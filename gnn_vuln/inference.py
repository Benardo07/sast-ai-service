"""
inference.py — Clean inference API for all model architectures.

Produces a structured dict from a single CPG graph, suitable for downstream
consumption (SAST tools, APIs, notebooks) without exposing raw tensors.

Usage
-----
    from gnn_vuln.inference import load_model, predict, predict_from_file

    model, class_names = load_model(
        checkpoint="checkpoints/<run>/best_lmgat_codebert.pt",
        config="configs/lmgat_codebert/multiclass.yaml",
        device="cuda",
    )

    # From a pre-built PyG Data object
    result = predict(model, data, class_names)

    # Directly from a CPG .json file
    result = predict_from_file(model, "path/to/func.json", class_names)

    print(result["prediction"])          # "CWE-119"
    print(result["confidence"])          # 0.87
    print(result["suspicious_lines"])    # [{"line": 14, "statement": "strcpy(dst, src);", "score": 0.92}, ...]

Output schema
-------------
{
    "prediction":         str   — predicted class name ("benign", "CWE-119", ...)
    "class_id":           int   — predicted class index
    "is_vulnerable":      bool  — True if class_id > 0
    "confidence":         float — softmax probability of the predicted class [0,1]
    "class_probabilities": {    — softmax probability for every class
        "benign":   float,
        "CWE-119":  float,
        ...
    },
    "suspicious_lines": [       — statements ranked by vulnerability score, descending
        {
            "line":       int,
            "statement":  str,    — source of the statement at that line (spans
                                    continuation lines of a multi-line statement);
                                    null if the source is unavailable
            "score":      float,  — vulnerability score [0,1]; higher = more suspicious
            # Only present for lmgat_mcs (multiclass statement head):
            "predicted_cwe":        str,
            "class_probabilities":  { class_name: float, ... }
        },
        ...
    ]
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.graph_builder_lm import build_func_text, build_from_parsed, parse_cpg
from gnn_vuln.data.node_embedder import LMNodeEmbedder
from gnn_vuln.models.registry import build_model
from gnn_vuln.utils import get_device, load_checkpoint


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(
    checkpoint: str | Path,
    config: str | Path,
    device: str = "cpu",
) -> tuple[torch.nn.Module, list[str] | None]:
    """
    Load a trained model from a checkpoint + config pair.

    Returns (model, class_names) where class_names is the ordered list of
    class labels (e.g. ["benign","CWE-119",...]) or None for binary mode.
    """
    cfg = (
        Config.from_yaml(config)
        if Path(config).exists()
        else load_default_config()
    )
    dev = get_device(device)

    # in_channels is fixed by the node feature layout (773D)
    from gnn_vuln.data.node_embedder import NODE_FEAT_DIM
    in_channels = NODE_FEAT_DIM

    model = build_model(cfg, in_channels).to(dev)
    load_checkpoint(model, checkpoint, device=str(dev))
    model.eval()

    # Derive class_names from config
    if cfg.data.mode == "multiclass":
        vocab_path = cfg.data.raw_dir / "cwe_vocab.json"
        if vocab_path.exists():
            import json
            with open(vocab_path) as f:
                vocab: dict[str, int] = json.load(f)
            class_names = [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        else:
            class_names = [str(i) for i in range(cfg.model.num_classes)]
    else:
        class_names = ["benign", "vulnerable"]

    return model, class_names


# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: torch.nn.Module,
    data,
    class_names: list[str] | None = None,
    device: str | torch.device | None = None,
    top_k_lines: int | None = None,
) -> dict:
    """
    Run inference on a single PyG Data object and return a structured result.

    Parameters
    ----------
    model       : trained model (any architecture)
    data        : single PyG Data object with x, edge_index, node_line, etc.
    class_names : ordered class label list; falls back to "class_0", "class_1"...
    device      : override device; defaults to the device of model parameters
    top_k_lines : if set, only return the top-k suspicious lines (all by default)

    Returns
    -------
    Structured dict — see module docstring for full schema.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    model.eval()
    data = data.to(device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    node_line = getattr(data, "node_line", None)
    edge_attr = getattr(data, "edge_attr", None)
    rwse = getattr(data, "rwse", None)

    def _with_batch_dim(t):
        # Data stores [max_length]; model expects [B, max_length]
        return t.unsqueeze(0) if t is not None and t.dim() == 1 else t

    # Forward pass — route func tokens when present
    if hasattr(model, "codebert"):
        func_input_ids = _with_batch_dim(getattr(data, "func_input_ids", None))
        func_attention_mask = _with_batch_dim(getattr(data, "func_attention_mask", None))
        func_token_lines = _with_batch_dim(getattr(data, "func_token_lines", None))
        out = model(
            data.x, data.edge_index, batch, node_line, edge_attr,
            func_input_ids, func_attention_mask, func_token_lines,
            func_line_cls=getattr(data, "func_line_cls", None),
            func_line_ids=getattr(data, "func_line_ids", None),
            func_line_cls_batch=getattr(data, "func_line_cls_batch", None),
            rwse=rwse,
            return_repr=True,
        )
    else:
        out = model(data.x, data.edge_index, batch, node_line, edge_attr,
                    rwse=rwse, return_repr=True)

    # return_repr=True appends the pre-head function representation as the last
    # element. Strip it before the length-based unpack below.
    out = tuple(out)
    cls_repr = out[-1]
    out = out[:-1]

    # Unpack return tuple — mirrors trainer._forward (2/3/4/5-tuple model outputs)
    if len(out) == 5:
        logit_func, _logit_group, _logit_binary, stmt_scores_list, _z = out
    elif len(out) == 4:
        logit_func, _logit_group, _logit_binary, stmt_scores_list = out
    elif len(out) == 3:
        logit_func, stmt_scores_list, _z = out
    else:
        logit_func, stmt_scores_list = out

    # ── Function-level result ────────────────────────────────────────────────
    probs = F.softmax(logit_func[0], dim=-1)          # [num_classes]
    class_id = int(probs.argmax().item())
    confidence = float(probs[class_id].item())
    num_classes = probs.shape[0]

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    class_probabilities = {
        class_names[i]: round(float(probs[i].item()), 6)
        for i in range(num_classes)
    }

    prediction = class_names[class_id] if class_id < len(class_names) else str(class_id)
    is_vulnerable = class_id > 0

    # ── Statement-level result ───────────────────────────────────────────────
    suspicious_lines: list[dict] = []

    # Source lines, so each suspicious line can carry the statement it refers to.
    raw_func = getattr(data, "raw_func", None)
    src_lines = str(raw_func).split("\n") if raw_func else None

    def _statement_text(line: int, node_lines_sorted: list[int]) -> Optional[str]:
        """Full source of the statement at `line`, spanning any continuation lines
        up to the next statement (multi-line calls, conditions, split strings)."""
        if not src_lines or line < 1 or line > len(src_lines):
            return None
        nxt = next((l for l in node_lines_sorted if l > line), len(src_lines) + 1)
        span = src_lines[line - 1:nxt - 1]
        while span and not span[-1].strip():
            span.pop()
        return "\n".join(span).strip() or None

    if stmt_scores_list is not None and node_line is not None:
        scores_raw = stmt_scores_list[0]   # tensor for graph 0

        if len(scores_raw) == 0:
            pass  # no valid lines — leave suspicious_lines empty

        elif scores_raw.dim() == 2:
            # Architecture 4 (lmgat_mcs): [n_stmts, num_classes]
            stmt_probs = F.softmax(scores_raw, dim=-1)           # [n_stmts, num_classes]
            vuln_scores = (1.0 - stmt_probs[:, 0]).cpu()         # [n_stmts]

            valid = node_line >= 0
            unique_lines = node_line[valid].unique(sorted=True).cpu().tolist()

            order = torch.argsort(vuln_scores, descending=True)
            lines_sorted = [unique_lines[i] for i in order.tolist()]
            scores_sorted = vuln_scores[order].tolist()
            probs_sorted = stmt_probs[order].cpu()               # [n_stmts, num_classes]

            entries = zip(lines_sorted, scores_sorted, probs_sorted)
            if top_k_lines is not None:
                entries = list(entries)[:top_k_lines]

            for line, score, sp in entries:
                stmt_class_id = int(sp.argmax().item())
                stmt_entry = {
                    "line": int(line),
                    "statement": _statement_text(int(line), unique_lines),
                    "score": round(float(score), 6),
                    "predicted_cwe": (
                        class_names[stmt_class_id]
                        if stmt_class_id < len(class_names)
                        else str(stmt_class_id)
                    ),
                    "class_probabilities": {
                        class_names[i]: round(float(sp[i].item()), 6)
                        for i in range(num_classes)
                    },
                }
                suspicious_lines.append(stmt_entry)

        else:
            # Architectures 1–3: [n_stmts] binary scalar
            scores = torch.sigmoid(scores_raw).cpu()             # [n_stmts]

            valid = node_line >= 0
            unique_lines = node_line[valid].unique(sorted=True).cpu().tolist()

            order = torch.argsort(scores, descending=True)
            lines_sorted = [unique_lines[i] for i in order.tolist()]
            scores_sorted = scores[order].tolist()

            entries = list(zip(lines_sorted, scores_sorted))
            if top_k_lines is not None:
                entries = entries[:top_k_lines]

            suspicious_lines = [
                {
                    "line": int(ln),
                    "statement": _statement_text(int(ln), unique_lines),
                    "score": round(float(sc), 6),
                }
                for ln, sc in entries
            ]

    # Function-level pre-head representation (vector fed to the classification
    # head) for drift detection / similarity search. [1, D] → flat list of floats.
    cls_embedding = (
        cls_repr[0].float().cpu().tolist()
        if cls_repr is not None else None
    )

    return {
        "prediction": prediction,
        "class_id": class_id,
        "is_vulnerable": is_vulnerable,
        "confidence": round(confidence, 6),
        "class_probabilities": class_probabilities,
        "suspicious_lines": suspicious_lines,
        "cls_embedding": cls_embedding,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper: predict directly from a CPG file
# ---------------------------------------------------------------------------

def predict_from_file(
    model: torch.nn.Module,
    cpg_path: str | Path,
    class_names: list[str] | None = None,
    pretrained_lm: str = "microsoft/codebert-base",
    func_lm: str = "",
    func_max_length: int = 512,
    label: int = 0,
    flaw_lines: list[int] | None = None,
    max_nodes: int = 1000,
    device: str | torch.device | None = None,
    top_k_lines: int | None = None,
) -> Optional[dict]:
    """
    Parse a CPG .json/.xml file, embed nodes with CodeBERT, and run predict().

    Parameters
    ----------
    model        : trained model
    cpg_path     : path to a Joern-exported CPG file (.json or .xml/.graphml)
    class_names  : ordered class label list
    pretrained_lm: HuggingFace model name for node embedding (frozen node features)
    func_lm      : HuggingFace model name for the live function-level branch;
                   falls back to pretrained_lm if empty
    func_max_length: tokenizer max_length for the func_lm branch
    label        : ground-truth label (only stored in Data.y, does not affect output)
    flaw_lines   : known flaw lines (optional; stored in flaw_line_mask)
    max_nodes    : skip graphs larger than this
    device       : inference device
    top_k_lines  : if set, only return the top-k suspicious lines

    Returns
    -------
    Structured result dict, or None if the CPG was empty / too large.
    """
    if device is None and hasattr(model, "parameters"):
        device = next(model.parameters()).device
    device = torch.device(device or "cpu")

    cpg = parse_cpg(cpg_path, max_nodes=max_nodes)
    if cpg is None:
        return None

    embedder = LMNodeEmbedder(model_name=pretrained_lm, device=str(device))
    codes = cpg["codes"]
    embed_batch = 256
    parts = [
        embedder.embed_batch(codes[i: i + embed_batch])
        for i in range(0, len(codes), embed_batch)
    ]
    cls_feats = torch.cat(parts, dim=0)

    # Tokenize function text when the model has a live CodeBERT/UniXcoder branch
    func_input_ids = func_attention_mask = func_token_lines = None
    if hasattr(model, "codebert"):
        from gnn_vuln.data.dataset_lm import _load_tokenizer, _compute_func_token_lines
        tokenizer = _load_tokenizer(func_lm or pretrained_lm)
        func_text = build_func_text(cpg)
        enc = tokenizer(
            func_text, max_length=func_max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        func_input_ids = enc["input_ids"]        # [1, func_max_length]
        func_attention_mask = enc["attention_mask"]
        func_token_lines = _compute_func_token_lines(
            func_text, func_max_length, tokenizer
        ).unsqueeze(0)                            # [1, func_max_length]

    data = build_from_parsed(cpg, cls_feats, label, flaw_lines,
                              func_input_ids, func_attention_mask, func_token_lines)

    return predict(model, data, class_names, device=device, top_k_lines=top_k_lines)


# ---------------------------------------------------------------------------
# OOP wrapper
# ---------------------------------------------------------------------------

class VulnPredictor:
    """
    High-level inference object. Wraps a trained model + class names.

    Usage
    -----
        predictor = VulnPredictor.from_checkpoint(
            checkpoint="checkpoints/run/best.pt",
            config="configs/lmgat_codebert_mtl/multiclass_mtl_livable_f1stop.yaml",
        )
        result = predictor.predict(data)
        result = predictor.predict_from_file("path/to/cpg.json")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str] | None,
        device: torch.device,
        pretrained_lm: str = "microsoft/codebert-base",
        func_lm: str = "",
        func_max_length: int = 512,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.device = device
        self.pretrained_lm = pretrained_lm
        self.func_lm = func_lm or pretrained_lm
        self.func_max_length = func_max_length

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        config: str | Path,
        device: str = "cpu",
    ) -> "VulnPredictor":
        """Load from checkpoint + config. Returns a ready-to-use VulnPredictor."""
        model, class_names = load_model(checkpoint, config, device)
        cfg = Config.from_yaml(config) if Path(config).exists() else load_default_config()
        pretrained_lm = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
        func_lm = getattr(cfg.model, "func_lm", "")
        func_max_length = getattr(cfg.model, "func_max_length", 512)
        return cls(model, class_names, get_device(device), pretrained_lm, func_lm, func_max_length)

    def predict(self, data, top_k_lines: int | None = None) -> dict:
        """Run inference on a single PyG Data object."""
        return predict(self.model, data, self.class_names,
                       device=self.device, top_k_lines=top_k_lines)

    def predict_from_file(
        self,
        cpg_path: str | Path,
        label: int = 0,
        flaw_lines: list[int] | None = None,
        max_nodes: int = 1000,
        top_k_lines: int | None = None,
    ) -> Optional[dict]:
        """Parse + embed + predict from a CPG file (lower-level; see predict_code)."""
        return predict_from_file(
            self.model, cpg_path,
            class_names=self.class_names,
            pretrained_lm=self.pretrained_lm,
            func_lm=self.func_lm,
            func_max_length=self.func_max_length,
            label=label,
            flaw_lines=flaw_lines,
            max_nodes=max_nodes,
            device=self.device,
            top_k_lines=top_k_lines,
        )

    def predict_code(
        self,
        code: str,
        joern_cli: str | Path,
        max_nodes: int = 2500,
        top_k_lines: int | None = None,
    ) -> Optional[dict]:
        """Predict directly from a function SOURCE STRING — the high-level entry point.

        Hides the Joern CPG step: builds the Code Property Graph in a private temp dir,
        embeds + runs the model, and returns the result dict (prediction, confidence,
        class_probabilities, suspicious_lines, cls_embedding). Returns None if Joern
        produced no CPG (unparseable / empty function). `joern_cli` = the joern-cli dir."""
        import tempfile

        from gnn_vuln.data.joern_runner import process_function

        with tempfile.TemporaryDirectory() as td:
            cpg = process_function(code, 0, Path(td), joern_cli_dir=Path(joern_cli))
            if cpg is None:
                return None
            result = self.predict_from_file(cpg, max_nodes=max_nodes, top_k_lines=top_k_lines)
            if result:  # attach the source text of each suspicious line (1-indexed)
                src = code.splitlines()
                for sl in result.get("suspicious_lines", []):
                    ln = sl.get("line")
                    if isinstance(ln, int) and 1 <= ln <= len(src):
                        sl["code"] = src[ln - 1]
            return result

    def predict_codes(
        self,
        codes: list[str],
        joern_cli: str | Path,
        max_nodes: int = 2500,
        top_k_lines: int | None = None,
    ) -> list[Optional[dict]]:
        """Batch predict_code — one result dict (or None on Joern failure) per input string."""
        return [self.predict_code(c, joern_cli, max_nodes, top_k_lines) for c in codes]
