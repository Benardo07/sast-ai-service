"""EmbeddingPGD — EDAT-style adversarial training on identifier embeddings.

Algorithm (matches EDAT code, not paper idealisation):
  1. Clean forward → loss_clean (caller provides, graph retained).
  2. K-step FGSM-sign perturbation δ from ∇loss_clean w.r.t. word_embeddings.weight.
     δ_{t+1} = clamp(δ_t + α·sign(∇L), -ε, ε)   — same grad each step (not re-computed).
  3. Apply δ to identifier-token rows of embedding matrix only.
  4. Adversarial forward → loss_adv.
  5. Restore embedding rows.
  6. Caller: total_loss = loss_clean + loss_adv → backward.

Only identifier tokens (variable/function names extracted via tree-sitter C, C keywords
excluded) are perturbed — preserves syntactic validity of code semantics.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
from loguru import logger

# ── tree-sitter C identifier extractor ───────────────────────────────────────

try:
    import tree_sitter_c
    from tree_sitter import Language, Parser as _Parser

    _C_LANGUAGE = Language(tree_sitter_c.language())
    _C_PARSER   = _Parser()
    _C_PARSER.language = _C_LANGUAGE
    _C_QUERY    = _C_LANGUAGE.query("(identifier) @identifier")
    _TREE_SITTER_OK = True
    logger.debug("EmbeddingPGD: tree-sitter-c loaded")
except Exception as _ts_err:
    _TREE_SITTER_OK = False
    logger.warning(f"EmbeddingPGD: tree-sitter-c unavailable ({_ts_err}), using regex fallback")

_ID_RE = re.compile(r'\b[a-zA-Z_]\w*\b')

_C_KEYWORDS: frozenset[str] = frozenset({
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while",
    "_Bool", "_Complex", "_Imaginary", "NULL", "true", "false",
})


def _extract_ts(code: str) -> list[str]:
    b = code.encode("utf-8", errors="replace")
    tree = _C_PARSER.parse(b)
    out: list[str] = []
    for match in _C_QUERY.matches(tree.root_node):
        node = match[0]
        ident = b[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        if ident not in _C_KEYWORDS:
            out.append(ident)
    return out


def _extract_re(code: str) -> list[str]:
    return [t for t in _ID_RE.findall(code) if t not in _C_KEYWORDS]


def _extract_identifiers(code: str) -> list[str]:
    if _TREE_SITTER_OK:
        try:
            return _extract_ts(code)
        except Exception:
            pass
    return _extract_re(code)


# ── Embedding weight finder ───────────────────────────────────────────────────

def _find_emb_weight(model: nn.Module) -> nn.Parameter | None:
    """Auto-detect the live LM's word embedding weight.

    UniXcoder / CodeBERT: codebert.embeddings.word_embeddings.weight
    CodeT5+ encoder:      codebert.encoder.embed_tokens.weight
    Falls back to first trainable param whose name ends with one of the known suffixes.
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "word_embeddings.weight" in name or "embed_tokens.weight" in name:
            return param
    return None


# ── EmbeddingPGD ─────────────────────────────────────────────────────────────

class EmbeddingPGD:
    """EDAT Embedding-layer Driven Adversarial Training.

    Parameters
    ----------
    model     : the full detector (must have a live LM branch).
    epsilon   : perturbation L∞ bound on the embedding table.
    alpha     : FGSM step size per ascent step.
    n_steps   : number of FGSM-sign ascent steps K.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.02,
        alpha:   float = 1e-2,
        n_steps: int   = 3,
    ) -> None:
        self.model   = model
        self.epsilon = epsilon
        self.alpha   = alpha
        self.n_steps = n_steps
        self.emb     = _find_emb_weight(model)
        if self.emb is None:
            raise ValueError(
                "EmbeddingPGD: no trainable word_embeddings/embed_tokens weight "
                "found in model — is live_lm != 'none'?"
            )
        logger.info(
            f"EmbeddingPGD ready | ε={epsilon} α={alpha} K={n_steps} | "
            f"emb shape={tuple(self.emb.shape)}"
        )

    # ── internal ─────────────────────────────────────────────────────────────

    def _compute_perturbation(self, loss: torch.Tensor) -> torch.Tensor:
        """K-step FGSM-sign perturbation δ ∈ [-ε, ε]^(vocab×d).

        Gradient is computed once from loss and reused across K steps
        (matches EDAT code behaviour, not proper multi-step PGD).
        retain_graph=True keeps the caller's computation graph alive so
        the final backward through loss_clean + loss_adv works correctly.
        """
        δ = torch.zeros_like(self.emb.data)
        for _ in range(self.n_steps):
            grad = torch.autograd.grad(
                loss, self.emb,
                retain_graph=True,
                create_graph=False,
            )[0]
            δ = δ + self.alpha * torch.sign(grad)
            δ = torch.clamp(δ, -self.epsilon, self.epsilon)
        return δ

    def _identifier_token_ids(
        self,
        func_input_ids: torch.Tensor,
        tokenizer,
    ) -> torch.Tensor:
        """Decode func_input_ids → source text → identifier tokens present in batch."""
        texts = tokenizer.batch_decode(func_input_ids.cpu(), skip_special_tokens=True)

        all_idents: set[str] = set()
        for text in texts:
            all_idents.update(_extract_identifiers(text))

        if not all_idents:
            return torch.zeros(0, dtype=torch.long, device=func_input_ids.device)

        enc = tokenizer(
            list(all_idents),
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None,
        )
        ident_ids: set[int] = set()
        for ids in enc["input_ids"]:
            ident_ids.update(ids)

        batch_ids = set(func_input_ids.reshape(-1).unique().cpu().tolist())
        target = sorted(ident_ids & batch_ids)

        if not target:
            return torch.zeros(0, dtype=torch.long, device=func_input_ids.device)

        return torch.tensor(target, dtype=torch.long, device=func_input_ids.device)

    # ── public API ────────────────────────────────────────────────────────────

    def adv_loss(
        self,
        loss_clean:     torch.Tensor,
        func_input_ids: torch.Tensor | None,
        tokenizer,
        forward_fn,
    ) -> torch.Tensor:
        """Compute adversarial loss; caller adds to loss_clean before backward.

        Parameters
        ----------
        loss_clean    : clean-batch loss (graph must be retained — do NOT call
                        .backward() on it before this returns).
        func_input_ids: function token IDs [B, L] used for identifier extraction.
        tokenizer     : the func-LM tokenizer (HF tokenizer).
        forward_fn    : callable() → loss scalar — re-runs forward on the same
                        batch with the perturbed embedding table.

        Returns
        -------
        loss_adv : adversarial loss scalar (0.0 tensor if no identifiers found).
        """
        if func_input_ids is None:
            return loss_clean.new_zeros(())

        δ = self._compute_perturbation(loss_clean)

        target_ids = self._identifier_token_ids(func_input_ids, tokenizer)
        if len(target_ids) == 0:
            return loss_clean.new_zeros(())

        orig = self.emb.data[target_ids].clone()
        with torch.no_grad():
            self.emb.data[target_ids] += δ[target_ids]

        try:
            loss_adv = forward_fn()
        finally:
            with torch.no_grad():
                self.emb.data[target_ids] = orig

        return loss_adv
