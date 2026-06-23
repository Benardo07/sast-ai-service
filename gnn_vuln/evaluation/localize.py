"""LocalizationExtractor — runs model inference and collects per-function scores."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gnn_vuln.metrics import LocalizationMetrics


class LocalizationExtractor:
    """
    Runs a trained model over a DataLoader and collects:
      - function-level predictions (y_true, y_pred, y_prob, confidence)
      - per-function statement-level localization dicts
    """

    def __init__(self, model: nn.Module, loader: DataLoader, device: torch.device) -> None:
        self.model = model
        self.loader = loader
        self.device = device

    @torch.no_grad()
    def run(self):
        """
        Returns
        -------
        y_true      : np.ndarray [N]
        y_pred      : np.ndarray [N]
        y_prob      : np.ndarray [N, C]
        confidence  : np.ndarray [N]
        loc_results : list[dict]  — one per function
        """
        import numpy as np

        self.model.eval()
        all_y, all_pred, all_prob, all_conf = [], [], [], []
        loc_results: list[dict] = []

        for batch in self.loader:
            batch = batch.to(self.device)
            node_line = getattr(batch, "node_line", None)
            flaw_mask = getattr(batch, "flaw_line_mask", None)
            edge_attr  = getattr(batch, "edge_attr", None)

            out = self._forward(batch, node_line, edge_attr)

            logit_func, stmt_scores_list = self._unpack(out)

            probs = torch.softmax(logit_func, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)

            all_y.extend(batch.y.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.append(probs)
            all_conf.extend(confs.tolist())

            if stmt_scores_list is not None and node_line is not None:
                B = int(batch.batch.max().item()) + 1
                for b in range(B):
                    loc_results.append(
                        self._extract_func(
                            stmt_scores_list[b], batch.batch, node_line, flaw_mask, b
                        )
                    )
            else:
                B = int(batch.batch.max().item()) + 1
                for _ in range(B):
                    loc_results.append(LocalizationMetrics.make_result([], [], []))

        import numpy as np
        return (
            np.array(all_y),
            np.array(all_pred),
            np.vstack(all_prob) if all_prob else np.zeros((0, 2)),
            np.array(all_conf),
            loc_results,
        )

    def _forward(self, batch, node_line, edge_attr):
        # Only read rwse if encoder uses PE — avoids PyG batch issues for non-PE configs.
        rwse = None
        _enc = getattr(self.model, "encoder", None)
        if _enc is not None and getattr(_enc, "use_pe", False):
            rwse = getattr(batch, "rwse", None)
        if hasattr(self.model, "codebert"):
            func_input_ids      = getattr(batch, "func_input_ids", None)
            func_attention_mask = getattr(batch, "func_attention_mask", None)
            func_token_lines    = getattr(batch, "func_token_lines", None)
            return self.model(
                batch.x, batch.edge_index, batch.batch, node_line, edge_attr,
                func_input_ids, func_attention_mask, func_token_lines,
                rwse=rwse,
            )
        return self.model(batch.x, batch.edge_index, batch.batch, node_line, edge_attr, rwse=rwse)

    @staticmethod
    def _unpack(out):
        """Unpack model output regardless of tuple length."""
        if isinstance(out, (tuple, list)):
            if len(out) >= 5:
                return out[0], out[3]
            if len(out) == 4:
                return out[0], out[3]
            if len(out) == 3:
                return out[0], out[2]
            return out[0], out[1]
        return out, None

    @staticmethod
    def _extract_func(scores_b, batch_idx, node_line, flaw_mask, b) -> dict:
        """Extract per-function localization data for graph b."""
        # scores_b: per-unique-line tensor for graph b (from StmtHead.score()[b])
        # batch_idx/node_line/flaw_mask: per-node tensors for the full batch
        if scores_b is None or len(scores_b) == 0:
            return LocalizationMetrics.make_result([], [], [])

        # scores_b is 1D (binary stmt) or 2D (multiclass stmt) — convert to scalar
        scores_b_cpu = scores_b.detach().cpu()
        if scores_b_cpu.dim() == 2:
            probs = torch.softmax(scores_b_cpu, dim=-1)
            scores_scalar = (1.0 - probs[:, 0])   # shape [n_unique_lines]
        else:
            scores_scalar = torch.sigmoid(scores_b_cpu)  # shape [n_unique_lines]

        # Slice per-node tensors to this graph using batch_idx
        graph_mask  = (batch_idx == b).cpu()
        node_line_b = node_line.cpu()[graph_mask]
        flaw_b = (
            flaw_mask.cpu()[graph_mask] if flaw_mask is not None
            else torch.zeros_like(node_line_b)
        )

        valid = node_line_b >= 0
        if not valid.any():
            return LocalizationMetrics.make_result([], [], [])

        lines_b = node_line_b[valid]
        flaw_b  = flaw_b[valid]
        # StmtHead.score() iterates lines_b.unique(sorted=True) in the same order
        unique_lines = lines_b.unique(sorted=True)

        if len(unique_lines) != len(scores_scalar):
            # Fallback: truncate to shorter (safety guard against edge cases)
            n = min(len(unique_lines), len(scores_scalar))
            unique_lines  = unique_lines[:n]
            scores_scalar = scores_scalar[:n]

        line_scores: list[float] = []
        line_labels: list[int]  = []
        for i, line in enumerate(unique_lines):
            # Index by position i — scores_scalar[i] matches unique_lines[i]
            line_scores.append(scores_scalar[i].item())
            node_mask = lines_b == line
            line_labels.append(int(flaw_b[node_mask].any().item()))

        return LocalizationMetrics.make_result(
            unique_lines.cpu().tolist(),
            line_scores,
            line_labels,
        )
