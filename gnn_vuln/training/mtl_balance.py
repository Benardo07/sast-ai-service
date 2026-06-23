"""Multi-task learning balance methods.

- UncertaintyWeights — Kendall, Gal & Cipolla CVPR 2018 (arxiv:1705.07115)
- pcgrad_project    — Yu et al. NeurIPS 2020 (arxiv:2001.06782)
- diagnose_mtl      — measures gradient conflict, loss imbalance, norm imbalance
"""

from __future__ import annotations
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyWeights(nn.Module):
    """Kendall 2018 homoscedastic uncertainty MTL weighting.

    Parameterizes log_sigma_squared (s = log σ²) per task for numerical stability.
    Combined loss = Σ_i [0.5 * exp(-s_i) * L_i + 0.5 * s_i]
                  = Σ_i [1/(2σ_i²) * L_i + log σ_i]    (paper eq 7, regression form)

    Paper also offers a classification-specific form (eq 10) with prefactor 1/σ_i²
    (no 0.5) for cross-entropy losses, derived from softmax temperature scaling.
    We use the regression form uniformly — simpler, equivalent in practice once
    σ is learned.

    s_i initialized at 0 → σ_i² = 1, so initial combined loss equals raw sum.
    """

    def __init__(self, task_names: list[str]) -> None:
        super().__init__()
        self.task_names = list(task_names)
        # log_sigma_squared per task, shape [num_tasks]
        self.log_sigma_sq = nn.Parameter(torch.zeros(len(task_names)))

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine raw per-task losses with learned uncertainty weights.

        losses: dict mapping task_name → scalar loss tensor.
        Returns: combined scalar loss = Σ [0.5*exp(-s_i)*L_i + 0.5*s_i] over tasks
                 present in losses (others skipped — handles missing rank/MIL etc).
        """
        total = losses[self.task_names[0]].new_zeros(())
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            s = self.log_sigma_sq[i]
            total = total + 0.5 * torch.exp(-s) * losses[name] + 0.5 * s
        return total

    @torch.no_grad()
    def get_sigmas(self) -> dict[str, float]:
        """Return current σ per task for logging."""
        return {
            name: float(torch.exp(0.5 * self.log_sigma_sq[i]).item())
            for i, name in enumerate(self.task_names)
        }


def pcgrad_project(
    losses: dict[str, torch.Tensor],
    params: list[nn.Parameter],
    *,
    retain_graph: bool = False,
) -> list[torch.Tensor]:
    """PCGrad (Yu NeurIPS 2020) projection — Algorithm 1.

    For each task i, iterate over other tasks j in random order. If gradients
    conflict (g_i · g_j < 0), project g_i onto orthogonal complement of g_j:
        g_i = g_i - (g_i · g_j / ||g_j||²) g_j
    Return Δθ = Σ_i g_i^PC.

    Args:
        losses: dict {task_name: scalar loss tensor}. Must have ≥2 tasks for
                projection to do anything; 1 task = pass-through.
        params: shared parameters to compute per-task gradients on.
        retain_graph: whether to retain graph between per-task .grad() calls.
                      Last call doesn't need it. Caller controls if outer scope
                      needs the graph again.

    Returns:
        list of tensors (same shape/order as params) — the projected sum.
    """
    task_names = list(losses.keys())
    if len(task_names) == 0:
        return [torch.zeros_like(p) for p in params]

    # Step 1: compute per-task gradients
    per_task_grads: dict[str, list[torch.Tensor]] = {}
    for i, name in enumerate(task_names):
        loss = losses[name]
        # retain_graph needed for all but last call when graph not needed externally
        need_retain = retain_graph or (i < len(task_names) - 1)
        grads = torch.autograd.grad(
            loss, params, retain_graph=need_retain, allow_unused=True,
        )
        # Replace None (unused) with zeros for consistent shape
        per_task_grads[name] = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, params)
        ]

    if len(task_names) == 1:
        return per_task_grads[task_names[0]]

    # Step 2: PCGrad projection — per Algorithm 1
    # For each task i, randomly iterate other tasks j; project g_i onto orth of g_j
    # if dot(g_i, g_j) < 0. Use flattened dot product / norm over all params.
    pc_grads: dict[str, list[torch.Tensor]] = {
        name: [g.clone() for g in grads]
        for name, grads in per_task_grads.items()
    }
    others_template = task_names[:]
    for name in task_names:
        others = [n for n in others_template if n != name]
        random.shuffle(others)
        for other in others:
            g_i = pc_grads[name]
            g_j = per_task_grads[other]  # use ORIGINAL g_j per paper
            # Flat dot product across all params
            dot = sum((gi * gj).sum() for gi, gj in zip(g_i, g_j))
            if dot < 0:
                norm_sq = sum((gj * gj).sum() for gj in g_j)
                if norm_sq > 0:
                    scale = dot / norm_sq
                    pc_grads[name] = [
                        gi - scale * gj for gi, gj in zip(g_i, g_j)
                    ]

    # Step 3: sum projected gradients across tasks
    result = [torch.zeros_like(p) for p in params]
    for name in task_names:
        for k, g in enumerate(pc_grads[name]):
            result[k] = result[k] + g
    return result


def diagnose_mtl(
    losses: dict[str, torch.Tensor],
    params: list[nn.Parameter],
    *,
    retain_graph: bool = False,
) -> dict[str, float]:
    """Diagnose MTL gradient/loss properties — NO surgery.

    Computes per-task gradients (autograd.grad), then for each task and each
    pair of tasks returns:
      - loss/<name>    : raw scalar loss value
      - gnorm/<name>   : ||∇ task loss|| on shared params (flat L2 norm)
      - cos/<a>_<b>    : cosine similarity between gradients of tasks a and b
      - conflict/<a>_<b>: 1.0 if cos < 0 else 0.0 (binary conflict indicator)

    Per PCGrad paper (Yu 2020):
      - High % cos < 0 across batches → PCGrad likely helps
      - High loss/grad-norm ratio → Kendall uncertainty likely helps

    Returns flat dict suitable for CSV logging.
    """
    names = list(losses.keys())
    if len(names) == 0:
        return {}

    # Compute per-task gradients (similar to pcgrad_project step 1)
    grads: dict[str, list[torch.Tensor]] = {}
    for i, name in enumerate(names):
        need_retain = retain_graph or (i < len(names) - 1)
        g = torch.autograd.grad(
            losses[name], params, retain_graph=need_retain, allow_unused=True,
        )
        grads[name] = [
            gi if gi is not None else torch.zeros_like(p)
            for gi, p in zip(g, params)
        ]

    out: dict[str, float] = {}
    # Per-task scalars
    for name in names:
        out[f"loss/{name}"] = float(losses[name].detach().item())
        # ||g||² = sum of element-wise squared, then sqrt
        norm_sq = sum((g * g).sum() for g in grads[name])
        out[f"gnorm/{name}"] = float(norm_sq.sqrt().item())

    # Pairwise cosine sim (only upper triangle)
    flats: dict[str, torch.Tensor] = {
        name: torch.cat([g.flatten() for g in grads[name]])
        for name in names
    }
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cs = F.cosine_similarity(flats[a].unsqueeze(0), flats[b].unsqueeze(0), dim=1)
            cs_val = float(cs.item())
            out[f"cos/{a}_{b}"] = cs_val
            out[f"conflict/{a}_{b}"] = 1.0 if cs_val < 0 else 0.0
    return out
