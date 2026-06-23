"""Loss functions for vulnerability detection training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ── Focal loss ────────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Focal loss for multiclass classification.
    gamma=0 reduces to standard cross-entropy.
    label_smoothing > 0 applies label smoothing before focal modulation.
    """
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none",
                         label_smoothing=label_smoothing)
    if gamma == 0.0:
        return ce.mean()
    p_t = torch.exp(-ce)
    return ((1.0 - p_t) ** gamma * ce).mean()


# ── Epoch-adaptive inverse-frequency class weights ────────────────────────────
# Simple approximation: ramps class weights from uniform → inverse-frequency
# over training epochs. NOT the real LIVABLE paper loss.

def epoch_adaptive_class_weights(
    counts: torch.Tensor,
    epoch: int,
    total_epochs: int,
    num_classes: int,
    device: torch.device,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Epoch-adaptive inverse-frequency class weights.
    w_i(t) = (N / (K * n_i)) ^ (t / T)
    Ramps from uniform at epoch=0 to full inverse-frequency at epoch=total_epochs.

    This is a simple approximation inspired by LIVABLE's time-shift idea,
    but is NOT the real LIVABLE paper loss. For the real LIVABLE two-branch
    loss (focal + LSCE blended by T), use livable_loss().
    """
    t_ratio = epoch / total_epochs
    base = counts.sum().float() / (num_classes * counts.float().clamp(min=1))
    return torch.clamp(base ** t_ratio, max=max_weight).to(device)


# ── Real LIVABLE loss (arXiv:2306.06935, Eq. 11-12) ──────────────────────────

def livable_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    total_epochs: int,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Real LIVABLE adaptive re-weighting loss (arXiv:2306.06935, Eq. 11-12).

    L = T * L_FL + (1-T) * L_LSCE

    where:
      T = 1 - (epoch / total_epochs)^2   (time-shift: starts at 1, decays to 0)
      L_FL   = focal loss (focuses on hard/tail samples, dominant early)
      L_LSCE = label-smooth CE (reduces overconfidence on head, dominant late)

    Both losses are applied to ALL samples — no head/tail splitting.
    The head/medium/tail tiers in the paper are for evaluation reporting only.

    Parameters
    ----------
    logits         : [B, C] raw logits
    targets        : [B]    integer class labels
    epoch          : current epoch (1-indexed)
    total_epochs   : total training epochs
    focal_gamma    : focal loss modulating factor (paper uses 2.0)
    label_smoothing: smoothing ε for LSCE branch (paper uses 0.1)
    weight         : optional per-class weight tensor [C]
    """
    T = 1.0 - (epoch / total_epochs) ** 2

    L_FL = focal_loss(logits, targets, gamma=focal_gamma, weight=weight)
    L_LSCE = F.cross_entropy(logits, targets, weight=weight,
                             label_smoothing=label_smoothing)

    return T * L_FL + (1.0 - T) * L_LSCE


# ── MIL losses ────────────────────────────────────────────────────────────────

def mil_loss(
    stmt_scores_list: list[torch.Tensor],
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    MIL binary cross-entropy for statement-level localisation (binary stmt head).
    Top-k statements per function:
      benign (label=0)      → pseudo-label 0
      vulnerable (label>0)  → pseudo-label 1
    """
    device = labels.device
    binary_labels = (labels > 0).float()  # compute once, no per-graph .item() sync
    losses: list[torch.Tensor] = []
    for scores, bl in zip(stmt_scores_list, binary_labels):
        if len(scores) == 0:
            continue
        actual_k = min(k, len(scores))
        topk_scores = scores.topk(actual_k).values
        losses.append(F.binary_cross_entropy_with_logits(topk_scores, bl.expand(actual_k)))
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def mil_loss_multiclass(
    stmt_scores_list: list[torch.Tensor],
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    MIL multiclass cross-entropy for [n_stmts, num_classes] stmt heads (lmgat_mcs).
    Per function of class c, select top-k statements by score[:, c] and apply CE.
    """
    device = labels.device
    losses: list[torch.Tensor] = []
    for scores, label in zip(stmt_scores_list, labels):
        if scores.shape[0] == 0:
            continue
        actual_k = min(k, scores.shape[0])
        topk_idx = scores[:, label].topk(actual_k).indices
        topk_scores = scores[topk_idx]          # [k, num_classes]
        targets = label.long().expand(actual_k)
        losses.append(F.cross_entropy(topk_scores, targets))
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


# ── Ranking loss ──────────────────────────────────────────────────────────────

def ranking_loss(
    stmt_scores_list: list[torch.Tensor],
    batch_idx: torch.Tensor,
    node_line: torch.Tensor,
    flaw_line_mask: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Margin ranking loss: flaw lines must score higher than non-flaw by ≥ margin.
    Only applies to vulnerable functions (label > 0) with binary stmt heads.
    """
    device = labels.device
    # Single sync: get all vulnerable graph indices as Python ints
    vuln_indices: list[int] = (labels > 0).nonzero(as_tuple=False).squeeze(1).tolist()
    losses: list[torch.Tensor] = []
    for b in vuln_indices:
        scores = stmt_scores_list[b]
        if len(scores) == 0:
            continue
        mask = batch_idx == b
        lines_b = node_line[mask]
        flaw_b  = flaw_line_mask[mask]
        valid = lines_b >= 0
        if not valid.any():
            continue
        lines_b = lines_b[valid]
        flaw_b  = flaw_b[valid].float()
        unique_lines, inv = torch.unique(lines_b, sorted=True, return_inverse=True)
        n_unique = unique_lines.shape[0]
        # Vectorized: scatter amax to get per-stmt flaw flag (replaces list comprehension)
        flaw_max = torch.zeros(n_unique, device=device)
        flaw_max.scatter_reduce_(0, inv, flaw_b, reduce='amax', include_self=True)
        flaw_flags = flaw_max.bool()
        if not flaw_flags.any() or flaw_flags.all():
            continue
        flaw_scores = scores[flaw_flags]
        safe_scores = scores[~flaw_flags]
        diff = flaw_scores.unsqueeze(1) - safe_scores.unsqueeze(0)
        losses.append(F.relu(margin - diff).mean())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()
