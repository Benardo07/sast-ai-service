"""supcon_head.py — Projection head for SupCon loss (MLP → L2-norm)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConProjector(nn.Module):
    """2-layer MLP projector for supervised contrastive learning.

    Maps fused representation to a unit-sphere embedding space.
    Dropout between layers enables two stochastic views from one input,
    which is required for self-supervised L_self (NT-Xent) loss.
    Separate from classification head — projector is discarded at inference.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.to(next(self.parameters()).dtype)), dim=-1)
