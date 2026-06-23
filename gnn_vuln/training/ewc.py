"""
EWC-DR: Elastic Weight Consolidation Done Right (arXiv:2603.18596).

Fixes vanilla EWC's gradient-vanishing problem via Logits Reversal (LR):
  z̃_k = -z_k  (negate logits)
  p̃_k = softmax(-z_k)
  Ω_i^LR = E[(y_k - p̃_k)² · (∂z̃_k/∂θ_i)²]

When model is confident (p_c → 1): LR gives p̃_c → 0 → (1 - 0)² = 1,
preserving gradient magnitude and correctly marking important weights.

Regularization loss added during task B training:
  L_ewc = (λ / 2) · Σ_i Ω_i · (θ_i - θ*_i)²

Configurable scope:
  "all" — protect all model parameters
  "lm"  — protect only the LM (codebert) branch (keep pretrained code knowledge)
  "gnn" — protect only GNN parameters (keep structural feature extraction)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class EWCDR:
    """
    EWC Done Right with Logits Reversal importance estimation.

    Parameters
    ----------
    model : nn.Module
        Model at its task-A optimal weights θ*.
    dataloader : DataLoader
        Task-A data used to estimate weight importance.
    device : torch.device
    ewc_weight : float
        λ — regularization strength. Higher = more conservative.
    scope : str
        Which parameters to protect: "all" | "lm" | "gnn".
    n_batches : int
        Number of batches for FIM estimation (0 = use all).
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        ewc_weight: float = 1000.0,
        scope: str = "all",
        n_batches: int = 0,
    ):
        self.ewc_weight  = ewc_weight
        self._scope      = scope
        self._star: dict[str, torch.Tensor] = {}   # θ* stored on CPU
        self._omega: dict[str, torch.Tensor] = {}  # Ω stored on CPU

        self._compute(model, dataloader, device, n_batches)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _named_params(self, model: nn.Module) -> dict[str, nn.Parameter]:
        """Return the parameter subset determined by scope."""
        if self._scope == "lm" and hasattr(model, "codebert"):
            return {
                f"codebert.{n}": p
                for n, p in model.codebert.named_parameters()
                if p.requires_grad
            }
        if self._scope == "gnn":
            lm_ids = {id(p) for p in model.lm_parameters()} if hasattr(model, "lm_parameters") else set()
            return {
                n: p for n, p in model.named_parameters()
                if p.requires_grad and id(p) not in lm_ids
            }
        # "all"
        return {n: p for n, p in model.named_parameters() if p.requires_grad}

    def _compute(self, model: nn.Module, dataloader, device: torch.device, n_batches: int) -> None:
        model.eval()
        model.zero_grad()

        params = self._named_params(model)
        for name, param in params.items():
            self._star[name]  = param.data.detach().cpu().clone()
            self._omega[name] = torch.zeros_like(param.data, device="cpu")

        n_samples = 0
        for i, batch in enumerate(dataloader):
            if n_batches > 0 and i >= n_batches:
                break
            batch = batch.to(device)

            # Forward pass — same dispatch as Trainer._forward
            if hasattr(model, "codebert"):
                func_ids  = getattr(batch, "func_input_ids",      None)
                func_mask = getattr(batch, "func_attention_mask",  None)
                out = model(
                    batch.x, batch.edge_index, batch.batch,
                    getattr(batch, "node_line", None),
                    getattr(batch, "edge_attr", None),
                    func_ids, func_mask,
                )
            else:
                out = model(
                    batch.x, batch.edge_index, batch.batch,
                    getattr(batch, "node_line", None),
                    getattr(batch, "edge_attr", None),
                )

            logits = out[0]  # primary logit (first element for all architectures)

            # Logits Reversal: negate logits → fixes gradient vanishing in standard FIM
            loss_lr = F.cross_entropy(-logits, batch.y)
            loss_lr.backward()

            # Accumulate squared gradients weighted by batch size
            for name, param in params.items():
                if param.grad is not None:
                    self._omega[name] += (
                        param.grad.data.pow(2).detach().cpu() * batch.num_graphs
                    )

            n_samples += batch.num_graphs
            model.zero_grad()

        # Normalise by total samples
        n_samples = max(n_samples, 1)
        for name in self._omega:
            self._omega[name] /= n_samples

        n_params = sum(p.numel() for p in self._omega.values())
        logger.info(
            f"EWC-DR importance computed | scope={self._scope} | "
            f"params={n_params:,} | samples={n_samples} | λ={self.ewc_weight}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        EWC-DR regularization term: (λ/2) · Σ_i Ω_i · (θ_i − θ*_i)²
        Add to task-B training loss to prevent forgetting task A.
        """
        device = next(model.parameters()).device
        loss   = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in self._omega:
                omega = self._omega[name].to(device)
                star  = self._star[name].to(device)
                # Class-incremental head growth: task-A importance covers the old
                # classes only (e.g. func_head [26, F]); the task-B model head is
                # larger ([36, F]). Penalise just the overlapping old-class slice —
                # new-class rows have no task-A importance and stay free to learn.
                if param.shape != star.shape:
                    sl = tuple(slice(0, s) for s in star.shape)
                    param = param[sl]
                loss  = loss + (omega * (param - star).pow(2)).sum()
        return 0.5 * self.ewc_weight * loss

    def save(self, path: str | Path) -> None:
        """Persist θ* and Ω to disk (avoid recomputing on repeat runs)."""
        torch.save({
            "star":       self._star,
            "omega":      self._omega,
            "ewc_weight": self.ewc_weight,
            "scope":      self._scope,
        }, path)
        logger.info(f"EWC-DR state saved → {path}")

    @classmethod
    def from_file(cls, path: str | Path, ewc_weight: float | None = None) -> "EWCDR":
        """
        Load pre-computed importance from disk — skips expensive FIM pass.
        Optionally override ewc_weight (e.g. to tune λ without recomputing).
        """
        obj = cls.__new__(cls)
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj._star       = data["star"]
        obj._omega      = data["omega"]
        obj.ewc_weight  = ewc_weight if ewc_weight is not None else data["ewc_weight"]
        obj._scope      = data.get("scope", "all")
        logger.info(
            f"EWC-DR state loaded ← {path} | "
            f"scope={obj._scope} | λ={obj.ewc_weight}"
        )
        return obj
