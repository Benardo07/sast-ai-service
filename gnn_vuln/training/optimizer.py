"""Optimizer and LR scheduler factory."""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from loguru import logger

from gnn_vuln.config import Config
from gnn_vuln.models.base import VulnDetectorBase


_LAYER_RE = re.compile(r"(?:^|\.)(?:layer|block)\.(\d+)\.")


def _group_lm_params_by_layer(
    lm_module: nn.Module,
) -> tuple[dict, int] | None:
    """Group LM parameters by transformer layer index for LLRD.

    Returns dict with keys:
      "embeddings" → list[Parameter]   (input embedding layer)
      "head"       → list[Parameter]   (pooler / lm_head / classifier above transformer stack)
      0..N-1       → list[Parameter]   (transformer layer i)
    plus the total layer count N. Returns None if no transformer layers found
    (e.g. CodeT5+ embedding model without a transformer stack).
    """
    groups: dict = {"embeddings": [], "head": []}
    max_layer = -1
    for name, p in lm_module.named_parameters():
        m = _LAYER_RE.search(name)
        if m:
            idx = int(m.group(1))
            groups.setdefault(idx, []).append(p)
            max_layer = max(max_layer, idx)
        elif "embeddings" in name:
            groups["embeddings"].append(p)
        else:
            groups["head"].append(p)
    if max_layer < 0:
        return None
    return groups, max_layer + 1

def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: Config,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, object, bool]:
    """
    Build optimizer + LR scheduler for the given model and config.

    Returns
    -------
    (optimizer, scheduler, step_scheduler_per_batch)

    step_scheduler_per_batch=True  → linear warmup; call scheduler.step() after
                                     each batch inside train_epoch().
    step_scheduler_per_batch=False → epoch-level scheduler (plateau or cosine).
    """
    # Fine-tune path = the model has a live LM branch. Robust to live_lm=none
    # (lmgat_codebert without a func LM → plain Adam, no separate LM lr group).
    is_ft = isinstance(model, VulnDetectorBase) and model.has_live_lm()
    lr_scheduler = getattr(cfg.train, "lr_scheduler", "plateau").lower()

    if is_ft:
        lm_lr        = getattr(cfg.train, "lm_lr", 2e-5)
        warmup_ratio = getattr(cfg.train, "warmup_ratio", 0.1)
        llrd_decay   = float(getattr(cfg.train, "lm_llrd_decay", 1.0))

        # Use base class method when available; fall back to attribute check
        if isinstance(model, VulnDetectorBase):
            lm_params    = model.lm_parameters()
            lm_param_ids = {id(p) for p in lm_params}
        else:
            lm_param_ids = {id(p) for p in model.codebert.parameters()}
            lm_params    = [p for p in model.parameters() if id(p) in lm_param_ids]

        other_params = [p for p in model.parameters() if id(p) not in {id(p) for p in lm_params}]

        param_groups: list[dict] = []
        if llrd_decay < 1.0 and llrd_decay > 0.0 and hasattr(model, "codebert"):
            parsed = _group_lm_params_by_layer(model.codebert)
            if parsed is not None:
                groups, n_layers = parsed
                if groups["head"]:
                    param_groups.append({"params": groups["head"], "lr": lm_lr, "weight_decay": 0.01})
                for i in range(n_layers - 1, -1, -1):
                    if not groups.get(i):
                        continue
                    lr_i = lm_lr * (llrd_decay ** (n_layers - 1 - i))
                    param_groups.append({"params": groups[i], "lr": lr_i, "weight_decay": 0.01})
                if groups["embeddings"]:
                    emb_lr = lm_lr * (llrd_decay ** n_layers)
                    param_groups.append({"params": groups["embeddings"], "lr": emb_lr, "weight_decay": 0.0})
                logger.info(
                    f"LLRD enabled: decay={llrd_decay} | top_layer_lr={lm_lr:.2e} | "
                    f"bottom_layer_lr={lm_lr * (llrd_decay ** (n_layers - 1)):.2e} | "
                    f"emb_lr={lm_lr * (llrd_decay ** n_layers):.2e}"
                )
            else:
                logger.warning("lm_llrd_decay set but LM has no transformer stack — falling back to uniform LM lr")
        if not param_groups:
            param_groups.append({"params": lm_params, "lr": lm_lr, "weight_decay": 0.01})
        param_groups.append({"params": other_params, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay})
        optimizer = torch.optim.AdamW(param_groups)

        if lr_scheduler == "cosine":
            # Cosine decay over all epochs — no warmup for cosine (simpler, EDAT-style)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.train.epochs, eta_min=0.0
            )
            step_per_batch = False
            logger.info(
                f"AdamW: LM lr={lm_lr:.1e}  GNN lr={cfg.train.lr:.1e} | "
                f"cosine decay over {cfg.train.epochs} epochs"
            )
        else:
            # Default: linear warmup → constant (original behavior)
            warmup_steps = max(1, int(total_steps * warmup_ratio))
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            step_per_batch = True
            logger.info(
                f"AdamW: LM lr={lm_lr:.1e}  GNN lr={cfg.train.lr:.1e} | "
                f"linear warmup {warmup_steps}/{total_steps} steps"
            )
    else:
        # Only optimize params with requires_grad=True. Identical to
        # model.parameters() for full-training runs (everything trainable);
        # for cRT it restricts the optimizer to the unfrozen func_head.
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        if lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.train.epochs, eta_min=0.0
            )
            step_per_batch = False
            logger.info(
                f"Adam: lr={cfg.train.lr:.1e} | cosine decay over {cfg.train.epochs} epochs"
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )
            step_per_batch = False

    return optimizer, scheduler, step_per_batch
