"""
pretrain_jepa.py — Node-masked JEPA self-supervised pretraining for the GNN encoder.

Masked-latent prediction (I-JEPA / GraphMAE flavor, NO METIS patches — the P-series
showed METIS patchify collapses CPGs). The online encoder sees a graph with a random
subset of node features replaced by a learned [MASK] token; an EMA target encoder
sees the full graph. A small MLP predictor maps the online latents at masked nodes to
the target latents there. Loss = SmoothL1 (or cosine) in latent space, masked nodes
only. EMA weights + stop-grad prevent collapse (pure SSL, no label anchor).

Saves the EMA (target) encoder state_dict — downstream loads it into model.encoder
via train.gnn_init_checkpoint (finetune) or +freeze_gnn (frozen linear probe).

Run (cloud, Linux):
  PYTHONPATH=src python -m gnn_vuln.pretrain_jepa --config configs/ablation/jepa/Q0_jepa_pretrain_n48.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.models.registry import build_model
from gnn_vuln.train import TrainingSession
from gnn_vuln.utils import set_seed, setup_logging, get_device


class JEPAPredictor(nn.Module):
    """Small MLP predictor — kept simpler than the encoder (Graph-JEPA: a weak
    predictor is crucial to avoid representation collapse)."""

    def __init__(self, dim: int, hidden: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if layers <= 1:
            self.net = nn.Linear(dim, dim)
        else:
            mods: list[nn.Module] = [nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(layers - 2):
                mods += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
            mods += [nn.Linear(hidden, dim)]
            self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def _ema_update(target: nn.Module, online: nn.Module, m: float) -> None:
    """EMA over floating-point tensors (params AND buffers — covers BatchNorm
    running_mean/var). Integer buffers (num_batches_tracked) are copied."""
    t_sd, o_sd = target.state_dict(), online.state_dict()
    for k, t in t_sd.items():
        o = o_sd[k]
        if t.is_floating_point():
            t.mul_(m).add_(o.detach(), alpha=1.0 - m)
        else:
            t.copy_(o)


def _ema_momentum(step: int, total: int, m0: float, m1: float) -> float:
    """Linear ramp m0 -> m1 over training — matches I-JEPA's momentum_scheduler
    (src/train.py: ema[0] + i*(ema[1]-ema[0])/total). step=0 -> m0, step=total -> m1."""
    if total <= 1:
        return m1
    return m0 + (m1 - m0) * min(step / total, 1.0)


def _forward_enc(enc, batch):
    return enc(batch.x, batch.edge_index, getattr(batch, "edge_attr", None),
               batch=batch.batch, rwse=getattr(batch, "rwse", None))


@torch.no_grad()
def _recalibrate_bn(enc: nn.Module, loader, device, n_batches: int = 0) -> None:
    """Recalibrate encoder BatchNorm running stats to the FULL-graph input
    distribution. The EMA target's BN buffers tracked the online encoder, which saw
    MASKED inputs — mismatched for downstream (which feeds full graphs). One pass of
    full-graph forwards with cumulative averaging fixes the stats (matters for the
    frozen probe, where BN stays frozen)."""
    bns = [m for m in enc.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    saved = []
    for m in bns:
        m.reset_running_stats()
        saved.append(m.momentum)
        m.momentum = None          # cumulative moving average
    enc.train()
    seen = 0
    for i, batch in enumerate(tqdm(loader, desc="BN recalib", leave=False, dynamic_ncols=True)):
        if n_batches and i >= n_batches:
            break
        _forward_enc(enc, batch.to(device))
        seen += 1
    for m, mom in zip(bns, saved):
        m.momentum = mom
    enc.eval()
    logger.info(f"BN recalibration on full graphs: {len(bns)} BN layers over {seen} batches")


def main() -> None:
    ap = argparse.ArgumentParser(description="Node-masked JEPA SSL pretraining for the GNN encoder")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else load_default_config()
    set_seed(cfg.train.seed, deterministic=getattr(cfg.train, "deterministic", False))
    setup_logging(getattr(cfg.train, "log_dir", "logs"))
    device = get_device(cfg.train.device)

    # JEPA hyperparameters — read from train.* via getattr (no config schema change).
    mask_ratio  = float(getattr(cfg.train, "jepa_mask_ratio", 0.5))
    ema_start   = float(getattr(cfg.train, "jepa_ema_start", 0.996))
    ema_end     = float(getattr(cfg.train, "jepa_ema_end", 1.0))
    loss_type   = str(getattr(cfg.train, "jepa_loss", "smoothl1"))
    pred_layers = int(getattr(cfg.train, "jepa_predictor_layers", 2))
    pred_hidden = int(getattr(cfg.train, "jepa_predictor_hidden", cfg.model.hidden_dim))
    pred_drop   = float(getattr(cfg.train, "jepa_predictor_dropout", 0.0))
    target_ln   = bool(getattr(cfg.train, "jepa_target_layernorm", True))   # I-JEPA: layer_norm targets over feature dim
    epochs      = int(getattr(cfg.train, "jepa_epochs", getattr(cfg.train, "epochs", 100)))
    lr          = float(getattr(cfg.train, "jepa_lr", 1e-3))
    wd          = float(getattr(cfg.train, "weight_decay", 0.0))
    grad_clip   = float(getattr(cfg.train, "grad_clip", 0.0))
    out_dir     = Path(getattr(cfg.train, "jepa_out_dir", "checkpoints/jepa/run"))

    # Reuse the standard dataset/loader (labels ignored for SSL).
    session = TrainingSession(cfg)
    dataset, loaders, _ = session._setup_dataset()
    train_loader = loaders[0]
    in_channels = dataset[0].x.size(1)
    logger.info(
        f"JEPA pretrain: {len(dataset)} graphs | in_channels={in_channels} | "
        f"mask_ratio={mask_ratio} loss={loss_type} target_ln={target_ln} epochs={epochs} "
        f"ema={ema_start}->{ema_end} predictor={pred_layers}L/{pred_hidden}"
    )

    # Build the full detector ONLY to grab an identically-constructed encoder
    # (live_lm=none → no LM branch). Guarantees downstream load_state_dict match.
    model = build_model(cfg, in_channels).to(device)
    online = model.encoder
    target = copy.deepcopy(online).to(device)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()

    predictor = JEPAPredictor(cfg.model.hidden_dim, pred_hidden, pred_layers, pred_drop).to(device)
    mask_token = nn.Parameter(torch.zeros(in_channels, device=device))
    nn.init.normal_(mask_token, std=0.02)

    params = list(online.parameters()) + list(predictor.parameters()) + [mask_token]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    total_steps = max(1, len(train_loader) * epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    use_amp = bool(getattr(cfg.train, "use_amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    step = 0
    m = ema_start
    epoch_times = []
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        online.train()
        predictor.train()
        run_loss = run_std = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch = batch.to(device)
            x = batch.x
            N = x.size(0)
            mask = torch.rand(N, device=device) < mask_ratio          # uniform node mask (GraphMAE)
            if not mask.any():
                mask[torch.randint(N, (1,), device=device)] = True
            x_ctx = x.clone()
            x_ctx[mask] = mask_token.to(x.dtype)                      # masked features → [MASK]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                h_ctx = online(x_ctx, batch.edge_index, getattr(batch, "edge_attr", None),
                               batch=batch.batch, rwse=getattr(batch, "rwse", None))
                with torch.no_grad():
                    h_tgt = _forward_enc(target, batch)
                    if target_ln:
                        h_tgt = F.layer_norm(h_tgt, (h_tgt.size(-1),))   # I-JEPA target normalization
                pred = predictor(h_ctx[mask])
                tgt = h_tgt[mask].detach()
                if loss_type == "cosine":
                    loss = (1.0 - F.cosine_similarity(pred.float(), tgt.float(), dim=-1)).mean()
                else:
                    loss = F.smooth_l1_loss(pred, tgt)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            m = _ema_momentum(step, total_steps, ema_start, ema_end)
            _ema_update(target, online, m)
            step += 1

            _l = loss.item()
            _s = tgt.float().std(dim=0).mean().item()                 # collapse monitor
            run_loss += _l
            run_std += _s
            n_batches += 1
            pbar.set_postfix(loss=f"{_l:.4f}", tgt_std=f"{_s:.3f}", m=f"{m:.4f}")

        avg_loss = run_loss / max(1, n_batches)
        avg_std = run_std / max(1, n_batches)
        ep_time = time.time() - t_epoch
        epoch_times.append(ep_time)
        logger.info(
            f"epoch {epoch:3d}/{epochs} | jepa_loss={avg_loss:.5f} | "
            f"target_std={avg_std:.4f} | ema_m={m:.5f} | lr={sched.get_last_lr()[0]:.2e} | "
            f"epoch_time={ep_time:.1f}s"
        )
        history.append({"epoch": epoch, "jepa_loss": avg_loss, "target_std": avg_std,
                        "ema_m": m, "epoch_time_s": ep_time})
        if avg_std < 1e-3:
            logger.warning(f"target_std={avg_std:.5f} near 0 — possible representation collapse")

    # Recalibrate target BN stats on full graphs, then save the EMA encoder (the one
    # the paper uses downstream). Also save online for reference.
    _recalibrate_bn(target, train_loader, device)
    ema_path = out_dir / "encoder_ema.pt"
    torch.save(target.state_dict(), ema_path)
    torch.save(online.state_dict(), out_dir / "encoder_online.pt")
    with open(out_dir / "jepa_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": str(args.config), "epochs": epochs, "mask_ratio": mask_ratio,
            "loss": loss_type, "target_layernorm": target_ln,
            "ema_start": ema_start, "ema_end": ema_end,
            "in_channels": in_channels, "hidden_dim": cfg.model.hidden_dim,
            "predictor_layers": pred_layers, "predictor_hidden": pred_hidden,
            "avg_epoch_time_s": sum(epoch_times) / max(1, len(epoch_times)),
            "total_time_s": sum(epoch_times),
            "history": history,
        }, f, indent=2)
    _avg = sum(epoch_times) / max(1, len(epoch_times))
    logger.info(f"saved EMA encoder → {ema_path}")
    logger.info(f"pretrain done: avg_epoch={_avg:.1f}s | total={sum(epoch_times)/60:.1f}min")
    logger.info(f"downstream: set train.gnn_init_checkpoint: {ema_path}")


if __name__ == "__main__":
    main()
