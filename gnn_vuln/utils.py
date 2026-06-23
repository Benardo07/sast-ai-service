"""Utility helpers shared across the project."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed applied to python `random`, numpy, torch (CPU+CUDA),
        cuDNN, and PYTHONHASHSEED.
    deterministic : bool
        When True, also enable bit-exact CUDA determinism. This makes scatter
        atomics, FlashAttention-2 backward, and non-deterministic cuBLAS
        kernels produce identical output across runs. **Significant slowdown**
        (typically 20-40% on live-LM training) — only enable when you need
        bit-exact reproducibility (e.g. replication studies). For statistical
        ablation comparisons, prefer multi-seed runs with this flag off.

    Notes
    -----
    - `warn_only=False` on `torch.use_deterministic_algorithms` — raises
      RuntimeError on any non-deterministic op so issues are caught immediately.
      In PyTorch 2.x, scatter_add_ and scatter_reduce_(mean/sum/amax) are all
      deterministic; only scatter_reduce_(prod) raises.
    - `CUBLAS_WORKSPACE_CONFIG=:16:8` is required for deterministic cuBLAS
      matmul/GEMM operations (16 MB workspace × 8 streams = ~128 MB total).
      `:4096:8` also works but reserves 4096 MB × 8 streams → OOM on training GPUs.
      Set before CUDA context is initialized — we set it here before any torch.cuda call.
    - `FLASH_ATTENTION_DETERMINISTIC=1` triggers FA2 v2.4.1+ deterministic
      backward pass in HuggingFace transformers.
    """
    # Set CUBLAS env var BEFORE any CUDA call that might init the CUDA context.
    # torch.cuda.manual_seed_all() below initializes the CUDA context, so this
    # must come first or cuBLAS determinism has no effect.
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: don't crash on ops without deterministic impl (e.g. some
        # PyG scatter ops). Just log a warning and continue.
        torch.use_deterministic_algorithms(True, warn_only=False)
        logger.info(
            f"Random seed set to {seed} | deterministic=True "
            "(CUDA atomics + FA2 + cuDNN deterministic; 20-40% slowdown)"
        )
    else:
        # Previous default behavior — cudnn deterministic kept on for cuDNN
        # conv op reproducibility (cheap), benchmark off (avoids picking
        # different algorithms across runs).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {seed} | deterministic=False (fast mode)")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_dir: str | Path, level: str = "INFO") -> None:
    """Configure loguru to write to console and a rotating file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "run_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}",
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(model: torch.nn.Module, path: str | Path, **extra) -> None:
    """Save model weights + optional metadata (no optimizer — for best-model checkpoints)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **extra}, path)
    logger.info(f"Checkpoint saved → {path}")


def save_resume_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    patience_counter: int,
    **extra,
) -> None:
    """
    Save full training state so that training can be resumed exactly.
    Writes to a temp file first then renames, so a crash mid-write never
    corrupts the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            **extra,
        },
        tmp,
    )
    tmp.replace(path)
    logger.debug(f"Resume checkpoint saved → {path}")


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: str = "cpu") -> dict:
    """Load model weights from a best-model checkpoint. Returns metadata dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    logger.info(f"Checkpoint loaded ← {path}")
    return {k: v for k, v in state.items() if k != "model_state_dict"}


def load_checkpoint_expandable(model: torch.nn.Module, path: str | Path, device: str = "cpu") -> dict:
    """
    Load weights into a model whose classifier head may be LARGER than the
    checkpoint's (class-incremental head expansion). Parameters with matching
    shape are copied exactly. Parameters whose shape differs only by growth
    (e.g. func_head.weight [26, F] -> [36, F]) get their overlapping leading
    slice copied; the new rows keep their fresh initialization. Missing keys are
    left at init. Use for CIL task-B init from a task-A checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    msd = model.state_dict()
    exact, expanded, skipped = 0, [], []
    with torch.no_grad():
        for name, dst in msd.items():
            src = sd.get(name)
            if src is None:
                skipped.append(name)
                continue
            if src.shape == dst.shape:
                dst.copy_(src)
                exact += 1
            elif src.dim() == dst.dim() and all(s <= d for s, d in zip(src.shape, dst.shape)):
                sl = tuple(slice(0, s) for s in src.shape)   # copy overlapping leading block
                dst[sl].copy_(src)
                expanded.append((name, tuple(src.shape), tuple(dst.shape)))
            else:
                skipped.append(name)
    model.load_state_dict(msd)
    logger.info(
        f"Expandable load ← {path}: {exact} exact, "
        f"{len(expanded)} expanded {[e[0] for e in expanded]}, {len(skipped)} left-at-init"
    )
    return {k: v for k, v in state.items() if k != "model_state_dict"}


def load_resume_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cpu",
) -> dict:
    """
    Restore full training state from a resume checkpoint.
    Returns metadata dict with epoch, best_val_loss, patience_counter, etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    meta = {k: v for k, v in state.items()
            if k not in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")}
    best_display = meta.get("best_val_f1", meta.get("best_val_loss", float("nan")))
    best_key = "best_val_f1" if "best_val_f1" in meta else "best_val_loss"
    logger.info(
        f"Resumed from {path}  "
        f"(epoch {meta['epoch']}, {best_key}={best_display:.4f})"
    )
    return meta


# ---------------------------------------------------------------------------
# Checkpoint manager (OOP wrapper)
# ---------------------------------------------------------------------------


class CheckpointManager:
    """
    Manages best + resume checkpoints for a single training run.

    Usage
    -----
        cm = CheckpointManager(run_dir, arch="lmgat_codebert")
        cm.save_best(model, epoch=10, val_f1=0.72)
        cm.save_last(model, optimizer, scheduler, epoch=10, ...)
        meta = cm.load_resume(model, optimizer, scheduler, device="cuda")
    """

    def __init__(self, run_dir: str | Path, arch: str) -> None:
        self.run_dir = Path(run_dir)
        self.arch    = arch
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self) -> Path:
        return self.run_dir / f"best_{self.arch}.pt"

    @property
    def last_path(self) -> Path:
        return self.run_dir / f"last_{self.arch}.pt"

    def save_best(self, model: torch.nn.Module, **meta) -> None:
        save_checkpoint(model, self.best_path, **meta)

    def save_last(self, model, optimizer, scheduler, epoch: int,
                  best_val_loss: float, patience_counter: int, **extra) -> None:
        save_resume_checkpoint(self.last_path, model, optimizer, scheduler,
                               epoch=epoch, best_val_loss=best_val_loss,
                               patience_counter=patience_counter, **extra)

    def load_model(self, model: torch.nn.Module, device: str = "cpu") -> dict:
        return load_checkpoint(model, self.best_path, device)

    def load_resume(self, model, optimizer, scheduler, device: str = "cpu") -> dict:
        return load_resume_checkpoint(self.last_path, model, optimizer, scheduler, device)

    def has_resume(self) -> bool:
        return self.last_path.exists()


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def get_device(requested: str = "cpu") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if requested == "cuda":
            logger.warning("CUDA requested but not available — falling back to CPU.")
        else:
            logger.info("Using CPU.")
    return device
