"""
train.py — Training entry point.

Usage (via uv):
    uv run train --config configs/lmgcn/binary.yaml
    uv run train --config configs/lmgat/binary.yaml
    uv run train --config configs/lmgat_codebert/multiclass.yaml
    uv run train --config configs/lmgat_mcs/multiclass.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch_geometric.loader import DataLoader
from loguru import logger

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset
from gnn_vuln.losses import HierarchicalSupConLoss
from gnn_vuln.models.registry import build_model, _parse_active_heads
from gnn_vuln.training.ewc import EWCDR
from gnn_vuln.training.losses import epoch_adaptive_class_weights
from gnn_vuln.training.optimizer import build_optimizer_and_scheduler
from gnn_vuln.training.trainer import Trainer
from gnn_vuln.utils import (
    set_seed, setup_logging, get_device,
    CheckpointManager,
)

__all__ = ["build_model", "_parse_active_heads", "TrainingSession"]


# ---------------------------------------------------------------------------
# TrainingSession — encapsulates full training setup + loop
# ---------------------------------------------------------------------------

class TrainingSession:
    """
    Encapsulates a full training run: dataset, model, optimizer, loop, checkpointing.

    Usage
    -----
        session = TrainingSession.from_args(args)
        session.run()
    """

    def __init__(self, cfg: "Config", resume: bool = False) -> None:
        self.cfg    = cfg
        self.resume = resume
        self.device = get_device(cfg.train.device)

        self._active_heads       = _parse_active_heads(cfg)
        self._mil_k              = getattr(cfg.model, "mil_k", 3)
        self._mil_weight         = getattr(cfg.model, "mil_weight", 0.5)
        self._rank_loss_weight   = getattr(cfg.model, "rank_loss_weight", 0.0)
        self._loc_supervised_weight     = getattr(cfg.model, "loc_supervised_weight", 0.0)
        self._loc_supervised_pos_weight = getattr(cfg.model, "loc_supervised_pos_weight", 0.0)
        self._group_loss_weight  = getattr(cfg.model, "group_loss_weight", 0.0)
        self._binary_loss_weight = getattr(cfg.model, "binary_loss_weight", 0.0)
        self._focal_gamma        = getattr(cfg.train, "focal_loss_gamma", 0.0)
        self._grad_clip          = getattr(cfg.train, "grad_clip", 0.0)
        self._is_binary          = getattr(cfg.data, "mode", "binary") == "binary"
        self._use_class_weights  = getattr(cfg.train, "use_class_weights", True)
        # Support old name (livable_loss/livable_adaptive) for backward compat with existing configs
        self._use_epoch_adaptive = (
            getattr(cfg.train, "epoch_adaptive_weights", False) or
            getattr(cfg.train, "livable_adaptive", False) or
            getattr(cfg.train, "livable_loss_old", False)  # not used, just safety
        ) and self._use_class_weights
        self._use_livable_real   = getattr(cfg.train, "livable_loss", False)

        if self._active_heads:
            if "group"  not in self._active_heads: self._group_loss_weight  = 0.0
            if "binary" not in self._active_heads: self._binary_loss_weight = 0.0

        self._use_supcon         = getattr(cfg.model, "use_supcon", False)
        self._supcon_weight      = getattr(cfg.model, "supcon_weight", 0.1) if self._use_supcon else 0.0
        self._self_supcon_weight = getattr(cfg.model, "supcon_self_weight", 0.0) if self._use_supcon else 0.0

    @classmethod
    def from_args(cls, args) -> "TrainingSession":
        paths = args.config if isinstance(args.config, (list, tuple)) else [args.config]
        cfg = (Config.from_yamls(paths)
               if all(Path(p).exists() for p in paths) else load_default_config())
        if getattr(args, "seed", None) is not None:
            cfg.train.seed = args.seed
        if getattr(args, "split_seed", None) is not None:
            cfg.data.split_seed = args.split_seed
        set_seed(cfg.train.seed, deterministic=getattr(cfg.train, "deterministic", False))
        setup_logging(cfg.train.log_dir)
        return cls(cfg, resume=getattr(args, "resume", False))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, config_path: str | Path | None = None) -> None:
        cfg    = self.cfg
        device = self.device

        supcon_fn     = self._build_supcon()
        dataset, loaders, train_idx = self._setup_dataset()
        train_loader, val_loader, test_loader = loaders

        in_channels = dataset[0].x.size(1)
        logger.info(f"Dataset: {len(dataset)} graphs | in_channels={in_channels}")
        if dataset.num_classes != cfg.model.num_classes:
            raise ValueError(
                f"Config model.num_classes={cfg.model.num_classes} but dataset has "
                f"{dataset.num_classes} classes."
            )

        class_weight, train_counts = self._setup_class_weights(dataset, train_idx)
        if self._use_livable_real and class_weight is not None:
            logger.info("livable_loss=true → disabling static class_weight")
            class_weight = None

        model = build_model(cfg, in_channels, self._active_heads).to(device)

        # cRT (decoupled stage 2): load frozen backbone, re-init + unfreeze only
        # func_head. Must run before the optimizer is built so it sees the frozen
        # requires_grad flags. Returns the trainable head (or None when disabled).
        crt_module = self._setup_crt(model)
        # JEPA downstream init (Q-series): load a node-masked-JEPA-pretrained GNN
        # encoder. Mutually exclusive with cRT (cRT takes precedence). Frozen-probe
        # mode returns func_head so the trainer keeps the backbone in eval(), reusing
        # the cRT eval mechanism; finetune mode returns None (normal full training).
        if crt_module is None:
            crt_module = self._setup_jepa_init(model)

        # LSUV init (Mishkin & Matas ICLR 2016) — applied to GNN encoder only.
        # Runs orthonormal init + per-layer variance normalization using first batch.
        if getattr(cfg.model, "gnn_lsuv_init", False) and hasattr(model, "encoder"):
            from gnn_vuln.models.encoders import apply_lsuv_encoder
            try:
                first_batch = next(iter(train_loader)).to(device)
                tol = getattr(cfg.model, "gnn_lsuv_tol", 0.1)
                max_trials = getattr(cfg.model, "gnn_lsuv_max_trials", 10)
                final_vars = apply_lsuv_encoder(model.encoder, first_batch,
                                                tol=tol, max_trials=max_trials)
                logger.info(f"LSUV init done — final variances: "
                            f"{ {k: round(v, 3) for k, v in final_vars.items()} }")
            except Exception as e:
                logger.warning(f"LSUV init failed, falling back to default init: {e}")

        # Vectorized StmtHead: scatter-based pooling, no Python inner loop
        if getattr(cfg.train, "stmt_head_vectorized", False):
            from gnn_vuln.models.heads import StmtHead
            for m in model.modules():
                if isinstance(m, StmtHead):
                    m._vectorized = True
            logger.info("StmtHead: vectorized scatter mode enabled")

        ewc   = self._setup_ewc(model, train_loader, in_channels)
        if ewc is not None and getattr(cfg.ewc, "compute_only", False):
            logger.info("ewc.compute_only=true → importance cache saved, exiting before training.")
            return
        replay_loader, replay_weight = self._setup_replay()

        # torch.compile — fuses kernels, ~20-50% speedup (PyTorch 2.0+, CUDA only)
        if getattr(cfg.train, "compile_model", False) and device.type == "cuda":
            try:
                import torch._dynamo as _dynamo  # alias: avoid rebinding local `torch`
                _dynamo.config.capture_scalar_outputs = True
                model = torch.compile(model, mode="default", dynamic=True)
                logger.info("torch.compile enabled (mode=reduce-overhead)")
            except Exception as e:
                logger.warning(f"torch.compile failed, skipping: {e}")

        use_amp, amp_dtype, scaler = self._setup_amp()
        total_steps = len(train_loader) * cfg.train.epochs
        optimizer, scheduler, step_per_batch = build_optimizer_and_scheduler(model, cfg, total_steps)

        # MTL balance method (Kendall uncertainty / PCGrad). Default "fixed" uses
        # mil_weight/rank_loss_weight from config; kendall learns weights via
        # uncertainty params; pcgrad projects conflicting per-task gradients.
        loss_balance_method = getattr(cfg.train, "loss_balance_method", "fixed")
        uncertainty_weights = None
        if loss_balance_method == "kendall":
            from gnn_vuln.training.mtl_balance import UncertaintyWeights
            # Task list covers all losses that may be present; UncertaintyWeights
            # silently skips tasks not in raw_losses dict for a given batch.
            task_names = ["cls", "group", "binary", "mil", "rank"]
            uncertainty_weights = UncertaintyWeights(task_names).to(device)
            # Add learnable log_sigma² params to optimizer (use same LR as model).
            optimizer.add_param_group({"params": list(uncertainty_weights.parameters())})
            logger.info(f"Kendall uncertainty weighting enabled — tasks: {task_names}")
        elif loss_balance_method == "pcgrad":
            logger.info("PCGrad gradient surgery enabled — AMP-compatible via scaler.scale() per task")

        # ULMFiT gradual unfreezing (optional). When schedule is non-empty,
        # all LM layers are frozen up-front and progressively unfrozen by epoch.
        unfreezer = None
        _unfreeze_schedule = getattr(cfg.train, "lm_unfreeze_schedule", []) or []
        if _unfreeze_schedule and hasattr(model, "codebert") and model.has_live_lm():
            from gnn_vuln.training.unfreezer import GradualUnfreezer
            unfreezer = GradualUnfreezer(model.codebert, _unfreeze_schedule)
            logger.info(
                f"Gradual unfreezing enabled: schedule={_unfreeze_schedule} | "
                f"{unfreezer.n_layers} LM layers detected"
            )

        # EDAT adversarial training setup
        pgd = pgd_tokenizer = None
        if getattr(cfg.train, "use_edat", False):
            if getattr(cfg.train, "compile_model", False):
                logger.warning("use_edat=true conflicts with compile_model=true — EDAT disabled")
            elif not (hasattr(model, "codebert") or
                      (hasattr(model, "module") and hasattr(model.module, "codebert"))):
                logger.warning("use_edat=true but model has no live LM — EDAT disabled")
            else:
                from gnn_vuln.training.pgd import EmbeddingPGD
                pgd = EmbeddingPGD(
                    model,
                    epsilon=getattr(cfg.train, "edat_epsilon", 0.02),
                    alpha=getattr(cfg.train,   "edat_alpha",   1e-2),
                    n_steps=getattr(cfg.train, "edat_steps",   3),
                )
                # Reuse dataset's func tokenizer (initialised on first data access)
                if hasattr(dataset, "_tok_cache") and dataset._tok_cache is not None:
                    pgd_tokenizer = dataset._tok_cache
                else:
                    from gnn_vuln.data.dataset_lm import _load_tokenizer
                    _func_lm = getattr(cfg.model, "func_lm", "") or getattr(
                        cfg.model, "pretrained_lm", "microsoft/unixcoder-base"
                    )
                    pgd_tokenizer = _load_tokenizer(_func_lm)
                logger.info("EDAT enabled — adversarial identifier perturbation active")

        grad_accum_steps = getattr(cfg.train, "grad_accum_steps", 1)
        trainer = Trainer(
            model=model, optimizer=optimizer, scheduler=scheduler,
            step_per_batch=step_per_batch, device=device,
            mil_k=self._mil_k, mil_weight=self._mil_weight,
            rank_loss_weight=self._rank_loss_weight, focal_gamma=self._focal_gamma,
            loc_supervised_weight=self._loc_supervised_weight,
            loc_supervised_pos_weight=self._loc_supervised_pos_weight,
            group_loss_weight=self._group_loss_weight, binary_loss_weight=self._binary_loss_weight,
            supcon_fn=supcon_fn, supcon_weight=self._supcon_weight,
            self_supcon_weight=self._self_supcon_weight,
            use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler, ewc=ewc,
            replay_loader=replay_loader, replay_weight=replay_weight,
            grad_accum_steps=grad_accum_steps,
            label_smoothing=getattr(cfg.train, "label_smoothing", 0.0),
            use_livable_real=self._use_livable_real,
            livable_focal_gamma=getattr(cfg.train, "focal_loss_gamma", 2.0),
            livable_label_smoothing=getattr(cfg.train, "label_smoothing", 0.1),
            pgd=pgd, pgd_tokenizer=pgd_tokenizer,
            loss_balance_method=loss_balance_method,
            uncertainty_weights=uncertainty_weights,
            mtl_diagnose=getattr(cfg.train, "mtl_diagnose", False),
            mtl_diagnose_every=getattr(cfg.train, "mtl_diagnose_every", 10),
        )
        trainer.set_grad_clip(self._grad_clip)
        if crt_module is not None:
            trainer.set_crt_mode(crt_module)

        # Balanced-Mixup / Remix: pass per-class train counts for imbalance-aware
        # label mixing. The model does the feature mix (mixup_alpha); the trainer
        # builds the two-target loss.
        if getattr(cfg.train, "mixup_alpha", 0.0) > 0.0:
            _counts = train_counts
            if _counts is None:
                _all_y = dataset.get_all_labels()
                _tl = _all_y[torch.tensor(train_idx, dtype=torch.long)]
                _counts = torch.bincount(_tl, minlength=cfg.model.num_classes).float()
            trainer.set_mixup(
                remix=getattr(cfg.train, "mixup_remix", True),
                kappa=getattr(cfg.train, "mixup_remix_kappa", 3.0),
                tau=getattr(cfg.train, "mixup_remix_tau", 0.5),
                counts=_counts.to(device),
            )
            logger.info(
                f"Balanced-Mixup enabled: alpha={cfg.train.mixup_alpha} | "
                f"remix={getattr(cfg.train, 'mixup_remix', True)} | "
                f"kappa={getattr(cfg.train, 'mixup_remix_kappa', 3.0)} | "
                f"tau={getattr(cfg.train, 'mixup_remix_tau', 0.5)}"
            )

        # Logit Adjustment loss: pass log(class priors) so CE is offset by tau*log(pi_y).
        if getattr(cfg.train, "logit_adjustment", False):
            _la_counts = train_counts
            if _la_counts is None:
                _all_y = dataset.get_all_labels()
                _tl = _all_y[torch.tensor(train_idx, dtype=torch.long)]
                _la_counts = torch.bincount(_tl, minlength=cfg.model.num_classes).float()
            _prior = _la_counts / _la_counts.sum()
            _log_prior = torch.log(_prior.clamp(min=1e-12)).to(device)
            _la_tau = getattr(cfg.train, "logit_adjustment_tau", 1.0)
            trainer.set_logit_adjustment(_log_prior, _la_tau)
            logger.info(f"Logit Adjustment loss enabled: tau={_la_tau}")

        # FLAG adversarial node-feature training (Kong et al. 2020).
        if getattr(cfg.train, "use_flag", False):
            if getattr(cfg.train, "use_amp", False):
                logger.warning("use_flag=true with use_amp=true — set use_amp=false for FLAG")
            _flag_ss = getattr(cfg.train, "flag_step_size", 1e-3)
            _flag_steps = getattr(cfg.train, "flag_steps", 3)
            trainer.set_flag(_flag_ss, _flag_steps)
            logger.info(f"FLAG enabled: step_size={_flag_ss} steps={_flag_steps}")

        run_id, run_dir = self._setup_run_dir()
        if config_path:
            paths = config_path if isinstance(config_path, (list, tuple)) else [config_path]
            for i, p in enumerate(paths):
                if Path(p).exists():
                    name = "config.yaml" if len(paths) == 1 else f"config_{i}_{Path(p).name}"
                    shutil.copy(p, run_dir / name)

        cm = CheckpointManager(run_dir, cfg.model.architecture)
        stop_on_f1 = getattr(cfg.train, "early_stop_metric", "f1") == "f1"
        best_val_f1 = -1.0; best_val_loss = float("inf"); patience_counter = 0; start_epoch = 1

        if self.resume and cm.has_resume():
            meta = cm.load_resume(model, optimizer, scheduler, device=str(device))
            start_epoch      = meta["epoch"] + 1
            best_val_f1      = meta.get("best_val_f1",  -1.0)
            best_val_loss    = meta.get("best_val_loss", float("inf"))
            patience_counter = meta["patience_counter"]
            logger.info(f"Resuming from epoch {start_epoch}")
        elif self.resume:
            logger.warning(f"--resume: {cm.last_path} not found — starting fresh.")

        self._training_loop(
            trainer, train_loader, val_loader, test_loader,
            cm, class_weight, train_counts, stop_on_f1,
            best_val_f1, best_val_loss, patience_counter, start_epoch,
            step_per_batch, optimizer, scheduler, unfreezer,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_supcon(self):
        if not self._use_supcon:
            return None
        cfg = self.cfg
        _cwe_vocab_path = (
            Path(getattr(cfg.data, "processed_dir", Path("data/processed"))).parent
            / "raw" / getattr(cfg.data, "source", "megavul") / "cwe_vocab.json"
        )
        _cwe_vocab = None
        _dist_matrix_path = None
        if getattr(cfg.model, "supcon_use_distance_matrix", False):
            if _cwe_vocab_path.exists():
                with open(_cwe_vocab_path, encoding="utf-8") as f:
                    _cwe_vocab = json.load(f)
            _p = Path(getattr(cfg.model, "cwe_dist_matrix", "data/cwe/cwe_distance_matrix.json"))
            _dist_matrix_path = _p if _p.exists() else None
        fn = HierarchicalSupConLoss(
            temperature=getattr(cfg.model, "supcon_temperature", 0.07),
            alpha=getattr(cfg.model, "supcon_alpha", 0.5),
            dist_matrix_path=_dist_matrix_path, cwe_vocab=_cwe_vocab,
            weight_fn=getattr(cfg.model, "supcon_weight_fn", "linear"),
            exp_scale=getattr(cfg.model, "supcon_exp_scale", 5.0),
            power=getattr(cfg.model, "supcon_power", 2.0),
            min_weight=getattr(cfg.model, "supcon_min_weight", 0.0),
            intragroup_only=getattr(cfg.model, "supcon_intragroup_only", True),
            self_temperature=getattr(cfg.model, "supcon_self_temperature", 0.5),
            class_averaging=getattr(cfg.model, "supcon_class_averaging", False),
        )
        return fn.to(self.device)

    def _setup_crt(self, model):
        """Decoupled cRT (Kang et al. 2020, ICLR) stage-2 setup.

        Loads a frozen backbone checkpoint, optionally re-initializes the
        classifier head, then freezes every parameter except ``func_head``.
        Returns the trainable head module (so the trainer can keep the backbone
        in eval()), or None when cRT is disabled.
        """
        cfg = self.cfg
        ckpt = getattr(cfg.train, "crt_init_checkpoint", "") or ""
        if not ckpt:
            return None
        from gnn_vuln.utils import load_checkpoint
        p = Path(ckpt)
        if not p.exists():
            raise FileNotFoundError(f"crt_init_checkpoint not found: {p}")
        load_checkpoint(model, p, device=str(self.device))
        logger.info(f"cRT: loaded frozen backbone ← {p}")

        if getattr(cfg.train, "crt_reinit_head", True):
            n_reinit = 0
            for m in model.func_head.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
                    n_reinit += 1
            logger.info(f"cRT: re-initialized func_head ({n_reinit} layer(s) reset)")

        n_train = n_frozen = 0
        for name, prm in model.named_parameters():
            if name.startswith("func_head."):
                prm.requires_grad = True
                n_train += prm.numel()
            else:
                prm.requires_grad = False
                n_frozen += prm.numel()
        logger.info(
            f"cRT: trainable={n_train:,} (func_head) | frozen={n_frozen:,} (backbone) | "
            f"class_balanced_sampling={getattr(cfg.train, 'class_balanced_sampling', False)}"
        )
        return model.func_head

    def _setup_jepa_init(self, model):
        """JEPA downstream init (Q-series). Loads a node-masked-JEPA-pretrained GNN
        encoder (gnn_vuln.pretrain_jepa → encoder_ema.pt) into ``model.encoder``.

        freeze_gnn=false → finetune: encoder is just initialized, all params trainable
        (returns None → normal training).
        freeze_gnn=true → frozen linear probe: freeze every param except ``func_head``,
        re-init the head, and return it so the trainer keeps the backbone in eval()
        (reuses the cRT mechanism). This is the canonical JEPA SSL-quality measure.
        Returns the trainable head module, or None when finetuning / disabled.
        """
        cfg = self.cfg
        ckpt = getattr(cfg.train, "gnn_init_checkpoint", "") or ""
        if not ckpt:
            return None
        if not hasattr(model, "encoder"):
            raise ValueError("train.gnn_init_checkpoint set but model has no .encoder")
        p = Path(ckpt)
        if not p.exists():
            raise FileNotFoundError(f"gnn_init_checkpoint not found: {p}")
        sd = torch.load(p, map_location=str(self.device))
        missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
        logger.info(
            f"JEPA: loaded GNN encoder ← {p} "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )
        if missing or unexpected:
            logger.warning(
                f"JEPA encoder load mismatch — missing={list(missing)} unexpected={list(unexpected)}"
            )

        if not getattr(cfg.train, "freeze_gnn", False):
            logger.info("JEPA: finetune mode — encoder initialized, all params trainable")
            return None

        # Frozen linear probe — mirror cRT: re-init head, freeze all but func_head.
        if getattr(cfg.train, "crt_reinit_head", True):
            n_reinit = 0
            for m in model.func_head.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
                    n_reinit += 1
            logger.info(f"JEPA frozen probe: re-initialized func_head ({n_reinit} layer(s) reset)")
        n_train = n_frozen = 0
        for name, prm in model.named_parameters():
            if name.startswith("func_head."):
                prm.requires_grad = True
                n_train += prm.numel()
            else:
                prm.requires_grad = False
                n_frozen += prm.numel()
        logger.info(
            f"JEPA frozen probe: trainable={n_train:,} (func_head) | frozen={n_frozen:,} (encoder+rest)"
        )
        return model.func_head

    def _setup_dataset(self):
        cfg = self.cfg
        pretrained_lm   = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
        func_lm         = getattr(cfg.model, "func_lm", "") or pretrained_lm
        add_func_tokens = getattr(cfg.model, "add_func_tokens", False)
        func_lm_source  = getattr(cfg.model, "func_lm_source", "raw")
        source_val      = getattr(cfg.data, "source_val",  "")
        source_test     = getattr(cfg.data, "source_test", "")
        use_official    = bool(source_val and source_test)
        _use_balanced   = self._use_supcon and getattr(cfg.train, "supcon_balanced_sampling", False)
        _classes_per_batch = getattr(cfg.train, "supcon_classes_per_batch", 8)
        _use_cb_sampling = getattr(cfg.train, "class_balanced_sampling", False)

        kwargs = dict(
            root=str(cfg.data.processed_dir.parent), max_nodes=cfg.data.max_nodes,
            embedder_device=str(self.device), mode=cfg.data.mode,
            pretrained_lm=pretrained_lm, func_lm=func_lm,
            add_func_tokens=add_func_tokens, func_lm_source=func_lm_source,
            top_cwe=getattr(cfg.data, "top_cwe", 0),
            cwe_list=getattr(cfg.data, "cwe_list", None),
            cwe_groups=getattr(cfg.data, "cwe_groups", None),
            filter_owasp=getattr(cfg.data, "filter_owasp", False),
            filter_top25_dangerous=getattr(cfg.data, "filter_top25_dangerous", False),
            max_per_class=getattr(cfg.data, "max_per_class", 0),
            resample_seed=getattr(cfg.data, "resample_seed", 42),
            func_max_length=getattr(cfg.model, "func_max_length", 512),
            storage=getattr(cfg.data, "storage", "inmemory"),
            precompute_line_cls=getattr(cfg.model, "precompute_line_cls", False),
            ds_name_suffix=getattr(cfg.data, "ds_name_suffix", ""),
            target_vocab=getattr(cfg.data, "target_vocab", None),
        )
        bs          = cfg.train.batch_size
        num_workers = getattr(cfg.train, "num_workers",    4)
        prefetch    = getattr(cfg.train, "prefetch_factor", 2)
        pin_mem     = self.device.type == "cuda"

        # Strip heavy func-token tensors from collation when there's no live LM.
        # lmgat_codebert with live_lm=none uses no func_input_ids but the _ft
        # dataset still carries them — 64×1024 token stacks per batch for nothing.
        _needs_func_tokens = getattr(cfg.model, "live_lm", "func") != "none"
        _precompute_line_cls = getattr(cfg.model, "precompute_line_cls", False)
        _line_context_lines  = getattr(cfg.model, "line_context_lines", 0)
        _FUNC_TOKEN_KEYS = ("func_input_ids", "func_attention_mask", "func_token_lines")
        # follow_batch=['func_line_cls'] → Batch.from_data_list creates func_line_cls_batch
        # [total_lines] with graph index per line, needed when using cached line embeddings.
        _follow_batch = ["func_line_cls"] if _precompute_line_cls else []
        def _strip_collate_fn(batch):
            from torch_geometric.data import Batch
            if not _needs_func_tokens:
                for g in batch:
                    for k in _FUNC_TOKEN_KEYS:
                        if hasattr(g, k):
                            delattr(g, k)
            return Batch.from_data_list(batch, follow_batch=_follow_batch)

        _seed = cfg.train.seed

        def _worker_init_fn(worker_id):
            # torch already sets initial_seed() = base_seed + worker_id per worker
            worker_seed = torch.initial_seed() % (2 ** 32)
            import random as _random
            import numpy as _np
            _random.seed(worker_seed)
            _np.random.seed(worker_seed)

        _g = torch.Generator().manual_seed(_seed)

        dl_kw = dict(
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch if num_workers > 0 else None,
            collate_fn=_strip_collate_fn,
            worker_init_fn=_worker_init_fn,
            generator=_g,
        )

        dataset = CodeBERTGraphDataset(source=getattr(cfg.data, "source", "bigvul"), **kwargs)
        _dataset_pt = Path(dataset.processed_paths[0]).name
        self._dataset_pt = _dataset_pt
        if getattr(cfg.data, "split_file", ""):
            dataset.load_split_file(cfg.data.split_file)
        if use_official:
            val_ds  = CodeBERTGraphDataset(source=source_val,  **kwargs)
            test_ds = CodeBERTGraphDataset(source=source_test, **kwargs)
            if _precompute_line_cls:
                _lm_dev = str(self.device)
                dataset.precompute_line_cls_all(_lm_dev, context_lines=_line_context_lines)
                val_ds.precompute_line_cls_all(_lm_dev, context_lines=_line_context_lines)
                test_ds.precompute_line_cls_all(_lm_dev, context_lines=_line_context_lines)
            train_idx = list(range(len(dataset)))
            if _use_balanced:
                from gnn_vuln.training.sampler import SupConBalancedSampler
                _all_labels = dataset.get_all_labels().tolist()
                _bs = SupConBalancedSampler(_all_labels, bs, _classes_per_batch, seed=_seed)
                logger.info(f"SupConBalancedSampler: {_bs.classes_per_batch} classes × {_bs.samples_per_class} samples/class per batch")
                train_dl = DataLoader(dataset, batch_sampler=_bs, **dl_kw)
            elif _use_cb_sampling:
                from gnn_vuln.training.sampler import class_balanced_sampler
                _cb_labels = dataset.get_all_labels().tolist()
                _cb = class_balanced_sampler(_cb_labels, seed=_seed)
                logger.info(f"class_balanced_sampler (cRT): {len(set(_cb_labels))} classes | {len(_cb_labels)} draws/epoch")
                train_dl = DataLoader(dataset, batch_size=bs, sampler=_cb, **dl_kw)
            else:
                train_dl = DataLoader(dataset, batch_size=bs, shuffle=True, **dl_kw)
            loaders = (
                train_dl,
                DataLoader(val_ds,  batch_size=bs, **dl_kw),
                DataLoader(test_ds, batch_size=bs, **dl_kw),
            )
        else:
            if _precompute_line_cls:
                dataset.precompute_line_cls_all(str(self.device), context_lines=_line_context_lines)
            train_idx, val_idx, test_idx = dataset.get_splits(
                train_ratio=getattr(cfg.data, "train_ratio", 0.8),
                val_ratio=getattr(cfg.data, "val_ratio", 0.1),
                seed=getattr(cfg.data, "split_seed", None) or cfg.train.seed,
            )
            if _use_balanced:
                from gnn_vuln.training.sampler import SupConBalancedSampler
                _all_labels = dataset.get_all_labels()
                _train_labels = _all_labels[torch.tensor(train_idx, dtype=torch.long)].tolist()
                _bs = SupConBalancedSampler(_train_labels, bs, _classes_per_batch, seed=_seed)
                logger.info(f"SupConBalancedSampler: {_bs.classes_per_batch} classes × {_bs.samples_per_class} samples/class per batch")
                train_dl = DataLoader(dataset[train_idx], batch_sampler=_bs, **dl_kw)
            elif _use_cb_sampling:
                from gnn_vuln.training.sampler import class_balanced_sampler
                _all_labels = dataset.get_all_labels()
                _cb_labels = _all_labels[torch.tensor(train_idx, dtype=torch.long)].tolist()
                _cb = class_balanced_sampler(_cb_labels, seed=_seed)
                logger.info(f"class_balanced_sampler (cRT): {len(set(_cb_labels))} classes | {len(_cb_labels)} draws/epoch")
                train_dl = DataLoader(dataset[train_idx], batch_size=bs, sampler=_cb, **dl_kw)
            else:
                train_dl = DataLoader(dataset[train_idx], batch_size=bs, shuffle=True, **dl_kw)
            loaders = (
                train_dl,
                DataLoader(dataset[val_idx],   batch_size=bs, **dl_kw),
                DataLoader(dataset[test_idx],  batch_size=bs, **dl_kw),
            )
        # Stash split for split.json (val/test only exist on the seeded-split branch)
        self._split_dataset = dataset
        self._split_train_idx = train_idx
        self._split_val_idx = locals().get("val_idx")
        self._split_test_idx = locals().get("test_idx")
        return dataset, loaders, train_idx

    def _setup_class_weights(self, dataset, train_idx):
        if not self._use_class_weights:
            return None, None
        cfg = self.cfg
        all_y = dataset.get_all_labels()
        train_labels = all_y[torch.tensor(train_idx, dtype=torch.long)]
        counts = torch.bincount(train_labels, minlength=cfg.model.num_classes).float()
        if self._use_epoch_adaptive:
            w = epoch_adaptive_class_weights(counts, 1, cfg.train.epochs, cfg.model.num_classes, self.device)
        else:
            w = torch.clamp(counts.sum() / (counts * cfg.model.num_classes), max=10.0).to(self.device)
        return w, counts

    def _setup_ewc(self, model, train_loader, in_channels):
        cfg = self.cfg
        _ewc_cfg = getattr(cfg, "ewc", None)
        if _ewc_cfg is None or not getattr(_ewc_cfg, "enabled", False):
            return None
        _ewc_weight  = getattr(_ewc_cfg, "weight",             1000.0)
        _ewc_scope   = getattr(_ewc_cfg, "scope",              "all")
        _ewc_cache   = getattr(_ewc_cfg, "importance_cache",   "")
        _ewc_ckpt    = getattr(_ewc_cfg, "source_checkpoint",  "")
        _ewc_nbatch  = getattr(_ewc_cfg, "n_batches",          0)
        # Always initialise from task-A weights first — continual learning starts
        # from the previous task. Must happen BEFORE the cache early-return so task-B
        # runs (cache already present) still load task-A weights. Expandable load
        # handles CIL head growth (e.g. 26→36): old-class rows copied, new rows at init.
        from gnn_vuln.utils import load_checkpoint_expandable as _lce
        if _ewc_ckpt and Path(_ewc_ckpt).exists():
            _lce(model, _ewc_ckpt, device=str(self.device))
        if _ewc_cache and Path(_ewc_cache).exists():
            return EWCDR.from_file(_ewc_cache, ewc_weight=_ewc_weight)
        if not _ewc_ckpt or not Path(_ewc_ckpt).exists():
            raise ValueError(f"EWC enabled but source_checkpoint not found: {_ewc_ckpt!r}")
        ewc = EWCDR(model=model, dataloader=train_loader, device=self.device,
                    ewc_weight=_ewc_weight, scope=_ewc_scope, n_batches=_ewc_nbatch)
        if _ewc_cache:
            Path(_ewc_cache).parent.mkdir(parents=True, exist_ok=True)
            ewc.save(_ewc_cache)
        model.__class__ = build_model(self.cfg, in_channels, self._active_heads).__class__
        ewc._star = {k: v.cpu() for k, v in ewc._star.items()}
        return ewc

    def _setup_replay(self):
        """Experience Replay (Chaudhry et al. 2019): build a cyclic loader over a
        small per-class memory buffer of the task-A dataset. Returns (loader, weight)
        or (None, 1.0) when disabled."""
        cfg = self.cfg
        rcfg = getattr(cfg, "replay", None)
        if rcfg is None or not getattr(rcfg, "enabled", False):
            return None, 1.0
        from torch_geometric.loader import DataLoader
        from torch_geometric.data import Batch
        pretrained_lm = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
        func_lm = getattr(cfg.model, "func_lm", "") or pretrained_lm
        kwargs = dict(
            root=str(cfg.data.processed_dir.parent), max_nodes=cfg.data.max_nodes,
            embedder_device=str(self.device), mode=cfg.data.mode,
            pretrained_lm=pretrained_lm, func_lm=func_lm,
            add_func_tokens=getattr(cfg.model, "add_func_tokens", False),
            func_lm_source=getattr(cfg.model, "func_lm_source", "raw"),
            cwe_list=getattr(cfg.data, "cwe_list", None),
            cwe_groups=getattr(cfg.data, "cwe_groups", None),
            filter_owasp=getattr(cfg.data, "filter_owasp", False),
            func_max_length=getattr(cfg.model, "func_max_length", 512),
            storage=getattr(cfg.data, "storage", "inmemory"),
            precompute_line_cls=getattr(cfg.model, "precompute_line_cls", False),
            ds_name_suffix=getattr(rcfg, "ds_name_suffix", ""),
        )
        # Dataset-identity params: prefer replay-config overrides (task-A subset),
        # else inherit cfg.data. Lets the replay buffer load task-A's megavul .pt even
        # when task-B's data block differs (CIL: megavul_cil filter-off vs task-A filter_top25).
        def _rd(k, default):
            v = getattr(rcfg, k, None)
            return v if v is not None else getattr(cfg.data, k, default)
        kwargs["top_cwe"]                = _rd("top_cwe", 0)
        kwargs["filter_top25_dangerous"] = _rd("filter_top25_dangerous", False)
        kwargs["max_per_class"]          = _rd("max_per_class", 0)
        kwargs["resample_seed"]          = _rd("resample_seed", 42)
        ds = CodeBERTGraphDataset(source=getattr(rcfg, "source", ""), **kwargs)
        train_idx, _, _ = ds.get_splits(
            train_ratio=getattr(cfg.data, "train_ratio", 0.8),
            val_ratio=getattr(cfg.data, "val_ratio", 0.1),
            seed=getattr(cfg.data, "split_seed", None) or cfg.train.seed,
        )
        bpc = int(getattr(rcfg, "buffer_per_class", 0))
        if bpc > 0:
            import collections
            import random as _random
            rng = _random.Random(int(getattr(rcfg, "buffer_seed", 42)))
            by_class: dict[int, list[int]] = collections.defaultdict(list)
            for i in train_idx:
                by_class[int(ds[i].y)].append(i)
            buf: list[int] = []
            for _c, idxs in by_class.items():
                rng.shuffle(idxs)
                buf.extend(idxs[:bpc])
        else:
            buf = list(train_idx)

        _needs_func_tokens = getattr(cfg.model, "live_lm", "func") != "none"
        _KEYS = ("func_input_ids", "func_attention_mask", "func_token_lines")
        def _collate(batch):
            if not _needs_func_tokens:
                for g in batch:
                    for k in _KEYS:
                        if hasattr(g, k):
                            delattr(g, k)
            return Batch.from_data_list(batch)

        loader = DataLoader(ds[buf], batch_size=cfg.train.batch_size, shuffle=True,
                            collate_fn=_collate, num_workers=0)
        logger.info(
            f"Experience replay: buffer={len(buf)} from task-A "
            f"'{getattr(rcfg, 'source', '')}{getattr(rcfg, 'ds_name_suffix', '')}' "
            f"(per_class={bpc or 'all'}), weight={getattr(rcfg, 'weight', 1.0)}"
        )
        return loader, float(getattr(rcfg, "weight", 1.0))

    def _setup_amp(self):
        cfg = self.cfg
        use_amp = self.device.type == "cuda" and getattr(cfg.train, "use_amp", True)
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        scaler = GradScaler() if use_amp and amp_dtype == torch.float16 else None
        return use_amp, amp_dtype, scaler

    def _setup_run_dir(self):
        cfg = self.cfg
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.model.architecture}_{cfg.data.mode}"
        if self.resume:
            existing = sorted(cfg.train.checkpoint_dir.glob(
                f"*_{cfg.model.architecture}_{cfg.data.mode}/last_{cfg.model.architecture}.pt"
            ))
            if existing:
                run_dir = existing[-1].parent
                run_id  = run_dir.name
                logger.info(f"Resuming run: {run_id}")
                return run_id, run_dir
            logger.warning("--resume: no previous run found — starting fresh.")
        run_dir = cfg.train.checkpoint_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def _training_loop(
        self, trainer, train_loader, val_loader, test_loader,
        cm: CheckpointManager, class_weight, train_counts,
        stop_on_f1, best_val_f1, best_val_loss, patience_counter, start_epoch,
        step_per_batch, optimizer, scheduler, unfreezer=None,
    ) -> None:
        cfg = self.cfg
        save_last_every = getattr(cfg.train, "save_last_every", 1)
        train_start = time.time()
        epoch_log: list[dict] = []

        for epoch in range(start_epoch, cfg.train.epochs + 1):
            if unfreezer is not None:
                msg = unfreezer.step(epoch)
                if msg:
                    logger.info(msg)
            if self._use_epoch_adaptive and train_counts is not None:
                class_weight = epoch_adaptive_class_weights(
                    train_counts, epoch, cfg.train.epochs, cfg.model.num_classes, self.device
                )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            train_loss = trainer.train_epoch(train_loader, epoch, cfg.train.epochs, class_weight)
            val_m = trainer.evaluate(val_loader, self._is_binary, class_weight)
            val_loss, val_acc, val_conf = val_m["loss"], val_m["acc"], val_m["conf"]
            val_f1, val_f1w = val_m["f1_macro"], val_m["f1_weighted"]
            val_prec, val_rec = val_m["precision_macro"], val_m["recall_macro"]
            if not step_per_batch:
                # CosineAnnealingLR takes no argument; ReduceLROnPlateau takes val_loss
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                else:
                    scheduler.step(val_loss)

            epoch_time = time.time() - t0
            improved = (val_f1 > best_val_f1) if stop_on_f1 else (val_loss < best_val_loss)
            lr_now = optimizer.param_groups[-1]['lr']
            logger.info(
                f"Epoch {epoch:03d}/{cfg.train.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"acc={val_acc:.4f} | f1={val_f1:.4f} | f1w={val_f1w:.4f} | "
                f"prec={val_prec:.4f} | rec={val_rec:.4f} | "
                f"lr={lr_now:.2e} | "
                f"patience={patience_counter}/{cfg.train.patience} | "
                f"{epoch_time:.0f}s" + (" *" if improved else "")
            )
            epoch_peak_vram = 0.0
            if torch.cuda.is_available():
                epoch_peak_vram = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            epoch_entry = {
                "epoch": epoch, "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 6),
                "val_f1": round(val_f1, 6), "val_f1w": round(val_f1w, 6),
                "val_prec": round(val_prec, 6), "val_rec": round(val_rec, 6),
                "lr": lr_now, "epoch_time_s": round(epoch_time, 1),
                "peak_vram_gb": epoch_peak_vram, "best": improved,
            }
            # MTL diagnostics — flatten trainer.last_diag into epoch row
            if getattr(trainer, "last_diag", None):
                for k, v in trainer.last_diag.items():
                    epoch_entry[f"diag_{k.replace('/', '_')}"] = round(v, 6)
            epoch_log.append(epoch_entry)
            if improved:
                best_val_f1 = val_f1; best_val_loss = val_loss; patience_counter = 0
                cm.save_best(trainer.model, epoch=epoch, val_loss=val_loss,
                             val_acc=val_acc, val_conf=val_conf, val_f1=val_f1, val_f1_weighted=val_f1w)
            else:
                patience_counter += 1

            if save_last_every > 0 and epoch % save_last_every == 0:
                cm.save_last(trainer.model, optimizer, scheduler, epoch=epoch,
                             best_val_f1=best_val_f1, best_val_loss=best_val_loss,
                             patience_counter=patience_counter, val_loss=val_loss,
                             val_acc=val_acc, val_conf=val_conf, val_f1=val_f1, val_f1_weighted=val_f1w)

            if patience_counter >= cfg.train.patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        test_m = trainer.evaluate(test_loader, self._is_binary, class_weight)
        test_acc, test_conf = test_m["acc"], test_m["conf"]
        test_f1, test_f1w = test_m["f1_macro"], test_m["f1_weighted"]
        test_prec, test_rec = test_m["precision_macro"], test_m["recall_macro"]
        if len(getattr(test_loader, "dataset", []) or []) == 0:
            logger.info("Empty test split (train_ratio + val_ratio = 1.0) — test metrics skipped.")
        else:
            logger.info(
                f"Test  acc={test_acc:.4f} | f1={test_f1:.4f} | f1w={test_f1w:.4f} | "
                f"prec={test_prec:.4f} | rec={test_rec:.4f} | conf={test_conf:.4f}"
            )
        total_time = int(time.time() - train_start)
        h, rem = divmod(total_time, 3600)
        m, s = divmod(rem, 60)
        logger.info(f"Total time: {h:02d}h {m:02d}m {s:02d}s")
        logger.info(f"Best checkpoint → {cm.best_path}")
        logger.info(f"To evaluate: uv run evaluate --checkpoint {cm.best_path}")

        # Save training artifacts to results_dir (separate from model weights)
        import csv as _csv, json as _json
        res_dir = cfg.train.results_dir / cm.run_dir.name
        res_dir.mkdir(parents=True, exist_ok=True)
        # Under the API (GNN_VULN_API_MODE=1) skip research-only artifacts — the per-epoch
        # CSV log and the rendered curves plot are never consumed by the service and only
        # waste compute + disk. split.json + training_summary.json (handoff) are still written.
        _api_mode = os.environ.get("GNN_VULN_API_MODE") == "1"

        if epoch_log and not _api_mode:
            log_path = res_dir / "training_log.csv"
            with open(log_path, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=epoch_log[0].keys())
                writer.writeheader(); writer.writerows(epoch_log)
            logger.info(f"training_log.csv → {log_path}")

        epoch_times = [r["epoch_time_s"] for r in epoch_log]
        num_params = sum(p.numel() for p in trainer.model.parameters())
        peak_vram_gb = peak_reserved_gb = 0.0
        gpu_name = "cpu"
        if torch.cuda.is_available():
            peak_vram_gb     = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            peak_reserved_gb = round(torch.cuda.max_memory_reserved()  / 1024**3, 3)
            gpu_name         = torch.cuda.get_device_name(0)
        # Aggregate MTL diagnostics across all epochs (mean of per-epoch means)
        mtl_summary = {}
        diag_keys = [k for k in (epoch_log[0].keys() if epoch_log else []) if k.startswith("diag_")]
        if diag_keys:
            for k in diag_keys:
                vals = [r.get(k, 0.0) for r in epoch_log if k in r]
                if vals:
                    mtl_summary[f"{k}_mean"]   = round(sum(vals) / len(vals), 6)
                    mtl_summary[f"{k}_max"]    = round(max(vals), 6)
                    mtl_summary[f"{k}_min"]    = round(min(vals), 6)
                    # Verdict on whether MTL methods worth trying:
                    # conflict_pct keys → if > 0.20 average, PCGrad likely helps
                    if k.startswith("diag_conflict_"):
                        mtl_summary[f"{k}_verdict"] = (
                            "severe" if mtl_summary[f"{k}_mean"] > 0.30 else
                            "moderate" if mtl_summary[f"{k}_mean"] > 0.15 else
                            "mild" if mtl_summary[f"{k}_mean"] > 0.05 else
                            "none"
                        )

        summary_path = res_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            _json.dump({
                "run_id":              cm.run_dir.name,
                "architecture":        cfg.model.architecture,
                "dataset_pt":          getattr(self, "_dataset_pt", ""),
                "num_classes":         cfg.model.num_classes,
                "num_params":          num_params,
                "epochs_trained":      len(epoch_log),
                "best_val_f1":         round(best_val_f1, 6),
                "best_val_loss":       round(best_val_loss, 6),
                "test_acc":            round(test_acc, 6),
                "test_f1":             round(test_f1, 6),
                "test_f1w":            round(test_f1w, 6),
                "test_prec":           round(test_prec, 6),
                "test_rec":            round(test_rec, 6),
                "test_prec_weighted":  round(test_m["precision_weighted"], 6),
                "test_rec_weighted":   round(test_m["recall_weighted"],    6),
                "test_per_class":      test_m["per_class"],
                "test_conf":           round(test_conf, 6),
                "total_time_s":        total_time,
                "avg_epoch_time_s":    round(sum(epoch_times) / len(epoch_times), 1) if epoch_times else 0,
                "min_epoch_time_s":    round(min(epoch_times), 1) if epoch_times else 0,
                "max_epoch_time_s":    round(max(epoch_times), 1) if epoch_times else 0,
                "gpu":                 gpu_name,
                "peak_vram_gb":        peak_vram_gb,
                "peak_reserved_gb":    peak_reserved_gb,
                **({"mtl_diagnostics": mtl_summary} if mtl_summary else {}),
            }, f, indent=2)
        logger.info(f"training_summary.json → {summary_path}")

        # split.json — map dataset indices to parquet_ids (seeded-split runs only)
        _sd = getattr(self, "_split_dataset", None)
        _train_idx = getattr(self, "_split_train_idx", None)
        _val_idx   = getattr(self, "_split_val_idx", None)
        _test_idx  = getattr(self, "_split_test_idx", None)
        if _sd is not None and _val_idx is not None and _test_idx is not None:
            _pids = _sd.get_all_parquet_ids().tolist()
            _split = {
                "seed": cfg.train.seed,
                "train_ratio": getattr(cfg.data, "train_ratio", 0.8),
                "val_ratio": getattr(cfg.data, "val_ratio", 0.1),
                "train": [_pids[i] for i in _train_idx],
                "val":   [_pids[i] for i in _val_idx],
                "test":  [_pids[i] for i in _test_idx],
            }
            (res_dir / "split.json").write_text(_json.dumps(_split), encoding="utf-8")
            logger.info(f"split.json → {res_dir / 'split.json'}")

        # Training curves plot (research-only — skipped under the API)
        if epoch_log and not _api_mode:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                epochs    = [r["epoch"]      for r in epoch_log]
                tr_loss   = [r["train_loss"] for r in epoch_log]
                val_loss_ = [r["val_loss"]   for r in epoch_log]
                val_f1_   = [r["val_f1"]     for r in epoch_log]
                val_f1w_  = [r["val_f1w"]    for r in epoch_log]
                vram_     = [r.get("peak_vram_gb", 0.0) for r in epoch_log]
                best_ep   = next((r["epoch"] for r in epoch_log if r["best"]), None)
                has_vram  = any(v > 0 for v in vram_)

                nrows = 3 if has_vram else 2
                fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)
                ax1, ax2 = axes[0], axes[1]

                ax1.plot(epochs, tr_loss,   label="train loss",  color="steelblue")
                ax1.plot(epochs, val_loss_, label="val loss",    color="tomato")
                if best_ep:
                    ax1.axvline(best_ep, color="gray", linestyle="--", alpha=0.6, label=f"best (ep {best_ep})")
                ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
                ax1.set_title("Training Curves")

                ax2.plot(epochs, val_f1_,  label="val F1 macro",    color="seagreen")
                ax2.plot(epochs, val_f1w_, label="val F1 weighted",  color="darkorange", linestyle="--")
                if best_ep:
                    ax2.axvline(best_ep, color="gray", linestyle="--", alpha=0.6)
                ax2.set_ylabel("F1"); ax2.legend(); ax2.grid(True, alpha=0.3)
                if not has_vram:
                    ax2.set_xlabel("Epoch")

                if has_vram:
                    ax3 = axes[2]
                    ax3.plot(epochs, vram_, label="peak VRAM (GB)", color="mediumpurple")
                    ax3.set_ylabel("VRAM (GB)"); ax3.set_xlabel("Epoch")
                    ax3.legend(); ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                curves_path = res_dir / "training_curves.png"
                fig.savefig(curves_path, dpi=150)
                plt.close(fig)
                logger.info(f"training_curves.png → {curves_path}")
            except Exception as e:
                logger.warning(f"Could not save training_curves.png: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Python 3.14 switched DataLoader workers to 'forkserver' (pickles worker args,
    # breaks local worker_init_fn/collate_fn). Force 'fork' to match prior behavior.
    import multiprocessing as _mp
    try:
        _mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Train GNN vulnerability detector")
    parser.add_argument("--config", type=str, nargs="+", default=["configs/lmgcn/binary.yaml"],
                        help="One config file (classic monolithic), or several split "
                             "files (e.g. data.yaml model.yaml train.yaml) merged in order.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest last_*.pt for this arch/mode.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override cfg.train.seed for multi-seed variance runs.")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Override the train/val/test split seed (keeps the split fixed while train.seed varies).")
    args = parser.parse_args()

    session = TrainingSession.from_args(args)
    session.run(config_path=args.config)



if __name__ == "__main__":
    main()
