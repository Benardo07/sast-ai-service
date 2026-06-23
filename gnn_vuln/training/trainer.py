"""Trainer class: forward pass dispatch, train loop, evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class _CUDAPrefetcher:
    """Overlap CPU collation with GPU compute using a background CUDA stream."""

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self._use_cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream() if self._use_cuda else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._next = None
        self._preload()
        return self

    def _preload(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next = None
            return
        if self._use_cuda:
            with torch.cuda.stream(self.stream):
                batch = batch.to(self.device, non_blocking=True)
        self._next = batch

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self._use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next
        self._preload()
        return batch

from gnn_vuln.training.losses import (
    focal_loss,
    livable_loss,
    mil_loss,
    mil_loss_multiclass,
    ranking_loss,
)

# Import lazily to avoid circular imports; resolved at runtime
_EWCDR_TYPE = None


def _ewcdr_type():
    global _EWCDR_TYPE
    if _EWCDR_TYPE is None:
        from gnn_vuln.training.ewc import EWCDR
        _EWCDR_TYPE = EWCDR
    return _EWCDR_TYPE


class Trainer:
    """
    Encapsulates the training and evaluation loop for all architectures.

    Handles:
      - Unified forward dispatch (2-tuple / 3-tuple / 5-tuple returns)
      - MTL auxiliary losses (group, binary, SupCon)
      - MIL and ranking localisation losses
      - AMP (automatic mixed precision)
      - Gradient clipping
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        step_per_batch: bool,
        device: torch.device,
        *,
        mil_k: int = 3,
        mil_weight: float = 0.0,
        rank_loss_weight: float = 0.0,
        focal_gamma: float = 0.0,
        group_loss_weight: float = 0.0,
        binary_loss_weight: float = 0.0,
        supcon_fn: nn.Module | None = None,
        supcon_weight: float = 0.0,
        self_supcon_weight: float = 0.0,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        scaler: GradScaler | None = None,
        ewc=None,   # EWCDR | None
        replay_loader=None,        # cyclic DataLoader over task-A memory buffer | None
        replay_weight: float = 1.0,
        grad_accum_steps: int = 1,
        label_smoothing: float = 0.0,
        use_livable_real: bool = False,
        livable_focal_gamma: float = 2.0,
        livable_label_smoothing: float = 0.1,
        pgd=None,             # EmbeddingPGD | None
        pgd_tokenizer=None,   # HF tokenizer for func-LM
        loss_balance_method: str = "fixed",   # "fixed" | "kendall" | "pcgrad"
        uncertainty_weights: nn.Module | None = None,
        mtl_diagnose: bool = False,            # if True, log per-task gradient/loss diagnostics
        mtl_diagnose_every: int = 10,          # sample 1/N batches
    ):
        self.model              = model
        self.optimizer          = optimizer
        self.scheduler          = scheduler
        self.step_per_batch     = step_per_batch
        self.device             = device
        self.loss_balance_method = loss_balance_method
        self.uncertainty_weights = uncertainty_weights
        self._raw_losses: dict[str, torch.Tensor] = {}
        # MTL diagnostic state — populated by train_epoch when mtl_diagnose=True
        self.mtl_diagnose = mtl_diagnose
        self.mtl_diagnose_every = max(1, mtl_diagnose_every)
        self._diag_buffer: list[dict[str, float]] = []   # samples this epoch
        self.last_diag: dict[str, float] = {}            # epoch-mean diagnostics
        self.mil_k              = mil_k
        self.mil_weight         = mil_weight
        self.rank_loss_weight   = rank_loss_weight
        self.focal_gamma        = focal_gamma
        self.group_loss_weight  = group_loss_weight
        self.binary_loss_weight = binary_loss_weight
        self.supcon_fn          = supcon_fn
        self.supcon_weight      = supcon_weight
        self.self_supcon_weight = self_supcon_weight
        self.use_amp            = use_amp
        self.amp_dtype          = amp_dtype
        self.scaler             = scaler
        self.ewc                = ewc
        self.replay_loader      = replay_loader
        self.replay_weight      = replay_weight
        self._replay_iter       = iter(replay_loader) if replay_loader is not None else None
        self.grad_accum_steps   = max(1, grad_accum_steps)
        self.label_smoothing    = label_smoothing
        self.use_livable_real   = use_livable_real
        self.livable_focal_gamma = livable_focal_gamma
        self.livable_label_smoothing = livable_label_smoothing
        self.pgd                = pgd
        self.pgd_tokenizer      = pgd_tokenizer
        self._current_epoch     = 1
        self._total_epochs      = 100
        # cRT: when set, only this module trains; the rest of the model is kept
        # in eval() during train_epoch so backbone BatchNorm running stats and
        # dropout stay fixed (decoupled stage-2 keeps representations frozen).
        self._crt_train_module: nn.Module | None = None
        # Balanced-Mixup / Remix: when class_counts is set, the classification loss
        # becomes a two-target mix using the model's per-batch mixup perm + lam.
        self.mixup_remix: bool = True
        self.mixup_remix_kappa: float = 3.0
        self.mixup_remix_tau: float = 0.5
        self.class_counts: torch.Tensor | None = None
        # Logit Adjustment (Menon et al. 2021): when set, classification CE is computed
        # on logit + tau*log_prior (raw logit kept for inference/metrics).
        self._la_log_prior: torch.Tensor | None = None
        self._la_tau: float = 1.0
        # FLAG (Kong et al. 2020): adversarial node-feature perturbation training.
        self._flag_enabled: bool = False
        self._flag_step_size: float = 1e-3
        self._flag_steps: int = 3

    def set_crt_mode(self, train_module: nn.Module) -> None:
        """Enable cRT: keep backbone in eval, train only ``train_module`` (func_head)."""
        self._crt_train_module = train_module

    def set_logit_adjustment(self, log_prior: torch.Tensor, tau: float) -> None:
        """Enable Logit Adjustment loss. ``log_prior`` = log(class priors) [C] on device."""
        self._la_log_prior = log_prior
        self._la_tau = tau

    def set_flag(self, step_size: float, steps: int) -> None:
        """Enable FLAG adversarial node-feature training (Kong et al. 2020)."""
        self._flag_enabled = True
        self._flag_step_size = step_size
        self._flag_steps = max(1, steps)

    def _flag_step(self, batch, class_weight) -> torch.Tensor:
        """One FLAG training step (Kong et al. 2020, reference flag()): perturb node
        features, M unbounded ascent steps (delta += step_size*sign(grad), no clamp),
        param grads accumulated over the M perturbed forwards (loss/=M), one optimizer
        step. Returns the (un-normalized) last-step loss."""
        self.optimizer.zero_grad()
        x = batch.x
        m = self._flag_steps
        ss = self._flag_step_size
        delta = torch.empty_like(x).uniform_(-ss, ss)
        delta.requires_grad_(True)
        last = None
        for _ in range(m):
            loss_t = self._forward(batch, class_weight, x_override=x + delta)[1] / m
            loss_t.backward()                       # accumulates param grad (1/m) + delta.grad
            last = loss_t
            with torch.no_grad():
                delta.data.add_(ss * delta.grad.sign())   # unbounded ascent, no clamp (paper)
            delta.grad = None
        if getattr(self, "_grad_clip", 0.0) > 0.0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return (last * m).detach()

    def set_mixup(self, remix: bool, kappa: float, tau: float, counts: torch.Tensor) -> None:
        """Enable Balanced-Mixup loss. ``counts`` = per-class train sample counts
        (on the compute device) used for the Remix imbalance-aware label mixing."""
        self.mixup_remix = remix
        self.mixup_remix_kappa = kappa
        self.mixup_remix_tau = tau
        self.class_counts = counts

    def _remix_lambda_y(self, y_i: torch.Tensor, y_j: torch.Tensor, lam: float) -> torch.Tensor:
        """Per-sample label-mix weight on y_i (Chou et al. 2020, Remix). Vanilla mixup
        uses lam for every sample; Remix reassigns the full label to the MINORITY class
        of each pair when one class is >= kappa times rarer and the feature ratio is
        extreme enough (guarded by tau)."""
        lam_y = torch.full_like(y_i, float(lam), dtype=torch.float)
        if self.class_counts is None or not self.mixup_remix:
            return lam_y
        n_i = self.class_counts[y_i].float()
        n_j = self.class_counts[y_j].float()
        kappa, tau = self.mixup_remix_kappa, self.mixup_remix_tau
        # i is the majority (n_i >> n_j) and the feature mix isn't i-dominated → label to minority j
        to_j = (n_i / n_j >= kappa) & (lam < tau)
        # j is the majority → label to minority i
        to_i = (n_j / n_i >= kappa) & ((1.0 - lam) < tau)
        lam_y = torch.where(to_j, torch.zeros_like(lam_y), lam_y)
        lam_y = torch.where(to_i, torch.ones_like(lam_y), lam_y)
        return lam_y

    # ── Forward ──────────────────────────────────────────────────────────────

    def _forward(
        self,
        batch,
        class_weight: torch.Tensor | None = None,
        x_override: torch.Tensor | None = None,
        add_ewc: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass → (logit_func, total_loss).

        x_override replaces batch.x in the model call (FLAG adversarial node features).

        Handles all return-tuple lengths:
          2-tuple (logit, stmt_scores)                                 — standard
          3-tuple (logit, stmt_scores, z)                              — SupCon
          5-tuple (logit_cwe, logit_group, logit_binary, stmt, z)      — MTL+SupCon
        """
        _x = batch.x if x_override is None else x_override
        node_line  = getattr(batch, "node_line",  None)
        edge_attr  = getattr(batch, "edge_attr",  None)
        # Only pass rwse if model's GNN encoder actually uses PE — avoid PyG batch
        # confusion for configs that don't enable PE but share dataset with PE configs.
        rwse = None
        _enc = getattr(self.model, "encoder", None)
        if _enc is not None and getattr(_enc, "use_pe", False):
            rwse = getattr(batch, "rwse", None)

        if hasattr(self.model, "codebert"):
            func_ids        = getattr(batch, "func_input_ids",      None)
            func_mask       = getattr(batch, "func_attention_mask", None)
            func_tlines     = getattr(batch, "func_token_lines",    None)
            func_line_cls   = getattr(batch, "func_line_cls",       None)
            func_line_ids   = getattr(batch, "func_line_ids",       None)
            func_line_cls_b = getattr(batch, "func_line_cls_batch", None)
            out = self.model(
                _x, batch.edge_index, batch.batch,
                node_line, edge_attr, func_ids, func_mask, func_tlines,
                func_line_cls=func_line_cls,
                func_line_ids=func_line_ids,
                func_line_cls_batch=func_line_cls_b,
                rwse=rwse,
            )
        elif hasattr(batch, "subgraphs_nodes_mapper"):
            # graph_vit offline: pass the batch so it reads precomputed patch fields
            out = self.model(_x, batch.edge_index, batch.batch, node_line, edge_attr, rwse=rwse, data=batch)
        else:
            out = self.model(_x, batch.edge_index, batch.batch, node_line, edge_attr, rwse=rwse)

        # Unpack return tuple
        if len(out) == 5:
            logit_func, logit_group, logit_binary, stmt_scores, z_combined = out
        elif len(out) == 4:
            logit_func, logit_group, logit_binary, stmt_scores = out
            z_combined = None
        elif len(out) == 3:
            logit_func, stmt_scores, z_combined = out
            logit_group = logit_binary = None
        else:
            logit_func, stmt_scores = out
            logit_group = logit_binary = z_combined = None

        # Compute raw (unweighted) per-task losses → stored in self._raw_losses for
        # MTL balance methods (kendall/pcgrad). Combined into total `loss` below.
        raw: dict[str, torch.Tensor] = {}
        use_balance = self.loss_balance_method in ("kendall", "pcgrad")

        # Logit Adjustment: classification loss is computed on the offset logit
        # (logit + tau*log_prior); the raw logit_func is kept for metrics/inference.
        _logit_cls = logit_func
        if self._la_log_prior is not None:
            _logit_cls = logit_func + self._la_tau * self._la_log_prior

        # Primary classification loss (CE / focal / livable)
        _mixup_perm = getattr(self.model, "_mixup_perm", None)
        if _mixup_perm is not None and self.model.training:
            # Balanced-Mixup: logit comes from mixed features → two-target CE with
            # Remix imbalance-aware per-sample label weight. Uses plain CE branches
            # (focal/livable not combined with mixup).
            lam = self.model._mixup_lam
            y_i = batch.y
            y_j = batch.y[_mixup_perm]
            lam_y = self._remix_lambda_y(y_i, y_j, lam)
            ce_i = F.cross_entropy(_logit_cls, y_i, weight=class_weight,
                                   label_smoothing=self.label_smoothing, reduction="none")
            ce_j = F.cross_entropy(_logit_cls, y_j, weight=class_weight,
                                   label_smoothing=self.label_smoothing, reduction="none")
            cls_loss = (lam_y * ce_i + (1.0 - lam_y) * ce_j).mean()
        elif self.use_livable_real:
            cls_loss = livable_loss(
                _logit_cls, batch.y,
                epoch=self._current_epoch,
                total_epochs=self._total_epochs,
                focal_gamma=self.livable_focal_gamma,
                label_smoothing=self.livable_label_smoothing,
                weight=class_weight,
            )
        elif self.focal_gamma > 0.0:
            cls_loss = focal_loss(_logit_cls, batch.y, gamma=self.focal_gamma,
                                  weight=class_weight, label_smoothing=self.label_smoothing)
        else:
            cls_loss = F.cross_entropy(_logit_cls, batch.y, weight=class_weight,
                                       label_smoothing=self.label_smoothing)
        raw["cls"] = cls_loss
        loss = cls_loss

        # MTL auxiliary losses
        if logit_group is not None and self.group_loss_weight > 0.0:
            group_labels = getattr(batch, "group_id", None)
            if group_labels is not None:
                g_loss = F.cross_entropy(logit_group, group_labels)
                raw["group"] = g_loss
                if not use_balance:
                    loss = loss + self.group_loss_weight * g_loss

        if logit_binary is not None and self.binary_loss_weight > 0.0:
            binary_labels = (batch.y > 0).long()
            b_loss = F.cross_entropy(logit_binary, binary_labels)
            raw["binary"] = b_loss
            if not use_balance:
                loss = loss + self.binary_loss_weight * b_loss

        # MIL localisation loss
        if stmt_scores is not None and self.mil_weight > 0.0:
            is_mc_stmt = len(stmt_scores) > 0 and stmt_scores[0].dim() == 2
            if is_mc_stmt:
                m_loss = mil_loss_multiclass(stmt_scores, batch.y, self.mil_k)
            else:
                m_loss = mil_loss(stmt_scores, batch.y, self.mil_k)
            raw["mil"] = m_loss
            if not use_balance:
                loss = loss + self.mil_weight * m_loss

        # Ranking loss (binary stmt heads only)
        if (
            stmt_scores is not None
            and self.rank_loss_weight > 0.0
            and node_line is not None
            and (len(stmt_scores) == 0 or stmt_scores[0].dim() == 1)
        ):
            flaw_mask = getattr(batch, "flaw_line_mask", None)
            if flaw_mask is not None:
                rl = ranking_loss(
                    stmt_scores, batch.batch, node_line, flaw_mask, batch.y
                )
                raw["rank"] = rl
                if not use_balance:
                    loss = loss + self.rank_loss_weight * rl

        # MTL balance: combine raw losses via learned uncertainty (kendall) or
        # leave for PCGrad to split per-task gradients (pcgrad uses raw dict directly).
        if self.loss_balance_method == "kendall" and self.uncertainty_weights is not None:
            loss = self.uncertainty_weights(raw)
        # pcgrad case: total `loss` = cls only (computed above before raw additions);
        # train_epoch handles per-task backward + projection separately using self._raw_losses.
        self._raw_losses = raw

        # MoE load-balance aux loss (Shazeer 2017 / GMoE). Added directly to total
        # loss for all balance methods — it regularizes routing, not a task loss.
        moe_aux = getattr(self.model, "moe_aux_loss", None)
        if moe_aux is not None and torch.is_tensor(moe_aux) and moe_aux.requires_grad:
            loss = loss + moe_aux

        # Hierarchical SupCon — matrix distance only (no group requirement).
        # group_ids falls back to batch.y: each class = its own group, so
        # same_group is never True for different-class pairs → matrix handles
        # all inter-CWE weights, intragroup_only=False in the loss config.
        if z_combined is not None and self.supcon_fn is not None and self.supcon_weight > 0.0:
            group_ids = getattr(batch, "group_id", batch.y)
            cwe_vocab_ids = getattr(batch, "cwe_id", None)
            sc = self.supcon_fn(z_combined, batch.y, group_ids, cwe_vocab_ids)
            loss = loss + self.supcon_weight * sc

        # Self-supervised NT-Xent loss (L_self from HierarchicalSupCon EMNLP 2024).
        # Runs projector a second time (dropout active in training) → different view z2.
        # Prevents intra-class collapse by anchoring each sample to its own stochastic views.
        if (
            self.self_supcon_weight > 0.0
            and self.supcon_fn is not None
            and z_combined is not None
            and self.model.training
            and hasattr(self.model, "supcon_head")
            and self.model.supcon_head is not None
            and hasattr(self.model, "_fused_for_supcon")
            and self.model._fused_for_supcon is not None
        ):
            z2 = self.model.supcon_head(self.model._fused_for_supcon)
            sl = self.supcon_fn.self_supervised_loss(z_combined, z2)
            loss = loss + self.self_supcon_weight * sl

        # EWC-DR continual learning regularization (skip on replay forwards to
        # avoid double-counting the penalty within one optimization step).
        if self.ewc is not None and add_ewc:
            loss = loss + self.ewc.penalty(self.model)

        return logit_func, loss

    def _next_replay_batch(self):
        """Next batch from the cyclic task-A memory buffer (experience replay)."""
        try:
            return next(self._replay_iter)
        except StopIteration:
            self._replay_iter = iter(self.replay_loader)
            return next(self._replay_iter)

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
        class_weight: torch.Tensor | None = None,
    ) -> float:
        self.model.train()
        # cRT: freeze the backbone in eval() (BN running stats + dropout fixed);
        # only func_head trains. Root model.training=False also disables the
        # forward-pass graph augmentation block.
        if self._crt_train_module is not None:
            self.model.eval()
            self._crt_train_module.train()
        self._current_epoch = epoch
        self._total_epochs  = total_epochs
        accum = self.grad_accum_steps
        self.optimizer.zero_grad()
        # Reset per-epoch diagnostic buffer
        if self.mtl_diagnose:
            self._diag_buffer = []

        # Accumulate loss on GPU — avoids per-batch .item() sync which stalls
        # training waiting for GPU. One sync at end of epoch + throttled tqdm.
        loss_sum = torch.zeros(1, device=self.device)
        n_graphs = 0
        n_steps  = len(loader)
        # Throttle tqdm refresh: update display ~100 times/epoch regardless of size
        refresh_every = max(1, n_steps // 100)

        pbar = tqdm(loader, desc=f"  Train {epoch:03d}/{total_epochs}", unit="batch", leave=False)

        for step, batch in enumerate(pbar):
            batch = batch.to(self.device, non_blocking=True)
            is_last = (step == n_steps - 1)
            should_step = ((step + 1) % accum == 0) or is_last

            # FLAG: self-contained M-step adversarial step (own backward + optimizer
            # step). Bypasses the normal forward/accum/backward path below.
            if self._flag_enabled:
                flag_loss = self._flag_step(batch, class_weight)
                if self.step_per_batch:
                    self.scheduler.step()
                loss_sum = loss_sum + flag_loss * batch.num_graphs
                n_graphs += batch.num_graphs
                if (step % refresh_every == 0) or is_last:
                    pbar.set_postfix(loss=f"{(loss_sum / n_graphs).item():.4f}")
                continue

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self._forward(batch, class_weight)

            # Experience Replay (Chaudhry et al. 2019): mix a task-A memory batch
            # into this step's loss. add_ewc=False so the EWC penalty is counted once.
            if self.replay_loader is not None:
                rb = self._next_replay_batch().to(self.device, non_blocking=True)
                with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    _, r_loss = self._forward(rb, class_weight, add_ewc=False)
                loss = loss + self.replay_weight * r_loss

            # EDAT adversarial training: perturb identifier embeddings,
            # add adversarial loss. Runs adversarial forward in FP32 (no AMP)
            # so the retained clean graph and adv graph can be backprop-ed
            # together via loss_clean + loss_adv.
            if self.pgd is not None and self.pgd_tokenizer is not None:
                func_ids = getattr(batch, "func_input_ids", None)
                _cw = class_weight
                _batch = batch
                loss_adv = self.pgd.adv_loss(
                    loss_clean=loss,
                    func_input_ids=func_ids,
                    tokenizer=self.pgd_tokenizer,
                    forward_fn=lambda: self._forward(_batch, _cw)[1],
                )
                loss = loss + loss_adv

            # MTL diagnostic — sample 1/N batches, compute per-task gradient
            # cosine/conflict/norm. NO surgery, NO change to optimization.
            # Logged each epoch into trainer.last_diag (mean across samples).
            # Runs before loss /= accum so raw losses are unscaled.
            if (
                self.mtl_diagnose
                and len(self._raw_losses) >= 2
                and (step % self.mtl_diagnose_every == 0)
            ):
                from gnn_vuln.training.mtl_balance import diagnose_mtl
                _shared = [p for p in self.model.parameters() if p.requires_grad]
                # Need retain_graph=True because main backward happens below
                self._diag_buffer.append(
                    diagnose_mtl(self._raw_losses, _shared, retain_graph=True)
                )

            # Scale loss so gradient magnitude is independent of accum_steps
            loss = loss / accum

            if self.loss_balance_method in ("pcgrad", "pcgrad_encoder") and len(self._raw_losses) >= 2:
                # PCGrad: project conflicting per-task gradients on shared params.
                # AMP-compatible: scale each task loss via scaler.scale() before
                # autograd.grad(). Projection is scale-invariant — dot/norm_sq ratio
                # is unchanged when all gradients are multiplied by the same S —
                # so projected grads are correctly scaled. scaler.unscale_() + step()
                # handle the rest identically to the normal AMP path.
                #
                # "pcgrad"         — project ALL shared params (original; heads get
                #                    cross-task projection noise — no conflict there).
                # "pcgrad_encoder" — project ONLY encoder params (N28b fix). Heads get
                #                    standard per-task grads (each head serves 1 task,
                #                    no conflict). Only the shared encoder is de-conflicted.
                from gnn_vuln.training.mtl_balance import pcgrad_project
                use_scaler = self.use_amp and self.scaler is not None
                losses_for_pcgrad = {
                    name: (self.scaler.scale(l / accum) if use_scaler else l / accum)
                    for name, l in self._raw_losses.items()
                }
                if self.loss_balance_method == "pcgrad_encoder" and hasattr(self.model, "encoder"):
                    # 1. Per-task grads on ENCODER only (retain graph for backward below).
                    enc_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
                    projected = pcgrad_project(losses_for_pcgrad, enc_params, retain_graph=True)
                    # 2. Standard backward of summed loss → correct grads for heads + all.
                    summed = sum(losses_for_pcgrad.values())
                    summed.backward()
                    # 3. Overwrite encoder grads with PCGrad-projected version.
                    for p, g in zip(enc_params, projected):
                        p.grad = g
                else:
                    # Original: project all shared params, assign directly.
                    shared_params = [p for p in self.model.parameters() if p.requires_grad]
                    projected = pcgrad_project(losses_for_pcgrad, shared_params)
                    for p, g in zip(shared_params, projected):
                        p.grad = (p.grad + g) if p.grad is not None else g
                if should_step:
                    if use_scaler:
                        if hasattr(self, "_grad_clip") and self._grad_clip > 0.0:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if hasattr(self, "_grad_clip") and self._grad_clip > 0.0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
            elif self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                if should_step:
                    if hasattr(self, "_grad_clip") and self._grad_clip > 0.0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if should_step:
                    if hasattr(self, "_grad_clip") and self._grad_clip > 0.0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if self.step_per_batch and should_step:
                self.scheduler.step()

            # Accumulate on GPU (no sync). Multiply by accum to undo the earlier
            # loss / accum scaling so total_loss reflects un-normalized loss sum.
            loss_sum = loss_sum + loss.detach() * (accum * batch.num_graphs)
            n_graphs += batch.num_graphs

            # Only sync for tqdm display every refresh_every steps
            if (step % refresh_every == 0) or is_last:
                pbar.set_postfix(loss=f"{(loss_sum / n_graphs).item():.4f}")

        # Aggregate MTL diagnostic samples → mean per metric for this epoch
        if self.mtl_diagnose and self._diag_buffer:
            keys = set()
            for d in self._diag_buffer:
                keys.update(d.keys())
            self.last_diag = {
                k: sum(d.get(k, 0.0) for d in self._diag_buffer) / max(1, len(self._diag_buffer))
                for k in keys
            }
        else:
            self.last_diag = {}

        # Single sync at epoch end
        return (loss_sum.item() / n_graphs) if n_graphs > 0 else 0.0

    def set_grad_clip(self, clip: float) -> None:
        self._grad_clip = clip

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def evaluate(
        self,
        loader: DataLoader,
        is_binary: bool = True,
        class_weight: torch.Tensor | None = None,
    ) -> dict:
        """Return metrics dict: loss, acc, conf, f1_macro, f1_weighted, precision_macro, recall_macro, precision_weighted, recall_weighted, per_class."""
        self.model.eval()
        loss_sum  = torch.zeros(1, device=self.device)
        conf_sum  = torch.zeros(1, device=self.device)
        preds_buf:  list[torch.Tensor] = []
        labels_buf: list[torch.Tensor] = []

        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            logits, loss = self._forward(batch, class_weight)
            probs = F.softmax(logits, dim=-1)
            preds_buf.append(logits.argmax(dim=-1))
            labels_buf.append(batch.y)
            loss_sum = loss_sum + loss.detach() * batch.num_graphs
            conf_sum = conf_sum + probs.max(dim=-1).values.sum()

        if not preds_buf:                 # empty loader (e.g. test split at ratio 0) — no crash
            z = 0.0
            return {"loss": z, "acc": z, "conf": z, "f1_macro": z, "f1_weighted": z,
                    "precision_macro": z, "recall_macro": z, "precision_weighted": z,
                    "recall_weighted": z, "per_class": {}}
        # Single CPU sync for all accumulated tensors
        all_preds  = torch.cat(preds_buf).cpu().tolist()
        all_labels = torch.cat(labels_buf).cpu().tolist()
        n          = len(all_labels)
        avg = "binary" if is_binary else "macro"
        f1_macro         = f1_score(all_labels, all_preds, average=avg,        zero_division=0)
        f1_weighted      = f1_score(all_labels, all_preds, average="weighted",  zero_division=0)
        precision_main   = precision_score(all_labels, all_preds, average=avg,        zero_division=0)
        recall_main      = recall_score(all_labels, all_preds, average=avg,           zero_division=0)
        precision_w      = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall_w         = recall_score(all_labels, all_preds, average="weighted",    zero_division=0)
        acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        per_class = {
            k: {
                "precision": round(v["precision"], 6),
                "recall":    round(v["recall"],    6),
                "f1":        round(v["f1-score"],  6),
                "support":   int(v["support"]),
            }
            for k, v in report.items()
            if k not in ("macro avg", "weighted avg", "accuracy")
        }
        return {
            "loss":               (loss_sum / n).item(),
            "acc":                acc,
            "conf":               (conf_sum / n).item(),
            "f1_macro":           float(f1_macro),
            "f1_weighted":        float(f1_weighted),
            "precision_macro":    float(precision_main),
            "recall_macro":       float(recall_main),
            "precision_weighted": float(precision_w),
            "recall_weighted":    float(recall_w),
            "per_class":          per_class,
        }

    # ── Localisation ──────────────────────────────────────────────────────────

    @torch.inference_mode()
    def localise(self, data, top_k: int = 5) -> list[tuple[int, float]]:
        """Return top-k (line, score) for a single graph."""
        self.model.eval()
        data  = data.to(self.device, non_blocking=True)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
        node_line = getattr(data, "node_line", None)

        if hasattr(self.model, "codebert"):
            fids  = getattr(data, "func_input_ids",      None)
            fmask = getattr(data, "func_attention_mask", None)
            fids  = fids.unsqueeze(0)  if fids  is not None else None
            fmask = fmask.unsqueeze(0) if fmask is not None else None
            out = self.model(data.x, data.edge_index, batch, node_line, None, fids, fmask)
        else:
            out = self.model(data.x, data.edge_index, batch, node_line)

        # Extract stmt_scores
        stmt_scores_list = out[1] if len(out) >= 2 else None
        if stmt_scores_list is None or len(stmt_scores_list[0]) == 0:
            return []

        scores_raw = stmt_scores_list[0]
        scores = (
            1.0 - torch.softmax(scores_raw, dim=-1)[:, 0]
            if scores_raw.dim() == 2
            else torch.sigmoid(scores_raw)
        )

        valid_lines = data.node_line[data.node_line >= 0].unique(sorted=True)
        k = min(top_k, len(valid_lines))
        top_scores, top_idx = scores.topk(k)
        return [
            (int(valid_lines[i].item()), float(top_scores[j].item()))
            for j, i in enumerate(top_idx)
        ]
