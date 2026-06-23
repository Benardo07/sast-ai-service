"""Abstract base class for all vulnerability detectors."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoConfig, AutoModel
import transformers.pytorch_utils as _tpu
import torch.utils.checkpoint as _cp

# Gradient checkpointing: old reentrant mode doesn't preserve autocast context →
# backward recompute runs in fp32 even when forward was bf16, doubling activation
# memory. Patch checkpoint to default use_reentrant=False so autocast is preserved.
_orig_checkpoint = _cp.checkpoint
def _checkpoint_non_reentrant(fn, *args, use_reentrant=False, **kwargs):
    return _orig_checkpoint(fn, *args, use_reentrant=use_reentrant, **kwargs)
_cp.checkpoint = _checkpoint_non_reentrant

# Jina-v2 custom code imports find_pruneable_heads_and_indices from
# transformers.pytorch_utils, which was removed in newer transformers.
if not hasattr(_tpu, "find_pruneable_heads_and_indices"):
    from typing import List, Set, Tuple

    def _find_pruneable_heads_and_indices(
        heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
    ) -> Tuple[Set[int], torch.LongTensor]:
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index: torch.LongTensor = torch.arange(len(mask))[mask].long()
        return heads, index

    _tpu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

from gnn_vuln.models._lm_utils import (
    lm_hidden_dim, lm_pool, lm_pool_windowed, lm_full_windowed, lm_per_line_embed,
    lm_per_line_raw, lm_full_codet5p, lm_full_codet5p_raw, _is_codet5p_embedding,
    WindowAttentionPool, CrossWindowAttn, WindowMixerPool, _is_decoder_only,
)


class VulnDetectorBase(nn.Module):
    """
    Shared logic for all vulnerability detectors:
      - Live LM branch setup (_build_lm_branch)
      - LM embedding helper (_lm_embed)
      - Optimizer param grouping (lm_parameters / has_live_lm)
      - Config-driven construction classmethod (from_config)

    Subclasses implement forward(). Statement scoring is handled by
    StmtHead / MulticlassStmtHead in heads.py.
    """

    # ── LM branch (optional) ─────────────────────────────────────────────────
    # Populated by _build_lm_branch(); absent on frozen-embedding models.
    codebert: nn.Module
    _lm_dim: int
    _is_enc_dec: bool
    _matryoshka_dim: int | None
    _func_chunk_size: int   # 0 = disabled (single forward pass)
    _func_chunk_stride: int

    # ── LM branch helpers ─────────────────────────────────────────────────────

    def _build_lm_branch(
        self,
        pretrained_lm: str,
        func_lm: str,
        matryoshka_dim: int | None = None,
        func_chunk_size: int = 0,
        func_chunk_stride: int = 0,
        use_flash_attention: bool = False,
        compile_lm: bool = False,
        use_grad_checkpoint: bool = True,
        lm_per_line: bool = False,
        codet5p_raw_encoder: bool = False,
        codet5p_normalize_per_token: bool = False,
        freeze_func_lm: bool = False,
        window_attn_pool: bool = False,
        window_attn_hidden: bool = False,
        window_center_weight: bool = False,
        cross_window_attn: bool = False,
        window_mixer: bool = False,
        window_mixer_max: int = 6,
    ) -> None:
        """
        Load a live LM and store as self.codebert.
        Call once from subclass __init__ when a live LM is needed.

        Parameters
        ----------
        func_chunk_size : int
            Sliding-window chunk size in tokens. 0 = disabled (single forward pass).
            Should match the model's trained max length (e.g. 512 for UniXcoder).
        func_chunk_stride : int
            Step between windows. 0 = defaults to chunk_size // 2 (50% overlap).
            Only used when func_chunk_size > 0.
        use_flash_attention : bool
            Load the LM with flash_attention_2 if available. Requires flash-attn package.
        freeze_func_lm : bool
            When True, freeze all func_lm weights (requires_grad=False). The LM still
            runs forward passes for feature extraction but weights are not updated.
            Disables gradient checkpointing (no grads flow through). No LM optimizer
            group is created (has_live_lm returns False, lm_parameters returns []).
        """
        _func_lm = func_lm if func_lm else pretrained_lm
        _cfg = AutoConfig.from_pretrained(_func_lm, trust_remote_code=True)
        if not hasattr(_cfg, "is_decoder"):
            _cfg.is_decoder = False
        if not hasattr(_cfg, "add_cross_attention"):
            _cfg.add_cross_attention = False
        # Activate SDPA inside jina-v2's custom attention (attn_implementation='torch').
        # Ignored by standard HF models (unknown config attr). Always set as baseline
        # so jina uses PyTorch's memory-efficient scaled_dot_product_attention by default.
        _cfg.attn_implementation = "torch"
        load_kwargs: dict = {"config": _cfg, "trust_remote_code": True}
        if torch.cuda.is_available():
            load_kwargs["dtype"] = torch.bfloat16
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info(f"flash_attention_2 enabled for {_func_lm} (flash-attn {flash_attn.__version__})")
            except ImportError:
                # HF-level sdpa for standard models (ModernBERT etc.). Jina uses config-level
                # attn_implementation='torch' set above instead of this HF dispatch path.
                logger.info(f"flash-attn not installed — trying sdpa for {_func_lm}")
                load_kwargs["attn_implementation"] = "sdpa"
        try:
            self.codebert = AutoModel.from_pretrained(_func_lm, **load_kwargs)
        except ValueError as _fa2_err:
            if "Flash Attention 2.0" in str(_fa2_err) and load_kwargs.get("attn_implementation") == "flash_attention_2":
                # Model doesn't support HF flash_attention_2 dispatch (e.g. jina-v2).
                # Fall back to config-level SDPA (attn_implementation='torch' set above).
                logger.warning(f"{_func_lm} does not support flash_attention_2 — using config-level SDPA")
                load_kwargs.pop("attn_implementation")
                self.codebert = AutoModel.from_pretrained(_func_lm, **load_kwargs)
            else:
                raise
        self._is_decoder_only_lm = _is_decoder_only(self.codebert)
        self._freeze_func_lm = freeze_func_lm
        if freeze_func_lm:
            self.codebert.requires_grad_(False)
        elif use_grad_checkpoint and hasattr(self.codebert, "gradient_checkpointing_enable"):
            self.codebert.config.use_cache = False
            self.codebert.gradient_checkpointing_enable()
        if compile_lm:
            try:
                self.codebert = torch.compile(self.codebert, mode="reduce-overhead", dynamic=False)
            except Exception:
                pass  # unsupported platform or torch version — skip silently
        self._codet5p_raw = codet5p_raw_encoder and _is_codet5p_embedding(self.codebert)
        self._codet5p_norm_per_token = codet5p_normalize_per_token
        if self._codet5p_raw:
            d = getattr(self.codebert.config, "d_model", 768)
            self._lm_dim = min(d, matryoshka_dim) if matryoshka_dim else d
        else:
            self._lm_dim = lm_hidden_dim(self.codebert, matryoshka_dim)
        self._is_enc_dec = getattr(
            self.codebert.config, "is_encoder_decoder", False
        )
        self._matryoshka_dim = matryoshka_dim
        self._func_chunk_size = func_chunk_size
        # Default stride to chunk_size // 2 (50% overlap) when not explicitly set
        self._func_chunk_stride = func_chunk_stride if func_chunk_stride > 0 else max(1, func_chunk_size // 2)
        self._lm_per_line = lm_per_line
        # Window attention pool — replaces mean-pool over window CLS vectors when enabled.
        # Only meaningful when func_chunk_size > 0 (sliding window active).
        self._use_window_attn_pool = window_attn_pool and func_chunk_size > 0
        self._use_window_attn_hidden = window_attn_hidden and self._use_window_attn_pool
        self._use_window_center_weight = window_center_weight and func_chunk_size > 0
        # H10: MLP-Mixer over window CLS — alternative chunk aggregator (window↔window mix).
        self._use_window_mixer = window_mixer and func_chunk_size > 0
        # both poolers route through the same win_cls path in _lm_embed_full
        self._use_window_pool = self._use_window_attn_pool or self._use_window_mixer
        if self._use_window_attn_pool:
            self.window_attn_pool = WindowAttentionPool(self._lm_dim)
        if self._use_window_mixer:
            self.window_mixer_pool = WindowMixerPool(self._lm_dim, window_mixer_max)
        if cross_window_attn and self._use_window_pool:
            self.cross_window_attn_module = CrossWindowAttn(self._lm_dim)

    def _lm_embed_full(
        self,
        func_input_ids: torch.Tensor | None,
        func_attention_mask: torch.Tensor | None,
        B: int,
        device: torch.device,
        func_token_lines: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (cls_emb [B, lm_dim], last_hidden_state [B, L, lm_dim] or None).
        last_hidden_state is None for non-BERT LMs that don't expose it.

        When func_chunk_size > 0, uses sliding-window encoding via
        lm_full_windowed — per-token hidden states are aligned to original
        input positions, overlap regions averaged. Localization (mode=lm|both)
        gets full per-line LM features for functions longer than chunk_size.

        When lm_per_line is set (and func_token_lines given): function tokens
        are regrouped per source line, each line forwarded independently → the
        returned hidden carries each line's [CLS] at all its token positions
        (EDAT-style line isolation). Classification cls_emb still comes from a
        function-level forward.
        """
        if func_input_ids is None:
            return torch.zeros(B, self._lm_dim, device=device), None
        # CodeT5+ raw: skip proj/L2-norm, use <s> token hidden (d_model dim).
        if self._codet5p_raw:
            return lm_full_codet5p_raw(
                self.codebert, func_input_ids, func_attention_mask, self._matryoshka_dim,
            )
        # CodeT5+ embedding model — public forward is pooled-only. Pull per-token
        # states from the internal T5 encoder so localization=lm|both works.
        if _is_codet5p_embedding(self.codebert):
            return lm_full_codet5p(
                self.codebert, func_input_ids, func_attention_mask, self._matryoshka_dim,
                normalize_per_token=self._codet5p_norm_per_token,
            )
        # Per-line embedding — EDAT-style line isolation (reuses func tokens)
        if self._lm_per_line and func_token_lines is not None:
            try:
                cls_emb = self._lm_embed(func_input_ids, func_attention_mask, B, device)
                synth_hidden = lm_per_line_embed(
                    self.codebert, func_input_ids, func_token_lines,
                )
                if self._matryoshka_dim is not None:
                    synth_hidden = synth_hidden[:, :, :self._matryoshka_dim]
                return cls_emb, synth_hidden
            except (AttributeError, TypeError):
                return self._lm_embed(func_input_ids, func_attention_mask, B, device), None
        # Sliding-window full forward — per-token hidden aligned to input positions
        if self._func_chunk_size > 0:
            try:
                if getattr(self, "_use_window_pool", False):
                    win_cls, win_mask, hidden = lm_full_windowed(
                        self.codebert, self._is_enc_dec,
                        func_input_ids, func_attention_mask,
                        chunk_size=self._func_chunk_size,
                        stride=self._func_chunk_stride,
                        matryoshka_dim=self._matryoshka_dim,
                        return_window_cls=True,
                        use_center_weight=getattr(self, "_use_window_center_weight", False),
                    )
                    if hasattr(self, "cross_window_attn_module"):
                        hidden = self.cross_window_attn_module(hidden, win_cls, win_mask)
                    if getattr(self, "_use_window_mixer", False):
                        cls = self.window_mixer_pool(win_cls, win_mask)   # H10: MLP-Mixer over window CLS
                    elif getattr(self, "_use_window_attn_hidden", False):
                        cls, win_weights = self.window_attn_pool(win_cls, win_mask, return_weights=True)
                        # Scale per-token hidden by their window's attention weight.
                        # Designed for non-overlapping windows (stride >= chunk_size).
                        L = hidden.size(1)
                        n_wins = win_weights.size(1)
                        tok_win = (torch.arange(L, device=hidden.device) // self._func_chunk_stride).clamp(max=n_wins - 1)
                        scale = win_weights[:, tok_win]           # [B, L]
                        hidden = hidden * scale.unsqueeze(-1)     # [B, L, D]
                    else:
                        cls = self.window_attn_pool(win_cls, win_mask)
                    return cls, hidden
                return lm_full_windowed(
                    self.codebert, self._is_enc_dec,
                    func_input_ids, func_attention_mask,
                    chunk_size=self._func_chunk_size,
                    stride=self._func_chunk_stride,
                    matryoshka_dim=self._matryoshka_dim,
                    use_center_weight=getattr(self, "_use_window_center_weight", False),
                )
            except (AttributeError, TypeError):
                # Non-BERT LM (e.g. CodeT5+) — fall back to pooled CLS-only
                return self._lm_embed(func_input_ids, func_attention_mask, B, device), None
        try:
            out = self.codebert(
                input_ids=func_input_ids,
                attention_mask=func_attention_mask,
            )
            hidden = out.last_hidden_state  # [B, L, hidden]
            if getattr(self, "_is_decoder_only_lm", False):
                if func_attention_mask is not None:
                    last_idx = func_attention_mask.sum(dim=1) - 1  # [B]
                    cls = hidden[torch.arange(B, device=hidden.device), last_idx]
                else:
                    cls = hidden[:, -1]
            else:
                cls = hidden[:, 0]          # [B, hidden] — CLS token
            if self._matryoshka_dim is not None:
                cls    = cls[:, :self._matryoshka_dim]
                hidden = hidden[:, :, :self._matryoshka_dim]
            return cls, hidden
        except (AttributeError, TypeError):
            # CodeT5+ or other models that don't return last_hidden_state in standard form
            return self._lm_embed(func_input_ids, func_attention_mask, B, device), None

    def _lm_embed(
        self,
        func_input_ids: torch.Tensor | None,
        func_attention_mask: torch.Tensor | None,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return LM embedding [B, lm_dim], or zeros when no input provided.

        When func_chunk_size > 0, uses sliding-window encoding so sequences
        longer than chunk_size are split into overlapping windows and
        mean-pooled, keeping each window within the model's trained length.
        """
        if func_input_ids is None:
            return torch.zeros(B, self._lm_dim, device=device)

        if self._codet5p_raw:
            cls, _ = lm_full_codet5p_raw(
                self.codebert, func_input_ids, func_attention_mask, self._matryoshka_dim,
            )
            return cls

        if self._func_chunk_size > 0:
            return lm_pool_windowed(
                self.codebert,
                self._is_enc_dec,
                func_input_ids,
                func_attention_mask,
                chunk_size=self._func_chunk_size,
                stride=self._func_chunk_stride,
                matryoshka_dim=self._matryoshka_dim,
            )

        return lm_pool(
            self.codebert,
            self._is_enc_dec,
            func_input_ids,
            func_attention_mask,
            matryoshka_dim=self._matryoshka_dim,
        )

    @torch.compiler.disable
    def _lm_embed_per_line_raw(
        self,
        func_input_ids: torch.Tensor,
        func_token_lines: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Per-line LM forward → (line_cls [n, lm_dim], uniq_sid [n], B, L).

        Used by live_lm=line: each source line forwarded through the LM
        independently → per-line [CLS]. No whole-function forward. The caller
        (a line-level transformer) recovers cross-line context.
        """
        return lm_per_line_raw(self.codebert, func_input_ids, func_token_lines)

    # ── Optimizer helpers ─────────────────────────────────────────────────────

    def lm_parameters(self) -> list[nn.Parameter]:
        """Return LM parameters for a separate optimizer param group.
        Returns empty list for frozen-embedding models or when freeze_func_lm=True."""
        if hasattr(self, "codebert") and not getattr(self, "_freeze_func_lm", False):
            return list(self.codebert.parameters())
        return []

    def has_live_lm(self) -> bool:
        """True when the model has a trainable live LM branch (not frozen)."""
        return hasattr(self, "codebert") and not getattr(self, "_freeze_func_lm", False)

    # ── Config constructor ────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg, in_channels: int, **kwargs):
        """
        Build model from a Config object.
        Must be implemented by every subclass.
        """
        raise NotImplementedError(
            f"{cls.__name__}.from_config() not implemented."
        )

    # ── Forward (abstract) ────────────────────────────────────────────────────

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_line: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        func_input_ids: torch.Tensor | None = None,
        func_attention_mask: torch.Tensor | None = None,
        func_token_lines: torch.Tensor | None = None,
    ):
        ...
