"""Shared helpers for live LM branches across model architectures."""

from __future__ import annotations

import torch
from loguru import logger


def _is_codet5p_embedding(model) -> bool:
    """True for Salesforce/codet5p-*-embedding — returns raw pooled tensor, not ModelOutput."""
    return getattr(model.config, "model_type", "") == "codet5p_embedding"


def _is_t5_like(model) -> bool:
    """True for T5-family models (enc-dec or enc-only), False for BERT-family."""
    if _is_codet5p_embedding(model):
        return False
    cfg = model.config
    return (
        getattr(cfg, "is_encoder_decoder", False)
        or "t5" in getattr(cfg, "model_type", "").lower()
    )


def _is_decoder_only(model) -> bool:
    """True for decoder-only (GPT/Qwen/LLaMA-family) models."""
    cfg = model.config
    return (
        not getattr(cfg, "is_encoder_decoder", False)
        and getattr(cfg, "model_type", "") in {
            "qwen2", "gpt2", "llama", "mistral", "gemma", "phi", "falcon", "bloom",
        }
    )


def lm_hidden_dim(model, matryoshka_dim: int | None = None) -> int:
    """Return hidden size of the LM; returns matryoshka_dim if set."""
    if matryoshka_dim is not None:
        return matryoshka_dim
    if _is_codet5p_embedding(model):
        # Internal T5 hidden_size (768) != projected output (256). Probe to get real dim.
        device = next(model.parameters()).device
        dummy = torch.zeros(1, 2, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=dummy)
        return out.shape[-1]
    cfg = model.config
    if _is_t5_like(model):
        return getattr(cfg, "d_model", cfg.hidden_size)
    return cfg.hidden_size


def lm_pool(
    model,
    is_enc_dec: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    matryoshka_dim: int | None = None,
    mean_pool: bool = False,
) -> torch.Tensor:
    """
    Extract fixed-size LM representation.
    T5 enc-dec / enc-only: mean-pool over encoder output.
    Decoder-only (Qwen2 etc.): last non-padding token.
    BERT-family: CLS token (position 0), or mask-mean-pool when mean_pool=True.
    Truncates to matryoshka_dim if set.

    mean_pool: BERT-family only. Mask-mean-pool over tokens instead of taking
    position 0. Used for sliding-window chunks — a chunk that starts mid-function
    has no real <s> at position 0, so its [CLS] is meaningless; mean-pool gives
    a valid per-chunk vector regardless of where the slice begins.
    """
    if _is_codet5p_embedding(model):
        # codet5p-110m-embedding uses T5 attention internally — relative-position
        # bias overflows in fp16/bf16 under AMP → NaN. Force float32.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            emb = model(input_ids=input_ids, attention_mask=attention_mask)
        emb = emb.float()
    elif _is_t5_like(model):
        enc = model.encoder if is_enc_dec else model
        # T5 relative-position bias overflows in fp16/bf16 → NaN loss under AMP.
        # Force float32 for just the encoder forward regardless of outer autocast.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            out = enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hs = out.last_hidden_state.float()  # [B, seq, d_model]
        mask = (
            attention_mask.unsqueeze(-1).float()
            if attention_mask is not None
            else torch.ones(*input_ids.shape, 1, device=input_ids.device)
        )
        emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif _is_decoder_only(model):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state  # [B, seq, hidden]
        if attention_mask is not None:
            left_pad = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_pad:
                emb = hs[:, -1]
            else:
                last_idx = attention_mask.sum(dim=1) - 1  # [B]
                emb = hs[torch.arange(hs.size(0), device=hs.device), last_idx]
        else:
            emb = hs[:, -1]
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state                       # [B, seq, hidden]
        if mean_pool:
            mask = (
                attention_mask.unsqueeze(-1).to(hs.dtype)
                if attention_mask is not None
                else torch.ones(*input_ids.shape, 1, device=input_ids.device, dtype=hs.dtype)
            )
            emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            emb = hs[:, 0]

    if matryoshka_dim is not None:
        emb = emb[:, :matryoshka_dim]
    return emb


def lm_pool_windowed(
    model,
    is_enc_dec: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    chunk_size: int = 512,
    stride: int = 256,
    matryoshka_dim: int | None = None,
) -> torch.Tensor:
    """
    Sliding-window LM encoding for sequences longer than chunk_size.

    The sequence is split into overlapping windows of `chunk_size` tokens,
    stepping by `stride` tokens each time (overlap = chunk_size - stride).
    Each window is encoded independently with lm_pool(), then all window
    embeddings are mean-pooled into a single [B, hidden] vector.

    Falls back to a single lm_pool() call when the sequence fits in one window,
    so there is no overhead for short functions.

    Parameters
    ----------
    chunk_size : int
        Tokens per window. Must be ≤ model's trained max length (e.g. 512).
    stride : int
        Step between window starts. stride < chunk_size → overlapping windows.
        stride == chunk_size → non-overlapping (faster, less context sharing).
        Recommended: stride = chunk_size // 2 for 50% overlap.
    """
    B, L = input_ids.shape

    # Fast path — sequence fits in a single window
    if L <= chunk_size:
        return lm_pool(model, is_enc_dec, input_ids, attention_mask, matryoshka_dim)

    # Clamp stride to [1, chunk_size] to avoid infinite loops or no-ops
    stride = max(1, min(stride, chunk_size))

    chunk_embs: list[torch.Tensor] = []   # [B, hidden] per window
    valid_counts: list[torch.Tensor] = [] # [B] float — 1.0 if sample has real tokens in window

    start = 0
    while start < L:
        end = min(start + chunk_size, L)
        ids_chunk  = input_ids[:, start:end]
        mask_chunk = attention_mask[:, start:end] if attention_mask is not None else None

        # Per-sample: which samples have at least one real token in this window?
        if mask_chunk is not None:
            per_sample_valid = (mask_chunk.sum(dim=1) > 0).float()  # [B]
        else:
            per_sample_valid = torch.ones(B, device=input_ids.device)

        # Skip window entirely only when NO sample has real tokens
        if per_sample_valid.sum() == 0:
            if end == L:
                break
            start += stride
            continue

        # mean_pool=True — a sliding-window chunk has no real <s> at position 0
        # (only chunk 0 starts at the function's <s>). Mask-mean-pool the chunk's
        # tokens for a valid per-chunk vector.
        emb = lm_pool(model, is_enc_dec, ids_chunk, mask_chunk, matryoshka_dim,
                      mean_pool=True)  # [B, hidden]
        # Zero out embedding for samples that have no real tokens in this window
        emb = emb * per_sample_valid.unsqueeze(-1)
        chunk_embs.append(emb)
        valid_counts.append(per_sample_valid)

        if end == L:
            break
        start += stride

    # Fall back to first-window single pass if every window was all-padding (degenerate)
    if not chunk_embs:
        ids_fb = input_ids[:, :chunk_size]
        mask_fb = attention_mask[:, :chunk_size] if attention_mask is not None else None
        return lm_pool(model, is_enc_dec, ids_fb, mask_fb, matryoshka_dim, mean_pool=True)

    # Per-sample weighted mean: divide by number of valid windows per sample
    embs  = torch.stack(chunk_embs, dim=1)        # [B, n_windows, hidden]
    counts = torch.stack(valid_counts, dim=1)      # [B, n_windows]
    sum_embs = embs.sum(dim=1)                     # [B, hidden]
    count    = counts.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
    return sum_embs / count


def lm_full_codet5p(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    matryoshka_dim: int | None = None,
    normalize_per_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token + pooled embeddings for a CodeT5+ embedding model.

    The public forward of `codet5p-*-embedding` returns only the pooled
    [B, d] vector. For statement localization we also need per-token states:
    run the internal T5 encoder, then apply the model's projection head per
    token so the per-token features live in the SAME projected space as the
    pooled embedding.

    normalize_per_token: apply F.normalize(dim=-1) to per_token so it matches
    the unit-norm scale of the pooled embedding (F6 ablation).

    Returns (pooled [B, d], per_token [B, L, d]).
    T5 relative-position bias overflows in fp16/bf16 → force fp32.
    """
    import torch.nn.functional as F
    dev = input_ids.device
    with torch.autocast(device_type=dev.type, enabled=False):
        enc = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = enc.last_hidden_state.float()                  # [B, L, d_model]
        per_token = model.proj(hs)                          # [B, L, d]
        # Reuse per_token[<s>] for pooled — avoids a second full encoder forward.
        # codet5p-*-embedding pools the <s> token (index 0), projects, then L2-normalises.
        pooled = F.normalize(per_token[:, 0, :], dim=-1).float()       # [B, d]
    if normalize_per_token:
        per_token = F.normalize(per_token, dim=-1)
    if matryoshka_dim is not None:
        pooled    = pooled[:, :matryoshka_dim]
        per_token = per_token[:, :, :matryoshka_dim]
    return pooled, per_token


def lm_full_codet5p_raw(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    matryoshka_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw encoder hidden states for CodeT5+ — no proj, no L2-norm.

    Uses the <s> token (position 0) as the classification embedding,
    analogous to [CLS] in BERT/UniXcoder. Per-token = full encoder output.
    dim = d_model (768 for codet5p-110m-embedding).

    Returns (cls [B, d_model], per_token [B, L, d_model]).
    """
    dev = input_ids.device
    with torch.autocast(device_type=dev.type, enabled=False):
        enc = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = enc.last_hidden_state.float()   # [B, L, d_model]
    cls = hs[:, 0, :]                        # <s> token [B, d_model]
    if matryoshka_dim is not None:
        cls = cls[:, :matryoshka_dim]
        hs  = hs[:, :, :matryoshka_dim]
    return cls, hs


class WindowAttentionPool(torch.nn.Module):
    """Attention pooling over per-window CLS vectors → single global CLS.

    Replaces mean-pool over sliding-window CLS with a learned attention score,
    so the model can up-weight windows most relevant to the CWE class.

    Input : window_cls [B, N, hidden], valid_mask [B, N] bool (optional)
    Output: [B, hidden] — attention-weighted sum over valid windows
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = torch.nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        window_cls: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(window_cls.to(self.score.weight.dtype)).squeeze(-1)  # [B, N]
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float("-inf"))
        attn = torch.softmax(logits, dim=-1)                  # [B, N]
        # Guard: if all windows invalid for a sample, attn = NaN → zero out
        if valid_mask is not None:
            any_valid = valid_mask.any(dim=-1, keepdim=True)  # [B, 1]
            attn = torch.where(any_valid, attn, torch.zeros_like(attn))
        cls = (attn.unsqueeze(-1) * window_cls).sum(dim=1)   # [B, hidden]
        if return_weights:
            return cls, attn                                  # [B, hidden], [B, N]
        return cls


class WindowMixerPool(torch.nn.Module):
    """MLP-Mixer over per-window CLS vectors → single global CLS (H10).

    The ONLY chunk-aggregator with window↔window mixing (mean / WindowAttentionPool
    just average / reweight, no cross-window interaction). MixerBlock with reference
    mlp_mixer.py dims (token_mix nhid*4, channel_mix nhid//2) over [B, W, D] padded to
    max_windows (token-mix MLP is fixed-size), then masked mean over valid windows.

    Input : window_cls [B, N, D], valid_mask [B, N] bool (optional)
    Output: [B, D]
    """

    def __init__(self, hidden_dim: int, max_windows: int, dropout: float = 0.0):
        super().__init__()
        self.max_windows = max_windows
        ch = max(hidden_dim // 2, 1)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.token_mix = torch.nn.Sequential(
            torch.nn.Linear(max_windows, hidden_dim * 4), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim * 4, max_windows))
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.channel_mix = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, ch), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(ch, hidden_dim))

    def forward(self, window_cls, valid_mask=None):
        w_dtype = self.norm1.weight.dtype
        x = window_cls.to(w_dtype)
        B, W, _ = x.shape
        if valid_mask is None:
            valid_mask = torch.ones(B, W, dtype=torch.bool, device=x.device)
        # pad / truncate to max_windows (fixed-size token-mix MLP)
        if W < self.max_windows:
            pad = self.max_windows - W
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            valid_mask = torch.nn.functional.pad(valid_mask, (0, pad), value=False)
        elif W > self.max_windows:
            x, valid_mask = x[:, :self.max_windows], valid_mask[:, :self.max_windows]
        # MixerBlock: token-mix over windows + channel-mix over features
        y = self.norm1(x).transpose(1, 2)                 # [B, D, maxW]
        x = x + self.token_mix(y).transpose(1, 2)
        x = x + self.channel_mix(self.norm2(x))
        m = valid_mask.unsqueeze(-1).to(x.dtype)          # masked mean over valid windows
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)   # [B, D]


class CrossWindowAttn(torch.nn.Module):
    """Cross-attention from per-token hidden states to window CLS vectors.

    Each token attends over window-level summaries to refine its representation
    with global function context. Residual + LayerNorm.

    Input:
        hidden  : [B, L, D] — per-token hidden states (overlap-averaged)
        win_cls : [B, W, D] — per-window CLS vectors
        win_mask: [B, W] bool — True = valid window (optional)
    Output: [B, L, D] — refined hidden states
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        win_cls: torch.Tensor,
        win_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kpm = (~win_mask) if win_mask is not None else None
        w_dtype = next(self.parameters()).dtype
        attn_out, _ = self.cross_attn(
            hidden.to(w_dtype), win_cls.to(w_dtype), win_cls.to(w_dtype),
            key_padding_mask=kpm,
        )
        return self.norm(hidden.to(w_dtype) + attn_out)


def lm_full_windowed(
    model,
    is_enc_dec: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    chunk_size: int = 512,
    stride: int = 256,
    matryoshka_dim: int | None = None,
    return_window_cls: bool = False,
    use_center_weight: bool = False,
) -> tuple[torch.Tensor, ...]:
    """
    Sliding-window LM forward that returns per-token hidden states aligned to
    the original input positions. Used by `_lm_embed_full` for localization
    (mode=lm|both) so long functions retain LM features past the model's
    trained max length.

    For input [B, L], slides chunk_size windows with stride. Each original
    token position's hidden state = mean of all chunks that contain it.
    Overlap regions average naturally via accumulator + count.

    Returns (return_window_cls=False, default):
        cls    : [B, hidden] — mean-pool over all windows
        hidden : [B, L, hidden] — per-original-position averaged hidden states

    Returns (return_window_cls=True):
        window_cls  : [B, N_windows, hidden] — per-window mean-pooled CLS
        valid_mask  : [B, N_windows] bool — True if window has real tokens for that sample
        hidden      : [B, L, hidden]

    Fast path: if L ≤ chunk_size, returns a single forward (N_windows=1).
    """
    B, L = input_ids.shape
    device = input_ids.device

    # Fast path — fits one window, no sliding needed
    if L <= chunk_size:
        out = model.encoder(input_ids=input_ids, attention_mask=attention_mask) \
              if is_enc_dec else \
              model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if matryoshka_dim is not None:
            hidden = hidden[:, :, :matryoshka_dim]
        if _is_decoder_only(model):
            if attention_mask is not None:
                last_idx = attention_mask.sum(dim=1) - 1     # [B]
                cls = hidden[torch.arange(B, device=device), last_idx]
            else:
                cls = hidden[:, -1]
        else:
            cls = hidden[:, 0]
        if return_window_cls:
            # Single window: [B, 1, H], all valid
            win_cls  = cls.unsqueeze(1)                       # [B, 1, H]
            win_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            return win_cls, win_mask, hidden
        return cls, hidden

    stride = max(1, min(stride, chunk_size))

    # Accumulate in fp32 for precision; cast back to LM dtype at the end.
    hidden_dim = model.config.hidden_size if matryoshka_dim is None else matryoshka_dim
    hidden_acc = torch.zeros(B, L, hidden_dim, device=device, dtype=torch.float32)
    count      = torch.zeros(B, L, 1,          device=device, dtype=torch.float32)
    # Per-window CLS accumulators
    win_cls_list:   list[torch.Tensor] = []   # [B, H] per window
    win_valid_list: list[torch.Tensor] = []   # [B] bool per window
    # Fallback mean-pool accumulators (return_window_cls=False)
    cls_acc = torch.zeros(B, hidden_dim, device=device, dtype=torch.float32)
    cls_cnt = torch.zeros(B,             device=device, dtype=torch.float32)
    out_dtype = None

    start = 0
    while start < L:
        end = min(start + chunk_size, L)
        ids_chunk  = input_ids[:, start:end]
        mask_chunk = attention_mask[:, start:end] if attention_mask is not None else None

        # Skip windows where ALL samples have no real tokens
        if mask_chunk is not None and mask_chunk.sum() == 0:
            if end == L:
                break
            start += stride
            continue

        out = model.encoder(input_ids=ids_chunk, attention_mask=mask_chunk) \
              if is_enc_dec else \
              model(input_ids=ids_chunk, attention_mask=mask_chunk)
        chunk_hidden = out.last_hidden_state              # [B, end-start, H]
        if matryoshka_dim is not None:
            chunk_hidden = chunk_hidden[:, :, :matryoshka_dim]
        if out_dtype is None:
            out_dtype = chunk_hidden.dtype

        chunk_fp32 = chunk_hidden.float()
        # Center weighting: downweight boundary tokens (less context at window edges).
        # Applied only to hidden_acc/count — win_cls stays plain mean-pooled.
        chunk_len = end - start
        if use_center_weight and chunk_len > 1:
            half = (chunk_len - 1) / 2.0
            pos = torch.arange(chunk_len, device=device, dtype=torch.float32)
            cw = (1.0 - 0.9 * (pos - half).abs() / half).view(1, chunk_len, 1)
        else:
            cw = None
        if mask_chunk is not None:
            w = mask_chunk.unsqueeze(-1).float()          # [B, end-start, 1]
            eff_w = w * cw if cw is not None else w       # center-weighted mask
            hidden_acc[:, start:end] += chunk_fp32 * eff_w
            count[:, start:end]      += eff_w
            w_1d    = mask_chunk.float()                  # [B, end-start]
            valid_b = (w_1d.sum(dim=1) > 0)              # [B] bool
            win_mean = (chunk_fp32 * w_1d.unsqueeze(-1)).sum(dim=1) \
                       / w_1d.sum(dim=1, keepdim=True).clamp(min=1)  # [B, H]
            win_cls_list.append(win_mean)
            win_valid_list.append(valid_b)
            cls_acc += win_mean * valid_b.float().unsqueeze(-1)
            cls_cnt += valid_b.float()
        else:
            eff_w = cw if cw is not None else 1.0
            hidden_acc[:, start:end] += chunk_fp32 * eff_w
            count[:, start:end]      += (cw if cw is not None else 1.0)
            win_mean = chunk_fp32.mean(dim=1)
            win_cls_list.append(win_mean)
            win_valid_list.append(torch.ones(B, dtype=torch.bool, device=device))
            cls_acc += win_mean
            cls_cnt += 1.0

        if end == L:
            break
        start += stride

    _dtype = out_dtype or input_ids.new_zeros(1).float().dtype
    hidden = (hidden_acc / count.clamp(min=1)).to(_dtype)

    if return_window_cls:
        if win_cls_list:
            win_cls  = torch.stack(win_cls_list,  dim=1).to(_dtype)  # [B, N, H]
            win_mask = torch.stack(win_valid_list, dim=1)             # [B, N] bool
        else:
            # Degenerate: all windows empty — single zero window
            win_cls  = torch.zeros(B, 1, hidden_dim, device=device, dtype=_dtype)
            win_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return win_cls, win_mask, hidden

    cls = (cls_acc / cls_cnt.unsqueeze(-1).clamp(min=1)).to(_dtype)
    return cls, hidden


_PERLINE_MAX_LINE = 100_000   # statement-id base — must exceed any source line number


def lm_per_line_raw(
    model,
    input_ids: torch.Tensor,
    token_lines: torch.Tensor,
    sub_batch: int = 512,
    max_line_len: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Per-line LM forward — EDAT-style line isolation, reusing func tokens.

    Each source line's tokens (grouped via token_lines) are re-wrapped with
    [CLS]/[SEP] and forwarded through the LM independently → per-line [CLS].
    No separate per-line tokenization, no .pt rebuild.

    Returns (line_cls [n, lm_dim], uniq_sid [n], B, L) where n = total source
    lines across the batch and uniq_sid = sorted (b*_PERLINE_MAX_LINE + line).
    The per-graph line index is uniq_sid // _PERLINE_MAX_LINE.

    Fully vectorized — the only loop is the sub-batched LM forward.

    Parameters
    ----------
    input_ids   : [B, L] function token ids (from func_input_ids)
    token_lines : [B, L] per-token source line (-1 = special/pad)
    sub_batch   : per-line forward sub-batch size (memory guard)
    max_line_len: cap tokens per line (incl. [CLS]/[SEP])
    """
    B, L = input_ids.shape
    device = input_ids.device
    cfg = model.config
    cls_id = cfg.bos_token_id if getattr(cfg, "bos_token_id", None) is not None else 0
    sep_id = cfg.eos_token_id if getattr(cfg, "eos_token_id", None) is not None else 2
    pad_id = cfg.pad_token_id if getattr(cfg, "pad_token_id", None) is not None else 1
    lm_dim = cfg.hidden_size

    # ── Flatten + keep valid tokens ────────────────────────────────────────
    tok_b   = torch.arange(B, device=device).unsqueeze(1).expand(B, L).reshape(-1)
    tl_flat = token_lines.reshape(-1)
    id_flat = input_ids.reshape(-1)
    valid   = tl_flat >= 0
    if not valid.any():
        return (torch.zeros(0, lm_dim, device=device),
                torch.zeros(0, dtype=torch.long, device=device), B, L)

    v_b, v_tl, v_ids = tok_b[valid], tl_flat[valid], id_flat[valid]
    sid = v_b * _PERLINE_MAX_LINE + v_tl                          # statement id per token
    uniq_sid, inv, counts = torch.unique(
        sid, sorted=True, return_inverse=True, return_counts=True)
    n = uniq_sid.shape[0]

    # ── Within-statement token position (vectorized) ──────────────────────
    order      = torch.argsort(inv, stable=True)                  # group tokens by stmt
    inv_sorted = inv[order]
    ids_sorted = v_ids[order]
    seg_start  = torch.zeros(n, dtype=torch.long, device=device)
    seg_start[1:] = torch.cumsum(counts, 0)[:-1]
    within = torch.arange(inv_sorted.shape[0], device=device) - seg_start[inv_sorted]

    # ── Build padded [n, max_len] : [CLS] line_tokens [SEP] ───────────────
    max_tok = int(counts.max().clamp(max=max_line_len - 2).item())
    max_len = max_tok + 2
    batch_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    keep = within < max_tok                                       # truncate over-long lines
    batch_ids[inv_sorted[keep], within[keep] + 1] = ids_sorted[keep]
    batch_ids[:, 0] = cls_id
    line_len = counts.clamp(max=max_tok)                          # tokens kept per stmt
    batch_ids[torch.arange(n, device=device), line_len + 1] = sep_id
    ar = torch.arange(max_len, device=device).unsqueeze(0)
    batch_msk = (ar <= (line_len + 1).unsqueeze(1)).long()        # [n, max_len]

    # ── Per-line forward, sub-batched (memory guard) ──────────────────────
    cls_out = torch.zeros(n, lm_dim, device=device)
    for st in range(0, n, sub_batch):
        en = min(st + sub_batch, n)
        out = model(input_ids=batch_ids[st:en], attention_mask=batch_msk[st:en])
        cls_out[st:en] = out.last_hidden_state[:, 0].float()

    return cls_out, uniq_sid, B, L


@torch.compiler.disable
def scatter_lines_to_tokens(
    per_line: torch.Tensor,
    uniq_sid: torch.Tensor,
    token_lines: torch.Tensor,
    B: int,
    L: int,
) -> torch.Tensor:
    """Scatter per-line vectors → synthetic per-token hidden [B, L, d].

    Every token of line ℓ carries line ℓ's vector. Downstream per-line pooling
    (StmtHead, statement_features) recovers it unchanged — so no downstream code
    needs modification. per_line [n, d] is indexed by uniq_sid (sorted).
    """
    device = per_line.device
    d = per_line.shape[-1]
    n = uniq_sid.shape[0]
    if n == 0:
        return torch.zeros(B, L, d, device=device)
    tl_flat = token_lines.reshape(-1)
    tok_b   = torch.arange(B, device=device).unsqueeze(1).expand(B, L).reshape(-1)
    valid   = tl_flat >= 0
    synth   = torch.zeros(B * L, d, device=device, dtype=per_line.dtype)
    tok_sid = tok_b * _PERLINE_MAX_LINE + tl_flat                 # [B*L]
    pos = torch.searchsorted(uniq_sid, tok_sid).clamp(0, n - 1)
    matched = (uniq_sid[pos] == tok_sid) & valid
    synth[matched] = per_line[pos[matched]]
    return synth.reshape(B, L, d)


def lm_per_line_embed(
    model,
    input_ids: torch.Tensor,
    token_lines: torch.Tensor,
    sub_batch: int = 512,
    max_line_len: int = 128,
) -> torch.Tensor:
    """Per-line LM embedding → synthetic per-token hidden [B, L, lm_dim].

    Thin wrapper: per-line forward (lm_per_line_raw) then scatter each line's
    [CLS] back onto its tokens. Used by live_lm=func_and_line.
    """
    cls_out, uniq_sid, B, L = lm_per_line_raw(
        model, input_ids, token_lines, sub_batch, max_line_len)
    return scatter_lines_to_tokens(cls_out, uniq_sid, token_lines, B, L)
