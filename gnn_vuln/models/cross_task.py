"""cross_task.py — Phase 2 bidirectional cross-task modules for lmgat_codebert.

Localization is **per-statement** (per-line) — like EDAT's per-line MMOE. The
cross-task conditions each statement individually; it does NOT collapse the
localization view to a per-graph vector. Classification stays per-graph.

Each module forward:
    forward(fused, loc_feats, stmt_graph, h, batch, B, lm_hidden, func_token_lines)
      fused      [B, fused_dim]  — classification representation (per-graph)
      loc_feats  [S, loc_dim]    — per-statement localization features
      stmt_graph [S]             — graph index of each statement
    returns (fused_mod [B, fused_dim], stmt_cond [S, loc_dim])

Methods (config `cross_task_method`):
  cross_attention — decoder cross-attention (statement ↔ nodes, fused ↔ statements)
  self_attention  — EDAT-style: statements self-attend, query biased by classification
  mmoe            — Multi-gate Mixture-of-Experts (Ma et al. 2018) — EDAT's code
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

_MAX_LINE = 100_000   # must match StmtHead's statement-id formula


def loc_dim(mode: str, hidden_dim: int, lm_dim: int) -> int:
    """Dimension of the per-statement localization representation."""
    if mode == "gnn":
        return hidden_dim
    if mode == "lm":
        return lm_dim
    return hidden_dim + lm_dim   # both


def statement_features(
    h: torch.Tensor,
    batch: torch.Tensor,
    node_line: torch.Tensor,
    lm_hidden: torch.Tensor | None,
    func_token_lines: torch.Tensor | None,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-statement mean-pooled localization features.

    Returns (feats [S, loc_dim], stmt_graph [S], B). S statements are ordered by
    sorted (graph, line) — identical ordering to StmtHead's vectorized scorer, so
    a [S]-indexed cond aligns directly with StmtHead's statements.
    """
    device = h.device
    B = int(batch.max().item()) + 1
    valid = node_line >= 0
    if not valid.any():
        ld = loc_dim(mode, h.shape[1],
                     lm_hidden.shape[-1] if lm_hidden is not None else 0)
        return (torch.zeros(0, ld, device=device),
                torch.zeros(0, dtype=torch.long, device=device), B)

    hv, lv, bv = h[valid], node_line[valid], batch[valid]
    sid = bv * _MAX_LINE + lv
    unique_sid, inv = torch.unique(sid, sorted=True, return_inverse=True)
    S = unique_sid.shape[0]
    stmt_graph = unique_sid // _MAX_LINE

    parts: list[torch.Tensor] = []

    if mode in ("gnn", "both"):
        D = h.shape[1]
        gsum = torch.zeros(S, D, device=device, dtype=hv.dtype)
        gcnt = torch.zeros(S, 1, device=device, dtype=hv.dtype)
        gsum.scatter_add_(0, inv.unsqueeze(1).expand(-1, D), hv)
        gcnt.scatter_add_(0, inv.unsqueeze(1),
                          torch.ones(hv.shape[0], 1, device=device, dtype=hv.dtype))
        parts.append(gsum / gcnt.clamp(min=1))

    if mode in ("lm", "both"):
        LM_D = lm_hidden.shape[-1]
        lm_feat = torch.zeros(S, LM_D, device=device, dtype=lm_hidden.dtype)
        if func_token_lines is not None:
            L = lm_hidden.shape[1]
            g_tok = torch.arange(B, device=device).unsqueeze(1).expand(-1, L).reshape(-1)
            tl_tok = func_token_lines.reshape(-1)
            lm_flat = lm_hidden.reshape(-1, LM_D)
            vt = tl_tok >= 0
            if vt.any():
                g_tok, tl_tok, lm_flat = g_tok[vt], tl_tok[vt], lm_flat[vt]
                tsid = g_tok * _MAX_LINE + tl_tok
                u_tsid, inv_t = torch.unique(tsid, sorted=True, return_inverse=True)
                ST = u_tsid.shape[0]
                ssum = torch.zeros(ST, LM_D, device=device, dtype=lm_flat.dtype)
                scnt = torch.zeros(ST, 1, device=device, dtype=lm_flat.dtype)
                ssum.scatter_add_(0, inv_t.unsqueeze(1).expand(-1, LM_D), lm_flat)
                scnt.scatter_add_(0, inv_t.unsqueeze(1),
                                  torch.ones(lm_flat.shape[0], 1, device=device, dtype=lm_flat.dtype))
                lm_mean = ssum / scnt.clamp(min=1)
                pos = torch.searchsorted(u_tsid, unique_sid).clamp(0, ST - 1)
                found = u_tsid[pos] == unique_sid
                lm_feat[found] = lm_mean[pos[found]]
        parts.append(lm_feat)

    return torch.cat(parts, dim=-1), stmt_graph, B


def _pool_to_graph(feats: torch.Tensor, stmt_graph: torch.Tensor, B: int) -> torch.Tensor:
    """Per-statement [S, d] → per-graph mean [B, d]."""
    device, d = feats.device, feats.shape[1]
    g = torch.zeros(B, d, device=device, dtype=feats.dtype)
    c = torch.zeros(B, 1, device=device, dtype=feats.dtype)
    if feats.shape[0] > 0:
        g.scatter_add_(0, stmt_graph.unsqueeze(1).expand(-1, d), feats)
        c.scatter_add_(0, stmt_graph.unsqueeze(1),
                       torch.ones(feats.shape[0], 1, device=device, dtype=feats.dtype))
    return g / c.clamp(min=1)


class _LineLevelEncoder(nn.Module):
    """Transformer encoder over per-statement features — EDAT's
    line_level_encoder pattern. Statements within each graph self-attend, then
    project to D. Used as the MMOE per-task encoder for the LOCALIZATION path
    when `mmoe_loc_transformer=true`. Cross-statement context recovered at
    line granularity (vs the general MLP encoder which processes each
    statement independently).
    """

    def __init__(self, in_dim: int, D: int, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, D)
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads, dim_feedforward=D * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, stmt_graph: torch.Tensor, B: int) -> torch.Tensor:
        """x [S, in_dim], stmt_graph [S], B = num graphs → [S, D]."""
        if x.shape[0] == 0:
            return self.proj(x)
        x_proj = self.proj(x.float())                              # [S, D]
        x_dense, mask = to_dense_batch(x_proj, stmt_graph, batch_size=B)   # [B, Smax, D]
        kpad = ~mask
        empty = ~mask.any(dim=1)
        if empty.any():
            kpad = kpad.clone(); kpad[empty, 0] = False
        out = self.encoder(x_dense, src_key_padding_mask=kpad)     # [B, Smax, D]
        return out[mask]                                           # [S, D] unpadded


class MMOECrossTask(nn.Module):
    """Multi-gate Mixture-of-Experts (Ma et al. 2018, KDD) — matches EDAT's code.

    Shared expert MLP pool; one softmax gate per task. Classification runs the
    experts per-graph, localization runs them per-statement. Cross-task transfer
    = the shared experts are trained by both losses.

    Task-specific encoders before the experts:
      • cls_in (per-graph fused vector): Linear or MLP — `task_encoder` flag.
      • loc_in (per-statement features): Linear / MLP (general) OR transformer
        (EDAT line_level_encoder style — statements self-attend) when
        `loc_transformer=true`. The transformer recovers cross-statement
        context, especially useful with per-line LM embedding.
    """

    def __init__(self, fused_dim: int, hidden_dim: int, lm_dim: int, mode: str,
                 expert_num: int = 4, expert_dim: int | None = None,
                 task_encoder: bool = False, residual: bool = True,
                 loc_transformer: bool = False,
                 loc_transformer_layers: int = 2,
                 loc_transformer_heads: int = 4):
        super().__init__()
        self._residual = residual
        self._loc_is_transformer = loc_transformer
        ld = loc_dim(mode, hidden_dim, lm_dim)
        D = expert_dim or hidden_dim

        def _general(in_dim: int) -> nn.Module:
            """General task-specific encoder — operates on a single vector per
            unit (per-graph for cls, per-statement for loc)."""
            if task_encoder:
                return nn.Sequential(
                    nn.Linear(in_dim, D), nn.LayerNorm(D), nn.ReLU(),
                    nn.Dropout(0.1), nn.Linear(D, D),
                )
            return nn.Linear(in_dim, D)

        # Classification: always uses the general encoder (single per-graph vector;
        # transformer attention over a single token would degenerate to FFN).
        self.cls_in = _general(fused_dim)

        # Localization: optionally transformer over statements (line-level).
        if loc_transformer:
            self.loc_in = _LineLevelEncoder(
                ld, D,
                num_layers=loc_transformer_layers,
                num_heads=loc_transformer_heads,
            )
        else:
            self.loc_in = _general(ld)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, D), nn.LayerNorm(D), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(D, D), nn.LayerNorm(D), nn.ReLU(),
            ) for _ in range(expert_num)
        ])
        self.gate_cls = nn.Linear(D, expert_num)
        self.gate_loc = nn.Linear(D, expert_num)
        self.cls_out = nn.Linear(D, fused_dim)
        self.loc_out = nn.Linear(D, ld)
        self.res_gate_cls = nn.Parameter(torch.zeros(1))
        self.res_gate_loc = nn.Parameter(torch.zeros(1))

    def _moe(self, x: torch.Tensor, gate: nn.Module) -> torch.Tensor:
        """x [*, D] → gated expert mixture [*, D] (works per-graph or per-statement)."""
        experts = torch.stack([ex(x) for ex in self.experts], dim=1)   # [*,N,D]
        weights = torch.softmax(gate(x), dim=-1).unsqueeze(-1)         # [*,N,1]
        return (experts * weights).sum(dim=1)

    def forward(self, fused, loc_feats, stmt_graph, h, batch, B,
                lm_hidden, func_token_lines):
        cross_c = self.cls_out(self._moe(self.cls_in(fused.float()), self.gate_cls))
        if self._loc_is_transformer:
            x_loc = self.loc_in(loc_feats, stmt_graph, B)
        else:
            x_loc = self.loc_in(loc_feats.float())
        cross_l = self.loc_out(self._moe(x_loc, self.gate_loc))
        if self._residual:
            fused_mod = fused + self.res_gate_cls * cross_c
            stmt_cond = self.res_gate_loc * cross_l
        else:
            fused_mod = cross_c
            stmt_cond = cross_l
        return fused_mod, stmt_cond


class CrossTaskAttn(nn.Module):
    """Decoder-style cross-attention, per-statement localization.

    loc direction: each statement queries its graph's encoder units (nodes / LM
                   tokens) → per-statement conditioning.
    cls direction: the fused vector queries the graph's statements → per-graph.
    """

    def __init__(self, fused_dim: int, hidden_dim: int, lm_dim: int,
                 mode: str, num_heads: int = 4, residual: bool = True):
        super().__init__()
        self._mode = mode
        self._residual = residual
        ld = loc_dim(mode, hidden_dim, lm_dim)
        self._ld = ld
        A = hidden_dim                                  # internal attention dim
        self._A = A
        # loc direction — statement query, encoder-unit K/V
        self.q_stmt   = nn.Linear(ld, A)
        self.kv_node  = nn.Linear(hidden_dim, A) if mode in ("gnn", "both") else None
        self.kv_tok   = nn.Linear(lm_dim, A)     if mode in ("lm", "both")  else None
        self.attn_loc = nn.MultiheadAttention(A, num_heads, batch_first=True)
        self.loc_out  = nn.Linear(A, ld)
        # cls direction — fused query, statement K/V
        self.q_fused  = nn.Linear(fused_dim, A)
        self.kv_stmt  = nn.Linear(ld, A)
        self.attn_cls = nn.MultiheadAttention(A, num_heads, batch_first=True)
        self.to_cls   = nn.Linear(A, fused_dim)
        self.gate_cls = nn.Parameter(torch.zeros(1))
        self.gate_loc = nn.Parameter(torch.zeros(1))

    def forward(self, fused, loc_feats, stmt_graph, h, batch, B,
                lm_hidden, func_token_lines):
        fused = fused.float()
        loc_feats = loc_feats.float()
        h = h.float()
        if lm_hidden is not None:
            lm_hidden = lm_hidden.float()
        S = loc_feats.shape[0]

        # dense statement tensors (shared by both directions)
        stmt_dense, stmt_mask = to_dense_batch(loc_feats, stmt_graph, batch_size=B)  # [B,Sm,ld]

        # ── loc direction: statements query encoder units → per-statement ──
        q = self.q_stmt(stmt_dense)                                     # [B,Sm,A]
        kv_parts, kpad_parts = [], []
        if self.kv_node is not None:
            kvn, nmask = to_dense_batch(h, batch, batch_size=B)
            kv_parts.append(self.kv_node(kvn))
            kpad_parts.append(~nmask)
        if self.kv_tok is not None:
            kvt = lm_hidden
            if func_token_lines is not None:
                tpad = func_token_lines < 0
                allp = tpad.all(dim=1)
                if allp.any():
                    tpad = tpad.clone(); tpad[allp, 0] = False
            else:
                tpad = torch.zeros(*lm_hidden.shape[:2], dtype=torch.bool, device=h.device)
            kv_parts.append(self.kv_tok(kvt))
            kpad_parts.append(tpad)
        kv = torch.cat(kv_parts, dim=1)                                 # [B,Σ,A]
        kpad = torch.cat(kpad_parts, dim=1)
        loc_ref, _ = self.attn_loc(q, kv, kv,
                                   key_padding_mask=kpad, need_weights=False)  # [B,Sm,A]
        stmt_cond = self.loc_out(loc_ref)[stmt_mask]                    # [S, ld]

        # ── cls direction: fused queries the graph's statements → per-graph ──
        q_c = self.q_fused(fused).unsqueeze(1)                          # [B,1,A]
        kv_s = self.kv_stmt(stmt_dense)                                 # [B,Sm,A]
        empty = ~stmt_mask.any(dim=1)
        kpad_s = ~stmt_mask
        if empty.any():
            kpad_s = kpad_s.clone(); kpad_s[empty, 0] = False
        cls_ref, _ = self.attn_cls(q_c, kv_s, kv_s,
                                   key_padding_mask=kpad_s, need_weights=False)
        cross_c = self.to_cls(cls_ref.squeeze(1))                       # [B, fused_dim]

        if self._residual:
            fused_mod = fused + self.gate_cls * cross_c
            stmt_cond = self.gate_loc * stmt_cond
        else:
            fused_mod = cross_c
        return fused_mod, stmt_cond


class SelfAttnCrossTask(nn.Module):
    """EDAT-style — statements self-attend within their graph, query biased by
    the classification signal. Mirrors EDAT's line_level_encoder (lines attend
    each other) at per-statement granularity.

    loc direction: statements self-attend (query biased by fused) → per-statement.
    cls direction: statements self-attend (query biased by nothing extra), then
                   pool per graph → per-graph classification signal.
    """

    def __init__(self, fused_dim: int, hidden_dim: int, lm_dim: int,
                 mode: str, num_heads: int = 4, residual: bool = True):
        super().__init__()
        self._mode = mode
        self._residual = residual
        ld = loc_dim(mode, hidden_dim, lm_dim)
        self._ld = ld
        A = hidden_dim
        self.stmt_in  = nn.Linear(ld, A)
        self.bias_cls = nn.Linear(fused_dim, A)        # cls signal → query bias
        self.attn     = nn.MultiheadAttention(A, num_heads, batch_first=True)
        self.loc_out  = nn.Linear(A, ld)
        self.to_cls   = nn.Linear(A, fused_dim)
        self.gate_cls = nn.Parameter(torch.zeros(1))
        self.gate_loc = nn.Parameter(torch.zeros(1))

    def forward(self, fused, loc_feats, stmt_graph, h, batch, B,
                lm_hidden, func_token_lines):
        fused = fused.float()
        loc_feats = loc_feats.float()

        stmt_dense, stmt_mask = to_dense_batch(loc_feats, stmt_graph, batch_size=B)  # [B,Sm,ld]
        units = self.stmt_in(stmt_dense)                                # [B,Sm,A]
        kpad = ~stmt_mask
        empty = ~stmt_mask.any(dim=1)
        if empty.any():
            kpad = kpad.clone(); kpad[empty, 0] = False
        mf = stmt_mask.unsqueeze(-1).to(units.dtype)

        # loc direction: statements self-attend, query biased by cls
        q_l = units + self.bias_cls(fused.detach()).unsqueeze(1)
        ref_l, _ = self.attn(q_l, units, units, key_padding_mask=kpad, need_weights=False)
        stmt_cond = self.loc_out(ref_l)[stmt_mask]                      # [S, ld]

        # cls direction: statements self-attend (own query), pool per graph
        ref_c, _ = self.attn(units, units, units, key_padding_mask=kpad, need_weights=False)
        pooled = (ref_c * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)    # [B, A]
        cross_c = self.to_cls(pooled)                                   # [B, fused_dim]

        if self._residual:
            fused_mod = fused + self.gate_cls * cross_c
            stmt_cond = self.gate_loc * stmt_cond
        else:
            fused_mod = cross_c
        return fused_mod, stmt_cond


def build_cross_task(method: str, fused_dim: int, hidden_dim: int,
                     num_classes: int, lm_dim: int, localization_encoder: str,
                     num_heads: int = 4, mmoe_task_encoder: bool = False,
                     residual: bool = True,
                     mmoe_loc_transformer: bool = False) -> nn.Module | None:
    """Factory — returns the cross-task module for `method`, or None for 'none'."""
    if method == "none":
        return None
    if method == "cross_attention":
        return CrossTaskAttn(fused_dim, hidden_dim, lm_dim, localization_encoder,
                             num_heads, residual=residual)
    if method == "self_attention":
        return SelfAttnCrossTask(fused_dim, hidden_dim, lm_dim, localization_encoder,
                                 num_heads, residual=residual)
    if method == "mmoe":
        return MMOECrossTask(fused_dim, hidden_dim, lm_dim, localization_encoder,
                             task_encoder=mmoe_task_encoder, residual=residual,
                             loc_transformer=mmoe_loc_transformer)
    raise ValueError(
        "cross_task_method must be none|cross_attention|self_attention|mmoe, "
        f"got {method!r}"
    )
