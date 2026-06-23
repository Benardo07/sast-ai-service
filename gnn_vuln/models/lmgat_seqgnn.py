"""lmgat_seqgnn.py — Sequential two-stage GNN (localize → classify), GNN-only.

Two N48-style GNN encoders run in sequence:
  stage-1 (loc_encoder)  → per-node suspicion s_i + per-statement scores (localization)
  stage-2 (cls_encoder)  → classification on suspicion-augmented nodes [x, s_i]

The localization prediction is fed forward into classification — "find the
vulnerable line, then categorize it". Unlike LOSVER (separate localization and
classification models), both stages train JOINTLY: the standard (logit,
stmt_scores) return drives classification CE + MIL + rank localization loss in
one optimizer step. Both encoders follow N48 (jknet pool + gnn_plus block).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import build_gnn_encoder
from gnn_vuln.models.heads import FuncHead, LinearFuncHead, StmtHead

NODE_FEAT_DIM = 773
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGATSeqGNNVulnDetector(VulnDetectorBase):
    def __init__(self, in_channels=NODE_FEAT_DIM, hidden_dim=256, num_layers=4,
                 dropout=0.3, num_classes=26, num_heads=4, edge_dim=31,
                 add_self_loops=False, use_skip=True,
                 gnn_block_style="gnn_plus", gnn_norm_type="batch", gnn_activation="elu",
                 gnn_use_ffn=True, gnn_ffn_expansion=2,
                 gnn_model="gat", num_relations=7, num_bases=None,
                 jknet_mode="concat", jknet_readout="meanmax", jknet_loc=True,
                 func_head_type="linear",
                 seq_stage2_input="raw",        # "raw" → [x, s_i] | "loc" → [h_loc, s_i]
                 seq_susp_pool=False,           # stage-2 pool weighted by suspicion
                 seq_susp_pool_k=4.0,           # suspicious node weight = 1 + k·s_i (k=4 → ~5×)
                 seq_detach_susp=False):        # detach s_i feeding stage-2 (decouple stages)
        super().__init__()
        self._live_lm = "none"
        self._lm_dim = 0
        self._jknet_mode = jknet_mode
        self._jknet_readout = jknet_readout
        self._jknet_loc = bool(jknet_loc) and jknet_mode != "max"
        self._stage2_input = seq_stage2_input
        self._susp_pool = bool(seq_susp_pool)
        self._susp_k = float(seq_susp_pool_k)
        self._detach_susp = bool(seq_detach_susp)

        def _enc(inch: int) -> nn.Module:
            return build_gnn_encoder(
                gnn_model, inch, hidden_dim, num_layers, dropout,
                num_heads=num_heads, edge_dim=edge_dim, add_self_loops=add_self_loops,
                use_skip=use_skip, num_relations=num_relations, num_bases=num_bases,
                block_style=gnn_block_style, norm_type=gnn_norm_type, activation=gnn_activation,
                use_ffn=gnn_use_ffn, ffn_expansion=gnn_ffn_expansion,
            )

        # Stage 1 — localization
        self.loc_encoder = _enc(in_channels)
        loc_node_dim = num_layers * hidden_dim if self._jknet_loc else hidden_dim
        self.loc_stmt_max  = nn.Linear(loc_node_dim, 1)
        self.loc_stmt_mean = nn.Linear(loc_node_dim, 1)
        self.stmt_head = StmtHead(loc_node_dim, lm_dim=0, localization_encoder="gnn")

        # Stage 2 — classification on suspicion-augmented nodes
        base_dim = loc_node_dim if seq_stage2_input == "loc" else in_channels
        self.cls_encoder = _enc(base_dim + 1)        # +1 for suspicion channel
        self._pool_out_dim = hidden_dim if jknet_mode == "max" else num_layers * hidden_dim
        if func_head_type == "linear":
            self.func_head = LinearFuncHead(self._pool_out_dim, num_classes, dropout=dropout)
        else:
            self.func_head = FuncHead(self._pool_out_dim, hidden_dim, num_classes, dropout)
        # Inference-only capture of the pre-head function representation (h_graph),
        # for drift detection / similarity search. Read when forward(return_repr=True).
        self._cls_repr: torch.Tensor | None = None
        self.func_head.register_forward_pre_hook(
            lambda _m, _inp: None if _m.training else setattr(self, "_cls_repr", _inp[0].detach())
        )

    # ── helpers ────────────────────────────────────────────────────────────────
    def _jknet_node(self, encoder: nn.Module, h: torch.Tensor) -> torch.Tensor:
        layers = getattr(encoder, "_layer_hiddens", [])
        if not layers:
            return h
        if self._jknet_mode == "max":
            return torch.stack(layers, dim=0).amax(dim=0)
        return torch.cat(layers, dim=-1)

    def _jknet_pool(self, h_jk: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r = self._jknet_readout
        if r == "max":
            return global_max_pool(h_jk, batch)
        if r == "mean":
            return global_mean_pool(h_jk, batch)
        if r == "add":
            return global_add_pool(h_jk, batch)
        return _ALPHA_MAX * global_max_pool(h_jk, batch) + _ALPHA_MEAN * global_mean_pool(h_jk, batch)

    def _node_suspicion(self, h_loc: torch.Tensor) -> torch.Tensor:
        raw = _ALPHA_MAX * self.loc_stmt_max(h_loc) + _ALPHA_MEAN * self.loc_stmt_mean(h_loc)
        return torch.sigmoid(raw).squeeze(-1)        # [N]

    # ── forward ──────────────────────────────────────────────────────────────────
    def forward(self, *args, return_repr: bool = False, **kwargs):
        """Public entry. return_repr=True (inference only) appends the pre-head
        function representation (h_graph) as the last tuple element, for drift
        detection / similarity search. Training (default) returns as before."""
        out = self._forward_impl(*args, **kwargs)
        if return_repr:
            tup = out if isinstance(out, tuple) else (out,)
            return (*tup, self._cls_repr)
        return out

    def _forward_impl(self, x, edge_index, batch, node_line=None, edge_attr=None, **kwargs):
        # Stage 1: localize
        h1 = self.loc_encoder(x, edge_index, edge_attr, batch=batch)
        h_loc = self._jknet_node(self.loc_encoder, h1) if self._jknet_loc else h1
        s_i = self._node_suspicion(h_loc)            # [N]
        s_feed = s_i.detach() if self._detach_susp else s_i

        # Stage 2: classify on suspicion-augmented nodes
        base = h_loc if self._stage2_input == "loc" else x
        x_aug = torch.cat([base, s_feed.unsqueeze(-1)], dim=-1)
        h2 = self.cls_encoder(x_aug, edge_index, edge_attr, batch=batch)
        h_jk2 = self._jknet_node(self.cls_encoder, h2)

        if self._susp_pool:
            w = (1.0 + self._susp_k * s_i).unsqueeze(-1)   # suspicious ≈ (1+k)× a clean node
            h_graph = global_add_pool(h_jk2 * w, batch) / global_add_pool(w, batch).clamp(min=1e-6)
        else:
            h_graph = self._jknet_pool(h_jk2, batch)

        logit = self.func_head(h_graph)
        stmt_scores = (
            self.stmt_head.score(h_loc, batch, node_line)
            if node_line is not None else None
        )
        return logit, stmt_scores

    @classmethod
    def from_config(cls, cfg, in_channels, **kwargs):
        return cls(
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            add_self_loops=getattr(cfg.model, "add_self_loops", False),
            use_skip=getattr(cfg.model, "use_skip", False),
            gnn_block_style=getattr(cfg.model, "gnn_block_style", "resnet"),
            gnn_norm_type=getattr(cfg.model, "gnn_norm_type", "batch"),
            gnn_activation=getattr(cfg.model, "gnn_activation", "relu"),
            gnn_use_ffn=getattr(cfg.model, "gnn_use_ffn", False),
            gnn_ffn_expansion=getattr(cfg.model, "gnn_ffn_expansion", 2),
            gnn_model=getattr(cfg.model, "gnn_model", "gat"),
            num_relations=getattr(cfg.model, "num_relations", 7),
            num_bases=getattr(cfg.model, "num_bases", None),
            jknet_mode=getattr(cfg.model, "jknet_mode", "concat"),
            jknet_readout=getattr(cfg.model, "jknet_readout", "meanmax"),
            jknet_loc=getattr(cfg.model, "jknet_loc", True),
            func_head_type=getattr(cfg.model, "func_head_type", "linear"),
            seq_stage2_input=getattr(cfg.model, "seq_stage2_input", "raw"),
            seq_susp_pool=getattr(cfg.model, "seq_susp_pool", False),
            seq_susp_pool_k=getattr(cfg.model, "seq_susp_pool_k", 4.0),
            seq_detach_susp=getattr(cfg.model, "seq_detach_susp", False),
        )
