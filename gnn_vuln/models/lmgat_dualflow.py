"""lmgat_dualflow.py — Arch10: Dual-flow GATv2 + live LM."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import GATEncoder
from gnn_vuln.models.heads import FuncHead, StmtHead

NODE_FEAT_DIM = 773
_ALPHA_MAX = 0.8
_ALPHA_MEAN = 0.6


class LMGATDualFlowVulnDetector(VulnDetectorBase):
    def __init__(self, pretrained_lm="microsoft/unixcoder-base", func_lm="",
                 in_channels=NODE_FEAT_DIM, hidden_dim=256, num_layers=4,
                 dropout=0.3, num_classes=11, num_heads=4, edge_dim=7,
                 add_self_loops=False, use_skip=False, matryoshka_dim=None,
                 func_chunk_size=0, func_chunk_stride=0,
                 localization_encoder="gnn", use_flash_attention=False, compile_lm=False,
                 use_grad_checkpoint=True):
        super().__init__()
        self._build_lm_branch(pretrained_lm, func_lm, matryoshka_dim, func_chunk_size, func_chunk_stride, use_flash_attention, compile_lm, use_grad_checkpoint)
        self._loc_enc = localization_encoder
        self.dropout = dropout

        # Stage 1: localization GNN
        self.loc_encoder = GATEncoder(in_channels, hidden_dim, num_layers, num_heads,
                                      dropout, edge_dim, add_self_loops, use_skip)
        self.loc_stmt_max  = nn.Linear(hidden_dim, 1)
        self.loc_stmt_mean = nn.Linear(hidden_dim, 1)

        # Stage 2: classification GNN (input: 773+1=774)
        self.cls_encoder = GATEncoder(in_channels + 1, hidden_dim, num_layers, num_heads,
                                      dropout, edge_dim, add_self_loops, use_skip)

        # Fusion: focal [hidden] + context [hidden] + lm [lm_dim]
        self.func_head = FuncHead(hidden_dim * 2 + self._lm_dim, hidden_dim * 2, num_classes, dropout)
        self.stmt_head = StmtHead(hidden_dim, lm_dim=self._lm_dim if localization_encoder in ("lm", "both") else 0, localization_encoder=localization_encoder)

    def _node_suspicion(self, h_loc: torch.Tensor) -> torch.Tensor:
        raw = _ALPHA_MAX * self.loc_stmt_max(h_loc) + _ALPHA_MEAN * self.loc_stmt_mean(h_loc)
        return torch.sigmoid(raw).squeeze(-1)  # [N]

    def forward(self, x, edge_index, batch, node_line=None, edge_attr=None,
                func_input_ids=None, func_attention_mask=None,
                func_token_lines=None):
        # Stage 1
        h_loc = self.loc_encoder(x, edge_index, edge_attr)
        s_i = self._node_suspicion(h_loc)  # [N]

        # Stage 2
        x_aug = torch.cat([x, s_i.unsqueeze(-1)], dim=-1)  # [N, 774]
        h_cls = self.cls_encoder(x_aug, edge_index, edge_attr)

        # Dual pooling
        s_w = s_i.unsqueeze(-1)
        focal_emb   = global_add_pool(h_cls * s_w, batch) / global_add_pool(s_w, batch).clamp(min=1e-6)
        context_emb = global_mean_pool(h_cls, batch)

        B = focal_emb.size(0)
        if self._loc_enc != "gnn":
            lm_emb, lm_hidden = self._lm_embed_full(func_input_ids, func_attention_mask, B, x.device)
        else:
            lm_emb = self._lm_embed(func_input_ids, func_attention_mask, B, x.device)
            lm_hidden = None

        z_combined = torch.cat([focal_emb, context_emb, lm_emb], dim=-1)
        logit = self.func_head(z_combined)

        stmt_scores = (
            self.stmt_head.score(h_loc, batch, node_line, lm_hidden, func_token_lines)
            if node_line is not None else None
        )
        return logit, stmt_scores, z_combined

    @classmethod
    def from_config(cls, cfg, in_channels, **kwargs):
        pretrained_lm = getattr(cfg.model, "pretrained_lm", "microsoft/unixcoder-base")
        func_lm = getattr(cfg.model, "func_lm", "") or pretrained_lm
        return cls(
            pretrained_lm=pretrained_lm, func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            add_self_loops=getattr(cfg.model, "add_self_loops", False),
            use_skip=getattr(cfg.model, "use_skip", False),
            matryoshka_dim=getattr(cfg.model, "matryoshka_dim", None),
            func_chunk_size=getattr(cfg.model, "func_chunk_size", 0),
            func_chunk_stride=getattr(cfg.model, "func_chunk_stride", 0),
            localization_encoder=getattr(cfg.model, "localization_encoder", "gnn"),
            use_flash_attention=getattr(cfg.train, "use_flash_attention", False),
            compile_lm=getattr(cfg.train, "compile_lm", False),
            use_grad_checkpoint=getattr(cfg.model, "use_grad_checkpoint", True),
        )
