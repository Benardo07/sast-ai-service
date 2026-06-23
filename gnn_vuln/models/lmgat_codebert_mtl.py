"""lmgat_codebert_mtl.py — Arch11: GATv2Conv + live LM + 3 MTL heads."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.heads import StmtHead, MTLHeads
from gnn_vuln.models._lm_utils import lm_hidden_dim, lm_pool
from transformers import AutoConfig, AutoModel

NODE_FEAT_DIM = 773
EDGE_FEAT_DIM = 7
_EDGE_GROUP_MAP = torch.tensor([0, 1, 1, 2, 2, 0, 2], dtype=torch.long)
_NUM_EDGE_GROUPS = 3


class LMGATCodeBERTMTLVulnDetector(VulnDetectorBase):
    def __init__(
        self,
        pretrained_lm="microsoft/codebert-base",
        func_lm="",
        in_channels=NODE_FEAT_DIM,
        hidden_dim=256,
        num_layers=4,
        dropout=0.3,
        num_classes=11,
        num_groups=16,
        num_heads=4,
        edge_dim=EDGE_FEAT_DIM,
        use_group_cond=True,
        add_self_loops=False,
        use_skip=False,
        use_edge_emb=False,
        edge_emb_dim=32,
        edge_coarse_dim=16,
        matryoshka_dim=None,
        func_chunk_size=0,
        func_chunk_stride=0,
        localization_encoder="gnn",
        use_flash_attention=False,
        compile_lm=False,
        use_grad_checkpoint=True,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip
        self.use_edge_emb = use_edge_emb

        self._loc_enc = localization_encoder

        # Live LM
        self._build_lm_branch(pretrained_lm, func_lm, matryoshka_dim,
                               func_chunk_size, func_chunk_stride, use_flash_attention, compile_lm, use_grad_checkpoint)

        # Edge embeddings
        if use_edge_emb:
            self.edge_fine_emb   = nn.Embedding(EDGE_FEAT_DIM, edge_emb_dim)
            self.edge_coarse_emb = nn.Embedding(_NUM_EDGE_GROUPS, edge_coarse_dim)
            self.register_buffer("_edge_group_map", _EDGE_GROUP_MAP)
            _gat_edge_dim = edge_emb_dim + edge_coarse_dim
        else:
            _gat_edge_dim = edge_dim

        # GATv2 encoder (manual — needs edge_emb preprocessing in _encode)
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_dim, heads=num_heads, concat=False,
                                    dropout=dropout, edge_dim=_gat_edge_dim, add_self_loops=add_self_loops))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                                        dropout=dropout, edge_dim=_gat_edge_dim, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.res_projs = nn.ModuleList([nn.Linear(in_channels, hidden_dim, bias=False)])
            for _ in range(num_layers - 1):
                self.res_projs.append(nn.Identity())

        fused_dim = hidden_dim + self._lm_dim
        self.mtl_heads = MTLHeads(fused_dim, hidden_dim, num_classes, num_groups, dropout, use_group_cond)
        self.stmt_head  = StmtHead(hidden_dim, lm_dim=self._lm_dim if localization_encoder in ("lm", "both") else 0, localization_encoder=localization_encoder)

    def _encode(self, x, edge_index, edge_attr):
        if self.use_edge_emb and edge_attr is not None and edge_attr.shape[0] > 0:
            type_idx  = edge_attr.argmax(dim=-1)
            group_idx = self._edge_group_map[type_idx]
            edge_attr = torch.cat([self.edge_coarse_emb(group_idx), self.edge_fine_emb(type_idx)], dim=-1)
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x + residual) if residual is not None else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, batch, node_line=None, edge_attr=None,
                func_input_ids=None, func_attention_mask=None,
                func_token_lines=None):
        h = self._encode(x, edge_index, edge_attr)
        h_graph = global_mean_pool(h, batch)
        if self._loc_enc != "gnn":
            lm_emb, lm_hidden = self._lm_embed_full(func_input_ids, func_attention_mask, h_graph.size(0), x.device)
        else:
            lm_emb = self._lm_embed(func_input_ids, func_attention_mask, h_graph.size(0), x.device)
            lm_hidden = None
        fused   = torch.cat([h_graph, lm_emb], dim=-1)

        logit_cwe, logit_group, logit_binary = self.mtl_heads(fused)
        stmt_scores = (
            self.stmt_head.score(h, batch, node_line, lm_hidden, func_token_lines)
            if node_line is not None else None
        )
        return logit_cwe, logit_group, logit_binary, stmt_scores, fused

    @classmethod
    def from_config(cls, cfg, in_channels, **kwargs):
        pretrained_lm = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
        func_lm = getattr(cfg.model, "func_lm", "") or pretrained_lm
        active_heads = kwargs.get("active_heads", frozenset())
        use_group_cond = (
            ("group" in active_heads and "cwe" in active_heads) if active_heads else True
        ) and getattr(cfg.model, "use_group_cond", True)
        return cls(
            pretrained_lm=pretrained_lm, func_lm=func_lm,
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_groups=getattr(cfg.model, "num_groups", 16),
            num_heads=cfg.model.heads,
            edge_dim=getattr(cfg.model, "edge_dim", 7),
            use_group_cond=use_group_cond,
            add_self_loops=getattr(cfg.model, "add_self_loops", False),
            use_skip=getattr(cfg.model, "use_skip", False),
            use_edge_emb=getattr(cfg.model, "use_edge_emb", False),
            edge_emb_dim=getattr(cfg.model, "edge_emb_dim", 32),
            edge_coarse_dim=getattr(cfg.model, "edge_coarse_dim", 16),
            matryoshka_dim=getattr(cfg.model, "matryoshka_dim", None),
            func_chunk_size=getattr(cfg.model, "func_chunk_size", 0),
            func_chunk_stride=getattr(cfg.model, "func_chunk_stride", 0),
            localization_encoder=getattr(cfg.model, "localization_encoder", "gnn"),
            use_flash_attention=getattr(cfg.train, "use_flash_attention", False),
            compile_lm=getattr(cfg.train, "compile_lm", False),
            use_grad_checkpoint=getattr(cfg.model, "use_grad_checkpoint", True),
        )
