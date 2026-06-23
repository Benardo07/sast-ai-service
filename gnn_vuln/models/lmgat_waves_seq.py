"""lmgat_waves_seq.py — Arch8: Transformer stmt localiser + GATv2 classifier + live LM."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import GATEncoder
from gnn_vuln.models.heads import FuncHead

NODE_FEAT_DIM = 773
CODEBERT_DIM  = 768  # slice [1:769] of node features


class LMGATWavesSeqVulnDetector(VulnDetectorBase):
    def __init__(
        self,
        pretrained_lm="microsoft/unixcoder-base",
        func_lm="",
        in_channels=NODE_FEAT_DIM,
        hidden_dim=256,
        num_layers=4,
        dropout=0.3,
        num_classes=11,
        num_heads=4,
        edge_dim=7,
        stmt_transformer_layers=2,
        stmt_transformer_heads=4,
        add_self_loops=False,
        use_skip=False,
        matryoshka_dim=None,
        func_chunk_size=0,
        func_chunk_stride=0,
        use_flash_attention=False,
        compile_lm=False,
        use_grad_checkpoint=True,
    ):
        super().__init__()
        self._build_lm_branch(pretrained_lm, func_lm, matryoshka_dim, func_chunk_size, func_chunk_stride, use_flash_attention, compile_lm, use_grad_checkpoint)
        self.dropout = dropout

        # Stage 1: transformer-based statement encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=CODEBERT_DIM, nhead=stmt_transformer_heads,
            dropout=dropout, batch_first=True,
        )
        self.stmt_transformer  = nn.TransformerEncoder(enc_layer, num_layers=stmt_transformer_layers, enable_nested_tensor=False)
        self.stmt_score_head   = nn.Linear(CODEBERT_DIM, 1)

        # Stage 2: GNN on x + suspicion (774D)
        self.gnn_encoder = GATEncoder(in_channels + 1, hidden_dim, num_layers, num_heads, dropout, edge_dim, add_self_loops, use_skip)

        self.func_head = FuncHead(hidden_dim + self._lm_dim, hidden_dim, num_classes, dropout)

    def _stage1(self, x, batch, node_line):
        """Transformer over per-statement CodeBERT embeddings -> node suspicion + stmt scores."""
        device = x.device
        B = int(batch.max().item()) + 1
        node_susp = torch.zeros(x.size(0), device=device)
        stmt_scores_list = []

        cb_feats = x[:, 1:769]  # [N, 768] CodeBERT slice

        for b in range(B):
            mask  = batch == b
            h_b   = cb_feats[mask]  # [n_b, 768]
            lines = node_line[mask] if node_line is not None else torch.full((h_b.size(0),), -1, device=device)
            valid = lines >= 0
            if not valid.any():
                stmt_scores_list.append(torch.zeros(0, device=device))
                continue

            h_v, l_v = h_b[valid], lines[valid]
            unique_lines = l_v.unique(sorted=True)
            stmt_feats = torch.stack([h_v[l_v == line].mean(dim=0) for line in unique_lines])  # [S, 768]
            out = self.stmt_transformer(stmt_feats.unsqueeze(0)).squeeze(0)  # [S, 768]
            scores = self.stmt_score_head(out).squeeze(-1)  # [S]
            stmt_scores_list.append(scores)

            # Assign suspicion to nodes by line
            for i, line in enumerate(unique_lines):
                node_susp[mask.nonzero(as_tuple=True)[0][l_v == line]] = torch.sigmoid(scores[i]).detach()

        return node_susp, stmt_scores_list

    def forward(self, x, edge_index, batch, node_line=None, edge_attr=None,
                func_input_ids=None, func_attention_mask=None,
                func_token_lines=None):
        node_susp, stmt_scores = self._stage1(x, batch, node_line)

        x_aug  = torch.cat([x, node_susp.unsqueeze(-1)], dim=-1)
        h      = self.gnn_encoder(x_aug, edge_index, edge_attr)
        h_graph = global_mean_pool(h, batch)

        B      = h_graph.size(0)
        lm_emb = self._lm_embed(func_input_ids, func_attention_mask, B, x.device)

        logit = self.func_head(torch.cat([h_graph, lm_emb], dim=-1))
        return logit, stmt_scores

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
            stmt_transformer_layers=getattr(cfg.model, "stmt_transformer_layers", 2),
            stmt_transformer_heads=getattr(cfg.model, "stmt_transformer_heads", 4),
            add_self_loops=getattr(cfg.model, "add_self_loops", False),
            use_skip=getattr(cfg.model, "use_skip", False),
            matryoshka_dim=getattr(cfg.model, "matryoshka_dim", None),
            func_chunk_size=getattr(cfg.model, "func_chunk_size", 0),
            func_chunk_stride=getattr(cfg.model, "func_chunk_stride", 0),
            use_flash_attention=getattr(cfg.train, "use_flash_attention", False),
            compile_lm=getattr(cfg.train, "compile_lm", False),
            use_grad_checkpoint=getattr(cfg.model, "use_grad_checkpoint", True),
        )
