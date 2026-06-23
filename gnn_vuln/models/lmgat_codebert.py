"""lmgat_codebert.py — Unified GATv2 + (optional) live LM.

live_lm modes:
  - none          : GNN only (replaces old lmgat). fused = h_graph. No LM forwards.
  - func          : func-level [CLS] (sliding window if func_chunk_size>0). Default.
  - func_and_line : func-level [CLS] for cls + per-line LM forward for localization
                    (EDAT-style line isolation). Reuses func_input_ids.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import to_dense_batch, dropout_edge, dropout_node, mask_feature
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.encoders import build_gnn_encoder
from gnn_vuln.models.heads import FuncHead, ThinFuncHead, LinearFuncHead, StmtHead
from gnn_vuln.models.cross_task import build_cross_task, statement_features, _LineLevelEncoder
from gnn_vuln.models._lm_utils import scatter_lines_to_tokens, _PERLINE_MAX_LINE
from gnn_vuln.models.supcon_head import SupConProjector

NODE_FEAT_DIM = 773

_VALID_LIVE_LM = ("none", "func", "func_and_line", "line")


class CNNReadout(nn.Module):
    """Multi-scale 1D CNN graph readout.

    Treats sorted node hiddens as a sequence, applies parallel Conv1d with
    kernel_sizes=(3,5,7), max-pools each over the node dimension, concatenates,
    then projects back to out_dim. Captures local node-order patterns that
    permutation-invariant mean/max pool cannot.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 kernel_sizes: tuple[int, ...] = (3, 5, 7),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.proj = nn.Linear(out_dim * len(kernel_sizes), out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    @torch.compiler.disable
    def forward(self, h: torch.Tensor, batch: torch.Tensor, B: int) -> torch.Tensor:
        x_dense, mask = to_dense_batch(h.float(), batch, batch_size=B)  # [B, N_max, D]
        x = x_dense.permute(0, 2, 1)                                    # [B, D, N_max]
        valid = mask.any(dim=1)                                          # [B]
        parts = []
        for conv in self.convs:
            c = F.relu(conv(x))                                          # [B, out_dim, N_max]
            c = c.masked_fill(~mask.unsqueeze(1), float("-inf"))
            c_max = c.max(dim=-1).values                                 # [B, out_dim]
            c_max[~valid] = 0.0
            parts.append(c_max)
        out = torch.cat(parts, dim=-1)                                   # [B, out_dim*K]
        out = self.drop(F.relu(self.proj(out)))
        return self.norm(out)


class LMGATCodeBERTVulnDetector(VulnDetectorBase):
    def __init__(self, pretrained_lm="microsoft/unixcoder-base", func_lm="",
                 in_channels=NODE_FEAT_DIM, hidden_dim=256, num_layers=4,
                 dropout=0.3, num_classes=11, num_heads=4, edge_dim=7,
                 add_self_loops=False, use_skip=False, gnn_block_style="resnet",
                 gnn_norm_type="batch", gnn_activation="relu",
                 gnn_use_ffn=False, gnn_ffn_expansion=2,
                 gnn_use_pe=False, gnn_pe_walk_length=16, gnn_pe_dim=28,
                 gnn_balanced_init=False, gnn_balanced_init_beta=2.0,
                 gnn_g_init=False, gnn_g_init_d=2.0,
                 gnn_moe_ffn=False, gnn_gmoe=False, gnn_edge_moe=False,
                 gnn_moe_experts=8, gnn_moe_experts_1hop=4,
                 gnn_moe_k=2, gnn_moe_coef=1e-2,
                 func_head_type="fat",
                 num_groups=16, mtl_use_group_cond=True, mtl_use_linear_heads=False,
                 imtl_mid_layer=1,
                 matryoshka_dim=None,
                 func_chunk_size=0, func_chunk_stride=0,
                 localization_encoder="gnn", use_flash_attention=False, compile_lm=False,
                 use_grad_checkpoint=True,
                 stmt_both_mode="concat", stmt_lm_alpha=0.5,
                 cross_task_method="none", graph_pool="mean", graph_pool_proj_dim=0, jknet_mode="concat",
                 jknet_readout="meanmax", jknet_loc=False,
                 mmoe_task_encoder=False, cross_task_residual=True,
                 mmoe_loc_transformer=False, live_lm="func",
                 gnn_model="gat", num_relations=7, num_bases=None,
                 codet5p_raw_encoder=False, codet5p_normalize_per_token=False,
                 normalize_gnn_output=False, freeze_func_lm=False,
                 graph_aug_drop_edge=0.0, graph_aug_drop_node=0.0,
                 graph_aug_mask_feature=0.0, graph_aug_mask_mode="col",
                 window_attn_pool=False,
                 window_attn_hidden=False,
                 window_center_weight=False,
                 cross_window_attn=False,
                 window_mixer=False,
                 window_mixer_max=6,
                 supcon_proj_dim=0,
                 supcon_proj_hidden=256,
                 supcon_proj_dropout=0.1,
                 mixup_alpha=0.0):
        super().__init__()
        self._normalize_gnn_output = normalize_gnn_output
        # Balanced-Mixup / Remix (Chou et al. 2020): manifold mixup on the pooled
        # graph embedding h_graph. Training-time only. The trainer reads _mixup_perm
        # and _mixup_lam to build the (imbalance-aware) two-target classification loss.
        self._mixup_alpha = float(mixup_alpha)
        self._mixup_perm: torch.Tensor | None = None
        self._mixup_lam: float | None = None
        # Structural graph augmentation (training-time only, resampled every forward pass).
        # drop_edge  — DropEdge (Rong et al. 2020): Bernoulli edge mask, M ~ Bern(1-p),
        #              edge_attr re-indexed in sync. Unbiased: E[neighbor aggregation] unchanged.
        # drop_node  — NodeDropping (graph aug survey): Bernoulli node mask; dropped nodes
        #              both lose their incident edges (via dropout_node's subgraph) AND have
        #              their feature rows zeroed, so they cannot leak into pooling.
        # mask_feature — FeatureMasking (graph aug survey): Bernoulli mask over feature
        #              columns ("col", masks whole channels — GraphCL-style) or entries ("all").
        self._aug_drop_edge = float(graph_aug_drop_edge)
        self._aug_drop_node = float(graph_aug_drop_node)
        self._aug_mask_feature = float(graph_aug_mask_feature)
        self._aug_mask_mode = graph_aug_mask_mode
        assert live_lm in _VALID_LIVE_LM, \
            f"live_lm must be one of {_VALID_LIVE_LM}, got {live_lm!r}"
        self._live_lm = live_lm
        # When no live LM, localization must be gnn-only (lm/both need LM hidden).
        # Cross-task methods with mode='gnn' work without LM (lm_hidden path gated by kv_tok).
        if live_lm == "none":
            assert localization_encoder == "gnn", (
                f"live_lm='none' requires localization_encoder='gnn', "
                f"got {localization_encoder!r}. Live LM hidden states are unavailable."
            )
            self._lm_dim = 0
        else:
            self._build_lm_branch(
                pretrained_lm, func_lm, matryoshka_dim,
                func_chunk_size, func_chunk_stride,
                use_flash_attention, compile_lm, use_grad_checkpoint,
                lm_per_line=(live_lm == "func_and_line"),
                codet5p_raw_encoder=codet5p_raw_encoder,
                codet5p_normalize_per_token=codet5p_normalize_per_token,
                freeze_func_lm=freeze_func_lm,
                window_attn_pool=window_attn_pool,
                window_attn_hidden=window_attn_hidden,
                window_center_weight=window_center_weight,
                cross_window_attn=cross_window_attn,
                window_mixer=window_mixer,
                window_mixer_max=window_mixer_max,
            )
        # Line-level transformer (live_lm=line): contextualizes per-line LM
        # embeddings across the function. Classification = meanmax pool of its
        # output; localization = its per-line output. No whole-function forward.
        self.line_encoder = (
            _LineLevelEncoder(self._lm_dim, self._lm_dim, num_layers=2, num_heads=num_heads)
            if live_lm == "line" else None
        )
        self._loc_enc = localization_encoder
        self.encoder = build_gnn_encoder(
            gnn_model, in_channels, hidden_dim, num_layers, dropout,
            num_heads=num_heads, edge_dim=edge_dim, add_self_loops=add_self_loops,
            use_skip=use_skip, num_relations=num_relations, num_bases=num_bases,
            block_style=gnn_block_style,
            norm_type=gnn_norm_type, activation=gnn_activation,
            use_ffn=gnn_use_ffn, ffn_expansion=gnn_ffn_expansion,
            use_pe=gnn_use_pe, pe_walk_length=gnn_pe_walk_length, pe_dim=gnn_pe_dim,
            balanced_init=gnn_balanced_init, balanced_init_beta=gnn_balanced_init_beta,
            g_init=gnn_g_init, g_init_d=gnn_g_init_d,
            moe_ffn=gnn_moe_ffn, gmoe=gnn_gmoe, edge_moe=gnn_edge_moe,
            moe_experts=gnn_moe_experts, moe_experts_1hop=gnn_moe_experts_1hop,
            moe_k=gnn_moe_k, moe_coef=gnn_moe_coef,
        )
        # Graph-level pooling: mean | max | add | meanmax | meanmaxadd | attention | dualflow | cnn | jknet
        assert graph_pool in ("mean", "max", "add", "meanmax", "meanmaxadd", "attention", "dualflow", "cnn", "jknet"), \
            f"graph_pool must be mean|max|add|meanmax|meanmaxadd|attention|dualflow|cnn|jknet, got {graph_pool!r}"
        self._graph_pool = graph_pool
        self._jknet_mode = jknet_mode
        assert jknet_readout in ("meanmax", "max", "mean", "add"), \
            f"jknet_readout must be meanmax|max|mean|add, got {jknet_readout!r}"
        self._jknet_readout = jknet_readout
        self.attn_pool = (
            AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
            if graph_pool == "attention" else None
        )
        # dualflow: per-node suspicion head → focal (suspicion-weighted) + context (mean)
        self.node_susp = nn.Linear(hidden_dim, 1) if graph_pool == "dualflow" else None
        # cnn: multi-scale Conv1d readout over sorted node hiddens
        self.cnn_pool = (
            CNNReadout(hidden_dim, hidden_dim, kernel_sizes=(3, 5, 7), dropout=dropout)
            if graph_pool == "cnn" else None
        )
        # Thin head only for in-path MMOE (residual off + mmoe): MMOE's
        # task encoder + shared experts do the adaptation → head can be thin.
        # Attention methods don't carry that adaptation depth → keep fat head.
        # func_head_type override: "fat" (MLP) | "thin" (LN+Linear) | "linear" (GNN+ style)
        # | "mtl" (hierarchical: group_head + group-conditioned cwe_head + binary_head).
        assert func_head_type in ("fat", "thin", "linear", "mtl", "imtl", "imtl_cwe"), \
            f"func_head_type must be fat|thin|linear|mtl|imtl|imtl_cwe, got {func_head_type!r}"
        # Pool output dim: meanmaxadd concats mean+max+add → 3× hidden_dim.
        # jknet concats L layer node hiddens then pools → num_layers × hidden_dim.
        # Others keep hidden_dim (mean / meanmax score-level / attention / dualflow / cnn).
        if graph_pool == "meanmaxadd":
            self._pool_out_dim = 3 * hidden_dim
        elif graph_pool == "jknet":
            # concat → [N, L*hidden]; max → element-wise max across layers → [N, hidden]
            self._pool_out_dim = hidden_dim if jknet_mode == "max" else num_layers * hidden_dim
        else:
            self._pool_out_dim = hidden_dim
        # Optional projection on the graph-pool output to rebalance the GNN:LM ratio
        # in the fused vector (e.g. jknet 4×hidden drowning the 768D LM 4:1). Projected
        # before fusion so func_head / cross_task / supcon all see the balanced dim.
        if graph_pool_proj_dim and graph_pool_proj_dim > 0:
            self.graph_proj = nn.Linear(self._pool_out_dim, graph_pool_proj_dim)
            self._pool_out_dim = graph_pool_proj_dim
        else:
            self.graph_proj = None
        _fused_dim = self._pool_out_dim + self._lm_dim
        self._mtl = func_head_type == "mtl"
        self._imtl = func_head_type == "imtl"
        self._imtl_cwe = func_head_type == "imtl_cwe"
        self._imtl_mid_layer = imtl_mid_layer
        if func_head_type == "imtl":
            from gnn_vuln.models.heads import IntermediateMTLHeads
            self.func_head = IntermediateMTLHeads(hidden_dim, num_classes, num_groups, dropout)
        elif func_head_type == "mtl":
            # Hierarchical group→CWE: cwe_head conditioned on softmax(group_logits).
            # Sidesteps 26-class tail few-shot — groups are well-populated.
            if mtl_use_linear_heads:
                from gnn_vuln.models.heads import LinearMTLHeads
                self.func_head = LinearMTLHeads(_fused_dim, num_classes, num_groups,
                                                dropout, use_group_cond=mtl_use_group_cond)
            else:
                from gnn_vuln.models.heads import MTLHeads
                self.func_head = MTLHeads(_fused_dim, hidden_dim, num_classes, num_groups,
                                          dropout, use_group_cond=mtl_use_group_cond)
        elif func_head_type in ("linear", "imtl_cwe"):
            self.func_head = LinearFuncHead(_fused_dim, num_classes, dropout=dropout)
        elif func_head_type == "thin" or (cross_task_method == "mmoe" and not cross_task_residual):
            self.func_head = ThinFuncHead(_fused_dim, num_classes)
        else:
            self.func_head = FuncHead(_fused_dim, hidden_dim, num_classes, dropout)
        lm_dim = self._lm_dim if localization_encoder in ("lm", "both") else 0
        # jknet_loc: feed the JK-concat node vector [N, L*hidden] to localization (multi-scale
        # per-line), not just the last-layer node vector. Only meaningful for jknet concat mode.
        self._jknet_loc = bool(jknet_loc) and graph_pool == "jknet" and jknet_mode != "max"
        loc_gnn_dim = num_layers * hidden_dim if self._jknet_loc else hidden_dim
        self.stmt_head = StmtHead(loc_gnn_dim, lm_dim=lm_dim, localization_encoder=localization_encoder,
                                  both_mode=stmt_both_mode, lm_alpha=stmt_lm_alpha)
        self._cross_task_method = cross_task_method
        self.cross_task = build_cross_task(
            cross_task_method, self._pool_out_dim + self._lm_dim, hidden_dim, num_classes,
            self._lm_dim, localization_encoder, num_heads,
            mmoe_task_encoder=mmoe_task_encoder, residual=cross_task_residual,
            mmoe_loc_transformer=mmoe_loc_transformer,
        )
        _fused_dim_total = self._pool_out_dim + self._lm_dim
        self.supcon_head = (
            SupConProjector(_fused_dim_total, supcon_proj_hidden, supcon_proj_dim,
                            dropout=supcon_proj_dropout)
            if supcon_proj_dim > 0 else None
        )
        self._fused_for_supcon: torch.Tensor | None = None
        # Inference-only capture of the function-level representation fed to the
        # classification head (the vector right before the output head), for drift
        # detection / similarity search. Populated only in eval mode; read when
        # forward(return_repr=True). Training is untouched.
        self._cls_repr: torch.Tensor | None = None
        self.func_head.register_forward_pre_hook(
            lambda _m, _inp: None if _m.training else setattr(self, "_cls_repr", _inp[0].detach())
        )

    def _maybe_mixup(self, h_graph: torch.Tensor) -> torch.Tensor:
        """Manifold mixup on the pooled graph embedding (Balanced-Mixup / Remix).
        Training-time only: h~ = lam*h + (1-lam)*h[perm], lam ~ Beta(a,a). Stores
        perm + lam so the trainer can build the (imbalance-aware) two-target loss.
        Returns h_graph unchanged (and clears state) when disabled / eval."""
        if not (self.training and self._mixup_alpha > 0.0):
            self._mixup_perm = None
            self._mixup_lam = None
            return h_graph
        lam = float(torch.distributions.Beta(self._mixup_alpha, self._mixup_alpha).sample())
        perm = torch.randperm(h_graph.size(0), device=h_graph.device)
        self._mixup_perm = perm
        self._mixup_lam = lam
        return lam * h_graph + (1.0 - lam) * h_graph[perm]

    def forward(self, *args, return_repr: bool = False, **kwargs):
        """Public entry. With return_repr=True (inference only) the function-level
        pre-head representation (vector fed to the classification head) is appended
        as the last tuple element, for drift detection / similarity search. Default
        (return_repr=False) returns exactly as before — training path unchanged."""
        out = self._forward_impl(*args, **kwargs)
        if return_repr:
            tup = out if isinstance(out, tuple) else (out,)
            return (*tup, self._cls_repr)
        return out

    def _forward_impl(self, x, edge_index, batch, node_line=None, edge_attr=None,
                func_input_ids=None, func_attention_mask=None,
                func_token_lines=None,
                func_line_cls=None, func_line_ids=None, func_line_cls_batch=None,
                rwse=None):
        # Structural graph augmentation — training only, fresh Bernoulli sample per
        # forward pass (stricter than DropEdge's "once per epoch", strictly unbiased).
        if self.training:
            if self._aug_drop_edge > 0.0:
                edge_index, _edge_mask = dropout_edge(edge_index, p=self._aug_drop_edge, training=True)
                if edge_attr is not None:
                    edge_attr = edge_attr[_edge_mask]
            if self._aug_drop_node > 0.0:
                edge_index, _edge_mask, _node_mask = dropout_node(
                    edge_index, p=self._aug_drop_node, num_nodes=x.size(0), training=True)
                if edge_attr is not None:
                    edge_attr = edge_attr[_edge_mask]
                x = x * _node_mask.unsqueeze(-1).to(x.dtype)
            if self._aug_mask_feature > 0.0:
                x, _ = mask_feature(x, p=self._aug_mask_feature, mode=self._aug_mask_mode, training=True)
        h = self.encoder(x, edge_index, edge_attr, batch=batch, rwse=rwse)
        # MoE load-balance aux loss (0 if no MoE). Read by trainer, added to total.
        self.moe_aux_loss = getattr(self.encoder, "_moe_aux_loss", None)
        h_jk = None   # JK-concat node vector, set in jknet branch; reused by jknet_loc
        if self._graph_pool == "attention":
            h_graph = self.attn_pool(h, batch)
        elif self._graph_pool == "meanmax":
            h_graph = 0.8 * global_max_pool(h, batch) + 0.6 * global_mean_pool(h, batch)
        elif self._graph_pool == "meanmaxadd":
            h_graph = torch.cat([
                global_mean_pool(h, batch),
                global_max_pool(h, batch),
                global_add_pool(h, batch),
            ], dim=-1)
        elif self._graph_pool == "max":
            h_graph = global_max_pool(h, batch)
        elif self._graph_pool == "add":
            h_graph = global_add_pool(h, batch)
        elif self._graph_pool == "dualflow":
            # focal: per-node suspicion-weighted pool + context: mean pool
            s = torch.sigmoid(self.node_susp(h))                      # [N, 1]
            focal = global_add_pool(h * s, batch) / global_add_pool(s, batch).clamp(min=1e-6)
            h_graph = focal + global_mean_pool(h, batch)
        elif self._graph_pool == "cnn":
            B_hint = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            h_graph = self.cnn_pool(h, batch, B_hint)
        elif self._graph_pool == "jknet":
            # JK-Net layer aggregation (Xu et al. 2018): concat → [N, L*hidden_dim],
            # or element-wise max across layers → [N, hidden_dim]. Then meanmax graph pool.
            layer_hiddens = getattr(self.encoder, "_layer_hiddens", [])
            if not layer_hiddens:
                h_jk = h
            elif self._jknet_mode == "max":
                h_jk = torch.stack(layer_hiddens, dim=0).amax(dim=0)
            else:
                h_jk = torch.cat(layer_hiddens, dim=-1)
            if self._jknet_readout == "max":
                h_graph = global_max_pool(h_jk, batch)
            elif self._jknet_readout == "mean":
                h_graph = global_mean_pool(h_jk, batch)
            elif self._jknet_readout == "add":
                h_graph = global_add_pool(h_jk, batch)
            else:
                h_graph = 0.8 * global_max_pool(h_jk, batch) + 0.6 * global_mean_pool(h_jk, batch)
        else:
            h_graph = global_mean_pool(h, batch)
        if self.graph_proj is not None:
            h_graph = self.graph_proj(h_graph)
        B = h_graph.size(0)
        # Per-node GNN features for localization (optionally unit-normed, symmetric to F6 per_token norm).
        # jknet_loc → use the JK-concat node vector [N, L*hidden] (multi-scale per line) instead of
        # the last-layer node vector h.
        node_src = h_jk if (self._jknet_loc and h_jk is not None) else h
        h_loc = F.normalize(node_src, dim=-1) if self._normalize_gnn_output else node_src
        # Intermediate-MTL: pool the mid-layer node hiddens for group head.
        # Group gradient only flows through layers 0..mid_layer; CWE gradient through all.
        h_mid_graph = None
        if self._imtl or self._imtl_cwe:
            _layers = getattr(self.encoder, "_layer_hiddens", [])
            _mid_idx = self._imtl_mid_layer
            _h_mid = _layers[_mid_idx] if _mid_idx < len(_layers) else h
            h_mid_graph = 0.8 * global_max_pool(_h_mid, batch) + 0.6 * global_mean_pool(_h_mid, batch)
        # ── LM branch ─────────────────────────────────────────────────────────
        if self._live_lm == "none":
            # GNN-only: fused = h_graph. Skip all LM forwards.
            ct = self._cross_task_method
            if ct == "none" or node_line is None:
                stmt_scores = (
                    self.stmt_head.score(h_loc, batch, node_line)
                    if node_line is not None else None
                )
                # MTL hierarchical head: group_head + group-conditioned cwe_head +
                # binary_head. Returns (cwe, group, binary, stmt) 4-tuple — trainer
                # adds group/binary CE via group_loss_weight/binary_loss_weight.
                if self._mtl:
                    logit_cwe, logit_group, logit_binary = self.func_head(h_graph)
                    return logit_cwe, logit_group, logit_binary, stmt_scores
                if self._imtl:
                    logit_cwe, logit_group, logit_binary = self.func_head(h_mid_graph, h_graph)
                    return logit_cwe, logit_group, logit_binary, stmt_scores
                if self._imtl_cwe:
                    logit = self.func_head(h_mid_graph)
                    return logit, stmt_scores
                h_cls = self._maybe_mixup(h_graph)
                logit = self.func_head(h_cls)
                # SupCon projection on the graph embedding (GNN-only fused = h_graph).
                if self.supcon_head is not None:
                    self._fused_for_supcon = h_graph
                    proj_z = self.supcon_head(h_graph)
                    return logit, stmt_scores, proj_z
                return logit, stmt_scores
            # Cross-task with GNN-only path. statement_features + cross_task with
            # mode='gnn' work without LM (kv_tok=None, lm_hidden=None skipped).
            loc_feats, stmt_graph, _ = statement_features(
                h_loc, batch, node_line, None, None, self._loc_enc,
            )
            fused_mod, stmt_cond = self.cross_task(
                h_graph, loc_feats.detach(), stmt_graph, h_loc, batch, B,
                None, None,
            )
            logit = self.func_head(fused_mod)
            stmt_scores = self.stmt_head.score(h_loc, batch, node_line, cond=stmt_cond)
            return logit, stmt_scores

        if self._live_lm == "line":
            # Hierarchical: per-line LM forward → line transformer (cross-line
            # context). Classification = meanmax pool; localization = per-line.
            # No whole-function forward — function length is unbounded.
            # Fast path: use precomputed per-line CLS from dataset cache when available
            # (set by precompute_line_cls=True + freeze_func_lm=True). DataLoader
            # follow_batch creates func_line_cls_batch [total_lines] = graph index.
            if func_line_cls is not None and func_line_cls_batch is not None:
                line_cls   = func_line_cls.to(x.device)        # [total_lines, lm_dim]
                line_graph = func_line_cls_batch.to(x.device)  # [total_lines]
                uniq_sid   = line_graph * _PERLINE_MAX_LINE + func_line_ids.to(x.device)
            else:
                line_cls, uniq_sid, _, _ = self._lm_embed_per_line_raw(
                    func_input_ids, func_token_lines,
                )
                line_graph = (uniq_sid // _PERLINE_MAX_LINE).long()
            line_ctx = self.line_encoder(line_cls, line_graph, B)        # [n, lm_dim]
            lm_emb = (0.8 * global_max_pool(line_ctx, line_graph, size=B)
                      + 0.6 * global_mean_pool(line_ctx, line_graph, size=B))
            lm_hidden = scatter_lines_to_tokens(
                line_ctx, uniq_sid, func_token_lines, B, func_input_ids.size(1),
            )
        elif self._loc_enc != "gnn":
            lm_emb, lm_hidden = self._lm_embed_full(
                func_input_ids, func_attention_mask, B, x.device, func_token_lines,
            )
        else:
            lm_emb = self._lm_embed(func_input_ids, func_attention_mask, B, x.device)
            lm_hidden = None
        if self._normalize_gnn_output:
            h_graph = F.normalize(h_graph, dim=-1)
        fused = torch.cat([h_graph, lm_emb], dim=-1)
        self._fused_for_supcon = fused
        proj_z = self.supcon_head(fused) if self.supcon_head is not None else None

        ct = self._cross_task_method
        if ct == "none" or node_line is None:
            logit = self.func_head(fused)
            stmt_scores = (
                self.stmt_head.score(h_loc, batch, node_line, lm_hidden, func_token_lines)
                if node_line is not None else None
            )
            if proj_z is not None:
                return logit, stmt_scores, proj_z
            return logit, stmt_scores

        # cross_attention | self_attention | mmoe — per-statement loc conditioning.
        # statement_features uses the SAME sid formula as StmtHead → cond [S,
        # loc_dim] aligns directly with StmtHead's statements.
        loc_feats, stmt_graph, _ = statement_features(
            h_loc, batch, node_line, lm_hidden, func_token_lines, self._loc_enc,
        )
        fused_mod, stmt_cond = self.cross_task(
            fused, loc_feats.detach(), stmt_graph, h_loc, batch, B,
            lm_hidden, func_token_lines,
        )
        logit = self.func_head(fused_mod)
        stmt_scores = self.stmt_head.score(h_loc, batch, node_line, lm_hidden, func_token_lines, cond=stmt_cond)
        if proj_z is not None:
            return logit, stmt_scores, proj_z
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
            add_self_loops=getattr(cfg.model, "add_self_loops", False),
            use_skip=getattr(cfg.model, "use_skip", False),
            gnn_block_style=getattr(cfg.model, "gnn_block_style", "resnet"),
            gnn_norm_type=getattr(cfg.model, "gnn_norm_type", "batch"),
            gnn_activation=getattr(cfg.model, "gnn_activation", "relu"),
            gnn_use_ffn=getattr(cfg.model, "gnn_use_ffn", False),
            gnn_ffn_expansion=getattr(cfg.model, "gnn_ffn_expansion", 2),
            gnn_use_pe=getattr(cfg.model, "gnn_use_pe", False),
            gnn_pe_walk_length=getattr(cfg.model, "gnn_pe_walk_length", 16),
            gnn_pe_dim=getattr(cfg.model, "gnn_pe_dim", 28),
            gnn_balanced_init=getattr(cfg.model, "gnn_balanced_init", False),
            gnn_balanced_init_beta=getattr(cfg.model, "gnn_balanced_init_beta", 2.0),
            gnn_g_init=getattr(cfg.model, "gnn_g_init", False),
            gnn_g_init_d=getattr(cfg.model, "gnn_g_init_d", 2.0),
            gnn_moe_ffn=getattr(cfg.model, "gnn_moe_ffn", False),
            gnn_gmoe=getattr(cfg.model, "gnn_gmoe", False),
            gnn_edge_moe=getattr(cfg.model, "gnn_edge_moe", False),
            gnn_moe_experts=getattr(cfg.model, "gnn_moe_experts", 8),
            gnn_moe_experts_1hop=getattr(cfg.model, "gnn_moe_experts_1hop", 4),
            gnn_moe_k=getattr(cfg.model, "gnn_moe_k", 2),
            gnn_moe_coef=getattr(cfg.model, "gnn_moe_coef", 1e-2),
            func_head_type=getattr(cfg.model, "func_head_type", "fat"),
            num_groups=getattr(cfg.model, "num_groups", 16),
            mtl_use_group_cond=getattr(cfg.model, "mtl_use_group_cond", True),
            mtl_use_linear_heads=getattr(cfg.model, "mtl_use_linear_heads", False),
            imtl_mid_layer=getattr(cfg.model, "imtl_mid_layer", 1),
            matryoshka_dim=getattr(cfg.model, "matryoshka_dim", None),
            func_chunk_size=getattr(cfg.model, "func_chunk_size", 0),
            func_chunk_stride=getattr(cfg.model, "func_chunk_stride", 0),
            localization_encoder=getattr(cfg.model, "localization_encoder", "gnn"),
            use_flash_attention=getattr(cfg.train, "use_flash_attention", False),
            compile_lm=getattr(cfg.train, "compile_lm", False),
            use_grad_checkpoint=getattr(cfg.model, "use_grad_checkpoint", True),
            stmt_both_mode=getattr(cfg.model, "stmt_both_mode", "concat"),
            stmt_lm_alpha=getattr(cfg.model, "stmt_lm_alpha", 0.5),
            cross_task_method=getattr(cfg.model, "cross_task_method", "none"),
            mmoe_task_encoder=getattr(cfg.model, "mmoe_task_encoder", False),
            cross_task_residual=getattr(cfg.model, "cross_task_residual", True),
            graph_pool=getattr(cfg.model, "graph_pool", "mean"),
            graph_pool_proj_dim=getattr(cfg.model, "graph_pool_proj_dim", 0),
            jknet_mode=getattr(cfg.model, "jknet_mode", "concat"),
            jknet_readout=getattr(cfg.model, "jknet_readout", "meanmax"),
            jknet_loc=getattr(cfg.model, "jknet_loc", False),
            mmoe_loc_transformer=getattr(cfg.model, "mmoe_loc_transformer", False),
            live_lm=getattr(cfg.model, "live_lm", "func"),
            gnn_model=getattr(cfg.model, "gnn_model", "gat"),
            num_relations=getattr(cfg.model, "num_relations", 7),
            num_bases=getattr(cfg.model, "num_bases", None),
            codet5p_raw_encoder=getattr(cfg.model, "codet5p_raw_encoder", False),
            codet5p_normalize_per_token=getattr(cfg.model, "codet5p_normalize_per_token", False),
            normalize_gnn_output=getattr(cfg.model, "normalize_gnn_output", False),
            freeze_func_lm=getattr(cfg.model, "freeze_func_lm", False),
            graph_aug_drop_edge=getattr(cfg.model, "graph_aug_drop_edge", 0.0),
            graph_aug_drop_node=getattr(cfg.model, "graph_aug_drop_node", 0.0),
            graph_aug_mask_feature=getattr(cfg.model, "graph_aug_mask_feature", 0.0),
            graph_aug_mask_mode=getattr(cfg.model, "graph_aug_mask_mode", "col"),
            window_attn_pool=getattr(cfg.model, "window_attn_pool", False),
            window_attn_hidden=getattr(cfg.model, "window_attn_hidden", False),
            window_center_weight=getattr(cfg.model, "window_center_weight", False),
            cross_window_attn=getattr(cfg.model, "cross_window_attn", False),
            window_mixer=getattr(cfg.model, "window_mixer", False),
            window_mixer_max=getattr(cfg.model, "window_mixer_max", 6),
            supcon_proj_dim=getattr(cfg.model, "supcon_proj_dim", 0),
            supcon_proj_hidden=getattr(cfg.model, "supcon_proj_hidden", 256),
            supcon_proj_dropout=getattr(cfg.model, "supcon_proj_dropout", 0.1),
            mixup_alpha=getattr(cfg.train, "mixup_alpha", 0.0),
        )
