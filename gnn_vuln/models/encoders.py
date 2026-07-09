"""Reusable GNN encoder blocks shared across architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, GCNConv, GINEConv, GatedGraphConv, RGCNConv, ResGatedGraphConv,
)
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import degree


def _compute_rwse(edge_index: torch.Tensor, num_nodes: int, walk_length: int = 16) -> torch.Tensor:
    """Random Walk Structural Encoding (Dwivedi 2022, used in GNN+ 2025).
    Returns [N, walk_length] tensor: diagonal of (D^-1 A)^k for k=1..walk_length.
    Each row = probability of returning to start node after k steps.
    """
    device = edge_index.device
    row, col = edge_index[0], edge_index[1]
    deg_inv = 1.0 / degree(row, num_nodes=num_nodes).clamp(min=1.0)
    # Build sparse D^-1 A as edge weights (CSR for efficient mm)
    edge_w = deg_inv[row]
    adj = torch.sparse_coo_tensor(edge_index, edge_w, size=(num_nodes, num_nodes), device=device).coalesce()
    pe = torch.zeros(num_nodes, walk_length, device=device)
    # Iteratively: M_k = (D^-1 A)^k. Track diagonal each step.
    M_k = adj
    for k in range(walk_length):
        # diag(M_k) — sum over output index for entries where row == col
        idx = M_k.indices()
        vals = M_k.values()
        self_mask = idx[0] == idx[1]
        if self_mask.any():
            pe[idx[0][self_mask], k] = vals[self_mask]
        if k < walk_length - 1:
            M_k = torch.sparse.mm(M_k, adj).coalesce()
    return pe


def _build_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    """Build per-layer normalization. 'batch' (default) or 'graph' (Cai 2021 ICML)."""
    if norm_type == "graph":
        return GraphNorm(hidden_dim)
    if norm_type == "batch":
        return nn.BatchNorm1d(hidden_dim)
    raise ValueError(f"norm_type must be 'batch' or 'graph', got {norm_type!r}")


def _activation(act: str) -> callable:
    """Pick activation. 'relu' (default) or 'elu' (original GAT 2018)."""
    if act == "relu":
        return F.relu
    if act == "elu":
        return F.elu
    raise ValueError(f"activation must be 'relu' or 'elu', got {act!r}")


@torch.no_grad()
def apply_balanced_init(layers, beta: float = 2.0) -> None:
    """Mustafa et al. NeurIPS 2023 'Are GATs Out of Balance?' — Procedure 2.6.

    Approximated BalO for GATv2Conv layers:
      1. Zero attention vector a^l for all layers l (Xav+ZeroAtt)
      2. Apply orthogonal init to lin_l / lin_r weights (LL-Ortho base)
      3. Scale first layer row norms to sqrt(beta) (default beta=2.0)
      4. Balance inter-layer per-neuron norms (simplified: skip for multi-head GAT
         due to shape complexity — Xav+Bal approximation still effective)

    Paper showed BalO → 80.2% Cora vs Xavier 39.3% at L=10.
    """
    if not layers:
        return
    for conv in layers:
        # Zero ALL attention parameters (handles GATConv 'att_src/dst' and GATv2Conv 'att')
        for attr in ("att", "att_src", "att_dst", "att_l", "att_r"):
            if hasattr(conv, attr):
                p = getattr(conv, attr)
                if isinstance(p, nn.Parameter):
                    p.zero_()
        # Orthogonal init for linear weights (LL-Ortho base)
        for lin_name in ("lin", "lin_l", "lin_r", "lin_src", "lin_dst", "lin_edge"):
            if hasattr(conv, lin_name):
                lin = getattr(conv, lin_name)
                if lin is not None and hasattr(lin, "weight") and lin.weight is not None:
                    nn.init.orthogonal_(lin.weight)
                    if hasattr(lin, "bias") and lin.bias is not None:
                        lin.bias.zero_()
    # Step 3: Scale first layer row norms to sqrt(beta)
    first = layers[0]
    for lin_name in ("lin_l", "lin_r"):
        if hasattr(first, lin_name):
            lin = getattr(first, lin_name)
            if lin is None or not hasattr(lin, "weight"):
                continue
            W = lin.weight  # [out, in]
            norms = W.norm(dim=1, keepdim=True).clamp(min=1e-6)
            W.copy_(W / norms * (beta ** 0.5))

@torch.no_grad()
def apply_g_init(layers, d_i: float = 2.0) -> None:
    """G-Init (Kelesis et al. 2024, Applied Intelligence) — generalizes Kaiming to GNNs.

    Paper formula (Section 3, eq for sigma):
      sigma = sqrt(2 * d_i / n_l)
      W ~ N(0, sigma^2)

    Where:
      d_i = fixed hyperparameter (default 2.0 — paper uses 2.0 for most datasets,
            1.6 for ogbn-arxiv). NOT computed from real graph degrees.
      n_l = layer dimensionality (paper assumes square W of size n_l x n_l for GCN).
            For non-square GAT linear layers, we use fan_in (PyTorch convention).

    Paper SCOPE:
      - Applies ONLY to weight matrices W (linear projections).
      - NOT to bias, NOT to attention vectors. Paper tested on GCN only.
      - For GAT we apply to lin_l, lin_r, lin_edge (the W-equivalents).
      - Attention vectors a_l, a_r keep default PyG Xavier init.

    Effect: Kaiming with sqrt(2) larger std (sqrt(4/n_l) vs sqrt(2/n_l)).
    Larger maximum singular values → resists oversmoothing at depth.
    Reference: arxiv:2410.23830.
    """
    if not layers:
        return
    # Apply only to weight matrices W (per paper).
    # Skip attention vectors — paper does not extend G-Init to them.
    for conv in layers:
        for lin_name in ("lin", "lin_l", "lin_r", "lin_src", "lin_dst", "lin_edge"):
            if hasattr(conv, lin_name):
                lin = getattr(conv, lin_name)
                if lin is not None and hasattr(lin, "weight") and lin.weight is not None:
                    fan_in = lin.weight.size(-1)
                    sigma = (2.0 * d_i / fan_in) ** 0.5
                    nn.init.normal_(lin.weight, mean=0.0, std=sigma)
                    # Paper: only W initialized. Leave bias default (zero/whatever PyG sets).


@torch.no_grad()
def apply_lsuv_encoder(encoder, sample_batch, tol: float = 0.1,
                       max_trials: int = 10, verbose: bool = False) -> dict:
    """LSUV (Mishkin & Matas, ICLR 2016, arxiv:1511.06422) — Algorithm 1.

    Two steps:
      1. Pre-init: orthonormal init for all nn.Linear weights in encoder.
      2. Sequential variance normalization. For each Linear layer L:
         while |Var(out_L) - 1.0| >= tol and trials < max_trials:
             forward pass with sample batch
             measure Var(out_L)
             W_L = W_L / sqrt(Var(out_L))

    Args:
      encoder: GATEncoder (or any nn.Module exposing forward(x, edge_index, edge_attr=, batch=, rwse=))
      sample_batch: a PyG Batch with .x, .edge_index, .edge_attr, .batch (and .rwse if PE)
      tol: variance tolerance, paper says 0.01-0.1 works in broad range
      max_trials: cap on rescale iterations per layer

    Returns:
      dict {layer_name: final_variance} for logging.
    """
    # Step 1: orthonormal init for all Linear layers in encoder (paper: "Pre-initialize
    # network with orthonormal matrices as in Saxe et al. (2014)"). PyTorch's orthogonal_
    # handles both square and rectangular weights via QR/SVD.
    for m in encoder.modules():
        if isinstance(m, nn.Linear) and m.weight is not None:
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Collect target Linear layers in encoder
    linear_modules = [(n, m) for n, m in encoder.named_modules() if isinstance(m, nn.Linear)]
    if not linear_modules:
        return {}

    # Setup forward hooks to capture per-layer outputs
    activations = {}
    hooks = []
    for name, layer in linear_modules:
        def make_hook(lname):
            def hook(mod, inp, out):
                # out may be tuple from PyG layers; take tensor
                t = out if isinstance(out, torch.Tensor) else out[0]
                activations[lname] = t.detach()
            return hook
        h = layer.register_forward_hook(make_hook(name))
        hooks.append(h)

    # Build forward args from sample batch
    x = sample_batch.x
    edge_index = sample_batch.edge_index
    edge_attr = getattr(sample_batch, "edge_attr", None)
    b = getattr(sample_batch, "batch", None)
    rwse = getattr(sample_batch, "rwse", None) if getattr(encoder, "use_pe", False) else None

    encoder.eval()
    final_vars = {}
    try:
        # Step 2: per-layer variance scaling (sequential in forward order)
        for name, layer in linear_modules:
            for _trial in range(max_trials):
                activations.clear()
                _ = encoder(x, edge_index, edge_attr, batch=b, rwse=rwse)
                if name not in activations:
                    break
                var = activations[name].var().item()
                if not (var > 0):
                    break
                if abs(var - 1.0) < tol:
                    final_vars[name] = var
                    break
                layer.weight.data.div_((var ** 0.5))
                final_vars[name] = var
    finally:
        for h in hooks:
            h.remove()
        encoder.train()

    return final_vars


# ── Mixture-of-Experts helpers (Shazeer 2017 / GMoE Wang NeurIPS 2023) ─────────

def _cv_squared(x: torch.Tensor) -> torch.Tensor:
    """Squared coefficient of variation = var/mean². Load-balance loss term —
    pushes expert usage toward uniform. 0 for single expert. (Shazeer 2017)."""
    eps = 1e-10
    if x.numel() <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean() ** 2 + eps)


def _prob_in_top_k(clean_values, noisy_values, noise_std, noisy_top_values, k):
    """Differentiable P(expert in top-k) under the gating noise (Shazeer 2017,
    moe.py:_prob_in_top_k). Lets the `load` term backprop into gate params.
    noisy_top_values: top-(k+1) noisy logits [N, k+1]."""
    device = clean_values.device
    batch = clean_values.size(0)
    m = noisy_top_values.size(1)
    top_flat = noisy_top_values.flatten()
    thresh_pos_if_in = torch.arange(batch, device=device) * m + k
    thresh_if_in = torch.gather(top_flat, 0, thresh_pos_if_in).unsqueeze(1)
    is_in = noisy_values > thresh_if_in
    thresh_pos_if_out = thresh_pos_if_in - 1
    thresh_if_out = torch.gather(top_flat, 0, thresh_pos_if_out).unsqueeze(1)
    normal = torch.distributions.Normal(
        torch.zeros((), device=device), torch.ones((), device=device))
    prob_if_in = normal.cdf((clean_values - thresh_if_in) / noise_std)
    prob_if_out = normal.cdf((clean_values - thresh_if_out) / noise_std)
    return torch.where(is_in, prob_if_in, prob_if_out)


def _noisy_top_k_gating(x, w_gate, w_noise, k, training, softplus, noise_eps=1e-2):
    """Noisy top-k gating (Shazeer 2017, moe.py:noisy_top_k_gating). Per-node gate.

    Returns (gates [N, n_experts], load [n_experts]).
      gates — sparse: top-k softmax, rest 0.
      load  — differentiable expected #tokens per expert (prob_in_top_k) during
              noisy training; else hard count (gates>0).sum(0).
    importance = gates.sum(0) computed by caller. Both feed load-balance loss.
    """
    n_experts = w_gate.size(1)
    clean_logits = x @ w_gate
    if training:
        raw_noise = x @ w_noise
        noise_std = softplus(raw_noise) + noise_eps
        noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        logits = noisy_logits
    else:
        logits = clean_logits
    top_logits, top_idx = logits.topk(min(k + 1, n_experts), dim=1)
    top_k_logits = top_logits[:, :k]
    top_k_idx = top_idx[:, :k]
    top_k_gates = torch.softmax(top_k_logits, dim=1)
    gates = torch.zeros_like(logits).scatter(1, top_k_idx, top_k_gates)
    if training and k < n_experts:
        load = _prob_in_top_k(clean_logits, noisy_logits, noise_std, top_logits, k).sum(0)
    else:
        load = (gates > 0).sum(0).float()
    return gates, load


class NodeMoEFFN(nn.Module):
    """Switch-style per-node MoE FFN (Fedus 2021 / Shazeer 2017), drop-in for the
    GNN+ FFN sub-block. N expert FFNs; each node routed to top-k via noisy gate.
    Dense compute (all experts run), sparse gating (non-top-k weight = 0).

    Expert FFN matches GNN+ _ff_block inner: Linear(D→eD) → Act → Drop → Linear(eD→D).
    Returns (out [N, D], load_balance_loss scalar).
    """

    def __init__(self, hidden_dim, ffn_expansion, num_experts, k, dropout,
                 act_fn, coef=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.dropout = dropout
        self.act_fn = act_fn
        self.coef = coef
        eD = hidden_dim * ffn_expansion
        self.experts = nn.ModuleList([
            nn.ModuleList([nn.Linear(hidden_dim, eD), nn.Linear(eD, hidden_dim)])
            for _ in range(num_experts)
        ])
        self.w_gate = nn.Parameter(torch.zeros(hidden_dim, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(hidden_dim, num_experts))
        self.softplus = nn.Softplus()

    def forward(self, x):
        gates, load = _noisy_top_k_gating(x, self.w_gate, self.w_noise, self.k,
                                          self.training, self.softplus)
        importance = gates.sum(0)
        lb_loss = (_cv_squared(importance) + _cv_squared(load)) * self.coef
        outs = []
        for w1, w2 in self.experts:
            h = self.act_fn(w1(x))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = w2(h)
            outs.append(h)
        stacked = torch.stack(outs, dim=1)                  # [N, n_experts, D]
        out = (gates.unsqueeze(-1) * stacked).sum(dim=1)    # weighted sum
        return out, lb_loss


def _two_hop_edge_index(edge_index, num_nodes, cap_factor=4):
    """Compute 2-hop edge_index via sparse A@A (GMoE Wang 2023). Removes self-loops
    and 1-hop edges so hop-2 experts see only genuinely distant neighbors.

    GMoE was tuned on small sparse molecule graphs (~25 nodes). CPG graphs (mean
    334 nodes, dense) explode the 2-hop set (~N·deg²) → OOM even at 48GB. We CAP
    the 2-hop edges to cap_factor × (#1-hop edges), keeping the highest A@A values
    (= most length-2 paths between i,j → strongest 2-hop connections). Deterministic,
    bounds memory regardless of graph density.

    Returns edge_index_2hop [2, E2] (no edge_attr — hop-2 GAT runs edge-attr-free)."""
    device = edge_index.device
    E1 = edge_index.size(1)
    val = torch.ones(E1, device=device)
    A = torch.sparse_coo_tensor(edge_index, val, (num_nodes, num_nodes)).coalesce()
    A2 = torch.sparse.mm(A, A).coalesce()
    idx2 = A2.indices()
    vals2 = A2.values()
    # Remove self-loops
    mask = idx2[0] != idx2[1]
    idx2, vals2 = idx2[:, mask], vals2[mask]
    # Remove pairs already connected at 1-hop
    one_hop_keys = edge_index[0] * num_nodes + edge_index[1]
    two_hop_keys = idx2[0] * num_nodes + idx2[1]
    keep = ~torch.isin(two_hop_keys, one_hop_keys)
    idx2, vals2 = idx2[:, keep], vals2[keep]
    # Cap: keep top (cap_factor × E1) edges by A@A path-count → bounds memory.
    max_e2 = cap_factor * E1
    if idx2.size(1) > max_e2:
        topk_idx = torch.topk(vals2, max_e2, sorted=False).indices
        idx2 = idx2[:, topk_idx]
    return idx2


class GMoEConv(nn.Module):
    """Graph Mixture-of-Experts conv (Wang et al. NeurIPS 2023, arxiv:2304.02806).

    N GATv2 experts: first `num_experts_1hop` aggregate 1-hop neighbors, the rest
    aggregate 2-hop neighbors. Per-node noisy top-k gate selects experts → each
    node adaptively picks its receptive field. Dense compute, sparse gates.

    Hop-2 experts run edge-attr-free (2-hop edges have no single CPG edge type).
    Returns (out [N, D], load_balance_loss).
    """

    def __init__(self, in_dim, out_dim, num_heads, num_experts, num_experts_1hop,
                 k, dropout, edge_dim, add_self_loops, fill_value, coef=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_1hop = num_experts_1hop
        self.k = min(k, num_experts)
        self.coef = coef
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            # 1-hop experts use edge features; 2-hop experts edge-attr-free.
            ed = edge_dim if i < num_experts_1hop else None
            self.experts.append(GATv2Conv(
                in_dim, out_dim, heads=num_heads, concat=False, dropout=dropout,
                edge_dim=ed, add_self_loops=add_self_loops, fill_value=fill_value,
            ))
        self.bns = nn.ModuleList([nn.BatchNorm1d(out_dim) for _ in range(num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(in_dim, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(in_dim, num_experts))
        self.softplus = nn.Softplus()

    def forward(self, x, edge_index, edge_attr, edge_index_2hop):
        gates, load = _noisy_top_k_gating(x, self.w_gate, self.w_noise, self.k,
                                          self.training, self.softplus)
        importance = gates.sum(0)
        lb_loss = (_cv_squared(importance) + _cv_squared(load)) * self.coef
        outs = []
        for i, (expert, bn) in enumerate(zip(self.experts, self.bns)):
            if i < self.num_experts_1hop:
                h = expert(x, edge_index, edge_attr=edge_attr)
            else:
                h = expert(x, edge_index_2hop)
            outs.append(bn(h))
        stacked = torch.stack(outs, dim=1)                  # [N, n_experts, D]
        # Weighted SUM per GMoE paper eq 2 (σ(Σ_o G_o·E_o)) + Shazeer 2017 + Switch.
        # gates are top-k softmax (sum to 1) → convex combination of selected experts.
        # NOTE: GMoE reference CODE uses .mean(dim=1) (÷num_experts) — an impl quirk
        # present since its first commit, contradicting its own paper eq 2 and the
        # davidmrau base it cites. Post-BN the two are equivalent (1/n scale cancels),
        # so we use .sum to match the published equation + standard MoE.
        out = (gates.unsqueeze(-1) * stacked).sum(dim=1)
        return out, lb_loss


# Maps each of the 30 CPG edge-type indices (gnn_vuln.data.cpg.constants.EDGE_TYPES
# order) to one of 5 semantic super-relation groups. Used by EdgeTypeMoEConv so each
# expert aggregates one family of CPG relations.
#   0 = syntax     (AST, CONTAINS, REF, CONDITION, *_BODY, FOR_*, EVAL_TYPE, CATCH/TRY)
#   1 = control    (CFG, CDG, DOMINATE, POST_DOMINATE)
#   2 = dataflow   (REACHING_DEF/DDG, ARGUMENT, PARAMETER_LINK)
#   3 = call       (CALL, RECEIVER, IS_CALL_FOR_IMPORT)
#   4 = misc/type  (ALIAS_OF, BINDS, CAPTURE, IMPORTS, INHERITS_FROM, SOURCE_FILE, TAGGED_BY)
# Index order matches EDGE_TYPES: ALIAS_OF, ARGUMENT, AST, BINDS, CALL, CAPTURE,
# CATCH_BODY, CDG, CFG, CONDITION, CONTAINS, DOMINATE, EVAL_TYPE, FALSE_BODY,
# FINALLY_BODY, FOR_BODY, FOR_INIT, FOR_UPDATE, IMPORTS, INHERITS_FROM,
# IS_CALL_FOR_IMPORT, PARAMETER_LINK, POST_DOMINATE, REACHING_DEF, RECEIVER, REF,
# SOURCE_FILE, TAGGED_BY, TRUE_BODY, TRY_BODY.
_CPG_EDGE_GROUP_MAP = [
    4, 2, 0, 4, 3, 4, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 4,
    3, 2, 1, 2, 3, 0, 4, 4, 0, 0,
]
NUM_EDGE_GROUPS = 5


class EdgeTypeMoEConv(nn.Module):
    """Edge-type Mixture-of-Experts conv — GMoE per-node gating (Wang 2023) over
    RGCN/HGT-style relation-specific experts.

    Instead of hop-based experts (which explode via A@A on dense CPGs), each expert
    aggregates ONE family of CPG edge types (syntax / control / dataflow / call /
    misc). Edges are PARTITIONED by type → Σ expert edges = total edges (no growth,
    no 2-hop). Memory ≈ a single GATv2. Per-node noisy top-k gate lets each node
    route to the relations relevant to its vulnerability (e.g. taint-sink → dataflow,
    branch → control).

    edge_attr layout (gnn_vuln.data.cpg.features._edge_attr): first 30 dims = edge-type
    one-hot, dim 30 = has_var flag → type index = edge_attr[:, :30].argmax(1).

    Returns (out [N, D], load_balance_loss).
    """

    def __init__(self, in_dim, out_dim, num_heads, k, dropout, edge_dim,
                 add_self_loops, fill_value, num_groups=NUM_EDGE_GROUPS, coef=1e-2):
        super().__init__()
        self.num_experts = num_groups
        self.k = min(k, num_groups)
        self.coef = coef
        # One GATv2 expert per relation group; each uses the (subset's) edge features.
        self.experts = nn.ModuleList([
            GATv2Conv(in_dim, out_dim, heads=num_heads, concat=False, dropout=dropout,
                      edge_dim=edge_dim, add_self_loops=add_self_loops, fill_value=fill_value)
            for _ in range(num_groups)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(out_dim) for _ in range(num_groups)])
        self.w_gate = nn.Parameter(torch.zeros(in_dim, num_groups))
        self.w_noise = nn.Parameter(torch.zeros(in_dim, num_groups))
        self.softplus = nn.Softplus()
        # group_map [30] long buffer (idx → group). Moves with .to(device).
        self.register_buffer("group_map", torch.tensor(_CPG_EDGE_GROUP_MAP, dtype=torch.long))

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        gates, load = _noisy_top_k_gating(x, self.w_gate, self.w_noise, self.k,
                                          self.training, self.softplus)
        importance = gates.sum(0)
        lb_loss = (_cv_squared(importance) + _cv_squared(load)) * self.coef
        # Edge type idx → group id (per edge).
        etype = edge_attr[:, :len(_CPG_EDGE_GROUP_MAP)].argmax(dim=1)   # [E]
        egroup = self.group_map[etype]                                  # [E] in 0..num_groups-1
        outs = []
        for g, (expert, bn) in enumerate(zip(self.experts, self.bns)):
            emask = egroup == g
            if emask.any():
                ei_g = edge_index[:, emask]
                ea_g = edge_attr[emask]
                h = expert(x, ei_g, edge_attr=ea_g)
                h = bn(h)
            else:
                # No edges of this relation in the batch → expert contributes zeros.
                h = x.new_zeros(N, bn.num_features)
            outs.append(h)
        stacked = torch.stack(outs, dim=1)                              # [N, num_groups, D]
        out = (gates.unsqueeze(-1) * stacked).sum(dim=1)                # weighted sum (Shazeer/GMoE eq2)
        return out, lb_loss


# CPG edge types: AST, CFG, CDG, DDG, PDG, CALL, REACHING_DEF
NUM_EDGE_TYPES = 7


# ── Shared residual projection helper ─────────────────────────────────────────

def _build_res_projs(
    in_channels: int, hidden_dim: int, num_layers: int
) -> nn.ModuleList:
    """Residual projections: Linear for layer 0, Identity for rest."""
    projs = nn.ModuleList()
    projs.append(nn.Linear(in_channels, hidden_dim, bias=False))
    for _ in range(num_layers - 1):
        projs.append(nn.Identity())
    return projs


# ── Conv factory — lets the GNN+ recipe wrap any classic backbone ──────────────
# GNN+ (Luo ICML 2025) tested GCN / GIN / GatedGCN. We add them as drop-in convs so
# the SAME recipe (block_style, FFN, norm, ELU, skip) wraps each → fair vs GATv2.
_EDGE_AWARE_CONVS = {"gat", "gatedgcn", "gine"}   # take edge_attr; "gcn" is edge-agnostic


def _make_conv(conv_type, in_dim, out_dim, num_heads, dropout, edge_dim,
               add_self_loops, fill_value):
    """Build one message-passing layer of the given type, edge-feature aware where
    supported. All output out_dim (heads concat=False for GAT)."""
    if conv_type == "gat":
        return GATv2Conv(in_dim, out_dim, heads=num_heads, concat=False,
                         dropout=dropout, edge_dim=edge_dim,
                         add_self_loops=add_self_loops, fill_value=fill_value)
    if conv_type == "gatedgcn":
        # ResGatedGraphConv (Bresson & Laurent 2017) — edge-gated message passing,
        # GNN+'s overall-best backbone. Uses edge features via edge_dim.
        return ResGatedGraphConv(in_dim, out_dim, edge_dim=edge_dim)
    if conv_type == "gine":
        # GINE (Hu 2020) — GIN with edge features. MLP update per GIN.
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        return GINEConv(mlp, edge_dim=edge_dim)
    if conv_type == "gcn":
        # Edge-agnostic (ignores CPG edge types).
        return GCNConv(in_dim, out_dim, add_self_loops=add_self_loops)
    raise ValueError(f"conv_type must be gat|gatedgcn|gine|gcn, got {conv_type!r}")


# ── Faithful GNN+ GatedGCN (Luo ICML 2025 / Bresson & Laurent 2017) ───────────
# Ported 1:1 from github.com/LUOyk1999/GNNPlus GatedGCNLayer: 5 linears A-E,
# maintained+updated edge state e, normalized soft-gating, edge BN/residual, and
# self-contained block (BN→act→drop→residual + FFN) INSIDE the layer.

from torch_geometric.nn.conv import MessagePassing as _MP
from torch_geometric.utils import scatter as _pyg_scatter


class GatedGCNLayer(_MP):
    """GNN+ GatedGCN layer (faithful). Maintains node x AND edge e across layers.
    forward(x, e, edge_index) → (x, e). Self-contained: BN, act, dropout, residual,
    and FFN all internal (matches reference gatedgcn_layer.py exactly)."""

    def __init__(self, in_dim, out_dim, dropout, residual=True, ffn=True, act="relu"):
        super().__init__()
        a = _activation(act)
        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim, out_dim, bias=True)
        self.D = nn.Linear(in_dim, out_dim, bias=True)
        self.E = nn.Linear(in_dim, out_dim, bias=True)
        self.act_fn_x = a
        self.act_fn_e = a
        self.dropout = dropout
        self.residual = residual
        self.ffn = ffn
        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self._e = None
        if ffn:
            self.norm1_local = nn.BatchNorm1d(out_dim)
            self.ff_linear1 = nn.Linear(out_dim, out_dim * 2)
            self.ff_linear2 = nn.Linear(out_dim * 2, out_dim)
            self.act_fn_ff = a
            self.norm2 = nn.BatchNorm1d(out_dim)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, x, e, edge_index):
        x_in, e_in = x, e
        Ax, Bx, Dx, Ex = self.A(x), self.B(x), self.D(x), self.E(x)
        Ce = self.C(e)
        x, e = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, Ax=Ax)
        x = self.bn_node_x(x); e = self.bn_edge_e(e)
        x = self.act_fn_x(x);  e = self.act_fn_e(e)
        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        if self.residual:
            x = x_in + x; e = e_in + e
        if self.ffn:
            x = self.norm1_local(x)
            x = x + self._ff_block(x)
            x = self.norm2(x)
        return x, e

    def message(self, Dx_i, Ex_j, Ce):
        e_ij = Dx_i + Ex_j + Ce
        self._e = e_ij
        return torch.sigmoid(e_ij)

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        dim_size = Bx.shape[0]
        num = _pyg_scatter(sigma_ij * Bx_j, index, 0, dim_size, reduce="sum")
        den = _pyg_scatter(sigma_ij, index, 0, dim_size, reduce="sum")
        return num / (den + 1e-6)

    def update(self, aggr_out, Ax):
        x = Ax + aggr_out
        e_out = self._e
        self._e = None
        return x, e_out


class GatedGCNEncoder(nn.Module):
    """Stack of faithful GNN+ GatedGCN layers. Projects node + edge features to
    hidden once (GNNPreMP-style), then L self-contained GatedGCN layers maintaining
    (x, e). Returns final node embedding. Matches GNN+ CustomGNN(gatedgcn) recipe."""

    def __init__(self, in_channels, hidden_dim, num_layers, dropout, edge_dim,
                 residual=True, ffn=True, act="relu"):
        super().__init__()
        self.node_pre = nn.Linear(in_channels, hidden_dim)
        self.edge_pre = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GatedGCNLayer(hidden_dim, hidden_dim, dropout, residual=residual,
                          ffn=ffn, act=act)
            for _ in range(num_layers)
        ])
        # No MoE here; expose attribute for the model's aux-loss read (always 0).
        self.use_pe = False
        self._moe_aux_loss = torch.zeros(())

    def forward(self, x, edge_index, edge_attr=None, batch=None, rwse=None):
        x = self.node_pre(x)
        e = self.edge_pre(edge_attr) if edge_attr is not None else \
            x.new_zeros(edge_index.size(1), x.size(1))
        for layer in self.layers:
            x, e = layer(x, e, edge_index)
        self._moe_aux_loss = torch.zeros((), device=x.device)
        return x


# ── GAT Encoder ───────────────────────────────────────────────────────────────

class GATEncoder(nn.Module):
    """
    Stack of GATv2Conv layers with Norm + Activation + Dropout.
    Optional residual skip connections.

    block_style:
      - "resnet"   (legacy default): Conv → Norm → +residual → Act → Dropout
      - "gnn_plus" (Luo 2025 ICML SOTA): Conv → Norm → Act → Dropout → +residual
    norm_type:
      - "batch" (default) — BatchNorm1d
      - "graph" — GraphNorm (Cai 2021 ICML, per-graph normalization, needs batch index)
    activation:
      - "relu" (default) — ReLU
      - "elu" — ELU (original GAT 2018 activation)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        edge_dim: int = NUM_EDGE_TYPES,
        add_self_loops: bool = False,
        use_skip: bool = False,
        fill_value: float = 0.0,
        block_style: str = "resnet",
        norm_type: str = "batch",
        activation: str = "relu",
        use_ffn: bool = False,
        ffn_expansion: int = 2,
        use_pe: bool = False,
        pe_walk_length: int = 16,
        pe_dim: int = 28,
        balanced_init: bool = False,
        balanced_init_beta: float = 2.0,
        g_init: bool = False,
        g_init_d: float = 2.0,
        moe_ffn: bool = False,
        gmoe: bool = False,
        edge_moe: bool = False,
        moe_experts: int = 8,
        moe_experts_1hop: int = 4,
        moe_k: int = 2,
        moe_coef: float = 1e-2,
        conv_type: str = "gat",
    ):
        super().__init__()
        assert block_style in ("resnet", "gnn_plus"), \
            f"block_style must be 'resnet' or 'gnn_plus', got {block_style!r}"
        self.conv_type = conv_type
        self._edge_aware = conv_type in _EDGE_AWARE_CONVS
        self.dropout = dropout
        self.use_skip = use_skip
        self.block_style = block_style
        self.norm_type = norm_type
        self.act_fn = _activation(activation)
        self._needs_batch = (norm_type == "graph")
        self.use_ffn = use_ffn
        # MoE flags. moe_ffn = Switch-style FFN experts (replaces FFN sub-block).
        # gmoe = Graph-MoE hop experts (replaces main conv). aux load-balance loss
        # accumulated in self._moe_aux_loss each forward, read by the model+trainer.
        self.moe_ffn = moe_ffn
        self.gmoe = gmoe
        self.edge_moe = edge_moe
        self._moe_aux_loss = torch.zeros(())
        self.use_pe = use_pe
        self.pe_walk_length = pe_walk_length
        self._layer_hiddens: list[torch.Tensor] = []  # populated each forward, used by jknet/imtl

        # PE encoder (GNN+ 2025 RWSE): random walk PE → BN → Linear → dim_pe.
        # Concatenated to node features before first conv. Increases in_channels by pe_dim.
        if use_pe:
            self.pe_raw_norm = nn.BatchNorm1d(pe_walk_length)
            self.pe_encoder = nn.Linear(pe_walk_length, pe_dim)
            in_channels_eff = in_channels + pe_dim
        else:
            in_channels_eff = in_channels

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        if gmoe:
            # Graph-MoE: each layer = GMoEConv (hop-1 + hop-2 GATv2 experts, per-node gate).
            for li in range(num_layers):
                in_d = in_channels_eff if li == 0 else hidden_dim
                self.convs.append(GMoEConv(
                    in_d, hidden_dim, num_heads, moe_experts, moe_experts_1hop,
                    moe_k, dropout, edge_dim, add_self_loops, fill_value, coef=moe_coef,
                ))
                self.bns.append(_build_norm(norm_type, hidden_dim))
        elif edge_moe:
            # Edge-type MoE: each layer = EdgeTypeMoEConv (5 relation-group GATv2 experts,
            # per-node gate). Edges partitioned by CPG type → no 2-hop explosion.
            for li in range(num_layers):
                in_d = in_channels_eff if li == 0 else hidden_dim
                self.convs.append(EdgeTypeMoEConv(
                    in_d, hidden_dim, num_heads, moe_k, dropout, edge_dim,
                    add_self_loops, fill_value, coef=moe_coef,
                ))
                self.bns.append(_build_norm(norm_type, hidden_dim))
        else:
            # conv_type selects backbone (gat default; gcn/gatedgcn/gine for GNN+ parity).
            self.convs.append(_make_conv(
                conv_type, in_channels_eff, hidden_dim, num_heads, dropout,
                edge_dim, add_self_loops, fill_value))
            self.bns.append(_build_norm(norm_type, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(_make_conv(
                    conv_type, hidden_dim, hidden_dim, num_heads, dropout,
                    edge_dim, add_self_loops, fill_value))
                self.bns.append(_build_norm(norm_type, hidden_dim))

        if use_skip:
            self.res_projs = _build_res_projs(in_channels_eff, hidden_dim, num_layers)

        # Apply Mustafa NeurIPS 2023 balanced init AFTER all layers built but BEFORE FFN add.
        # Zeros attention vectors + orthogonal init + sqrt(beta) first-layer row scaling.
        if balanced_init:
            apply_balanced_init(self.convs, beta=balanced_init_beta)
        # Apply G-Init (Kelesis 2024) — Kaiming-generalized variance with d_i factor.
        # Mutually exclusive with BalO (last one wins if both set).
        if g_init:
            apply_g_init(self.convs, d_i=g_init_d)

        # Per-layer FFN block (GNN+ 2025 — matches official github.com/LUOyk1999/GNNPlus _ff_block).
        # Block: BN(norm1) → [Linear → Act → Drop → Linear → Drop] → +residual → BN(norm2)
        # 2x expansion (W1: D→2D, W2: 2D→D). Three BNs total per layer when FFN is on.
        if use_ffn:
            ffn_dim = hidden_dim * ffn_expansion
            self.ffn_norm1 = nn.ModuleList([_build_norm(norm_type, hidden_dim) for _ in range(num_layers)])
            self.ffn_norm2 = nn.ModuleList([_build_norm(norm_type, hidden_dim) for _ in range(num_layers)])
            if moe_ffn:
                # Switch-style per-node FFN experts replace the dense FFN.
                self.moe_ffn_layers = nn.ModuleList([
                    NodeMoEFFN(hidden_dim, ffn_expansion, moe_experts, moe_k,
                               dropout, self.act_fn, coef=moe_coef)
                    for _ in range(num_layers)
                ])
            else:
                self.ffn_w1 = nn.ModuleList([nn.Linear(hidden_dim, ffn_dim) for _ in range(num_layers)])
                self.ffn_w2 = nn.ModuleList([nn.Linear(ffn_dim, hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        rwse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # PE: prefer precomputed RWSE from dataset (batch.rwse). Fallback: compute on-the-fly.
        if self.use_pe:
            if rwse is None:
                rwse = _compute_rwse(edge_index, x.size(0), walk_length=self.pe_walk_length)
            pe = self.pe_raw_norm(rwse)
            pe = self.pe_encoder(pe)
            x = torch.cat([x, pe], dim=-1)
        # Reset per-forward MoE aux loss accumulator.
        aux = torch.zeros((), device=x.device)
        self._layer_hiddens = []
        # GMoE needs 2-hop edges (computed once per forward, shared across layers).
        edge_index_2hop = None
        if self.gmoe:
            edge_index_2hop = _two_hop_edge_index(edge_index, x.size(0))
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            if self.gmoe:
                x, lb = conv(x, edge_index, edge_attr, edge_index_2hop)
                aux = aux + lb
            elif self.edge_moe:
                x, lb = conv(x, edge_index, edge_attr)
                aux = aux + lb
            elif self._edge_aware:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)   # GCN: edge-agnostic
            x = bn(x, batch) if self._needs_batch else bn(x)
            if self.block_style == "gnn_plus":
                x = self.act_fn(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if residual is not None:
                    x = x + residual
            else:
                x = self.act_fn(x + residual) if residual is not None else self.act_fn(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # FFN sub-block (GNN+ 2025, matches official _ff_block exactly):
            # x → BN(norm1) → [Linear1 → Act → Drop → Linear2 → Drop] → +x → BN(norm2)
            # MoE-FFN variant: per-node expert FFNs replace the dense FFN.
            if self.use_ffn:
                x = self.ffn_norm1[i](x, batch) if self._needs_batch else self.ffn_norm1[i](x)
                if self.moe_ffn:
                    ff, lb = self.moe_ffn_layers[i](x)
                    aux = aux + lb
                else:
                    ff = self.act_fn(self.ffn_w1[i](x))
                    ff = F.dropout(ff, p=self.dropout, training=self.training)
                    ff = self.ffn_w2[i](ff)
                    ff = F.dropout(ff, p=self.dropout, training=self.training)
                x = x + ff
                x = self.ffn_norm2[i](x, batch) if self._needs_batch else self.ffn_norm2[i](x)
            self._layer_hiddens.append(x)
        # Average aux load-balance loss over layers (reference conv.py:359 /= num_layer).
        n_layers = len(self.convs)
        self._moe_aux_loss = aux / n_layers if n_layers > 0 else aux
        return x


# ── GCN Encoder ───────────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """
    Stack of GCNConv layers. Edge features are ignored (GCN is edge-agnostic).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        add_self_loops: bool = True,
        use_skip: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim, add_self_loops=add_self_loops))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.res_projs = _build_res_projs(in_channels, hidden_dim, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,  # ignored
    ) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x + residual) if residual is not None else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ── RGCN Encoder ──────────────────────────────────────────────────────────────

class RGCNEncoder(nn.Module):
    """
    Relational GCN: one weight matrix per CPG edge type.
    Converts one-hot edge_attr [E, num_relations] → integer edge_type [E].
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_relations: int = NUM_EDGE_TYPES,
        num_bases: int | None = None,
        use_skip: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_dim, num_relations, num_bases=num_bases))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.res_projs = _build_res_projs(in_channels, hidden_dim, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_attr is not None and edge_attr.shape[0] > 0:
            edge_type = edge_attr.argmax(dim=-1)
        else:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            x = conv(x, edge_index, edge_type=edge_type)
            x = bn(x)
            x = F.relu(x + residual) if residual is not None else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ── GGNN Encoder ──────────────────────────────────────────────────────────────

class GGNNEncoder(nn.Module):
    """
    Linear projection + GatedGraphConv (GatedGraphConv requires in==out).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_skip: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,  # ignored
    ) -> torch.Tensor:
        proj = self.input_proj(x)
        h = self.ggnn(proj, edge_index)
        if self.use_skip:
            h = F.relu(h + proj)
        else:
            h = F.relu(h)
        return F.dropout(h, p=self.dropout, training=self.training)


# ── GIN Encoder ───────────────────────────────────────────────────────────────

def _gin_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim, out_dim),
    )


class GINEncoder(nn.Module):
    """
    GINEConv with per-layer edge feature projection.
    Layer 0: edge projection 7→in_channels; layers 1+: 7→hidden_dim.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        edge_dim: int = NUM_EDGE_TYPES,
        use_skip: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # layer 0: edge proj to in_channels
        self.edge_projs = nn.ModuleList()
        self.edge_projs.append(nn.Linear(edge_dim, in_channels))
        self.convs.append(GINEConv(_gin_mlp(in_channels, hidden_dim, dropout), edge_dim=in_channels))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.edge_projs.append(nn.Linear(edge_dim, hidden_dim))
            self.convs.append(GINEConv(_gin_mlp(hidden_dim, hidden_dim, dropout), edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_skip:
            self.res_projs = _build_res_projs(in_channels, hidden_dim, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), NUM_EDGE_TYPES, device=x.device)

        for i, (ep, conv, bn) in enumerate(zip(self.edge_projs, self.convs, self.bns)):
            residual = self.res_projs[i](x) if self.use_skip else None
            ea = ep(edge_attr)
            x = conv(x, edge_index, edge_attr=ea)
            x = bn(x)
            x = F.relu(x + residual) if residual is not None else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ── Encoder factory ───────────────────────────────────────────────────────────

def build_gnn_encoder(
    gnn_model: str,
    in_channels: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_heads: int = 4,
    edge_dim: int = NUM_EDGE_TYPES,
    add_self_loops: bool = False,
    use_skip: bool = False,
    num_relations: int = NUM_EDGE_TYPES,
    num_bases: int | None = None,
    block_style: str = "resnet",
    norm_type: str = "batch",
    activation: str = "relu",
    use_ffn: bool = False,
    ffn_expansion: int = 2,
    use_pe: bool = False,
    pe_walk_length: int = 16,
    pe_dim: int = 28,
    balanced_init: bool = False,
    balanced_init_beta: float = 2.0,
    g_init: bool = False,
    g_init_d: float = 2.0,
    moe_ffn: bool = False,
    gmoe: bool = False,
    edge_moe: bool = False,
    moe_experts: int = 8,
    moe_experts_1hop: int = 4,
    moe_k: int = 2,
    moe_coef: float = 1e-2,
) -> nn.Module:
    """Build a GNN encoder by name. All encoders share forward(x, edge_index, edge_attr).

    gat  — GATv2Conv  (uses num_heads, edge_dim, add_self_loops)
    gcn  — GCNConv     (edge-agnostic; uses add_self_loops)
    gin  — GINEConv    (uses edge_dim)
    rgcn — RGCNConv    (uses num_relations, num_bases)
    ggnn — GatedGraphConv (edge-agnostic)

    block_style: "resnet" (legacy) or "gnn_plus" (Luo 2025) — currently only GAT supports.
    """
    m = gnn_model.lower()
    if m == "gat":
        return GATEncoder(in_channels, hidden_dim, num_layers, num_heads, dropout,
                          edge_dim, add_self_loops, use_skip,
                          block_style=block_style, norm_type=norm_type, activation=activation,
                          use_ffn=use_ffn, ffn_expansion=ffn_expansion,
                          use_pe=use_pe, pe_walk_length=pe_walk_length, pe_dim=pe_dim,
                          balanced_init=balanced_init, balanced_init_beta=balanced_init_beta,
                          g_init=g_init, g_init_d=g_init_d,
                          moe_ffn=moe_ffn, gmoe=gmoe, edge_moe=edge_moe, moe_experts=moe_experts,
                          moe_experts_1hop=moe_experts_1hop, moe_k=moe_k, moe_coef=moe_coef)
    if m == "gatedgcn":
        # Faithful GNN+ GatedGCN (ported 1:1 from GNNPlus GatedGCNLayer).
        return GatedGCNEncoder(in_channels, hidden_dim, num_layers, dropout, edge_dim,
                               residual=use_skip, ffn=use_ffn, act=activation)
    if m == "gin":
        return GINEncoder(in_channels, hidden_dim, num_layers, dropout,
                          edge_dim, use_skip)
    if m == "rgcn":
        return RGCNEncoder(in_channels, hidden_dim, num_layers, dropout,
                           num_relations, num_bases, use_skip)
    if m == "ggnn":
        return GGNNEncoder(in_channels, hidden_dim, num_layers, dropout, use_skip)
    raise ValueError(
        f"gnn_model must be gat|gcn|gin|rgcn|ggnn, got {gnn_model!r}"
    )
