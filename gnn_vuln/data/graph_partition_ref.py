"""VERBATIM reference partition (Graph-ViT/MLP-Mixer, He et al. 2023).

Copied op-for-op from src/graph-vit-mlpmixer core/transform_utils/subgraph_extractors.py
(k_hop_subgraph, metis_subgraph) + core/transform.py (to_sparse, combine_subgraphs,
cal_coarsen_adj). Used by the OFFLINE build on cloud so the patched .pt is produced by
the AUTHORS' EXACT code — not the plain-torch reimplementation in graph_partition.py
(which stays for local online-fallback + testing, and is verified equal to this).

Cloud-only: needs torch_sparse (k_hop SparseTensor) + metis + networkx + libmetis.
torch_sparse imported lazily so this module loads even where it's absent.
"""
from __future__ import annotations

import numpy as np
import torch

import metis as _metis          # noqa: E402  (cloud-only)
import networkx as nx           # noqa: E402

from gnn_vuln.data.graph_partition import SubgraphsData, _random_walk


# ── subgraph_extractors.py (verbatim) ──────────────────────────────────────────

def k_hop_subgraph(edge_index, num_nodes, num_hops, is_directed=False):
    from torch_sparse import SparseTensor  # lazy: cloud-only
    if is_directed:
        row, col = edge_index
        birow, bicol = torch.cat([row, col]), torch.cat([col, row])
        edge_index = torch.stack([birow, bicol])
    else:
        row, col = edge_index
    sparse_adj = SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)]
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator == -1) & next_mask] = i + 1
    hop_indicator = hop_indicator.T  # N x N
    node_mask = (hop_indicator >= 0)
    return node_mask


def metis_subgraph(g, n_patches, drop_rate=0.0, num_hops=1, is_directed=False):
    # undirected branch (verbatim); returns base membership too (for localization)
    if g.num_nodes < n_patches:
        membership = torch.randperm(n_patches)
    else:
        adjlist = g.edge_index.t()
        arr = torch.rand(len(adjlist))
        selected = arr > drop_rate
        G = nx.Graph()
        G.add_nodes_from(np.arange(g.num_nodes))
        G.add_edges_from(adjlist[selected].tolist())
        cuts, membership = _metis.part_graph(G, n_patches, recursive=True)

    assert len(membership) >= g.num_nodes
    membership = torch.tensor(np.array(membership[:g.num_nodes]))
    max_patch_id = torch.max(membership) + 1
    membership = membership + (n_patches - max_patch_id)

    node_mask = torch.stack([membership == i for i in range(n_patches)])

    if num_hops > 0:
        subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero().T
        k_hop_node_mask = k_hop_subgraph(
            g.edge_index, g.num_nodes, num_hops, is_directed)
        node_mask.index_add_(0, subgraphs_batch,
                             k_hop_node_mask[subgraphs_node_mapper])

    edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
    return node_mask, edge_mask, membership


# ── transform.py (verbatim) ─────────────────────────────────────────────────────

def cal_coarsen_adj(subgraphs_nodes_mask):
    mask = subgraphs_nodes_mask.to(torch.float)
    return torch.matmul(mask, mask.t())


def to_sparse(node_mask, edge_mask):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    return subgraphs_nodes, subgraphs_edges


def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1
    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)
    inc = torch.arange(num_selected) * num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]
    return combined_subgraphs


# ── wrapper → SubgraphsData with the fields graph_vit reads ─────────────────────

def partition_data_ref(data, n_patches, num_hops=1, drop_rate=0.0, patch_rw_dim=8):
    """Reference GraphPartitionTransform (verbatim metis+k_hop+combine) → SubgraphsData."""
    node_mask, edge_mask, membership = metis_subgraph(
        data, n_patches=n_patches, drop_rate=drop_rate, num_hops=num_hops)
    subgraphs_nodes, subgraphs_edges = to_sparse(node_mask, edge_mask)
    combined = combine_subgraphs(
        data.edge_index, subgraphs_nodes, subgraphs_edges,
        num_selected=n_patches, num_nodes=data.num_nodes)
    patch_pe = _random_walk(cal_coarsen_adj(node_mask), patch_rw_dim)

    sd = SubgraphsData(**{k: v for k, v in data})
    sd.n_patches = n_patches
    sd.subgraphs_nodes_mapper = subgraphs_nodes[1]
    sd.subgraphs_batch = subgraphs_nodes[0]
    sd.subgraphs_edges_mapper = subgraphs_edges[1]
    sd.combined_subgraphs = combined
    sd.patch_membership = membership.long()         # base home patch per node (localization)
    sd.patch_pe = patch_pe
    mask = torch.zeros(n_patches, dtype=torch.bool)
    mask[subgraphs_nodes[0]] = True
    sd.patch_mask = mask.unsqueeze(0)
    return sd
