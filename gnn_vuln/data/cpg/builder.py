"""CPG builder: assemble PyG Data from parsed CPG + LM embeddings."""
from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Data

from gnn_vuln.data.node_embedder import LMNodeEmbedder
from gnn_vuln.data.cpg.constants import EDGE_TYPES
from gnn_vuln.data.cpg.features import (
    _node_type_onehot, _edge_attr, _node_line, _node_end_line,
    _compute_distance_features, _dangerous_api_flags, _extra_node_features,
)
from gnn_vuln.data.cpg.parser import parse_cpg


def build_func_text(cpg: dict) -> str:
    """
    Reconstruct readable function body from CPG node code snippets,
    sorted by source line number. Used to tokenize the full function for
    live LM injection.
    """
    nodes = cpg["nodes"]
    codes = cpg["codes"]
    line_codes: dict[int, str] = {}
    for code, node in zip(codes, nodes):
        line = _node_line(node)
        if line >= 0 and code.strip() and line not in line_codes:
            line_codes[line] = code
    if line_codes:
        return "\n".join(line_codes[l] for l in sorted(line_codes))
    return " ".join(c for c in codes[:50] if c.strip()) or "<empty>"


def build_from_parsed(
    cpg: dict,
    cls_feats: torch.Tensor,
    label: int,
    flaw_lines: list[int] | None = None,
    func_input_ids: torch.Tensor | None = None,
    func_attention_mask: torch.Tensor | None = None,
    func_token_lines: torch.Tensor | None = None,
) -> Data:
    """
    Build a PyG Data object from a pre-parsed CPG + pre-computed LM node embeddings.

    Parameters
    ----------
    cpg       : dict from parse_cpg() — contains nodes, edges, codes
    cls_feats : (N, lm_dim) LM CLS embeddings per node
    label     : graph-level class label
    flaw_lines: 1-indexed vulnerable line numbers (empty → all-zero mask)

    Returns
    -------
    Data(x=[N, NON_LM_FEAT_DIM+lm_dim], edge_index=[2,E], edge_attr=[E,18],
         y=[1], node_line=[N], flaw_line_mask=[N])
    """
    nodes = cpg["nodes"]
    edges = cpg["edges"]
    codes = cpg["codes"]

    node_ids = [n["id"] for n in nodes]
    node_idx = {nid: i for i, nid in enumerate(node_ids)}

    type_oh      = torch.tensor([_node_type_onehot(n) for n in nodes], dtype=torch.float)
    dist_feats   = _compute_distance_features(nodes, edges)
    danger_flags = _dangerous_api_flags(nodes, codes)
    extra_feats  = _extra_node_features(nodes)

    x = torch.cat([type_oh, cls_feats, dist_feats, danger_flags, extra_feats], dim=1)

    node_lines = [_node_line(n) for n in nodes]
    node_line  = torch.tensor(node_lines, dtype=torch.long)
    flaw_set   = set(flaw_lines) if flaw_lines else set()

    flaw_line_mask = torch.tensor(
        [1 if (ln >= 0 and any(ln <= fl <= max(ln, _node_end_line(n)) for fl in flaw_set))
         else 0
         for ln, n in zip(node_lines, nodes)],
        dtype=torch.long,
    )

    src_list, dst_list, edge_attr_list = [], [], []
    for e in edges:
        src, dst = e.get("src"), e.get("dst")
        if src in node_idx and dst in node_idx:
            src_list.append(node_idx[src])
            dst_list.append(node_idx[dst])
            edge_attr_list.append(_edge_attr(e))

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, len(EDGE_TYPES) + 1), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        node_line=node_line,
        flaw_line_mask=flaw_line_mask,
    )
    if func_input_ids is not None:
        data.func_input_ids = func_input_ids
        data.func_attention_mask = func_attention_mask
    if func_token_lines is not None:
        data.func_token_lines = func_token_lines
    return data


def build_lm_graph_from_json(
    path,
    label: int,
    embedder: LMNodeEmbedder,
    max_nodes: int = 500,
    embed_batch_size: int = 256,
    flaw_lines: list[int] | None = None,
) -> Optional[Data]:
    """Convenience: parse + embed + build in one call (single-graph use)."""
    cpg = parse_cpg(path, max_nodes)
    if cpg is None:
        return None
    codes = cpg["codes"]
    cls_parts = [
        embedder.embed_batch(codes[i: i + embed_batch_size])
        for i in range(0, len(codes), embed_batch_size)
    ]
    cls_feats = torch.cat(cls_parts, dim=0)
    return build_from_parsed(cpg, cls_feats, label, flaw_lines)
