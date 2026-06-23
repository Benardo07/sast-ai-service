"""CPG subpackage — Code Property Graph parsing, features, and PyG builder."""

from gnn_vuln.data.cpg.constants import (
    EDGE_TYPES,
    EDGE_TYPE_TO_IDX,
    DANGEROUS_APIS,
    _check_enum,
)
from gnn_vuln.data.cpg.parser import (
    _parse_graphml,
    _parse_megavul_json,
    parse_cpg,
)
from gnn_vuln.data.cpg.features import (
    _node_type_onehot,
    _edge_attr,
    _node_line,
    _node_end_line,
    _compute_distance_features,
    _dangerous_api_flags,
    _extra_node_features,
)
from gnn_vuln.data.cpg.builder import (
    build_func_text,
    build_from_parsed,
    build_lm_graph_from_json,
)

__all__ = [
    # constants
    "EDGE_TYPES", "EDGE_TYPE_TO_IDX", "DANGEROUS_APIS",
    # parser
    "parse_cpg", "_parse_graphml", "_parse_megavul_json",
    # features
    "_node_type_onehot", "_edge_attr", "_node_line", "_node_end_line",
    "_compute_distance_features", "_dangerous_api_flags", "_extra_node_features",
    # builder
    "build_func_text", "build_from_parsed", "build_lm_graph_from_json",
]
