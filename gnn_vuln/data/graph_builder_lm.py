"""
graph_builder_lm.py — backward-compatibility shim.
All logic has moved to gnn_vuln.data.cpg subpackage.
"""
from gnn_vuln.data.cpg import *  # noqa: F401, F403
from gnn_vuln.data.cpg import (
    EDGE_TYPES, EDGE_TYPE_TO_IDX, DANGEROUS_APIS,
    parse_cpg, build_from_parsed, build_func_text, build_lm_graph_from_json,
    _parse_graphml, _parse_megavul_json,
    _node_type_onehot, _edge_attr, _node_line,
    _compute_distance_features, _dangerous_api_flags, _extra_node_features,
)
from gnn_vuln.data.cpg.constants import (
    _NUM_NODE_TYPES, _CST_TO_IDX, _EVAL_TO_IDX,
    _DISPATCH_IDX, _DISPATCH_VALS, _check_enum,
)
