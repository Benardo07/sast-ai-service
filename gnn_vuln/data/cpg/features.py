"""CPG feature extraction: node one-hots, edge attrs, distance features."""
from __future__ import annotations

from collections import deque

import torch

from gnn_vuln.data.node_embedder import NODE_TYPE_TO_IDX
from gnn_vuln.data.cpg.constants import (
    EDGE_TYPES, EDGE_TYPE_TO_IDX,
    DANGEROUS_APIS, _TOKEN_RE,
    _NUM_NODE_TYPES, _CST_TO_IDX, _EVAL_TO_IDX,
    _DISPATCH_IDX, _DISPATCH_VALS, _INTEGER_TYPES,
    _MAX_ARG_IDX, _check_enum,
)

# ---------------------------------------------------------------------------
# Node type one-hot
# ---------------------------------------------------------------------------

def _node_type_onehot(attrs: dict) -> list[float]:
    """24D one-hot vector for node label. Unknown → UNKNOWN bucket."""
    label = str(attrs.get("labelV", attrs.get("label", "UNKNOWN")))
    if label not in NODE_TYPE_TO_IDX:
        _check_enum("node_label", label, set(NODE_TYPE_TO_IDX))
    idx = NODE_TYPE_TO_IDX.get(label, NODE_TYPE_TO_IDX["UNKNOWN"])
    vec = [0.0] * _NUM_NODE_TYPES
    vec[idx] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Edge attributes
# ---------------------------------------------------------------------------

def _edge_attr(edge_attrs: dict) -> list[float]:
    """18D: 17D edge-type one-hot + 1D has_variable flag."""
    etype = edge_attrs.get("label", "")
    if etype not in EDGE_TYPE_TO_IDX:
        _check_enum("edge_type", etype, set(EDGE_TYPE_TO_IDX))
        one_hot = [0.0] * len(EDGE_TYPES)
    else:
        one_hot = [0.0] * len(EDGE_TYPES)
        one_hot[EDGE_TYPE_TO_IDX[etype]] = 1.0
    var = edge_attrs.get("variable", edge_attrs.get("VARIABLE", edge_attrs.get("property", "")))
    has_var = 1.0 if var and str(var).strip() else 0.0
    return one_hot + [has_var]


# ---------------------------------------------------------------------------
# Line number helpers
# ---------------------------------------------------------------------------

def _node_line(attrs: dict) -> int:
    """Source line number from a CPG node; -1 if absent."""
    ln = attrs.get("lineNumber", attrs.get("LINE_NUMBER", None))
    try:
        return int(ln) if ln is not None else -1
    except (ValueError, TypeError):
        return -1


def _node_end_line(nd: dict) -> int:
    """End source line number; -1 if absent."""
    try:
        v = nd.get("lineNumberEnd", nd.get("LINE_NUMBER_END", None))
        return int(v) if v is not None else -1
    except (ValueError, TypeError):
        return -1


# ---------------------------------------------------------------------------
# Distance features
# ---------------------------------------------------------------------------

def _bfs_distances(sources: list, adjacency: dict) -> dict:
    """Multi-source BFS. Returns {node_id: hop_distance}."""
    dist: dict = {}
    queue: deque = deque()
    for s in sources:
        if s not in dist:
            dist[s] = 0
            queue.append(s)
    while queue:
        curr = queue.popleft()
        for nb in adjacency.get(curr, []):
            if nb not in dist:
                dist[nb] = dist[curr] + 1
                queue.append(nb)
    return dist


def _compute_distance_features(nodes: list[dict], edges: list[dict]) -> torch.Tensor:
    """
    (N, 3) float32 tensor:
      [0] dist from METHOD entry (forward)
      [1] dist to RETURN/METHOD_RETURN (reverse)
      [2] dist to nearest CALL (reverse)
    All normalised to [0, 1].
    """
    n = len(nodes)
    norm = max(n - 1, 1)

    adj:  dict = {nd["id"]: [] for nd in nodes}
    radj: dict = {nd["id"]: [] for nd in nodes}
    for e in edges:
        s, d = e.get("src"), e.get("dst")
        if s in adj and d in adj:
            adj[s].append(d)
            radj[d].append(s)

    def _label(nd: dict) -> str:
        return str(nd.get("labelV", nd.get("label", "UNKNOWN"))).upper()

    entry_ids = [nd["id"] for nd in nodes if _label(nd) == "METHOD"] or [nodes[0]["id"]]
    exit_ids  = [nd["id"] for nd in nodes if _label(nd) in ("RETURN", "METHOD_RETURN")]
    call_ids  = [nd["id"] for nd in nodes if _label(nd) == "CALL"]

    d_entry = _bfs_distances(entry_ids, adj)
    d_exit  = _bfs_distances(exit_ids,  radj)
    d_call  = _bfs_distances(call_ids,  radj)

    feats = [
        [
            min(d_entry.get(nd["id"], n) / norm, 1.0),
            min(d_exit.get(nd["id"],  n) / norm, 1.0),
            min(d_call.get(nd["id"],  n) / norm, 1.0),
        ]
        for nd in nodes
    ]
    return torch.tensor(feats, dtype=torch.float)


# ---------------------------------------------------------------------------
# Dangerous API detection
# ---------------------------------------------------------------------------

def _dangerous_api_flags(nodes: list[dict], codes: list[str]) -> torch.Tensor:
    """(N, 1): 1 if node code or methodFullName matches a known dangerous API."""
    flags = []
    for node, code in zip(nodes, codes):
        code_hit = any(t in DANGEROUS_APIS for t in _TOKEN_RE.findall(code))
        mfn = str(node.get("methodFullName", node.get("METHOD_FULL_NAME", "")))
        mfn_name = mfn.split(".")[-1].split(":")[0].split("<")[0]
        mfn_hit = mfn_name in DANGEROUS_APIS
        flags.append(1.0 if code_hit or mfn_hit else 0.0)
    return torch.tensor(flags, dtype=torch.float).unsqueeze(1)


# ---------------------------------------------------------------------------
# Extra node features
# ---------------------------------------------------------------------------

def _extra_node_features(nodes: list[dict]) -> torch.Tensor:
    """
    (N, 27) float32 tensor of additional CPG node attributes.
    All values in [0, 1]. 0 always means "absent/not applicable".

      [0]     is_external        — binary: isExternal flag
      [1-12]  ctrl_struct_type   — 12D one-hot: [none,IF,ELSE,WHILE,FOR,SWITCH,BREAK,CONTINUE,GOTO,TRY,CATCH,FINALLY]
      [13]    has_type_info      — binary: typeFullName present and non-empty
      [14]    is_pointer_type    — binary: typeFullName contains pointer
      [15]    is_integer_type    — binary: typeFullName is known integer type
      [16]    is_void_type       — binary: typeFullName is void/ANY
      [17-20] evaluation_strategy — 4D one-hot: [missing,BY_VALUE,BY_REFERENCE,BY_SHARING]
      [21]    argument_index     — (raw+1)/17: 0=missing/-1, 0.06=idx0, ..., 1.0=idx16
      [22-24] dispatch_type      — 3D one-hot: [missing,STATIC_DISPATCH,DYNAMIC_DISPATCH]
      [25]    is_variadic        — binary: variadic param (..., va_list)
      [26]    span_normalized    — (lineNumberEnd-lineNumber)/50, capped at 1.0
    """
    rows = []
    for n in nodes:
        is_ext = 1.0 if str(n.get("isExternal", n.get("IS_EXTERNAL", "false"))).lower() == "true" else 0.0

        cst = str(n.get("controlStructureType", n.get("CONTROL_STRUCTURE_TYPE", ""))).upper()
        if cst:
            _check_enum("controlStructureType", cst, _CST_TO_IDX)
        cst_vec = [0.0] * 14  # 0=none, 1-13 for CST types
        cst_vec[_CST_TO_IDX.get(cst, 0)] = 1.0

        tfn = str(n.get("typeFullName", n.get("TYPE_FULL_NAME", ""))).strip()
        if tfn:
            has_type = 1.0
            is_ptr  = 1.0 if ("*" in tfn or "[]" in tfn) else 0.0
            is_int  = 1.0 if tfn.lower() in _INTEGER_TYPES else 0.0
            is_void = 1.0 if tfn.lower() in ("void", "any") else 0.0
        else:
            has_type = is_ptr = is_int = is_void = 0.0

        es = str(n.get("evaluationStrategy", n.get("EVALUATION_STRATEGY", ""))).upper()
        if es:
            _check_enum("evaluationStrategy", es, _EVAL_TO_IDX)
        es_vec = [0.0] * 4
        es_vec[_EVAL_TO_IDX.get(es, 0)] = 1.0

        try:
            _raw = n.get("argumentIndex", n.get("ARGUMENT_INDEX", None))
            raw_ai = float(_raw) if _raw is not None else -1.0
            raw_ai = max(-1.0, min(raw_ai, _MAX_ARG_IDX))
            ai = (raw_ai + 1.0) / (_MAX_ARG_IDX + 1.0)
        except (ValueError, TypeError):
            ai = 0.0

        dt = str(n.get("dispatchType", n.get("DISPATCH_TYPE", ""))).upper()
        if dt:
            _check_enum("dispatchType", dt, _DISPATCH_VALS)
        dt_vec = [0.0] * 4  # 0=missing, 1=STATIC, 2=DYNAMIC, 3=INLINED
        dt_vec[_DISPATCH_IDX.get(dt, 0)] = 1.0

        is_var = 1.0 if str(n.get("isVariadic", n.get("IS_VARIADIC", "false"))).lower() == "true" else 0.0

        try:
            ln_start = int(n.get("lineNumber", n.get("LINE_NUMBER", 0)) or 0)
            ln_end   = int(n.get("lineNumberEnd", n.get("LINE_NUMBER_END", ln_start)) or ln_start)
            span_norm = min(max(0, ln_end - ln_start), 50) / 50.0
        except (ValueError, TypeError):
            span_norm = 0.0

        rows.append([is_ext] + cst_vec + [has_type, is_ptr, is_int, is_void] + es_vec + [ai] + dt_vec + [is_var, span_norm])

    return torch.tensor(rows, dtype=torch.float)
