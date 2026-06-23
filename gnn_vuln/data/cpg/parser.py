"""CPG parsers: GraphML (Joern), MegaVul JSON, and unified parse_cpg()."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from gnn_vuln.data.cpg.constants import _GRAPHML_NS


def _parse_graphml(path: str | Path) -> dict:
    """
    Parse a Joern GraphML (.xml) export into unified dict:
        {"nodes": [{id, labelV, code, lineNumber, ...}],
         "edges": [{src, dst, label, property}]}
    """
    def _tag(name: str) -> str:
        return f"{{{_GRAPHML_NS}}}{name}"

    tree = ET.parse(path)
    root = tree.getroot()

    key_map: dict[str, str] = {}
    for key_el in root.iter(_tag("key")):
        kid = key_el.get("id", "")
        aname = key_el.get("attr.name", kid)
        key_map[kid] = aname

    nodes: list[dict] = []
    edges: list[dict] = []

    for graph_el in root.iter(_tag("graph")):
        for node_el in graph_el.findall(_tag("node")):
            nd: dict = {"id": node_el.get("id")}
            for data_el in node_el.findall(_tag("data")):
                kid = data_el.get("key", "")
                aname = key_map.get(kid, kid)
                val = data_el.text or ""
                if aname == "labelV":
                    nd["labelV"] = val
                elif aname == "CODE":
                    nd["code"] = val
                elif aname == "LINE_NUMBER":
                    nd["lineNumber"] = val
                else:
                    nd[aname] = val
            nodes.append(nd)

        for edge_el in graph_el.findall(_tag("edge")):
            ed: dict = {
                "src": edge_el.get("source"),
                "dst": edge_el.get("target"),
            }
            for data_el in edge_el.findall(_tag("data")):
                kid = data_el.get("key", "")
                aname = key_map.get(kid, kid)
                val = data_el.text or ""
                ed["label" if aname == "labelE" else aname] = val
            edges.append(ed)

    return {"nodes": nodes, "edges": edges}


def _parse_megavul_json(path: str | Path) -> dict:
    """
    Parse MegaVul pre-built graph JSON into unified dict.

    MegaVul schema:
      node: {_label, id (int), code, lineNumber, ...}
      edge: {inNode (dst), outNode (src), etype (label), variable}
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    nodes: list[dict] = []
    for n in raw.get("nodes", []):
        nd = dict(n)
        if "_label" in nd:
            nd["labelV"] = nd.pop("_label")
        nd["id"] = nd.get("id", "")
        nodes.append(nd)

    edges: list[dict] = []
    for e in raw.get("edges", []):
        edges.append({
            "src":      e.get("outNode", ""),
            "dst":      e.get("inNode",  ""),
            "label":    e.get("etype",   ""),
            "variable": e.get("variable", ""),
        })

    return {"nodes": nodes, "edges": edges}


def parse_cpg_hdf5_group(fg, max_nodes: int = 5000) -> Optional[dict]:
    """
    Read a CPG from an open h5py group (benign/func_X or vulnerable/func_X).
    Returns same {nodes, edges, codes} dict as parse_cpg(), or None if too large.
    Requires the group to have nodes_json + edges_json datasets (written by convert_raw_to_hdf5.py).
    """
    if "nodes_json" not in fg or "edges_json" not in fg:
        return None

    raw_nodes = fg["nodes_json"][()]
    raw_edges = fg["edges_json"][()]
    nodes_str = raw_nodes.decode() if isinstance(raw_nodes, (bytes, bytearray)) else str(raw_nodes)
    edges_str = raw_edges.decode() if isinstance(raw_edges, (bytes, bytearray)) else str(raw_edges)

    nodes = json.loads(nodes_str)
    edges = json.loads(edges_str)

    if not nodes or len(nodes) > max_nodes:
        return None

    return {"nodes": nodes, "edges": edges, "codes": [str(n.get("code", "")) for n in nodes]}


def parse_cpg(path: str | Path, max_nodes: int = 500) -> Optional[dict]:
    """
    Parse a CPG file (.xml/.graphml = Joern GraphML, .json = MegaVul or plain JSON)
    and validate its size.

    Returns dict with {nodes, edges, codes} ready for build_from_parsed(),
    or None if empty or exceeds max_nodes.
    """
    path = Path(path)
    if path.suffix.lower() in (".xml", ".graphml"):
        cpg = _parse_graphml(path)
    elif path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        first_node = (raw.get("nodes") or [{}])[0]
        first_edge = (raw.get("edges") or [{}])[0]
        if "_label" in first_node or "inNode" in first_edge:
            cpg = _parse_megavul_json(path)
        else:
            cpg = raw
    else:
        with open(path, encoding="utf-8") as f:
            cpg = json.load(f)

    nodes = cpg.get("nodes", [])
    if not nodes or len(nodes) > max_nodes:
        return None

    cpg["codes"] = [str(n.get("code", "")) for n in nodes]
    return cpg
