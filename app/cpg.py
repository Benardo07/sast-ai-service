"""CPG generation: function source -> Joern CPG JSON, via the vendored gnn_vuln.

Used by the backend's dataset materializer (single source of truth) which caches
the result in its own ``graph_cache`` table. This service only does the parsing.

NOTE: requires Joern (joern-cli) installed on the deploy machine. The exact CPG
JSON shape must match what gnn_vuln's training/inference graph builder expects;
confirm on the deploy machine.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.config import settings


def build_cpg(code: str, language: str | None = None) -> dict:
    # Joern's JSON export is unreliable on Joern v4 (produces nothing); GraphML export
    # works (it's what inference uses). Export GraphML and parse it into the same
    # {nodes, edges, codes} dict the training/inference graph builder consumes — the
    # downstream relearn dataset stays JSON-shaped, so nothing else changes.
    from gnn_vuln.data.cpg.parser import parse_cpg
    from gnn_vuln.data.joern_runner import process_function

    joern_cli = Path(settings.joern_cli) if settings.joern_cli else None
    with tempfile.TemporaryDirectory() as out:
        dest = process_function(
            code,
            0,
            Path(out),
            joern_cli_dir=joern_cli,
            fmt="graphml",
            lang=language or None,
        )
        if dest is None:
            raise RuntimeError("Joern produced no CPG (check joern-cli install and language)")
        cpg = parse_cpg(dest, max_nodes=100000)
        if not cpg or not cpg.get("nodes"):
            raise RuntimeError("Parsed CPG has no nodes")
        return {"cpg_json": cpg, "node_count": len(cpg.get("nodes", []))}
