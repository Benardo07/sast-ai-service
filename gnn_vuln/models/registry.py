"""Model registry: central mapping from architecture name → class + build_model factory."""

from __future__ import annotations

from gnn_vuln.config import Config
from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.lmgat_codebert import LMGATCodeBERTVulnDetector
from gnn_vuln.models.lmgat_codebert_mtl import LMGATCodeBERTMTLVulnDetector
from gnn_vuln.models.lmgat_mcs import LMGATMCSVulnDetector
from gnn_vuln.models.lmgat_interp import LMGATInterpVulnDetector
from gnn_vuln.models.lmgat_seq import LMGATSeqVulnDetector
from gnn_vuln.models.lmgat_seqgnn import LMGATSeqGNNVulnDetector
from gnn_vuln.models.lmgat_waves_seq import LMGATWavesSeqVulnDetector
from gnn_vuln.models.lmgat_dualflow import LMGATDualFlowVulnDetector
from gnn_vuln.models.lmgat_hcdfgat import LMGATHCDFGATVulnDetector
from gnn_vuln.models.graph_vit import GraphViTVulnDetector

# ── MTL head validation ────────────────────────────────────────────────────────

_MTL_ARCHS   = frozenset({"lmgat_codebert_mtl", "lmgat_hcdfgat"})
_VALID_HEADS = frozenset({"binary", "group", "cwe"})


def _parse_active_heads(cfg: Config) -> frozenset[str]:
    """
    Parse model.active_heads from config for MTL architectures.
    Returns empty frozenset for non-MTL archs.
    Raises ValueError on mode/head conflicts.
    """
    arch = getattr(cfg.model, "architecture", "").lower()
    if arch not in _MTL_ARCHS:
        return frozenset()

    raw = getattr(cfg.model, "active_heads", None)
    if raw is None:
        return frozenset({"binary", "group", "cwe"})

    active = frozenset(str(h).lower() for h in raw)
    unknown = active - _VALID_HEADS
    if unknown:
        raise ValueError(
            f"model.active_heads contains unknown heads: {sorted(unknown)}. "
            f"Valid: {sorted(_VALID_HEADS)}"
        )
    mode = getattr(cfg.data, "mode", "multiclass")
    if mode == "group" and "group" not in active:
        raise ValueError("Conflict: data.mode='group' requires 'group' in model.active_heads")
    if mode == "multiclass" and "cwe" not in active:
        raise ValueError("Conflict: data.mode='multiclass' requires 'cwe' in model.active_heads")
    return active


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, type[VulnDetectorBase]] = {
    "lmgat_codebert":     LMGATCodeBERTVulnDetector,
    "lmgat_codebert_mtl": LMGATCodeBERTMTLVulnDetector,
    "lmgat_mcs":          LMGATMCSVulnDetector,
    "lmgat_interp":       LMGATInterpVulnDetector,
    "lmgat_seq":          LMGATSeqVulnDetector,
    "lmgat_seqgnn":       LMGATSeqGNNVulnDetector,
    "lmgat_waves_seq":    LMGATWavesSeqVulnDetector,
    "lmgat_dualflow":     LMGATDualFlowVulnDetector,
    "lmgat_hcdfgat":      LMGATHCDFGATVulnDetector,
    "graph_vit":          GraphViTVulnDetector,
}


def build_model(
    cfg: Config,
    in_channels: int,
    active_heads: frozenset[str] = frozenset(),
) -> VulnDetectorBase:
    """
    Build a model from config using the model registry.
    Each model class implements from_config(cfg, in_channels, **kwargs).
    """
    arch = cfg.model.architecture.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {arch!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[arch].from_config(cfg, in_channels, active_heads=active_heads)
