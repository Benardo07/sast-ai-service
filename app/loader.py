from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from app.config import settings


@dataclass
class LoadedRelease:
    model_version_id: str | None
    checkpoint_path: str
    config_path: str
    device: str
    architecture: str | None
    data_source: str | None
    data_mode: str | None
    num_classes: int | None
    loaded_at: datetime


class ReleaseManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._predictor = None
        self._release: LoadedRelease | None = None
        self._load_error: str | None = None

    def _ensure_source_repo_on_path(self) -> None:
        source_repo = settings.source_repo_path
        src_dir = source_repo / "src"
        if not src_dir.exists():
            raise FileNotFoundError(f"AI source repo src directory not found: {src_dir}")
        src_dir_text = str(src_dir)
        if src_dir_text not in sys.path:
            sys.path.insert(0, src_dir_text)

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (settings.workspace_root / raw_path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        return path

    def _default_config_path(self, checkpoint_path: Path) -> Path:
        candidate = checkpoint_path.parent / "config.yaml"
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            "config_path was not provided and no sibling config.yaml was found next to the checkpoint"
        )

    def _load_config_summary(self, config_path: Path) -> dict[str, Any]:
        self._ensure_source_repo_on_path()
        from gnn_vuln.config import Config

        cfg = Config.from_yaml(config_path)
        return {
            "architecture": getattr(cfg.model, "architecture", None),
            "data_source": getattr(cfg.data, "source", None),
            "data_mode": getattr(cfg.data, "mode", None),
            "num_classes": getattr(cfg.model, "num_classes", None),
        }

    def _persist_release(self, release: LoadedRelease) -> None:
        target = settings.active_release_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(release)
        payload["loaded_at"] = release.loaded_at.astimezone(UTC).isoformat()
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_predictor(self, checkpoint_path: Path, config_path: Path, device: str):
        self._ensure_source_repo_on_path()
        from gnn_vuln.inference import VulnPredictor

        return VulnPredictor.from_checkpoint(
            checkpoint=checkpoint_path,
            config=config_path,
            device=device,
        )

    def load_release(
        self,
        *,
        model_version_id: str | None,
        checkpoint_path: str,
        config_path: str | None,
        device: str | None,
        force_reload: bool = False,
    ) -> LoadedRelease:
        with self._lock:
            resolved_checkpoint = self._resolve_workspace_path(checkpoint_path)
            resolved_config = self._resolve_workspace_path(config_path) if config_path else self._default_config_path(resolved_checkpoint)
            resolved_device = device or settings.default_device

            if (
                not force_reload
                and self._release is not None
                and self._release.checkpoint_path == str(resolved_checkpoint)
                and self._release.config_path == str(resolved_config)
                and self._release.device == resolved_device
            ):
                return self._release

            summary = self._load_config_summary(resolved_config)
            predictor = self._load_predictor(resolved_checkpoint, resolved_config, resolved_device)
            release = LoadedRelease(
                model_version_id=model_version_id,
                checkpoint_path=str(resolved_checkpoint),
                config_path=str(resolved_config),
                device=resolved_device,
                architecture=summary.get("architecture"),
                data_source=summary.get("data_source"),
                data_mode=summary.get("data_mode"),
                num_classes=summary.get("num_classes"),
                loaded_at=datetime.now(UTC),
            )
            self._predictor = predictor
            self._release = release
            self._load_error = None
            self._persist_release(release)
            return release

    def active_release(self) -> LoadedRelease | None:
        with self._lock:
            return self._release

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            if self._release is None:
                detail = self._load_error or "No release loaded"
                return {
                    "status": "idle",
                    "active_model_version_id": None,
                    "active_checkpoint_path": None,
                    "loaded_at": None,
                    "detail": detail,
                }
            return {
                "status": "ready",
                "active_model_version_id": self._release.model_version_id,
                "active_checkpoint_path": self._release.checkpoint_path,
                "loaded_at": self._release.loaded_at,
                "detail": None,
            }

    def active_release_payload(self) -> dict[str, Any]:
        with self._lock:
            if self._release is None:
                return {
                    "model_version_id": None,
                    "checkpoint_path": None,
                    "config_path": None,
                    "device": None,
                    "architecture": None,
                    "data_source": None,
                    "data_mode": None,
                    "num_classes": None,
                    "loaded_at": None,
                    "ready": False,
                    "source_repo_root": str(settings.source_repo_path),
                }
            payload = asdict(self._release)
            payload["ready"] = True
            payload["source_repo_root"] = str(settings.source_repo_path)
            return payload

    def predict_from_cpg(
        self,
        *,
        cpg_path: str,
        top_k_lines: int | None,
        max_nodes: int,
        label: int,
        flaw_lines: list[int] | None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._release is None or self._predictor is None:
                raise RuntimeError("No release loaded. Call /load-release first.")
            resolved_cpg = self._resolve_workspace_path(cpg_path)
            result = self._predictor.predict_from_file(
                resolved_cpg,
                label=label,
                flaw_lines=flaw_lines,
                max_nodes=max_nodes,
                top_k_lines=top_k_lines,
            )
            if result is None:
                raise RuntimeError("Prediction returned no result. CPG may be empty or exceed max_nodes.")
            return {
                "model_version_id": self._release.model_version_id,
                "checkpoint_path": self._release.checkpoint_path,
                "config_path": self._release.config_path,
                "result": result,
                "predicted_at": datetime.now(UTC),
            }

    def autoload_if_enabled(self) -> None:
        if not settings.autoload_persisted_release:
            return
        persisted = settings.active_release_path
        if not persisted.exists():
            return
        payload = json.loads(persisted.read_text(encoding="utf-8"))
        self.load_release(
            model_version_id=payload.get("model_version_id"),
            checkpoint_path=payload["checkpoint_path"],
            config_path=payload.get("config_path"),
            device=payload.get("device"),
            force_reload=True,
        )


release_manager = ReleaseManager()
