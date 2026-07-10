"""Relearn (continual-learning) jobs for the standalone AI service.

Ported and adapted from the original training repo. Key differences:
  * Self-contained: trains via the VENDORED ``gnn_vuln`` package (no tugas-akhir).
  * Stateless w.r.t. the domain model: this service owns NO model/dataset registry.
    The caller (sast-backend, the single source of truth) supplies the base
    checkpoint, the base config payload, and the data sources explicitly, and is
    responsible for persisting the resulting ModelVersion / metrics.
  * Job state is kept in-memory + on disk under ``var/jobs/<job_id>/`` only so the
    backend can poll ``GET /relearn/{job_id}`` for status, the new checkpoint, and
    the training metrics.

Methods: finetune (base weights, no protection), EWC (EWC-DR penalty),
ER (experience replay), EWC-ER (both), retrain (fresh weights, no base).
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from app import storage
from app.config import settings

_REQUIRES_BASE = {"ER", "EWC", "EWC-ER", "finetune"}
_EWC_METHODS = {"EWC", "EWC-ER"}
_REPLAY_METHODS = {"ER", "EWC-ER"}
VALID_METHODS = {"finetune", "EWC", "ER", "EWC-ER", "retrain"}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RelearnJobState:
    job_id: str
    status: str  # queued | running | done | failed
    method: str
    data_source: str
    base_checkpoint_path: str | None = None
    config_path: str | None = None
    log_path: str | None = None
    result_checkpoint_path: str | None = None
    result_config_path: str | None = None
    metrics: dict[str, Any] | None = None
    model_version_id: str | None = None
    message: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)


class RelearnManager:
    """Runs relearn jobs in background threads using the vendored gnn_vuln package."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._jobs: dict[str, RelearnJobState] = {}

    # ── persistence (on-disk, no DB) ──────────────────────────────────────
    def _job_dir(self, job_id: str) -> Path:
        return settings.jobs_root / job_id

    def _save(self, job: RelearnJobState) -> None:
        job.updated_at = _utc_now_iso()
        with self._lock:
            self._jobs[job.job_id] = job
        d = self._job_dir(job.job_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")

    def get(self, job_id: str) -> RelearnJobState | None:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]
        # Fall back to disk (survives restart).
        path = self._job_dir(job_id) / "job.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return RelearnJobState(**data)
        return None

    def list(self) -> list[RelearnJobState]:
        jobs: dict[str, RelearnJobState] = {}
        if settings.jobs_root.exists():
            for jp in settings.jobs_root.glob("*/job.json"):
                try:
                    data = json.loads(jp.read_text(encoding="utf-8"))
                    jobs[data["job_id"]] = RelearnJobState(**data)
                except Exception:  # noqa: BLE001
                    continue
        with self._lock:
            jobs.update(self._jobs)
        return sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)

    # ── config generation (method -> ewc/replay blocks) ───────────────────
    def _build_train_config(
        self,
        *,
        method: str,
        base_config: dict[str, Any],
        data_source: str,
        num_classes: int | None,
        epochs: int | None,
        base_checkpoint_path: str | None,
        base_class_names: list[str] | None,
        replay_source: str | None,
        device: str | None,
        job_dir: Path,
        materialized: bool = False,
    ) -> tuple[Path, Path | None]:
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("data", {})
        cfg["data"]["source"] = data_source
        if materialized:
            # Point gnn_vuln at the CPGs we just wrote under the service data root.
            cfg["data"]["raw_dir"] = str(settings.data_root / "raw")
            cfg["data"]["processed_dir"] = str(settings.data_root / "processed")
        if num_classes is not None:
            cfg.setdefault("model", {})["num_classes"] = num_classes
        if epochs:
            cfg.setdefault("train", {})["epochs"] = epochs
        if device:
            cfg.setdefault("train", {})["device"] = device
        # Make gnn_vuln write checkpoints/results where this service looks for them
        # (otherwise it falls back to its own ./checkpoints and the result is "not found").
        cfg.setdefault("train", {})["checkpoint_dir"] = str(settings.checkpoints_root)
        cfg["train"]["results_dir"] = str(settings.results_root)

        # Continual-learning label alignment: when continuing a base model, remap task-B labels
        # onto the base model's class space so ids never clash. Known CWEs keep their canonical
        # id; brand-new CWEs (class-incremental) extend the head and num_classes grows to fit.
        # gnn_vuln applies the remap at load via data.target_vocab. retrain (no base) skips this.
        if method != "retrain" and base_class_names and materialized:
            tv = {name: i for i, name in enumerate(base_class_names)}
            vpath = settings.data_root / "raw" / data_source / "cwe_vocab.json"
            if vpath.exists():
                for name in json.loads(vpath.read_text(encoding="utf-8")):
                    if name not in tv:
                        tv[name] = len(tv)          # new CWE -> extended id (head grows)
            cfg["data"]["target_vocab"] = tv
            cfg.setdefault("model", {})["num_classes"] = len(tv)

        cfg.pop("ewc", None)
        cfg.pop("replay", None)
        importance_cfg_path: Path | None = None

        if method != "retrain":
            staged_ckpt = ""
            cache = ""
            if base_checkpoint_path:
                stage_dir = job_dir / "base"
                stage_dir.mkdir(parents=True, exist_ok=True)
                import shutil

                staged = stage_dir / "best_model.pt"
                if not staged.exists():
                    shutil.copy2(base_checkpoint_path, staged)
                staged_ckpt = str(staged)
                cache = str(stage_dir / "ewc_importance.pt")

            weight = 1000.0 if method in _EWC_METHODS else 0.0
            cfg["ewc"] = {
                "enabled": True,
                "weight": weight,
                "scope": "all",
                "source_checkpoint": staged_ckpt,
                "importance_cache": cache,
                "n_batches": 0,
            }

            if method in _REPLAY_METHODS and replay_source:
                cfg["replay"] = {
                    "enabled": True,
                    "source": replay_source,
                    "buffer_per_class": 50,
                    "weight": 1.0,
                    "buffer_seed": 42,
                }

            # EWC importance pass (task-A data) if cache missing.
            if method in _EWC_METHODS and replay_source and cache and not Path(cache).exists():
                imp = copy.deepcopy(cfg)
                imp.setdefault("data", {})["source"] = replay_source
                imp["ewc"] = {**cfg["ewc"], "weight": 1000.0, "compute_only": True}
                imp.pop("replay", None)
                importance_cfg_path = job_dir / "importance.yaml"
                importance_cfg_path.write_text(yaml.safe_dump(imp, sort_keys=False), encoding="utf-8")

        train_cfg_path = job_dir / "train.yaml"
        train_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return train_cfg_path, importance_cfg_path

    # ── result discovery ──────────────────────────────────────────────────
    def _find_latest_checkpoint(self, started_at: float) -> Path | None:
        root = settings.checkpoints_root
        if not root.exists():
            return None
        candidates = [
            c for c in root.glob("*_*")
            if c.is_dir() and not c.name.startswith("base") and c.stat().st_mtime >= started_at - 1
        ]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in candidates:
            best = next(run_dir.glob("best_*.pt"), None)
            if best:
                return best
        return None

    @staticmethod
    def _scalar_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
        """Flatten one level of a summary dict into scalar numeric metrics.

        Keeps int/float values (drops bool); recurses one level into nested
        dicts (e.g. metrics_summary's per-section blocks) with a dotted key.
        Arrays/curves are skipped, so the result is safe to store as run metrics.
        """
        out: dict[str, float] = {}
        for k, v in data.items():
            key = f"{prefix}{k}"
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out[key] = float(v)
            elif isinstance(v, dict) and not prefix:
                for sk, sv in v.items():
                    if not isinstance(sv, bool) and isinstance(sv, (int, float)):
                        out[f"{k}.{sk}"] = float(sv)
        return out

    def _parse_metrics(self, checkpoint: Path) -> dict[str, Any] | None:
        """Best-effort: read training_summary.json + metrics_summary.json near the
        checkpoint / results dir, merging all scalar numeric metrics found."""
        search_dirs = [checkpoint.parent, settings.results_root]
        filenames = ("training_summary.json", "metrics_summary.json")
        metrics: dict[str, float] = {}
        for d in search_dirs:
            if not d or not d.exists():
                continue
            for fname in filenames:
                for summary in sorted(
                    d.glob(f"**/{fname}"), key=lambda p: p.stat().st_mtime, reverse=True
                ):
                    try:
                        data = json.loads(summary.read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        continue
                    if isinstance(data, dict):
                        metrics.update(self._scalar_metrics(data))
                    break  # newest file of this kind per dir is enough
        return metrics or None

    @staticmethod
    def _log_tail(path: Path, n: int = 12, maxlen: int = 600) -> str:
        """Last meaningful log lines — folded into the job message so the real
        subprocess error reaches the caller without log-spelunking."""
        try:
            lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
            return "\n".join(lines[-n:])[-maxlen:]
        except Exception:  # noqa: BLE001
            return ""

    # ── job runner ────────────────────────────────────────────────────────
    def _run_job(self, job: RelearnJobState, train_cfg: Path, importance_cfg: Path | None) -> None:
        log = Path(job.log_path) if job.log_path else self._job_dir(job.job_id) / "run.log"
        started_at = datetime.now(UTC).timestamp()
        # GNN_VULN_API_MODE tells the library it runs under a service: skip research-only
        # artifacts (training_log.csv, training_curves.png) and use the metrics-only eval
        # path. split.json + training_summary.json (+ metrics_summary.json) are still written.
        env = {**os.environ, "GNN_VULN_API_MODE": "1"}
        try:
            job.status = "running"
            self._save(job)
            with open(log, "w", encoding="utf-8") as lf:
                if importance_cfg is not None:
                    lf.write(f"== EWC importance: {importance_cfg} ==\n")
                    lf.flush()
                    subprocess.run(
                        [sys.executable, "-m", "gnn_vuln.train", "--config", str(importance_cfg)],
                        check=True, cwd=str(settings.service_root), env=env,
                        stdout=lf, stderr=subprocess.STDOUT,
                    )
                lf.write(f"== train: {train_cfg} ==\n")
                lf.flush()
                subprocess.run(
                    [sys.executable, "-m", "gnn_vuln.train", "--config", str(train_cfg)],
                    check=True, cwd=str(settings.service_root), env=env,
                    stdout=lf, stderr=subprocess.STDOUT,
                )
            best = self._find_latest_checkpoint(started_at)
            if best is None:
                job.status = "failed"
                job.message = "training finished but no checkpoint was found"
            else:
                # Metrics-only evaluation → metrics_summary.json (function-level + localization).
                # Best-effort: training metrics (training_summary.json) are still captured if eval fails.
                try:
                    with open(log, "a", encoding="utf-8") as lf:
                        lf.write(f"== evaluate (metrics-only): {best} ==\n")
                        lf.flush()
                        subprocess.run(
                            [sys.executable, "-m", "gnn_vuln.evaluate", "--checkpoint", str(best),
                             "--config", str(train_cfg), "--metrics-only"],
                            check=True, cwd=str(settings.service_root), env=env,
                            stdout=lf, stderr=subprocess.STDOUT,
                        )
                except Exception as e:  # noqa: BLE001
                    with open(log, "a", encoding="utf-8") as lf:
                        lf.write(f"WARN evaluate (metrics-only) failed: {type(e).__name__}: {e}\n")
                job.result_checkpoint_path = str(best)
                job.result_config_path = str(train_cfg)
                job.metrics = self._parse_metrics(best)
                # In S3 mode push the checkpoint (+ its config sibling) to object storage so
                # any backend/worker node can deploy it later — `result_checkpoint_path`
                # becomes the s3:// pointer the backend stores as the model version source_uri.
                if storage.using_s3():
                    try:
                        prefix = f"relearn/{best.parent.name}"
                        uri = storage.put_file(settings.s3_bucket_checkpoints, f"{prefix}/{best.name}", best)
                        cfg_sibling = best.parent / "config.yaml"
                        cfg_src = cfg_sibling if cfg_sibling.exists() else train_cfg
                        storage.put_file(settings.s3_bucket_checkpoints, f"{prefix}/config.yaml", cfg_src)
                        job.result_checkpoint_path = uri
                    except Exception as e:  # noqa: BLE001 - keep local path if upload fails
                        with open(log, "a", encoding="utf-8") as lf:
                            lf.write(f"WARN checkpoint upload to object storage failed: {type(e).__name__}: {e}\n")
                job.status = "done"
        except subprocess.CalledProcessError as e:
            tail = self._log_tail(log)
            job.status = "failed"
            job.message = (f"training failed (exit {e.returncode}):\n{tail}" if tail
                           else f"training failed (exit {e.returncode}); see log {log}")
        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.message = f"{type(e).__name__}: {e}"
        self._save(job)

    # ── submission ──────────────────────────────────────────────────────
    def _install_bundle(self, uri: str) -> str:
        """Download + extract a prepared dataset bundle (.tar.gz of the gnn_vuln `data/`
        layout) into the data root, and return the data_source name (the raw/<source> dir).
        The leading `data/` prefix is stripped so files land at data_root/processed + raw."""
        import tarfile

        local = storage.ensure_local(uri)  # downloads if s3://, else resolves the path
        data_root = settings.data_root
        data_root.mkdir(parents=True, exist_ok=True)
        source: str | None = None
        with tarfile.open(local, mode="r:gz") as tf:
            members = tf.getmembers()
            for m in members:
                parts = m.name.split("/")
                if source is None and len(parts) >= 3 and parts[0] == "data" and parts[1] == "raw" and parts[2]:
                    source = parts[2]
                # strip the leading "data/" so it extracts into data_root directly
                if parts and parts[0] == "data":
                    m.name = "/".join(parts[1:])
                if m.name:
                    tf.extract(m, path=data_root)
        if not source:
            # fall back: infer from a processed .pt filename lm_dataset_<source>_<mode>_...
            for p in (data_root / "processed").glob("lm_dataset_*.pt"):
                source = p.stem.split("_")[2] if len(p.stem.split("_")) > 2 else None
                break
        if not source:
            raise ValueError("Could not determine data_source from bundle (no data/raw/<source> or processed .pt)")
        return source

    def _write_inline_dataset(self, source: str, entries: list[dict], mode: str) -> int:
        """Write materialized CPG entries to data_root/raw/<source>/{benign,vulnerable}
        with .meta.json sidecars and cwe_vocab.json, as gnn_vuln expects."""
        base = settings.data_root / "raw" / source
        (base / "benign").mkdir(parents=True, exist_ok=True)
        (base / "vulnerable").mkdir(parents=True, exist_ok=True)
        vocab: dict[str, int] = {"benign": 0}
        next_id = 1
        for i, e in enumerate(entries):
            vuln = bool(e.get("is_vulnerable"))
            sub = "vulnerable" if vuln else "benign"
            fname = f"func_{i}"
            (base / sub / f"{fname}.json").write_text(json.dumps(e.get("cpg_json")), encoding="utf-8")
            if vuln:
                cwe = e.get("cwe") or "UNKNOWN"
                if mode == "multiclass":
                    if cwe not in vocab:
                        vocab[cwe] = next_id
                        next_id += 1
                    class_id = vocab[cwe]
                else:
                    class_id = 1
                meta = {"class_id": class_id, "cwe": cwe, "flaw_lines": e.get("flaw_lines") or []}
                (base / sub / f"{fname}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (base / "cwe_vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
        return len(vocab)

    # ── evaluation (score an existing checkpoint on a held-out set) ─────────
    def evaluate_checkpoint(
        self,
        *,
        checkpoint_path: str,
        base_config: dict[str, Any],
        dataset: list[dict],
        source: str | None = None,
        base_class_names: list[str] | None = None,
        device: str | None = None,
    ) -> dict[str, Any]:
        """Run gnn_vuln.evaluate for one checkpoint over an inline CPG dataset used as a
        100% held-out test set. No training. Returns the metrics_summary. Synchronous —
        the caller (backend) runs this from a background task. Reuses the relearn dataset
        writer + target_vocab alignment so labels map onto the model's class space."""
        if not dataset:
            raise ValueError("evaluate requires a non-empty inline dataset")
        if storage.is_s3_uri(checkpoint_path):
            checkpoint_path = str(storage.ensure_local(checkpoint_path))
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        job_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        job_dir = self._job_dir(f"eval_{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)
        mode = (base_config.get("data") or {}).get("mode", "multiclass") if isinstance(base_config, dict) else "multiclass"
        eff_source = source or f"eval_{job_id}"

        # EVALUATION keeps the model's head FIXED at the checkpoint's classes — unlike relearn,
        # it never grows the head (that would desync from the checkpoint weights → load error).
        # A benchmark CWE the model was never trained on is unanswerable by a fixed-class model,
        # so drop those vulnerable samples (benign is always in-vocab). Their count is reported.
        dropped = 0
        if base_class_names and mode == "multiclass":
            known = set(base_class_names)
            kept = [e for e in dataset if (not e.get("is_vulnerable")) or (str(e.get("cwe") or "") in known)]
            dropped = len(dataset) - len(kept)
            dataset = kept
            if not dataset:
                raise ValueError("no benchmark samples have a CWE in the model's class space")

        self._write_inline_dataset(eff_source, dataset, mode)

        cfg = copy.deepcopy(base_config)
        cfg.setdefault("data", {})
        cfg["data"]["source"] = eff_source
        cfg["data"]["raw_dir"] = str(settings.data_root / "raw")
        cfg["data"]["processed_dir"] = str(settings.data_root / "processed")
        # 100% test split: train + val = 0 puts every sample in the test set.
        cfg["data"]["train_ratio"] = 0.0
        cfg["data"]["val_ratio"] = 0.0
        cfg.setdefault("train", {})
        cfg["train"]["results_dir"] = str(job_dir)          # isolate metrics per eval
        cfg["train"]["checkpoint_dir"] = str(settings.checkpoints_root)
        if device:
            cfg["train"]["device"] = device
        # Fixed target_vocab = the model's exact class list (no extension); num_classes matches
        # the checkpoint head. The benchmark was already filtered to these classes above.
        if base_class_names and mode == "multiclass":
            cfg["data"]["target_vocab"] = {name: i for i, name in enumerate(base_class_names)}
            cfg.setdefault("model", {})["num_classes"] = len(base_class_names)

        cfg_path = job_dir / "eval.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        log = job_dir / "eval.log"
        env = {**os.environ, "GNN_VULN_API_MODE": "1"}
        with open(log, "a", encoding="utf-8") as lf:
            subprocess.run(
                [sys.executable, "-m", "gnn_vuln.evaluate", "--checkpoint", str(checkpoint_path),
                 "--config", str(cfg_path), "--metrics-only"],
                check=True, cwd=str(settings.service_root), env=env,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        # metrics_summary.json is written under results_dir/<checkpoint-parent-name>/
        summary = next(job_dir.glob("**/metrics_summary.json"), None)
        if summary is None:
            raise RuntimeError(f"evaluation produced no metrics (see {log})")
        data = json.loads(summary.read_text(encoding="utf-8"))
        return {
            "job_id": job_id,
            "checkpoint_path": checkpoint_path,
            "metrics": self._scalar_metrics(data) if isinstance(data, dict) else {},
            "num_samples": len(dataset),
            "num_dropped": dropped,
        }

    def submit(
        self,
        *,
        method: str,
        base_config: dict[str, Any],
        data_source: str | None = None,
        source: str | None = None,
        dataset: list[dict] | None = None,
        dataset_bundle_uri: str | None = None,
        num_classes: int | None = None,
        epochs: int | None = None,
        base_checkpoint_path: str | None = None,
        base_class_names: list[str] | None = None,
        replay_source: str | None = None,
        device: str | None = None,
        model_version_id: str | None = None,
        run_name: str | None = None,
    ) -> RelearnJobState:
        if method not in VALID_METHODS:
            raise ValueError(f"Unknown method '{method}'. Allowed: {sorted(VALID_METHODS)}")
        if method in _REQUIRES_BASE and not base_checkpoint_path:
            raise ValueError(f"method '{method}' requires base_checkpoint_path")
        # The base checkpoint may be an s3:// pointer (model version source_uri) — pull it
        # to a local path so the trainer can stage it.
        if base_checkpoint_path and storage.is_s3_uri(base_checkpoint_path):
            base_checkpoint_path = str(storage.ensure_local(base_checkpoint_path))

        job_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        mode = (base_config.get("data") or {}).get("mode", "multiclass") if isinstance(base_config, dict) else "multiclass"
        materialized = False
        if dataset:
            effective_source = source or data_source or f"relearn_{job_id}"
            vocab_size = self._write_inline_dataset(effective_source, dataset, mode)
            data_source = effective_source
            materialized = True
            if num_classes is None and mode == "multiclass":
                # Weight-loading relearn (finetune/EWC/ER) must keep the base model's class
                # count so the checkpoint head loads; only size to the data vocab when
                # training from scratch (retrain) or when the base has no class count.
                cfg_nc = (base_config.get("model") or {}).get("num_classes") if isinstance(base_config, dict) else None
                if method != "retrain" and cfg_nc:
                    num_classes = int(cfg_nc)
                else:
                    num_classes = vocab_size
        elif dataset_bundle_uri:
            # Prepared bundle (.tar.gz of the gnn_vuln data/ layout): download + extract into
            # the data root and train directly from its processed .pt — no Joern/materialize.
            data_source = self._install_bundle(dataset_bundle_uri)
            materialized = True
        if not data_source:
            raise ValueError("an inline dataset, a dataset_bundle_uri, or a data_source is required")

        job = RelearnJobState(
            job_id=job_id,
            status="queued",
            method=method,
            data_source=data_source,
            base_checkpoint_path=base_checkpoint_path,
            log_path=str(job_dir / "run.log"),
            model_version_id=model_version_id,
            message=run_name,
        )
        train_cfg, importance_cfg = self._build_train_config(
            method=method,
            base_config=base_config,
            data_source=data_source,
            num_classes=num_classes,
            epochs=epochs,
            base_checkpoint_path=base_checkpoint_path,
            base_class_names=base_class_names,
            replay_source=replay_source,
            device=device,
            job_dir=job_dir,
            materialized=materialized,
        )
        job.config_path = str(train_cfg)
        self._save(job)
        threading.Thread(
            target=self._run_job, args=(job, train_cfg, importance_cfg), daemon=True
        ).start()
        return job


relearn_manager = RelearnManager()
