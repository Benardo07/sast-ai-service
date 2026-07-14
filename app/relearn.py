"""Relearn (continual-learning) jobs for the standalone AI service.

Ported and adapted from the original training repo. Key differences:
  * Trains via the installed ``gnn-vuln`` library package (its ``python -m`` module
    entrypoints), invoked in a Celery worker process — no vendored source tree.
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
    class_names: list[str] | None = None
    model_version_id: str | None = None
    message: str | None = None
    # Dataset-bundle export (durable .pt builds). `materialized` = the dataset was built under
    # this service's data root, so there is a .pt to export; `export_bundle_key` is the
    # backend's <dataset_key>-<build_key>. The last three are filled after a successful train.
    materialized: bool = False
    export_bundle_key: str | None = None
    exported_bundle_uri: str | None = None
    ds_name: str | None = None
    num_graphs: int | None = None
    # Durable EWC importance. `imported_importance` = submit() pre-seeded the base version's
    # cached importance into the stage dir, so this run computes nothing and exports nothing.
    # `exported_importance_uri` is filled only when we DID run the pass and uploaded the result.
    imported_importance: bool = False
    exported_importance_uri: str | None = None
    # Cumulative replay pool (>1 ancestor bundle). submit() does NOT install or merge these:
    # installing tens of thousands of lazy graphs + a merge is far outside the backend's 300s
    # `POST /relearn` timeout, and blowing it splits the brain (backend fails the run while this
    # service keeps training). submit() only DERIVES the pool name from the URIs and persists the
    # plan here; `_run_job_locked` installs + merges in the Celery worker, where there is no clock.
    # Both fields must survive the process hop to the worker (same reason as `imported_importance`).
    replay_pool_uris: list[str] | None = None
    replay_pool_source: str | None = None
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
        val_source: str | None = None,
        test_source: str | None = None,
        split: dict | None = None,
        importance_source: str | None = None,
        pool_merged: bool = False,
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
        # results_dir is per-job (NOT the shared results_root) so two jobs never read each
        # other's summary — _parse_metrics reads this job's own dir (review #11).
        cfg["train"]["results_dir"] = str(job_dir)

        # Continual-learning label alignment: when continuing a base model, remap task-B labels
        # onto the base model's class space so ids never clash. Known CWEs keep their canonical
        # id; brand-new CWEs (class-incremental) extend the head and num_classes grows to fit.
        # gnn_vuln applies the remap at load via data.target_vocab. retrain (no base) skips this.
        # Applied for ANY continued base (materialized inline/bundle OR a pre-staged data_source):
        # the vpath.exists() check below guards head extension, so a data_source-only job whose
        # raw/<source>/cwe_vocab.json exists is aligned too — its labels no longer clash (#13).
        if method != "retrain" and base_class_names:
            tv = {name: i for i, name in enumerate(base_class_names)}
            vpath = settings.data_root / "raw" / data_source / "cwe_vocab.json"
            if vpath.exists():
                for name in json.loads(vpath.read_text(encoding="utf-8")):
                    if name not in tv:
                        tv[name] = len(tv)          # new CWE -> extended id (head grows)
            cfg["data"]["target_vocab"] = tv
            cfg.setdefault("model", {})["num_classes"] = len(tv)

        # Train/val/test selection. Role mode (explicit val source) wins: the library sets
        # use_official = bool(source_val), trains on 100% of train and early-stops on val — so we
        # must NOT also set train_ratio/val_ratio. A test source is only honored WITH a val source
        # (library ignores test-only), so drop test_source if val_source is absent. Inline val/test
        # share the main featurization, so no source_val_params/source_test_params. Otherwise fall
        # back to the auto-split spec, setting only the keys the caller provided (library defaults
        # 0.8/0.1/42 fill the rest).
        if val_source:
            cfg["data"]["source_val"] = val_source
            if test_source:
                cfg["data"]["source_test"] = test_source
        elif split:
            if "train_ratio" in split:
                cfg["data"]["train_ratio"] = split["train_ratio"]
            if "val_ratio" in split:
                cfg["data"]["val_ratio"] = split["val_ratio"]
            if "seed" in split:
                cfg.setdefault("train", {})["seed"] = split["seed"]

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

            # Experience replay (Chaudhry 2019): `replay_source` is the base model's CUMULATIVE
            # lineage pool (merged in submit()), NOT the t-1 dataset — episodic memory spans every
            # past task, else the 3rd generation forgets task A.
            if method in _REPLAY_METHODS and replay_source:
                # A merged pool REBUILDS the vocab as ["benign"] + sorted(rest) (gnn_vuln
                # data/merge.py:72-74), while our head is FREQUENCY-ordered. Without an explicit
                # target_vocab the replay buffer would feed correctly-embedded graphs under WRONG
                # label ids — silently replaying the wrong CWEs, corrupting exactly what ER
                # protects. No best-effort here: fail the job.
                if pool_merged and not cfg["data"].get("target_vocab"):
                    raise ValueError(
                        "merged replay pool requires data.target_vocab (the base model's class "
                        "space): the merge rebuilds the label space alphabetically while the base "
                        "head is frequency-ordered, so replaying without it corrupts labels. "
                        "Pass base_class_names for this relearn, or send a single replay bundle."
                    )
                cfg["replay"] = {
                    "enabled": True,
                    "source": replay_source,
                    "buffer_per_class": 50,
                    "weight": 1.0,
                    "buffer_seed": 42,
                    # Explicit even though _setup_replay's `_rd` would inherit both from cfg.data —
                    # this is where a future reader looks, and the vocab one is load-bearing.
                    "storage": cfg["data"].get("storage", "inmemory"),
                    "target_vocab": cfg["data"].get("target_vocab"),
                }

            # EWC importance pass over the t-1 dataset ONLY (EWC-DR Eq(1)) — never the cumulative
            # pool: protection of older tasks is transitive via the anchor. Skipped when the
            # backend pre-seeded the base version's cached importance at `cache`.
            if method in _EWC_METHODS and importance_source and cache and not Path(cache).exists():
                imp = copy.deepcopy(cfg)
                imp.setdefault("data", {})["source"] = importance_source
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

    @staticmethod
    def _derive_class_names(train_cfg: Path, data_source: str) -> list[str] | None:
        """The trained model's ordered class list, so the backend can tag the new version
        (serving labels #1 + next relearn's base class space #8). Prefer the grown
        target_vocab written into the job's train config; fall back to the source's
        cwe_vocab.json. Returns None when neither is available (e.g. binary mode)."""
        try:
            cfg = yaml.safe_load(train_cfg.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            cfg = {}
        tv = (cfg.get("data") or {}).get("target_vocab") if isinstance(cfg, dict) else None
        if isinstance(tv, dict) and tv:
            names = [""] * (max(int(v) for v in tv.values()) + 1)
            for name, idx in tv.items():
                if 0 <= int(idx) < len(names):
                    names[int(idx)] = name
            return names
        vpath = settings.data_root / "raw" / data_source / "cwe_vocab.json"
        if vpath.exists():
            try:
                vocab = json.loads(vpath.read_text(encoding="utf-8"))
                if isinstance(vocab, dict) and vocab:
                    names = [""] * (max(int(v) for v in vocab.values()) + 1)
                    for name, idx in vocab.items():
                        if 0 <= int(idx) < len(names):
                            names[int(idx)] = name
                    return names
            except Exception:  # noqa: BLE001
                pass
        return None

    def _parse_metrics(self, checkpoint: Path, results_dir: Path | None = None) -> dict[str, Any] | None:
        """Best-effort: read training_summary.json + metrics_summary.json near the
        checkpoint / results dir, merging all scalar numeric metrics found."""
        search_dirs = [checkpoint.parent, results_dir, settings.results_root]
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
    def run_job(self, job: RelearnJobState, train_cfg: Path, importance_cfg: Path | None) -> None:
        # Runs inside the Celery worker process (see app.tasks.run_relearn). Serializing
        # concurrent trainings is now the worker's job (``--concurrency=1``) instead of an
        # in-process semaphore, so the heavy train runs in a SEPARATE process from the API —
        # inference is never starved of CPU by a training.
        self._run_job_locked(job, train_cfg, importance_cfg)

    def _run_job_locked(self, job: RelearnJobState, train_cfg: Path, importance_cfg: Path | None) -> None:
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
                # Cumulative replay pool: install the ancestor bundles + merge them HERE, where
                # there is no HTTP clock (submit() only planned it — see RelearnJobState). Must
                # precede the importance pass and the train: both read the pool by name. Any
                # failure propagates and fails the job — no fallback to single-bundle replay.
                if job.replay_pool_uris and len(job.replay_pool_uris) > 1:
                    self._install_and_merge_replay_pool(job, lf, log)
                if importance_cfg is not None:
                    lf.write(f"== EWC importance: {importance_cfg} ==\n")
                    lf.flush()
                    subprocess.run(
                        [sys.executable, "-m", "gnn_vuln.train", "--config", str(importance_cfg)],
                        check=True, cwd=str(settings.service_root), env=env,
                        stdout=lf, stderr=subprocess.STDOUT,
                    )
                    # We just paid for the importance pass over the task-A dataset, so publish it:
                    # it is a pure function of the BASE version's immutable (checkpoint, dataset),
                    # and the backend caches the URI on that version so no future EWC relearn off
                    # the same base pays again. Done here rather than after training — the
                    # importance is valid whether or not the subsequent train succeeds.
                    # Best-effort, like the checkpoint upload below: an upload failure must never
                    # fail an otherwise-successful job; the next run simply recomputes.
                    cache = self._job_dir(job.job_id) / "base" / "ewc_importance.pt"
                    if cache.exists() and not job.imported_importance and storage.using_s3():
                        try:
                            job.exported_importance_uri = storage.put_file(
                                settings.s3_bucket_checkpoints,
                                f"importance/{job.job_id}.ewc_importance.pt",
                                cache,
                            )
                            lf.write(f"== EWC importance exported: {job.exported_importance_uri} ==\n")
                        except Exception as e:  # noqa: BLE001
                            lf.write(
                                f"WARN EWC importance upload to object storage failed: "
                                f"{type(e).__name__}: {e}\n"
                            )
                        lf.flush()
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
                job.metrics = self._parse_metrics(best, self._job_dir(job.job_id))
                # The trained model's ordered class list — the backend tags the new version
                # with it (correct serving CWE labels + base class space for the next relearn).
                job.class_names = self._derive_class_names(train_cfg, job.data_source)
                # For retrain, drop a cwe_vocab.json next to the checkpoint so the class space
                # is recoverable straight from artifacts (review #8).
                if job.class_names:
                    try:
                        (best.parent / "cwe_vocab.json").write_text(
                            json.dumps({n: i for i, n in enumerate(job.class_names)}, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:  # noqa: BLE001
                        pass
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
                        # Ship the class vocab alongside so a redeployed node can recover the
                        # class space from artifacts alone (review #8).
                        vocab_sibling = best.parent / "cwe_vocab.json"
                        if vocab_sibling.exists():
                            storage.put_file(settings.s3_bucket_checkpoints, f"{prefix}/cwe_vocab.json", vocab_sibling)
                        job.result_checkpoint_path = uri
                    except Exception as e:  # noqa: BLE001 - keep local path if upload fails
                        with open(log, "a", encoding="utf-8") as lf:
                            lf.write(f"WARN checkpoint upload to object storage failed: {type(e).__name__}: {e}\n")
                # Durable .pt build: push the dataset we just embedded to object storage so the
                # next relearn on the same raw set + featurization reuses it (the CPGs are cached,
                # the unixcoder embedding pass is not). Best-effort — never fails a good train.
                if job.export_bundle_key and job.materialized:
                    self._export_dataset_bundle(job, train_cfg, log)
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

    def _export_dataset_bundle(self, job: RelearnJobState, train_cfg: Path, log: Path) -> None:
        """Inverse of `_install_bundle`: tar.gz the built gnn_vuln dataset and upload it to
        `<datasets>/builds/<export_bundle_key>.tar.gz`, recording the uri + ds_name (+ n_graphs)
        on the job. Emits exactly the layout _install_bundle consumes:
            data/raw/<source>/cwe_vocab.json          (also how it recovers the source name)
            data/processed/<ds_name>_meta.pt + <ds_name>_graphs/**   (lazy storage)
            data/processed/<ds_name>.pt                              (inmemory storage)
        Best-effort: any failure is logged and leaves the successful job untouched."""
        import shutil
        import tarfile
        import tempfile

        def _warn(msg: str) -> None:
            with open(log, "a", encoding="utf-8") as lf:
                lf.write(f"WARN dataset bundle export: {msg}\n")

        try:
            source = job.data_source
            processed = settings.data_root / "processed"
            try:
                cfg = yaml.safe_load(train_cfg.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001
                cfg = {}
            mode = (cfg.get("data") or {}).get("mode", "multiclass")
            # Pin the mode segment right after the source: the role sources are named
            # <source>_val / <source>_test, so a bare `lm_dataset_<source>_*` would match them too.
            newest = lambda ps: sorted(ps, key=lambda p: p.stat().st_mtime, reverse=True)  # noqa: E731
            metas = newest(processed.glob(f"lm_dataset_{source}_{mode}_*_meta.pt"))
            if metas:
                lazy, ds_name = True, metas[0].name[: -len("_meta.pt")]
            else:
                flat = newest(
                    p for p in processed.glob(f"lm_dataset_{source}_{mode}_*.pt")
                    if not p.name.endswith("_meta.pt")
                )
                if not flat:
                    _warn(f"no built .pt for source '{source}' under {processed} — skipped")
                    return
                lazy, ds_name = False, flat[0].stem

            num_graphs: int | None = None
            if lazy:  # the meta is a tiny {"n_graphs", "class_names"} dict; the inmemory .pt is not
                try:
                    import torch

                    meta = torch.load(processed / f"{ds_name}_meta.pt", weights_only=False)
                    if isinstance(meta, dict) and meta.get("n_graphs") is not None:
                        num_graphs = int(meta["n_graphs"])
                except Exception:  # noqa: BLE001
                    num_graphs = None

            # s3 mode: a temp tar, deleted right after upload (bundles are GBs, VM disk is tight).
            # fs mode: put_file returns the path itself, so it must survive — keep it under data_root.
            if storage.using_s3():
                tar_dir = Path(tempfile.mkdtemp(prefix="ds_bundle_"))
            else:
                tar_dir = settings.data_root / "bundles"
                tar_dir.mkdir(parents=True, exist_ok=True)
            tar_path = tar_dir / f"{job.export_bundle_key}.tar.gz"
            try:
                with tarfile.open(tar_path, "w:gz") as tf:
                    vocab = settings.data_root / "raw" / source / "cwe_vocab.json"
                    if vocab.exists():
                        tf.add(vocab, arcname=f"data/raw/{source}/cwe_vocab.json")
                    else:
                        _warn(f"no cwe_vocab.json for source '{source}' (source inferred on install)")
                    if lazy:
                        tf.add(processed / f"{ds_name}_meta.pt", arcname=f"data/processed/{ds_name}_meta.pt")
                        tf.add(processed / f"{ds_name}_graphs", arcname=f"data/processed/{ds_name}_graphs")
                    else:
                        tf.add(processed / f"{ds_name}.pt", arcname=f"data/processed/{ds_name}.pt")
                uri = storage.put_file(
                    settings.s3_bucket_datasets, f"builds/{job.export_bundle_key}.tar.gz", tar_path
                )
            finally:
                if storage.using_s3():
                    shutil.rmtree(tar_dir, ignore_errors=True)
            job.exported_bundle_uri = uri
            job.ds_name = ds_name
            job.num_graphs = num_graphs
            with open(log, "a", encoding="utf-8") as lf:
                lf.write(f"== dataset bundle exported: {uri} (ds_name={ds_name}, n_graphs={num_graphs}) ==\n")
        except Exception as e:  # noqa: BLE001 - export never fails a successful training
            try:
                _warn(f"{type(e).__name__}: {e}")
            except Exception:  # noqa: BLE001
                pass

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

    @staticmethod
    def _pool_source_name(uris: list[str]) -> str:
        """The cumulative replay pool's source name, derived from the ordered ancestor bundle
        URIs — NOT from their installed source names, which are only knowable after downloading
        multi-GB tarballs. That is the whole point: submit() must name the pool without paying
        for it, so the name it writes into `train.yaml`'s `replay.source` and the name the worker
        later passes to `--out-source` are the same string by construction.

        The pool is ordered, so the hash is order-sensitive. No underscore in the name:
        `_ds_name` is `lm_dataset_{source}_{mode}_...` and `_install_bundle`'s fallback splits on
        "_" — same rule as `dsv-`.
        """
        import hashlib

        return "pool-" + hashlib.sha256("|".join(uris).encode()).hexdigest()[:12]

    def _write_merge_config(self, cfg: dict[str, Any], job_dir: Path) -> Path:
        """Write `replay_merge.yaml` — the cfg half of the merge. Called from submit() (cheap: it
        only reshuffles the base config, no I/O beyond one small file) so a config that can never
        merge fails the HTTP request instead of a queued job. The `--sources` / `--out-source` half
        is CLI args the worker supplies once the bundles are actually installed."""
        data_cfg = cfg.get("data") or {}

        # gnn_vuln 0.1.15 cannot merge cwe_list/cwe_groups-filtered datasets: both `_build_ds` and
        # `_out_processed_path` (merge.py:33-47,157) drop these two when deriving `_ds_name`
        # (`_filter_suffix(None, None, ...)`), while the real dataset — and therefore
        # `_setup_replay` — folds them into `_fsuffix` (dataset_lm.py:423). The merge would read
        # and write .pt names the trainer then cannot resolve, and die rebuilding from raw CPG.
        # filter_owasp/filter_top25_dangerous ARE forwarded and merge fine. Fail early and clearly.
        if data_cfg.get("cwe_list") or data_cfg.get("cwe_groups"):
            raise ValueError(
                "cannot merge a cumulative replay pool for a config with data.cwe_list or "
                "data.cwe_groups: gnn_vuln's merge ignores both when resolving the dataset name, "
                "so the merged pool would be unreadable by the trainer. Send a single "
                "replay_bundle_uri for this model, or drop the cwe_list/cwe_groups filter."
            )

        # Mirror cfg's NAME-DETERMINING params exactly: merge resolves the output path via the same
        # `_ds_name` that `_setup_replay` (through `_rd`'s cfg.data fallback) will resolve when it
        # loads the pool. Any drift here and the trainer looks for a file the merge never wrote.
        # Keys absent from cfg stay absent, so both sides take the same library default.
        model_cfg = cfg.get("model") or {}
        merge_cfg: dict[str, Any] = {
            "data": {
                k: data_cfg[k]
                for k in (
                    "mode", "max_nodes", "top_cwe", "cwe_list", "cwe_groups", "filter_owasp",
                    "filter_top25_dangerous", "max_per_class", "resample_seed", "storage",
                    "ds_name_suffix",
                )
                if k in data_cfg
            },
            "model": {
                k: model_cfg[k]
                for k in ("pretrained_lm", "func_lm", "func_lm_source", "add_func_tokens",
                          "func_max_length")
                if k in model_cfg
            },
        }
        merge_cfg["data"]["raw_dir"] = str(settings.data_root / "raw")       # merge writes the
        merge_cfg["data"]["processed_dir"] = str(settings.data_root / "processed")  # unified vocab
        merge_yaml = job_dir / "replay_merge.yaml"
        merge_yaml.write_text(yaml.safe_dump(merge_cfg, sort_keys=False), encoding="utf-8")
        return merge_yaml

    def _install_and_merge_replay_pool(self, job: RelearnJobState, lf: Any, log: Path) -> None:
        """Worker-side half of the cumulative replay pool: install the ancestor bundles and merge
        them into `job.replay_pool_source`. Runs in the Celery worker, NOT in submit() — installing
        N multi-GB bundles (a torch.load + torch.save per lazy graph) cannot fit the backend's 300s
        `POST /relearn` timeout.

        The out-source is READ FROM THE JOB, never recomputed: submit() already wrote that exact
        string into `train.yaml`'s `replay.source`, and the trainer will look for a pool under
        precisely that name. Recomputing the hash here would be a second source of truth.

        Raises on any failure — a job whose pool did not merge must fail loudly rather than fall
        back to single-bundle replay, which would silently restore the 1-generation forgetting this
        pool exists to fix and quietly invalidate the run's numbers.
        """
        uris = job.replay_pool_uris or []
        out_source = job.replay_pool_source
        if not out_source:
            raise RuntimeError("replay pool requested but job.replay_pool_source is unset")
        merge_yaml = self._job_dir(job.job_id) / "replay_merge.yaml"
        if not merge_yaml.exists():
            raise RuntimeError(f"replay pool merge config missing: {merge_yaml}")

        lf.write(f"== replay pool {out_source}: installing {len(uris)} ancestor bundle(s) ==\n")
        lf.flush()
        sources: list[str] = []
        for uri in uris:
            s = self._install_bundle(uri)
            if s not in sources:            # dedup source names, preserve pool order
                sources.append(s)
            lf.write(f"== installed {uri} -> {s} ==\n")
            lf.flush()
        self._merge_replay_pool(sources, out_source, merge_yaml, lf, log)

    def _merge_replay_pool(
        self, sources: list[str], out_source: str, merge_yaml: Path, lf: Any, log: Path
    ) -> str:
        """Merge the base model's ordered ancestor datasets into ONE cumulative replay pool at the
        processed .pt level (gnn_vuln.data.merge — concatenate finished .pt + unify the vocab; no
        Joern, no re-embedding) under the caller-supplied `out_source`."""
        try:
            merge_cfg = yaml.safe_load(merge_yaml.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            merge_cfg = {}
        mode = (merge_cfg.get("data") or {}).get("mode", "multiclass")
        processed = settings.data_root / "processed"

        # Preflight: merge's `_build_ds` LOADS each source, but a miss silently falls through to
        # BUILDING it from raw CPG — i.e. the Joern wall, failing minutes later with a confusing
        # error. Lives here (not in submit) because it needs the sources INSTALLED to check them.
        # Mode-pinned glob (a bare `<source>_*` also matches `<source>_val`/`<source>_test`).
        for s in sources:
            if not any(processed.glob(f"lm_dataset_{s}_{mode}_*")):
                raise ValueError(
                    f"replay pool source '{s}' has no built dataset under {processed} "
                    f"(expected lm_dataset_{s}_{mode}_*.pt or _meta.pt). Its bundle installed no "
                    f"processed .pt, so merging it would try to rebuild from raw CPG."
                )

        # Idempotent: the pool is a pure function of (ordered sources, featurization), both of which
        # the output path encodes — so an existing one is reusable as-is. Resolved through the
        # library's own helper off the yaml submit() wrote (the same load the subprocess does), so
        # the check can never disagree with the merge. Best-effort: if it throws, just merge again.
        try:
            from gnn_vuln.config import Config
            from gnn_vuln.data.merge import _out_processed_path

            out_path = _out_processed_path(
                settings.data_root, out_source, Config.from_yamls([str(merge_yaml)])
            )
            if out_path.exists():
                lf.write(f"== replay pool {out_source} already merged ({out_path}) — reused ==\n")
                lf.flush()
                return out_source
        except Exception as e:  # noqa: BLE001
            lf.write(f"WARN replay pool reuse check failed ({type(e).__name__}: {e}) — merging\n")
            lf.flush()

        # `--dedup` drops duplicate functions by `raw_func` hash (merge.py:48-52); a graph without
        # `raw_func` is silently kept. Fine here: `_write_inline_dataset` always writes raw_func and
        # lazy `torch.save(g, ...)` round-trips the whole Data.
        lf.write(f"== merge replay pool -> {out_source} from {sources} ==\n")
        lf.flush()
        try:
            subprocess.run(
                [sys.executable, "-m", "gnn_vuln.data.merge", "--config", str(merge_yaml),
                 "--sources", *sources, "--out-source", out_source, "--dedup"],
                check=True, cwd=str(settings.service_root),
                env={**os.environ, "GNN_VULN_API_MODE": "1"},
                stdout=lf, stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            # NO fallback to single-bundle replay: that would silently restore the 1-generation
            # behavior this pool exists to fix, and quietly invalidate the run's numbers.
            lf.flush()   # the tail we are about to read is still in our own buffer
            raise RuntimeError(
                f"replay pool merge failed (exit {e.returncode}) for sources {sources}:\n"
                f"{self._log_tail(log)}"
            ) from e
        return out_source

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
            # `raw_func` (function source) is the func-LM branch input for hybrid_graph_lm &
            # sequential — written for BOTH vulnerable and benign, else those heads see empty
            # text. `row_id` = sample_uid keeps per-sample provenance back to the platform.
            raw_func = e.get("code") or ""
            sample_uid = e.get("sample_uid")
            if vuln:
                cwe = e.get("cwe") or "UNKNOWN"
                if mode == "multiclass":
                    if cwe not in vocab:
                        vocab[cwe] = next_id
                        next_id += 1
                    class_id = vocab[cwe]
                else:
                    class_id = 1
                meta = {"class_id": class_id, "cwe": cwe, "flaw_lines": e.get("flaw_lines") or [],
                        "raw_func": raw_func}
            else:
                meta = {"class_id": 0, "cwe": "benign", "flaw_lines": [], "raw_func": raw_func}
            if sample_uid:
                meta["row_id"] = sample_uid
            (base / sub / f"{fname}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (base / "cwe_vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
        return len(vocab)

    # Cap on embedding rows returned for the drift baseline: MMD needs ~100 vectors; a full
    # 1000×D matrix bloats the /evaluate response. Confidence/error are 1 float each, kept whole.
    _BASELINE_EMB_CAP = 600

    def _read_baseline_from_artifacts(self, results_dir: Path) -> dict[str, Any] | None:
        """Assemble the drift-baseline signals from a FULL evaluate's research artifacts:
        per-sample confidence + error (1=wrong/0=correct) from predictions.csv, and the
        pre-head function embeddings (capped) from embeddings.npz. Returns None when no
        predictions.csv was produced (e.g. a metrics-only/test-empty run)."""
        import csv

        import numpy as np

        pred_csv = next(results_dir.glob("**/predictions.csv"), None)
        if pred_csv is None:
            return None
        confidence: list[float] = []
        error: list[int] = []
        correct: list[bool] = []
        with open(pred_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    confidence.append(float(row["confidence"]))
                except (KeyError, ValueError):
                    continue
                is_correct = str(row.get("correct", "")).strip().lower() in ("true", "1")
                error.append(0 if is_correct else 1)
                correct.append(is_correct)
        if not confidence:
            return None

        emb_list: list[list[float]] = []
        npz = next(results_dir.glob("**/embeddings.npz"), None)
        if npz is not None:
            try:
                arr = np.load(npz)["embeddings"]
                if arr.ndim == 2 and arr.shape[0] > 0:
                    emb_list = [[float(v) for v in r] for r in arr[: self._BASELINE_EMB_CAP].tolist()]
            except Exception:  # noqa: BLE001 - embeddings are best-effort
                emb_list = []

        return {
            "num_samples": len(confidence),
            "accuracy": (sum(correct) / len(correct)) if correct else None,
            "confidence": confidence,
            "error": error,
            "embeddings": emb_list,
            "embedding_dim": (len(emb_list[0]) if emb_list else 0),
        }

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

        # The model's ACTUAL class space. A relearned checkpoint grows its head and stores the
        # grown target_vocab in its sibling config.yaml (staged next to the .pt by ensure_local);
        # the caller's base_class_names (from the base ConfigVersion) is stale for it. Prefer the
        # checkpoint's own config when present, else the caller-supplied class names.
        class_names = list(base_class_names) if base_class_names else None
        sibling_cfg = Path(checkpoint_path).parent / "config.yaml"
        if sibling_cfg.exists():
            try:
                sc = yaml.safe_load(sibling_cfg.read_text(encoding="utf-8")) or {}
                tv = (sc.get("data") or {}).get("target_vocab")
                if isinstance(tv, dict) and tv:
                    names = [""] * (max(int(v) for v in tv.values()) + 1)
                    for name, idx in tv.items():
                        if 0 <= int(idx) < len(names):
                            names[int(idx)] = name
                    class_names = names
            except Exception:  # noqa: BLE001 - fall back to base_class_names
                pass

        # EVALUATION keeps the model's head FIXED at the checkpoint's classes — unlike relearn,
        # it never grows the head (that would desync from the checkpoint weights → load error).
        # A benchmark CWE the model was never trained on is unanswerable by a fixed-class model,
        # so drop those vulnerable samples (benign is always in-vocab). Their count is reported.
        dropped = 0
        if class_names and mode == "multiclass":
            known = set(class_names)
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
        if class_names and mode == "multiclass":
            cfg["data"]["target_vocab"] = {name: i for i, name in enumerate(class_names)}
            cfg.setdefault("model", {})["num_classes"] = len(class_names)

        cfg_path = job_dir / "eval.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        log = job_dir / "eval.log"
        # FULL research path (NOT --metrics-only, and WITHOUT GNN_VULN_API_MODE): the library
        # writes predictions.csv + embeddings.npz alongside metrics_summary.json. Those ARE the
        # drift-baseline signals — per-sample confidence + correctness (predictions.csv) and the
        # pre-head function vectors (embeddings.npz). embeddings.npz holds the SAME vector
        # predict() returns as cls_embedding, so the MMD baseline is directly comparable to the
        # embeddings production inference produces (apples-to-apples).
        env = {k: v for k, v in os.environ.items() if k != "GNN_VULN_API_MODE"}
        with open(log, "a", encoding="utf-8") as lf:
            subprocess.run(
                [sys.executable, "-m", "gnn_vuln.evaluate", "--checkpoint", str(checkpoint_path),
                 "--config", str(cfg_path)],
                check=True, cwd=str(settings.service_root), env=env,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        # metrics_summary.json is written under results_dir/<checkpoint-parent-name>/
        summary = next(job_dir.glob("**/metrics_summary.json"), None)
        if summary is None:
            raise RuntimeError(f"evaluation produced no metrics (see {log})")
        data = json.loads(summary.read_text(encoding="utf-8"))
        # Drift-baseline signals, assembled from the research artifacts the full eval just wrote
        # (predictions.csv + embeddings.npz). Best-effort: any gap must not fail the evaluation.
        baseline = None
        try:
            baseline = self._read_baseline_from_artifacts(job_dir)
        except Exception:  # noqa: BLE001
            baseline = None
        return {
            "job_id": job_id,
            "checkpoint_path": checkpoint_path,
            "metrics": self._scalar_metrics(data) if isinstance(data, dict) else {},
            "num_samples": len(dataset),
            "num_dropped": dropped,
            "baseline": baseline,
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
        replay_bundle_uri: str | None = None,
        replay_bundle_uris: list[str] | None = None,
        ewc_importance_uri: str | None = None,
        device: str | None = None,
        model_version_id: str | None = None,
        run_name: str | None = None,
        val_dataset: list[dict] | None = None,
        val_source: str | None = None,
        val_dataset_bundle_uri: str | None = None,
        test_dataset: list[dict] | None = None,
        test_source: str | None = None,
        test_dataset_bundle_uri: str | None = None,
        split: dict | None = None,
        export_bundle_key: str | None = None,
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

        # Role datasets (manual split): materialize VAL/TEST exactly like the main dataset. They
        # share the main featurization (no source_*_params), so their labels align via target_vocab.
        val_src: str | None = None
        if val_dataset:
            val_src = val_source or f"{data_source}_val"
            self._write_inline_dataset(val_src, val_dataset, mode)
        elif val_dataset_bundle_uri:
            val_src = self._install_bundle(val_dataset_bundle_uri)
        test_src: str | None = None
        if test_dataset:
            test_src = test_source or f"{data_source}_test"
            self._write_inline_dataset(test_src, test_dataset, mode)
        elif test_dataset_bundle_uri:
            test_src = self._install_bundle(test_dataset_bundle_uri)

        # TWO DISTINCT SOURCES — do not collapse them:
        #   importance_source = the t-1 base dataset, ALWAYS a single bundle. EWC-DR Eq(1) takes
        #     the Fisher from task t-1 only; older tasks are protected transitively via the anchor.
        #     Computing it over the cumulative pool would violate the method.
        #   replay_source     = the base's CUMULATIVE lineage (Chaudhry 2019 episodic memory over
        #     ALL past tasks), merged from the ancestor bundles. With only t-1, the 3rd generation
        #     loses task A.
        # Both are installed from durable bundles derived by the backend from the champion being
        # relearned, and override any manual replay_source. No-op for finetune/retrain.
        # Captured BEFORE replay_source is reassigned to the pool below: a caller that passes only
        # a manual replay_source (no bundles) keeps today's behavior — that source drives the
        # importance pass. A bundle overrides it.
        importance_source: str | None = (
            replay_source if method in (_REPLAY_METHODS | _EWC_METHODS) else None
        )
        # The t-1 bundle is normally also the pool's last entry — install each URI at most once
        # (they are multi-GB tarballs to fetch and extract).
        _installed: dict[str, str] = {}

        def _install(uri: str) -> str:
            if uri not in _installed:
                _installed[uri] = self._install_bundle(uri)
            return _installed[uri]

        if replay_bundle_uri and method in (_REPLAY_METHODS | _EWC_METHODS):
            importance_source = _install(replay_bundle_uri)

        # Cumulative replay pool. submit() runs INSIDE the backend's 300s `POST /relearn` request,
        # so it must stay cheap: installing the ancestor bundles (tens of thousands of lazy graphs,
        # a torch.load + torch.save each, times N ancestors) and merging them would blow that budget
        # and split the brain — the backend would time out and fail the run while this service kept
        # working. So the >1 case only PLANS the pool here and defers the work to the worker.
        pool_merged = False
        replay_pool_uris: list[str] | None = None
        replay_pool_source: str | None = None
        if method in _REPLAY_METHODS:
            uris = list(dict.fromkeys(replay_bundle_uris or []))     # dedup, preserve pool order
            if not uris and replay_bundle_uri:
                uris = [replay_bundle_uri]                           # single-ancestor / legacy caller
            if len(uris) > 1:
                # Name the pool from the URIs, not the installed source names: the latter are only
                # knowable by downloading the bundles, which is exactly what we are deferring.
                replay_pool_uris = uris
                replay_pool_source = self._pool_source_name(uris)
                pool_merged = True
                # Fails the request now if the config can never merge (cwe_list/cwe_groups).
                self._write_merge_config(base_config, job_dir)
            elif uris:
                # Single bundle: unchanged — install here and use its source name.
                replay_source = _install(uris[0])
            # else: no bundles — keep the caller-supplied manual replay_source as-is.

        # Durable EWC importance: the base version's Fisher/importance is a pure function of
        # (base checkpoint, base replay dataset, scope="all", n_batches=0) — all immutable — so the
        # backend caches it per version and hands it back here. Pre-seed it at exactly the path
        # `_build_train_config` puts in cfg.ewc.importance_cache; that function's cache-exists guard
        # then skips emitting importance.yaml, and gnn_vuln's train.py::_setup_ewc loads it via
        # EWCDR.from_file instead of running the ~1h pass over the task-A dataset.
        imported_importance = False
        if ewc_importance_uri and method in _EWC_METHODS:
            # MUST match `_build_train_config`'s stage_dir (`job_dir / "base"`) — that coupling is
            # load-bearing: a mismatch silently re-runs the pass. `mkdir(exist_ok=True)` there makes
            # pre-creating it safe.
            try:
                stage_dir = job_dir / "base"
                stage_dir.mkdir(parents=True, exist_ok=True)
                import shutil

                local = storage.ensure_local(ewc_importance_uri)  # downloads if s3://, else resolves
                shutil.copy2(local, stage_dir / "ewc_importance.pt")
                imported_importance = True
            except Exception as e:  # noqa: BLE001 - never fail a job over a cache miss
                # Worst case the importance pass just runs again — correct, only slower. Warn into
                # a sibling of run.log, which `_run_job_locked` opens with "w" (would truncate this).
                try:
                    (job_dir / "submit.log").write_text(
                        f"WARN EWC importance cache fetch failed ({ewc_importance_uri}): "
                        f"{type(e).__name__}: {e}\n",
                        encoding="utf-8",
                    )
                except Exception:  # noqa: BLE001
                    pass

        job = RelearnJobState(
            job_id=job_id,
            status="queued",
            method=method,
            data_source=data_source,
            base_checkpoint_path=base_checkpoint_path,
            log_path=str(job_dir / "run.log"),
            model_version_id=model_version_id,
            message=run_name,
            materialized=materialized,
            export_bundle_key=export_bundle_key,
            imported_importance=imported_importance,
            replay_pool_uris=replay_pool_uris,
            replay_pool_source=replay_pool_source,
        )
        # THE INVARIANT: the pool name written into `train.yaml`'s `replay.source` here must be
        # byte-identical to the `--out-source` the worker later merges to, or the trainer looks for
        # a pool that does not exist. Both sides read `job.replay_pool_source` — one hash, computed
        # once, carried on the job across the process hop. Never recompute it.
        replay_source_final = job.replay_pool_source or replay_source
        train_cfg, importance_cfg = self._build_train_config(
            method=method,
            base_config=base_config,
            data_source=data_source,
            num_classes=num_classes,
            epochs=epochs,
            base_checkpoint_path=base_checkpoint_path,
            base_class_names=base_class_names,
            replay_source=replay_source_final,
            device=device,
            job_dir=job_dir,
            materialized=materialized,
            val_source=val_src,
            test_source=test_src,
            split=split,
            importance_source=importance_source,
            pool_merged=pool_merged,
        )
        job.config_path = str(train_cfg)
        self._save(job)
        # Hand the heavy training off to the Celery worker (a SEPARATE process). The job spec is
        # already persisted to disk, so the worker resolves it by id and the API polls status
        # from disk via GET /relearn/{job_id}. Lazy import avoids a circular import at module load.
        from app.tasks import run_relearn as _run_relearn_task

        _run_relearn_task.delay(
            job.job_id, str(train_cfg), str(importance_cfg) if importance_cfg else None
        )
        return job


relearn_manager = RelearnManager()
