from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8001
    backend_base_url: str = "http://127.0.0.1:8000"
    source_repo_root: str = "../tugas-akhir"
    active_release_file: str = "./var/active_release.json"
    default_device: str = "cpu"
    autoload_persisted_release: bool = False
    # Relearn/training working dirs (relative paths resolve under service_root).
    jobs_dir: str = "./var/jobs"
    checkpoints_dir: str = "./var/checkpoints"
    results_dir: str = "./var/results"
    data_dir: str = "./var/data"
    # Joern CLI dir for CPG generation (e.g. C:/joern/joern-cli). None = auto-detect.
    joern_cli: str | None = None
    # ── Artifact storage (production-ready: pull/push checkpoints from object storage) ──
    # "fs" = local filesystem (dev); "s3" = MinIO/AWS via boto3 (prod, no shared disk).
    storage_backend: str = "fs"
    s3_endpoint: str = ""  # e.g. http://localhost:9000 for MinIO; blank = AWS default
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_region: str = "us-east-1"
    s3_bucket_checkpoints: str = "checkpoints"
    # Built .pt dataset bundles (s3://datasets/builds/<key>.tar.gz) — reused by later relearns.
    s3_bucket_datasets: str = "datasets"
    # Local cache for checkpoints downloaded from object storage.
    checkpoint_cache_dir: str = "./var/checkpoint-cache"
    # Cap on cached checkpoint dirs (LRU-evicted by last-access). 0 = unbounded (review #16).
    checkpoint_cache_max_entries: int = 32
    # ── Async job queue (Celery) — long jobs (relearn/train) run in a SEPARATE worker
    # process so the API container stays reserved for inference (the model forward keeps
    # responding while a multi-minute training runs elsewhere). Its OWN Redis DBs (3/4) +
    # a dedicated queue name keep it isolated from sast-backend's Celery on the same Redis.
    celery_broker_url: str = "redis://localhost:6379/3"
    celery_result_backend: str = "redis://localhost:6379/4"
    relearn_queue: str = "ai_relearn"

    model_config = {
        "env_prefix": "SAST_AI_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @property
    def service_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def workspace_root(self) -> Path:
        return self.service_root.parent.parent

    @property
    def source_repo_path(self) -> Path:
        path = Path(self.source_repo_root)
        if not path.is_absolute():
            path = (self.service_root / path).resolve()
        else:
            path = path.resolve()
        return path

    @property
    def active_release_path(self) -> Path:
        path = Path(self.active_release_file)
        if not path.is_absolute():
            path = (self.service_root / path).resolve()
        else:
            path = path.resolve()
        return path

    def _resolve(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = (self.service_root / path).resolve()
        return path

    @property
    def jobs_root(self) -> Path:
        return self._resolve(self.jobs_dir)

    @property
    def checkpoints_root(self) -> Path:
        return self._resolve(self.checkpoints_dir)

    @property
    def results_root(self) -> Path:
        return self._resolve(self.results_dir)

    @property
    def data_root(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def checkpoint_cache_root(self) -> Path:
        return self._resolve(self.checkpoint_cache_dir)


settings = Settings()
