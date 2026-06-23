"""Pluggable artifact storage for the AI service — local filesystem (dev) or S3/MinIO.

The service is a **stateless compute worker**: checkpoints it trains are pushed to object
storage, and a checkpoint it must serve is pulled down on demand. This removes the
shared-disk assumption so the service can run as an ephemeral / multi-instance deployment
(a worker can load a model it never trained — only the buckets are durable + shared).

Same boto3 path for AWS S3 and self-hosted MinIO; only ``SAST_AI_S3_ENDPOINT`` differs.
Selected via ``SAST_AI_STORAGE_BACKEND`` (fs|s3). ``boto3`` is imported lazily.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.config import settings

_S3_PREFIX = "s3://"


class StorageError(RuntimeError):
    pass


def is_s3_uri(uri: str | None) -> bool:
    return isinstance(uri, str) and uri.startswith(_S3_PREFIX)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not is_s3_uri(uri):
        raise ValueError(f"Not an s3 uri: {uri!r}")
    bucket, _, key = uri[len(_S3_PREFIX):].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri (need bucket and key): {uri!r}")
    return bucket, key


def using_s3() -> bool:
    return settings.storage_backend.strip().lower() == "s3"


def _client():
    try:
        import boto3  # lazy: only needed in s3 mode
    except ImportError as exc:  # pragma: no cover - deploy-only path
        raise StorageError("boto3 is required for STORAGE_BACKEND=s3 (uv add boto3)") from exc
    kwargs: dict = {
        "aws_access_key_id": settings.s3_access_key,
        "aws_secret_access_key": settings.s3_secret_key,
        "region_name": settings.s3_region,
    }
    if settings.s3_endpoint:
        kwargs["endpoint_url"] = settings.s3_endpoint
    return boto3.client("s3", **kwargs)


def _ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:  # noqa: BLE001
        try:
            client.create_bucket(Bucket=bucket)
        except Exception:  # pragma: no cover
            pass


def put_file(bucket: str, key: str, local_path: str | Path) -> str:
    """Upload a local file; returns ``s3://bucket/key`` (s3 mode) or the path (fs mode)."""
    local_path = Path(local_path)
    if using_s3():
        client = _client()
        _ensure_bucket(client, bucket)
        client.upload_file(str(local_path), bucket, key)
        return f"{_S3_PREFIX}{bucket}/{key}"
    return str(local_path.resolve())


def _download(client, bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    client.download_file(bucket, key, str(tmp))
    tmp.replace(dest)  # atomic: avoid a half-written cache entry


def ensure_local(uri: str) -> Path:
    """Return a LOCAL path for an artifact pointer. For an ``s3://`` URI the blob is
    downloaded into a per-artifact cache dir (idempotent by URI hash); the sibling
    ``config.yaml`` (same key prefix) is fetched too so the loader's config-sibling
    lookup keeps working. For a normal path it is returned as-is. ``/load-release`` calls
    this so it can load a checkpoint regardless of where it physically lives."""
    if not is_s3_uri(uri):
        return Path(uri)
    bucket, key = parse_s3_uri(uri)
    digest = hashlib.sha256(uri.encode()).hexdigest()[:16]
    art_dir = settings.checkpoint_cache_root / digest
    dest = art_dir / Path(key).name
    if not dest.exists():
        client = _client()
        _download(client, bucket, key, dest)
        # best-effort: bring the sibling config next to the cached checkpoint
        prefix = key.rsplit("/", 1)[0] if "/" in key else ""
        for cfg_name in ("config.yaml", "config.yml"):
            cfg_key = f"{prefix}/{cfg_name}" if prefix else cfg_name
            try:
                _download(client, bucket, cfg_key, art_dir / cfg_name)
                break
            except Exception:  # noqa: BLE001 - no sibling config in the bucket
                continue
    return dest
