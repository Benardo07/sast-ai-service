"""Celery tasks for the AI service — thin orchestration around ``RelearnManager``.

The heavy work (the training/eval subprocess that drives the gnn_vuln library) lives in
``RelearnManager``; a task just resolves the persisted job and runs it inside the worker
process, so the API process never blocks on a multi-minute train. Job state is persisted
to disk under ``var/jobs/<job_id>/`` (a volume shared with the API container), which the
API reads back for ``GET /relearn/{job_id}`` — no shared in-memory state needed.
"""
from __future__ import annotations

from pathlib import Path

from app.celery_app import celery_app


@celery_app.task(name="run_relearn", bind=True)
def run_relearn(self, job_id: str, train_cfg: str, importance_cfg: str | None) -> dict:
    # Lazy import: RelearnManager pulls in the ML stack, which the API process already
    # holds — keep it out of the Celery app's import graph until a task actually runs.
    from app.relearn import relearn_manager

    job = relearn_manager.get(job_id)
    if job is None:
        return {"job_id": job_id, "status": "failed", "message": "job not found on worker"}
    relearn_manager.run_job(
        job,
        Path(train_cfg),
        Path(importance_cfg) if importance_cfg else None,
    )
    latest = relearn_manager.get(job_id)
    return {"job_id": job_id, "status": latest.status if latest else "unknown"}
