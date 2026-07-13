"""Celery app for the AI service — runs long jobs (relearn/train) off the API process.

The worker is a SEPARATE process/container so the API container is reserved for
inference: the model forward stays responsive while a multi-minute training runs
elsewhere (this is what stops training from starving inference of CPU).

    celery -A app.celery_app worker --loglevel=info -Q ai_relearn --concurrency=1

Broker + result backend = Redis (settings.celery_broker_url / celery_result_backend),
on their OWN DBs (3/4) + a dedicated queue name, so this never collides with
sast-backend's Celery running on the same Redis server. Tasks live in ``app.tasks``;
the gnn_vuln LIBRARY knows nothing about Celery — tasks orchestrate, the library computes.
"""
from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "sast_ai_service",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_track_started=True,          # report STARTED, not just PENDING -> SUCCESS
    task_acks_late=True,              # re-deliver if a worker dies mid-task
    worker_prefetch_multiplier=1,     # one heavy job at a time per worker
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86_400,            # keep task results 1 day
    task_default_queue=settings.relearn_queue,
    task_routes={"run_relearn": {"queue": settings.relearn_queue}},
)
