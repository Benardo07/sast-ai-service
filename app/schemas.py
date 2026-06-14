from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str = "sast-ai-service"
    active_model_version_id: str | None = None
    active_checkpoint_path: str | None = None
    loaded_at: datetime | None = None
    detail: str | None = None


class LoadReleaseRequest(BaseModel):
    model_version_id: str | None = None
    checkpoint_path: str
    config_path: str | None = None
    device: str | None = None
    force_reload: bool = False


class ActiveReleaseResponse(BaseModel):
    model_version_id: str | None = None
    checkpoint_path: str | None = None
    config_path: str | None = None
    device: str | None = None
    architecture: str | None = None
    data_source: str | None = None
    data_mode: str | None = None
    num_classes: int | None = None
    loaded_at: datetime | None = None
    ready: bool
    source_repo_root: str


class PredictRequest(BaseModel):
    cpg_path: str = Field(..., description="Path to a Joern-exported CPG JSON file")
    top_k_lines: int | None = 10
    max_nodes: int = 2500
    label: int = 0
    flaw_lines: list[int] | None = None


class PredictResponse(BaseModel):
    model_version_id: str | None = None
    checkpoint_path: str
    config_path: str
    result: dict
    predicted_at: datetime
