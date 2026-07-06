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


class ValidateModelRequest(BaseModel):
    checkpoint_path: str
    config_path: str | None = None
    device: str | None = None


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
    # Provide EITHER a function source string (Joern runs internally) OR a prebuilt CPG path.
    code: str | None = Field(default=None, description="Function source string")
    language: str | None = None
    cpg_path: str | None = Field(default=None, description="Path to a Joern-exported CPG JSON file")
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


class RelearnDatasetEntry(BaseModel):
    cpg_json: dict
    is_vulnerable: bool = False
    cwe: str | None = None
    flaw_lines: list[int] = []
    func_name: str | None = None
    sample_uid: str | None = None

    model_config = {"extra": "allow"}


class RelearnRequest(BaseModel):
    method: str = Field(..., description="finetune | EWC | ER | EWC-ER | retrain")
    base_config: dict = Field(..., description="Base training config payload (from backend ConfigVersion)")
    data_source: str | None = Field(None, description="Task-B data source name (subdir under the data root). Used when no inline dataset is sent.")
    source: str | None = Field(None, description="Name for the materialized inline dataset (when `dataset` is provided)")
    dataset: list[RelearnDatasetEntry] | None = Field(None, description="Inline CPG dataset materialized by the backend from DatasetVersion(s)")
    dataset_bundle_uri: str | None = Field(None, description="URI (s3://… or path) to a prepared dataset bundle (.tar.gz of the gnn_vuln data/ layout) to train from directly")
    num_classes: int | None = None
    epochs: int | None = None
    base_checkpoint_path: str | None = Field(None, description="Task-A checkpoint (.pt) for finetune/EWC/ER/EWC-ER")
    replay_source: str | None = Field(None, description="Task-A data source for replay / EWC importance")
    device: str | None = None
    model_version_id: str | None = Field(None, description="Backend-precreated ModelVersion id, for correlation")
    run_name: str | None = None


class BuildCpgRequest(BaseModel):
    code: str = Field(..., description="Function source code")
    language: str | None = Field(None, description="Language hint: c, cpp, java, js, py")


class BuildCpgResponse(BaseModel):
    cpg_json: dict
    node_count: int | None = None
    detail: str | None = None


class RelearnJobOut(BaseModel):
    job_id: str
    status: str
    method: str
    data_source: str
    base_checkpoint_path: str | None = None
    config_path: str | None = None
    log_path: str | None = None
    result_checkpoint_path: str | None = None
    result_config_path: str | None = None
    metrics: dict | None = None
    model_version_id: str | None = None
    message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
