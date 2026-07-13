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
    # The model's ordered class list (backend-owned, from the ModelVersion class_names tag).
    # Overrides the predictor's checkpoint-derived vocab so /predict returns the right CWE
    # names for relearned/CIL models (whose sibling cwe_vocab.json is the task-B vocab).
    class_names: list[str] | None = None


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
    # Full function source. Written to the sample's meta as `raw_func` — the func-LM branch
    # (hybrid_graph_lm, sequential) tokenizes it; without it those architectures train/eval
    # on empty LM input. Ignored by the pure-GNN graph_based architecture.
    code: str | None = None

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
    base_class_names: list[str] | None = Field(None, description="Base model's ordered class names. When continuing a base model, task-B labels are remapped onto this class space (known CWEs keep their id, new CWEs extend the head).")
    replay_source: str | None = Field(None, description="Task-A data source for replay / EWC importance")
    replay_bundle_uri: str | None = Field(None, description="URI (s3://… or path) to the base model's training dataset bundle (.tar.gz of the gnn_vuln data/ layout). Installed into the data root and used as the replay/EWC source — the durable equivalent of replay_source, derived by the backend from the champion being relearned.")
    device: str | None = None
    model_version_id: str | None = Field(None, description="Backend-precreated ModelVersion id, for correlation")
    run_name: str | None = None


class EvaluateRequest(BaseModel):
    checkpoint_path: str = Field(..., description="Checkpoint to score (local path or s3:// URI)")
    base_config: dict = Field(..., description="Base training config (sections: data/model/train)")
    dataset: list[RelearnDatasetEntry] = Field(..., description="Inline CPG dataset used as a 100% held-out test set")
    source: str | None = Field(None, description="Name for the materialized eval dataset")
    base_class_names: list[str] | None = Field(None, description="Model's ordered class names for label alignment")
    device: str | None = None


class EvaluateResponse(BaseModel):
    job_id: str
    checkpoint_path: str
    metrics: dict
    num_samples: int
    num_dropped: int = 0
    # Drift-baseline signals captured during the eval pass (per-sample confidence/error +
    # capped pre-head embeddings). None when the evaluator produced no sidecar.
    baseline: dict | None = None


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
    # The trained model's ordered class list (index = class id). The backend stores this as the
    # new version's class_names tag so /predict labels CWEs correctly and the next relearn aligns
    # task-B labels onto this class space. None for binary/unknown-vocab runs (review #8).
    class_names: list[str] | None = None
    model_version_id: str | None = None
    message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
