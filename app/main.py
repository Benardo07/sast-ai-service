from __future__ import annotations

from contextlib import asynccontextmanager
import traceback

from fastapi import FastAPI, HTTPException

from dataclasses import asdict

from app.config import settings
from app.loader import release_manager
from app.relearn import relearn_manager
from app.schemas import (
    ActiveReleaseResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    LoadReleaseRequest,
    ValidateModelRequest,
    BuildCpgRequest,
    BuildCpgResponse,
    PredictRequest,
    PredictResponse,
    RelearnJobOut,
    RelearnRequest,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        release_manager.autoload_if_enabled()
    except Exception as exc:
        # Service must still boot even if persisted release fails to load.
        print(f"[sast-ai-service] autoload skipped: {exc}")
    yield


app = FastAPI(
    title="SAST AI Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(**release_manager.health_payload())


@app.get("/release", response_model=ActiveReleaseResponse)
async def active_release() -> ActiveReleaseResponse:
    return ActiveReleaseResponse(**release_manager.active_release_payload())


@app.get("/runtimes")
async def runtimes() -> dict:
    """What this runtime can run — used by the platform's import compatibility check."""
    return release_manager.runtimes_payload()


@app.post("/validate-model")
def validate_model(body: ValidateModelRequest) -> dict:
    """Dry-load a checkpoint+config (no side effects) to confirm gnn_vuln can run it."""
    return release_manager.validate(
        checkpoint_path=body.checkpoint_path, config_path=body.config_path, device=body.device,
    )


@app.post("/load-release", response_model=ActiveReleaseResponse)
def load_release(body: LoadReleaseRequest) -> ActiveReleaseResponse:
    try:
        release_manager.load_release(
            model_version_id=body.model_version_id,
            checkpoint_path=body.checkpoint_path,
            config_path=body.config_path,
            device=body.device,
            force_reload=body.force_reload,
            class_names=body.class_names,
        )
    except Exception as exc:
        release_manager.remember_load_error(exc)
        print("[sast-ai-service] load-release failed:")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(exc))
    return ActiveReleaseResponse(**release_manager.active_release_payload())


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if not (body.code and body.code.strip()) and not (body.cpg_path and body.cpg_path.strip()):
        raise HTTPException(status_code=422, detail="Provide 'code' or 'cpg_path'")
    try:
        if body.code and body.code.strip():
            payload = release_manager.predict_from_code(
                code=body.code,
                top_k_lines=body.top_k_lines,
                max_nodes=body.max_nodes,
            )
        else:
            payload = release_manager.predict_from_cpg(
                cpg_path=body.cpg_path,
                top_k_lines=body.top_k_lines,
                max_nodes=body.max_nodes,
                label=body.label,
                flaw_lines=body.flaw_lines,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PredictResponse(**payload)


@app.post("/relearn", response_model=RelearnJobOut)
async def relearn(body: RelearnRequest) -> RelearnJobOut:
    try:
        job = relearn_manager.submit(
            method=body.method,
            base_config=body.base_config,
            data_source=body.data_source,
            source=body.source,
            dataset=[e.model_dump() for e in body.dataset] if body.dataset else None,
            dataset_bundle_uri=body.dataset_bundle_uri,
            num_classes=body.num_classes,
            epochs=body.epochs,
            base_checkpoint_path=body.base_checkpoint_path,
            base_class_names=body.base_class_names,
            replay_source=body.replay_source,
            replay_bundle_uri=body.replay_bundle_uri,
            device=body.device,
            model_version_id=body.model_version_id,
            run_name=body.run_name,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return RelearnJobOut(**asdict(job))


@app.get("/relearn/{job_id}", response_model=RelearnJobOut)
async def relearn_status(job_id: str) -> RelearnJobOut:
    job = relearn_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    return RelearnJobOut(**asdict(job))


@app.get("/relearn", response_model=list[RelearnJobOut])
async def relearn_list() -> list[RelearnJobOut]:
    return [RelearnJobOut(**asdict(j)) for j in relearn_manager.list()]


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(body: EvaluateRequest) -> EvaluateResponse:
    """Score one checkpoint over an inline CPG dataset used as a 100% held-out test set,
    returning function-level + localization metrics. Used by the backend evaluation gate
    to compare champion vs candidate on the same benchmark.

    Declared `def` (NOT `async def`) on purpose: evaluate_checkpoint runs a blocking
    gnn_vuln.evaluate subprocess for minutes. A sync endpoint runs in FastAPI's threadpool,
    so the event loop stays free to serve /predict, /release, /health while eval runs — an
    `async def` here would freeze the whole service for the duration of the subprocess."""
    try:
        result = relearn_manager.evaluate_checkpoint(
            checkpoint_path=body.checkpoint_path,
            base_config=body.base_config,
            dataset=[e.model_dump() for e in body.dataset],
            source=body.source,
            base_class_names=body.base_class_names,
            device=body.device,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return EvaluateResponse(**result)


@app.post("/build-cpg", response_model=BuildCpgResponse)
def build_cpg_endpoint(body: BuildCpgRequest) -> BuildCpgResponse:
    """Generate a Joern CPG for one function. The backend caches the result."""
    from app.cpg import build_cpg

    try:
        result = build_cpg(body.code, body.language)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"CPG generation failed: {exc}")
    return BuildCpgResponse(**result)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "sast-ai-service",
        "status": "ok",
        "backend_base_url": settings.backend_base_url,
    }
