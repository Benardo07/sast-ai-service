from __future__ import annotations

from contextlib import asynccontextmanager
import traceback

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.loader import release_manager
from app.schemas import (
    ActiveReleaseResponse,
    HealthResponse,
    LoadReleaseRequest,
    PredictRequest,
    PredictResponse,
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


@app.post("/load-release", response_model=ActiveReleaseResponse)
async def load_release(body: LoadReleaseRequest) -> ActiveReleaseResponse:
    try:
        release_manager.load_release(
            model_version_id=body.model_version_id,
            checkpoint_path=body.checkpoint_path,
            config_path=body.config_path,
            device=body.device,
            force_reload=body.force_reload,
        )
    except Exception as exc:
        release_manager.remember_load_error(exc)
        print("[sast-ai-service] load-release failed:")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(exc))
    return ActiveReleaseResponse(**release_manager.active_release_payload())


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    try:
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


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "sast-ai-service",
        "status": "ok",
        "backend_base_url": settings.backend_base_url,
    }
