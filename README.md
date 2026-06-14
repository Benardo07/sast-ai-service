# SAST AI Service

Standalone FastAPI service for loading released AI model checkpoints and serving inference separately from `sast-backend`.

## Scope

- `GET /health`
- `GET /release`
- `POST /load-release`
- `POST /predict`

This service is intentionally separated from `sast-backend` so model runtime, checkpoint loading, and GPU concerns do not affect the main backend process.

## Current inference contract

Current `predict` expects a ready CPG file path:

- `cpg_path`: path to a Joern-exported CPG `.json`

This is the shortest path to a working service because the sibling `tugas-akhir` repo already exposes `gnn_vuln.inference`.

## Bootstrap

1. Create venv and install service dependencies:

```powershell
cd "D:\TA Proj\sast-ai-service"
uv sync
```

2. Install the sibling AI training package into the same environment:

```powershell
uv pip install -e ..\tugas-akhir
```

This step is required because the service calls `gnn_vuln.inference` from the training repo.

3. Copy env file:

```powershell
Copy-Item .env.example .env
```

4. Run the service:

```powershell
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

## Load a release

Use a checkpoint imported earlier into the backend registry. Example:

```json
{
  "model_version_id": "optional-db-id",
  "checkpoint_path": "D:/TA Proj/tugas-akhir/checkpoints/imported/20260610_152801_lmgat_codebert_multiclass_checkpoints/checkpoints/20260610_152801_lmgat_codebert_multiclass/best_lmgat_codebert.pt",
  "config_path": "D:/TA Proj/tugas-akhir/checkpoints/imported/20260610_152801_lmgat_codebert_multiclass_checkpoints/checkpoints/20260610_152801_lmgat_codebert_multiclass/config.yaml",
  "device": "cpu"
}
```

If `config_path` is omitted, the service will look for `config.yaml` next to the checkpoint.

## Predict

```json
{
  "cpg_path": "D:/path/to/sample.json",
  "top_k_lines": 10,
  "max_nodes": 2500
}
```

## Next planned integration

- backend deploy action calls `POST /load-release`
- backend stores deployment result as `release_event`
- backend scan pipeline calls `POST /predict`
- optional support for raw code input instead of pre-built CPG path
