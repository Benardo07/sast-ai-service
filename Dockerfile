# sast-ai-service image — the AI compute endpoint (inference + relearn).
#
# torch / torch-geometric / transformers + Joern (CPG) all live INSIDE this image, so the
# host never installs the heavy ML stack — that is the whole point of running it in Docker.
# CPU build by default; for GPU pass the cu124 wheel indexes as build args (see compose
# overlay docker-compose.gpu.yml at the repo root).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SAST_AI_JOERN_CLI=/opt/joern/joern-cli \
    SAST_AI_HOST=0.0.0.0 \
    SAST_AI_PORT=8001 \
    HF_HOME=/app/.hf

# ── system deps: JDK (required by Joern) + fetch/build tools ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless curl unzip git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1

# torch + PyG sparse extensions need their own wheel indexes — install them FIRST so the
# project's transitive torch deps are already satisfied (and we control CPU vs CUDA).
# GPU build, e.g.:
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
#   --build-arg PYG_FIND_LINKS=https://data.pyg.org/whl/torch-2.6.0+cu124.html
ARG TORCH_VERSION=2.6.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PYG_FIND_LINKS=https://data.pyg.org/whl/torch-2.6.0+cpu.html
RUN uv pip install --system torch==${TORCH_VERSION} --index-url ${TORCH_INDEX} \
    && uv pip install --system torch-geometric>=2.5.0 \
    && uv pip install --system torch-scatter torch-sparse -f ${PYG_FIND_LINKS}

# ── Joern (CPG generation). Pinned + placed before the app layer so a code change reuses
# this cached ~1.7GB download. curl -f fails on 404; test -x asserts the CLI extracted. ──
ARG JOERN_VERSION=v4.0.526
RUN curl -fL "https://github.com/joernio/joern/releases/download/${JOERN_VERSION}/joern-cli.zip" \
        -o /tmp/joern-cli.zip \
    && mkdir -p /opt/joern \
    && unzip -q /tmp/joern-cli.zip -d /opt/joern \
    && rm /tmp/joern-cli.zip \
    && chmod +x /opt/joern/joern-cli/joern* \
    && test -x /opt/joern/joern-cli/joern-parse

# ── app + vendored gnn_vuln. torch/pyg above already satisfy the heavy deps; the rest
# (transformers, numpy, boto3, ...) resolve from PyPI via our pyproject. ──
COPY pyproject.toml README.md ./
COPY app ./app
COPY gnn_vuln ./gnn_vuln
RUN uv pip install --system .

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
