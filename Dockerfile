# Glyphh Runtime Dockerfile
#
# Usage:
#   docker build -t glyphh/runtime .
#   docker run -p 8002:8002 -e DATABASE_URL=... glyphh/runtime
#
# Or use docker compose:
#   docker compose up

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash glyphh
WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=glyphh:glyphh . /app

# Install the package itself (SDK + runtime + LLM backend)
RUN pip install --no-cache-dir -e ".[runtime,llm]"

# Download Qwen3-1.5B model at build time (public model, no token required)
# Pass --secret id=hf_token,env=HF_TOKEN at build time to avoid rate limits
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) \
    python -c "\
import os; \
from huggingface_hub import hf_hub_download; \
dest = os.path.expanduser('~glyphh/.local/share/glyphh/models'); \
os.makedirs(dest, exist_ok=True); \
hf_hub_download('Qwen/Qwen3-1.7B-GGUF', 'Qwen3-1.7B-Q4_K_M.gguf', \
    local_dir=dest, local_dir_use_symlinks=False)"

USER glyphh

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "glyphh.server:app", "--host", "0.0.0.0", "--port", "8002", "--limit-concurrency", "200", "--timeout-keep-alive", "30"]
