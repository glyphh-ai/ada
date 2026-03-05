# Glyphh Runtime Dockerfile
# Builds the runtime server with all dependencies.
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

# Install the package itself (SDK + runtime)
RUN pip install --no-cache-dir -e ".[runtime]"

USER glyphh

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--limit-concurrency", "200", "--timeout-keep-alive", "30"]
