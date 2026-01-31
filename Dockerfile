# Glyphh Runtime Dockerfile
# Multi-stage build supporting full (with NL query) and lite variants
#
# Build args:
#   SDK_REF: Git ref for SDK (branch, tag, or commit)
#   SDK_TOKEN: GitHub token for private SDK repo
#   HF_TOKEN: HuggingFace token for downloading models (full only)
#
# Build examples:
#   Lite:  docker build --target lite -t glyphh-runtime:lite .
#   Full:  docker build --target full --build-arg HF_TOKEN=xxx -t glyphh-runtime:full .

# =============================================================================
# Base Stage - Common dependencies
# =============================================================================
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash glyphh
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install base Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# SDK Stage - Install Glyphh SDK
# =============================================================================
FROM base AS sdk

ARG SDK_REF=main
ARG SDK_TOKEN

# Install SDK from private GitHub repo
RUN if [ -n "$SDK_TOKEN" ]; then \
        pip install --no-cache-dir "glyphh @ git+https://${SDK_TOKEN}@github.com/glyphh/glyphh-sdk.git@${SDK_REF}"; \
    else \
        echo "Warning: SDK_TOKEN not provided, skipping SDK installation"; \
    fi

# =============================================================================
# Lite Stage - Without NL Query (default)
# =============================================================================
FROM sdk AS lite

ENV ENABLE_NL_QUERY=false

# Copy application code
COPY --chown=glyphh:glyphh . /app

# Switch to non-root user
USER glyphh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Full Stage - With NL Query (LLM support)
# =============================================================================
FROM sdk AS full

ARG HF_TOKEN

# Install NL query dependencies
RUN pip install --no-cache-dir \
    transformers>=4.35.0 \
    torch>=2.0.0 \
    accelerate>=0.24.0 \
    sentencepiece>=0.1.99

# Download Phi-3.5-mini-instruct model (if HF_TOKEN provided)
RUN if [ -n "$HF_TOKEN" ]; then \
        python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" && \
        python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
            AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct'); \
            AutoModelForCausalLM.from_pretrained('microsoft/Phi-3.5-mini-instruct', trust_remote_code=True)"; \
    else \
        echo "Warning: HF_TOKEN not provided, model will be downloaded on first use"; \
    fi

ENV ENABLE_NL_QUERY=true

# Copy application code
COPY --chown=glyphh:glyphh . /app

# Switch to non-root user
USER glyphh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
