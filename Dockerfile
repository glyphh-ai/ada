FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential curl \
  && rm -rf /var/lib/apt/lists/*

COPY glyphh-sdk /app/glyphh-sdk
COPY glyphh-runtime /app/glyphh-runtime

ARG GLYPH_SDK_WHEEL_URL
ARG HF_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV PIP_PROGRESS_BAR=off
RUN pip install --no-cache-dir -r /app/glyphh-runtime/requirements.txt \
  && if [ -n "$GLYPH_SDK_WHEEL_URL" ]; then pip install --no-cache-dir "$GLYPH_SDK_WHEEL_URL"; fi \
  && pip install --no-cache-dir -e /app/glyphh-sdk \
  && pip install --no-cache-dir "huggingface_hub>=0.22.0" \
  && python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = "sentence-transformers/all-MiniLM-L6-v2"
local_dir = "/app/glyphh-runtime/models/intent/all-MiniLM-L6-v2"
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=token,
)
PY

ENV PYTHONPATH=/app/glyphh-runtime

WORKDIR /app/glyphh-runtime

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
