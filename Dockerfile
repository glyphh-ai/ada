FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential curl \
  && rm -rf /var/lib/apt/lists/*

COPY glyphh-sdk /app/glyphh-sdk
COPY glyphh-runtime /app/glyphh-runtime

RUN pip install --no-cache-dir -r /app/glyphh-runtime/requirements.txt \
  && pip install --no-cache-dir -e /app/glyphh-sdk

ENV PYTHONPATH=/app/glyphh-runtime

WORKDIR /app/glyphh-runtime

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
