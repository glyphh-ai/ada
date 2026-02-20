# Glyphh Runtime

Hyperdimensional computing runtime for deterministic, explainable AI.

Glyphh encodes natural language into high-dimensional vector representations using Vector Symbolic Architecture (VSA). No LLM in the loop — just math. Same input, same output, every time.

## Features

- **MCP Server** — Model Context Protocol interface for LLM sidecar integration
- **GraphQL API** — Query knowledge graphs and fact trees with confidence scores
- **CLI** — Manage models, deploy runtimes, and interact with the Glyphh Hub
- **Deterministic** — Auditable, reproducible results grounded in cosine similarity

## Install

Setup your python environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the package:

```bash
pip install glyphh
```

With runtime dependencies (PostgreSQL + pgvector):

```bash
pip install glyphh[runtime]
```

## Quick Start

The runtime requires PostgreSQL with pgvector. Pick whichever option fits your setup:

### Option 1 — Docker Compose (recommended)

Create a `docker-compose.yml` in your project directory:

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: glyphh_runtime
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  runtime:
    image: ghcr.io/glyphh-ai/glyphh-runtime:latest
    ports:
      - "8002:8002"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/glyphh_runtime
      - DEPLOYMENT_MODE=local
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

volumes:
  pgdata:
```

Then run:

```bash
docker compose up -d
```

Verify it's running:

```bash
curl http://localhost:8002/health
```

### Option 2 — Docker (manual)

Run the database and runtime as individual containers:

```bash
docker run -d --name glyphh-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=glyphh_runtime \
  -p 5432:5432 \
  pgvector/pgvector:pg16

docker pull ghcr.io/glyphh-ai/glyphh-runtime:latest

docker run -p 8002:8002 \
  -e DATABASE_URL=postgresql+asyncpg://postgres:postgres@host.docker.internal:5432/glyphh_runtime \
  ghcr.io/glyphh-ai/glyphh-runtime:latest
```

Or with an existing database:

```bash
docker run -p 8002:8002 \
  -e DATABASE_URL=postgresql+asyncpg://user:pass@your-db-host:5432/glyphh \
  ghcr.io/glyphh-ai/glyphh-runtime:latest
```

### Option 3 — pip install

If you already have PostgreSQL with pgvector:

```bash
pip install glyphh[runtime]
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/glyphh_runtime
glyphh serve
```

If you don't have PostgreSQL locally, start it with Docker first:

```bash
docker run -d --name glyphh-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=glyphh_runtime \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Then:

```bash
pip install glyphh[runtime]
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/glyphh_runtime
glyphh serve
```

### Query a deployed model

```bash
glyphh query "What is the refund policy?"
```

## How It Works

1. Your LLM sends a natural language query via MCP
2. Glyphh encodes it into a high-dimensional vector using stored procedures
3. The encoded query resolves against a knowledge graph via GraphQL
4. Fact trees with confidence scores are returned to ground the LLM's response

## License

MIT
