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

Glyphh ships as a single package with different install profiles:

| Profile | Command | What you get |
|---------|---------|-------------|
| SDK | `pip install glyphh` | Encoder, similarity, CLI, model packaging. Lightweight — just numpy, pyyaml, click, httpx. |
| Runtime | `pip install glyphh[runtime]` | Everything in SDK + FastAPI server, SQLAlchemy, pgvector, Alembic, Pydantic. For running the runtime locally. |
| Dev | `pip install glyphh[dev]` | Everything in SDK + pytest, hypothesis, black, ruff, mypy. For contributing to Glyphh. |

Most users want either SDK (build and package models) or Runtime (deploy and serve them).

## Quick Start

The runtime requires PostgreSQL with pgvector. Pick whichever option fits your setup:

### Option 1 — Docker Compose (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose plugin).

The CLI can scaffold the Docker files for you:

```bash
pip install glyphh[runtime]
glyphh docker init
docker pull ghcr.io/glyphh-ai/glyphh-runtime:latest
docker compose up -d
```

`glyphh docker init` writes a `docker-compose.yml` and `init.sql` into your current directory. The compose file runs PostgreSQL with pgvector and the published runtime image — no build step needed.

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

### Option 3 — pip install (bring your own Postgres)

If you already have PostgreSQL with pgvector running:

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
