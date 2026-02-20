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

Spins up PostgreSQL + pgvector and the runtime together:

```bash
git clone https://github.com/glyphh-ai/glyphh-runtime.git
cd glyphh-runtime
docker compose up
```

### Option 2 — pip install + existing Postgres

If you already have PostgreSQL with pgvector running:

```bash
pip install glyphh[runtime]
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/glyphh_runtime
glyphh serve
```

If you don't have PostgreSQL locally, start just the database with Docker:

```bash
docker compose up -d db
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/glyphh_runtime
glyphh serve
```

### Query a deployed model

```bash
glyphh query "What is the refund policy?"
```

## Docker

Pull the image directly:

```bash
docker pull ghcr.io/glyphh-ai/glyphh-runtime:latest
```

## How It Works

1. Your LLM sends a natural language query via MCP
2. Glyphh encodes it into a high-dimensional vector using stored procedures
3. The encoded query resolves against a knowledge graph via GraphQL
4. Fact trees with confidence scores are returned to ground the LLM's response

## License

MIT
