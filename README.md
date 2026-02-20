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

Install the package

```bash
pip install glyphh
```

With runtime dependencies (PostgreSQL + pgvector):

```bash
pip install glyphh[runtime]
```

## Docker

```bash
docker pull ghcr.io/glyphh-ai/glyphh-runtime:latest
```

## Quick Start

```bash
# Start the runtime
glyphh serve

# Query a deployed model
glyphh query "What is the refund policy?"
```

## How It Works

1. Your LLM sends a natural language query via MCP
2. Glyphh encodes it into a high-dimensional vector using stored procedures
3. The encoded query resolves against a knowledge graph via GraphQL
4. Fact trees with confidence scores are returned to ground the LLM's response

## License

MIT
