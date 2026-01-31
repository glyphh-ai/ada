# Glyphh Runtime

Execution environment for deployed `.glyphh` models. Serves models through REST and MCP APIs with persistent storage, multi-model management, licensing, and authentication.

## Quick Start

### Local Development

```bash
# Clone and setup
cd glyphh-runtime
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Start PostgreSQL with pgvector
docker-compose up -d db

# Copy and configure environment
cp .env.example .env

# Run the server
python main.py
```

### Docker

```bash
# Full image (with NL Query support)
docker pull ghcr.io/glyphh/runtime:latest

# Lite image (no NL Query, smaller footprint)
docker pull ghcr.io/glyphh/runtime:lite

# Run
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e DEPLOYMENT_MODE=local \
  ghcr.io/glyphh/runtime:latest
```

## Deployment Modes

| Mode | Auth Required | License | Use Case |
|------|--------------|---------|----------|
| `local` | No | No | Development |
| `self-hosted` | Yes (JWT) | Yes (call-home) | Customer infrastructure |
| `cloud` | Yes (JWT) | Internal | Glyphh-hosted |

## API Endpoints

### CLI-Facing (Deployment)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/deploy` | POST | Deploy .glyphh model |
| `/api/status` | GET | Runtime status |
| `/api/models` | GET | List deployed models |
| `/api/models/{id}` | DELETE | Remove model |
| `/api/logs` | GET | Runtime logs |
| `/api/tokens` | GET | List tokens |
| `/api/tokens/{id}` | DELETE | Revoke token |

### Query API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/{namespace}/search` | POST | Similarity search |
| `/api/v1/{namespace}/fact-tree` | POST | Generate fact tree |
| `/api/v1/{namespace}/predict` | POST | Temporal prediction |
| `/api/v1/{namespace}/query` | POST | Natural language query (if enabled) |

### MCP

| Endpoint | Description |
|----------|-------------|
| `/{org_id}/{model_id}/mcp` | MCP server (cloud mode) |
| `/mcp/{namespace}` | MCP server (local/self-hosted) |

### Health

| Endpoint | Description |
|----------|-------------|
| `/health` | Liveness probe |
| `/health/ready` | Readiness probe |
| `/metrics` | Prometheus metrics |

## Configuration

See `.env.example` for all configuration options.

## License

Proprietary - Glyphh AI
