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

# Run database migrations
alembic upgrade head

# Run the server
python main.py
```

### Docker

```bash
# Full image (with NL Query support)
docker pull ghcr.io/glyphh/runtime:latest

# Lite image (no NL Query, smaller footprint)
docker pull ghcr.io/glyphh/runtime:lite

# Run with docker-compose (recommended)
docker-compose up

# Or run standalone
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

### Local Mode

For development and testing. No authentication required.

```bash
DEPLOYMENT_MODE=local
```

### Self-Hosted Mode

For customer-managed deployments. Requires JWT authentication and license validation.

```bash
DEPLOYMENT_MODE=self-hosted
JWT_SECRET_KEY=your-secret-key
LICENSE_KEY=your-license-key
PLATFORM_API_URL=https://platform.glyphh.com/api
```

### Cloud Mode

For Glyphh-managed cloud deployments. Internal license validation.

```bash
DEPLOYMENT_MODE=cloud
JWT_SECRET_KEY=your-secret-key
```

## Deployment Options

### Docker Compose (Recommended for Development)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f runtime

# Stop services
docker-compose down
```

### Heroku

```bash
# Set stack to container
heroku stack:set container

# Add PostgreSQL addon
heroku addons:create heroku-postgresql:essential-0

# Set config vars
heroku config:set DEPLOYMENT_MODE=self-hosted
heroku config:set JWT_SECRET_KEY=your-secret-key
heroku config:set LICENSE_KEY=your-license-key

# Deploy
git push heroku main
```

### Kubernetes

Example deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: glyphh-runtime
spec:
  replicas: 2
  selector:
    matchLabels:
      app: glyphh-runtime
  template:
    metadata:
      labels:
        app: glyphh-runtime
    spec:
      containers:
      - name: runtime
        image: ghcr.io/glyphh/runtime:lite
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: glyphh-secrets
              key: database-url
        - name: DEPLOYMENT_MODE
          value: "self-hosted"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

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

All configuration is done via environment variables. See `.env.example` for all options.

### Required Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql://localhost:5432/glyphh_runtime` |
| `DEPLOYMENT_MODE` | `local`, `self-hosted`, or `cloud` | `local` |

### Authentication (self-hosted/cloud)

| Variable | Description |
|----------|-------------|
| `JWT_SECRET_KEY` | Secret key for JWT signing |
| `JWT_ALGORITHM` | JWT algorithm (default: HS256) |

### Licensing (self-hosted)

| Variable | Description |
|----------|-------------|
| `LICENSE_KEY` | License key from Glyphh |
| `PLATFORM_API_URL` | Platform API URL for validation |
| `LICENSE_GRACE_PERIOD_DAYS` | Grace period on validation failure (default: 7) |

### Resource Quotas

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_NAMESPACE_MEMORY_MB` | Memory quota per namespace | 1024 |
| `DEFAULT_NAMESPACE_STORAGE_GB` | Storage quota per namespace | 10 |
| `RATE_LIMIT_PER_MINUTE` | Rate limit per minute | 60 |

### NL Query (full image only)

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_NL_QUERY` | Enable NL query interface | false |
| `NL_MODEL` | Model for NL translation | microsoft/Phi-3.5-mini-instruct |

### Telemetry

| Variable | Description | Default |
|----------|-------------|---------|
| `TELEMETRY_ENABLED` | Enable telemetry export | true |
| `TELEMETRY_ENDPOINT` | Endpoint for metrics export | (none) |

## Health Checks

| Endpoint | Description |
|----------|-------------|
| `/health` | Liveness probe - returns 200 if service is running |
| `/health/ready` | Readiness probe - checks database, model manager |
| `/metrics` | Prometheus-compatible metrics |
| `/resources` | Detailed resource usage per namespace |

## Monitoring

### Prometheus Metrics

The `/metrics` endpoint exposes:
- `runtime_uptime_seconds` - Server uptime
- `runtime_models_loaded` - Number of loaded models
- `runtime_memory_mb` - Process memory usage
- `runtime_total_glyphs` - Total glyphs across namespaces
- `runtime_total_edges` - Total edges across namespaces

### Logging

Structured JSON logging is enabled by default:

```json
{"timestamp": "2024-01-15T10:30:00", "level": "INFO", "logger": "main", "message": "..."}
```

Set `LOG_LEVEL` to control verbosity (DEBUG, INFO, WARNING, ERROR).

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker-compose ps db

# Check pgvector extension
docker-compose exec db psql -U postgres -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

### License Validation Failures

- Check `LICENSE_KEY` is set correctly
- Verify network access to `PLATFORM_API_URL`
- Runtime enters grace period (7 days) on validation failure

### Memory Issues

- Reduce `DEFAULT_NAMESPACE_MEMORY_MB` quota
- Use lite image (no LLM dependencies)
- Scale horizontally with multiple instances

## License

This project is licensed under the [Glyphh AI Community License](LICENSE).

- ✓ Free to download and use for development
- ✓ Free tier available for production (1 model, 1,000 glyphs)
- ✗ Not open source — cannot redistribute or build competing products

Production use beyond the free tier requires a license key from https://glyphh.com.
