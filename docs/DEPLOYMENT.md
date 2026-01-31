# Glyphh Runtime Deployment Guide

This guide covers deploying the Glyphh Runtime in different environments.

## Deployment Modes

### Local Mode

For development and testing. No authentication required.

```bash
DEPLOYMENT_MODE=local
```

### Self-Hosted Mode

For customer-managed infrastructure. Requires:
- JWT authentication
- License validation (calls Glyphh Platform API)

```bash
DEPLOYMENT_MODE=self-hosted
JWT_SECRET_KEY=your-secret-key
LICENSE_KEY=your-license-key
```

### Cloud Mode

For Glyphh-managed deployments. Internal license validation.

```bash
DEPLOYMENT_MODE=cloud
JWT_SECRET_KEY=your-secret-key
```

---

## Docker Deployment

### Image Variants

| Image | Size | Features |
|-------|------|----------|
| `ghcr.io/glyphh/runtime:latest` | ~3GB | Full with NL Query LLM |
| `ghcr.io/glyphh/runtime:lite` | ~500MB | Rules-only, no LLM |

### Quick Start

```bash
# Pull the lite image
docker pull ghcr.io/glyphh/runtime:lite

# Run with PostgreSQL
docker run -d \
  --name glyphh-runtime \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e DEPLOYMENT_MODE=self-hosted \
  -e JWT_SECRET_KEY=your-secret \
  -e LICENSE_KEY=your-license \
  ghcr.io/glyphh/runtime:lite
```

### Docker Compose

```yaml
version: '3.8'

services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: glyphh_runtime
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  runtime:
    image: ghcr.io/glyphh/runtime:lite
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/glyphh_runtime
      - DEPLOYMENT_MODE=self-hosted
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - LICENSE_KEY=${LICENSE_KEY}
    depends_on:
      db:
        condition: service_healthy

volumes:
  pgdata:
```

### Building Custom Images

```bash
# Build lite image
docker build \
  --build-arg ENABLE_NL_QUERY=false \
  --build-arg SDK_REF=v1.0.0 \
  --build-arg SDK_TOKEN=$GITHUB_TOKEN \
  -t my-runtime:lite .

# Build full image with LLM
docker build \
  --build-arg ENABLE_NL_QUERY=true \
  --build-arg SDK_REF=v1.0.0 \
  --build-arg SDK_TOKEN=$GITHUB_TOKEN \
  --build-arg HF_TOKEN=$HUGGINGFACE_TOKEN \
  -t my-runtime:full .
```

---

## Heroku Deployment

### Prerequisites

- Heroku CLI installed
- Heroku account with container registry access

### Steps

```bash
# Login to Heroku
heroku login
heroku container:login

# Create app
heroku create my-glyphh-runtime

# Set stack to container
heroku stack:set container -a my-glyphh-runtime

# Add PostgreSQL
heroku addons:create heroku-postgresql:essential-0 -a my-glyphh-runtime

# Set config vars
heroku config:set \
  DEPLOYMENT_MODE=self-hosted \
  JWT_SECRET_KEY=your-secret-key \
  LICENSE_KEY=your-license-key \
  -a my-glyphh-runtime

# Deploy
git push heroku main
```

### Scaling

```bash
# Scale web dynos
heroku ps:scale web=2 -a my-glyphh-runtime

# View logs
heroku logs --tail -a my-glyphh-runtime
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- PostgreSQL with pgvector

### Deployment Manifest

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: glyphh

---
apiVersion: v1
kind: Secret
metadata:
  name: glyphh-secrets
  namespace: glyphh
type: Opaque
stringData:
  database-url: postgresql://user:pass@postgres:5432/glyphh
  jwt-secret-key: your-secret-key
  license-key: your-license-key

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: glyphh-runtime
  namespace: glyphh
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
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: glyphh-secrets
              key: jwt-secret-key
        - name: LICENSE_KEY
          valueFrom:
            secretKeyRef:
              name: glyphh-secrets
              key: license-key
        - name: DEPLOYMENT_MODE
          value: "self-hosted"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: glyphh-runtime
  namespace: glyphh
spec:
  selector:
    app: glyphh-runtime
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: glyphh-runtime
  namespace: glyphh
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
  - host: runtime.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: glyphh-runtime
            port:
              number: 80
```

### Apply

```bash
kubectl apply -f deployment.yaml
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: glyphh-runtime-hpa
  namespace: glyphh
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: glyphh-runtime
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Database Setup

### PostgreSQL with pgvector

The runtime requires PostgreSQL 14+ with the pgvector extension.

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

### Connection Pooling

For production, use PgBouncer or similar:

```bash
# PgBouncer config
[databases]
glyphh_runtime = host=postgres port=5432 dbname=glyphh_runtime

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
```

### Migrations

Run Alembic migrations on startup:

```bash
alembic upgrade head
```

---

## Monitoring

### Health Checks

| Endpoint | Purpose | Frequency |
|----------|---------|-----------|
| `/health` | Liveness | 10s |
| `/health/ready` | Readiness | 5s |

### Metrics

The `/metrics` endpoint provides Prometheus-compatible metrics:

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'glyphh-runtime'
    static_configs:
      - targets: ['runtime:8000']
    metrics_path: /metrics
```

### Logging

Structured JSON logs are written to stdout:

```json
{"timestamp": "2024-01-15T10:30:00", "level": "INFO", "message": "..."}
```

Configure log aggregation (ELK, Loki, CloudWatch) as needed.

---

## Troubleshooting

### Common Issues

**Database Connection Failed**
```
Check DATABASE_URL format: postgresql://user:pass@host:5432/db
Verify PostgreSQL is running and accessible
Check pgvector extension is installed
```

**License Validation Failed**
```
Verify LICENSE_KEY is correct
Check network access to platform.glyphh.com
Runtime enters 7-day grace period on failure
```

**Out of Memory**
```
Reduce DEFAULT_NAMESPACE_MEMORY_MB
Use lite image (no LLM)
Scale horizontally with more replicas
```

**Slow Queries**
```
Check pgvector index: CREATE INDEX ON glyphs USING ivfflat (embedding vector_cosine_ops)
Increase PostgreSQL shared_buffers
Enable query logging to identify slow queries
```

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=DEBUG
```

### Support

- Documentation: https://docs.glyphh.com
- Issues: https://github.com/glyphh/glyphh-runtime/issues
- Email: support@glyphh.com
