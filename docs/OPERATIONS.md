# Glyphh Runtime Operations Guide

Guide for operators managing Glyphh Runtime deployments.

## Model Management

### Loading Models

Models are loaded via the deployment API:

```bash
# Deploy a model
curl -X POST http://localhost:8000/api/deploy \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @model.glyphh
```

Response includes the namespace and endpoints:

```json
{
  "model_id": "model_abc123",
  "endpoints": {
    "search": "/api/v1/model_abc123/search"
  }
}
```

### Listing Models

```bash
curl http://localhost:8000/api/models \
  -H "Authorization: Bearer $TOKEN"
```

### Unloading Models

```bash
curl -X DELETE http://localhost:8000/api/models/model_abc123 \
  -H "Authorization: Bearer $TOKEN"
```

### Updating Configuration

Update similarity weights, beam width, or tree depth without re-encoding:

```bash
curl -X PATCH http://localhost:8000/api/models/model_abc123/config \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "similarity_weights": {"similarity": 1.0, "contrast": 0.3},
    "beam_width": 10
  }'
```

### Re-encoding

Re-encode all glyphs when encoder configuration changes:

```bash
curl -X POST http://localhost:8000/api/models/model_abc123/re-encode \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"regenerate_edges": true, "background": true}'
```

Check re-encode job status:

```bash
curl http://localhost:8000/api/models/model_abc123/re-encode/job_xyz789 \
  -H "Authorization: Bearer $TOKEN"
```

### Clearing Data

Clear all glyphs and edges while preserving configuration:

```bash
curl -X DELETE http://localhost:8000/api/models/model_abc123/data \
  -H "Authorization: Bearer $TOKEN"
```

---

## Monitoring

### Health Checks

**Liveness Probe** - Is the service running?
```bash
curl http://localhost:8000/health
```

**Readiness Probe** - Is the service ready to accept traffic?
```bash
curl http://localhost:8000/health/ready
```

### Metrics

Prometheus-compatible metrics:

```bash
curl http://localhost:8000/metrics
```

Key metrics:
- `runtime_uptime_seconds` - Server uptime
- `runtime_models_loaded` - Number of loaded models
- `runtime_memory_mb` - Process memory usage
- `runtime_total_glyphs` - Total glyphs across namespaces
- `runtime_cpu_percent` - CPU utilization

### Resource Usage

Detailed per-namespace resource usage:

```bash
curl http://localhost:8000/resources
```

### Logs

Logs are structured JSON written to stdout:

```json
{"timestamp": "2024-01-15T10:30:00", "level": "INFO", "logger": "main", "message": "Model loaded"}
```

View recent logs:

```bash
curl http://localhost:8000/api/logs?lines=100 \
  -H "Authorization: Bearer $TOKEN"
```

---

## Performance Tuning

### Database Optimization

**Connection Pooling**

Configure pool size based on workload:

```bash
# In DATABASE_URL or separate config
?pool_size=20&max_overflow=10
```

**pgvector Index**

Ensure IVFFlat index exists for fast similarity search:

```sql
CREATE INDEX idx_glyph_embedding ON glyphs 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

Tune `lists` parameter based on data size:
- < 1M vectors: lists = 100
- 1M - 10M vectors: lists = 1000
- > 10M vectors: lists = sqrt(n)

**Query Optimization**

Enable query logging to identify slow queries:

```sql
ALTER SYSTEM SET log_min_duration_statement = 100;
SELECT pg_reload_conf();
```

### Memory Management

**Namespace Quotas**

Set per-namespace memory limits:

```bash
DEFAULT_NAMESPACE_MEMORY_MB=1024
```

**Encoder Caching**

Encoders are cached in memory. Monitor memory usage and adjust quotas.

### Concurrency

**Rate Limiting**

Configure rate limits:

```bash
RATE_LIMIT_PER_MINUTE=60
```

**Connection Limits**

WebSocket connections have built-in limits:
- Heartbeat interval: 30 seconds
- Connection timeout: 5 minutes

---

## Backup and Recovery

### Database Backup

**pg_dump**

```bash
pg_dump -h localhost -U postgres -d glyphh_runtime > backup.sql
```

**Continuous Archiving**

Configure WAL archiving for point-in-time recovery:

```bash
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'
```

### Model Backup

Models are stored as `.glyphh` files. Back up the original files.

### Recovery Procedures

**Database Recovery**

```bash
# Restore from backup
psql -h localhost -U postgres -d glyphh_runtime < backup.sql

# Run migrations
alembic upgrade head
```

**Model Recovery**

Re-deploy models from backup `.glyphh` files:

```bash
curl -X POST http://localhost:8000/api/deploy \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @backup/model.glyphh
```

---

## Security

### Token Management

**List Tokens**

```bash
curl http://localhost:8000/api/tokens \
  -H "Authorization: Bearer $TOKEN"
```

**Revoke Token**

```bash
curl -X DELETE http://localhost:8000/api/tokens/token_id \
  -H "Authorization: Bearer $TOKEN"
```

### Audit Logging

All API requests are logged with:
- Correlation ID
- User ID (from JWT)
- Namespace
- Operation
- Timestamp

### Network Security

- Use HTTPS in production
- Configure CORS appropriately
- Use network policies in Kubernetes

---

## Alerting

### Recommended Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Memory | memory_percent > 90% | Warning |
| High CPU | cpu_percent > 80% for 5m | Warning |
| Database Down | health/ready database != ok | Critical |
| License Expiring | grace_period_remaining < 3d | Warning |
| High Error Rate | errors/requests > 5% | Warning |

### Prometheus Alert Rules

```yaml
groups:
- name: glyphh-runtime
  rules:
  - alert: HighMemoryUsage
    expr: runtime_memory_percent > 90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High memory usage on Glyphh Runtime

  - alert: DatabaseUnhealthy
    expr: glyphh_health_database != 1
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Database health check failing
```

---

## Maintenance

### Graceful Shutdown

The runtime handles SIGTERM gracefully:
1. Stops accepting new connections
2. Drains existing connections (30s timeout)
3. Closes WebSocket connections
4. Closes database connections
5. Flushes logs

### Rolling Updates

In Kubernetes, use rolling update strategy:

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

### Database Migrations

Run migrations before deploying new versions:

```bash
alembic upgrade head
```

Check migration status:

```bash
alembic current
alembic history
```

---

## Troubleshooting

### Common Issues

**Model Load Failure**
- Check `.glyphh` file integrity
- Verify SDK version compatibility
- Check available memory

**Slow Queries**
- Check pgvector index exists
- Analyze query plans with EXPLAIN
- Consider increasing `lists` parameter

**Connection Timeouts**
- Check database connection pool
- Verify network connectivity
- Check for connection leaks

**License Validation Failure**
- Verify LICENSE_KEY
- Check network access to platform.glyphh.com
- Runtime continues in grace period (7 days)

### Debug Commands

```bash
# Check runtime status
curl http://localhost:8000/api/status

# Check database connectivity
curl http://localhost:8000/health/ready

# View resource usage
curl http://localhost:8000/resources

# View recent logs
curl http://localhost:8000/api/logs?lines=50
```

### Support

- Documentation: https://docs.glyphh.com
- Issues: https://github.com/glyphh/glyphh-runtime/issues
- Email: support@glyphh.com
