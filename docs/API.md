# Glyphh Runtime API Reference

Complete API documentation for the Glyphh Runtime.

## Base URL

- Local: `http://localhost:8000`
- Self-hosted: `https://your-domain.com`
- Cloud: `https://runtime.glyphh.com`

## Authentication

All endpoints (except health checks) require authentication in `self-hosted` and `cloud` modes.

### JWT Token

Include the JWT token in the `Authorization` header:

```
Authorization: Bearer <token>
```

Tokens are obtained from the Glyphh Platform UI at `platform.glyphh.com`.

### Local Mode

In `local` mode, authentication is bypassed for development convenience.

---

## Health Endpoints

### GET /health

Liveness probe. Returns 200 if the service is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /health/ready

Readiness probe. Checks all dependencies.

**Response:**
```json
{
  "status": "ready",
  "checks": {
    "database": "ok",
    "model_manager": "ok",
    "license": "ok"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /metrics

Prometheus-compatible metrics.

**Response:**
```json
{
  "runtime_uptime_seconds": 3600,
  "runtime_models_loaded": 2,
  "runtime_memory_mb": 512.5,
  "runtime_total_glyphs": 50000,
  "runtime_total_edges": 150000
}
```

### GET /resources

Detailed resource usage per namespace.

**Response:**
```json
{
  "system": {
    "process": {
      "memory_mb": 512.5,
      "cpu_percent": 15.2
    },
    "namespaces": {
      "count": 2,
      "total_glyphs": 50000
    }
  },
  "namespaces": [
    {
      "namespace": "model_abc123",
      "memory_mb": 256.0,
      "storage_mb": 1024.0,
      "glyph_count": 25000
    }
  ]
}
```

---

## Deployment Endpoints

### POST /api/deploy

Deploy a `.glyphh` model.

**Request:**
- Content-Type: `application/octet-stream`
- Body: Binary `.glyphh` file

**Response:**
```json
{
  "model_id": "model_abc123",
  "version": "1.0.0",
  "endpoints": {
    "search": "/api/v1/model_abc123/search",
    "fact_tree": "/api/v1/model_abc123/fact-tree",
    "predict": "/api/v1/model_abc123/predict"
  },
  "webhook_token": "whk_..."
}
```

### GET /api/status

Get runtime status.

**Response:**
```json
{
  "version": "1.0.0",
  "models_loaded": 2,
  "uptime": "2h 30m",
  "deployment_mode": "self-hosted"
}
```

### GET /api/models

List deployed models.

**Response:**
```json
{
  "models": [
    {
      "model_id": "model_abc123",
      "name": "My Model",
      "version": "1.0.0",
      "deployed_at": "2024-01-15T10:00:00Z",
      "status": "Active"
    }
  ]
}
```

### DELETE /api/models/{model_id}

Remove a deployed model.

**Response:**
```json
{
  "status": "deleted",
  "model_id": "model_abc123"
}
```

### PATCH /api/models/{model_id}/config

Update model configuration.

**Request:**
```json
{
  "similarity_weights": {
    "similarity": 1.0,
    "contrast": 0.5
  },
  "beam_width": 10,
  "max_tree_depth": 5
}
```

**Response:**
```json
{
  "namespace": "model_abc123",
  "similarity_weights": {...},
  "beam_width": 10,
  "max_tree_depth": 5,
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### POST /api/models/{model_id}/re-encode

Re-encode all glyphs with current encoder.

**Request:**
```json
{
  "regenerate_edges": true,
  "background": true
}
```

**Response:**
```json
{
  "status": "started",
  "job_id": "job_xyz789"
}
```

### DELETE /api/models/{model_id}/data

Clear all glyphs and edges, keep configuration.

**Response:**
```json
{
  "namespace": "model_abc123",
  "glyphs_deleted": 25000,
  "edges_deleted": 75000
}
```

---

## Glyph Endpoints

### POST /api/v1/{namespace}/glyphs

Create a new glyph.

**Request:**
```json
{
  "concept": "Machine learning is a subset of AI",
  "metadata": {
    "source": "api",
    "tags": ["ml", "ai"]
  }
}
```

**Response:**
```json
{
  "glyph_id": "550e8400-e29b-41d4-a716-446655440000",
  "namespace": "model_abc123",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### POST /api/v1/{namespace}/glyphs/batch

Create multiple glyphs.

**Request:**
```json
{
  "concepts": [
    "First concept",
    "Second concept",
    "Third concept"
  ],
  "metadata": {
    "batch": true
  }
}
```

**Response:**
```json
{
  "created": 3,
  "failed": 0,
  "results": [
    {"index": 0, "glyph_id": "...", "status": "created"},
    {"index": 1, "glyph_id": "...", "status": "created"},
    {"index": 2, "glyph_id": "...", "status": "created"}
  ],
  "errors": []
}
```

### GET /api/v1/{namespace}/glyphs/{glyph_id}

Get a glyph by ID.

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "namespace": "model_abc123",
  "concept_text": "Machine learning is a subset of AI",
  "metadata": {"source": "api"},
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### PUT /api/v1/{namespace}/glyphs/{glyph_id}

Update a glyph.

**Request:**
```json
{
  "concept": "Updated concept text",
  "metadata": {"updated": true}
}
```

### DELETE /api/v1/{namespace}/glyphs/{glyph_id}

Delete a glyph.

**Response:**
```json
{
  "status": "deleted",
  "glyph_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### GET /api/v1/{namespace}/glyphs

List glyphs with pagination.

**Query Parameters:**
- `limit` (int, default: 100): Maximum results
- `offset` (int, default: 0): Pagination offset

**Response:**
```json
{
  "glyphs": [...],
  "total": 25000,
  "limit": 100,
  "offset": 0
}
```

---

## Query Endpoints

### POST /api/v1/{namespace}/search

Similarity search.

**Request:**
```json
{
  "query": "machine learning algorithms",
  "top_k": 10,
  "filters": {
    "source": "api"
  },
  "include_embeddings": false
}
```

**Response:**
```json
{
  "results": [
    {
      "glyph": {...},
      "similarity_score": 0.95,
      "security_weight": 1.0,
      "final_score": 0.95
    }
  ],
  "total_count": 10,
  "query_time_ms": 15.5
}
```

### POST /api/v1/{namespace}/fact-tree

Generate fact tree for claim verification.

**Request:**
```json
{
  "claim": "Python is a programming language",
  "max_depth": 3,
  "branching_factor": 5
}
```

**Response:**
```json
{
  "root_claim": "Python is a programming language",
  "nodes": [
    {
      "id": "root",
      "claim": "Python is a programming language",
      "supporting_glyphs": ["..."],
      "confidence": 0.92,
      "children": []
    }
  ],
  "confidence": 0.92,
  "citations": [
    {
      "glyph_id": "...",
      "concept_text": "Python is a high-level programming language",
      "relevance_score": 0.95
    }
  ],
  "generation_time_ms": 45.2
}
```

### POST /api/v1/{namespace}/predict

Temporal prediction.

**Request:**
```json
{
  "current_state": ["initial state", "context"],
  "steps_ahead": 3,
  "beam_width": 5,
  "direction": "forward"
}
```

**Response:**
```json
{
  "predictions": [
    {
      "state_glyphs": ["..."],
      "confidence": 0.85,
      "path_score": 2.5,
      "deltas": [
        {
          "glyph_id": "...",
          "change_type": "added",
          "magnitude": 0.8
        }
      ]
    }
  ],
  "prediction_time_ms": 120.5
}
```

---

## MCP Tools

The MCP server exposes the following tools:

### glyph_similarity_search

Find similar glyphs.

**Parameters:**
```json
{
  "query": "search text",
  "top_k": 10
}
```

### glyph_fact_tree

Generate verification report.

**Parameters:**
```json
{
  "claim": "claim to verify",
  "max_depth": 3
}
```

### glyph_temporal_predict

Predict future states.

**Parameters:**
```json
{
  "current_state": ["state1", "state2"],
  "steps_ahead": 3
}
```

### glyph_create

Create a new glyph.

**Parameters:**
```json
{
  "concept": "concept text",
  "metadata": {}
}
```

### glyph_get

Get glyph by ID.

**Parameters:**
```json
{
  "glyph_id": "uuid"
}
```

### glyph_list

List glyphs.

**Parameters:**
```json
{
  "limit": 100,
  "offset": 0
}
```

---

## WebSocket Listener

Connect to `/api/v1/{namespace}/listener` for real-time glyph ingestion.

### Protocol

**Create Glyph:**
```json
{"type": "create_glyph", "concept": "...", "metadata": {...}}
```

**Response:**
```json
{"type": "glyph_created", "glyph_id": "...", "status": "success"}
```

**Ping/Pong:**
```json
{"type": "ping"}
{"type": "pong"}
```

**Heartbeat (server-initiated):**
```json
{"type": "heartbeat"}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {...},
    "correlation_id": "req_abc123",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid token |
| `AUTHORIZATION_DENIED` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `QUOTA_EXCEEDED` | 429 | Resource quota exceeded |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected server error |
