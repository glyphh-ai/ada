# Glyphh Runtime

Runtime service for executing `.glyphh` model deployments.

Handles ingest/query/stream, listeners, and MCP tooling. This can run in our
cloud or locally for data scientists/customers.

## Local Development

Internal dev setup is documented in `glyphh-platform/README.md`.

## Local Runtime Install (Manual)

This section documents the manual tar/zip install flow for offline/local runtimes.

### Prerequisites

- Postgres 16+ with pgvector enabled
- A signed local runtime license file from the Glyphh admin console
- A `.glyphh` bundle to import

### Download + Unpack

1. Download the runtime release archive from the customer portal.
2. Unpack it and choose an install directory.

Example:

```
tar -xzf glyphh-runtime-vX.Y.Z.tar.gz
cd glyphh-runtime
```

### Configure (Environment)

Set the runtime environment values (or place them in your shell profile).

Required:

- `GLYPH_DATABASE_URL` (runtime DB connection string)

Recommended:

- `GLYPH_RUNTIME_STORAGE_PATH` (local storage directory)
- `GLYPH_RUNTIME_PORT` (default 8080)
- `GLYPH_RUNTIME_LOG_LEVEL` (default INFO)

Example:

```
export GLYPH_DATABASE_URL="postgresql+psycopg://user:pass@host:5432/glyphh_runtime"
export GLYPH_RUNTIME_STORAGE_PATH="/var/lib/glyphh/runtime"
export GLYPH_RUNTIME_PORT=8080
export GLYPH_RUNTIME_LOG_LEVEL=INFO
```

### License File

Place the signed license file in the runtime config directory. The runtime
expects a local license file to run offline.

Example:

```
mkdir -p ~/.glyphh/runtime
cp ~/Downloads/license-dev.json ~/.glyphh/runtime/license.json
```

### Start the Runtime

Run the API server:

```
python3 -m uvicorn api.main:app --host 0.0.0.0 --port ${GLYPH_RUNTIME_PORT:-8080}
```

### Import a Bundle

Use the SDK CLI to import a `.glyphh` bundle:

```
glyphh runtime-import --auto --dir /path/to/bundle --runtime-url http://localhost:8080
```

### License Issuance Flow (Admin)

1. User runs the local runtime binary and generates a license request payload.
2. Admin opens the org runtime screen and selects **Create New Runtime (Local)**.
3. Admin pastes the license request payload and issues the signed license.
4. Admin provides the license file back to the user for placement on disk.

## Notes

- The runtime expects Postgres + pgvector.
- Use a dedicated runtime database (`glyphh_runtime`); do not point it at the platform DB.
- Studio uses `VITE_RUNTIME_BASE` to reach this service.
- Runtime includes Alembic; generate migrations with `alembic revision --autogenerate` from `glyphh-runtime/`.
