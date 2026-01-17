# Glyphh Runtime

Runtime service for executing `.glyphh` model deployments.

Handles ingest/query/stream, listeners, and MCP tooling. This can run in our
cloud or locally for data scientists/customers.

## Local Development

Internal dev setup is documented in `glyphh-platform/README.md`.

## Local Runtime Install (Manual)

Local/offline install instructions live in the public releases repo:

- https://github.com/glyphh-ai/glyphh-releases

Refer to that README for download, configuration, licensing, and startup steps.

## Activation Key (CLI)

For local activation during development, use:

```bash
python3 glyphh-runtime/scripts/activate_runtime.py \
  --platform-url https://api.glyphh.ai/api/v1 \
  --org-id ORG_ID \
  --runtime-id RUNTIME_ID \
  --activation-key gk_...
```

## Notes

- The runtime expects Postgres + pgvector.
- Use a dedicated runtime database (`glyphh_runtime`); do not point it at the platform DB.
- Studio uses `VITE_RUNTIME_BASE` to reach this service.
- Runtime includes Alembic; generate migrations with `alembic revision --autogenerate` from `glyphh-runtime/`.
