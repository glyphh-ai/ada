# Glyphh Runtime

Runtime service for executing `.glyphh` model deployments.

Handles ingest/query/stream, listeners, and MCP tooling. This can run in our
cloud or locally for data scientists/customers.

## Local Development

Internal dev setup is documented in `glyphh-platform/README.md`.

## Notes

- The runtime expects Postgres + pgvector.
- Use a dedicated runtime database (`glyphh_runtime`); do not point it at the platform DB.
- Studio uses `VITE_RUNTIME_BASE` to reach this service.
- Runtime includes Alembic; generate migrations with `alembic revision --autogenerate` from `glyphh-runtime/`.
