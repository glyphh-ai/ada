from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool, text
from sqlalchemy.ext.asyncio import AsyncEngine

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
GLYPHH_SDK_ROOT = REPO_ROOT / "glyphh-sdk"
for path in (str(REPO_ROOT), str(GLYPHH_SDK_ROOT), str(RUNTIME_ROOT)):
    if path not in sys.path and Path(path).exists():
        sys.path.insert(0, path)

from api.core.db import Base  # noqa: E402

config = context.config

env_url = os.getenv("GLYPH_DATABASE_URL")
if env_url:
    config.set_main_option("sqlalchemy.url", env_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    if isinstance(connectable, AsyncEngine):
        async def async_run():
            async with connectable.connect() as connection:
                await connection.run_sync(do_run_migrations)
        asyncio.run(async_run())
    else:
        with connectable.connect() as connection:
            do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
