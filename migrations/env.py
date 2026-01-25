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

env_db_url = os.getenv("GLYPH_DATABASE_URL")
debug_enabled = os.getenv("GLYPH_MIGRATIONS_DEBUG") == "1"

if env_db_url:
    config.set_main_option("sqlalchemy.url", env_db_url.strip().strip('"').strip("'"))

masked_url = config.get_main_option("sqlalchemy.url")
if masked_url:
    masked_url = masked_url.replace("postgresql://", "postgresql+psycopg://", 1)
    masked_url = masked_url.rsplit("@", 1)[-1]
    print(f"[migrations] using db host: {masked_url}")

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
    if debug_enabled:
        try:
            db_name = connection.execute(text("select current_database()")).scalar_one()
            schema = connection.execute(text("select current_schema()")).scalar_one()
            user = connection.execute(text("select current_user")).scalar_one()
            search_path = connection.execute(text("show search_path")).scalar_one()
            server_addr = connection.execute(text("select inet_server_addr()")).scalar_one()
            server_port = connection.execute(text("select inet_server_port()")).scalar_one()
            table_count = connection.execute(
                text("select count(*) from information_schema.tables where table_schema='public'")
            ).scalar_one()
            print(f"[migrations] debug: cwd={os.getcwd()}")
            print(f"[migrations] debug: config={config.config_file_name}")
            print(f"[migrations] debug: db={db_name} schema={schema} user={user}")
            print(f"[migrations] debug: search_path={search_path}")
            print(f"[migrations] debug: server={server_addr}:{server_port}")
            print(f"[migrations] debug: public tables before={table_count}")
        except Exception as exc:
            print(f"[migrations] debug: failed to inspect connection: {exc!r}")

    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )

    context.run_migrations()

    if debug_enabled:
        try:
            table_count = connection.execute(
                text("select count(*) from information_schema.tables where table_schema='public'")
            ).scalar_one()
            alembic_table = connection.execute(text("select to_regclass('public.alembic_version')")).scalar_one()
            print(f"[migrations] debug: public tables after={table_count}")
            print(f"[migrations] debug: alembic_version={alembic_table}")
        except Exception as exc:
            print(f"[migrations] debug: failed to inspect tables after: {exc!r}")


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    if isinstance(connectable, AsyncEngine):
        async def async_run() -> None:
            async with connectable.begin() as connection:
                await connection.run_sync(do_run_migrations)

        asyncio.run(async_run())
    else:
        with connectable.begin() as connection:
            do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
