"""
Destructive database reset utility for the runtime service.

Drops all tables defined via `db.Base` and recreates them. Use only in
local/dev environments; it will wipe all data.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "glyphh-runtime"
GLYPHH_SDK_ROOT = REPO_ROOT / "glyphh-sdk"
for path in (REPO_ROOT, RUNTIME_ROOT, GLYPHH_SDK_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from api.core.db import Base, engine  # noqa: E402

VERSIONS_DIR = RUNTIME_ROOT / "migrations" / "versions"


def remove_migration_files() -> None:
    if not VERSIONS_DIR.exists():
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        return
    for entry in VERSIONS_DIR.iterdir():
        if entry.name == "__init__.py":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def run_alembic(*args: str) -> None:
    alembic_cfg = str(RUNTIME_ROOT / "alembic.ini")
    env = os.environ.copy()
    db_url = env.get("GLYPH_DATABASE_URL")
    if db_url:
        env["GLYPH_DATABASE_URL"] = db_url.strip().strip('"').strip("'")
    subprocess.run(
        ["alembic", "-c", alembic_cfg, *args],
        cwd=str(RUNTIME_ROOT),
        check=True,
        env=env,
    )


def patch_latest_migration_for_pgvector() -> None:
    versions = list(VERSIONS_DIR.glob("*.py"))
    if not versions:
        return
    latest = max(versions, key=lambda path: path.stat().st_mtime)
    contents = latest.read_text()
    if "pgvector.sqlalchemy.vector.VECTOR" not in contents:
        return
    if "import pgvector" in contents:
        return
    lines = contents.splitlines()
    insert_at = None
    for idx, line in enumerate(lines):
        if line.startswith("import sqlalchemy as sa"):
            insert_at = idx + 1
            break
    if insert_at is None:
        insert_at = 0
    lines.insert(insert_at, "import pgvector")
    latest.write_text("\n".join(lines) + "\n")


def main(force: bool) -> None:
    if not force:
        confirm = input(
            "This will DROP all tables and reset Alembic migrations. Type 'yes' to continue: "
        ).strip()
        if confirm.lower() != "yes":
            print("Aborted.")
            return

    if not os.getenv("GLYPH_DATABASE_URL"):
        print("GLYPH_DATABASE_URL is not set. Aborting.")
        sys.exit(1)

    print("Dropping tables...")
    with engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(text(f'DROP TABLE IF EXISTS "{table.name}" CASCADE'))
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))

    print("Removing existing migrations...")
    remove_migration_files()

    print("Creating new base migration...")
    run_alembic("revision", "--autogenerate", "-m", "base", "--head", "base", "--splice")
    patch_latest_migration_for_pgvector()

    print("Running Alembic upgrade head...")
    run_alembic("upgrade", "head")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    main(args.yes)
