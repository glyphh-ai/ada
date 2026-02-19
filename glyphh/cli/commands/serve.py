"""
CLI serve command — start the Glyphh runtime server.

glyphh serve              Start with defaults (port 8002)
glyphh serve --port 9000  Custom port
glyphh serve --reload     Auto-reload on code changes
"""

import click
import os
import sys

from .. import theme


@click.command("serve")
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", "-p", default=8002, type=int, help="Port number")
@click.option("--reload", is_flag=True, help="Auto-reload on changes")
@click.option("--workers", "-w", default=1, type=int, help="Number of workers")
def serve_command(host, port, reload, workers):
    """Start the Glyphh runtime server.

    Requires: pip install glyphh[runtime]
    """
    # Check runtime dependencies are installed
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
        import sqlalchemy  # noqa: F401
    except ImportError:
        click.secho("  Runtime dependencies not installed.", fg=theme.ERROR)
        click.secho("  Run: pip install glyphh[runtime]", fg=theme.ACCENT)
        sys.exit(1)

    # Check database connectivity
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        click.secho("  No DATABASE_URL set.", fg=theme.WARNING)
        click.secho("  Set DATABASE_URL or run: docker compose up -d db", fg=theme.MUTED)
        click.secho("  Example: export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/glyphh_runtime", fg=theme.TEXT_DIM)
        sys.exit(1)

    click.echo()
    click.secho(f"  Starting Glyphh Runtime on {host}:{port}", fg=theme.TEXT)
    click.secho(f"  Database: {_mask_db_url(db_url)}", fg=theme.TEXT_DIM)
    if reload:
        click.secho("  Auto-reload: enabled", fg=theme.TEXT_DIM)
    click.echo()

    # Find the main.py module — it's at the runtime repo root
    runtime_root = _find_runtime_root()
    if runtime_root:
        sys.path.insert(0, str(runtime_root))

    import uvicorn
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        log_level="info",
    )


def _mask_db_url(url: str) -> str:
    """Mask password in database URL for display."""
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


def _find_runtime_root():
    """Find the runtime root directory (where main.py lives)."""
    from pathlib import Path

    # Walk up from this file to find main.py
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "main.py").exists():
            return current
        current = current.parent

    # Fallback: check cwd
    cwd = Path.cwd()
    if (cwd / "main.py").exists():
        return cwd

    return None
