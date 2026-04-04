"""
CLI serve command — start the Glyphh runtime as a long-running server.

This is the headless entry point for keeping the runtime alive in the
background (for MCP servers, listeners, and API access).

glyphh serve              Start with defaults (port 8002)
glyphh serve --port 9000  Custom port
glyphh serve --reload     Auto-reload on code changes (development)
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
    """Start the Glyphh runtime as a long-running server.

    Use this to keep the runtime alive for MCP servers, listeners,
    and API access after code init. The interactive shell embeds a
    server that dies on exit — this command keeps it running.

    Examples:
        glyphh serve                     # foreground, port 8002
        glyphh serve -p 9000             # custom port
        glyphh serve &                   # background (shell)
        nohup glyphh serve > /dev/null & # background (survives logout)

    Requires: pip install glyphh
    """
    # Check runtime dependencies are installed
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
        import sqlalchemy  # noqa: F401
    except ImportError:
        click.secho("  Runtime dependencies not installed.", fg=theme.ERROR)
        click.secho("  Run: pip install glyphh", fg=theme.ACCENT)
        sys.exit(1)

    # Set defaults for local use
    os.environ.setdefault("ENABLE_DOCS", "true")
    os.environ.setdefault("CORS_ALLOW_ALL", "true")

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        click.secho("  No DATABASE_URL set — using SQLite (glyphh_dev.db)", fg=theme.MUTED)

    click.echo()
    click.secho(f"  Starting Glyphh Runtime on {host}:{port}", fg=theme.TEXT)
    click.secho(f"  Database: {_mask_db_url(db_url)}", fg=theme.TEXT_DIM)
    if reload:
        click.secho("  Auto-reload: enabled", fg=theme.TEXT_DIM)
    click.echo()

    import uvicorn
    uvicorn.run(
        "glyphh.server:app",
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
