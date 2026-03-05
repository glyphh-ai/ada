"""
CLI dev command — zero-config local development server.

glyphh dev               Start from current directory (looks for manifest.yaml)
glyphh dev ./mymodel     Start from a specific model directory
glyphh dev --port 9000   Custom port
glyphh dev -d            Run as background daemon
glyphh dev stop          Stop the running daemon

No DATABASE_URL, no login, no token required. Uses SQLite by default.
MCP endpoint is auth-free in local mode.
"""

import json
import os
import sys
from pathlib import Path

import click

from .. import theme


_PID_DIR   = Path.home() / ".glyphh"
_PID_FILE  = _PID_DIR / "dev.pid"
_PORT_FILE = _PID_DIR / "dev.port"
_LOG_FILE  = _PID_DIR / "dev.log"

# Known subcommands — used to distinguish "glyphh dev stop" from "glyphh dev ./stop-dir"
_DEV_SUBCOMMANDS = {"stop"}


def _find_free_port(start: int, max_attempts: int = 20) -> int:
    """Find a free port starting from `start`, scanning upward."""
    import socket
    for offset in range(max_attempts):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start  # fallback — let uvicorn report the error


def _find_model_dir(start: Path) -> Path | None:
    """Look for manifest.yaml in start or its direct children."""
    if (start / "manifest.yaml").exists():
        return start
    for child in sorted(start.iterdir()):
        if child.is_dir() and (child / "manifest.yaml").exists():
            return child
    return None


def _read_manifest(model_dir: Path) -> dict:
    try:
        import yaml
        with open(model_dir / "manifest.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _run_server(path: str, port: int, no_reload: bool, daemon: bool) -> None:
    """Core logic: locate model, set env vars, start uvicorn."""
    # ── Check runtime dependencies ─────────────────────────────────────────
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
        import sqlalchemy  # noqa: F401
    except ImportError:
        click.secho("  Runtime dependencies not installed.", fg=theme.ERROR)
        click.secho("  Run: pip install 'glyphh[runtime]'", fg=theme.ACCENT)
        sys.exit(1)

    # ── Find the model directory ───────────────────────────────────────────
    target = Path(path).resolve()
    model_dir = _find_model_dir(target)

    model_id = None
    model_name = None
    model_version = None

    if model_dir:
        manifest = _read_manifest(model_dir)
        model_id = manifest.get("model_id") or model_dir.name
        model_name = manifest.get("name") or model_id
        model_version = manifest.get("version", "")
    else:
        click.secho(
            f"  No manifest.yaml found in {target} or its subdirectories.",
            fg=theme.WARNING,
        )
        click.secho(
            "  Starting server without a pre-loaded model.",
            fg=theme.MUTED,
        )

    # ── Find a free port ──────────────────────────────────────────────────
    actual_port = _find_free_port(port)
    if actual_port != port:
        click.secho(
            f"  Port {port} in use — using {actual_port} instead.",
            fg=theme.WARNING,
        )
    port = actual_port

    # ── Set environment variables before uvicorn starts ────────────────────
    os.environ.setdefault("DEPLOYMENT_MODE", "local")
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./glyphh_dev.db")

    if model_dir:
        os.environ["GLYPHH_DEV_MODEL_DIR"] = str(model_dir)

    # Clear lru_cache so the server picks up the fresh env vars
    try:
        from infrastructure.config import get_settings
        get_settings.cache_clear()
    except ImportError:
        pass  # Server-side modules not yet imported — will pick up env naturally

    # ── Determine effective storage label for display ──────────────────────
    db_url = os.environ.get("DATABASE_URL", "")
    storage_label = "SQLite (glyphh_dev.db)"
    if "postgresql" in db_url or "postgres" in db_url:
        import re
        masked = re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", db_url)
        storage_label = f"pgvector ({masked})"
    elif "sqlite" in db_url and ":memory:" in db_url:
        storage_label = "SQLite (in-memory)"
    elif "sqlite" in db_url:
        db_file = db_url.split("///")[-1]
        storage_label = f"SQLite ({db_file})"

    # Effective org_id in local mode (matches auth.py mock user)
    org_id = "local-dev-org"
    mcp_url  = f"http://localhost:{port}/{org_id}/{model_id}/mcp"  if model_id else None

    # ── Print startup header ───────────────────────────────────────────────
    click.echo()
    if model_name and model_version:
        click.secho(
            f"  Glyphh Dev  ·  {model_name} v{model_version}",
            fg=theme.TEXT,
            bold=True,
        )
    elif model_name:
        click.secho(f"  Glyphh Dev  ·  {model_name}", fg=theme.TEXT, bold=True)
    else:
        click.secho("  Glyphh Dev", fg=theme.TEXT, bold=True)
    click.echo()

    click.secho(f"  Storage:   {storage_label}", fg=theme.TEXT_DIM)
    if mcp_url:
        click.secho(f"  MCP:       {mcp_url}", fg=theme.ACCENT)
    click.secho(f"  Docs:      http://localhost:{port}/docs", fg=theme.TEXT_DIM)
    click.secho("  Auth:      none (local mode)", fg=theme.TEXT_DIM)

    if daemon:
        click.secho(f"  Log:       {_LOG_FILE}", fg=theme.TEXT_DIM)
        click.secho(f"  PID file:  {_PID_FILE}", fg=theme.TEXT_DIM)

    click.echo()

    if mcp_url:
        mcp_config = json.dumps(
            {
                "mcpServers": {
                    model_id: {
                        "url": mcp_url,
                    }
                }
            },
            indent=4,
        )
        click.secho("  Add to Claude Desktop / Claude Code:", fg=theme.MUTED)
        click.echo()
        for line in mcp_config.splitlines():
            click.secho(f"    {line}", fg=theme.INFO)
        click.echo()

    # ── Persist dev server info so `glyphh` shell can display it ────────
    _PID_DIR.mkdir(parents=True, exist_ok=True)
    _PORT_FILE.write_text(str(port))
    _DEV_INFO = _PID_DIR / "dev.json"
    _DEV_INFO.write_text(json.dumps({
        "port": port,
        "org_id": org_id,
        "model_id": model_id,
        "model_name": model_name,
        "model_version": model_version,
        "mcp_url": mcp_url,
        "storage": storage_label,
    }))

    # ── Daemon mode ────────────────────────────────────────────────────────
    if daemon:
        _start_daemon(port, no_reload)
        return

    # ── Foreground mode ────────────────────────────────────────────────────
    import uvicorn

    try:
        uvicorn.run(
            "glyphh.server:app",
            host="0.0.0.0",
            port=port,
            reload=not no_reload,
            workers=1,
            log_level="warning",
        )
    finally:
        _PORT_FILE.unlink(missing_ok=True)
        (_PID_DIR / "dev.json").unlink(missing_ok=True)


def _start_daemon(port: int, no_reload: bool) -> None:
    """Fork uvicorn as a background process, write PID file."""
    import subprocess

    _PID_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing daemon
    if _PID_FILE.exists():
        try:
            pid = int(_PID_FILE.read_text().strip())
            os.kill(pid, 0)  # Signal 0 = check if process exists
            click.secho(
                f"  Dev server already running (PID {pid}). "
                "Run 'glyphh dev stop' to stop it.",
                fg=theme.WARNING,
            )
            return
        except (ProcessLookupError, ValueError):
            _PID_FILE.unlink(missing_ok=True)  # Stale PID

    # Build the uvicorn command using the same Python interpreter
    cmd = [
        sys.executable, "-m", "uvicorn",
        "glyphh.server:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--workers", "1",
        "--log-level", "warning",
    ]
    if not no_reload:
        cmd.append("--reload")

    log_fd = open(_LOG_FILE, "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=log_fd,
        start_new_session=True,
        env=os.environ.copy(),
    )

    _PID_FILE.write_text(str(proc.pid))

    click.secho(
        f"  Dev server started in background (PID {proc.pid}).",
        fg=theme.TEXT,
        bold=True,
    )
    click.secho("  Stop with: glyphh dev stop", fg=theme.TEXT_DIM)
    click.echo()


# ── Custom group that steals path-like first arg before Click sees it ─────────

class _DevGroup(click.Group):
    """Click Group subclass that treats the first non-option, non-subcommand arg
    as the model directory path, allowing 'glyphh dev .' to work alongside
    'glyphh dev stop'."""

    def parse_args(self, ctx: click.Context, args: list) -> list:
        # If the first arg doesn't start with '-' and isn't a known subcommand,
        # capture it as the path and remove it from args before Click tries to
        # look it up as a subcommand name.
        if args and not args[0].startswith("-") and args[0] not in _DEV_SUBCOMMANDS:
            ctx.meta["_dev_path"] = args[0]
            args = list(args[1:])
        return super().parse_args(ctx, args)


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group("dev", cls=_DevGroup, invoke_without_command=True)
@click.option("--port", "-p", default=8002, type=int, help="Port (default: 8002)")
@click.option("--no-reload", is_flag=True, default=False, help="Disable auto-reload")
@click.option("--daemon", "-d", is_flag=True, default=False, help="Run server in background")
@click.pass_context
def dev_group(ctx, port, no_reload, daemon):
    """Start a zero-config local dev server for a Glyphh model.

    Requires: pip install glyphh[runtime]

    Sets DEPLOYMENT_MODE=local automatically — no account, no token needed.
    MCP endpoint is open in local mode.
    Storage defaults to SQLite (glyphh_dev.db) if DATABASE_URL is not set.

    \b
    Examples:
      glyphh dev .            # start from current directory (foreground)
      glyphh dev ./mymodel    # start from specific model directory
      glyphh dev . -d         # start as background daemon
      glyphh dev stop         # stop the background daemon
    """
    if ctx.invoked_subcommand is not None:
        return  # Delegate to subcommand

    path = ctx.meta.get("_dev_path", ".")
    _run_server(path, port, no_reload, daemon)


@dev_group.command("stop")
def dev_stop():
    """Stop the background dev server started with 'glyphh dev -d'."""
    import signal

    if not _PID_FILE.exists():
        click.secho("  No dev server PID file found. Is a daemon running?", fg=theme.WARNING)
        return

    try:
        pid = int(_PID_FILE.read_text().strip())
    except ValueError:
        click.secho("  PID file is malformed.", fg=theme.ERROR)
        _PID_FILE.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, 0)  # Check process exists
    except ProcessLookupError:
        click.secho(f"  Process {pid} not found — removing stale PID file.", fg=theme.TEXT_DIM)
        _PID_FILE.unlink(missing_ok=True)
        _PORT_FILE.unlink(missing_ok=True)
        (_PID_DIR / "dev.json").unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
        _PID_FILE.unlink(missing_ok=True)
        _PORT_FILE.unlink(missing_ok=True)
        (_PID_DIR / "dev.json").unlink(missing_ok=True)
        click.secho(f"  Dev server stopped (PID {pid}).", fg=theme.TEXT, bold=True)
    except PermissionError:
        click.secho(f"  Permission denied stopping PID {pid}.", fg=theme.ERROR)
    except Exception as exc:
        click.secho(f"  Failed to stop server: {exc}", fg=theme.ERROR)


# ── Shell handler ─────────────────────────────────────────────────────────────

def handle_dev(func: str | None, args: str = ""):
    """Route dev subcommands from the interactive shell."""
    if func is None:
        func = "status"

    if func == "start":
        # Check if already running before printing the full header
        if _PID_FILE.exists():
            try:
                pid = int(_PID_FILE.read_text().strip())
                os.kill(pid, 0)
                click.secho(
                    f"  Dev server already running (PID {pid}). "
                    "Use 'dev restart' to restart.",
                    fg=theme.WARNING,
                )
                return
            except (ProcessLookupError, ValueError):
                _PID_FILE.unlink(missing_ok=True)  # stale
        path = args.strip() or "."
        _run_server(path, port=8002, no_reload=False, daemon=True)

    elif func == "stop":
        import signal

        if not _PID_FILE.exists():
            click.secho("  No dev server running.", fg=theme.WARNING)
            return
        try:
            pid = int(_PID_FILE.read_text().strip())
        except ValueError:
            click.secho("  PID file is malformed.", fg=theme.ERROR)
            _PID_FILE.unlink(missing_ok=True)
            return
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            click.secho(f"  Process {pid} not found — removing stale PID file.", fg=theme.TEXT_DIM)
            _PID_FILE.unlink(missing_ok=True)
            _PORT_FILE.unlink(missing_ok=True)
            (_PID_DIR / "dev.json").unlink(missing_ok=True)
            return
        try:
            os.kill(pid, signal.SIGTERM)
            _PID_FILE.unlink(missing_ok=True)
            _PORT_FILE.unlink(missing_ok=True)
            (_PID_DIR / "dev.json").unlink(missing_ok=True)
            click.secho(f"  Dev server stopped (PID {pid}).", fg=theme.TEXT, bold=True)
        except PermissionError:
            click.secho(f"  Permission denied stopping PID {pid}.", fg=theme.ERROR)
        except Exception as exc:
            click.secho(f"  Failed to stop server: {exc}", fg=theme.ERROR)

    elif func == "status":
        if not _PID_FILE.exists():
            click.secho("  No dev server running.", fg=theme.TEXT_DIM)
            return
        try:
            pid = int(_PID_FILE.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError):
            click.secho("  No dev server running (stale PID file).", fg=theme.TEXT_DIM)
            _PID_FILE.unlink(missing_ok=True)
            _PORT_FILE.unlink(missing_ok=True)
            (_PID_DIR / "dev.json").unlink(missing_ok=True)
            return

        # Read dev.json for details
        info_file = _PID_DIR / "dev.json"
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
                name = info.get("model_name") or info.get("model_id") or "unknown"
                ver = info.get("model_version")
                label = f"{name} v{ver}" if ver else name
                click.echo()
                click.secho(f"  Dev Server  ·  {label}  (PID {pid})", fg=theme.TEXT, bold=True)
                click.echo()
                click.secho(f"  Storage:   {info.get('storage', '?')}", fg=theme.TEXT_DIM)
                if info.get("mcp_url"):
                    click.secho(f"  MCP:       {info['mcp_url']}", fg=theme.ACCENT)
                port = info.get("port", 8002)
                click.secho(f"  Docs:      http://localhost:{port}/docs", fg=theme.TEXT_DIM)
                click.echo()
            except Exception:
                click.secho(f"  Dev server running (PID {pid}).", fg=theme.TEXT)
        else:
            click.secho(f"  Dev server running (PID {pid}).", fg=theme.TEXT)

    elif func in ("log", "logs", "tail"):
        if not _LOG_FILE.exists():
            click.secho("  No log file found.", fg=theme.TEXT_DIM)
            return
        # Show last N lines
        n = 30
        if args.strip().isdigit():
            n = int(args.strip())
        try:
            lines = _LOG_FILE.read_text().splitlines()
            tail = lines[-n:] if len(lines) > n else lines
            if not tail:
                click.secho("  Log file is empty.", fg=theme.TEXT_DIM)
                return
            click.echo()
            for line in tail:
                click.secho(f"  {line}", fg=theme.TEXT_DIM)
            click.echo()
            click.secho(f"  ({len(lines)} total lines, showing last {len(tail)})", fg=theme.MUTED)
        except Exception as exc:
            click.secho(f"  Failed to read log: {exc}", fg=theme.ERROR)

    elif func == "restart":
        # Stop then start
        handle_dev("stop", "")
        import time
        time.sleep(0.5)
        path = args.strip() or "."
        _run_server(path, port=8002, no_reload=False, daemon=True)

    else:
        click.secho(f"  Unknown dev command: {func}", fg=theme.WARNING)
        click.secho("  Available: start, stop, status, log, restart", fg=theme.TEXT_DIM)
