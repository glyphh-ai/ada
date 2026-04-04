"""
Interactive REPL shell for Glyphh CLI.

After login, starts the runtime server in a daemon thread (dies when
the shell exits — no PID files, no background daemon, no state files).
"""

import os
import socket
import sys
import threading

import click
from pathlib import Path

from .banner import print_banner
from .auth import is_logged_in, device_login, register_runtime
from .commands.auth import handle_auth
from .commands.model import handle_model
from .commands.token import handle_token
from .commands.query import handle_query
from .commands.chat import handle_chat
from .commands.ada import handle_ada
from .commands.config import handle_config
from .commands.docker import handle_docker
from .commands.hub import handle_hub
from .commands.license import handle_license
from . import theme

# Try to import readline for history/completion
try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

HISTORY_FILE = Path.home() / ".glyphh" / "history"

# Command routing table: category -> handler
COMMAND_HANDLERS = {
    "auth": handle_auth,
    "model": handle_model,
    "token": handle_token,
    "query": handle_query,
    "chat": handle_chat,
    "ada": handle_ada,
    "config": handle_config,
    "docker": handle_docker,
    "hub": handle_hub,
    "license": handle_license,
}


def _discover_plugins():
    """Discover installed model plugins via entry points.

    Plugins register under the 'glyphh.plugins' group:
        [project.entry-points."glyphh.plugins"]
        code = "glyphh_code.plugin:register"

    Each register() function returns a dict:
        {"handler": callable, "subcommands": ["init", "compile", "status"]}
    """
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        # Python 3.12+ returns a SelectableGroups, older returns dict
        if hasattr(eps, "select"):
            plugin_eps = eps.select(group="glyphh.plugins")
        else:
            plugin_eps = eps.get("glyphh.plugins", [])

        for ep in plugin_eps:
            try:
                register_fn = ep.load()
                plugin = register_fn()
                handler = plugin.get("handler")
                subcommands = plugin.get("subcommands", [])
                if handler:
                    COMMAND_HANDLERS[ep.name] = handler
                    _SUBCOMMANDS[ep.name] = subcommands
            except Exception:
                pass  # Skip broken plugins silently
    except Exception:
        pass


# Discover plugins on import
_discover_plugins()


import glob as _glob
import os as _os

# Cache deployed model IDs for tab completion (refreshed periodically)
_deployed_ids_cache: list[str] = []
_deployed_ids_ts: float = 0


def _get_deployed_model_ids() -> list[str]:
    """Return cached list of deployed model IDs for tab completion."""
    import time
    global _deployed_ids_cache, _deployed_ids_ts
    now = time.time()
    if now - _deployed_ids_ts < 10:  # cache for 10s
        return _deployed_ids_cache
    try:
        from .config import resolve_runtime_url, resolve_runtime_token
        from .auth import resolve_org_id
        import httpx
        runtime_url = resolve_runtime_url()
        token = resolve_runtime_token()
        org_id = resolve_org_id(runtime_url)
        if not org_id:
            return _deployed_ids_cache
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with httpx.Client(timeout=2) as client:
            res = client.get(f"{runtime_url}/{org_id}/models", headers=headers)
        if res.status_code == 200:
            _deployed_ids_cache = [m["model_id"] for m in res.json().get("models", [])]
            _deployed_ids_ts = now
    except Exception:
        pass
    return _deployed_ids_cache

# Commands that take file/directory path arguments
_FILE_ARG_COMMANDS = {
    ("model", "load"),
    ("model", "deploy"),
    ("model", "package"),
    ("model", "init"),
}

# Subcommands per category
_SUBCOMMANDS = {
    "auth": ["login", "logout", "status"],
    "model": ["list", "deploy", "load", "data", "count", "clear",
              "re-encode", "status", "undeploy", "init", "package", "test"],
    "token": ["create", "list", "revoke"],
    "query": [],
    "chat": [],
    "ada": [],
    "config": ["show", "set", "clear"],
    "docker": ["init"],
    "hub": ["list", "search", "install"],
    "license": ["show", "activate", "deactivate", "refresh"],
}

_CATEGORIES = list(_SUBCOMMANDS.keys()) + ["help", "clear", "home", "exit", "quit"]


def _completer(text, state):
    """Tab completer: commands for first two words, file paths for arguments."""
    line = readline.get_line_buffer()
    parts = line.split()
    # Number of complete words (if line ends with space, cursor is on next word)
    n_complete = len(parts) if line.endswith(" ") else max(0, len(parts) - 1)

    if n_complete == 0:
        # Completing first word — category
        options = [c + " " for c in _CATEGORIES if c.startswith(text)]
    elif n_complete == 1:
        # Completing second word — subcommand (+ deployed model IDs for "model")
        cat = parts[0].lower()
        subs = list(_SUBCOMMANDS.get(cat, []))
        if cat == "model":
            subs.extend(_get_deployed_model_ids())
        elif cat in ("chat", "query"):
            subs.extend(_get_deployed_model_ids())
        options = [s + " " for s in subs if s.startswith(text)]
    else:
        # Third word+ — file path completion
        prefix = text
        if prefix.startswith("~"):
            prefix = _os.path.expanduser(prefix)
        if _os.path.isdir(prefix) and not prefix.endswith(_os.sep):
            prefix += _os.sep
        matches = _glob.glob(prefix + "*")
        options = []
        for m in matches:
            display = m
            if text.startswith("~"):
                home = _os.path.expanduser("~")
                if display.startswith(home):
                    display = "~" + display[len(home):]
            if _os.path.isdir(m):
                display += _os.sep
            options.append(display)

    try:
        return options[state]
    except IndexError:
        return None


def setup_readline():
    """Setup readline for history and tab completion."""
    if not HAS_READLINE:
        return

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if HISTORY_FILE.exists():
        try:
            readline.read_history_file(str(HISTORY_FILE))
        except Exception:
            pass

    readline.set_history_length(1000)

    # Set up tab completion
    readline.set_completer(_completer)
    readline.set_completer_delims(" \t")

    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


def save_history():
    """Save command history."""
    if HAS_READLINE:
        try:
            readline.write_history_file(str(HISTORY_FILE))
        except Exception:
            pass


def get_prompt() -> str:
    """Get the shell prompt."""
    return click.style("glyphh", fg=theme.PRIMARY) + click.style("> ", fg=theme.TEXT)


# ── Embedded runtime server ─────────────────────────────────────────────────


def _find_free_port(start: int = 8002, max_tries: int = 20) -> int:
    """Find a free port starting from `start`."""
    for offset in range(max_tries):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _start_embedded_server() -> int | None:
    """Start uvicorn in a daemon thread. Returns the port, or None on failure."""
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
    except ImportError:
        click.secho("  Runtime dependencies not installed.", fg=theme.ERROR)
        click.secho("  Run: pip install glyphh", fg=theme.ACCENT)
        return None

    # Set up environment for the embedded server
    if "DATABASE_URL" not in os.environ:
        db_dir = Path.home() / ".glyphh"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "local.db"
        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"

    os.environ.setdefault("ENABLE_DOCS", "true")
    os.environ.setdefault("CORS_ALLOW_ALL", "true")

    # Clear settings cache so server picks up fresh env
    try:
        from infrastructure.config import get_settings
        get_settings.cache_clear()
    except ImportError:
        pass

    port = _find_free_port()

    # Redirect all server logging to a file so the shell stays clean
    import logging
    log_dir = Path.home() / ".glyphh"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "runtime.log"

    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "message": "%(message)s"}'
    ))

    # Route all loggers to file, remove console handlers
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)

    # Suppress noisy libraries from even the log file
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    import uvicorn

    config = uvicorn.Config(
        "glyphh.server:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Poll /health until ready
    import time
    import httpx

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=2) as client:
                res = client.get(f"http://127.0.0.1:{port}/health")
                if res.status_code == 200:
                    return port
        except Exception:
            pass
        time.sleep(0.3)

    click.secho("  Server failed to start within 30s.", fg=theme.ERROR)
    return None


def _provision_runtime_token(port: int) -> str | None:
    """Mint a local database token so all API calls validate locally.

    Uses the Platform JWT (once) to authenticate with the token endpoint,
    then saves the resulting glyphh_xxxx token as runtime_token in config.
    Subsequent calls via resolve_runtime_token() will use the local token
    and skip Platform JWT validation entirely.
    """
    import httpx

    from .auth import get_token, get_org_id, _load_config, _save_config

    # Check if we already have a working runtime token
    config = _load_config()
    existing = config.get("runtime_token", "").strip()
    if existing and existing.startswith("glyphh_"):
        # Verify it still works
        org_id = get_org_id()
        if org_id:
            try:
                with httpx.Client(timeout=5) as client:
                    res = client.get(
                        f"http://127.0.0.1:{port}/{org_id}/models",
                        headers={"Authorization": f"Bearer {existing}"},
                    )
                    if res.status_code != 401:
                        return existing
            except Exception:
                pass

    # Need to mint a new token using Platform JWT
    platform_jwt = get_token()
    org_id = get_org_id()
    if not platform_jwt or not org_id:
        return None

    try:
        with httpx.Client(timeout=10) as client:
            res = client.post(
                f"http://127.0.0.1:{port}/{org_id}/tokens",
                headers={"Authorization": f"Bearer {platform_jwt}"},
                json={
                    "name": "cli-session",
                    "permissions": ["read", "write", "admin"],
                    "expires_days": 365,
                },
            )
            if res.status_code != 200:
                return None

            raw_token = res.json().get("token")
            if not raw_token:
                return None

            # Save to config so resolve_runtime_token() picks it up
            config = _load_config()
            config["runtime_token"] = raw_token
            _save_config(config)
            return raw_token
    except Exception:
        return None


# ── Shell entry point ────────────────────────────────────────────────────────


@click.command()
@click.pass_context
def shell(ctx):
    """Start an interactive Glyphh shell."""
    setup_readline()
    print_banner()

    # If not logged in, prompt once
    if not is_logged_in():
        click.secho("  Press Enter to open the browser and log in, or type 'q' to quit.", fg=theme.MUTED)
        try:
            resp = input("  ")
        except (KeyboardInterrupt, EOFError):
            click.echo()
            return
        if resp.strip().lower() in ("q", "quit", "exit"):
            return
        success = device_login()
        if not success:
            click.echo()
            click.secho("  Run 'glyphh' again after logging in.", fg=theme.MUTED)
            return
        click.echo()
    else:
        register_runtime()

    # Start embedded runtime server
    click.secho("  Starting runtime...", fg=theme.MUTED)
    port = _start_embedded_server()
    if port is None:
        return

    # Mint a local database token for all subsequent API calls
    rt = _provision_runtime_token(port)
    if rt:
        click.secho("  Token:     local database token active", fg=theme.TEXT_DIM)
    else:
        click.secho("  Warning: could not provision runtime token — using Platform JWT", fg=theme.WARNING)

    url = f"http://localhost:{port}"
    click.echo()
    click.secho(f"  Runtime:   {url}", fg=theme.ACCENT)
    if os.environ.get("ENABLE_DOCS") == "true":
        click.secho(f"  API docs:  {url}/docs", fg=theme.TEXT_DIM)
    click.secho(f"  Logs:      ~/.glyphh/runtime.log", fg=theme.TEXT_DIM)
    click.echo()

    try:
        while True:
            try:
                prompt = get_prompt()
                line = input(prompt).strip()

                if not line:
                    continue

                if line.lower() in ("exit", "quit", "q"):
                    break
                elif line.lower() in ("clear", "home"):
                    click.clear()
                    print_banner()
                    continue
                elif line.lower() == "help":
                    _print_help()
                    continue
                elif line.startswith("!"):
                    # Shell escape: run arbitrary system commands
                    import subprocess
                    subprocess.run(line[1:].strip(), shell=True)
                    continue

                # Parse <category> <function> [args] format
                parts = line.split(None, 2)
                category = parts[0].lower()
                func = parts[1].lower() if len(parts) > 1 else None
                args = parts[2] if len(parts) > 2 else ""

                handler = COMMAND_HANDLERS.get(category)
                if handler:
                    handler(func, args)
                    # Exit shell after auth logout
                    if category == "auth" and func == "logout":
                        break
                else:
                    click.secho(f"  unknown command: {line}", fg=theme.MUTED)
                    click.secho("  type 'help' for available commands", fg=theme.TEXT_DIM)

            except KeyboardInterrupt:
                click.echo()
                continue
            except EOFError:
                break

    finally:
        save_history()
        click.echo()
        click.secho("Goodbye!", fg="cyan")


def _print_help():
    """Print available commands grouped by category."""
    click.echo()
    click.secho("  auth", fg=theme.ACCENT)
    click.secho("    auth login              Log in via browser", fg=theme.MUTED)
    click.secho("    auth logout             Log out and clear session", fg=theme.MUTED)
    click.secho("    auth status             Show auth status", fg=theme.MUTED)
    click.echo()
    click.secho("  model", fg=theme.ACCENT)
    click.secho("    model list              List deployed models", fg=theme.MUTED)
    click.secho("    model deploy [path]     Deploy model to runtime", fg=theme.MUTED)
    click.secho("    model load <file>       Load data from concepts.json", fg=theme.MUTED)
    click.secho("    model data              View stored glyphs", fg=theme.MUTED)
    click.secho("    model count             Show glyph/vector counts", fg=theme.MUTED)
    click.secho("    model clear             Clear all data (keep model)", fg=theme.MUTED)
    click.secho("    model re-encode         Re-encode all glyphs", fg=theme.MUTED)
    click.secho("    model status [id]       Check deployed status", fg=theme.MUTED)
    click.secho("    model undeploy [id]     Remove from runtime", fg=theme.MUTED)
    click.secho("    model init [name]       Scaffold new model", fg=theme.MUTED)
    click.secho("    model package [path]    Create .glyphh file", fg=theme.MUTED)
    click.echo()
    click.secho("  token", fg=theme.ACCENT)
    click.secho("    token create             Create an API token", fg=theme.MUTED)
    click.secho("    token list               List active tokens", fg=theme.MUTED)
    click.secho("    token revoke <id>        Revoke a token", fg=theme.MUTED)
    click.echo()
    click.secho("  query", fg=theme.ACCENT)
    click.secho("    query <model-id> <question>  Query a deployed model", fg=theme.MUTED)
    click.echo()
    click.secho("  chat", fg=theme.ACCENT)
    click.secho("    chat <model-id>          Open interactive chat REPL", fg=theme.MUTED)
    click.secho("    chat <model-id> <query>  Single query and return", fg=theme.MUTED)
    click.echo()
    click.secho("  ada", fg=theme.ACCENT)
    click.secho("    ada                      Talk to Ada (local LLM)", fg=theme.MUTED)
    click.secho("    ada <question>           Single query and return", fg=theme.MUTED)
    click.echo()
    click.secho("  hub", fg=theme.ACCENT)
    click.secho("    hub                      Browse model registry", fg=theme.MUTED)
    click.secho("    hub list                 Browse with paginated cards", fg=theme.MUTED)
    click.secho("    hub search <query>       Search by name, tag, category", fg=theme.MUTED)
    click.secho("    hub install <id>         Install a model from the registry", fg=theme.MUTED)
    click.echo()
    click.secho("  docker", fg=theme.ACCENT)
    click.secho("    docker init [--force]    Write docker-compose.yml + init.sql", fg=theme.MUTED)
    click.echo()
    click.secho("  license", fg=theme.ACCENT)
    click.secho("    license show             Display current license info", fg=theme.MUTED)
    click.secho("    license activate <jwt>   Activate a license token", fg=theme.MUTED)
    click.secho("    license deactivate       Remove license (free tier)", fg=theme.MUTED)
    click.secho("    license refresh          Re-fetch license from Platform", fg=theme.MUTED)
    click.echo()
    click.secho("  config", fg=theme.ACCENT)
    click.secho("    config show              Show current configuration", fg=theme.MUTED)
    click.secho("    config set endpoint <url> Set runtime endpoint", fg=theme.MUTED)
    click.secho("    config set token <jwt>   Set runtime auth token", fg=theme.MUTED)
    click.secho("    config clear             Clear all config", fg=theme.MUTED)
    click.echo()
    # Show installed plugin commands
    builtin_categories = {
        "auth", "model", "token", "query", "chat", "config", "docker", "hub", "license",
    }
    plugin_categories = [c for c in _SUBCOMMANDS if c not in builtin_categories]
    if plugin_categories:
        click.secho("  plugins", fg=theme.ACCENT)
        for cat in sorted(plugin_categories):
            subs = _SUBCOMMANDS.get(cat, [])
            sub_str = ", ".join(subs) if subs else "<command>"
            click.secho(f"    {cat} {sub_str}", fg=theme.MUTED)
        click.echo()

    click.secho("  general", fg=theme.ACCENT)
    click.secho("    clear, home             Clear screen and show banner", fg=theme.MUTED)
    click.secho("    !<command>              Run a shell command (e.g. !python3 script.py)", fg=theme.MUTED)
    click.secho("    exit, quit, q           Exit the shell", fg=theme.MUTED)
    click.secho("    help                    Show this message", fg=theme.MUTED)
    click.echo()
    click.secho("  background server", fg=theme.ACCENT)
    click.secho("    The shell embeds the runtime — it stops when you quit.", fg=theme.MUTED)
    click.secho("    To keep it running for MCP and listeners:", fg=theme.MUTED)
    click.secho("      glyphh serve          Foreground (Ctrl+C to stop)", fg=theme.TEXT_DIM)
    click.secho("      glyphh serve &        Background (shell job)", fg=theme.TEXT_DIM)
    click.echo()
