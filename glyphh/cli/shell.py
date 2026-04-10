"""
Interactive REPL shell for Glyphh CLI.

After login, starts the runtime server in a daemon thread (dies when
the shell exits — no PID files, no background daemon, no state files).

Default mode: free text goes to Ada's brain via the /mcp think tool.
Commands (auth, token, setup, etc.) are still available as prefixed words.
"""

import os
import socket
import sys
import threading

import click
from pathlib import Path

from .banner import print_banner
from .auth import is_logged_in, device_login, register_runtime
from .vault_env import load_vault_env, require_api_key
from .commands.auth import handle_auth
from .commands.token import handle_token
from .commands.ada import handle_dream, handle_memory, handle_derivative, handle_recall
from .commands.config import handle_config
from .commands.license import handle_license
from . import theme
from .spinner import GridSpinner

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
    "token": handle_token,
    "dream": handle_dream,
    "memory": handle_memory,
    "derivative": handle_derivative,
    "recall": handle_recall,
    "config": handle_config,
    "license": handle_license,
}


# Subcommands per category
_SUBCOMMANDS = {
    "auth": ["login", "logout", "status"],
    "token": ["create", "list", "revoke"],
    "dream": ["status", "start", "stop", "insights"],
    "memory": ["reset"],
    "derivative": ["status"],
    "recall": [],
    "config": ["show", "set", "clear"],
    "license": ["show", "activate", "deactivate", "refresh"],
    "setup": ["key", "model", "claude"],
}

_CATEGORIES = list(_SUBCOMMANDS.keys()) + ["help", "clear", "home", "exit", "quit"]


def _completer(text, state):
    """Tab completer: commands and subcommands."""
    line = readline.get_line_buffer()
    parts = line.split()
    n_complete = len(parts) if line.endswith(" ") else max(0, len(parts) - 1)

    if n_complete == 0:
        options = [c + " " for c in _CATEGORIES if c.startswith(text)]
    elif n_complete == 1:
        cat = parts[0].lower()
        subs = list(_SUBCOMMANDS.get(cat, []))
        options = [s + " " for s in subs if s.startswith(text)]
    else:
        options = []

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
    return click.style("ada", fg=theme.PRIMARY) + click.style("> ", fg=theme.TEXT)


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
                        f"http://127.0.0.1:{port}/health",
                        headers={"Authorization": f"Bearer {existing}"},
                    )
                    if res.status_code == 200:
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


# ── Ada brain interface ──────────────────────────────────────────────────────


def _think(text: str, port: int) -> None:
    """Send free text to Ada's brain via the MCP think tool."""
    import httpx
    import json

    try:
        # Call MCP endpoint directly via HTTP POST
        with GridSpinner("  ada> "):
            with httpx.Client(timeout=30) as client:
                res = client.post(
                    f"http://127.0.0.1:{port}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "think",
                            "arguments": {"input": text, "tool": "cli"},
                        },
                    },
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                )

        if res.status_code != 200:
            click.secho(f"  Error: server returned {res.status_code}", fg=theme.ERROR)
            return

        body = res.json()

        # MCP response: result.content[0].text contains JSON
        result = body.get("result", {})
        content = result.get("content", [])
        if not content:
            click.secho("  No response from Ada.", fg=theme.MUTED)
            return

        response_text = content[0].get("text", "")
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = {"response": response_text}

        response = data.get("response", "")
        capability = data.get("capability", "")
        confidence = data.get("confidence", 0)
        elapsed = data.get("elapsed_ms", 0)

        if data.get("error"):
            click.secho(f"  Error: {data['error']}", fg=theme.ERROR)
            return

        # Print response
        click.echo()
        click.secho("  ada", fg=theme.ACCENT, bold=True)

        # Word-wrap the response at ~80 cols
        if response:
            for line in response.split("\n"):
                col = 2
                click.echo("  ", nl=False)
                words = line.split(" ")
                for i, word in enumerate(words):
                    if col + len(word) + 1 > 80 and col > 2:
                        click.echo()
                        click.echo("  ", nl=False)
                        col = 2
                    if i > 0 and col > 2:
                        sys.stdout.write(" ")
                        col += 1
                    sys.stdout.write(word)
                    col += len(word)
                click.echo()
        else:
            click.secho("  (no response)", fg=theme.MUTED)

        # Debug line
        if capability or confidence:
            meta_parts = []
            if capability:
                meta_parts.append(capability)
            if confidence:
                meta_parts.append(f"{confidence:.0%}")
            if elapsed:
                meta_parts.append(f"{elapsed:.0f}ms")
            click.secho(f"  [{' · '.join(meta_parts)}]", fg=theme.TEXT_DIM)
        click.echo()

    except httpx.ConnectError:
        click.secho("  Error: cannot reach runtime server", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Error: {e}", fg=theme.ERROR)


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

    # Load vault secrets (API key, etc.) into os.environ
    load_vault_env()

    # Require Anthropic API key before starting
    if not require_api_key():
        click.secho("  Cannot start without an API key.", fg=theme.MUTED)
        return

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
    model = os.environ.get("ADA_MODEL", "claude-haiku-4-5-20251001")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    click.echo()
    click.secho(f"  MCP        {url}/mcp", fg=theme.ACCENT)
    click.secho(f"  Tool       think(input)", fg=theme.TEXT_DIM)
    click.secho(f"  LLM        {model}", fg=theme.TEXT_DIM)
    if has_key:
        click.echo(click.style("  Key        ", fg=theme.TEXT_DIM) + click.style("●", fg="green") + click.style(" set", fg=theme.TEXT_DIM))
    else:
        click.echo(click.style("  Key        ", fg=theme.TEXT_DIM) + click.style("○", fg=theme.WARNING) + click.style(" not set (run setup key)", fg=theme.TEXT_DIM))
    if os.environ.get("ENABLE_DOCS") == "true":
        click.secho(f"  Docs       {url}/docs", fg=theme.TEXT_DIM)
    click.secho(f"  Logs       ~/.glyphh/runtime.log", fg=theme.TEXT_DIM)
    click.echo()
    click.secho("  Just type. Ada hears everything. Type 'help' for commands.", fg=theme.MUTED)
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

                # Handle setup commands
                if line.lower().startswith("setup"):
                    _handle_setup(line)
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
                    # Not a command — send to Ada's brain
                    _think(line, port)

            except KeyboardInterrupt:
                click.echo()
                continue
            except EOFError:
                break

    finally:
        save_history()
        click.echo()
        click.secho("Goodbye!", fg="cyan")


def _handle_setup(line: str) -> None:
    """Handle setup commands (key, model, claude)."""
    from .vault_env import setup_key, setup_model, setup_claude_code

    parts = line.split()
    sub = parts[1].lower() if len(parts) > 1 else ""

    if sub == "key":
        setup_key()
    elif sub == "model":
        setup_model()
    elif sub in ("claude", "claude-code"):
        setup_claude_code()
    else:
        click.echo()
        click.secho("  setup key          Set Anthropic API key", fg=theme.MUTED)
        click.secho("  setup model        Change Ada's internal LLM model", fg=theme.MUTED)
        click.secho("  setup claude       Auto-configure Claude Code", fg=theme.MUTED)
        click.echo()


def _print_help():
    """Print available commands grouped by category."""
    click.echo()
    click.secho("  Just type anything — Ada hears it and responds.", fg=theme.TEXT)
    click.secho("  Commands below are prefixed words. Everything else goes to Ada.", fg=theme.MUTED)
    click.echo()
    click.secho("  dream", fg=theme.ACCENT)
    click.secho("    dream status            Background reasoning status", fg=theme.MUTED)
    click.secho("    dream start             Start background reasoning", fg=theme.MUTED)
    click.secho("    dream stop              Stop background reasoning", fg=theme.MUTED)
    click.secho("    dream insights          Show recent insights", fg=theme.MUTED)
    click.echo()
    click.secho("  memory", fg=theme.ACCENT)
    click.secho("    memory reset             Clear all memory", fg=theme.MUTED)
    click.echo()
    click.secho("  derivative", fg=theme.ACCENT)
    click.secho("    derivative               Show user derivative stats", fg=theme.MUTED)
    click.echo()
    click.secho("  recall", fg=theme.ACCENT)
    click.secho("    recall                   Show all thoughts in memory", fg=theme.MUTED)
    click.secho("    recall <query>           Search thought space by similarity", fg=theme.MUTED)
    click.echo()
    click.secho("  setup", fg=theme.ACCENT)
    click.secho("    setup key               Set Anthropic API key", fg=theme.MUTED)
    click.secho("    setup model             Change Ada's internal LLM model", fg=theme.MUTED)
    click.secho("    setup claude            Auto-configure Claude Code", fg=theme.MUTED)
    click.echo()
    click.secho("  auth", fg=theme.ACCENT)
    click.secho("    auth login              Log in via browser", fg=theme.MUTED)
    click.secho("    auth logout             Log out and clear session", fg=theme.MUTED)
    click.secho("    auth status             Show auth status", fg=theme.MUTED)
    click.echo()
    click.secho("  token", fg=theme.ACCENT)
    click.secho("    token create             Create an API token", fg=theme.MUTED)
    click.secho("    token list               List active tokens", fg=theme.MUTED)
    click.secho("    token revoke <id>        Revoke a token", fg=theme.MUTED)
    click.echo()
    click.secho("  license", fg=theme.ACCENT)
    click.secho("    license show             Display current license info", fg=theme.MUTED)
    click.secho("    license activate <jwt>   Activate a license token", fg=theme.MUTED)
    click.echo()
    click.secho("  config", fg=theme.ACCENT)
    click.secho("    config show              Show current configuration", fg=theme.MUTED)
    click.echo()
    click.secho("  general", fg=theme.ACCENT)
    click.secho("    clear, home             Clear screen and show banner", fg=theme.MUTED)
    click.secho("    !<command>              Run a shell command", fg=theme.MUTED)
    click.secho("    exit, quit, q           Exit the shell", fg=theme.MUTED)
    click.echo()
    click.secho("  server", fg=theme.ACCENT)
    click.secho("    The shell embeds the runtime — it stops when you quit.", fg=theme.MUTED)
    click.secho("      glyphh serve          Run standalone (Ctrl+C to stop)", fg=theme.TEXT_DIM)
    click.echo()
