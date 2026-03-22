"""
Interactive REPL shell for Glyphh CLI.

Supports the same <category> <function> commands as direct CLI subcommands.
"""

import click
from pathlib import Path

from .banner import print_banner
from .auth import is_logged_in, device_login, register_runtime
from .commands.auth import handle_auth
from .commands.model import handle_model
from .commands.token import handle_token
from .commands.query import handle_query
from .commands.chat import handle_chat
from .commands.dev import handle_dev
from .commands.config import handle_config
from .commands.docker import handle_docker
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
    "dev": handle_dev,
    "config": handle_config,
    "docker": handle_docker,
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

# Commands that take file/directory path arguments
_FILE_ARG_COMMANDS = {
    ("model", "load"),
    ("model", "deploy"),
    ("model", "package"),
    ("model", "init"),
    ("dev", "start"),
    ("dev", "restart"),
}

# Subcommands per category
_SUBCOMMANDS = {
    "auth": ["login", "logout", "status"],
    "model": ["list", "deploy", "load", "data", "count", "clear",
              "re-encode", "status", "undeploy", "init", "package", "test"],
    "token": ["create", "list", "revoke"],
    "query": [],
    "chat": [],
    "dev": ["start", "stop", "status", "log", "restart"],
    "config": ["show", "set", "clear"],
    "docker": ["init"],
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
        # Completing second word — subcommand
        cat = parts[0].lower()
        subs = _SUBCOMMANDS.get(cat, [])
        options = [s + " " for s in subs if s.startswith(text)]
    else:
        # Third word+ — file path completion
        # Expand ~ and complete paths
        prefix = text
        if prefix.startswith("~"):
            prefix = _os.path.expanduser(prefix)
        if _os.path.isdir(prefix) and not prefix.endswith(_os.sep):
            prefix += _os.sep
        matches = _glob.glob(prefix + "*")
        options = []
        for m in matches:
            # Re-apply ~ if user typed it
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


def _show_dev_server_info():
    """If a dev server is running, print its connection info."""
    import json as _json
    import os

    info_file = Path.home() / ".glyphh" / "dev.json"
    pid_file = Path.home() / ".glyphh" / "dev.pid"
    if not info_file.exists() or not pid_file.exists():
        return

    # Verify the process is actually running
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
    except (ValueError, ProcessLookupError, PermissionError):
        return

    try:
        info = _json.loads(info_file.read_text())
    except Exception:
        return

    name = info.get("model_name") or info.get("model_id") or "unknown"
    ver = info.get("model_version")
    label = f"{name} v{ver}" if ver else name

    click.echo()
    click.secho(f"  Dev Server  ·  {label}", fg=theme.TEXT, bold=True)
    click.echo()
    click.secho(f"  Storage:   {info.get('storage', '?')}", fg=theme.TEXT_DIM)
    if info.get("mcp_url"):
        click.secho(f"  MCP:       {info['mcp_url']}", fg=theme.ACCENT)
    port = info.get("port", 8002)
    click.secho(f"  Docs:      http://localhost:{port}/docs", fg=theme.TEXT_DIM)
    click.secho("  Auth:      none (local mode)", fg=theme.TEXT_DIM)
    click.echo()


@click.command()
@click.pass_context
def shell(ctx):
    """Start an interactive Glyphh shell."""
    setup_readline()
    print_banner()
    _show_dev_server_info()

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
    click.secho("    query <question>         Query the model", fg=theme.MUTED)
    click.echo()
    click.secho("  chat", fg=theme.ACCENT)
    click.secho("    chat                     Open interactive chat REPL", fg=theme.MUTED)
    click.secho("    chat <question>          Single query and return", fg=theme.MUTED)
    click.echo()
    click.secho("  dev", fg=theme.ACCENT)
    click.secho("    dev start [path]         Start dev server (background)", fg=theme.MUTED)
    click.secho("    dev stop                 Stop the dev server", fg=theme.MUTED)
    click.secho("    dev status               Show dev server status", fg=theme.MUTED)
    click.secho("    dev log [n]              Show last n lines of log (default 30)", fg=theme.MUTED)
    click.secho("    dev restart [path]       Restart the dev server", fg=theme.MUTED)
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
        "auth", "model", "token", "query", "chat", "dev", "config", "docker", "license",
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
