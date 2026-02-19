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
from .commands.catalog import handle_catalog
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
    "catalog": handle_catalog,
}


def setup_readline():
    """Setup readline for history."""
    if not HAS_READLINE:
        return

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if HISTORY_FILE.exists():
        try:
            readline.read_history_file(str(HISTORY_FILE))
        except Exception:
            pass

    readline.set_history_length(1000)

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
    click.secho("    model list              List local models", fg=theme.MUTED)
    click.secho("    model deploy [path]     Deploy model to runtime", fg=theme.MUTED)
    click.secho("    model status [id]       Check deployed status", fg=theme.MUTED)
    click.secho("    model undeploy [id]     Remove from runtime", fg=theme.MUTED)
    click.secho("    model init [name]       Scaffold new model", fg=theme.MUTED)
    click.secho("    model package [path]    Create .glyphh file", fg=theme.MUTED)
    click.echo()
    click.secho("  catalog", fg=theme.ACCENT)
    click.secho("    catalog list             Browse platform models", fg=theme.MUTED)
    click.secho("    catalog search <query>   Search by name/category", fg=theme.MUTED)
    click.secho("    catalog download <name>  Download .glyphh model", fg=theme.MUTED)
    click.secho("    catalog info <name>      Show model details", fg=theme.MUTED)
    click.echo()
    click.secho("  general", fg=theme.ACCENT)
    click.secho("    clear, home             Clear screen and show banner", fg=theme.MUTED)
    click.secho("    exit, quit, q           Exit the shell", fg=theme.MUTED)
    click.secho("    help                    Show this message", fg=theme.MUTED)
    click.echo()
