"""
Interactive REPL shell for Glyphh CLI.
"""

import click
from typing import Optional
from pathlib import Path

from .banner import print_banner
from .auth import is_logged_in, device_login, get_user, register_runtime
from . import theme

# Try to import readline for history/completion
try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

HISTORY_FILE = Path.home() / ".glyphh" / "history"


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

    # macOS uses libedit
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

    # Check if logged in — if not, prompt to login
    if not is_logged_in():
        click.echo()
        click.secho("  Not logged in.", fg=theme.WARNING)
        click.echo()
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
    else:
        user = get_user() or {}
        name = user.get("first_name", user.get("email", ""))
        if name:
            click.secho(f"  Logged in as {name}", fg=theme.MUTED)
            click.echo()
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
                    click.echo()
                    click.secho("  exit, quit, q   Exit the shell", fg=theme.MUTED)
                    click.secho("  clear, home     Clear screen and show banner", fg=theme.MUTED)
                    click.secho("  logout          Log out and clear session", fg=theme.MUTED)
                    click.secho("  help            Show this message", fg=theme.MUTED)
                    click.echo()
                    continue
                elif line.lower() == "logout":
                    from .auth import clear_session
                    clear_session()
                    click.secho("  Logged out.", fg=theme.MUTED)
                    break

                # Unrecognized input
                click.secho(f"  unknown command: {line}", fg=theme.MUTED)

            except KeyboardInterrupt:
                click.echo()
                continue
            except EOFError:
                break

    finally:
        save_history()
        click.echo()
        click.secho("Goodbye!", fg="cyan")
