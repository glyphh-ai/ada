"""
Banner for the Glyphh CLI shell.
"""

import click
from importlib.metadata import version as pkg_version
from . import theme
from .streaming import stream_text
from .auth import is_logged_in, get_user


def _get_runtime_version() -> str:
    try:
        return pkg_version("glyphh")
    except Exception:
        return "?"


def print_banner():
    """Print the welcome banner."""
    click.echo()
    stream_text("        _             _     _             _", fg=theme.PRIMARY)
    stream_text("   __ _| |_   _ _ __ | |__ | |__     __ _(_)", fg=theme.PRIMARY)
    stream_text("  / _` | | | | | '_ \\| '_ \\| '_ \\   / _` | |", fg=theme.PRIMARY)
    stream_text(" | (_| | | |_| | |_) | | | | | | | | (_| | |", fg=theme.ACCENT)
    stream_text("  \\__, |_|\\__, | .__/|_| |_|_| |_|  \\__,_|_|", fg="cyan")
    stream_text("  |___/   |___/|_|", fg="bright_cyan")
    click.echo()
    stream_text("  when your llm can't afford to be wrong", fg="bright_cyan")
    ver = _get_runtime_version()
    stream_text(f"  version: Ada {ver}", fg=theme.TEXT_DIM)
    click.echo()
    _print_status()


def _print_status():
    """Print auth status line below the banner."""
    if is_logged_in():
        user = get_user() or {}
        name = user.get("first_name", user.get("email", ""))
        dot = click.style("●", fg=theme.SUCCESS)
        label = click.style(f" logged in as {name}" if name else " logged in", fg=theme.MUTED)
        hint = click.style("  auth logout", fg=theme.PRIMARY) + click.style(" to exit", fg=theme.TEXT_DIM)
        click.echo(f"  {dot}{label}  {hint}")
    else:
        dot = click.style("●", fg=theme.ERROR)
        label = click.style(" not logged in", fg=theme.MUTED)
        hint = click.style("  auth login", fg=theme.PRIMARY) + click.style(" to connect", fg=theme.TEXT_DIM)
        click.echo(f"  {dot}{label}  {hint}")
    click.echo()
