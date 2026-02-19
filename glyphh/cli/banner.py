"""
Banner for the Glyphh CLI shell.
"""

import click
from . import theme
from .streaming import stream_text


def print_banner():
    """Print the welcome banner."""
    click.echo()
    stream_text("        _             _     _             _", fg=theme.PRIMARY)
    stream_text("   __ _| |_   _ _ __ | |__ | |__     __ _(_)", fg=theme.ACCENT)
    stream_text("  / _` | | | | | '_ \\| '_ \\| '_ \\   / _` | |", fg="cyan")
    stream_text(" | (_| | | |_| | |_) | | | | | | | | (_| | |", fg="cyan")
    stream_text("  \\__, |_|\\__, | .__/|_| |_|_| |_|  \\__,_|_|", fg="bright_cyan")
    stream_text("  |___/   |___/|_|", fg="bright_cyan")
    click.echo()
    stream_text("  when your llm can't afford to be wrong", fg="bright_cyan")
    click.echo()
