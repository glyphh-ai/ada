"""Shared helpers for screen rendering."""

import click
from .. import theme
from ..streaming import stream_text

# Fixed layout width — no dynamic sizing
_COLS = 120


def get_cols() -> int:
    """Return fixed column width. No dynamic detection."""
    return _COLS


def is_narrow() -> bool:
    return False


def box_width() -> int:
    return min(67, _COLS - 4)


def print_box(title: str, lines: list[str], color: str = theme.PRIMARY):
    """Print a bordered box."""
    w = box_width()
    inner = w - 4

    click.echo()
    stream_text(f"  ╭{'─' * (w - 2)}╮", fg=color)
    stream_text(f"  │{' ' * inner}  │", fg=color)

    # Title
    t = title[:inner]
    click.secho("  │  ", fg=color, nl=False)
    stream_text(t, fg=theme.TEXT_HIGHLIGHT, nl=False)
    click.secho(" " * (inner - len(t) - 2) + "  │", fg=color)

    stream_text(f"  │{' ' * inner}  │", fg=color)

    for line in lines:
        text = line[:inner]
        pad = inner - len(text)
        click.secho("  │  ", fg=color, nl=False)
        stream_text(text, fg=color, nl=False)
        click.secho(" " * max(0, pad - 2) + "  │", fg=color)

    stream_text(f"  │{' ' * inner}  │", fg=color)
    stream_text(f"  ╰{'─' * (w - 2)}╯", fg=color)
    click.echo()
