"""
SSE-style streaming text effect for the Glyphh CLI.

Renders text character-by-character left-to-right, line-by-line,
giving the appearance of server-sent event streaming.
"""

import sys
import time
import click
from typing import Optional


# Default characters-per-second for the streaming effect
DEFAULT_CPS = 800


def stream_text(
    text: str,
    fg: Optional[str] = None,
    bold: bool = False,
    cps: int = DEFAULT_CPS,
    nl: bool = True,
):
    """
    Print text character-by-character with optional color.

    Args:
        text: The string to stream.
        fg: Click color name (e.g. 'cyan', 'magenta').
        bold: Whether to render bold.
        cps: Characters per second (higher = faster).
        nl: Whether to print a newline at the end.
    """
    delay = 1.0 / cps
    styled = click.style(text, fg=fg, bold=bold) if (fg or bold) else text

    # We need to emit raw characters one at a time, but ANSI codes
    # should be emitted atomically. Walk the styled string and detect
    # escape sequences.
    i = 0
    while i < len(styled):
        if styled[i] == '\x1b':
            # Consume the entire ANSI escape sequence at once
            j = i + 1
            while j < len(styled) and styled[j] != 'm':
                j += 1
            sys.stdout.write(styled[i:j + 1])
            i = j + 1
        else:
            sys.stdout.write(styled[i])
            sys.stdout.flush()
            time.sleep(delay)
            i += 1

    if nl:
        sys.stdout.write('\n')
    sys.stdout.flush()


def stream_lines(
    lines: list[str],
    fg: Optional[str] = None,
    bold: bool = False,
    cps: int = DEFAULT_CPS,
    prefix: str = "",
):
    """
    Stream multiple lines sequentially, each left-to-right.

    Args:
        lines: List of text lines.
        fg: Click color name.
        bold: Bold text.
        cps: Characters per second.
        prefix: String prepended to each line (e.g. '  ' for indent).
    """
    for line in lines:
        full = f"{prefix}{line}"
        stream_text(full, fg=fg, bold=bold, cps=cps)


def stream_echo(
    *segments: tuple,
    cps: int = DEFAULT_CPS,
    nl: bool = True,
):
    """
    Stream a line composed of multiple styled segments.

    Each segment is a tuple of (text, fg, bold).
    Example:
        stream_echo(
            ("  -> ", "cyan", False),
            ("docs", "cyan", True),
            (" full guide", "bright_black", False),
        )
    """
    delay = 1.0 / cps
    for text, fg, bold in segments:
        styled = click.style(text, fg=fg, bold=bold) if (fg or bold) else text
        i = 0
        while i < len(styled):
            if styled[i] == '\x1b':
                j = i + 1
                while j < len(styled) and styled[j] != 'm':
                    j += 1
                sys.stdout.write(styled[i:j + 1])
                i = j + 1
            else:
                sys.stdout.write(styled[i])
                sys.stdout.flush()
                time.sleep(delay)
                i += 1

    if nl:
        sys.stdout.write('\n')
    sys.stdout.flush()
