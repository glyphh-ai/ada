"""
UI components for the Glyphh CLI.

Provides cards, tables, and other visual elements that adapt
to terminal width without wrapping awkwardly.
"""

import os
import click
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from . import theme


def get_terminal_width() -> int:
    """Get terminal width, with a sensible default."""
    from .screens.helpers import get_cols
    return get_cols()


def truncate(text: str, max_len: int, suffix: str = "…") -> str:
    """Truncate text to max length with suffix."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


# ============================================================================
# Cards
# ============================================================================

@dataclass
class Card:
    """A card component for displaying structured information."""
    title: str
    subtitle: Optional[str] = None
    body: Optional[str] = None
    footer: Optional[str] = None
    status: Optional[str] = None  # "success", "warning", "error", "info"
    metadata: Optional[Dict[str, str]] = None
    actions: Optional[List[str]] = None


def print_card(card: Card, width: Optional[int] = None, stream: bool = False):
    """Print a styled card. If stream=True, body text is rendered character-by-character."""
    from .streaming import stream_text as _stream_text

    term_width = width or get_terminal_width()
    card_width = min(60, term_width - 4)
    inner_width = card_width - 4
    
    # Determine border color based on status
    border_color = theme.MUTED
    status_icon = ""
    if card.status == "success":
        border_color = theme.SUCCESS
        status_icon = "✓ "
    elif card.status == "warning":
        border_color = theme.WARNING
        status_icon = "⚠ "
    elif card.status == "error":
        border_color = theme.ERROR
        status_icon = "✗ "
    elif card.status == "info":
        border_color = theme.TEXT_HIGHLIGHT
        status_icon = "ℹ "
    
    click.echo()
    
    # Top border
    click.secho(f"  ╭{'─' * (card_width - 2)}╮", fg=border_color)
    
    # Title row
    title_text = f"{status_icon}{card.title}"
    title_display = truncate(title_text, inner_width)
    padding = inner_width - len(title_display)
    click.secho("  │ ", fg=border_color, nl=False)
    click.secho(title_display, fg=theme.TEXT, bold=True, nl=False)
    click.echo(" " * padding, nl=False)
    click.secho(" │", fg=border_color)
    
    # Subtitle
    if card.subtitle:
        sub_display = truncate(card.subtitle, inner_width)
        padding = inner_width - len(sub_display)
        click.secho("  │ ", fg=border_color, nl=False)
        click.secho(sub_display, fg=theme.MUTED, nl=False)
        click.echo(" " * padding, nl=False)
        click.secho(" │", fg=border_color)
    
    # Separator after title
    click.secho(f"  ├{'─' * (card_width - 2)}┤", fg=border_color)
    
    # Body
    if card.body:
        # Word wrap body text
        words = card.body.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= inner_width:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            padding = inner_width - len(line)
            click.secho("  │ ", fg=border_color, nl=False)
            if stream:
                _stream_text(line, fg=theme.TEXT, nl=False)
            else:
                click.secho(line, fg=theme.TEXT, nl=False)
            click.echo(" " * padding, nl=False)
            click.secho(" │", fg=border_color)
    
    # Metadata (key: value pairs)
    if card.metadata:
        click.secho(f"  ├{'─' * (card_width - 2)}┤", fg=border_color)
        for key, value in card.metadata.items():
            kv_text = f"{key}: {value}"
            kv_display = truncate(kv_text, inner_width)
            padding = inner_width - len(kv_display)
            click.secho("  │ ", fg=border_color, nl=False)
            click.secho(f"{key}: ", fg=theme.MUTED, nl=False)
            remaining = inner_width - len(key) - 2
            click.secho(truncate(value, remaining), fg=theme.TEXT_HIGHLIGHT, nl=False)
            actual_len = len(key) + 2 + min(len(value), remaining)
            click.echo(" " * (inner_width - actual_len), nl=False)
            click.secho(" │", fg=border_color)
    
    # Actions
    if card.actions:
        click.secho(f"  ├{'─' * (card_width - 2)}┤", fg=border_color)
        actions_text = "  ".join(f"[{a}]" for a in card.actions)
        actions_display = truncate(actions_text, inner_width)
        padding = inner_width - len(actions_display)
        click.secho("  │ ", fg=border_color, nl=False)
        click.secho(actions_display, fg=theme.PRIMARY, nl=False)
        click.echo(" " * padding, nl=False)
        click.secho(" │", fg=border_color)
    
    # Footer
    if card.footer:
        click.secho(f"  ├{'─' * (card_width - 2)}┤", fg=border_color)
        footer_display = truncate(card.footer, inner_width)
        padding = inner_width - len(footer_display)
        click.secho("  │ ", fg=border_color, nl=False)
        click.secho(footer_display, fg=theme.MUTED, nl=False)
        click.echo(" " * padding, nl=False)
        click.secho(" │", fg=border_color)
    
    # Bottom border
    click.secho(f"  ╰{'─' * (card_width - 2)}╯", fg=border_color)
    click.echo()


def print_card_grid(cards: List[Card], columns: int = 2):
    """Print cards in a grid layout."""
    # For now, just print cards vertically
    # TODO: Side-by-side layout for wide terminals
    for card in cards:
        print_card(card)


# ============================================================================
# Tables
# ============================================================================

def print_table(
    headers: List[str],
    rows: List[List[str]],
    title: Optional[str] = None,
    max_col_width: int = 30,
    border_color: str = theme.MUTED,
):
    """
    Print a formatted table that adapts to terminal width.
    
    Args:
        headers: Column headers
        rows: List of rows, each row is a list of cell values
        title: Optional table title
        max_col_width: Maximum width for any column
        border_color: Color for table borders
    """
    if not headers or not rows:
        return
    
    term_width = get_terminal_width()
    num_cols = len(headers)
    
    # Calculate column widths based on content
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for row in rows:
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(min(max_width, max_col_width))
    
    # Adjust if total width exceeds terminal
    total_width = sum(col_widths) + (num_cols * 3) + 1  # borders and padding
    if total_width > term_width - 4:
        # Shrink columns proportionally
        available = term_width - 4 - (num_cols * 3) - 1
        ratio = available / sum(col_widths)
        col_widths = [max(5, int(w * ratio)) for w in col_widths]
    
    # Build separator line
    sep_parts = ["─" * (w + 2) for w in col_widths]
    
    click.echo()
    
    # Title
    if title:
        click.secho(f"  {title}", fg=theme.TEXT, bold=True)
        click.echo()
    
    # Top border
    click.secho(f"  ┌{'┬'.join(sep_parts)}┐", fg=border_color)
    
    # Header row
    click.secho("  │", fg=border_color, nl=False)
    for i, header in enumerate(headers):
        cell = truncate(header, col_widths[i]).ljust(col_widths[i])
        click.secho(f" {cell} ", fg=theme.TEXT, bold=True, nl=False)
        click.secho("│", fg=border_color, nl=False)
    click.echo()
    
    # Header separator
    click.secho(f"  ├{'┼'.join(sep_parts)}┤", fg=border_color)
    
    # Data rows
    for row in rows:
        click.secho("  │", fg=border_color, nl=False)
        for i, col_width in enumerate(col_widths):
            cell_value = str(row[i]) if i < len(row) else ""
            cell = truncate(cell_value, col_width).ljust(col_width)
            click.secho(f" {cell} ", fg=theme.TEXT, nl=False)
            click.secho("│", fg=border_color, nl=False)
        click.echo()
    
    # Bottom border
    click.secho(f"  └{'┴'.join(sep_parts)}┘", fg=border_color)
    click.echo()


def print_key_value_list(
    items: Dict[str, Any],
    title: Optional[str] = None,
    key_color: str = theme.MUTED,
    value_color: str = theme.TEXT,
):
    """Print a key-value list with aligned values."""
    if not items:
        return
    
    term_width = get_terminal_width()
    max_key_len = max(len(k) for k in items.keys())
    max_val_width = term_width - max_key_len - 8  # padding and borders
    
    click.echo()
    
    if title:
        click.secho(f"  {title}", fg=theme.TEXT, bold=True)
        click.echo()
    
    for key, value in items.items():
        value_str = str(value)
        value_display = truncate(value_str, max_val_width)
        click.secho(f"  {key.rjust(max_key_len)}: ", fg=key_color, nl=False)
        click.secho(value_display, fg=value_color)
    
    click.echo()


# ============================================================================
# Lists
# ============================================================================

def print_list(
    items: List[str],
    title: Optional[str] = None,
    numbered: bool = False,
    bullet: str = "•",
    color: str = theme.TEXT,
):
    """Print a formatted list."""
    click.echo()
    
    if title:
        click.secho(f"  {title}", fg=theme.TEXT, bold=True)
        click.echo()
    
    term_width = get_terminal_width()
    max_item_width = term_width - 8
    
    for i, item in enumerate(items):
        prefix = f"{i + 1}." if numbered else bullet
        item_display = truncate(item, max_item_width)
        click.secho(f"    {prefix} ", fg=theme.MUTED, nl=False)
        click.secho(item_display, fg=color)
    
    click.echo()


# ============================================================================
# Progress & Status
# ============================================================================

def print_progress_bar(
    current: int,
    total: int,
    label: Optional[str] = None,
    width: int = 40,
    fill_char: str = "█",
    empty_char: str = "░",
):
    """Print a progress bar."""
    if total <= 0:
        return
    
    percent = min(current / total, 1.0)
    filled = int(width * percent)
    empty = width - filled
    
    bar = fill_char * filled + empty_char * empty
    percent_text = f"{percent * 100:.0f}%"
    
    click.secho("  ", nl=False)
    if label:
        click.secho(f"{label} ", fg=theme.MUTED, nl=False)
    click.secho("[", fg=theme.MUTED, nl=False)
    click.secho(bar[:filled], fg=theme.PRIMARY, nl=False)
    click.secho(bar[filled:], fg=theme.MUTED, nl=False)
    click.secho("] ", fg=theme.MUTED, nl=False)
    click.secho(percent_text, fg=theme.TEXT)


def print_status_line(
    label: str,
    status: str,
    status_type: str = "info",  # "success", "warning", "error", "info"
):
    """Print a status line with colored indicator."""
    icons = {
        "success": ("✓", theme.SUCCESS),
        "warning": ("⚠", theme.WARNING),
        "error": ("✗", theme.ERROR),
        "info": ("•", theme.TEXT_HIGHLIGHT),
    }
    icon, color = icons.get(status_type, ("•", theme.TEXT))
    
    click.secho(f"  {icon} ", fg=color, nl=False)
    click.secho(f"{label}: ", fg=theme.MUTED, nl=False)
    click.secho(status, fg=color)
