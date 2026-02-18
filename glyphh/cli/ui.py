"""
UI utilities for the Glyphh CLI.
Provides styled boxes, error displays, and other visual elements.
"""

import click
from typing import Optional, List


# Box drawing characters
BOX_CHARS = {
    "tl": "╭",  # top-left
    "tr": "╮",  # top-right
    "bl": "╰",  # bottom-left
    "br": "╯",  # bottom-right
    "h": "─",   # horizontal
    "v": "│",   # vertical
}

# Alternative sharp corners
BOX_SHARP = {
    "tl": "┌",
    "tr": "┐",
    "bl": "└",
    "br": "┘",
    "h": "─",
    "v": "│",
}


def print_box(
    title: str,
    message: str,
    color: str = "white",
    border_color: str = "bright_black",
    width: int = 60,
    icon: Optional[str] = None,
    sharp: bool = False,
):
    """Print a styled box with title and message."""
    # Prepare title with icon
    if icon:
        title = f"{icon} {title}"

    chars = BOX_SHARP if sharp else BOX_CHARS

    # Fixed width
    width = min(width, 60)
    if width < 20:
        width = 20

    # Calculate inner width
    inner_width = width - 4  # borders + padding

    # Wrap message lines
    lines = []
    for line in message.split("\n"):
        while len(line) > inner_width:
            break_at = line.rfind(" ", 0, inner_width)
            if break_at == -1:
                break_at = inner_width
            lines.append(line[:break_at])
            line = line[break_at:].lstrip()
        lines.append(line)

    click.echo()

    # Top border with title
    title_display = f" {title} "
    if len(title_display) > width - 4:
        title_display = f" {title[:width - 6]} "
    remaining = width - len(title_display) - 2
    left_pad = 2
    right_pad = max(0, remaining - left_pad)

    click.secho(f"  {chars['tl']}{chars['h'] * left_pad}", fg=border_color, nl=False)
    click.secho(title_display, fg=color, bold=True, nl=False)
    click.secho(f"{chars['h'] * right_pad}{chars['tr']}", fg=border_color)

    # Empty line after title
    click.secho(f"  {chars['v']}", fg=border_color, nl=False)
    click.echo(" " * (width - 2), nl=False)
    click.secho(chars['v'], fg=border_color)

    # Message lines
    for line in lines:
        padding = inner_width - len(line)
        click.secho(f"  {chars['v']} ", fg=border_color, nl=False)
        click.secho(line, fg=color, nl=False)
        click.echo(" " * (padding + 1), nl=False)
        click.secho(chars['v'], fg=border_color)

    # Empty line before bottom
    click.secho(f"  {chars['v']}", fg=border_color, nl=False)
    click.echo(" " * (width - 2), nl=False)
    click.secho(chars['v'], fg=border_color)

    # Bottom border
    click.secho(f"  {chars['bl']}{chars['h'] * (width - 2)}{chars['br']}", fg=border_color)

    click.echo()


def print_error(title: str, message: str, suggestions: Optional[List[str]] = None):
    """Print a styled error box."""
    print_box(
        title=title,
        message=message,
        color="red",
        border_color="red",
        icon="✗",
    )
    
    if suggestions:
        click.secho("  Suggestions:", fg="yellow")
        for suggestion in suggestions:
            text = suggestion[:76] if len(suggestion) > 76 else suggestion
            click.echo(f"    • {text}")
        click.echo()


def print_warning(title: str, message: str):
    """Print a styled warning box."""
    print_box(
        title=title,
        message=message,
        color="yellow",
        border_color="yellow",
        icon="⚠",
    )


def print_success(title: str, message: str):
    """Print a styled success box."""
    print_box(
        title=title,
        message=message,
        color="green",
        border_color="green",
        icon="✓",
    )


def print_info(title: str, message: str):
    """Print a styled info box."""
    print_box(
        title=title,
        message=message,
        color="cyan",
        border_color="cyan",
        icon="ℹ",
    )


def print_command_error(command: str, error: str, hint: Optional[str] = None):
    """Print a command execution error with context."""
    message = f"Command '{command}' failed:\n{error}"
    if hint:
        message += f"\n\nHint: {hint}"
    print_error("Command Error", message)


def print_connection_error(url: str):
    """Print a connection error with helpful suggestions."""
    print_error(
        "Connection Failed",
        f"Could not connect to:\n{url}",
        suggestions=[
            "Check your internet connection",
            "Verify the API URL is correct",
            "Try again in a few moments",
            "Run 'auth set-api <url>' to change the API endpoint",
        ]
    )


def print_auth_error(action: str = "perform this action"):
    """Print an authentication error."""
    print_error(
        "Authentication Required",
        f"You must be logged in to {action}.",
        suggestions=[
            "Run 'auth login' to sign in",
            "Run 'auth signup' to create an account",
        ]
    )


def print_not_found_error(resource: str, name: str):
    """Print a resource not found error."""
    print_error(
        "Not Found",
        f"{resource} '{name}' was not found.",
        suggestions=[
            f"Check the {resource.lower()} name is correct",
            f"Run '{resource.lower()}s' to list available {resource.lower()}s",
        ]
    )
