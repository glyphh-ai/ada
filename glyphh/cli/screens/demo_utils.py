"""
Shared utilities for demo reel animations.

Provides banner rendering, step prompts, simulated typing,
status lines, and similarity tree display used across all reels.
"""

import sys
import os
import time
import click
from .. import theme
from ..streaming import stream_text, stream_echo


# Speeds
TYPE_CPS = 55        # user "typing" speed
SYSTEM_CPS = 1000    # system output speed
FLOW_CPS = 500       # narrative / response text


class DemoExit(Exception):
    """Raised when user wants to exit the demo."""
    pass


def get_cols() -> int:
    from .helpers import get_cols as _get_cols
    return _get_cols()


def print_banner_static():
    """Print a compact, non-streaming banner for the demo header."""
    click.echo()
    click.secho("        _             _     _             _", fg=theme.PRIMARY)
    click.secho("   __ _| |_   _ _ __ | |__ | |__     __ _(_)", fg=theme.ACCENT)
    click.secho("  / _` | | | | | '_ \\| '_ \\| '_ \\   / _` | |", fg="cyan")
    click.secho(" | (_| | | |_| | |_) | | | | | | | | (_| | |", fg="cyan")
    click.secho("  \\__, |_|\\__, | .__/|_| |_|_| |_|  \\__,_|_|", fg="bright_cyan")
    click.secho("  |___/   |___/|_|", fg="bright_cyan")
    click.echo()
    click.secho("  when your llm can't afford to be wrong", fg="bright_cyan")
    click.echo()


def clear_and_banner():
    """Clear screen and redraw the static banner."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    print_banner_static()


def step_prompt(step: int, total: int, label: str):
    """Show a step indicator and wait for user to press enter."""
    click.echo()
    click.secho(f"  [{step}/{total}] ", fg="cyan", nl=False)
    click.secho(label, fg=theme.MUTED, nl=False)
    click.secho("  press enter to continue", fg=theme.MUTED, nl=False)
    try:
        input("")
    except (EOFError, KeyboardInterrupt):
        raise DemoExit()


def type_prompt(text: str, prompt: str = "glyphh> "):
    """Simulate a user typing into the prompt."""
    click.secho(f"  {prompt}", fg=theme.PRIMARY, nl=False)
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(1.0 / TYPE_CPS)
    sys.stdout.write("\n")
    sys.stdout.flush()


def status_line(icon: str, label: str, value: str,
                icon_color: str = theme.MUTED, value_color: str = theme.TEXT):
    """Print a status/step line with streaming value."""
    click.secho(f"  {icon} ", fg=icon_color, nl=False)
    click.secho(f"{label}: ", fg=theme.MUTED, nl=False)
    stream_text(value, fg=value_color, cps=SYSTEM_CPS)


def flow_arrow():
    """Print a downward flow arrow."""
    stream_text("     ↓", fg=theme.ACCENT, cps=SYSTEM_CPS)


def header(text: str):
    """Print a section header."""
    stream_text(f"  {text}", fg=theme.TEXT_HIGHLIGHT, cps=SYSTEM_CPS)
    click.echo()


def similarity_tree(cortex: str, layer: str, segment: str, role: str,
                    color: str = theme.MUTED):
    """Print a similarity breakdown tree: cortex → layer → segment → role."""
    stream_text(f"    ├─ cortex:  {cortex}", fg=color, cps=SYSTEM_CPS)
    stream_text(f"    ├─ layer:   {layer}", fg=color, cps=SYSTEM_CPS)
    stream_text(f"    ├─ segment: {segment}", fg=color, cps=SYSTEM_CPS)
    stream_text(f"    └─ role:    {role}", fg=color, cps=SYSTEM_CPS)


def run_reel(scenes: list[tuple[str, callable]]):
    """Run a list of (label, scene_fn) as a step-by-step demo reel."""
    total = len(scenes)

    try:
        for i, (label, scene_fn) in enumerate(scenes, 1):
            clear_and_banner()

            # Step header
            click.secho(f"  ── demo ", fg=theme.ACCENT, bold=True, nl=False)
            click.secho(f"[{i}/{total}] ", fg="cyan", nl=False)
            click.secho(label, fg=theme.MUTED)
            click.echo()

            # Play the scene
            scene_fn()

            # Wait for user to advance (skip on last scene)
            if i < total:
                step_prompt(i, total, "next: " + scenes[i][0])

    except DemoExit:
        pass

    click.echo()
