"""
CLI ada subcommand — dream control and memory reset.

The brain handles all conversation via the MCP think tool.
These commands provide direct access to dream loop and memory management.
"""

import os
import shutil

import click

from .. import theme

from glyphh.memory.ada_cognitive import AdaCognitive
from glyphh.memory.glyph_dream import InsightKind


# ── Ada singleton ─────────────────────────────────────────────────────────

_MEMORY_DIR = os.path.expanduser("~/.glyphh/memory")
_ada: AdaCognitive | None = None


def _get_ada() -> AdaCognitive:
    global _ada
    if _ada is None:
        _ada = AdaCognitive()
    return _ada


# ── Commands ──────────────────────────────────────────────────────────────

def _do_reset() -> None:
    """Clear all of Ada's memory."""
    global _ada
    if _ada is not None:
        _ada.reset()
        _ada = None
    path = os.path.expanduser("~/.glyphh/memory")
    if os.path.exists(path):
        shutil.rmtree(path)
    click.secho("  Memory cleared.", fg=theme.ACCENT)


def _do_dream(text: str) -> None:
    """Control background reasoning."""
    ada = _get_ada()
    dream = ada._ensure_dream()
    cmd = text.strip().lower() if text else ""

    if cmd in ("start", "on", ""):
        if dream.is_running:
            click.secho("  Already thinking.", fg=theme.TEXT_DIM)
        else:
            ada.start_dreaming()
            click.secho("  background reasoning started", fg=theme.ACCENT)

    elif cmd in ("stop", "off"):
        ada.stop_dreaming()
        click.secho("  background reasoning stopped", fg=theme.ACCENT)

    elif cmd in ("status", "stats"):
        stats = ada.dream_stats()
        space = ada.thought_space
        click.echo()
        click.secho(f"  running:    {'yes' if ada.is_dreaming else 'no'}", fg=theme.TEXT)
        click.secho(f"  localized:  {stats.get('localized_cycles', 0)} cycles (REM)", fg=theme.TEXT)
        click.secho(f"  deep:       {stats.get('deep_cycles', 0)} cycles (slow-wave)", fg=theme.TEXT)
        click.secho(f"  chains:     {stats.get('total_chains', 0)}", fg=theme.TEXT)
        click.secho(f"  insights:   {stats.get('total_insights', 0)} ({stats.get('queued_insights', 0)} pending)", fg=theme.TEXT)
        click.secho(f"  thoughts:   {stats.get('thoughts', 0)} glyphs", fg=theme.TEXT)
        click.secho(f"  pathways:   {stats.get('pathways', 0)} activation patterns", fg=theme.TEXT)
        click.secho(f"  primitives: {space.primitives.count} words -> {len(space.primitives.all_roles())} roles", fg=theme.TEXT)

    elif cmd in ("insights", "thoughts"):
        insights = ada.drain_insights()
        if not insights:
            click.secho("  No new insights.", fg=theme.TEXT_DIM)
        else:
            _show_insights(insights)

    else:
        click.secho("  dream [start|stop|status|insights]", fg=theme.TEXT_DIM)


_INSIGHT_ICONS = {
    InsightKind.CONNECTION: ("!", theme.ACCENT),
    InsightKind.CONTRADICTION: ("!", "yellow"),
    InsightKind.CONVERGENCE: ("+", "cyan"),
    InsightKind.QUESTION: ("?", "yellow"),
    InsightKind.CRYSTALLIZATION: ("*", "bright_cyan"),
}


def _show_insights(insights, max_show: int = 5) -> None:
    if not insights:
        return
    click.echo()
    click.secho("  Ada noticed:", fg=theme.TEXT_DIM, bold=True)
    for insight in insights[:max_show]:
        icon, color = _INSIGHT_ICONS.get(insight.kind, (".", theme.TEXT))
        click.secho(f"  {icon} {insight.summary}", fg=color)
    remaining = len(insights) - max_show
    if remaining > 0:
        click.secho(f"  ... and {remaining} more (type 'ada dream insights')", fg=theme.TEXT_DIM)
    click.echo()


# ── CLI command (glyphh ada) ──────────────────────────────────────────────

@click.command("ada")
@click.argument("action", required=False)
@click.argument("text", required=False, nargs=-1)
def ada_command(action, text):
    """Ada brain commands — dream control and memory reset."""
    if action and action.lower() in ("reset", "dream"):
        args = " ".join(text) if text else ""
        if action.lower() == "reset":
            _do_reset()
        else:
            _do_dream(args)
        return

    click.secho("  ada dream [status|start|stop|insights]", fg=theme.MUTED)
    click.secho("  ada reset", fg=theme.MUTED)


# ── Handler for interactive shell ─────────────────────────────────────────

def handle_ada(func: str | None, args: str = ""):
    """Route ada subcommands from the interactive shell."""
    full = " ".join(p for p in [func, args] if p).strip()

    if full:
        parts = full.split(None, 1)
        cmd = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""
        if cmd == "reset":
            _do_reset()
            return
        if cmd == "dream":
            _do_dream(cmd_args)
            return

    click.secho("  ada dream [status|start|stop|insights]", fg=theme.MUTED)
    click.secho("  ada reset", fg=theme.MUTED)
