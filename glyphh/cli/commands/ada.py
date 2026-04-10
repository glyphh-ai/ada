"""
CLI dream and memory commands.

The brain handles all conversation via the MCP think tool.
These commands provide direct access to dream loop and memory management.
"""

import os
import shutil

import click

from .. import theme

from glyphh.memory.glyph_dream import InsightKind


def _get_brain():
    """Get the live brain from the running server."""
    from glyphh.server import brain
    return brain


def _get_cognitive():
    """Get the live AdaCognitive from the running brain."""
    b = _get_brain()
    if b is None:
        return None
    return b.cognitive


# ── Dream commands ───────────────────────────────────────────────────────

def handle_dream(func: str | None, args: str = ""):
    """Route dream subcommands from the interactive shell."""
    cmd = func.strip().lower() if func else "status"
    cognitive = _get_cognitive()

    if cognitive is None:
        click.secho("  Brain not running.", fg=theme.ERROR)
        return

    if cmd in ("start", "on"):
        if cognitive.is_dreaming:
            click.secho("  Already dreaming.", fg=theme.TEXT_DIM)
        else:
            cognitive.start_dreaming()
            click.secho("  Background reasoning started.", fg=theme.ACCENT)

    elif cmd in ("stop", "off"):
        cognitive.stop_dreaming()
        click.secho("  Background reasoning stopped.", fg=theme.ACCENT)

    elif cmd in ("status", "stats"):
        stats = cognitive.dream_stats()
        space = cognitive.thought_space
        click.echo()
        click.secho(f"  running:    {'yes' if cognitive.is_dreaming else 'no'}", fg=theme.TEXT)
        click.secho(f"  localized:  {stats.get('localized_cycles', 0)} cycles (REM)", fg=theme.TEXT)
        click.secho(f"  deep:       {stats.get('deep_cycles', 0)} cycles (slow-wave)", fg=theme.TEXT)
        click.secho(f"  chains:     {stats.get('total_chains', 0)}", fg=theme.TEXT)
        click.secho(f"  insights:   {stats.get('total_insights', 0)} ({stats.get('queued_insights', 0)} pending)", fg=theme.TEXT)
        click.secho(f"  thoughts:   {stats.get('thoughts', 0)} glyphs", fg=theme.TEXT)
        click.secho(f"  pathways:   {stats.get('pathways', 0)} activation patterns", fg=theme.TEXT)
        click.secho(f"  primitives: {space.primitives.count} words -> {len(space.primitives.all_roles())} roles", fg=theme.TEXT)

    elif cmd in ("insights", "thoughts"):
        insights = cognitive.drain_insights()
        if not insights:
            click.secho("  No new insights.", fg=theme.TEXT_DIM)
        else:
            _show_insights(insights)

    else:
        click.secho("  dream [status|start|stop|insights]", fg=theme.TEXT_DIM)


# ── Memory commands ──────────────────────────────────────────────────────

def handle_memory(func: str | None, args: str = ""):
    """Route memory subcommands from the interactive shell."""
    cmd = func.strip().lower() if func else ""

    if cmd == "reset":
        cognitive = _get_cognitive()
        if cognitive is not None:
            cognitive.reset()

        # Clear SQLite ada_thoughts table
        brain = _get_brain()
        if brain is not None and hasattr(brain, '_session_factory'):
            import asyncio
            from glyphh.memory.thought_persistence import clear_all_thoughts
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(clear_all_thoughts(brain._session_factory))
                loop.close()
            except Exception as e:
                click.secho(f"  Warning: could not clear database: {e}", fg=theme.WARNING)

        path = os.path.expanduser("~/.glyphh/memory")
        if os.path.exists(path):
            shutil.rmtree(path)
        click.secho("  Memory cleared.", fg=theme.ACCENT)
    else:
        click.secho("  memory reset              Clear all memory", fg=theme.TEXT_DIM)


# ── Derivative commands ──────────────────────────────────────────────────

def handle_derivative(func: str | None, args: str = ""):
    """Route derivative subcommands from the interactive shell."""
    brain = _get_brain()
    if brain is None:
        click.secho("  Brain not running.", fg=theme.ERROR)
        return

    deriv = brain.derivative
    cmd = func.strip().lower() if func else "status"

    if cmd in ("status", "show"):
        stats = deriv.stats()
        click.echo()
        click.secho(f"  signals:   {stats['total_signals']}", fg=theme.TEXT)
        click.secho(f"  strong:    {stats['strong_signals']}", fg=theme.TEXT)
        dims = stats.get("dimensions", {})
        if dims:
            for dim, count in dims.items():
                click.secho(f"  {dim:12s} {count} patterns", fg=theme.TEXT_DIM)
        style = deriv.style_summary()
        if style:
            click.echo()
            click.secho(f"  style:     {style}", fg=theme.ACCENT)
        click.echo()

    else:
        click.secho("  derivative                Show user derivative stats", fg=theme.TEXT_DIM)


# ── Insight display ──────────────────────────────────────────────────────

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
        click.secho(f"  ... and {remaining} more (type 'dream insights')", fg=theme.TEXT_DIM)
    click.echo()
