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

        brain = _get_brain()
        if brain is not None and hasattr(brain, '_session_factory'):
            import asyncio
            from glyphh.memory.thought_persistence import clear_all_thoughts
            from domains.brain.thread_persistence import clear_all_threads
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(clear_all_thoughts(brain._session_factory))
                loop.run_until_complete(clear_all_threads(brain._session_factory))
                loop.close()
            except Exception as e:
                click.secho(f"  Warning: could not clear database: {e}", fg=theme.WARNING)

        # Clear thread store in memory
        if brain is not None and hasattr(brain, '_thread_store'):
            from domains.brain.context_thread import ThreadStore
            brain._thread_store = ThreadStore()

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


# ── Recall commands ─────────────────────────────────────────────────────

def handle_recall(func: str | None, args: str = ""):
    """Query context threads and thought space."""
    brain = _get_brain()
    if brain is None:
        click.secho("  Brain not running.", fg=theme.ERROR)
        return

    # Combine func and args as the query text
    query = ""
    if func:
        query = func
        if args:
            query += " " + args
    query = query.strip()

    store = brain.thread_store

    if not query:
        # No query — show all threads, then thought count
        threads = store.all_threads()
        click.echo()
        if threads:
            click.secho(f"  {len(threads)} context threads:", fg=theme.ACCENT, bold=True)
            click.echo()
            for t in threads:
                status = "●" if t.active else "○"
                age = _format_age(t.updated_at)
                entities_str = ", ".join(t.entities[:5]) if t.entities else "-"
                click.secho(f"  {status} [{t.tool}] {t.topic}", fg=theme.ACCENT)
                click.secho(f"    entities: {entities_str}", fg=theme.TEXT)
                click.secho(f"    facts:    {len(t.facts)}  turns: {t.turn_count}  {age}", fg=theme.TEXT_DIM)
                for fact in t.facts[:3]:
                    click.secho(f"      · {fact}", fg=theme.TEXT)
                if len(t.facts) > 3:
                    click.secho(f"      ... and {len(t.facts) - 3} more", fg=theme.TEXT_DIM)
                click.echo()
        else:
            click.secho("  No context threads yet.", fg=theme.TEXT_DIM)
            click.echo()

        # Also show thought space count
        space = brain.cognitive.thought_space
        click.secho(f"  {space.count} thoughts in HDC space", fg=theme.TEXT_DIM)
        click.echo()
        return

    # Search: try threads first, then HDC fallback
    from domains.brain.thread_manager import ThreadManager, _extract_implicit_entities

    entities = _extract_implicit_entities(query)
    thread_results = store.recall(entities=entities if entities else None, limit=5)

    click.echo()
    click.secho(f"  recall: \"{query}\"", fg=theme.ACCENT, bold=True)
    if entities:
        click.secho(f"  entities: {', '.join(entities)}", fg=theme.TEXT_DIM)
    click.echo()

    if thread_results:
        click.secho("  ── threads ──", fg=theme.ACCENT)
        for t in thread_results:
            click.secho(f"  [{t.tool}] {t.topic}  ({len(t.facts)} facts)", fg="green")
            for fact in t.facts:
                if fact.startswith("[Q] "):
                    click.secho(f"    ? {fact[4:]}", fg=theme.TEXT_DIM)
                else:
                    click.secho(f"    · {fact}", fg=theme.TEXT)
        click.echo()

    # Also show HDC results for comparison
    space = brain.cognitive.thought_space
    results = space.recall(query, top_k=5)
    if results:
        click.secho("  ── HDC similarity ──", fg=theme.TEXT_DIM)
        for r in results[:5]:
            sim = r.global_similarity
            color = "green" if sim >= 0.5 else "yellow" if sim >= 0.3 else theme.TEXT_DIM
            click.secho(f"  {sim:.3f}  {r.thought.content}", fg=color)
        click.echo()


def _format_age(timestamp: float) -> str:
    """Format a timestamp as a human-readable age."""
    import time
    age_s = time.time() - timestamp
    if age_s < 60:
        return "just now"
    if age_s < 3600:
        return f"{int(age_s / 60)}m ago"
    if age_s < 86400:
        return f"{int(age_s / 3600)}h ago"
    return f"{int(age_s / 86400)}d ago"


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
