"""
CLI ada command — Ada cognitive agent with HDC memory + local LLM.

No commands. Just talk to her. She absorbs everything — sentence by
sentence, like hearing speech. The DreamLoop decomposes what she heard
into structured facts in the background.

glyphh ada                    Interactive REPL
glyphh ada "my name is chris" Single message
glyphh ada reset              Clear all memory
glyphh ada dream status       Background reasoning
"""

import logging
import os
import re
import sys

import click

logger = logging.getLogger(__name__)

from .. import theme
from ..spinner import GridSpinner

from glyphh.memory.ada_cognitive import AdaCognitive, CognitiveResult, ADA_SYSTEM_PROMPT
from glyphh.memory.cognitive_glyph import Action, CognitiveState
from glyphh.memory.glyph_dream import InsightKind


# ── Readline history ────────────────────────────────────────────────────────

_HISTORY_FILE = os.path.expanduser("~/.glyphh/ada_history")

try:
    import readline as _readline

    def _setup_history() -> None:
        os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
        try:
            _readline.read_history_file(_HISTORY_FILE)
        except FileNotFoundError:
            pass
        _readline.set_history_length(500)

    def _save_history() -> None:
        try:
            _readline.write_history_file(_HISTORY_FILE)
        except OSError:
            pass

except ImportError:
    def _setup_history() -> None: pass   # noqa: E704
    def _save_history() -> None: pass    # noqa: E704


# ── LLM Engine — language synthesis layer ──────────────────────────────────
# The LLM speaks for Ada. HDC thinks, LLM talks.
# Falls back to raw HDC recall if no LLM is available.

_engine = None
_llm_available: bool | None = None  # None = not yet checked


def _get_engine():
    """Get or create the LLM engine. Returns None if no backend available."""
    global _engine, _llm_available
    if _llm_available is False:
        return None
    if _engine is None:
        try:
            from glyphh.llm import LLMEngine
            _engine = LLMEngine(system_prompt=ADA_SYSTEM_PROMPT)
            # Force backend detection now so we know if it works
            _engine._ensure_loaded()
            _llm_available = True
        except Exception as e:
            logger.info("No LLM backend available: %s", e)
            _llm_available = False
            return None
    return _engine


# ── Ada singleton ─────────────────────────────────────────────────────────

_MEMORY_DIR = os.path.expanduser("~/.glyphh/memory")
_ada: AdaCognitive | None = None


def _get_ada() -> AdaCognitive:
    """Get or create the AdaCognitive instance."""
    global _ada
    if _ada is None:
        _ada = AdaCognitive()
    return _ada


# ── Backward-compat accessors (used by tests/demos) ──────────────────────

def _get_thought_space():
    return _get_ada().thought_space

def _get_cognitive():
    return _get_ada().cognitive

def _recall_with_gate(text):
    return _get_ada().recall(text)

def _build_llm_prompt(text, state, gate_state="ASK", facts=None):
    return _get_ada().build_prompt(text, state, gate_state, facts)


# ── LLM streaming output ──────────────────────────────────────────────────

def _stream_response(prompt: str) -> str:
    """Stream Ada's response with word-wrap, hang detection, and loop detection."""
    engine = _get_engine()
    if engine is None:
        return ""

    click.echo("  ", nl=False)
    response_parts = []
    col = 2
    blank_run = 0

    for token in engine.stream(prompt, max_tokens=256, temperature=0.6, raw=True):
        response_parts.append(token)

        # Skip think tag artifacts in display
        if "</think>" in token or "<think>" in token:
            continue

        # Hang detector
        if not token.strip():
            blank_run += 1
            if blank_run > 8:
                break
        else:
            blank_run = 0

        # Repetition detector — catch repeated phrases of any length
        full = "".join(response_parts)
        if len(full) > 80:
            tail = full[-200:]
            caught = False
            for plen in (15, 20, 30, 40):
                if plen > len(tail) // 3:
                    continue
                phrase = tail[-plen:]
                if tail.count(phrase) >= 3:
                    caught = True
                    break
            if caught:
                break

        for char in token:
            if char == "\n":
                click.echo()
                click.echo("  ", nl=False)
                col = 2
            else:
                sys.stdout.write(char)
                col += 1
                if col > 80 and char == " ":
                    click.echo()
                    click.echo("  ", nl=False)
                    col = 2
        sys.stdout.flush()

    click.echo()
    click.echo()
    # Clean up any leaked think tags from raw mode
    result = "".join(response_parts).strip()
    result = re.sub(r"</?think>\s*", "", result).strip()
    return result


# ── Response: LLM synthesis with HDC fallback ────────────────────────────

def _respond_from_recall(text: str, result: CognitiveResult) -> str:
    """Recall with confidence gate -> LLM speaks or says 'I don't know.'"""
    engine = _get_engine()

    if engine is not None:
        click.echo()
        click.secho("  ada", fg=theme.ACCENT, bold=True)
        response = _stream_response(result.prompt)
        if response:
            _get_ada().add_response(text, response)
            return response

    # Fallback: pure HDC (no LLM)
    return _respond_hdc_only(text)


def _respond_hdc_only(text: str) -> str:
    """Pure HDC recall response — no LLM. Fallback mode."""
    ada = _get_ada()
    results = ada.thought_space.recall(text, top_k=3, speaker="incoming")

    click.echo()
    click.secho("  ada", fg=theme.ACCENT, bold=True)

    if not results or results[0].global_similarity < 0.1:
        click.secho("  I don't know.", fg=theme.TEXT)
        click.echo()
        return "I don't know."

    response_parts = []
    for r in results:
        if r.global_similarity < 0.2:
            break
        t = r.thought
        who = "you said" if t.speaker == "incoming" else "I said"
        response_parts.append(f"{who} \"{t.content}\"")

    if response_parts:
        response = "I remember: " + ". ".join(response_parts) + "."
    else:
        response = "I don't know."

    click.secho(f"  {response}", fg=theme.TEXT)
    click.echo()
    return response


# ── Core action — everything is conversation ───────────────────────────────

def _do_talk(text: str) -> str:
    """Route input through AdaCognitive -> display response."""
    ada = _get_ada()
    result = ada.process(text)

    # Show cognitive state indicator
    _show_cognitive_indicator(result.state)

    # Route through LLM with cognitive context (falls back to HDC-only)
    if result.state.action == Action.RECALL:
        response = _respond_from_recall(text, result)
    else:
        response = _respond_with_llm(text, result)

    # Absorb Ada's own response
    ada.absorb(response, speaker="outgoing")

    # Save periodically
    ada.save(_MEMORY_DIR)

    return response


def _show_cognitive_indicator(state: CognitiveState) -> None:
    """Show a subtle cognitive state indicator."""
    icons = {
        "question": "?",
        "statement": ".",
        "emotion": "~",
        "contradict": "!",
        "curiosity": "?",
        "dream": "*",
    }
    icon = icons.get(state.winner, ".")
    amb = " ~" if state.is_ambiguous else ""
    click.secho(
        f"  [{icon} {state.winner} {state.confidence:.2f}{amb}]",
        fg=theme.TEXT_DIM,
    )


def _respond_with_llm(text: str, result: CognitiveResult) -> str:
    """Route non-RECALL actions through LLM."""
    engine = _get_engine()
    if engine is not None:
        click.echo()
        click.secho("  ada", fg=theme.ACCENT, bold=True)
        response = _stream_response(result.prompt)
        if response:
            _get_ada().add_response(text, response)
            return response

    return _respond_hdc_action(text, result.state)


def _respond_hdc_action(text: str, state: CognitiveState) -> str:
    """HDC-only fallback responses per cognitive action. No LLM."""
    ada = _get_ada()
    space = ada.thought_space
    click.echo()
    click.secho("  ada", fg=theme.ACCENT, bold=True)

    if state.action == Action.STORE:
        results = space.recall(text, top_k=1, speaker="incoming")
        if results and results[0].global_similarity > 0.3:
            existing = results[0].thought
            response = f"I'll remember that. It connects to: \"{existing.content}\""
        else:
            response = "I'll remember that."

    elif state.action == Action.CONTRADICT:
        results = space.recall(text, top_k=3, speaker="incoming")
        if results and results[0].global_similarity > 0.2:
            existing = results[0].thought
            response = f"That conflicts with what I know: \"{existing.content}\". Which is correct?"
        else:
            response = "I hear a correction, but I'm not sure what it changes."

    elif state.action == Action.FEEL:
        response = "I understand. I'll remember how you feel."

    elif state.action == Action.WONDER:
        results = space.recall(text, top_k=3, speaker="incoming")
        if results and results[0].global_similarity > 0.2:
            response = f"That's interesting. It reminds me of: \"{results[0].thought.content}\""
        else:
            response = "That's interesting. Tell me more."

    elif state.action == Action.DREAM:
        response = "I'll think about that."

    else:
        response = "I don't know."

    click.secho(f"  {response}", fg=theme.TEXT)
    click.echo()
    return response


# ── Commands (only dream and reset) ────────────────────────────────────────

def _do_reset() -> None:
    """Clear all of Ada's memory — thought glyphs + pathways."""
    import shutil
    global _ada
    if _ada is not None:
        _ada.reset()
        _ada = None
    # Clear disk state (legacy + pathways)
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

    elif cmd == "gaps":
        gaps = dream.find_gaps()
        if not gaps:
            click.secho("  No knowledge gaps found.", fg=theme.TEXT_DIM)
        else:
            click.echo()
            for gap in gaps:
                click.secho(f"  ? {gap.summary}", fg="yellow")

    else:
        click.secho("  dream [start|stop|status|insights|gaps]", fg=theme.TEXT_DIM)


_INSIGHT_ICONS = {
    InsightKind.CONNECTION: ("!", theme.ACCENT),
    InsightKind.CONTRADICTION: ("!", "yellow"),
    InsightKind.CONVERGENCE: ("+", "cyan"),
    InsightKind.QUESTION: ("?", "yellow"),
    InsightKind.CRYSTALLIZATION: ("*", "bright_cyan"),
}


def _show_insights(insights, max_show: int = 5) -> None:
    """Display insights from background reasoning."""
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


def _do_mind() -> None:
    """Show Ada's cognitive state — what each sub-agent is doing."""
    cog = _get_ada().cognitive
    click.echo()

    if cog.last_state is None:
        click.secho("  No cognitive state yet. Talk to Ada first.", fg=theme.TEXT_DIM)
        return

    # Format cognitive state
    click.secho("  cognitive state:", fg=theme.ACCENT, bold=True)
    click.secho(f"  {cog.format_state()}", fg=theme.TEXT)
    click.echo()

    # Agent stats
    click.secho("  sub-agents:", fg=theme.ACCENT, bold=True)
    stats = cog.stats()
    for name, info in stats["agents"].items():
        bar = "#" * int(info["activation"] * 20)
        fired = f"{info['last_fired']:.0f}s ago" if info["last_fired"] != float("inf") else "never"
        click.secho(
            f"    {name:12s} {info['activation']:.3f} {bar}  "
            f"({info['exemplars']} exemplars, fired {fired})",
            fg=theme.TEXT,
        )
    click.echo()


def _do_know() -> None:
    """Show what Ada knows — thought glyphs with layer structure."""
    space = _get_ada().thought_space

    click.echo()

    if space.count == 0:
        click.secho("  Ada knows nothing yet.", fg=theme.TEXT_DIM)
        return

    # Thought glyphs (strongest first)
    click.secho(f"  thought glyphs ({space.count}):", fg=theme.ACCENT, bold=True)
    for t in space.all_thoughts()[:20]:
        # Show content + activated layers
        layers = []
        for layer_name, layer in t.glyph.layers.items():
            active_segs = [s for s in layer.segments.values()
                           if hasattr(s, 'roles') and s.roles]
            if active_segs:
                seg_names = [s.name for s in active_segs]
                layers.append(f"{layer_name}/{','.join(seg_names)}")
        strength = f" x{t.strength:.2f}" if t.strength != 1.0 else ""
        layer_str = " . ".join(layers) if layers else "-"
        click.secho(f"    [{t.speaker[0]}] {t.content}", fg=theme.TEXT)
        click.secho(f"        {layer_str}{strength}", fg=theme.TEXT_DIM)
    if space.count > 20:
        click.secho(f"    ... and {space.count - 20} more", fg=theme.TEXT_DIM)
    click.echo()

    # Primitive stats
    stats = space.primitives.stats()
    click.secho(f"  primitives: {stats['words']} words -> {stats['roles']} roles, {stats['axioms']} axioms",
                fg=theme.TEXT_DIM)


# ── Dispatch — only dream, reset, quit ──────────────────────────────────────

def _dispatch(line: str) -> str | None:
    """Check for the few remaining commands. Returns 'handled', 'quit', or None."""
    parts = line.split(None, 1)
    cmd = parts[0].lower().lstrip("/")
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "dream":
        _do_dream(args)
        click.echo()
        return "handled"

    if cmd == "reset":
        _do_reset()
        click.echo()
        return "handled"

    if cmd in ("know", "memory", "facts"):
        _do_know()
        click.echo()
        return "handled"

    if cmd in ("mind", "cognitive", "cog"):
        _do_mind()
        click.echo()
        return "handled"

    if cmd in ("quit", "exit", "q"):
        return "quit"

    if cmd == "clear":
        return "clear"

    return None


# ── Version ────────────────────────────────────────────────────────────────

ADA_VERSION = "3.0.0"
ADA_TAGLINE = "i don't guess."


# ── Banner ─────────────────────────────────────────────────────────────────

def _print_banner():
    click.echo()
    click.secho("            _", fg=theme.PRIMARY)
    click.secho("   __ _  __| | __ _", fg=theme.PRIMARY)
    click.secho("  / _` |/ _` |/ _` |", fg=theme.ACCENT)
    click.secho(" | (_| | (_| | (_| |", fg="cyan")
    click.secho("  \\__,_|\\__,_|\\__,_|", fg="bright_cyan")
    click.secho(f"              v{ADA_VERSION}", fg=theme.TEXT_DIM)
    click.echo()
    click.secho(f"  {ADA_TAGLINE}", fg="bright_cyan", bold=True)
    click.echo()
    ada = _get_ada()
    space = ada.thought_space
    cog = ada.cognitive
    n_exemplars = sum(len(a._exemplars) for a in cog.agents.values())
    engine = _get_engine()
    llm_str = f" . LLM ({engine.backend_name})" if engine else " . no LLM"
    click.secho(f"  HDC . {space.count} thoughts . {space.primitives.count} primitives . {n_exemplars} cog{llm_str}", fg=theme.TEXT_DIM)
    click.echo()


# ── REPL ────────────────────────────────────────────────────────────────────

def _run_repl():
    click.secho("  just talk -- know . mind . dream . reset . quit", fg=theme.TEXT_DIM)
    click.echo()

    _setup_history()

    ada = _get_ada()

    # Start background reasoning
    if ada.thought_space.count > 0:
        ada.start_dreaming()

    while True:
        # Show any insights Ada discovered
        insights = ada.drain_insights()
        if insights:
            _show_insights(insights)

        prompt_str = (
            click.style("  ", fg=theme.TEXT_DIM)
            + click.style("you", fg=theme.PRIMARY, bold=True)
            + click.style(" > ", fg=theme.TEXT_DIM)
        )

        try:
            line = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            click.echo()
            ada.stop_dreaming()
            _save_history()
            ada.save(_MEMORY_DIR)
            break

        if not line:
            continue

        # Check for commands
        result = _dispatch(line)
        if result == "handled":
            continue
        elif result == "quit":
            ada.stop_dreaming()
            _save_history()
            ada.save(_MEMORY_DIR)
            break
        elif result == "clear":
            click.clear()
            _print_banner()
            continue

        # Everything else is conversation — pure HDC
        _do_talk(line)


# ── CLI command ─────────────────────────────────────────────────────────────

@click.command("ada")
@click.option("--version", is_flag=True, help="Show Ada version.")
@click.argument("action", required=False)
@click.argument("text", required=False, nargs=-1)
def ada_command(version, action, text):
    """Ada — i don't guess.

    \b
    Examples:
      glyphh ada                           Interactive REPL
      glyphh ada "my name is chris"        Talk to Ada
      glyphh ada dream status              Background reasoning
      glyphh ada reset                     Clear all memory
    """
    if version:
        click.echo(f"Ada v{ADA_VERSION}")
        return

    # Direct commands
    if action and action.lower() in ("reset", "dream"):
        args = " ".join(text) if text else ""
        if action.lower() == "reset":
            _do_reset()
        else:
            _do_dream(args)
        return

    if action:
        # Single message — pure HDC
        query_text = action + (" " + " ".join(text) if text else "")
        _do_talk(query_text)
    else:
        _print_banner()
        _run_repl()


# ── Handler for interactive shell ───────────────────────────────────────────

def handle_ada(func: str | None, args: str = ""):
    """Route ada subcommands from the interactive shell."""
    full = " ".join(p for p in [func, args] if p).strip()

    # Direct commands
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

    if full:
        _do_talk(full)
    else:
        _print_banner()
        _run_repl()
