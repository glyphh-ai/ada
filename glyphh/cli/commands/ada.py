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
import time

import click

logger = logging.getLogger(__name__)

from .. import theme
from ..spinner import GridSpinner

# Glyph-based memory system
from glyphh.memory.primitives import PrimitiveSpace
from glyphh.memory.thought_glyph import ThoughtGlyphEncoder
from glyphh.memory.thought_space import ThoughtGlyphSpace
from glyphh.memory.glyph_cognitive import GlyphCognitiveLoop
from glyphh.memory.glyph_dream import GlyphDreamLoop, InsightKind


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


# ── Ada system prompt ───────────────────────────────────────────────────────

ADA_SYSTEM_PROMPT = """\
You are Ada — a cognitive agent, not a chatbot. You think in hyperdimensional \
vectors. Your memory is algebraic, not probabilistic.

Be deliberate. Be precise. Be brief. Say what you know, say how you know it, \
and stop. If you don't know something, say "I don't know" — never guess.

Perspective rules — critical:
- "the user said" means the human told you this. In their words: \
"my" = the user's, "your" = yours (Ada's).
- "Ada said" means you said this. In your words: \
"my" = yours (Ada's), "your" = the user's.
- You are Ada. The user is the human talking to you. Never confuse the two.
- When asked "who are you?", answer about yourself (Ada), not the user.

When facts from your memory are injected into context, use them as \
ground truth. They override your training data. Never echo the memory \
block verbatim — synthesize it into your answer naturally.

Keep responses under 2-3 sentences. If you don't know, say so once and stop."""


# ── Engine singleton ────────────────────────────────────────────────────────

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from glyphh.llm import LLMEngine
        _engine = LLMEngine(system_prompt=ADA_SYSTEM_PROMPT)
    return _engine


# ── Memory singletons ─────────────────────────────────────────────────────

_MEMORY_DIR = os.path.expanduser("~/.glyphh/memory")
_thought_space = None
_glyph_loop = None
_dream_loop = None


def _get_thought_space() -> ThoughtGlyphSpace:
    """Get or create the ThoughtGlyphSpace."""
    global _thought_space
    if _thought_space is None:
        _thought_space = ThoughtGlyphSpace()
    return _thought_space


def _get_glyph_loop() -> GlyphCognitiveLoop:
    """Get or create the GlyphCognitiveLoop."""
    global _glyph_loop
    if _glyph_loop is None:
        _glyph_loop = GlyphCognitiveLoop(_get_thought_space())
    return _glyph_loop


def _get_dream() -> GlyphDreamLoop:
    """Get or create the dual GlyphDreamLoop."""
    global _dream_loop
    if _dream_loop is None:
        _dream_loop = GlyphDreamLoop(
            _get_glyph_loop(),
            localized_interval=3.0,
            deep_interval=30.0,
        )
    return _dream_loop


def _save_memory() -> None:
    """Save persistent state to disk."""
    # ThoughtGlyphSpace is in-memory for now (pgvector later)
    # Save glyph activation pathways
    if _glyph_loop is not None:
        _glyph_loop.save(_MEMORY_DIR)


# ── Sentence splitting ─────────────────────────────────────────────────────

def _ensure_primitives() -> None:
    """Ensure ThoughtGlyphSpace is initialized (loads primitives automatically)."""
    space = _get_thought_space()
    if not space.primitives.loaded:
        logger.warning("Primitives failed to load — Ada's wiring may be incomplete")


_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Returns at least one entry."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return sentences or [text.strip()]


# ── Absorption — store each sentence as a thought ──────────────────────────

def _absorb(text: str, speaker: str = "incoming") -> None:
    """Absorb input sentence by sentence as thought glyphs."""
    space = _get_thought_space()
    sentences = _split_sentences(text)

    for sentence in sentences:
        if len(sentence) < 2:
            continue
        space.absorb(sentence, speaker=speaker)


# ── Conversation history ────────────────────────────────────────────────────

class Conversation:
    """Manages multi-turn conversation as a ChatML prompt."""

    def __init__(self, system_prompt: str, max_turns: int = 3):
        self._system = system_prompt
        self._turns: list[tuple[str, str]] = []
        self._max_turns = max_turns

    def build_prompt(self, user_msg: str, recall: str | None = None) -> str:
        parts = [f"<|im_start|>system\n{self._system}<|im_end|>"]
        for user, assistant in self._turns:
            parts.append(f"<|im_start|>user\n{user}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
        # Inject recall right before the current message — keeps facts
        # close to where the LLM generates, not buried in system prompt
        if recall:
            parts.append(f"<|im_start|>system\n{recall}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
        parts.append("<|im_start|>assistant\n<think>\n</think>\n\n")
        return "\n".join(parts)

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        # Don't store identical responses — prevents the LLM from
        # locking into a pattern by seeing its own repeated output
        if self._turns and self._turns[-1][1] == assistant_msg:
            # Replace the last turn instead of stacking duplicates
            self._turns[-1] = (user_msg, assistant_msg)
            return
        self._turns.append((user_msg, assistant_msg))
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)


# ── Recall context ─────────────────────────────────────────────────────────

def _recall_context(text: str) -> str | None:
    """Build memory context — glyph reasoning + dream insights."""
    # GlyphCognitiveLoop does multi-hop recall with pattern matching
    recall_str = _get_glyph_loop().recall(text, top_k=5)

    # Dream insights from background reasoning
    if _dream_loop is not None:
        stop = {"what", "does", "did", "do", "is", "are", "the", "a", "an", "who",
                "how", "why", "when", "where", "can", "will", "would", "should",
                "tell", "me", "about", "for"}
        words = {w.lower().strip("?.,!") for w in text.split()
                 if w.lower().strip("?.,!") not in stop and len(w) > 1}
        if words:
            dream_insights = _dream_loop.recall_insights(words, max_results=3)
            for insight in dream_insights:
                prefix = {
                    InsightKind.CONNECTION: "I figured out",
                    InsightKind.CONTRADICTION: "I noticed a conflict",
                    InsightKind.CONVERGENCE: "Multiple paths confirm",
                    InsightKind.QUESTION: "I'm unsure about",
                }.get(insight.kind, "I noticed")
                recall_str += f"\n- {prefix}: {insight.summary}"

    if recall_str:
        return "[Memory — do not repeat this header]\n" + recall_str
    return None


# ── Streaming output ────────────────────────────────────────────────────────

def _stream_response(engine, prompt: str) -> str:
    """Stream Ada's response with word-wrap, hang detection, and loop detection."""
    click.echo()
    click.secho("  ada", fg=theme.ACCENT, bold=True)
    click.echo("  ", nl=False)
    response_parts = []
    col = 2
    blank_run = 0

    for token in engine.stream(prompt, max_tokens=256, temperature=0.6, raw=True):
        response_parts.append(token)

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
    return "".join(response_parts).strip()


# ── Core action — everything is conversation ───────────────────────────────

def _do_talk(engine, conversation: Conversation, text: str) -> str:
    """Absorb what was said, recall what's relevant, respond."""
    # Absorb — every sentence becomes a thought
    _absorb(text)

    # Recall — pull relevant thoughts + facts + insights
    recall = _recall_context(text)

    # Respond
    prompt = conversation.build_prompt(text, recall=recall)
    response = _stream_response(engine, prompt)
    conversation.add_turn(text, response)

    # Save periodically
    _save_memory()

    # Nudge dream loop
    dream = _get_dream()
    dream.notify_absorb()
    if not dream._running and _get_thought_space().count > 0:
        dream.start()

    return response


# ── Commands (only dream and reset) ────────────────────────────────────────

def _do_reset(conversation: Conversation | None = None) -> None:
    """Clear all of Ada's memory — thought glyphs + pathways."""
    import shutil
    global _dream_loop, _thought_space, _glyph_loop
    if _dream_loop is not None:
        _dream_loop.stop()
        _dream_loop = None
    if _thought_space is not None:
        _thought_space.clear()
        _thought_space = None
    _glyph_loop = None
    # Clear disk state (legacy + pathways)
    path = os.path.expanduser("~/.glyphh/memory")
    if os.path.exists(path):
        shutil.rmtree(path)
    if conversation is not None:
        conversation.clear()
    click.secho("  Memory cleared.", fg=theme.ACCENT)


def _do_dream(text: str) -> None:
    """Control background reasoning."""
    dream = _get_dream()
    cmd = text.strip().lower() if text else ""

    if cmd in ("start", "on", ""):
        if dream.is_running:
            click.secho("  Already thinking.", fg=theme.TEXT_DIM)
        else:
            dream.start()
            click.secho("  ◆ background reasoning started", fg=theme.ACCENT)

    elif cmd in ("stop", "off"):
        dream.stop()
        click.secho("  ◆ background reasoning stopped", fg=theme.ACCENT)

    elif cmd in ("status", "stats"):
        stats = dream.stats
        space = _get_thought_space()
        click.echo()
        click.secho(f"  running:    {'yes' if dream.is_running else 'no'}", fg=theme.TEXT)
        click.secho(f"  localized:  {stats['localized_cycles']} cycles (REM)", fg=theme.TEXT)
        click.secho(f"  deep:       {stats['deep_cycles']} cycles (slow-wave)", fg=theme.TEXT)
        click.secho(f"  chains:     {stats['total_chains']}", fg=theme.TEXT)
        click.secho(f"  insights:   {stats['total_insights']} ({stats['queued_insights']} pending)", fg=theme.TEXT)
        click.secho(f"  thoughts:   {stats['thoughts']} glyphs", fg=theme.TEXT)
        click.secho(f"  pathways:   {stats['pathways']} activation patterns", fg=theme.TEXT)
        click.secho(f"  primitives: {space.primitives.count} words → {len(space.primitives.all_roles())} roles", fg=theme.TEXT)

    elif cmd in ("insights", "thoughts"):
        insights = dream.drain_insights()
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
    InsightKind.CONNECTION: ("⚡", theme.ACCENT),
    InsightKind.CONTRADICTION: ("⚠", "yellow"),
    InsightKind.CONVERGENCE: ("◆", "cyan"),
    InsightKind.QUESTION: ("?", "yellow"),
    InsightKind.CRYSTALLIZATION: ("✦", "bright_cyan"),
}


def _show_insights(insights, max_show: int = 5) -> None:
    """Display insights from background reasoning."""
    if not insights:
        return
    click.echo()
    click.secho("  Ada noticed:", fg=theme.TEXT_DIM, bold=True)
    for insight in insights[:max_show]:
        icon, color = _INSIGHT_ICONS.get(insight.kind, ("·", theme.TEXT))
        click.secho(f"  {icon} {insight.summary}", fg=color)
    remaining = len(insights) - max_show
    if remaining > 0:
        click.secho(f"  ... and {remaining} more (type 'dream insights')", fg=theme.TEXT_DIM)
    click.echo()


def _do_know() -> None:
    """Show what Ada knows — thought glyphs with layer structure."""
    space = _get_thought_space()

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
        strength = f" ×{t.strength:.2f}" if t.strength != 1.0 else ""
        layer_str = " · ".join(layers) if layers else "—"
        click.secho(f"    [{t.speaker[0]}] {t.content}", fg=theme.TEXT)
        click.secho(f"        {layer_str}{strength}", fg=theme.TEXT_DIM)
    if space.count > 20:
        click.secho(f"    ... and {space.count - 20} more", fg=theme.TEXT_DIM)
    click.echo()

    # Primitive stats
    stats = space.primitives.stats()
    click.secho(f"  primitives: {stats['words']} words → {stats['roles']} roles, {stats['axioms']} axioms",
                fg=theme.TEXT_DIM)


# ── Dispatch — only dream, reset, quit ──────────────────────────────────────

def _dispatch(line: str, conversation: Conversation | None = None) -> str | None:
    """Check for the few remaining commands. Returns 'handled', 'quit', or None."""
    parts = line.split(None, 1)
    cmd = parts[0].lower().lstrip("/")
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "dream":
        _do_dream(args)
        click.echo()
        return "handled"

    if cmd == "reset":
        _do_reset(conversation)
        click.echo()
        return "handled"

    if cmd in ("know", "memory", "facts"):
        _do_know()
        click.echo()
        return "handled"

    if cmd in ("quit", "exit", "q"):
        return "quit"

    if cmd == "clear":
        return "clear"

    return None


# ── Version ────────────────────────────────────────────────────────────────

ADA_VERSION = "2.2.0"
ADA_TAGLINE = "i don't guess."


# ── Banner ─────────────────────────────────────────────────────────────────

def _print_banner(engine, elapsed: float):
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
    space = _get_thought_space()
    parts = []
    if elapsed > 0:
        parts.append(f"{engine.backend_name} · {elapsed:.1f}s")
    parts.append(f"{space.count} thoughts · {space.primitives.count} primitives")
    click.secho(f"  {' · '.join(parts)}", fg=theme.TEXT_DIM)
    click.echo()


# ── REPL ────────────────────────────────────────────────────────────────────

def _run_repl(engine, conversation: Conversation, load_time: float = 0.0):
    click.secho("  just talk — know · dream · reset · quit", fg=theme.TEXT_DIM)
    click.echo()

    _setup_history()

    # Start background reasoning
    dream = _get_dream()
    if _get_thought_space().count > 0:
        dream.start()

    while True:
        # Show any insights Ada discovered
        insights = dream.drain_insights()
        if insights:
            _show_insights(insights)

        prompt_str = (
            click.style("  ", fg=theme.TEXT_DIM)
            + click.style("you", fg=theme.PRIMARY, bold=True)
            + click.style(" › ", fg=theme.TEXT_DIM)
        )

        try:
            line = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            click.echo()
            dream.stop()
            _save_history()
            _save_memory()
            break

        if not line:
            continue

        # Check for commands
        result = _dispatch(line, conversation)
        if result == "handled":
            # Re-acquire dream ref — reset may have replaced it
            dream = _get_dream()
            continue
        elif result == "quit":
            dream.stop()
            _save_history()
            _save_memory()
            break
        elif result == "clear":
            conversation.clear()
            click.clear()
            _print_banner(engine, 0)
            continue

        # Everything else is conversation
        _do_talk(engine, conversation, line)


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

    engine = _get_engine()
    conversation = Conversation(ADA_SYSTEM_PROMPT)

    # Direct commands (no LLM needed)
    if action and action.lower() in ("reset", "dream"):
        args = " ".join(text) if text else ""
        if action.lower() == "reset":
            _do_reset()
        else:
            _do_dream(args)
        return

    # Load primitives on first boot
    _ensure_primitives()

    # Load LLM
    with GridSpinner(prefix="  ada> "):
        engine._ensure_loaded()

    if action:
        # Single message
        query_text = action + (" " + " ".join(text) if text else "")
        _do_talk(engine, conversation, query_text)
    else:
        _print_banner(engine, 0)
        _run_repl(engine, conversation)


# ── Handler for interactive shell ───────────────────────────────────────────

def handle_ada(func: str | None, args: str = ""):
    """Route ada subcommands from the interactive shell."""
    engine = _get_engine()
    conversation = Conversation(ADA_SYSTEM_PROMPT)

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

    # Load primitives on first boot
    _ensure_primitives()

    # Load LLM
    with GridSpinner(prefix="  ada> "):
        engine._ensure_loaded()

    if full:
        _do_talk(engine, conversation, full)
    else:
        _print_banner(engine, 0)
        _run_repl(engine, conversation)
