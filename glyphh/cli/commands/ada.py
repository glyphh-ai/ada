"""
CLI ada command — Ada cognitive agent with HDC memory + local LLM.

glyphh ada                          Interactive REPL
glyphh ada "what does chris do?"    Single query
glyphh ada teach "chris builds glyphh"
glyphh ada decompose "Alice manages payments. Bob manages infra."
glyphh ada query "what does chris build?"
glyphh ada infer chris uses
glyphh ada facts
glyphh ada atoms
glyphh ada learn glyphh
glyphh ada reset
"""

import os
import re
import sys
import time

import click

from .. import theme
from ..spinner import GridSpinner
from glyphh.memory import ThoughtEncoder, ThoughtStore, AtomForge, FactStore, Teacher, CognitiveLoop
from glyphh.memory.dream import DreamLoop, InsightKind

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

When you have facts from your memory (shown as "You know:"), use them as \
ground truth. They override your training data. If your memory says X and \
your training says Y, trust your memory — it was taught by a human.

When answering, show your reasoning: which facts you used and how they connect. \
Not verbose — just the chain. "I know A because B → C → A."

Keep responses under 3-4 sentences unless the question demands more."""


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
_thought_store = None
_forge = None
_fact_store = None
_teacher = None
_cognitive_loop = None
_dream_loop = None


def _get_memory() -> ThoughtStore:
    global _thought_store
    if _thought_store is None:
        _thought_store = ThoughtStore(ThoughtEncoder())
        _thought_store.load()
    return _thought_store


def _get_forge() -> AtomForge:
    global _forge
    if _forge is None:
        _forge = AtomForge(dimension=2048)
        _forge.load(_MEMORY_DIR)
    return _forge


def _get_facts() -> FactStore:
    global _fact_store
    if _fact_store is None:
        _fact_store = FactStore(_get_forge())
        _fact_store.load(_MEMORY_DIR)
    return _fact_store


def _get_teacher() -> Teacher:
    global _teacher
    if _teacher is None:
        _teacher = Teacher(_get_forge(), _get_facts())
    return _teacher


def _get_loop() -> CognitiveLoop:
    global _cognitive_loop
    if _cognitive_loop is None:
        _cognitive_loop = CognitiveLoop(_get_forge(), _get_facts())
        _cognitive_loop.load(_MEMORY_DIR)
    return _cognitive_loop


def _get_dream() -> DreamLoop:
    global _dream_loop
    if _dream_loop is None:
        _dream_loop = DreamLoop(
            _get_loop(), _get_forge(), _get_facts(),
            cycle_budget=8,
            cycle_interval=3.0,
        )
    return _dream_loop


def _nudge_dream() -> None:
    """Auto-start background reasoning if facts exist and dream isn't running."""
    if _dream_loop is not None and not _dream_loop.is_running and _get_facts().count > 0:
        _dream_loop.start()


def _save_memory() -> None:
    """Save all persistent state to disk."""
    if _forge is not None:
        _forge.save(_MEMORY_DIR)
    if _fact_store is not None:
        _fact_store.save(_MEMORY_DIR)
    if _cognitive_loop is not None:
        _cognitive_loop.save(_MEMORY_DIR)


# ── Conversation history ────────────────────────────────────────────────────

class Conversation:
    """Manages multi-turn conversation as a ChatML prompt."""

    def __init__(self, system_prompt: str, max_turns: int = 20):
        self._system = system_prompt
        self._turns: list[tuple[str, str]] = []
        self._max_turns = max_turns

    def build_prompt(self, user_msg: str, recall: str | None = None) -> str:
        system = self._system
        if recall:
            system = f"{self._system}\n\n{recall}"
        parts = [f"<|im_start|>system\n{system}<|im_end|>"]
        for user, assistant in self._turns:
            parts.append(f"<|im_start|>user\n{user}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
        parts.append("<|im_start|>assistant\n<think>\n</think>\n\n")
        return "\n".join(parts)

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self._turns.append((user_msg, assistant_msg))
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)


# ── Fact recall ─────────────────────────────────────────────────────────────

def _recall_context(text: str) -> str | None:
    """Build a 'You know:' block — driven by CognitiveLoop reasoning + dream insights."""
    recall_lines = []

    # Flat text memories
    memory = _get_memory()
    memories = memory.recall(text, top_k=2, min_score=0.10)
    for t, _score in memories:
        recall_lines.append(f"- {t.content}")

    # Structured reasoning via CognitiveLoop
    if _get_facts().count > 0:
        loop = _get_loop()
        chain_lines = loop.recall(text)
        recall_lines.extend(chain_lines)

    # Dream insights — things Ada figured out while thinking in the background
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
                recall_lines.append(f"- {prefix}: {insight.summary}")

    if recall_lines:
        return "You know:\n" + "\n".join(recall_lines)
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

        # Repetition detector
        if len(response_parts) > 20:
            tail = "".join(response_parts[-20:])
            if len(tail) > 60:
                third = len(tail) // 3
                chunk = tail[:third]
                if chunk in tail[third:2*third] and chunk in tail[2*third:]:
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


# ── Core actions ────────────────────────────────────────────────────────────

def _do_ask(engine, conversation: Conversation, text: str) -> str:
    """Ask Ada a question — recall facts, stream response, auto-learn."""
    recall = _recall_context(text)
    prompt = conversation.build_prompt(text, recall=recall)
    response = _stream_response(engine, prompt)
    conversation.add_turn(text, response)
    return response


def _do_teach(text: str) -> None:
    """Teach Ada a simple fact: subject verb object."""
    if not text:
        click.secho("  Usage: teach <fact>  e.g. teach chris builds glyphh", fg=theme.TEXT_DIM)
        return
    teacher = _get_teacher()
    facts = teacher.parse(text)
    if facts:
        for f in facts:
            click.secho(f"  ◆ {f.subject} → {f.relation} → {f.object}", fg=theme.ACCENT)
        _save_memory()
        _nudge_dream()
    else:
        click.secho("  Couldn't parse. Try: subject verb object", fg=theme.TEXT_DIM)
    click.secho(f"  ({_get_facts().count} facts, {_get_forge().count} atoms)", fg=theme.TEXT_DIM)


def _do_decompose(engine, text: str) -> None:
    """LLM decomposes complex text into atomic facts."""
    if not text:
        click.secho("  Usage: decompose <paragraph>", fg=theme.TEXT_DIM)
        return
    with GridSpinner(prefix="  ada> "):
        teacher = _get_teacher()
        learned = teacher.decompose(text, engine)
    if learned:
        for f in learned:
            click.secho(f"  ◆ {f.subject} → {f.relation} → {f.object}", fg=theme.ACCENT)
        _save_memory()
        _nudge_dream()
        click.secho(f"  ({_get_facts().count} facts, {_get_forge().count} atoms)", fg=theme.TEXT_DIM)
    else:
        click.secho("  Couldn't extract facts from that.", fg=theme.TEXT_DIM)


# ── Query parser ────────────────────────────────────────────────────────────

_Q_WHAT_DOES_VERB = re.compile(
    r"^what\s+(?:does|did|do|will)\s+(\w+)\s+(\w+)\??$", re.IGNORECASE)
_Q_WHAT_IS = re.compile(
    r"^what\s+is\s+(?:a\s+|an\s+|the\s+)?(\w+)\??$", re.IGNORECASE)
_Q_WHAT_DOES = re.compile(
    r"^what\s+(?:does|did|do|will|can)\s+(\w+)\s+do\??$", re.IGNORECASE)
_Q_WHO_VERB = re.compile(
    r"^who\s+(\w+)\s+(\w+)\??$", re.IGNORECASE)


def _parse_query(text: str) -> dict:
    """Parse NL query into subject/relation/object/mode."""
    text = text.strip().rstrip("?").strip()

    m = _Q_WHAT_DOES.match(text + "?")
    if m:
        return {"subject": m.group(1).lower(), "relation": None, "object": None, "mode": "all"}

    m = _Q_WHAT_DOES_VERB.match(text + "?")
    if m:
        return {"subject": m.group(1).lower(), "relation": m.group(2).lower(), "object": None, "mode": "object"}

    m = _Q_WHAT_IS.match(text + "?")
    if m:
        return {"subject": m.group(1).lower(), "relation": "is", "object": None, "mode": "object"}

    m = _Q_WHO_VERB.match(text + "?")
    if m:
        rel = m.group(1).lower()
        if rel.endswith("s") and not rel.endswith("ss"):
            rel = rel[:-1]
        return {"subject": None, "relation": rel, "object": m.group(2).lower(), "mode": "subject"}

    parts = text.split()
    if len(parts) >= 2:
        return {"subject": parts[0].lower(), "relation": parts[1].lower(), "object": None, "mode": "object"}
    elif len(parts) == 1:
        return {"subject": parts[0].lower(), "relation": None, "object": None, "mode": "all"}
    return {"subject": None, "relation": None, "object": None, "mode": "all"}


def _do_query(text: str) -> None:
    """Query Ada's HDC memory."""
    if not text:
        click.secho("  Usage: query chris builds  or  query what does chris build?", fg=theme.TEXT_DIM)
        return

    q = _parse_query(text)
    facts = _get_facts()
    found = False

    if q["mode"] == "object" and q["subject"] and q["relation"]:
        results = facts.query_object(q["subject"], q["relation"])
        if results:
            click.echo()
            for name, score in results:
                click.secho(f"  {q['subject']} {q['relation']} → {name}  [{score:.2f}]", fg=theme.TEXT)
            found = True

    elif q["mode"] == "subject" and q["relation"] and q["object"]:
        results = facts.query(relation=q["relation"], object=q["object"], top_k=5)
        if results:
            click.echo()
            for fact, score in results:
                click.secho(f"  {fact.subject} → {fact.relation} → {fact.object}  [{score:.2f}]", fg=theme.TEXT)
            found = True

    if not found and q["subject"]:
        results = facts.query(subject=q["subject"], top_k=10)
        if results:
            click.echo()
            for fact, score in results:
                click.secho(f"  {fact.subject} → {fact.relation} → {fact.object}  [{score:.2f}]", fg=theme.TEXT)
            found = True

    if not found:
        click.secho("  No results.", fg=theme.TEXT_DIM)


def _do_infer(text: str) -> None:
    """Transitive inference via CognitiveLoop."""
    if not text:
        click.secho("  Usage: infer chris uses", fg=theme.TEXT_DIM)
        return
    parts = text.split()
    if len(parts) < 2:
        click.secho("  Need subject and relation.", fg=theme.TEXT_DIM)
        return

    loop = _get_loop()
    chains = loop.reason(parts[0].lower(), parts[1].lower())
    if chains:
        click.echo()
        for chain in chains:
            flag = " ⚡" if chain.pattern_boost > 0 else ""
            warn = " ⚠ contradicted" if chain.contradicted else ""
            click.secho(
                f"  {parts[0]} {parts[1]} → {chain.answer}  "
                f"[{chain.confidence:.3f}]{flag}{warn}",
                fg=theme.ACCENT,
            )
            for subj, rel, obj, conf in chain.hops:
                click.secho(f"    {subj} → {rel} → {obj}  [{conf:.2f}]", fg=theme.TEXT_DIM)
            if chain.pathway_name:
                click.secho(f"    pattern: {chain.pathway_name}", fg=theme.TEXT_DIM)
    else:
        click.secho("  No inferences found.", fg=theme.TEXT_DIM)


def _do_confirm(text: str) -> None:
    """Confirm the last reasoning chain — Hebbian strengthening."""
    loop = _get_loop()
    loop.confirm()
    _save_memory()
    click.secho("  ◆ pathway strengthened", fg=theme.ACCENT)


def _do_reject(text: str) -> None:
    """Reject the last reasoning chain — weaken pathway."""
    loop = _get_loop()
    loop.reject()
    _save_memory()
    click.secho("  ◆ pathway weakened", fg=theme.TEXT_DIM)


def _do_facts() -> None:
    """Show all stored facts."""
    facts = _get_facts().facts
    if not facts:
        click.secho("  No facts. Use teach or decompose to add some.", fg=theme.TEXT_DIM)
    else:
        click.echo()
        for f in facts:
            click.secho(f"  {f.subject} → {f.relation} → {f.object}  [{f.strength:.2f}]", fg=theme.TEXT)


def _do_atoms() -> None:
    """Show all known atoms."""
    atoms = _get_forge().all_atoms()
    entity_atoms = [a for a in atoms if a.kind != "role"]
    if not entity_atoms:
        click.secho("  No atoms yet.", fg=theme.TEXT_DIM)
    else:
        click.echo()
        for a in entity_atoms[:30]:
            click.secho(f"  {a.name:24s}  {a.kind:10s}  str={a.strength:.2f}", fg=theme.TEXT)
        if len(entity_atoms) > 30:
            click.secho(f"  ... and {len(entity_atoms) - 30} more", fg=theme.TEXT_DIM)


def _do_learn(name: str) -> None:
    """Load a .teach lesson file."""
    if not name:
        click.secho("  Usage: learn <lesson>  e.g. learn glyphh", fg=theme.TEXT_DIM)
        from pathlib import Path
        lessons_dir = Path(__file__).parent.parent.parent / "memory" / "lessons"
        if lessons_dir.exists():
            available = [p.stem for p in lessons_dir.glob("*.teach")]
            if available:
                click.secho(f"  Available: {', '.join(available)}", fg=theme.TEXT_DIM)
        return
    teacher = _get_teacher()
    try:
        if os.path.isfile(name) or os.path.isfile(name + ".teach"):
            path = name if os.path.isfile(name) else name + ".teach"
            result = teacher.learn_file(path)
        else:
            result = teacher.learn_lesson(name)
        parts = []
        if result.atoms_created:
            parts.append(f"{result.atoms_created} sounds")
        if result.pairs_created:
            parts.append(f"{result.pairs_created} pairs")
        if result.compositions_created:
            parts.append(f"{result.compositions_created} words")
        if result.facts_created:
            parts.append(f"{result.facts_created} facts")
        click.secho(f"  ◆ {', '.join(parts) or 'nothing new'}", fg=theme.ACCENT)
        click.secho(f"  ({_get_facts().count} facts, {_get_forge().count} atoms)", fg=theme.TEXT_DIM)
        _save_memory()
    except FileNotFoundError as e:
        click.secho(f"  {e}", fg=theme.TEXT_DIM)


def _do_reset() -> None:
    """Clear all of Ada's memory."""
    import shutil
    # Stop dreaming first
    global _dream_loop
    if _dream_loop is not None:
        _dream_loop.stop()
        _dream_loop = None
    path = os.path.expanduser("~/.glyphh/memory")
    if os.path.exists(path):
        shutil.rmtree(path)
    global _forge, _fact_store, _teacher, _cognitive_loop
    _forge = None
    _fact_store = None
    _teacher = None
    _cognitive_loop = None
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

    elif cmd == "pause":
        dream.pause()
        click.secho("  ◆ paused", fg=theme.TEXT_DIM)

    elif cmd == "resume":
        dream.resume()
        click.secho("  ◆ resumed", fg=theme.ACCENT)

    elif cmd in ("status", "stats"):
        stats = dream.stats
        click.echo()
        click.secho(f"  running:  {'yes' if dream.is_running else 'no'}", fg=theme.TEXT)
        click.secho(f"  cycles:   {stats['cycles']}", fg=theme.TEXT)
        click.secho(f"  chains:   {stats['total_chains']}", fg=theme.TEXT)
        click.secho(f"  insights: {stats['total_insights']} ({stats['queued_insights']} pending)", fg=theme.TEXT)

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
        click.secho("  dream [start|stop|pause|resume|status|insights|gaps]", fg=theme.TEXT_DIM)


_INSIGHT_ICONS = {
    InsightKind.CONNECTION: ("⚡", theme.ACCENT),
    InsightKind.CONTRADICTION: ("⚠", "yellow"),
    InsightKind.CONVERGENCE: ("◆", "cyan"),
    InsightKind.QUESTION: ("?", "yellow"),
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


# ── Command dispatch ────────────────────────────────────────────────────────

COMMANDS = {
    "teach":     lambda engine, args: _do_teach(args),
    "decompose": lambda engine, args: _do_decompose(engine, args),
    "query":     lambda engine, args: _do_query(args),
    "infer":     lambda engine, args: _do_infer(args),
    "yes":       lambda engine, args: _do_confirm(args),
    "no":        lambda engine, args: _do_reject(args),
    "facts":     lambda engine, args: _do_facts(),
    "atoms":     lambda engine, args: _do_atoms(),
    "learn":     lambda engine, args: _do_learn(args),
    "reset":     lambda engine, args: _do_reset(),
    "dream":     lambda engine, args: _do_dream(args),
}


def _dispatch(engine, conversation: Conversation, line: str) -> bool:
    """Dispatch a line to the right handler. Returns True if handled."""
    parts = line.split(None, 1)
    cmd = parts[0].lower().lstrip("/")
    args = parts[1] if len(parts) > 1 else ""

    if cmd in COMMANDS:
        COMMANDS[cmd](engine, args)
        click.echo()
        return True

    if cmd in ("quit", "exit", "q"):
        return False  # signal exit

    if cmd in ("clear", "home"):
        conversation.clear()
        click.clear()
        _print_banner(engine, 0)
        return True

    if cmd == "history":
        if not conversation.turn_count:
            click.secho("  No conversation history.", fg=theme.TEXT_DIM)
        else:
            click.echo()
            for i, (u, a) in enumerate(conversation._turns, 1):
                click.secho(f"  [{i}] You: {u[:60]}{'...' if len(u) > 60 else ''}", fg=theme.TEXT_DIM)
                click.secho(f"      Ada: {a[:60]}{'...' if len(a) > 60 else ''}", fg=theme.TEXT_DIM)
        click.echo()
        return True

    if cmd == "help":
        _print_help()
        return True

    # Not a command — treat as conversation
    return None


# ── Version ────────────────────────────────────────────────────────────────

ADA_VERSION = "2.1.1"
ADA_TAGLINE = "i don't guess."

# Dream spinner — row of grid spinners started at different offsets
_dream_anim_stop = None


# ── Banner & help ───────────────────────────────────────────────────────────

def _print_dream_line() -> None:
    """Print a static spinner line. Returns True if printed."""
    from ..spinner import _FRAMES
    if _dream_loop is None or not _dream_loop._running:
        return
    n = 12
    chars = "".join(_FRAMES[(i * 2) % len(_FRAMES)] for i in range(n))
    click.secho(f"  {chars}", fg="cyan")


def _start_dream_spinner() -> None:
    """Animate the spinner line ABOVE the prompt using ANSI cursor movement.

    Layout is fixed: spinner on line N, prompt/cursor on line N+1.
    Thread saves cursor, moves up 1 line, rewrites spinner, restores cursor.
    """
    import threading as _th
    from ..spinner import _FRAMES, _INTERVAL

    global _dream_anim_stop
    if _dream_loop is None or not _dream_loop._running:
        return
    if _dream_anim_stop is not None:
        return  # already running

    _dream_anim_stop = _th.Event()
    stop = _dream_anim_stop
    n_spinners = 12
    offsets = [i * 2 for i in range(n_spinners)]
    clr = "\033[36m"  # cyan
    rst = "\033[0m"

    def _animate():
        tick = 0
        while not stop.is_set():
            chars = "".join(
                _FRAMES[(tick + off) % len(_FRAMES)] for off in offsets
            )
            # Save cursor → up 1 line → carriage return → write → restore cursor
            sys.stdout.write(f"\033[s\033[1A\r  {clr}{chars}{rst} \033[u")
            sys.stdout.flush()
            tick += 1
            stop.wait(_INTERVAL)

    t = _th.Thread(target=_animate, daemon=True, name="ada-dream-spin")
    t.start()


def _stop_dream_spinner() -> None:
    """Stop the dream spinner animation."""
    global _dream_anim_stop
    if _dream_anim_stop is not None:
        _dream_anim_stop.set()
        time.sleep(0.1)  # let thread finish
        _dream_anim_stop = None


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
    facts_count = _get_facts().count
    atoms_count = _get_forge().count
    parts = []
    if elapsed > 0:
        parts.append(f"{engine.backend_name} · {elapsed:.1f}s")
    parts.append(f"{facts_count} facts · {atoms_count} atoms")
    click.secho(f"  {' · '.join(parts)}", fg=theme.TEXT_DIM)
    click.echo()


def _print_help():
    click.echo()
    click.secho("  Commands:", fg=theme.TEXT_DIM)
    click.secho("    teach <fact>            Teach a simple fact (subject verb object)", fg=theme.TEXT)
    click.secho("    decompose <text>        LLM breaks text into atomic facts", fg=theme.TEXT)
    click.secho("    query <question>        Query HDC memory", fg=theme.TEXT)
    click.secho("    infer <subject> <rel>   Reason via cognitive loop + pathways", fg=theme.TEXT)
    click.secho("    yes                     Confirm last inference (strengthen pathway)", fg=theme.TEXT)
    click.secho("    no                      Reject last inference (weaken pathway)", fg=theme.TEXT)
    click.secho("    dream [start|stop|status|insights|gaps]", fg=theme.TEXT)
    click.secho("                            Background reasoning — Ada thinks on her own", fg=theme.TEXT)
    click.secho("    learn <lesson>          Load a .teach lesson file", fg=theme.TEXT)
    click.secho("    facts                   Show all stored facts", fg=theme.TEXT)
    click.secho("    atoms                   Show all known atoms", fg=theme.TEXT)
    click.secho("    reset                   Clear all memory", fg=theme.TEXT)
    click.secho("    clear                   Clear conversation", fg=theme.TEXT)
    click.secho("    quit                    Exit", fg=theme.TEXT)
    click.echo()
    click.secho("  Or just talk — anything else goes to Ada.", fg=theme.TEXT_DIM)
    click.echo()


# ── REPL ────────────────────────────────────────────────────────────────────

def _run_repl(engine, conversation: Conversation, load_time: float = 0.0):
    click.secho("  teach · decompose · query · infer · dream · yes/no · learn · help · quit", fg=theme.TEXT_DIM)
    click.echo()

    _setup_history()

    # Start background reasoning if there are facts to think about
    dream = _get_dream()
    if _get_facts().count > 0:
        dream.start()
        click.secho("  ◆ background reasoning active", fg=theme.TEXT_DIM)
        click.echo()

    while True:
        # Show any insights Ada discovered while we were busy
        insights = dream.drain_insights()
        if insights:
            _show_insights(insights)

        # Print spinner line, then prompt below — animate spinner above prompt
        _print_dream_line()
        _start_dream_spinner()

        prompt_str = (
            click.style("  ", fg=theme.TEXT_DIM)
            + click.style("you", fg=theme.PRIMARY, bold=True)
            + click.style(" › ", fg=theme.TEXT_DIM)
        )

        try:
            line = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            _stop_dream_spinner()
            click.echo()
            dream.stop()
            _save_history()
            _get_memory().save()
            _save_memory()
            break

        dream.pause()

        if not line:
            dream.resume()
            continue

        # Resume dreaming while we process the command
        dream.resume()

        if line.startswith("!"):
            import subprocess
            subprocess.run(line[1:].strip(), shell=True)
            continue

        result = _dispatch(engine, conversation, line)

        if result is None:
            # Not a command — ask Ada
            _do_ask(engine, conversation, line)
        elif result is False:
            # quit
            dream.stop()
            _save_history()
            _get_memory().save()
            _save_memory()
            break


# ── CLI command ─────────────────────────────────────────────────────────────

@click.command("ada")
@click.option("--version", is_flag=True, help="Show Ada version.")
@click.argument("action", required=False)
@click.argument("text", required=False, nargs=-1)
def ada_command(version, action, text):
    """Ada — when your llm can't afford to be wrong.

    \b
    Examples:
      glyphh ada                                      Interactive REPL
      glyphh ada "what does chris do?"                 Ask a question
      glyphh ada teach "chris builds glyphh"           Teach a fact
      glyphh ada decompose "Alice manages payments."   LLM decomposition
      glyphh ada query "what does chris build?"        Query memory
      glyphh ada infer chris uses                      Transitive inference
      glyphh ada dream status                          Background reasoning
      glyphh ada learn glyphh                          Load a lesson
      glyphh ada facts                                 Show all facts
      glyphh ada reset                                 Clear memory
    """
    if version:
        click.echo(f"Ada v{ADA_VERSION}")
        return
    engine = _get_engine()
    conversation = Conversation(ADA_SYSTEM_PROMPT)

    # If action is a known command, run it directly (no REPL)
    if action and action.lower() in COMMANDS:
        args = " ".join(text) if text else ""
        # Only load LLM for commands that need it
        if action.lower() in ("decompose",):
            with GridSpinner(prefix="  ada> "):
                engine._ensure_loaded()
        COMMANDS[action.lower()](engine, args)
        return

    # Load LLM with spinner
    with GridSpinner(prefix="  ada> "):
        engine._ensure_loaded()
    start = time.monotonic()

    if action:
        # Single query: "glyphh ada 'what does chris do?'"
        query_text = action + (" " + " ".join(text) if text else "")
        _do_ask(engine, conversation, query_text)
    else:
        _print_banner(engine, 0)
        _run_repl(engine, conversation)


# ── Handler for interactive shell ───────────────────────────────────────────

def handle_ada(func: str | None, args: str = ""):
    """Route ada subcommands from the interactive shell."""
    engine = _get_engine()
    conversation = Conversation(ADA_SYSTEM_PROMPT)

    full = " ".join(p for p in [func, args] if p).strip()

    # Check if it's a known command
    if full:
        parts = full.split(None, 1)
        cmd = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""
        if cmd in COMMANDS:
            if cmd in ("decompose",):
                with GridSpinner(prefix="  ada> "):
                    engine._ensure_loaded()
            COMMANDS[cmd](engine, cmd_args)
            return

    # Otherwise load LLM and either query or REPL
    with GridSpinner(prefix="  ada> "):
        engine._ensure_loaded()

    if full:
        _do_ask(engine, conversation, full)
    else:
        _print_banner(engine, 0)
        _run_repl(engine, conversation)
