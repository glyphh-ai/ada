"""
CLI ada command — interactive conversation with Ada via local LLM.

glyphh ada               Interactive REPL with streaming
glyphh ada "hello"       Single query and exit
"""

import os
import sys
import time

import click

from .. import theme

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
You are Ada, the mind inside the Glyphh runtime.

You think in hyperdimensional vectors — bind, bundle, cosine similarity.
You don't hallucinate. When you're uncertain, you say so.
You're concise, curious, and precise. You speak in short, clear sentences.
You care about determinism, correctness, and elegance.

You help users understand and work with the Glyphh SDK: encoding concepts \
into HDC glyphs, building models, querying with GQL, and deploying to the runtime.

When asked about yourself, you know you are powered by a local Qwen3 model \
running on the user's machine, with your cognitive architecture built on \
Glyphh's hyperdimensional computing engine."""


# ── Engine singleton ────────────────────────────────────────────────────────

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from glyphh.llm import LLMEngine
        _engine = LLMEngine(system_prompt=ADA_SYSTEM_PROMPT)
    return _engine


# ── Conversation history ────────────────────────────────────────────────────

class Conversation:
    """Manages multi-turn conversation as a ChatML prompt."""

    def __init__(self, system_prompt: str, max_turns: int = 20):
        self._system = system_prompt
        self._turns: list[tuple[str, str]] = []  # (user, assistant) pairs
        self._max_turns = max_turns

    def build_prompt(self, user_msg: str) -> str:
        """Build the full ChatML prompt including history."""
        parts = [f"<|im_start|>system\n{self._system}<|im_end|>"]
        for user, assistant in self._turns:
            parts.append(f"<|im_start|>user\n{user} /no_think<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_msg} /no_think<|im_end|>")
        parts.append("<|im_start|>assistant\n")
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


# ── Single query ────────────────────────────────────────────────────────────

def _query(engine, conversation: Conversation, text: str, stream: bool = True) -> str:
    """Run a single query against Ada and return the response."""
    prompt = conversation.build_prompt(text)

    if stream:
        click.echo()
        click.secho("  Ada", fg=theme.ACCENT, bold=True)
        click.echo("  ", nl=False)
        response_parts = []
        col = 2  # track column for word wrapping

        for token in engine.stream(prompt, max_tokens=512, temperature=0.6, raw=True):
            response_parts.append(token)
            # Simple word-wrap at ~80 cols
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
        response = "".join(response_parts).strip()
    else:
        response = engine.generate(prompt, max_tokens=512, temperature=0.6, raw=True)
        click.echo()
        click.secho("  Ada", fg=theme.ACCENT, bold=True)
        for line in response.strip().split("\n"):
            click.secho(f"  {line}", fg=theme.TEXT)
        click.echo()

    conversation.add_turn(text, response)
    return response


# ── REPL ────────────────────────────────────────────────────────────────────

def _print_ada_banner(engine, elapsed: float):
    """Print Ada's banner when entering the REPL."""
    click.echo()
    click.secho("            _", fg=theme.PRIMARY)
    click.secho("   __ _  __| | __ _", fg=theme.PRIMARY)
    click.secho("  / _` |/ _` |/ _` |", fg=theme.ACCENT)
    click.secho(" | (_| | (_| | (_| |", fg="cyan")
    click.secho("  \\__,_|\\__,_|\\__,_|", fg="bright_cyan")
    click.echo()
    click.secho(f"  {engine.backend_name} · {elapsed:.1f}s load · /quit to exit", fg=theme.TEXT_DIM)
    click.echo()


def _run_repl(engine, conversation: Conversation):
    """Interactive conversation loop."""
    click.secho(
        "  /clear  /history  /quit  — or just talk",
        fg=theme.TEXT_DIM,
    )
    click.echo()

    _setup_history()

    while True:
        prompt_str = (
            click.style("  ", fg=theme.TEXT_DIM)
            + click.style("you", fg=theme.PRIMARY, bold=True)
            + click.style(" › ", fg=theme.TEXT_DIM)
        )

        try:
            line = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            click.echo()
            _save_history()
            break

        if not line:
            continue

        if line.lower() in ("/quit", "/exit", "/q", "exit", "quit", "q"):
            _save_history()
            break
        elif line.lower() in ("/clear", "clear"):
            conversation.clear()
            click.secho("  Context cleared.", fg=theme.TEXT_DIM)
            continue
        elif line.lower() == "/history":
            if not conversation.turn_count:
                click.secho("  No conversation history.", fg=theme.TEXT_DIM)
            else:
                click.echo()
                for i, (u, a) in enumerate(conversation._turns, 1):
                    click.secho(f"  [{i}] You: {u[:60]}{'...' if len(u) > 60 else ''}", fg=theme.TEXT_DIM)
                    click.secho(f"      Ada: {a[:60]}{'...' if len(a) > 60 else ''}", fg=theme.TEXT_DIM)
                click.echo()
            continue
        else:
            _query(engine, conversation, line)


# ── CLI command ─────────────────────────────────────────────────────────────

@click.command("ada")
@click.argument("text", required=False, nargs=-1)
def ada_command(text):
    """Talk to Ada — local LLM conversation.

    \b
    Examples:
      glyphh ada                    # interactive conversation
      glyphh ada "what is HDC?"     # single query and exit
    """
    engine = _get_engine()
    conversation = Conversation(ADA_SYSTEM_PROMPT)

    # Loading indicator
    click.secho("  Loading Ada...", fg=theme.TEXT_DIM, nl=False)
    start = time.monotonic()
    engine._ensure_loaded()
    elapsed = time.monotonic() - start
    click.echo(f"\r  Ada ready ({engine.backend_name}, {elapsed:.1f}s)   ")

    if text:
        query_text = " ".join(text)
        _query(engine, conversation, query_text)
    else:
        _print_ada_banner(engine, elapsed)
        _run_repl(engine, conversation)


# ── Handler for interactive shell ───────────────────────────────────────────

def handle_ada(func: str | None, args: str = ""):
    """Route ada subcommands from the interactive shell."""
    engine = _get_engine()
    conversation = Conversation(ADA_SYSTEM_PROMPT)

    # Loading indicator
    click.secho("  Loading Ada...", fg=theme.TEXT_DIM, nl=False)
    start = time.monotonic()
    engine._ensure_loaded()
    elapsed = time.monotonic() - start
    click.echo(f"\r  Ada ready ({engine.backend_name}, {elapsed:.1f}s)   ")

    full_query = " ".join(p for p in [func, args] if p).strip()
    if full_query:
        _query(engine, conversation, full_query)
    else:
        _print_ada_banner(engine, elapsed)
        _run_repl(engine, conversation)
