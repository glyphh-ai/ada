"""
What is Glyphh AI? screen - explains the product and architecture.
"""

import click
from .. import theme
from ..streaming import stream_text, stream_echo
from .helpers import print_box


def show_what_is_glyphh():
    """Display the 'What is Glyphh AI?' information."""

    print_box(
        "WHAT IS GLYPHH AI?",
        [
            "The deterministic sidecar for your",
            "LLM + RAG stack. When accuracy matters,",
            "get grounded answers with citations",
            "— not guesses.",
        ],
    )

    stream_text("  THE PROBLEM", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_text("  Probabilistic AI is amazing,", fg=theme.MUTED)
    stream_text("  but in many cases you need determistic outpu:", fg=theme.MUTED)
    click.echo()
    stream_text("  LLMs       -> Hallucinate", fg=theme.MUTED)
    stream_text("  RAG        -> Still guesses", fg=theme.MUTED)
    stream_text("  KGs        -> Rigid & brittle", fg=theme.MUTED)
    stream_text("  Vector DBs -> No reasoning", fg=theme.MUTED)
    click.echo()

    stream_echo(("  THE SOLUTION: ", theme.TEXT_HIGHLIGHT, False), ("Vector Symbolic AI", "cyan", False))
    click.echo()
    stream_text("  Meaning encoded as math.", fg=theme.MUTED)
    stream_text("  Not embeddings. Not tokens.", fg=theme.MUTED)
    click.echo()

    # Properties table
    w = 63
    cw = (w - 5) // 4  # 4 columns + borders
    sep = "─" * cw

    click.secho(f"  ┌{sep}┬{sep}┬{sep}┬{sep}┐", fg=theme.ACCENT)
    click.secho("  │", fg=theme.ACCENT, nl=False)
    click.secho(f"{'Deterministic':^{cw}}", fg=theme.TEXT, nl=False)
    click.secho("│", fg=theme.ACCENT, nl=False)
    click.secho(f"{'Explainable':^{cw}}", fg=theme.TEXT, nl=False)
    click.secho("│", fg=theme.ACCENT, nl=False)
    click.secho(f"{'Auditable':^{cw}}", fg=theme.TEXT, nl=False)
    click.secho("│", fg=theme.ACCENT, nl=False)
    click.secho(f"{'Composable':^{cw}}", fg=theme.TEXT, nl=False)
    click.secho("│", fg=theme.ACCENT)
    click.secho(f"  └{sep}┴{sep}┴{sep}┴{sep}┘", fg=theme.ACCENT)
    click.echo()

    # How it works — simplified flow
    stream_text("  HOW IT WORKS", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_text("  LLM -> Glyphh -> NL -> GQL -> Fact Tree", fg=theme.ACCENT)
    click.echo()
    stream_text("  High confidence:", fg=theme.SUCCESS)
    stream_text("    Fact tree + citation + audit", fg=theme.MUTED)
    click.echo()
    stream_text("  Low confidence:", fg=theme.WARNING)
    stream_text("    Clarify / prompt / escalate", fg=theme.MUTED)
    click.echo()

    stream_text("  Your LLM handles ambiguity.", fg=theme.MUTED)
    stream_text("  Glyphh handles facts that", fg="cyan")
    stream_text("  can't be derrived probabilitically.", fg="cyan")
    click.echo()

    stream_text("  NEXT STEPS", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_echo(("  -> ", "cyan", False), ("2", theme.TEXT_HIGHLIGHT, False), (" build your first model", theme.MUTED, False))
    stream_echo(("  -> ", "cyan", False), ("3", theme.TEXT_HIGHLIGHT, False), (" browse pre-built models", theme.MUTED, False))
    stream_echo(("  -> ", "cyan", False), ("docs concepts", theme.TEXT_HIGHLIGHT, False), (" dive deeper", theme.MUTED, False))
    click.echo()
