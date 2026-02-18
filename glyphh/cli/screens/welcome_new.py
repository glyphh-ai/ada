from __future__ import annotations

"""
Welcome screen for first-time visitors.

Glyphh introduces itself in first person — conversational,
warm, but with enough substance to explain what makes it different.
"""

import os
import click
from .. import theme
from ..streaming import stream_text, stream_echo


def _is_web() -> bool:
    """Check if running in a web terminal session."""
    return os.environ.get("GLYPHH_WEB_TERMINAL") == "1"


def show_welcome_new(first_name: str | None = None):
    """Show the first-visit welcome where Glyphh introduces itself."""
    click.echo()

    # -- greeting --
    if first_name:
        stream_text(f"  hi {first_name}, welcome.", fg=theme.PRIMARY, bold=True, cps=500)
    else:
        stream_text("  hi there, welcome.", fg=theme.PRIMARY, bold=True, cps=500)
    click.echo()

    stream_echo(
        ("  my name is ", theme.MUTED, False),
        ("glyphh", "cyan", True),
        (" and I am built using glyphh ai vsa.", theme.MUTED, False),
        cps=500,
    )
    click.echo()

    # -- what makes me different --
    stream_text("  i'm not like other AI you've interacted with before.", fg=theme.MUTED, cps=600)
    stream_echo(
        ("  i'm not purely an LLM -- i have a series of small ", theme.MUTED, False),
        ('"brains"', theme.TEXT_HIGHLIGHT, False),
        (",", theme.MUTED, False),
        cps=600,
    )
    stream_echo(
        ("  made up of mathematical structures called ", theme.MUTED, False),
        ('"glyphs"', theme.TEXT_HIGHLIGHT, False),
        (".", theme.MUTED, False),
        cps=600,
    )
    click.echo()

    # -- how i work --
    stream_text("  how i work:", fg=theme.TEXT_HIGHLIGHT, cps=700)
    click.echo()
    stream_text("  i'm a semantic reasoning engine. i encode knowledge into", fg=theme.MUTED, cps=600)
    stream_text("  small, targeted neural models my internal LLM uses as", fg=theme.MUTED, cps=600)
    stream_text("  fact trees. this lets me interact with data deterministically", fg=theme.MUTED, cps=600)
    stream_echo(
        ("  -- no hallucinations, no guessing. just ", theme.MUTED, False),
        ("math", "cyan", True),
        (".", theme.MUTED, False),
        cps=600,
    )
    click.echo()

    # -- how i should be used --
    stream_text("  how glyphh should be used:", fg=theme.TEXT_HIGHLIGHT, cps=700)
    click.echo()
    stream_text("  pair glyphh with your LLM. let it handle the ambiguity --", fg=theme.MUTED, cps=600)
    stream_text("  the open-ended questions, the creative stuff.", fg=theme.MUTED, cps=600)
    stream_echo(
        ("  when your LLM needs to be ", theme.MUTED, False),
        ("deterministic", theme.SUCCESS, True),
        (", route it through", theme.MUTED, False),
        cps=600,
    )
    stream_text("  a model like me.", fg=theme.MUTED, cps=600)
    click.echo()
    stream_text("  we give you a grounded answers with confidence scores,", fg=theme.MUTED, cps=600)
    stream_text("  citation trails, and full audit paths.", fg=theme.MUTED, cps=600)
    stream_text("  if not confident, we ask for clarity instead of guessing.", fg=theme.MUTED, cps=600)
    click.echo()

    # -- explore more --
    stream_text("  explore more:", fg=theme.TEXT_HIGHLIGHT, cps=700)
    click.echo()
    if _is_web():
        click.secho("  ", nl=False)
        click.secho("[4]", fg="cyan", nl=False)
        click.secho(" get started -- install the SDK locally", fg=theme.MUTED)
        click.secho("  ", nl=False)
        click.secho("[5]", fg="cyan", nl=False)
        click.secho(" watch me in action with churn", fg=theme.MUTED)
        click.secho("  ", nl=False)
        click.secho("[6]", fg="cyan", nl=False)
        click.secho(" check out my product vibes", fg=theme.MUTED)
        click.secho("  ", nl=False)
        click.secho("[7]", fg="cyan", nl=False)
        click.secho(" this entire site runs on glyphh VSA. learn more", fg=theme.MUTED)
    else:
        click.secho("  ", nl=False)
        click.secho("[5]", fg="cyan", nl=False)
        click.secho(" watch me in action with churn", fg=theme.MUTED)
        click.secho("  ", nl=False)
        click.secho("[6]", fg="cyan", nl=False)
        click.secho(" check out my product vibes", fg=theme.MUTED)
        click.secho("  ", nl=False)
        click.secho("[7]", fg="cyan", nl=False)
        click.secho(" our stack runs on glyphh vsa. learn more", fg=theme.MUTED)
    click.echo()

    # -- CTA --
    if _is_web():
        stream_echo(
            ("  ready to build? type ", theme.MUTED, False),
            ("4", theme.SUCCESS, True),
            (" to get started with the local SDK.", theme.MUTED, False),
            cps=500,
        )
        click.echo()
        stream_echo(
            ("  or ", theme.MUTED, False),
            ("auth signup", theme.SUCCESS, True),
            (" to create your account.", theme.MUTED, False),
            cps=500,
        )
    else:
        stream_echo(
            ("  ready to try it? ", theme.MUTED, False),
            ("auth signup", theme.SUCCESS, True),
            (" to create your account.", theme.MUTED, False),
            cps=500,
        )
    click.echo()
