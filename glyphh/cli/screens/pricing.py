"""
Pricing screen - plan comparison and signup CTA.
"""

import click
from .. import theme
from ..streaming import stream_text, stream_echo
from .helpers import print_box


def show_pricing():
    """Display pricing information."""

    print_box(
        "PRICING & PLANS",
        ["Free to build. Pay when you ship."],
    )

    # Plans
    stream_echo(("  DEVELOPMENT", theme.TEXT_HIGHLIGHT, False), (" — ", theme.MUTED, False), ("FREE forever", theme.SUCCESS, False))
    click.echo()
    stream_text("  * Full SDK access", fg=theme.MUTED)
    stream_text("  * Local runtime (Docker)", fg=theme.MUTED)
    stream_text("  * 1 model, 1K glyphs", fg=theme.MUTED)
    stream_text("  * Community support", fg=theme.MUTED)
    click.echo()

    stream_echo(("  PRODUCTION", theme.TEXT_HIGHLIGHT, False), (" — ", theme.MUTED, False), ("$35/runtime/month", theme.TEXT, False))
    click.echo()
    stream_text("  * Everything in Development", fg=theme.MUTED)
    stream_text("  * Unlimited models & glyphs", fg=theme.MUTED)
    stream_text("  * Production license key", fg=theme.MUTED)
    stream_text("  * Priority support", fg=theme.MUTED)
    click.echo()

    stream_text("  HOW IT WORKS", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_echo(("  + ", theme.SUCCESS, False), ("Dev is free forever", theme.TEXT, False))
    stream_echo(("  + ", theme.SUCCESS, False), ("Pay only in production", theme.TEXT, False))
    stream_echo(("  + ", theme.SUCCESS, False), ("Self-hosted or cloud", theme.TEXT, False))
    click.echo()

    stream_text("  ENTERPRISE", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_text("  Custom SLAs, on-premise,", fg=theme.MUTED)
    stream_text("  volume pricing:", fg=theme.MUTED)
    stream_text("  sales@glyphh.ai", fg="cyan")
    click.echo()

    stream_text("  NEXT STEPS", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_echo(("  -> ", theme.PRIMARY, False), ("auth signup", theme.TEXT_HIGHLIGHT, False), (" free account", theme.MUTED, False))
    stream_echo(("  -> ", theme.PRIMARY, False), ("2", theme.TEXT_HIGHLIGHT, False), (" build a model", theme.MUTED, False))
    stream_echo(("  -> ", theme.PRIMARY, False), ("docs runtime", theme.TEXT_HIGHLIGHT, False), (" deploy options", theme.MUTED, False))
    click.echo()
