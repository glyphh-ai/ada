"""
Find a Model screen - guide for browsing the model hub.
"""

import click
from .. import theme
from ..streaming import stream_text, stream_echo
from .helpers import print_box


def show_find_model():
    """Display the 'Find a Model' guide."""

    print_box(
        "FIND A MODEL",
        [
            "Browse pre-built models from the",
            "Glyphh Hub. Use them directly or as",
            "starting points for your own.",
        ],
    )

    stream_text("  BROWSE THE HUB", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_echo(("  > hub", theme.TEXT_HIGHLIGHT, False), ("              All models", theme.MUTED, False))
    stream_echo(("  > hub --featured", theme.TEXT_HIGHLIGHT, False), ("    Popular", theme.MUTED, False))
    stream_echo(('  > hub search "q"', theme.TEXT_HIGHLIGHT, False), ("    Search", theme.MUTED, False))
    click.echo()

    stream_text("  CATEGORIES", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_text("  Documents  PDFs, knowledge bases", fg=theme.MUTED)
    stream_text("  Products   E-commerce, inventory", fg=theme.MUTED)
    stream_text("  Support    FAQ, ticket matching", fg=theme.MUTED)
    stream_text("  Content    Articles, media", fg=theme.MUTED)
    stream_text("  Code       Code search, docs", fg=theme.MUTED)
    click.echo()

    stream_text("  USING A MODEL", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_echo(("  > hub info <name>", theme.TEXT_HIGHLIGHT, False), ("      Details", theme.MUTED, False))
    stream_echo(("  > hub download <name>", theme.TEXT_HIGHLIGHT, False), ("  Local", theme.MUTED, False))
    stream_text("  > runtime deploy <name>", fg=theme.TEXT_HIGHLIGHT)
    stream_echo(("    --from-hub", theme.TEXT_HIGHLIGHT, False), ("          Deploy", theme.MUTED, False))
    click.echo()

    stream_text("  NEXT STEPS", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_echo(("  -> ", "cyan", False), ("hub --featured", theme.TEXT_HIGHLIGHT, False), (" popular models", theme.MUTED, False))
    stream_echo(("  -> ", "cyan", False), ("2", theme.TEXT_HIGHLIGHT, False), (" build your own model", theme.MUTED, False))
    stream_echo(("  -> ", "cyan", False), ("demo", theme.TEXT_HIGHLIGHT, False), (" interactive demo", theme.MUTED, False))
    click.echo()
