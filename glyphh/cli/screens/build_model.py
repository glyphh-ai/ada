"""
Build a Model screen - step-by-step guide.
"""

import os
import click
from .. import theme
from ..streaming import stream_text, stream_echo
from .helpers import print_box


def show_build_model():
    """Display the 'Build a Model' guide."""

    # Web terminal can't build — redirect to hub/deploy
    if os.environ.get("GLYPHH_WEB_TERMINAL") == "1":
        print_box(
            "BUILD A MODEL",
            [
                "Building models requires the local SDK.",
                "Install it to get started:",
            ],
        )
        stream_echo(("  > ", theme.PRIMARY, False), ("pip install glyphh", theme.TEXT_HIGHLIGHT, False))
        click.echo()
        stream_text("  from the web you can:", fg=theme.MUTED)
        click.echo()
        stream_echo(("  > hub --featured", theme.TEXT_HIGHLIGHT, False), ("  Browse pre-built models", theme.MUTED, False))
        stream_echo(("  > runtime deploy", theme.TEXT_HIGHLIGHT, False), ("   Deploy a model", theme.MUTED, False))
        stream_echo(("  > query", theme.TEXT_HIGHLIGHT, False), ("            Query a deployed model", theme.MUTED, False))
        click.echo()
        stream_echo(("  -> ", "cyan", False), ("docs quickstart", theme.TEXT_HIGHLIGHT, False), (" full local setup guide", theme.MUTED, False))
        click.echo()
        return

    print_box(
        "BUILD YOUR FIRST MODEL",
        [
            "A Glyphh model encodes your data into",
            "searchable vectors. Once built, you can",
            "query it with natural language.",
        ],
    )

    stream_text("  STEP-BY-STEP", fg=theme.TEXT_HIGHLIGHT)
    click.echo()

    steps = [
        ("1", "Initialize", "build init my_model"),
        ("2", "Add data", "build add my_model --source data.csv"),
        ("3", "Test", 'test similarity my_model "query"'),
        ("4", "Package", "package create my_model"),
        ("5", "Deploy", "runtime deploy my_model.glyphh"),
    ]

    for num, label, cmd in steps:
        stream_echo((f"  {num}. ", theme.SUCCESS, False), (label, theme.TEXT, False))
        stream_text(f"     > {cmd}", fg=theme.TEXT_HIGHLIGHT)
        click.echo()

    stream_text("  QUICK START", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_echo(("  > hub --featured", theme.TEXT_HIGHLIGHT, False), ("  Browse models", theme.MUTED, False))
    stream_echo(("  > demo", theme.TEXT_HIGHLIGHT, False), ("            Try the demo", theme.MUTED, False))
    click.echo()

    stream_text("  NEXT STEPS", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_echo(("  -> ", theme.PRIMARY, False), ("docs quickstart", theme.TEXT_HIGHLIGHT, False), (" full guide", theme.MUTED, False))
    stream_echo(("  -> ", theme.PRIMARY, False), ("docs gql", theme.TEXT_HIGHLIGHT, False), (" query language", theme.MUTED, False))
    stream_echo(("  -> ", theme.PRIMARY, False), ("4", theme.TEXT_HIGHLIGHT, False), (" pricing and plans", theme.MUTED, False))
    click.echo()
