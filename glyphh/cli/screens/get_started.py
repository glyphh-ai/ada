"""
Get Started screen for web terminal visitors.

Walks users through downloading and installing the Glyphh SDK
locally so they can build, test, package, deploy, and license --
all from the CLI. The web terminal is for discovery; real work
happens locally.
"""

import click
from .. import theme
from ..streaming import stream_text, stream_echo
from .helpers import print_box


def show_get_started():
    """Display the 'Get Started' guide -- download, install, work locally."""

    print_box(
        "GET STARTED",
        [
            "Everything you need lives in the CLI.",
            "Download it, install it, and you're ready.",
        ],
    )

    # -- download + install --
    stream_text("  DOWNLOAD & INSTALL", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_echo(
        ("  1. ", theme.SUCCESS, False),
        ("download the SDK", theme.TEXT, False),
        cps=600,
    )
    stream_echo(
        ("     ", theme.MUTED, False),
        ("github.com/glyphh-ai/public/releases", theme.TEXT_HIGHLIGHT, False),
        cps=500,
    )
    stream_text("     grab the latest glyphh-sdk .whl for your platform", fg=theme.MUTED, cps=600)
    click.echo()
    stream_echo(
        ("  2. ", theme.SUCCESS, False),
        ("install", theme.TEXT, False),
        cps=600,
    )
    stream_echo(
        ("     $ ", theme.MUTED, False),
        ("pip install glyphh-<version>.whl", theme.TEXT_HIGHLIGHT, False),
        cps=500,
    )
    click.echo()
    stream_echo(
        ("  3. ", theme.SUCCESS, False),
        ("verify", theme.TEXT, False),
        cps=600,
    )
    stream_echo(
        ("     $ ", theme.MUTED, False),
        ("glyphh --version", theme.TEXT_HIGHLIGHT, False),
        cps=500,
    )
    click.echo()
    stream_text("  no web dashboard needed. everything runs from your terminal.", fg=theme.MUTED, cps=600)
    click.echo()

    # -- what you can do --
    stream_text("  WHAT YOU CAN DO", fg=theme.TEXT_HIGHLIGHT)
    click.echo()

    actions = [
        ("build",    "create and train models from your data"),
        ("test",     "run similarity tests, validate encodings"),
        ("package",  "bundle into a .glyphh file"),
        ("deploy",   "push to a runtime (local, cloud, or self-hosted)"),
        ("query",    "run GQL or natural language queries"),
        ("license",  "activate and manage your runtime license"),
    ]

    for cmd, desc in actions:
        stream_echo(
            ("  > ", theme.PRIMARY, False),
            (f"{cmd:<10}", theme.TEXT_HIGHLIGHT, False),
            (desc, theme.MUTED, False),
            cps=800,
        )
    click.echo()

    # -- quick start flow --
    stream_text("  QUICK START", fg=theme.TEXT_HIGHLIGHT)
    click.echo()

    steps = [
        ("1.", "glyphh auth signup",              "create your account"),
        ("2.", "glyphh auth login",               "authenticate"),
        ("3.", "glyphh build init my_model",      "create a model"),
        ("4.", "glyphh build add my_model -s data.csv", "add your data"),
        ("5.", "glyphh package create my_model",  "package it"),
        ("6.", "glyphh runtime deploy my_model.glyphh", "deploy"),
    ]

    for num, cmd, desc in steps:
        stream_echo(
            (f"  {num} ", theme.SUCCESS, False),
            (cmd, theme.TEXT_HIGHLIGHT, False),
            cps=700,
        )
        stream_text(f"     {desc}", fg=theme.MUTED, cps=700)

    click.echo()

    # -- runtime options --
    stream_text("  RUNTIME OPTIONS", fg=theme.TEXT_HIGHLIGHT)
    click.echo()
    stream_text("  local      run on your machine for dev/testing", fg=theme.MUTED, cps=600)
    stream_text("  cloud      deploy to glyphh cloud (managed)", fg=theme.MUTED, cps=600)
    stream_text("  self-host  run on your own infra (heroku, aws, etc.)", fg=theme.MUTED, cps=600)
    click.echo()

    # -- CTA --
    stream_echo(
        ("  ready? ", theme.MUTED, False),
        ("auth signup", theme.SUCCESS, True),
        (" to create your account.", theme.MUTED, False),
        cps=500,
    )
    click.echo()
    stream_echo(
        ("  then grab the SDK: ", theme.MUTED, False),
        ("github.com/glyphh-ai/public/releases", theme.TEXT_HIGHLIGHT, False),
        cps=500,
    )
    click.echo()
