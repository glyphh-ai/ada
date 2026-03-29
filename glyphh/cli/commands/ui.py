"""
CLI ui command — open the browser dashboard.

From the interactive shell, the runtime is already running (embedded server).
This just opens the browser.
"""

import webbrowser

import click

from .. import theme


def handle_ui(func: str | None = None, args: str = ""):
    """Route ui command from the interactive shell."""
    click.echo()
    click.secho("  Opening dashboard in your browser...", fg=theme.MUTED)
    click.echo()

    from ..config import resolve_runtime_url
    url = resolve_runtime_url()
    browser_url = url.rstrip('/')
    if browser_url.endswith('/api'):
        browser_url = browser_url[:-4]

    webbrowser.open(browser_url)
    click.secho(f"  Opened: {browser_url}", fg=theme.ACCENT)
    click.echo()
