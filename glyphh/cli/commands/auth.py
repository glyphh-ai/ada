"""
CLI auth commands — authentication management.

auth login     Log in via browser (device flow)
auth logout    Log out and clear session
auth status    Show current auth status
"""

import click

from .. import theme
from ..auth import (
    is_logged_in,
    get_user,
    device_login,
    clear_session,
)


@click.group("auth")
def auth_group():
    """Authentication management."""
    pass


@auth_group.command("login")
def auth_login():
    """Log in via browser."""
    if is_logged_in():
        user = get_user() or {}
        name = user.get("first_name", user.get("email", ""))
        click.secho(f"  Already logged in as {name}.", fg=theme.MUTED)
        return
    device_login()


@auth_group.command("logout")
def auth_logout():
    """Log out and clear session."""
    clear_session()
    click.secho("  Logged out.", fg=theme.MUTED)


@auth_group.command("status")
def auth_status():
    """Show current auth status."""
    if is_logged_in():
        user = get_user() or {}
        name = user.get("first_name", user.get("email", ""))
        org_id = user.get("org_id", "—")
        click.echo()
        click.secho(f"  ● Logged in as {name}", fg=theme.SUCCESS)
        click.secho(f"    Org:     {org_id}", fg=theme.MUTED)

        # Show license tier and usage
        from ...licensing import load_license
        from ...metering import get_meter
        info = load_license()
        if info.is_free and not info.license_id:
            click.secho(f"    License: free (no license)", fg=theme.TEXT_DIM)
        else:
            click.secho(f"    License: {info.tier} ({info.format_limit()} ops/mo)", fg=theme.MUTED)

        meter = get_meter()
        usage = meter.get_usage(info.org_id)
        if not info.is_unlimited and usage > 0:
            pct = usage / info.max_encodings_per_month * 100
            if usage >= info.max_encodings_per_month:
                click.secho(f"    Usage:   {usage:,} / {info.format_limit()} ops (over limit)", fg=theme.WARNING)
            elif info.encoding_warning_threshold() and usage >= info.encoding_warning_threshold():
                click.secho(f"    Usage:   {usage:,} / {info.format_limit()} ops ({pct:.0f}%)", fg=theme.WARNING)
            else:
                click.secho(f"    Usage:   {usage:,} / {info.format_limit()} ops", fg=theme.MUTED)

        # Show runtime registration
        from ..auth import _load_config
        config = _load_config()
        if config.get("runtime_id"):
            click.secho(f"    Runtime: {config['runtime_id'][:8]}…", fg=theme.MUTED)

        click.echo()
    else:
        click.echo()
        click.secho(f"  ● Not logged in", fg=theme.ERROR)
        click.secho(f"  Run: auth login", fg=theme.TEXT_DIM)
        click.echo()


# ── Handler for interactive shell ──

def handle_auth(func: str | None, args: str = ""):
    """Route auth subcommands from the interactive shell."""
    if func == "login":
        ctx = click.Context(auth_login)
        ctx.invoke(auth_login)
    elif func == "logout":
        ctx = click.Context(auth_logout)
        ctx.invoke(auth_logout)
    elif func == "status":
        ctx = click.Context(auth_status)
        ctx.invoke(auth_status)
    else:
        click.secho("  usage: auth login | auth logout | auth status", fg=theme.MUTED)
