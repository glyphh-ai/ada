"""
CLI license commands — manage the runtime license (Ed25519 signed JWT).

glyphh license show           Display current license info
glyphh license activate <token> Save a signed JWT license token
glyphh license deactivate     Remove the license file
glyphh license refresh        Re-fetch license from Platform
"""

import click

from .. import theme
from ...licensing import load_license, save_license_token, remove_license, LICENSE_FILE, _verify_token
from ...metering import get_meter


def _show_usage(info):
    """Show current month's encoding operation usage."""
    meter = get_meter()
    usage = meter.get_usage(info.org_id)
    click.echo()
    click.secho(f"  Usage this month: {usage:,} ops", fg=theme.TEXT_DIM)
    if not info.is_unlimited:
        remaining = max(0, info.max_encodings_per_month - usage)
        pct = usage / info.max_encodings_per_month * 100
        if usage >= info.max_encodings_per_month:
            click.secho(f"  ⚠ Over limit ({pct:.0f}% used)", fg=theme.WARNING)
        elif info.encoding_warning_threshold() and usage >= info.encoding_warning_threshold():
            click.secho(f"  ⚠ {pct:.0f}% used — {remaining:,} ops remaining", fg=theme.WARNING)
        else:
            click.secho(f"  {pct:.0f}% used — {remaining:,} ops remaining", fg=theme.TEXT_DIM)
    click.echo()


@click.group("license")
def license_group():
    """Manage the runtime license."""
    pass


@license_group.command("show")
def license_show():
    """Display the current license information."""
    info = load_license()

    click.echo()
    click.secho("  Glyphh License", fg=theme.TEXT, bold=True)
    click.echo()

    if info.is_free and not info.license_id:
        click.secho("  Tier:      free (no license)", fg=theme.TEXT_DIM)
        click.secho(f"  Ops:       {info.format_limit()} / month", fg=theme.TEXT_DIM)
        click.secho(f"  Runtimes:  {info.max_runtimes}", fg=theme.TEXT_DIM)
        click.echo()
        click.secho("  Activate a license to unlock higher limits:", fg=theme.MUTED)
        click.secho("    license activate '<jwt-token>'", fg=theme.MUTED)
        click.echo()

        # Show current usage
        _show_usage(info)
        return

    click.secho(f"  License:   {info.license_id or '—'}", fg=theme.ACCENT)
    click.secho(f"  Org:       {info.org_id}", fg=theme.ACCENT)
    click.secho(f"  Tier:      {info.tier}", fg=theme.SUCCESS)
    click.secho("  Signature: verified", fg=theme.SUCCESS)

    click.secho(f"  Ops:       {info.format_limit()} / month", fg=theme.TEXT_DIM)
    click.secho(f"  Runtimes:  {info.max_runtimes}", fg=theme.TEXT_DIM)

    if info.expires_at:
        click.secho(f"  Expires:   {info.expires_at[:10]}", fg=theme.TEXT_DIM)
    else:
        click.secho("  Expires:   never", fg=theme.TEXT_DIM)

    # Show runtime_id for remote deployment
    from ..auth import _load_config
    runtime_id = _load_config().get("runtime_id")
    if runtime_id:
        click.echo()
        click.secho(f"  Runtime:   {runtime_id}", fg=theme.ACCENT)
        click.secho("             Set GLYPHH_RUNTIME_ID on remote runtimes to self-fetch this license.", fg=theme.TEXT_DIM)

    _show_usage(info)


@license_group.command("activate")
@click.argument("token")
def license_activate(token):
    """Activate a license. Pass the signed JWT token from the Platform.

    Example: glyphh license activate 'eyJhbGciOiJFZERTQSI...'
    """
    # Verify the token before saving
    claims = _verify_token(token)
    if not claims:
        click.secho("  Invalid or unverifiable license token.", fg=theme.ERROR)
        click.secho("  Get a valid token from the Glyphh dashboard.", fg=theme.MUTED)
        return

    path = save_license_token(token)
    tier = claims.get("tier", "unknown")
    click.echo()
    click.secho(f"  License activated: {tier} tier (signature verified)", fg=theme.SUCCESS)
    click.secho(f"  Saved to: {path}", fg=theme.TEXT_DIM)
    click.echo()
    click.secho("  Restart the runtime for changes to take effect.", fg=theme.MUTED)
    click.echo()


@license_group.command("refresh")
def license_refresh():
    """Re-fetch the license from the Platform (after plan upgrade, etc.)."""
    from ..auth import _load_config, get_token, get_api_url

    config = _load_config()
    runtime_id = config.get("runtime_id")
    token = get_token()

    if not runtime_id:
        click.secho("  No runtime registered. Run: auth login", fg=theme.ERROR)
        return

    if not token:
        click.secho("  Not logged in. Run: auth login", fg=theme.ERROR)
        return

    api_url = get_api_url()

    try:
        import httpx

        with httpx.Client(timeout=15) as client:
            res = client.get(
                f"{api_url}/runtimes/{runtime_id}/license",
                headers={"Authorization": f"Bearer {token}"},
            )

        if res.status_code == 200:
            data = res.json()
            jwt_token = data.get("token")
            if not jwt_token:
                click.secho("  No token in response.", fg=theme.ERROR)
                return

            # Verify before saving
            claims = _verify_token(jwt_token)
            if not claims:
                click.secho("  Received invalid license token from Platform.", fg=theme.ERROR)
                return

            path = save_license_token(jwt_token)
            tier = claims.get("tier", "unknown")
            click.echo()
            click.secho(f"  License refreshed: {tier} tier (signature verified)", fg=theme.SUCCESS)
            click.secho(f"  Saved to: {path}", fg=theme.TEXT_DIM)
            click.echo()
        elif res.status_code == 401:
            click.secho("  Session expired. Run: auth login", fg=theme.ERROR)
        elif res.status_code == 404:
            click.secho("  Runtime not found on Platform.", fg=theme.ERROR)
        else:
            click.secho(f"  Failed: {res.text}", fg=theme.ERROR)

    except Exception as e:
        click.secho(f"  Could not reach Platform: {e}", fg=theme.ERROR)


@license_group.command("deactivate")
def license_deactivate():
    """Remove the license file. Reverts to free tier."""
    if remove_license():
        click.secho("  License removed. Runtime will use free tier.", fg=theme.SUCCESS)
    else:
        click.secho("  No license file found.", fg=theme.MUTED)


# ── Handler for interactive shell ──

def handle_license(func: str | None, args: str = ""):
    """Route license subcommands from the interactive shell."""
    if func == "show":
        ctx = click.Context(license_show)
        ctx.invoke(license_show)
    elif func == "activate":
        token = args.strip().strip("'\"") if args else ""
        if not token:
            click.secho("  usage: license activate '<jwt-token>'", fg=theme.MUTED)
            return
        ctx = click.Context(license_activate)
        ctx.invoke(license_activate, token=token)
    elif func == "deactivate":
        ctx = click.Context(license_deactivate)
        ctx.invoke(license_deactivate)
    elif func == "refresh":
        ctx = click.Context(license_refresh)
        ctx.invoke(license_refresh)
    else:
        click.secho("  usage: license show | activate <token> | deactivate | refresh", fg=theme.MUTED)
