"""
CLI license commands — manage the runtime license file.

glyphh license show           Display current license info
glyphh license activate <key> Save a license (JSON string or key)
glyphh license deactivate     Remove the license file
"""

import json

import click

from .. import theme
from ...licensing import load_license, save_license, remove_license, LICENSE_FILE


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
        click.secho("  Models:    3 max", fg=theme.TEXT_DIM)
        click.secho("  Glyphs:    10,000 per model", fg=theme.TEXT_DIM)
        click.echo()
        click.secho("  Activate a license to unlock higher limits:", fg=theme.MUTED)
        click.secho("    glyphh license activate '<json>'", fg=theme.MUTED)
        click.echo()
        return

    click.secho(f"  License:   {info.license_id or '—'}", fg=theme.ACCENT)
    click.secho(f"  Org:       {info.org_id}", fg=theme.ACCENT)
    click.secho(f"  Tier:      {info.tier}", fg=theme.SUCCESS)

    models = "unlimited" if info.max_models == -1 else str(info.max_models)
    glyphs = "unlimited" if info.max_glyphs_per_model == -1 else f"{info.max_glyphs_per_model:,}"
    click.secho(f"  Models:    {models}", fg=theme.TEXT_DIM)
    click.secho(f"  Glyphs:    {glyphs} per model", fg=theme.TEXT_DIM)

    if info.expires_at:
        click.secho(f"  Expires:   {info.expires_at[:10]}", fg=theme.TEXT_DIM)
    else:
        click.secho("  Expires:   never", fg=theme.TEXT_DIM)

    click.echo()


@license_group.command("activate")
@click.argument("license_data")
def license_activate(license_data):
    """Activate a license. Pass a JSON string with tier info.

    Example: glyphh license activate '{"org_id":"my-org","tier":"pro"}'
    """
    try:
        data = json.loads(license_data)
    except json.JSONDecodeError:
        click.secho("  Invalid JSON. Expected format:", fg=theme.ERROR)
        click.secho('    \'{"org_id":"my-org","tier":"pro"}\'', fg=theme.MUTED)
        return

    if "tier" not in data:
        click.secho("  Missing 'tier' field in license data.", fg=theme.ERROR)
        return

    path = save_license(data)
    click.echo()
    click.secho(f"  License activated: {data.get('tier', 'unknown')} tier", fg=theme.SUCCESS)
    click.secho(f"  Saved to: {path}", fg=theme.TEXT_DIM)
    click.echo()
    click.secho("  Restart the runtime for changes to take effect.", fg=theme.MUTED)
    click.echo()


@license_group.command("deactivate")
def license_deactivate():
    """Remove the license file. Reverts to free tier."""
    if remove_license():
        click.secho("  License removed. Runtime will use free tier.", fg=theme.SUCCESS)
    else:
        click.secho("  No license file found.", fg=theme.MUTED)
