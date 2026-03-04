"""
CLI token commands — manage API tokens via the runtime.

token create     Generate a new API token (stored in runtime DB)
token list       List active tokens
token revoke     Revoke a token
"""

import click

from .. import theme
from ..config import resolve_runtime_url, resolve_runtime_token


@click.group("token")
def token_group():
    """Manage API tokens for runtime access."""
    pass


@token_group.command("create")
@click.option("--name", "-n", required=True, help="Token name (e.g. 'boomi-prod')")
@click.option("--org-id", "-o", required=True, help="Organization ID")
@click.option("--model-id", default=None, help="Scope to a specific model (omit for all models)")
@click.option("--permissions", "-p", default="read,write", help="Comma-separated: read,write,admin")
@click.option("--expires", "-e", default=365, type=int, help="Expiration in days (default: 365)")
def token_create(name, org_id, model_id, permissions, expires):
    """Create a new API token. The token is shown once — copy it."""
    runtime_url = resolve_runtime_url()
    admin_token = resolve_runtime_token()

    headers = {}
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"

    perms_list = [p.strip() for p in permissions.split(",")]

    try:
        import httpx

        with httpx.Client(timeout=15) as client:
            res = client.post(
                f"{runtime_url}/{org_id}/tokens",
                json={
                    "name": name,
                    "model_id": model_id,
                    "permissions": perms_list,
                    "expires_days": expires,
                },
                headers=headers,
            )

        if res.status_code in (200, 201):
            data = res.json()
            click.echo()
            click.secho(f"  Token created: {name}", fg=theme.SUCCESS)
            click.secho(f"    ID:      {data.get('id', '—')}", fg=theme.MUTED)
            click.secho(f"    Org:     {org_id}", fg=theme.MUTED)
            if model_id:
                click.secho(f"    Model:   {model_id}", fg=theme.MUTED)
            click.secho(f"    Perms:   {', '.join(perms_list)}", fg=theme.MUTED)
            click.secho(f"    Expires: {expires} days", fg=theme.MUTED)
            click.echo()
            click.secho("  Token (copy now — it won't be shown again):", fg=theme.WARNING)
            click.echo(f"  {data['token']}")
            click.echo()
        else:
            detail = res.text
            try:
                detail = res.json().get("detail", detail)
            except Exception:
                pass
            click.secho(f"  Failed: {detail}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {runtime_url}", fg=theme.ERROR)
        click.secho("  Is the runtime running?", fg=theme.MUTED)
    except Exception as e:
        click.secho(f"  Failed: {e}", fg=theme.ERROR)


@token_group.command("list")
@click.option("--org-id", "-o", required=True, help="Organization ID")
def token_list(org_id):
    """List active tokens."""
    runtime_url = resolve_runtime_url()
    admin_token = resolve_runtime_token()

    headers = {}
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"

    try:
        import httpx

        with httpx.Client(timeout=15) as client:
            res = client.get(
                f"{runtime_url}/{org_id}/tokens",
                headers=headers,
            )

        if res.status_code == 200:
            data = res.json()
            tokens = data.get("tokens", [])

            click.echo()
            click.secho(f"  Org: {org_id}", fg=theme.INFO)

            if not tokens:
                click.secho("  No active tokens.", fg=theme.MUTED)
                click.echo()
                return

            header = f"  {'PREFIX':<14} {'NAME':<24} {'PERMS':<14} EXPIRES"
            click.secho(header, fg=theme.TEXT_DIM)
            click.secho("  " + "─" * 64, fg=theme.TEXT_DIM)

            for t in tokens:
                prefix = t.get("token_prefix", "—")
                tname = (t.get("name") or "—")[:22]
                perms = ",".join(t.get("permissions", []))[:12]
                exp = (t.get("expires_at") or "never")[:10]
                click.echo(
                    click.style(f"  {prefix:<14} ", fg=theme.ACCENT)
                    + click.style(f"{tname:<24} ", fg=theme.TEXT)
                    + click.style(f"{perms:<14} ", fg=theme.INFO)
                    + click.style(exp, fg=theme.TEXT_DIM)
                )
            click.echo()
        else:
            click.secho(f"  Error: {res.text}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {runtime_url}", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Failed: {e}", fg=theme.ERROR)


@token_group.command("revoke")
@click.argument("token_id")
@click.option("--org-id", "-o", required=True, help="Organization ID")
def token_revoke(token_id, org_id):
    """Revoke a token by ID."""
    runtime_url = resolve_runtime_url()
    admin_token = resolve_runtime_token()

    headers = {}
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"

    try:
        import httpx

        with httpx.Client(timeout=15) as client:
            res = client.delete(
                f"{runtime_url}/{org_id}/tokens/{token_id}",
                headers=headers,
            )

        if res.status_code == 200:
            click.secho(f"  Token {token_id} revoked.", fg=theme.SUCCESS)
        elif res.status_code == 404:
            click.secho(f"  Token '{token_id}' not found or already revoked.", fg=theme.WARNING)
        else:
            click.secho(f"  Error: {res.text}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {runtime_url}", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Failed: {e}", fg=theme.ERROR)


# ── Handler for interactive shell ──

def handle_token(func: str | None, args: str = ""):
    """Route token subcommands from the interactive shell."""
    if func == "create":
        click.secho("  Use: glyphh token create --name <name> --org-id <org>", fg=theme.MUTED)
    elif func == "list":
        click.secho("  Use: glyphh token list --org-id <org>", fg=theme.MUTED)
    elif func == "revoke":
        if not args.strip():
            click.secho("  usage: token revoke <token-id> --org-id <org>", fg=theme.MUTED)
            return
        click.secho("  Use: glyphh token revoke <id> --org-id <org>", fg=theme.MUTED)
    else:
        click.echo()
        click.secho("  usage:", fg=theme.MUTED)
        click.secho("    token create --name <n> --org-id <org>   Create API token", fg=theme.MUTED)
        click.secho("    token list --org-id <org>                List active tokens", fg=theme.MUTED)
        click.secho("    token revoke <id> --org-id <org>         Revoke a token", fg=theme.MUTED)
        click.echo()
