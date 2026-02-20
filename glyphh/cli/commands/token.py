"""
CLI token commands — manage JWT tokens for runtime access.

token create     Generate a new JWT token
token list       List active tokens
token revoke     Revoke a token
token refresh    Refresh an expiring token
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from .. import theme
from ..auth import get_user, _load_config


TOKEN_STORE = Path.home() / ".glyphh" / "tokens.json"


def _get_secret():
    """Get JWT secret from env or config."""
    secret = os.environ.get("JWT_SECRET_KEY")
    if not secret:
        click.secho("  JWT_SECRET_KEY not set. Required for token management.", fg=theme.ERROR)
        click.secho("  export JWT_SECRET_KEY=your-secret-key", fg=theme.MUTED)
        return None
    return secret


def _get_org_id():
    """Get org_id from stored session."""
    config = _load_config()
    user = config.get("user", {})
    return user.get("org_id") or config.get("org_id")


def _load_store():
    """Load token metadata store."""
    if TOKEN_STORE.exists():
        return json.loads(TOKEN_STORE.read_text())
    return {"tokens": []}


def _save_store(store):
    """Save token metadata store."""
    TOKEN_STORE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_STORE.write_text(json.dumps(store, indent=2))


@click.group("token")
def token_group():
    """Manage JWT tokens for runtime access."""
    pass


@token_group.command("create")
@click.option("--name", "-n", required=True, help="Token name (e.g. 'boomi-integration')")
@click.option("--org-id", default=None, help="Organization ID (defaults to your logged-in org)")
@click.option("--role", default="service", help="Role: service, admin, user")
@click.option("--expires", "-e", default=365, help="Expiration in days (default: 365)")
def token_create(name, org_id, role, expires):
    """Generate a new JWT token for runtime access."""
    import uuid

    try:
        import jwt as pyjwt
    except ImportError:
        click.secho("  PyJWT required: pip install PyJWT", fg=theme.ERROR)
        return

    secret = _get_secret()
    if not secret:
        return

    # Resolve org_id from session if not provided
    if not org_id:
        org_id = _get_org_id()
    if not org_id:
        click.secho("  No org_id found. Provide --org-id or log in first.", fg=theme.ERROR)
        return

    now = int(time.time())
    token_id = str(uuid.uuid4())[:8]
    payload = {
        "sub": f"token_{token_id}",
        "org_id": org_id,
        "role": role,
        "type": "access",
        "name": name,
        "iat": now,
        "exp": now + (expires * 86400),
        "jti": token_id,
    }

    token = pyjwt.encode(payload, secret, algorithm="HS256")

    # Store metadata (not the token itself)
    store = _load_store()
    store["tokens"].append({
        "id": token_id,
        "name": name,
        "org_id": org_id,
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": datetime.fromtimestamp(payload["exp"], tz=timezone.utc).isoformat(),
        "revoked": False,
    })
    _save_store(store)

    click.echo()
    click.secho(f"  ✓ Token created: {name}", fg=theme.SUCCESS)
    click.secho(f"    ID:      {token_id}", fg=theme.MUTED)
    click.secho(f"    Org:     {org_id}", fg=theme.MUTED)
    click.secho(f"    Role:    {role}", fg=theme.MUTED)
    click.secho(f"    Expires: {expires} days", fg=theme.MUTED)
    click.echo()
    click.secho("  Token (copy now — it won't be shown again):", fg=theme.WARNING)
    click.echo(f"  {token}")
    click.echo()


@token_group.command("list")
def token_list():
    """List active tokens."""
    # Show org_id for convenience
    org_id = _get_org_id()
    if org_id:
        click.echo()
        click.secho(f"  Org: {org_id}", fg=theme.INFO)

    store = _load_store()
    active = [t for t in store["tokens"] if not t.get("revoked")]

    if not active:
        click.secho("  No active tokens.", fg=theme.MUTED)
        return

    click.echo()
    header = f"  {'ID':<10} {'NAME':<24} {'ORG':<20} {'ROLE':<10} EXPIRES"
    click.secho(header, fg=theme.TEXT_DIM)
    click.secho("  " + "─" * 80, fg=theme.TEXT_DIM)

    for t in active:
        exp = t.get("expires_at", "—")[:10]
        click.echo(
            click.style(f"  {t['id']:<10} ", fg=theme.ACCENT)
            + click.style(f"{t['name']:<24} ", fg=theme.TEXT)
            + click.style(f"{t['org_id']:<20} ", fg=theme.MUTED)
            + click.style(f"{t['role']:<10} ", fg=theme.INFO)
            + click.style(exp, fg=theme.TEXT_DIM)
        )
    click.echo()


@token_group.command("revoke")
@click.argument("token_id")
def token_revoke(token_id):
    """Revoke a token by ID."""
    store = _load_store()
    found = False
    for t in store["tokens"]:
        if t["id"] == token_id and not t.get("revoked"):
            t["revoked"] = True
            found = True
            break

    if not found:
        click.secho(f"  Token '{token_id}' not found or already revoked.", fg=theme.WARNING)
        return

    _save_store(store)
    click.secho(f"  ✓ Token {token_id} revoked.", fg=theme.SUCCESS)


@token_group.command("refresh")
@click.argument("token_id")
@click.option("--expires", "-e", default=365, help="New expiration in days")
def token_refresh(token_id, expires):
    """Refresh a token — generates a new JWT with extended expiration."""
    import uuid

    try:
        import jwt as pyjwt
    except ImportError:
        click.secho("  PyJWT required: pip install PyJWT", fg=theme.ERROR)
        return

    secret = _get_secret()
    if not secret:
        return

    store = _load_store()
    target = None
    for t in store["tokens"]:
        if t["id"] == token_id and not t.get("revoked"):
            target = t
            break

    if not target:
        click.secho(f"  Token '{token_id}' not found or revoked.", fg=theme.WARNING)
        return

    # Revoke old, create new
    target["revoked"] = True

    now = int(time.time())
    new_id = str(uuid.uuid4())[:8]
    payload = {
        "sub": f"token_{new_id}",
        "org_id": target["org_id"],
        "role": target["role"],
        "type": "access",
        "name": target["name"],
        "iat": now,
        "exp": now + (expires * 86400),
        "jti": new_id,
    }

    token = pyjwt.encode(payload, secret, algorithm="HS256")

    store["tokens"].append({
        "id": new_id,
        "name": target["name"],
        "org_id": target["org_id"],
        "role": target["role"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": datetime.fromtimestamp(payload["exp"], tz=timezone.utc).isoformat(),
        "revoked": False,
    })
    _save_store(store)

    click.echo()
    click.secho(f"  ✓ Token refreshed: {target['name']}", fg=theme.SUCCESS)
    click.secho(f"    Old ID: {token_id} (revoked)", fg=theme.MUTED)
    click.secho(f"    New ID: {new_id}", fg=theme.MUTED)
    click.echo()
    click.secho("  New token (copy now — it won't be shown again):", fg=theme.WARNING)
    click.echo(f"  {token}")
    click.echo()


# ── Handler for interactive shell ──

def handle_token(func: str | None, args: str = ""):
    """Route token subcommands from the interactive shell."""
    if func == "create":
        click.secho("  Use: glyphh token create --name <name> --org-id <org>", fg=theme.MUTED)
    elif func == "list":
        token_list.invoke(click.Context(token_list))
    elif func == "revoke":
        if not args.strip():
            click.secho("  usage: token revoke <token-id>", fg=theme.MUTED)
            return
        ctx = click.Context(token_revoke)
        token_revoke.invoke(ctx, token_id=args.strip())
    elif func == "refresh":
        if not args.strip():
            click.secho("  usage: token refresh <token-id>", fg=theme.MUTED)
            return
        ctx = click.Context(token_refresh)
        token_refresh.invoke(ctx, token_id=args.strip(), expires=365)
    else:
        click.echo()
        click.secho("  usage:", fg=theme.MUTED)
        click.secho("    token create --name <n> --org-id <org>  Generate JWT", fg=theme.MUTED)
        click.secho("    token list                              List active tokens", fg=theme.MUTED)
        click.secho("    token revoke <id>                       Revoke a token", fg=theme.MUTED)
        click.secho("    token refresh <id>                      Refresh expiring token", fg=theme.MUTED)
        click.echo()
