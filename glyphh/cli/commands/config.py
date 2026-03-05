"""
CLI config commands — manage persistent runtime connection settings.

glyphh config show                  Display current config
glyphh config set endpoint <url>    Persist runtime URL
glyphh config set token <jwt>       Persist runtime auth token
glyphh config clear                 Remove persisted config
"""

import os

import click

from .. import theme
from ..auth import _load_config, _save_config
from ..config import (
    DEFAULT_RUNTIME_URL,
    RuntimeConfig,
    ConfigurationError,
    resolve_runtime_url,
    resolve_runtime_token,
)


@click.group("config")
def config_group():
    """Manage CLI configuration for runtime connections."""
    pass


@config_group.command("show")
def config_show():
    """Display the current effective configuration."""
    url, url_source = _resolve_with_source("url")
    token, token_source = _resolve_with_source("token")

    click.echo()
    click.secho("  Glyphh Configuration", fg=theme.TEXT, bold=True)
    click.echo()

    click.secho(f"  Endpoint:  {url}", fg=theme.ACCENT)
    click.secho(f"             source: {url_source}", fg=theme.TEXT_DIM)
    click.echo()

    if token:
        masked = token[:8] + "..." + token[-4:] if len(token) > 16 else "***"
        click.secho(f"  Token:     {masked}", fg=theme.ACCENT)
    else:
        click.secho("  Token:     (none)", fg=theme.TEXT_DIM)
    click.secho(f"             source: {token_source}", fg=theme.TEXT_DIM)
    click.echo()

    try:
        rc = RuntimeConfig(runtime_url=url, jwt_token=token)
        if rc.is_local:
            click.secho("  Mode:      local (no auth required)", fg=theme.SUCCESS)
        elif rc.has_auth:
            click.secho("  Mode:      remote (authenticated)", fg=theme.SUCCESS)
        else:
            click.secho("  Mode:      remote (no token — auth may fail)", fg=theme.WARNING)
    except ConfigurationError:
        pass

    click.echo()


@config_group.group("set")
def config_set():
    """Set a configuration value."""
    pass


@config_set.command("endpoint")
@click.argument("url")
def config_set_endpoint(url):
    """Persist a runtime endpoint URL.

    Example: glyphh config set endpoint https://my-app.herokuapp.com
    """
    try:
        RuntimeConfig(runtime_url=url)
    except ConfigurationError as e:
        click.secho(f"  Invalid URL: {e}", fg=theme.ERROR)
        return

    config = _load_config()
    config["runtime_url"] = url.rstrip("/")
    _save_config(config)

    click.secho(f"  Endpoint saved: {url.rstrip('/')}", fg=theme.SUCCESS)


@config_set.command("token")
@click.argument("jwt")
def config_set_token(jwt):
    """Persist a runtime auth token (JWT).

    Example: glyphh config set token eyJhbGciOi...
    """
    config = _load_config()
    config["runtime_token"] = jwt
    _save_config(config)

    masked = jwt[:8] + "..." + jwt[-4:] if len(jwt) > 16 else "***"
    click.secho(f"  Token saved: {masked}", fg=theme.SUCCESS)


@config_group.command("clear")
@click.option("--endpoint", is_flag=True, help="Clear only the endpoint")
@click.option("--token", "clear_token", is_flag=True, help="Clear only the token")
def config_clear(endpoint, clear_token):
    """Remove persisted runtime configuration.

    Without flags, clears both endpoint and token.
    Does not affect session auth (glyphh auth login).
    """
    config = _load_config()

    clear_all = not endpoint and not clear_token

    removed = []
    if (clear_all or endpoint) and "runtime_url" in config:
        del config["runtime_url"]
        removed.append("endpoint")
    if (clear_all or clear_token) and "runtime_token" in config:
        del config["runtime_token"]
        removed.append("token")

    if removed:
        _save_config(config)
        click.secho(f"  Cleared: {', '.join(removed)}", fg=theme.SUCCESS)
    else:
        click.secho("  Nothing to clear.", fg=theme.MUTED)


def _resolve_with_source(kind: str) -> tuple:
    """Resolve a config value and return (value, source_description)."""
    config = _load_config()

    if kind == "url":
        env_val = os.environ.get("RUNTIME_URL", "").strip()
        if env_val:
            return env_val.rstrip("/"), "RUNTIME_URL env var"
        stored = config.get("runtime_url", "").strip()
        if stored:
            return stored.rstrip("/"), "~/.glyphh/config.json"
        return DEFAULT_RUNTIME_URL, "default (localhost)"

    elif kind == "token":
        env_val = os.environ.get("GLYPHH_TOKEN", "").strip()
        if env_val:
            return env_val, "GLYPHH_TOKEN env var"
        stored_rt = config.get("runtime_token", "").strip()
        if stored_rt:
            return stored_rt, "~/.glyphh/config.json (runtime_token)"
        return "", "none (run: auth logout → auth login to bootstrap)"

    return "", "unknown"


# ── Handler for interactive shell ──

def handle_config(func: str | None, args: str = ""):
    """Route config subcommands from the interactive shell."""
    if func == "show":
        ctx = click.Context(config_show)
        ctx.invoke(config_show)
    elif func == "set":
        parts = args.split(None, 1)
        if not parts:
            click.secho("  usage: config set endpoint <url> | config set token <jwt>", fg=theme.MUTED)
            return
        sub = parts[0].lower()
        val = parts[1] if len(parts) > 1 else ""
        if sub == "endpoint" and val:
            ctx = click.Context(config_set_endpoint)
            ctx.invoke(config_set_endpoint, url=val)
        elif sub == "token" and val:
            ctx = click.Context(config_set_token)
            ctx.invoke(config_set_token, jwt=val)
        else:
            click.secho("  usage: config set endpoint <url> | config set token <jwt>", fg=theme.MUTED)
    elif func == "clear":
        # Parse optional flags from args
        endpoint = "--endpoint" in args
        clear_token = "--token" in args
        ctx = click.Context(config_clear)
        ctx.invoke(config_clear, endpoint=endpoint, clear_token=clear_token)
    else:
        click.secho("  usage: config show | config set endpoint <url> | config set token <jwt> | config clear", fg=theme.MUTED)
