"""
CLI query command — interactive chat interface for querying models.

glyphh query              Start interactive query session
glyphh query "my question"  Single query
"""

import json

import click

from .. import theme
from ..auth import get_user, is_logged_in, resolve_org_id
from ..config import resolve_runtime_url, resolve_runtime_token
from ..packaging import find_model_dir, read_manifest


def _resolve_query_context(model_id_override=None):
    """Resolve org_id, model_id, token for query commands."""
    if not is_logged_in():
        click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
        return None

    token = resolve_runtime_token()
    if not token:
        click.secho("  No token available. Run: glyphh auth login", fg=theme.ERROR)
        return None

    runtime_url = resolve_runtime_url()
    org_id = resolve_org_id(runtime_url)
    if not org_id:
        click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
        return None

    model_id = model_id_override
    if not model_id:
        model_dir = find_model_dir()
        if model_dir:
            manifest = read_manifest(model_dir)
            model_id = manifest.get("model_id", model_dir.name)

    if not model_id:
        click.secho("  No model found. Provide --model-id or run from a model directory.", fg=theme.ERROR)
        return None

    return {
        "runtime_url": runtime_url,
        "token": token,
        "org_id": org_id,
        "model_id": model_id,
        "headers": {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
    }


def _do_query(ctx, query_text, tool="nl_query", debug=False):
    """Execute a single query against the MCP endpoint."""
    import httpx

    url = f"{ctx['runtime_url']}/{ctx['org_id']}/{ctx['model_id']}/mcp"
    payload = {
        "tool": tool,
        "arguments": {"query": query_text},
    }
    if debug:
        payload["arguments"]["debug"] = True

    try:
        with httpx.Client(timeout=30) as client:
            res = client.post(url, json=payload, headers=ctx["headers"])

        if res.status_code == 200:
            data = res.json()
            _print_result(data, tool)
        elif res.status_code == 401:
            click.secho("  Authentication failed. Check your GLYPHH_TOKEN.", fg=theme.ERROR)
        else:
            detail = res.text
            try:
                detail = res.json().get("detail", detail)
            except Exception:
                pass
            click.secho(f"  Error: {detail}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {ctx['runtime_url']}", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Query failed: {e}", fg=theme.ERROR)


def _print_result(data, tool):
    """Pretty-print a query result."""
    state = data.get("state", "")
    confidence = data.get("confidence")
    query_time = data.get("query_time_ms")
    fact_tree = data.get("fact_tree", {})
    children = fact_tree.get("children", [])

    click.echo()

    if not children:
        click.secho("  No matches found.", fg=theme.WARNING)
        click.echo()
        return

    for i, child in enumerate(children):
        desc = child.get("description", "")
        value = child.get("value")
        sample = child.get("data_sample", {})

        # Confidence bar
        conf_str = f"{value:.0%}" if value is not None else "—"
        click.echo(
            click.style(f"  [{conf_str}] ", fg=theme.ACCENT)
            + click.style(desc, fg=theme.TEXT)
        )

        # Show data sample
        if sample:
            for k, v in sample.items():
                val_str = str(v)[:100]
                click.secho(f"         {k}: {val_str}", fg=theme.MUTED)

    # Footer
    meta_parts = []
    if confidence is not None:
        meta_parts.append(f"confidence={confidence:.0%}")
    if query_time is not None:
        meta_parts.append(f"{query_time:.1f}ms")
    meta_parts.append(data.get("match_method", ""))

    click.echo()
    click.secho(f"  {' · '.join(p for p in meta_parts if p)}", fg=theme.TEXT_DIM)
    click.echo()


@click.command("query")
@click.argument("text", required=False)
@click.option("--model-id", "-m", default=None, help="Model ID (defaults to manifest)")
@click.option("--gql", is_flag=True, help="Use GQL query instead of natural language")
@click.option("--debug", is_flag=True, help="Include debug info in response")
def query_command(text, model_id, gql, debug):
    """Query a deployed model. Starts interactive mode if no query given.

    \b
    Examples:
      glyphh query "how do I reset my password"
      glyphh query --gql 'FIND SIMILAR TO "reset password" LIMIT 5'
      glyphh query                    # interactive mode
    """
    ctx = _resolve_query_context(model_id)
    if not ctx:
        return

    tool = "gql_query" if gql else "nl_query"

    if text:
        # Single query mode
        _do_query(ctx, text, tool=tool, debug=debug)
        return

    # Interactive chat mode
    click.echo()
    click.secho(f"  Querying {ctx['model_id']} (org: {ctx['org_id']})", fg=theme.INFO)
    click.secho("  Type a question, or /gql to switch to GQL mode. /quit to exit.", fg=theme.MUTED)
    click.echo()

    current_tool = tool

    while True:
        try:
            prompt = click.style("  > ", fg=theme.ACCENT)
            line = click.prompt(prompt, prompt_suffix="", default="", show_default=False).strip()
        except (EOFError, KeyboardInterrupt):
            click.echo()
            break

        if not line:
            continue

        if line.lower() in ("/quit", "/exit", "/q"):
            break
        elif line.lower() == "/gql":
            current_tool = "gql_query"
            click.secho("  Switched to GQL mode.", fg=theme.MUTED)
            continue
        elif line.lower() == "/nl":
            current_tool = "nl_query"
            click.secho("  Switched to natural language mode.", fg=theme.MUTED)
            continue
        elif line.lower() == "/debug":
            debug = not debug
            click.secho(f"  Debug: {'on' if debug else 'off'}", fg=theme.MUTED)
            continue

        _do_query(ctx, line, tool=current_tool, debug=debug)


# ── Handler for interactive shell ──

def handle_query(func: str | None, args: str = ""):
    """Route query subcommands from the interactive shell."""
    # Rejoin func + args since the shell splits on spaces
    # e.g. "query how do I reset" -> func="how", args="do I reset"
    full_query = " ".join(p for p in [func, args] if p).strip()
    if full_query:
        ctx = _resolve_query_context()
        if ctx:
            _do_query(ctx, full_query)
    else:
        click.secho("  usage: query <your question>", fg=theme.MUTED)
        click.secho("  Or run: glyphh query  (for interactive mode)", fg=theme.MUTED)
