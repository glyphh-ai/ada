"""
CLI chat command — terminal REPL that calls the same MCP endpoints as the web UI.

glyphh chat                  Interactive REPL (auto-detects local dev server)
glyphh chat "send to slack"  Single query and exit
glyphh chat --gql            Start in GQL mode
glyphh chat --url http://...  Target a specific runtime URL
glyphh chat --token <tok>    Pass an API token (required for non-local deployments)

In local mode (DEPLOYMENT_MODE=local or localhost:8002 reachable without auth),
no login or token is needed. In deployed mode, provide --token or set GLYPHH_TOKEN.
"""

import os
import sys

import click

from .. import theme


# ── Defaults ────────────────────────────────────────────────────────────────

_DEFAULT_URL = "http://localhost:8002"
_LOCAL_ORG   = "local-dev-org"


# ── Context resolution ───────────────────────────────────────────────────────

def _resolve_context(model_id_override=None, url_override=None, token_override=None):
    """
    Build the query context dict.

    Priority order:
      1. Explicit CLI flags (url_override, token_override, model_id_override)
      2. Environment variables (RUNTIME_URL, GLYPHH_TOKEN)
      3. Local session (glyphh auth login)
      4. Local dev defaults (localhost:8002, local-dev-org, no token)
    """
    runtime_url = (
        url_override
        or os.environ.get("RUNTIME_URL", "")
        or _DEFAULT_URL
    ).rstrip("/")

    token = token_override or os.environ.get("GLYPHH_TOKEN", "")

    # Detect whether we're talking to a local dev server.
    # localhost / 127.0.0.1 always run as local-dev-org regardless of session.
    _is_local_url = (
        "localhost" in runtime_url
        or "127.0.0.1" in runtime_url
    )

    # Try to pull org/model from a logged-in session — only for non-local URLs
    org_id = None
    model_id = model_id_override
    if not _is_local_url:
        try:
            from ..auth import is_logged_in, get_token as _get_token, _load_config
            if is_logged_in():
                if not token:
                    token = _get_token() or ""
                cfg = _load_config()
                org_id = cfg.get("user", {}).get("org_id")
        except Exception:
            pass

    # Fall back to local-dev-org for localhost (always) or when no session
    if not org_id:
        org_id = _LOCAL_ORG

    # Discover model_id from manifest if not provided
    if not model_id:
        try:
            from ..packaging import find_model_dir, read_manifest
            model_dir = find_model_dir()
            if model_dir:
                manifest = read_manifest(model_dir)
                model_id = manifest.get("model_id", model_dir.name)
        except Exception:
            pass

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return {
        "runtime_url": runtime_url,
        "org_id": org_id,
        "model_id": model_id,
        "headers": headers,
        "local": org_id == _LOCAL_ORG and not token,
    }


# ── Result rendering ─────────────────────────────────────────────────────────

def _match_nodes(ft):
    """Extract Match nodes from the 'results' intermediate node of a FactTree dict."""
    for child in (ft or {}).get("children", []):
        if child.get("description") == "results":
            return [c for c in child.get("children", [])
                    if c.get("description", "").startswith("Match")]
    return []


_STATE_COLOR = {
    "DONE":          "SUCCESS",
    "ASK":           "WARNING",
    "BLOCKED":       "WARNING",
    "AUTH_REQUIRED": "ERROR",
    "ERROR":         "ERROR",
}


def _print_result(data):
    """Render a MCP response to the terminal using the same logic as the web UI."""
    ft = data.get("result")  # fact_tree JSON

    # State lives in content[0].data.state (NL responses)
    state = (
        (data.get("content") or [{}])[0].get("data", {}).get("state")
        or ("DONE" if ft else None)
    )

    click.echo()

    # State badge
    if state:
        color_attr = _STATE_COLOR.get(state, "INFO")
        color = getattr(theme, color_attr, theme.INFO)
        click.secho(f"  {state}", fg=color, bold=True)

    if not ft:
        click.secho("  No result.", fg=theme.TEXT_DIM)
        click.echo()
        return

    # Error embedded in fact_tree (description == "Query Error")
    if ft.get("description") == "Query Error":
        err = next(
            (c for c in ft.get("children", []) if c.get("description") == "Error Details"),
            None,
        )
        msg = err.get("value", "Query error") if err else "Query error"
        click.secho(f"  ✕  {msg}", fg=theme.ERROR)
        click.echo()
        return

    # Similarity search / list matches
    matches = _match_nodes(ft)
    if matches:
        for match in matches:
            v = match.get("value") or {}
            concept = v.get("concept_text", "—")
            score   = v.get("final_score")
            if score is not None:
                pct    = int(score * 100)
                filled = round(score * 12)
                bar    = "█" * filled + "░" * (12 - filled)
                click.echo(
                    click.style(f"  {pct:>3}%  ", fg=theme.ACCENT, bold=True)
                    + click.style(f"[{bar}]  ", fg=theme.TEXT_DIM)
                    + click.style(concept, fg=theme.TEXT)
                )
            else:
                click.secho(f"  •  {concept}", fg=theme.TEXT)
    else:
        click.secho("  No matches found.", fg=theme.WARNING)

    # Timing / method footer
    parts = []
    ms = data.get("query_time_ms")
    if ms is not None:
        parts.append(f"{ms:.1f}ms")
    method = data.get("match_method", "")
    if method and method not in ("none", ""):
        parts.append(method)
    qtype = data.get("query_type", "")
    if qtype and qtype not in ("unknown", ""):
        parts.append(qtype)

    click.echo()
    if parts:
        click.secho(f"  {' · '.join(parts)}", fg=theme.TEXT_DIM)
    click.echo()


# ── Single query execution ───────────────────────────────────────────────────

def _do_query(ctx, query_text, tool="nl_query"):
    """POST one query to the MCP endpoint and print the result."""
    import httpx

    if not ctx["model_id"]:
        click.secho(
            "  No model_id found. Run from a model directory or pass --model-id.",
            fg=theme.ERROR,
        )
        return

    url = f"{ctx['runtime_url']}/{ctx['org_id']}/{ctx['model_id']}/mcp"
    payload = {"tool": tool, "arguments": {"query": query_text}}

    try:
        with httpx.Client(timeout=30) as client:
            res = client.post(url, json=payload, headers=ctx["headers"])

        if res.status_code == 200:
            _print_result(res.json())
        elif res.status_code == 401:
            click.secho(
                "  401 Unauthorized — pass --token or set GLYPHH_TOKEN.",
                fg=theme.ERROR,
            )
        elif res.status_code == 404:
            click.secho(
                f"  404 — model '{ctx['model_id']}' not found at {ctx['runtime_url']}. "
                "Is the server running?",
                fg=theme.ERROR,
            )
        else:
            detail = res.text
            try:
                detail = res.json().get("detail", detail)
            except Exception:
                pass
            click.secho(f"  Error {res.status_code}: {detail}", fg=theme.ERROR)

    except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            click.secho(
                f"  Could not connect to runtime at {ctx['runtime_url']}. "
                "Start it with: glyphh dev .",
                fg=theme.ERROR,
            )
        else:
            click.secho(f"  Request failed: {exc}", fg=theme.ERROR)


# ── CLI command ──────────────────────────────────────────────────────────────

@click.command("chat")
@click.argument("text", required=False)
@click.option("--model-id", "-m", default=None, help="Model ID (default: from manifest.yaml)")
@click.option("--gql", is_flag=True, help="Start in GQL mode")
@click.option("--url", default=None, help=f"Runtime URL (default: {_DEFAULT_URL})")
@click.option("--token", default=None, help="API token for non-local deployments")
def chat_command(text, model_id, gql, url, token):
    """Terminal chat REPL — same MCP endpoints as the web UI.

    \b
    Works without login in local dev mode (glyphh dev .):
      glyphh chat                        # interactive REPL
      glyphh chat "send a slack message" # single query

    \b
    For deployed models:
      glyphh chat --token <tok>
      GLYPHH_TOKEN=<tok> glyphh chat

    \b
    Slash commands inside the REPL:
      /gql     switch to GQL mode
      /nl      switch to natural language mode
      /quit    exit
    """
    ctx = _resolve_context(model_id, url, token)
    tool = "gql_query" if gql else "nl_query"

    # Single-query mode
    if text:
        _do_query(ctx, text, tool=tool)
        return

    # Interactive REPL
    model_label = ctx["model_id"] or "unknown"
    mode_label  = "local" if ctx["local"] else ctx["org_id"]

    click.echo()
    click.secho(
        f"  glyphh chat  ·  {model_label}  ·  {mode_label}",
        fg=theme.TEXT, bold=True,
    )
    click.secho(
        "  /gql  /nl  /quit  — or just type",
        fg=theme.TEXT_DIM,
    )
    click.echo()

    current_tool = tool

    while True:
        mode_indicator = click.style("GQL" if current_tool == "gql_query" else " NL", fg=theme.ACCENT)
        prompt = click.style("  [", fg=theme.TEXT_DIM) + mode_indicator + click.style("] › ", fg=theme.TEXT_DIM)

        try:
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
            click.secho("  → GQL mode", fg=theme.TEXT_DIM)
        elif line.lower() == "/nl":
            current_tool = "nl_query"
            click.secho("  → NL mode", fg=theme.TEXT_DIM)
        else:
            _do_query(ctx, line, tool=current_tool)
