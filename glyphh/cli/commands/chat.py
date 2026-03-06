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

# ── Readline history (optional — graceful fallback on Windows) ────────────────

_HISTORY_FILE = os.path.expanduser("~/.glyphh/chat_history")

try:
    import readline as _readline

    def _setup_history() -> None:
        os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
        try:
            _readline.read_history_file(_HISTORY_FILE)
        except FileNotFoundError:
            pass
        _readline.set_history_length(500)

    def _save_history() -> None:
        try:
            _readline.write_history_file(_HISTORY_FILE)
        except OSError:
            pass

except ImportError:
    def _setup_history() -> None: pass   # noqa: E704
    def _save_history() -> None: pass    # noqa: E704

import time

from .. import theme
from ..config import resolve_runtime_url, resolve_runtime_token


# ── Defaults ────────────────────────────────────────────────────────────────

_LOCAL_ORG   = "local-dev-org"
_READY_POLL_INTERVAL = 2  # seconds between readiness checks
_READY_TIMEOUT = 120      # max seconds to wait for model deployment


# ── Context resolution ───────────────────────────────────────────────────────

def _resolve_context(model_id_override=None, url_override=None, token_override=None):
    """
    Build the query context dict.

    Priority order:
      1. Explicit CLI flags (url_override, token_override, model_id_override)
      2. Environment variables (RUNTIME_URL, GLYPHH_TOKEN)
      3. ~/.glyphh/config.json (runtime_url, runtime_token)
      4. Local session (glyphh auth login)
      5. Local dev defaults (localhost:8002, local-dev-org, no token)
    """
    runtime_url = resolve_runtime_url(cli_override=url_override)
    token = resolve_runtime_token(cli_override=token_override) or ""

    # Detect whether we're talking to a local dev server.
    _is_local_url = (
        "localhost" in runtime_url
        or "127.0.0.1" in runtime_url
    )

    from ..auth import resolve_org_id

    model_id = model_id_override
    org_id = resolve_org_id(runtime_url) or _LOCAL_ORG

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


# ── Readiness check ──────────────────────────────────────────────────────────

def _wait_for_ready(ctx) -> bool:
    """Check if the runtime and model are ready. Wait with a spinner if deploying.

    Returns True if ready, False if timed out or unreachable.
    """
    import httpx

    if not ctx.get("model_id"):
        return True  # Will fail later with a clear message

    ready_url = f"{ctx['runtime_url']}/{ctx['org_id']}/{ctx['model_id']}/ready"
    health_url = f"{ctx['runtime_url']}/health"
    deadline = time.time() + _READY_TIMEOUT
    shown_waiting = False

    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5) as client:
                res = client.get(ready_url, headers=ctx["headers"])

            if res.status_code == 200:
                data = res.json()
                if data.get("ready"):
                    if shown_waiting:
                        click.secho("  Ready.", fg=theme.SUCCESS)
                    return True

                status = data.get("status", "unknown")
                if not shown_waiting:
                    click.secho(
                        f"  Runtime is starting up ({status}). Waiting for model to deploy...",
                        fg=theme.MUTED,
                    )
                    shown_waiting = True

            elif res.status_code == 404:
                # Model not deployed yet — runtime is up but model hasn't loaded
                if not shown_waiting:
                    click.secho(
                        "  Waiting for model to deploy...",
                        fg=theme.MUTED,
                    )
                    shown_waiting = True

        except Exception:
            # Runtime not reachable yet — check if Docker is even running
            if not shown_waiting:
                click.secho(
                    "  Waiting for runtime to start...",
                    fg=theme.MUTED,
                )
                shown_waiting = True

        time.sleep(_READY_POLL_INTERVAL)

    click.secho(
        f"  Timed out waiting for model to be ready ({_READY_TIMEOUT}s). "
        "Check: docker compose logs -f runtime",
        fg=theme.ERROR,
    )
    return False


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

    # Check for errors first
    if data.get("isError"):
        click.echo()
        error_msg = data.get("error") or ""
        # Also check content for text error messages
        content = (data.get("content") or [{}])[0]
        if content.get("type") == "text" and content.get("text"):
            error_msg = content["text"]
        click.secho(f"  Error: {error_msg}", fg=theme.ERROR)
        click.echo()
        return

    # State lives in content[0].data.state (NL responses with JSON content)
    content_data = {}
    content = (data.get("content") or [{}])[0]
    if content.get("type") == "json":
        content_data = content.get("data", {})
    state = content_data.get("state") or ("DONE" if ft else None)

    click.echo()

    # State badge
    if state:
        color_attr = _STATE_COLOR.get(state, "INFO")
        color = getattr(theme, color_attr, theme.INFO)
        click.secho(f"  {state}", fg=color, bold=True)

    # ASK state — show the question and numbered disambiguation options
    if state == "ASK":
        ask = content_data.get("ask", {})
        if ask:
            question = ask.get("question") or "Please clarify your request."
            click.secho(f"  {question}", fg=theme.WARNING)
            missing = ask.get("missing_slots") or []
            if missing:
                click.secho(f"  Missing: {', '.join(missing)}", fg=theme.TEXT_DIM)
            options = ask.get("disambiguation_options") or []
            option_labels = []
            for i, opt in enumerate(options, 1):
                label = opt.get("suggestion") or opt.get("intent") or str(opt)
                # Strip trailing match score like "(74% match)" for re-query
                import re as _re
                clean = _re.sub(r"\s*\(\d+%\s*match\)\s*$", "", label).strip()
                option_labels.append(clean)
                click.echo(
                    click.style(f"    {i}. ", fg=theme.ACCENT, bold=True)
                    + click.style(label, fg=theme.TEXT)
                )
            if option_labels:
                click.echo()
                click.secho("  Enter a number to select, or type a new query.", fg=theme.MUTED)
        else:
            option_labels = []
            click.secho("  Please clarify your request.", fg=theme.WARNING)
        click.echo()
        return option_labels

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
                pct    = score * 100
                filled = round(score * 12)
                bar    = "█" * filled + "░" * (12 - filled)
                click.echo(
                    click.style(f"  {pct:>5.1f}%  ", fg=theme.ACCENT, bold=True)
                    + click.style(f"[{bar}]  ", fg=theme.TEXT_DIM)
                    + click.style(concept, fg=theme.TEXT)
                )
            else:
                click.secho(f"  •  {concept}", fg=theme.TEXT)

            # Show response/metadata if present
            meta = v.get("metadata") or {}
            response_text = meta.get("response", "")
            if response_text:
                click.secho(f"          {response_text}", fg=theme.TEXT_DIM)
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

def _do_query(ctx, query_text, tool="nl_query", stage="auto", confirmed=False):
    """POST one query to the MCP endpoint and print the result.
    Returns a list of disambiguation option strings if ASK state, else None.
    """
    import httpx

    if not ctx["model_id"]:
        click.secho(
            "  No model_id found. Run from a model directory or pass --model-id.",
            fg=theme.ERROR,
        )
        return None

    url = f"{ctx['runtime_url']}/{ctx['org_id']}/{ctx['model_id']}/mcp"
    args = {"query": query_text}
    if tool == "nl_query" and stage != "auto":
        args["stage"] = stage
    if confirmed:
        args["confirmed"] = True
    payload = {"tool": tool, "arguments": args}

    try:
        with httpx.Client(timeout=30) as client:
            res = client.post(url, json=payload, headers=ctx["headers"])

        if res.status_code == 200:
            return _print_result(res.json())
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


# ── REPL loop (shared by CLI command and shell handler) ──────────────────────

_STAGE_VALUES = ("auto", "patterns", "data")


def _run_repl(ctx, tool="nl_query"):
    """Run the interactive chat REPL. Returns when the user exits."""
    model_label = ctx["model_id"] or "unknown"
    mode_label  = "local" if ctx["local"] else ctx["org_id"]

    click.echo()
    click.secho(
        f"  glyphh chat  ·  {model_label}  ·  {mode_label}",
        fg=theme.TEXT, bold=True,
    )
    click.secho(
        "  /gql  /nl  /stage <auto|patterns|data>  /quit  — or just type",
        fg=theme.TEXT_DIM,
    )
    click.echo()

    current_tool = tool
    current_stage = "auto"
    pending_options = []  # disambiguation options from last ASK response
    _setup_history()

    while True:
        mode_indicator = click.style("GQL" if current_tool == "gql_query" else " NL", fg=theme.ACCENT)
        stage_indicator = ""
        if current_tool == "nl_query" and current_stage != "auto":
            stage_indicator = click.style(f":{current_stage}", fg=theme.WARNING)
        prompt = click.style("  [", fg=theme.TEXT_DIM) + mode_indicator + stage_indicator + click.style("] › ", fg=theme.TEXT_DIM)

        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            click.echo()
            _save_history()
            break

        if not line:
            continue

        if line.lower() in ("/quit", "/exit", "/q"):
            _save_history()
            break
        elif line.lower() == "/gql":
            current_tool = "gql_query"
            click.secho("  → GQL mode", fg=theme.TEXT_DIM)
        elif line.lower() == "/nl":
            current_tool = "nl_query"
            click.secho("  → NL mode", fg=theme.TEXT_DIM)
        elif line.lower().startswith("/stage"):
            parts = line.split(None, 1)
            if len(parts) < 2 or parts[1].lower() not in _STAGE_VALUES:
                click.secho(f"  Usage: /stage <{'|'.join(_STAGE_VALUES)}>", fg=theme.TEXT_DIM)
                click.secho(f"  Current: {current_stage}", fg=theme.TEXT_DIM)
            else:
                current_stage = parts[1].lower()
                click.secho(f"  → stage: {current_stage}", fg=theme.TEXT_DIM)
        else:
            # If user typed a number and we have pending options, select that option
            query = line
            is_selection = False
            if pending_options and line.isdigit():
                idx = int(line) - 1
                if 0 <= idx < len(pending_options):
                    query = pending_options[idx]
                    is_selection = True
                    click.secho(f"  → {query}", fg=theme.TEXT_DIM)

            result = _do_query(ctx, query, tool=current_tool, stage=current_stage, confirmed=is_selection)
            pending_options = result if result else []


# ── CLI command ──────────────────────────────────────────────────────────────

@click.command("chat")
@click.argument("text", required=False)
@click.option("--model-id", "-m", default=None, help="Model ID (default: from manifest.yaml)")
@click.option("--gql", is_flag=True, help="Start in GQL mode")
@click.option("--url", default=None, help="Runtime URL (default: from config or localhost:8002)")
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

    if not _wait_for_ready(ctx):
        return

    if text:
        _do_query(ctx, text, tool=tool)
        return

    _run_repl(ctx, tool=tool)


# ── Handler for interactive shell ──

def handle_chat(func: str | None, args: str = ""):
    """Route chat subcommands from the interactive shell."""
    full_query = " ".join(p for p in [func, args] if p).strip()
    ctx = _resolve_context()
    if full_query:
        _do_query(ctx, full_query)
    else:
        _run_repl(ctx)
