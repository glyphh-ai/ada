"""
Hub — browse and install models from the Glyphh model registry.

    glyphh> hub                  Browse models (alias for hub list)
    glyphh> hub list             Browse models with pagination
    glyphh> hub search <query>   Search models by name/tag/category
    glyphh> hub install <id>     Deploy a model from the registry
"""

from pathlib import Path

import click

try:
    from glyphh.cli import theme
except ImportError:
    class _FallbackTheme:
        PRIMARY = "magenta"
        ACCENT = "bright_magenta"
        MUTED = "bright_black"
        SUCCESS = "green"
        WARNING = "yellow"
        ERROR = "red"
        INFO = "cyan"
        TEXT = "white"
        TEXT_DIM = "bright_black"
    theme = _FallbackTheme()


# ── Release download base URL ─────────────────────────────────────────────────
# Models are published as .glyphh files attached to GitHub Releases.
# Tag format: <model-dir>/v<version>  e.g. firewall/v0.9.0
# Asset name: <model-id>.glyphh       e.g. model-firewall.glyphh
#
# When the Platform hub API is ready, this switches to:
#   GET /hub/models/{id}/download → signed URL to .glyphh artifact

_RELEASES_REPO = "glyphh-ai/glyphh-models"
_RELEASES_API = f"https://api.github.com/repos/{_RELEASES_REPO}/releases"


# ── Model registry (hardcoded — will switch to Platform API) ────────────────

MODELS = [
    {
        "id": "toolrouter",
        "model_id": "model-toolrouter",
        "name": "SaaS Tool Router",
        "description": "Routes NL SaaS requests to tool functions across 8 domains. 38 tools, sub-10ms, zero tokens.",
        "category": "routing",
        "icon": "\U0001f500",
        "version": "2.2.0",
        "author": "Glyphh AI",
        "tags": ["tool-routing", "saas", "intent-matching"],
        "release_tag": "toolrouter/v2.2.0",
    },
    {
        "id": "faq-helpdesk",
        "model_id": "model-faq",
        "name": "Glyphh AI Knowledge Base",
        "description": "FAQ model covering all aspects of the Glyphh platform. 156 entries across 11 categories.",
        "category": "faq",
        "icon": "\U0001f4ac",
        "version": "0.2.0",
        "author": "Glyphh AI",
        "tags": ["faq", "knowledge-base", "documentation"],
        "release_tag": "faq/v0.2.0",
    },
    {
        "id": "customer-churn",
        "model_id": "model-churn",
        "name": "Customer Churn Predictor",
        "description": "Encodes customer usage metrics into HDC vectors to identify churn risk patterns via similarity.",
        "category": "prediction",
        "icon": "\U0001f4c9",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "tags": ["churn", "prediction", "customer-success"],
        "release_tag": "churn/v0.1.0",
    },
    {
        "id": "pipedream-router",
        "model_id": "model-pipedream",
        "name": "Pipedream Action Router",
        "description": "Routes NL to 3,000+ API actions across the Pipedream registry. Sub-millisecond, zero LLM calls.",
        "category": "routing",
        "icon": "\u26a1",
        "version": "0.7.2",
        "author": "Glyphh AI",
        "tags": ["pipedream", "api-routing", "automation"],
        "release_tag": "pipedream/v0.7.2",
    },
    {
        "id": "iris",
        "model_id": "model-iris",
        "name": "Glyphh Iris",
        "description": "Structured visual encoder — decomposes images into searchable glyphs. Face, pose, depth, OCR, objects.",
        "category": "vision",
        "icon": "\U0001f440",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "tags": ["vision", "image-search", "feature-extraction"],
        "release_tag": "iris/v0.1.0",
    },
    {
        "id": "marketing",
        "model_id": "model-marketing",
        "name": "Marketing Intelligence",
        "description": "Encodes campaigns, posts, and audiences as searchable HDC glyphs with performance vectors.",
        "category": "matching",
        "icon": "\U0001f4ca",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "tags": ["marketing", "campaigns", "analytics"],
        "release_tag": "marketing/v0.1.0",
    },
    {
        "id": "deals",
        "model_id": "model-deals",
        "name": "Deal Intelligence",
        "description": "Encodes sales deals as HDC glyphs with pipeline metrics. CRM sync via webhooks, MCP-native.",
        "category": "prediction",
        "icon": "\U0001f91d",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "tags": ["deals", "sales", "pipeline", "crm"],
        "release_tag": "deals/v0.1.0",
    },
    {
        "id": "code",
        "model_id": "model-code",
        "name": "Code Intelligence",
        "description": "File-level codebase search and drift scoring. MCP-native, sub-millisecond, zero LLM calls.",
        "category": "search",
        "icon": "\U0001f50d",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "tags": ["code-search", "drift-scoring", "mcp"],
        "release_tag": "code/v0.1.0",
        "has_commands": True,
    },
    {
        "id": "firewall",
        "model_id": "model-firewall",
        "name": "Prompt Injection Firewall",
        "description": "Detects prompt injection attacks in microseconds. 4-layer analysis, 6 attack families, full explainability.",
        "category": "security",
        "icon": "\U0001f512",
        "version": "0.9.0",
        "author": "Glyphh AI",
        "tags": ["prompt-injection", "firewall", "security", "guardrails"],
        "release_tag": "firewall/v0.9.0",
    },
    {
        "id": "voice",
        "model_id": "model-voice",
        "name": "Glyphh Voice",
        "description": "Voice identity and cognitive/emotional state via HDC + openSMILE. Speaker recognition, liveness, emotion.",
        "category": "identity",
        "icon": "\U0001f3a4",
        "version": "0.12.0",
        "author": "Glyphh AI",
        "tags": ["voice", "identity", "liveness", "biometrics"],
        "release_tag": "voice/v0.12.0",
    },
    {
        "id": "sentinel",
        "model_id": "model-sentinel",
        "name": "Sentinel Security",
        "description": "MITRE ATT&CK kill chain detection via HDC + Ada dreaming. Correlates security events no SIEM can find.",
        "category": "security",
        "icon": "\U0001f6a8",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "tags": ["security", "mitre", "kill-chain", "siem"],
        "release_tag": "sentinel/v0.1.0",
    },
]

PAGE_SIZE = 15


# ── Table rendering ────────────────────────────────────────────────────────

def _render_table(models: list[dict], page: int, total_pages: int, start_num: int):
    """Render a page of models as a table."""
    click.echo()
    header = f"  {'#':<4} {'MODEL':<24} {'VER':<8} DESCRIPTION"
    click.secho(header, fg=theme.TEXT_DIM)
    click.secho("  " + "\u2500" * 90, fg=theme.TEXT_DIM)

    for i, m in enumerate(models):
        num = start_num + i
        name = m["name"][:22]
        ver = f"v{m['version']}"[:7]
        # Truncate description at word boundary
        desc = m["description"]
        max_desc = 54
        if len(desc) > max_desc:
            cut = desc[:max_desc].rsplit(" ", 1)[0]
            desc = cut + "\u2026"
        click.echo(
            click.style(f"  {num:<4} ", fg=theme.MUTED)
            + click.style(f"{name:<24} ", fg=theme.TEXT)
            + click.style(f"{ver:<8} ", fg=theme.MUTED)
            + click.style(desc, fg=theme.TEXT_DIM)
        )

    click.echo()
    footer_parts = []
    if page > 1:
        footer_parts.append(click.style("back", fg=theme.ACCENT))
    if page < total_pages:
        footer_parts.append(click.style("next", fg=theme.ACCENT))
    footer_parts.append(click.style("<#>", fg=theme.ACCENT) + " for details")
    footer_parts.append(click.style("install <#>", fg=theme.ACCENT) + " to deploy")
    footer_parts.append(click.style("q", fg=theme.MUTED) + " to exit")
    click.echo("  " + "  \u00b7  ".join(footer_parts))
    click.secho(f"  Page {page}/{total_pages}", fg=theme.TEXT_DIM)
    click.echo()


def _render_detail(model: dict):
    """Render full model detail view."""
    click.echo()
    click.secho(f"  {model['icon']}  {model['name']}", fg=theme.TEXT, bold=True)
    click.secho(f"  {model['author']}  \u00b7  v{model['version']}", fg=theme.MUTED)
    click.echo()
    click.secho(f"  {model['description']}", fg=theme.TEXT_DIM)
    click.echo()

    tag_line = "  ".join(click.style(t, fg=theme.INFO) for t in model["tags"])
    click.echo(f"  {tag_line}")
    click.echo()

    click.secho("  Press Enter to go back.", fg=theme.MUTED)


# ── Interactive hub loop ────────────────────────────────────────────────────

def _hub_browse(models: list[dict]):
    """Interactive paginated table browser."""
    total_pages = max(1, (len(models) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = 1

    while True:
        start = (page - 1) * PAGE_SIZE
        page_models = models[start : start + PAGE_SIZE]
        start_num = start + 1

        _render_table(page_models, page, total_pages, start_num)

        try:
            resp = input(click.style("  hub> ", fg=theme.PRIMARY)).strip().lower()
        except (KeyboardInterrupt, EOFError):
            click.echo()
            return

        if resp in ("q", "quit", "exit", ""):
            return

        if resp in ("n", "next"):
            if page < total_pages:
                page += 1
            else:
                click.secho("  Already on last page.", fg=theme.MUTED)
            continue

        if resp in ("b", "back", "p", "prev", "previous"):
            if page > 1:
                page -= 1
            else:
                click.secho("  Already on first page.", fg=theme.MUTED)
            continue

        # install <name|number>
        if resp.startswith("install "):
            identifier = resp[8:].strip()
            if identifier:
                model = _resolve_model(identifier, models)
                if model:
                    _install_model(model)
                else:
                    click.secho(f"  Unknown model: {identifier}", fg=theme.WARNING)
            else:
                click.secho("  Usage: install <number or model-id>", fg=theme.MUTED)
            continue

        # Number selection — show detail
        try:
            num = int(resp)
            if 1 <= num <= len(models):
                _render_detail(models[num - 1])
                try:
                    input()
                except (KeyboardInterrupt, EOFError):
                    click.echo()
                    return
                continue
            else:
                click.secho(f"  Number out of range (1-{len(models)}).", fg=theme.WARNING)
        except ValueError:
            click.secho(f"  Unknown: {resp}  (try a number, next, back, install, or q)", fg=theme.MUTED)


# ── Download + install ─────────────────────────────────────────────────────

def _download_release(model: dict) -> Path | None:
    """Download .glyphh artifact from GitHub Releases.

    Tries the release API first. Falls back to constructing the direct
    download URL from the tag. Returns path to downloaded file or None.
    """
    import httpx

    release_tag = model.get("release_tag")
    model_id = model.get("model_id", model["id"])
    asset_name = f"{model_id}.glyphh"

    cache_dir = Path.home() / ".glyphh" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / asset_name

    if not release_tag:
        return None

    # Try GitHub Releases API to find the asset download URL
    tag_url = f"{_RELEASES_API}/tags/{release_tag}"
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            res = client.get(tag_url)
            if res.status_code == 200:
                for asset in res.json().get("assets", []):
                    if asset["name"] == asset_name:
                        download_url = asset["browser_download_url"]
                        click.secho(f"         Downloading {asset_name}...", fg=theme.TEXT_DIM)
                        r = client.get(download_url)
                        if r.status_code == 200:
                            dest.write_bytes(r.content)
                            size_kb = dest.stat().st_size / 1024
                            click.secho(f"         {size_kb:.0f} KB", fg=theme.TEXT_DIM)
                            return dest
    except Exception:
        pass

    return None


def _install_model(model: dict):
    """Download a .glyphh release artifact and deploy to the runtime."""
    import time

    click.echo()
    click.secho(f"  Installing {model['icon']}  {model['name']}...", fg=theme.TEXT)
    click.echo()

    model_id = model.get("model_id", model["id"])

    # Step 1: Download release artifact
    click.secho("  [1/2] Downloading...", fg=theme.MUTED)
    glyphh_file = _download_release(model)

    if not glyphh_file:
        click.secho("  No release found. Model may not be published yet.", fg=theme.WARNING)
        click.secho(f"  Expected release tag: {model.get('release_tag', '?')}", fg=theme.TEXT_DIM)
        return

    # Step 2: Deploy
    click.secho("  [2/2] Deploying...", fg=theme.MUTED)
    try:
        from glyphh.cli.auth import is_logged_in, get_org_id
        from glyphh.cli.config import resolve_runtime_url, resolve_runtime_token
        import httpx

        if not is_logged_in():
            click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
            return

        runtime_url = resolve_runtime_url()
        org_id = get_org_id()
        token = resolve_runtime_token()

        if not org_id:
            click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
            return

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        with httpx.Client(timeout=60) as client:
            with open(glyphh_file, "rb") as f:
                r = client.post(
                    f"{runtime_url}/{org_id}/{model_id}/model/deploy",
                    files={"file": (glyphh_file.name, f, "application/octet-stream")},
                    headers=headers,
                )

        if r.status_code not in (200, 201):
            detail = ""
            try:
                detail = r.json().get("detail", "") or r.json().get("error", {}).get("message", "")
            except Exception:
                pass
            click.secho(f"  Deploy failed ({r.status_code}): {detail}", fg=theme.ERROR)
            return

        # Wait for exemplar encoding to finish
        click.secho("         Deployed, loading exemplars...", fg=theme.TEXT_DIM)
        deadline = time.time() + 120
        glyph_count = 0
        while time.time() < deadline:
            time.sleep(1)
            try:
                with httpx.Client(timeout=5) as client:
                    res = client.get(
                        f"{runtime_url}/{org_id}/models",
                        headers=headers,
                    )
                if res.status_code == 200:
                    for m in res.json().get("models", []):
                        if m.get("model_id") == model_id:
                            status = m.get("status", "")
                            glyph_count = m.get("glyphs", 0)
                            if status != "encoding":
                                break
                    else:
                        continue
                    break
            except Exception:
                pass

        click.echo()
        dot = click.style("\u25cf", fg=theme.SUCCESS)
        click.echo(f"  {dot} {click.style('installed', fg=theme.SUCCESS)}")
        click.echo()
        click.secho(f"  Model:    {model_id}", fg=theme.TEXT_DIM)
        click.secho(f"  Version:  v{model.get('version', '?')}", fg=theme.TEXT_DIM)
        click.secho(f"  Glyphs:   {glyph_count}", fg=theme.TEXT_DIM)
        click.echo()

        if model.get("has_commands"):
            click.secho(f"  This model has custom commands. Enter the model REPL:", fg=theme.MUTED)
            click.secho(f"    model {model_id}", fg=theme.TEXT_DIM)
            click.echo()

    except ImportError:
        click.secho("  Runtime not installed. Run: pip install glyphh", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Deploy failed: {e}", fg=theme.ERROR)
    finally:
        # Clean up cached download
        try:
            if glyphh_file and glyphh_file.exists():
                glyphh_file.unlink()
        except Exception:
            pass


# ── Command handlers ────────────────────────────────────────────────────────

def _cmd_list(args: str):
    """hub list — browse all models."""
    _hub_browse(MODELS)


def _cmd_search(args: str):
    """hub search <query> — filter models by name/tag/category."""
    query = args.strip().lower()
    if not query:
        click.secho("  Usage: hub search <query>", fg=theme.MUTED)
        return

    filtered = [
        m for m in MODELS
        if query in m["name"].lower()
        or query in m["description"].lower()
        or query in m["category"].lower()
        or any(query in t for t in m["tags"])
    ]

    if not filtered:
        click.secho(f"  No models match \"{args.strip()}\".", fg=theme.TEXT_DIM)
        return

    click.secho(f"  {len(filtered)} model(s) matching \"{args.strip()}\"", fg=theme.TEXT_DIM)
    _hub_browse(filtered)


def _resolve_model(identifier: str, models: list[dict] | None = None) -> dict | None:
    """Resolve a model by number or id from a model list."""
    models = models or MODELS
    # Try number first
    try:
        num = int(identifier)
        if 1 <= num <= len(models):
            return models[num - 1]
    except ValueError:
        pass
    # Then by id
    return next((m for m in models if m["id"] == identifier), None)


def _cmd_install(args: str):
    """hub install <id|number> — deploy a model from the registry."""
    identifier = args.strip().lower()
    if not identifier:
        click.secho("  Usage: hub install <model-id or number>", fg=theme.MUTED)
        return

    model = _resolve_model(identifier)
    if not model:
        click.secho(f"  Unknown model: {identifier}", fg=theme.WARNING)
        click.secho("  Run 'hub list' to see available models.", fg=theme.MUTED)
        return

    _install_model(model)


def handle_hub(func: str | None, args: str = ""):
    """Route hub subcommands."""
    commands = {
        "list": _cmd_list,
        "search": _cmd_search,
        "install": _cmd_install,
    }

    if func is None:
        _cmd_list(args)
        return

    handler = commands.get(func)
    if handler:
        handler(args)
    else:
        click.secho(f"  Unknown: hub {func}", fg=theme.WARNING)
        click.secho("  Available: list, search, install", fg=theme.TEXT_DIM)
