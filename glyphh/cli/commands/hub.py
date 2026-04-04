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


# ── Model registry (hardcoded — will switch to Platform API) ────────────────

MODELS = [
    {
        "id": "toolrouter",
        "name": "SaaS Tool Router",
        "description": "Routes NL SaaS requests to tool functions across 8 domains. 38 tools, sub-10ms, zero tokens.",
        "category": "routing",
        "icon": "\U0001f500",
        "version": "2.2.0",
        "author": "Glyphh AI",
        "license": "MIT",
        "tags": ["tool-routing", "saas", "intent-matching"],
        "repo": "https://github.com/glyphh-ai/model-toolrouter",
    },
    {
        "id": "faq-helpdesk",
        "name": "Glyphh AI Knowledge Base",
        "description": "FAQ model covering all aspects of the Glyphh platform. 156 entries across 11 categories.",
        "category": "faq",
        "icon": "\U0001f4ac",
        "version": "0.2.0",
        "author": "Glyphh AI",
        "license": "MIT",
        "tags": ["faq", "knowledge-base", "documentation"],
        "repo": "https://github.com/glyphh-ai/model-faq",
    },
    {
        "id": "customer-churn",
        "name": "Customer Churn Predictor",
        "description": "Encodes customer usage metrics into HDC vectors to identify churn risk patterns via similarity.",
        "category": "prediction",
        "icon": "\U0001f4c9",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "license": "MIT",
        "tags": ["churn", "prediction", "customer-success"],
        "repo": "https://github.com/glyphh-ai/model-churn",
    },
    {
        "id": "pipedream-router",
        "name": "Pipedream Action Router",
        "description": "Routes NL to 3,000+ API actions across the Pipedream registry. Sub-millisecond, zero LLM calls.",
        "category": "routing",
        "icon": "\u26a1",
        "version": "0.7.2",
        "author": "Glyphh AI",
        "license": "MIT",
        "tags": ["pipedream", "api-routing", "automation"],
        "repo": "https://github.com/glyphh-ai/model-pipedream",
    },
    {
        "id": "iris",
        "name": "Glyphh Iris",
        "description": "Structured visual encoder — decomposes images into searchable glyphs. Face, pose, depth, OCR, objects.",
        "category": "vision",
        "icon": "\U0001f440",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "license": "MIT",
        "tags": ["vision", "image-search", "feature-extraction"],
        "repo": "https://github.com/glyphh-ai/model-iris",
    },
    {
        "id": "marketing",
        "name": "Marketing Intelligence",
        "description": "Encodes campaigns, posts, and audiences as searchable HDC glyphs with performance vectors.",
        "category": "matching",
        "icon": "\U0001f4ca",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "license": "MIT",
        "tags": ["marketing", "campaigns", "analytics"],
        "repo": "https://github.com/glyphh-ai/model-marketing",
    },
    {
        "id": "deals",
        "name": "Deal Intelligence",
        "description": "Encodes sales deals as HDC glyphs with pipeline metrics. CRM sync via webhooks, MCP-native.",
        "category": "prediction",
        "icon": "\U0001f91d",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "license": "MIT",
        "tags": ["deals", "sales", "pipeline", "crm"],
        "repo": "https://github.com/glyphh-ai/model-deals",
    },
    {
        "id": "code",
        "name": "Code Intelligence",
        "description": "File-level codebase search and drift scoring. MCP-native, sub-millisecond, zero LLM calls.",
        "category": "search",
        "icon": "\U0001f50d",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "license": "MIT",
        "tags": ["code-search", "drift-scoring", "mcp"],
        "repo": "https://github.com/glyphh-ai/glyphh-code",
    },
    {
        "id": "firewall",
        "name": "Prompt Injection Firewall",
        "description": "Detects prompt injection attacks in microseconds. 4-layer analysis, 6 attack families, full explainability.",
        "category": "security",
        "icon": "\U0001f512",
        "version": "0.9.0",
        "author": "Glyphh AI",
        "license": "AGPL-3.0",
        "tags": ["prompt-injection", "firewall", "security", "guardrails"],
        "repo": "https://github.com/glyphh-ai/model-firewall",
    },
    {
        "id": "voice",
        "name": "Glyphh Voice",
        "description": "Voice identity and cognitive/emotional state via HDC + openSMILE. Speaker recognition, liveness, emotion.",
        "category": "identity",
        "icon": "\U0001f3a4",
        "version": "0.12.0",
        "author": "Glyphh AI",
        "license": "AGPL-3.0",
        "tags": ["voice", "identity", "liveness", "biometrics"],
        "repo": "https://github.com/glyphh-ai/model-voice",
    },
    {
        "id": "sentinel",
        "name": "Sentinel Security",
        "description": "MITRE ATT&CK kill chain detection via HDC + Ada dreaming. Correlates security events no SIEM can find.",
        "category": "security",
        "icon": "\U0001f6a8",
        "version": "0.1.0",
        "author": "Glyphh AI",
        "license": "AGPL-3.0",
        "tags": ["security", "mitre", "kill-chain", "siem"],
        "repo": "https://github.com/glyphh-ai/model-sentinel",
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
    click.secho(f"  {model['author']}  \u00b7  v{model['version']}  \u00b7  {model['license']}", fg=theme.MUTED)
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


def _install_model(model: dict):
    """Clone a model from GitHub, package it, and deploy to the runtime."""
    import shutil
    import subprocess
    import time

    click.echo()
    click.secho(f"  Installing {model['icon']}  {model['name']}...", fg=theme.TEXT)
    click.echo()

    # Persistent model source dir: ~/.glyphh/models/<model-id>/
    models_dir = Path.home() / ".glyphh" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model_id early from registry
    registry_id = model["id"]

    # Step 1: Clone
    click.secho("  [1/4] Cloning...", fg=theme.MUTED)
    clone_dir = models_dir / registry_id

    # If already cloned, pull instead
    if clone_dir.exists() and (clone_dir / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(clone_dir), "pull", "--ff-only"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            click.secho(f"         Updated existing clone", fg=theme.TEXT_DIM)
        else:
            # Fresh clone
            shutil.rmtree(clone_dir, ignore_errors=True)
            clone_dir.mkdir(parents=True, exist_ok=True)
    else:
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)

    if not (clone_dir / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", f"{model['repo']}.git", str(clone_dir)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                click.secho(f"  Clone failed: {result.stderr.strip()[:120]}", fg=theme.ERROR)
                return
            click.secho(f"         Cloned to ~/.glyphh/models/{registry_id}/", fg=theme.TEXT_DIM)
        except FileNotFoundError:
            click.secho("  git not found. Install git and try again.", fg=theme.ERROR)
            return
        except subprocess.TimeoutExpired:
            click.secho("  Clone timed out.", fg=theme.ERROR)
            return

    # Step 2: Build (if build.py exists)
    click.secho("  [2/4] Building...", fg=theme.MUTED)
    build_script = clone_dir / "build.py"
    if build_script.exists():
        import sys
        result = subprocess.run(
            [sys.executable, str(build_script)],
            capture_output=True, text=True, timeout=300,
            cwd=str(clone_dir),
        )
        if result.returncode != 0:
            click.secho(f"  Build failed: {result.stderr.strip()[:200]}", fg=theme.ERROR)
            return
        click.secho(f"         Build complete", fg=theme.TEXT_DIM)
    else:
        click.secho(f"         No build.py, skipping", fg=theme.TEXT_DIM)

    # Step 3: Package
    click.secho("  [3/4] Packaging...", fg=theme.MUTED)
    try:
        from glyphh.cli.packaging import package_model, read_manifest, is_model_dir

        if not is_model_dir(clone_dir):
            click.secho("  Not a valid model (no manifest.yaml).", fg=theme.ERROR)
            return

        manifest = read_manifest(clone_dir)
        model_id = manifest.get("model_id", registry_id)
        glyphh_file = package_model(clone_dir)
        click.secho(f"         {glyphh_file.name}", fg=theme.TEXT_DIM)
    except Exception as e:
        click.secho(f"  Package failed: {e}", fg=theme.ERROR)
        return

    # Step 4: Deploy
    click.secho("  [4/4] Deploying...", fg=theme.MUTED)
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
        click.secho(f"  Source:   ~/.glyphh/models/{registry_id}/", fg=theme.TEXT_DIM)
        click.echo()

    except ImportError:
        click.secho("  Runtime not installed. Run: pip install glyphh[runtime]", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Deploy failed: {e}", fg=theme.ERROR)

    # Clean up packaged file (source dir is kept)
    try:
        if glyphh_file.exists():
            glyphh_file.unlink()
    except Exception:
        pass


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
