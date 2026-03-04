"""
CLI model commands — local model management.

model list       List local models found on disk
model deploy     Deploy a model to the runtime
model status     Check deployed model status
model undeploy   Remove a model from the runtime
model init       Scaffold a new model project
model package    Package a model directory into a .glyphh file
"""

import os
import click
from pathlib import Path

from .. import theme
from ..auth import get_token, get_api_url, is_logged_in
from ..config import resolve_runtime_url, resolve_runtime_token
from ..packaging import (
    discover_local_models,
    find_model_dir,
    is_model_dir,
    package_model,
    read_manifest,
    unpack_model,
)


@click.group("model")
def model_group():
    """Local model management."""
    pass


@model_group.command("list")
def model_list():
    """List local models found on disk."""
    models = discover_local_models()
    if not models:
        click.secho("  No models found in current directory.", fg=theme.MUTED)
        click.secho("  Try: catalog download <model-name>", fg=theme.TEXT_DIM)
        return

    # Table header
    click.echo()
    header = f"  {'NAME':<24} {'VERSION':<10} {'CATEGORY':<14} {'GLYPHS':<8} {'TYPE':<6} PATH"
    click.secho(header, fg=theme.TEXT_DIM)
    click.secho("  " + "─" * 90, fg=theme.TEXT_DIM)

    for m in models:
        name = m["name"][:22]
        ver = m["version"][:8] or "—"
        cat = m["category"][:12] or "—"
        glyphs = str(m["glyphs"]) if m["glyphs"] > 0 else "—"
        mtype = m["type"]
        path = m["path"]

        click.echo(
            click.style(f"  {name:<24} ", fg=theme.TEXT)
            + click.style(f"{ver:<10} ", fg=theme.MUTED)
            + click.style(f"{cat:<14} ", fg=theme.INFO)
            + click.style(f"{glyphs:<8} ", fg=theme.ACCENT)
            + click.style(f"{mtype:<6} ", fg=theme.TEXT_DIM)
            + click.style(path, fg=theme.TEXT_DIM)
        )
    click.echo()


@model_group.command("deploy")
@click.argument("path", default=".", type=click.Path(exists=True))
def model_deploy(path):
    """Deploy a model to the runtime.

    PATH can be a model directory or a .glyphh file.
    Defaults to current directory.
    """
    if not is_logged_in():
        click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
        return

    target = Path(path).resolve()

    # Determine what we're deploying
    if target.is_file() and target.suffix == ".glyphh":
        glyphh_file = target
        manifest = {}  # will be read from the file by the runtime
        click.secho(f"  Deploying {target.name}...", fg=theme.MUTED)
    elif target.is_dir() and is_model_dir(target):
        manifest = read_manifest(target)
        click.secho(f"  Packaging {manifest.get('name', target.name)}...", fg=theme.MUTED)
        glyphh_file = package_model(target)
        click.secho(f"  Deploying {glyphh_file.name}...", fg=theme.MUTED)
    else:
        click.secho("  Not a model directory (no manifest.yaml) or .glyphh file.", fg=theme.ERROR)
        return

    # Resolve model_id from manifest or filename
    model_id = manifest.get("model_id") or manifest.get("name") or glyphh_file.stem
    model_id = model_id.replace(".glyphh", "")

    # Upload to runtime (local or remote)
    runtime_url = resolve_runtime_url()
    token = resolve_runtime_token()

    # For local runtimes, use local-dev-org (matches runtime's local auth bypass)
    from urllib.parse import urlparse
    parsed = urlparse(runtime_url)
    is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    org_id = "local-dev-org" if is_local else (manifest.get("org_id") or "default")

    headers = {}
    if token and not is_local:
        headers["Authorization"] = f"Bearer {token}"

    deploy_url = f"{runtime_url}/{org_id}/{model_id}/model/deploy"

    try:
        import httpx

        with httpx.Client(timeout=60) as client:
            with open(glyphh_file, "rb") as f:
                res = client.post(
                    deploy_url,
                    files={"file": (glyphh_file.name, f, "application/octet-stream")},
                    headers=headers,
                )

            if res.status_code in (200, 201):
                data = res.json()
                click.echo()
                click.secho(f"  ✓ Deployed: {data.get('model_id', model_id)}", fg=theme.SUCCESS)
                if data.get("version"):
                    click.secho(f"    Version: {data['version']}", fg=theme.MUTED)
                click.echo()
            else:
                detail = res.text
                try:
                    body = res.json()
                    detail = body.get("detail") or body.get("error", {}).get("message", detail)
                except Exception:
                    pass
                click.secho(f"  Deploy failed: {detail}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {runtime_url}", fg=theme.ERROR)
        click.secho(f"  Is the runtime running? Try: RUNTIME_URL={runtime_url}", fg=theme.MUTED)
    except Exception as e:
        click.secho(f"  Deploy failed: {e}", fg=theme.ERROR)

    # Clean up packaged file if we created it from a directory
    if target.is_dir() and glyphh_file.exists() and glyphh_file != target:
        try:
            glyphh_file.unlink()
        except Exception:
            pass


@model_group.command("status")
@click.argument("model_id", required=False)
def model_status(model_id):
    """Check deployed model status.

    If MODEL_ID is omitted and you're in a model directory, uses that model.
    """
    if not is_logged_in():
        click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
        return

    # Resolve model_id from cwd if not provided
    if not model_id:
        model_dir = find_model_dir()
        if model_dir:
            manifest = read_manifest(model_dir)
            model_id = manifest.get("model_id", model_dir.name)
        else:
            click.secho("  Provide a model_id or run from a model directory.", fg=theme.MUTED)
            return

    runtime_url = resolve_runtime_url()
    token = resolve_runtime_token()

    from urllib.parse import urlparse
    parsed = urlparse(runtime_url)
    is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    org_id = "local-dev-org" if is_local else "default"

    headers = {}
    if token and not is_local:
        headers["Authorization"] = f"Bearer {token}"

    try:
        import httpx

        with httpx.Client(timeout=15) as client:
            res = client.get(
                f"{runtime_url}/{org_id}/{model_id}/ready",
                headers=headers,
            )

        if res.status_code == 200:
            data = res.json()
            click.echo()
            click.secho(f"  {data.get('meta_name', model_id)}", fg=theme.TEXT)
            click.secho(f"  Status: {data.get('status', 'unknown')}", fg=theme.INFO)
            click.secho(f"  Ready: {data.get('ready', False)}", fg=theme.SUCCESS if data.get('ready') else theme.WARNING)
            click.echo()
        elif res.status_code == 404:
            click.secho(f"  Model '{model_id}' not found.", fg=theme.WARNING)
        else:
            click.secho(f"  Error: {res.text}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {runtime_url}", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Could not check status: {e}", fg=theme.ERROR)


@model_group.command("undeploy")
@click.argument("model_id", required=False)
def model_undeploy(model_id):
    """Remove a model from the runtime.

    If MODEL_ID is omitted and you're in a model directory, uses that model.
    """
    if not is_logged_in():
        click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
        return

    if not model_id:
        model_dir = find_model_dir()
        if model_dir:
            manifest = read_manifest(model_dir)
            model_id = manifest.get("model_id", model_dir.name)
        else:
            click.secho("  Provide a model_id or run from a model directory.", fg=theme.MUTED)
            return

    runtime_url = resolve_runtime_url()
    token = resolve_runtime_token()

    from urllib.parse import urlparse
    parsed = urlparse(runtime_url)
    is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    org_id = "local-dev-org" if is_local else "default"

    headers = {}
    if token and not is_local:
        headers["Authorization"] = f"Bearer {token}"

    try:
        import httpx

        click.secho(f"  Undeploying {model_id}...", fg=theme.MUTED)
        with httpx.Client(timeout=30) as client:
            res = client.post(
                f"{runtime_url}/{org_id}/{model_id}/model/undeploy",
                headers=headers,
            )

        if res.status_code == 200:
            click.secho(f"  ✓ Undeployed: {model_id}", fg=theme.SUCCESS)
        else:
            detail = res.text
            try:
                detail = res.json().get("detail", detail)
            except Exception:
                pass
            click.secho(f"  Undeploy failed: {detail}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {runtime_url}", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Undeploy failed: {e}", fg=theme.ERROR)


@model_group.command("init")
@click.argument("name", required=False)
def model_init(name):
    """Scaffold a new model project.

    Creates a model directory with manifest.yaml and data/ folder.
    """
    import uuid

    if not name:
        name = Path.cwd().name

    model_dir = Path.cwd() / name if name != Path.cwd().name else Path.cwd()

    if model_dir != Path.cwd():
        model_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = model_dir / "manifest.yaml"
    if manifest_path.exists():
        click.secho(f"  manifest.yaml already exists in {model_dir.name}/", fg=theme.WARNING)
        return

    model_id = name.lower().replace(" ", "-")

    manifest = {
        "model_id": model_id,
        "name": name,
        "description": "",
        "version": "0.1.0",
        "author": "",
        "category": "",
        "load_on_startup": False,
    }

    import yaml
    manifest_path.write_text(yaml.dump(manifest, default_flow_style=False, sort_keys=False))
    (model_dir / "data").mkdir(exist_ok=True)

    click.echo()
    click.secho(f"  ✓ Created model project: {model_dir.name}/", fg=theme.SUCCESS)
    click.secho(f"    manifest.yaml", fg=theme.MUTED)
    click.secho(f"    data/", fg=theme.MUTED)
    click.echo()


@model_group.command("package")
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output .glyphh file path")
def model_package(path, output):
    """Package a model directory into a .glyphh file."""
    target = Path(path).resolve()

    if not is_model_dir(target):
        click.secho("  Not a model directory (no manifest.yaml).", fg=theme.ERROR)
        return

    manifest = read_manifest(target)
    out_path = Path(output) if output else None

    click.secho(f"  Packaging {manifest.get('name', target.name)}...", fg=theme.MUTED)
    result = package_model(target, out_path)
    click.secho(f"  ✓ {result.name} ({result.stat().st_size / 1024:.1f} KB)", fg=theme.SUCCESS)


@model_group.command("load")
@click.argument("file", type=click.Path(exists=True))
@click.option("--model-id", "-m", default=None, help="Model ID (defaults to manifest model_id)")
@click.option("--batch-size", "-b", default=50, type=int, help="Records per batch (default: 50)")
def model_load(file, model_id, batch_size):
    """Load data into a deployed model from a concepts.json file.

    The file should contain a JSON array of records matching your model's
    encoder config. Example concepts.json:

    \b
    [
      {"question": "how do I reset my password", "answer": "Go to Settings > Security."},
      {"question": "where are my invoices", "answer": "Go to Billing > Invoice History."}
    ]

    The CLI resolves org_id from your login session and model_id from the
    manifest.yaml in the current directory (or use --model-id).
    """
    import json as json_mod

    if not is_logged_in():
        click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
        return

    # Load the concepts file
    target = Path(file).resolve()
    try:
        raw = target.read_text()
        data = json_mod.loads(raw)
    except json_mod.JSONDecodeError as e:
        click.secho(f"  Invalid JSON: {e}", fg=theme.ERROR)
        return
    except Exception as e:
        click.secho(f"  Could not read file: {e}", fg=theme.ERROR)
        return

    # Accept either a bare array or {"records": [...]}
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "records" in data:
        records = data["records"]
    else:
        click.secho("  Expected a JSON array of records, or {\"records\": [...]}.", fg=theme.ERROR)
        return

    if not records:
        click.secho("  No records found in file.", fg=theme.WARNING)
        return

    # Resolve model_id from manifest if not provided
    if not model_id:
        model_dir = find_model_dir()
        if model_dir:
            manifest = read_manifest(model_dir)
            model_id = manifest.get("model_id", model_dir.name)
        else:
            click.secho("  No manifest.yaml found. Provide --model-id or run from a model directory.", fg=theme.ERROR)
            return

    # Resolve org_id from session
    from ..auth import _load_config
    config = _load_config()
    user = config.get("user", {})
    org_id = user.get("org_id")

    runtime_url = resolve_runtime_url()
    from urllib.parse import urlparse
    parsed = urlparse(runtime_url)
    is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1")

    if is_local and not org_id:
        org_id = "local-dev-org"
    elif not org_id:
        click.secho("  No org_id in session. Log in first: glyphh auth login", fg=theme.ERROR)
        return

    token = resolve_runtime_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    load_url = f"{runtime_url}/{org_id}/{model_id}/listener"

    click.secho(f"  Loading {len(records)} records into {model_id}...", fg=theme.MUTED)

    try:
        import httpx

        with httpx.Client(timeout=120) as client:
            res = client.post(
                load_url,
                json={"records": records, "batch_size": batch_size},
                headers=headers,
            )

        if res.status_code in (200, 201):
            body = res.json()
            job_id = body.get("job_id", "—")
            total = body.get("total_records", len(records))
            click.echo()
            click.secho(f"  ✓ Data load started", fg=theme.SUCCESS)
            click.secho(f"    Job:     {job_id}", fg=theme.MUTED)
            click.secho(f"    Records: {total}", fg=theme.MUTED)
            click.secho(f"    Status:  {body.get('status', 'queued')}", fg=theme.MUTED)
            click.echo()
        else:
            detail = res.text
            try:
                detail = res.json().get("detail", detail)
            except Exception:
                pass
            click.secho(f"  Load failed: {detail}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {runtime_url}", fg=theme.ERROR)
        click.secho(f"  Is the runtime running?", fg=theme.MUTED)
    except Exception as e:
        click.secho(f"  Load failed: {e}", fg=theme.ERROR)


def _resolve_context():
    """Resolve org_id, model_id, runtime_url, and auth headers for admin data commands.

    Uses the session token from `glyphh auth login`. In local mode the
    runtime allows unauthenticated access on lifecycle/admin endpoints,
    so a missing token is not fatal.
    """
    from ..auth import _load_config

    if not is_logged_in():
        click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
        return None

    runtime_url = resolve_runtime_url()
    token = resolve_runtime_token()

    config = _load_config()
    user = config.get("user", {})
    org_id = user.get("org_id")

    from urllib.parse import urlparse
    parsed = urlparse(runtime_url)
    is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1")

    if is_local and not org_id:
        org_id = "local-dev-org"
    elif not org_id:
        click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
        return None

    headers = {}
    if token and not is_local:
        headers["Authorization"] = f"Bearer {token}"

    model_dir = find_model_dir()
    if model_dir:
        manifest = read_manifest(model_dir)
        model_id = manifest.get("model_id", model_dir.name)
    else:
        model_id = None

    return {
        "runtime_url": runtime_url,
        "token": token,
        "org_id": org_id,
        "model_id": model_id,
        "headers": headers,
    }


@model_group.command("data")
@click.option("--model-id", "-m", default=None, help="Model ID (defaults to manifest)")
@click.option("--limit", "-l", default=20, type=int, help="Number of records to show")
@click.option("--offset", default=0, type=int, help="Offset for pagination")
def model_data(model_id, limit, offset):
    """View glyphs stored in a deployed model."""
    ctx = _resolve_context()
    if not ctx:
        return
    mid = model_id or ctx["model_id"]
    if not mid:
        click.secho("  Provide --model-id or run from a model directory.", fg=theme.ERROR)
        return

    try:
        import httpx

        with httpx.Client(timeout=15) as client:
            res = client.get(
                f"{ctx['runtime_url']}/{ctx['org_id']}/{mid}/data",
                params={"limit": limit, "offset": offset},
                headers=ctx["headers"],
            )

        if res.status_code == 200:
            data = res.json()
            total = data.get("total", 0)
            glyphs = data.get("glyphs", [])

            click.echo()
            click.secho(f"  {mid}: {total} glyphs total (showing {offset+1}-{offset+len(glyphs)})", fg=theme.INFO)
            click.secho("  " + "─" * 70, fg=theme.TEXT_DIM)

            for g in glyphs:
                text = g.get("concept_text", "")[:80]
                meta = g.get("metadata", {})
                # Show a compact summary of metadata keys
                meta_keys = list(meta.keys())[:4]
                meta_str = ", ".join(f"{k}={str(meta[k])[:20]}" for k in meta_keys)
                click.echo(
                    click.style(f"  {str(g['id'])[:8]}  ", fg=theme.ACCENT)
                    + click.style(meta_str or text, fg=theme.TEXT)
                )

            click.echo()
            if total > offset + limit:
                click.secho(f"  More: glyphh model data --offset {offset + limit}", fg=theme.MUTED)
        else:
            click.secho(f"  Error: {res.text}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {ctx['runtime_url']}", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Failed: {e}", fg=theme.ERROR)


@model_group.command("count")
@click.option("--model-id", "-m", default=None, help="Model ID (defaults to manifest)")
def model_count(model_id):
    """Show glyph and vector counts for a deployed model."""
    ctx = _resolve_context()
    if not ctx:
        return
    mid = model_id or ctx["model_id"]
    if not mid:
        click.secho("  Provide --model-id or run from a model directory.", fg=theme.ERROR)
        return

    try:
        import httpx

        with httpx.Client(timeout=15) as client:
            res = client.get(
                f"{ctx['runtime_url']}/{ctx['org_id']}/{mid}/data/count",
                headers=ctx["headers"],
            )

        if res.status_code == 200:
            data = res.json()
            click.echo()
            click.secho(f"  {mid}", fg=theme.TEXT)
            click.secho(f"    Glyphs:  {data.get('glyphs', 0)}", fg=theme.INFO)
            click.secho(f"    Vectors: {data.get('vectors', 0)}", fg=theme.MUTED)
            click.echo()
        else:
            click.secho(f"  Error: {res.text}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {ctx['runtime_url']}", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Failed: {e}", fg=theme.ERROR)


@model_group.command("clear")
@click.option("--model-id", "-m", default=None, help="Model ID (defaults to manifest)")
@click.confirmation_option(prompt="  This will delete all glyphs and edges. Continue?")
def model_clear(model_id):
    """Clear all data from a deployed model (keeps model loaded)."""
    if not is_logged_in():
        click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
        return

    from ..auth import _load_config
    config = _load_config()
    user = config.get("user", {})
    org_id = user.get("org_id")
    if not org_id:
        click.secho("  No org_id in session.", fg=theme.ERROR)
        return

    mid = model_id
    if not mid:
        model_dir = find_model_dir()
        if model_dir:
            manifest = read_manifest(model_dir)
            mid = manifest.get("model_id", model_dir.name)
        else:
            click.secho("  Provide --model-id or run from a model directory.", fg=theme.ERROR)
            return

    runtime_url = resolve_runtime_url()
    from urllib.parse import urlparse
    parsed = urlparse(runtime_url)
    is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1")

    headers = {}
    token = resolve_runtime_token()
    if token and not is_local:
        headers["Authorization"] = f"Bearer {token}"

    try:
        import httpx

        with httpx.Client(timeout=30) as client:
            res = client.delete(
                f"{runtime_url}/{org_id}/{mid}/data",
                headers=headers,
            )

        if res.status_code == 200:
            data = res.json()
            click.echo()
            click.secho(f"  ✓ Cleared {mid}", fg=theme.SUCCESS)
            click.secho(f"    Glyphs deleted: {data.get('glyphs_deleted', 0)}", fg=theme.MUTED)
            click.secho(f"    Edges deleted:  {data.get('edges_deleted', 0)}", fg=theme.MUTED)
            click.echo()
        else:
            click.secho(f"  Error: {res.text}", fg=theme.ERROR)

    except httpx.ConnectError:
        click.secho(f"  Could not connect to runtime at {runtime_url}", fg=theme.ERROR)
    except Exception as e:
        click.secho(f"  Failed: {e}", fg=theme.ERROR)


@model_group.command("re-encode")
@click.option("--model-id", "-m", default=None, help="Model ID (defaults to manifest)")
def model_re_encode(model_id):
    """Re-encode all glyphs for a deployed model."""
    if not is_logged_in():
        click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
        return

    from ..auth import _load_config
    config = _load_config()
    user = config.get("user", {})
    org_id = user.get("org_id")
    if not org_id:
        click.secho("  No org_id in session.", fg=theme.ERROR)
        return

    mid = model_id
    if not mid:
        model_dir = find_model_dir()
        if model_dir:
            manifest = read_manifest(model_dir)
            mid = manifest.get("model_id", model_dir.name)
        else:
            click.secho("  Provide --model-id or run from a model directory.", fg=theme.ERROR)
            return

    runtime_url = resolve_runtime_url()
    from urllib.parse import urlparse
    parsed = urlparse(runtime_url)
    is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1")

    headers = {}
    token = resolve_runtime_token()
    if token and not is_local:
        headers["Authorization"] = f"Bearer {token}"

    click.secho(f"  Re-encoding {mid}...", fg=theme.MUTED)

    try:
        import httpx

        with httpx.Client(timeout=120) as client:
            res = client.post(
                f"{runtime_url}/{org_id}/{mid}/model/re-encode",
                headers=headers,
            )

        if res.status_code == 200:
            data = res.json()
            click.echo()
            click.secho(f"  ✓ Re-encode started", fg=theme.SUCCESS)
            if data.get("job_id"):
                click.secho(f"    Job: {data['job_id']}", fg=theme.MUTED)
            click.secho(f"    Status: {data.get('status', '—')}", fg=theme.MUTED)
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
    except Exception as e:
        click.secho(f"  Failed: {e}", fg=theme.ERROR)


@model_group.command("test")
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("-k", "--keyword", type=str, default=None, help="Filter tests by keyword")
def model_test(path, verbose, keyword):
    """Run a model's test suite.

    PATH can be a model directory or a .glyphh file.
    Defaults to current directory.

    Looks for tests.py or tests/ directory in the model.
    """
    import subprocess
    import sys

    target = Path(path).resolve()

    # If it's a .glyphh file, unpack it first
    if target.is_file() and target.suffix == ".glyphh":
        click.secho(f"  Unpacking {target.name}...", fg=theme.MUTED)
        try:
            target = unpack_model(target)
        except Exception as e:
            click.secho(f"  Failed to unpack: {e}", fg=theme.ERROR)
            return

    if not target.is_dir():
        click.secho("  Not a model directory.", fg=theme.ERROR)
        return

    # Find test entry point
    tests_py = target / "tests.py"
    tests_dir = target / "tests"

    if tests_py.exists():
        click.secho(f"  Running tests for {target.name}...", fg=theme.MUTED)
        click.echo()

        cmd = [sys.executable, str(tests_py)]
        if verbose:
            cmd.append("-v")
        if keyword:
            cmd.extend(["-k", keyword])

        result = subprocess.run(cmd, cwd=str(target))

        click.echo()
        if result.returncode == 0:
            click.secho("  ✓ All tests passed", fg=theme.SUCCESS)
        else:
            click.secho(f"  ✗ Tests failed (exit code {result.returncode})", fg=theme.ERROR)

    elif tests_dir.exists() and tests_dir.is_dir():
        click.secho(f"  Running tests for {target.name}...", fg=theme.MUTED)
        click.echo()

        cmd = [sys.executable, "-m", "pytest", str(tests_dir)]
        if verbose:
            cmd.append("-v")
        if keyword:
            cmd.extend(["-k", keyword])

        result = subprocess.run(cmd, cwd=str(target))

        click.echo()
        if result.returncode == 0:
            click.secho("  ✓ All tests passed", fg=theme.SUCCESS)
        else:
            click.secho(f"  ✗ Tests failed (exit code {result.returncode})", fg=theme.ERROR)

    else:
        click.secho("  No tests found (expected tests.py or tests/ directory).", fg=theme.WARNING)
        click.secho("  See: https://docs.glyphh.ai/models/testing", fg=theme.MUTED)


# ── Handler for interactive shell ──

def handle_model(func: str | None, args: str = ""):
    """Route model subcommands from the interactive shell."""
    if func == "list":
        model_list.invoke(click.Context(model_list))
    elif func == "deploy":
        ctx = click.Context(model_deploy)
        model_deploy.invoke(ctx, path=args.strip() or ".")
    elif func == "status":
        ctx = click.Context(model_status)
        model_status.invoke(ctx, model_id=args.strip() or None)
    elif func == "undeploy":
        ctx = click.Context(model_undeploy)
        model_undeploy.invoke(ctx, model_id=args.strip() or None)
    elif func == "init":
        ctx = click.Context(model_init)
        model_init.invoke(ctx, name=args.strip() or None)
    elif func == "package":
        ctx = click.Context(model_package)
        model_package.invoke(ctx, path=args.strip() or ".", output=None)
    elif func == "load":
        if not args.strip():
            click.secho("  usage: model load <concepts.json>", fg=theme.MUTED)
            return
        ctx = click.Context(model_load)
        model_load.invoke(ctx, file=args.strip(), model_id=None, batch_size=50)
    elif func == "data":
        ctx = click.Context(model_data)
        model_data.invoke(ctx, model_id=None, limit=20, offset=0)
    elif func == "count":
        ctx = click.Context(model_count)
        model_count.invoke(ctx, model_id=None)
    elif func == "clear":
        ctx = click.Context(model_clear)
        model_clear.invoke(ctx, model_id=None)
    elif func == "re-encode":
        ctx = click.Context(model_re_encode)
        model_re_encode.invoke(ctx, model_id=None)
    elif func == "test":
        ctx = click.Context(model_test)
        model_test.invoke(ctx, path=args.strip() or ".", verbose=True, keyword=None)
    else:
        click.echo()
        click.secho("  usage:", fg=theme.MUTED)
        click.secho("    model list                 List local models", fg=theme.MUTED)
        click.secho("    model deploy [path]        Deploy model to runtime", fg=theme.MUTED)
        click.secho("    model load <file>          Load data from concepts.json", fg=theme.MUTED)
        click.secho("    model data                 View stored glyphs", fg=theme.MUTED)
        click.secho("    model count                Show glyph/vector counts", fg=theme.MUTED)
        click.secho("    model clear                Clear all data (keep model)", fg=theme.MUTED)
        click.secho("    model re-encode            Re-encode all glyphs", fg=theme.MUTED)
        click.secho("    model status [model_id]    Check deployed status", fg=theme.MUTED)
        click.secho("    model undeploy [model_id]  Remove from runtime", fg=theme.MUTED)
        click.secho("    model init [name]          Scaffold new model", fg=theme.MUTED)
        click.secho("    model package [path]       Create .glyphh file", fg=theme.MUTED)
        click.secho("    model test [path]          Run model test suite", fg=theme.MUTED)
        click.echo()
