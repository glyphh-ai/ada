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
from ..auth import get_token, get_api_url, is_logged_in, resolve_org_id
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
@click.option("--remote", is_flag=True, help="List models deployed on the remote runtime.")
def model_list(remote):
    """List models deployed on the runtime."""
    _list_remote_models()


def _list_remote_models():
    """Fetch and display models from the remote runtime."""
    if not is_logged_in():
        click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
        return

    runtime_url = resolve_runtime_url()
    token = resolve_runtime_token()
    org_id = resolve_org_id(runtime_url)
    if not org_id:
        click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
        return

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        import httpx
        with httpx.Client(timeout=15) as client:
            res = client.get(f"{runtime_url}/{org_id}/models", headers=headers)

        if res.status_code == 404:
            click.secho("  No models deployed yet. Deploy one first:", fg=theme.MUTED)
            click.secho("    glyphh model deploy", fg=theme.MUTED)
            return
        if res.status_code != 200:
            detail = res.text
            try:
                detail = res.json().get("detail", detail)
            except Exception:
                pass
            click.secho(f"  Failed: {detail}", fg=theme.ERROR)
            return

        data = res.json()
        models = data.get("models", [])
        if not models:
            click.secho("  No models deployed on runtime.", fg=theme.MUTED)
            return

        click.echo()
        click.secho(f"  {runtime_url}", fg=theme.TEXT_DIM)
        click.echo()
        header = f"  {'MODEL ID':<20} {'NAME':<28} {'VERSION':<10} {'GLYPHS':<10} STATUS"
        click.secho(header, fg=theme.TEXT_DIM)
        click.secho("  " + "─" * 80, fg=theme.TEXT_DIM)

        for m in models:
            mid = m.get("model_id", "?")[:18]
            name = (m.get("name") or mid)[:26]
            ver = (m.get("version") or "—")[:8]
            glyphs = str(m.get("glyphs", 0))
            status = m.get("status", "—")

            click.echo(
                click.style(f"  {mid:<20} ", fg=theme.ACCENT)
                + click.style(f"{name:<28} ", fg=theme.TEXT)
                + click.style(f"{ver:<10} ", fg=theme.MUTED)
                + click.style(f"{glyphs:<10} ", fg=theme.INFO)
                + click.style(status, fg=theme.SUCCESS)
            )
        click.echo()

    except Exception:
        click.secho(f"  Could not connect to runtime at {runtime_url}", fg=theme.WARNING)
        click.secho("  Is the runtime running? Check:", fg=theme.MUTED)
        click.secho("    glyphh dev start          (local dev server)", fg=theme.MUTED)
        click.secho("    docker compose up -d      (Docker)", fg=theme.MUTED)
        click.secho(f"    glyphh config set endpoint <url>", fg=theme.MUTED)


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

    # Resolve model_id: prefer explicit model_id, then directory name, then file stem.
    # Never use manifest "name" — that's a display label, not a technical identifier.
    if target.is_dir():
        model_id = manifest.get("model_id") or target.name
    else:
        model_id = manifest.get("model_id") or glyphh_file.stem
    model_id = model_id.replace(".glyphh", "")

    # Upload to runtime (local or remote)
    runtime_url = resolve_runtime_url()
    token = resolve_runtime_token()

    org_id = resolve_org_id(runtime_url)
    if not org_id:
        click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
        return

    headers = {}
    if token:
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

    org_id = resolve_org_id(runtime_url)
    if not org_id:
        click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
        return

    headers = {}
    if token:
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

    org_id = resolve_org_id(runtime_url)
    if not org_id:
        click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
        return

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        import httpx

        click.secho(f"  Undeploying {model_id}...", fg=theme.MUTED)
        with httpx.Client(timeout=30) as client:
            res = client.delete(
                f"{runtime_url}/{org_id}/{model_id}/model",
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

    # Load concepts file (JSON array, {"records": [...]}, or JSONL)
    target = Path(file).resolve()
    try:
        raw = target.read_text()
    except Exception as e:
        click.secho(f"  Could not read file: {e}", fg=theme.ERROR)
        return

    records = None
    # Try JSON first (array or {"records": [...]})
    try:
        data = json_mod.loads(raw)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and "records" in data:
            records = data["records"]
    except json_mod.JSONDecodeError:
        pass

    # Fall back to JSONL (one JSON object per line)
    if records is None:
        records = []
        for i, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json_mod.loads(line))
            except json_mod.JSONDecodeError as e:
                click.secho(f"  Invalid JSON on line {i}: {e}", fg=theme.ERROR)
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

    runtime_url = resolve_runtime_url()
    org_id = resolve_org_id(runtime_url)
    if not org_id:
        click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
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
    if not is_logged_in():
        click.secho("  Not logged in. Run: glyphh auth login", fg=theme.ERROR)
        return None

    runtime_url = resolve_runtime_url()
    token = resolve_runtime_token()

    org_id = resolve_org_id(runtime_url)
    if not org_id:
        click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
        return None

    headers = {}
    if token:
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

    runtime_url = resolve_runtime_url()
    org_id = resolve_org_id(runtime_url)
    if not org_id:
        click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
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

    headers = {}
    token = resolve_runtime_token()
    if token:
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

    runtime_url = resolve_runtime_url()
    org_id = resolve_org_id(runtime_url)
    if not org_id:
        click.secho("  No org_id in session. Run: glyphh auth login", fg=theme.ERROR)
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

    headers = {}
    token = resolve_runtime_token()
    if token:
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


# ── Dispatch a single model command ──

def _find_model_source(model_id: str | None) -> str | None:
    """Find the source directory for a model.

    Checks (in order):
    1. ~/.glyphh/models/<model-id>/  (hub-installed)
    2. Current working directory (if it's a model dir)
    """
    if model_id:
        hub_dir = Path.home() / ".glyphh" / "models"
        # Try exact model_id match
        candidate = hub_dir / model_id
        if candidate.is_dir() and is_model_dir(candidate):
            return str(candidate)
        # Try without "model-" prefix (hub uses short id like "firewall")
        if model_id.startswith("model-"):
            candidate = hub_dir / model_id[6:]
            if candidate.is_dir() and is_model_dir(candidate):
                return str(candidate)
    # Fall back to cwd
    model_dir = find_model_dir()
    if model_dir:
        return str(model_dir)
    return None


def _dispatch_model_cmd(cmd: str, args: str, model_id: str | None = None):
    """Execute a single model subcommand. model_id scopes data commands."""
    if cmd == "list":
        _list_remote_models()
    elif cmd == "deploy":
        arg = args.strip()
        if arg:
            # If explicit arg doesn't exist in cwd, try model source dir
            if not Path(arg).exists() and model_id:
                source = _find_model_source(model_id)
                if source and (Path(source) / arg).exists():
                    arg = str(Path(source) / arg)
            path = arg
        else:
            path = _find_model_source(model_id) or "."
        c = click.Context(model_deploy)
        c.invoke(model_deploy, path=path)
    elif cmd == "status":
        mid = args.strip() or model_id
        c = click.Context(model_status)
        c.invoke(model_status, model_id=mid)
    elif cmd == "undeploy":
        mid = args.strip() or model_id
        c = click.Context(model_undeploy)
        c.invoke(model_undeploy, model_id=mid)
    elif cmd == "init":
        c = click.Context(model_init)
        c.invoke(model_init, name=args.strip() or None)
    elif cmd == "package":
        path = args.strip() or _find_model_source(model_id) or "."
        c = click.Context(model_package)
        c.invoke(model_package, path=path, output=None)
    elif cmd == "load":
        if not args.strip():
            click.secho("  Usage: load <concepts.json>", fg=theme.MUTED)
            return
        c = click.Context(model_load)
        c.invoke(model_load, file=args.strip(), model_id=model_id, batch_size=50)
    elif cmd == "data":
        c = click.Context(model_data)
        c.invoke(model_data, model_id=model_id, limit=20, offset=0)
    elif cmd == "count":
        c = click.Context(model_count)
        c.invoke(model_count, model_id=model_id)
    elif cmd == "clear":
        c = click.Context(model_clear)
        c.invoke(model_clear, model_id=model_id)
    elif cmd == "re-encode":
        c = click.Context(model_re_encode)
        c.invoke(model_re_encode, model_id=model_id)
    elif cmd == "test":
        path = args.strip() or _find_model_source(model_id) or "."
        c = click.Context(model_test)
        c.invoke(model_test, path=path, verbose=True, keyword=None)
    elif cmd == "chat":
        from .chat import handle_chat
        if model_id:
            handle_chat(model_id, args)
        else:
            click.secho("  Enter a model first: model <model-id>", fg=theme.MUTED)
    elif cmd == "query":
        from .query import handle_query
        if model_id:
            handle_query(model_id, args)
        else:
            click.secho("  Enter a model first: model <model-id>", fg=theme.MUTED)
    else:
        return False  # unknown command
    return True


def _print_model_help(model_id: str | None = None):
    """Print help for the model REPL."""
    if model_id:
        click.echo()
        click.secho(f"  {model_id}", fg=theme.ACCENT, bold=True)
        click.echo()
        click.secho("    status                     Check model status", fg=theme.MUTED)
        click.secho("    data                       View stored glyphs", fg=theme.MUTED)
        click.secho("    count                      Show glyph/vector counts", fg=theme.MUTED)
        click.secho("    load <file>                Load data from file", fg=theme.MUTED)
        click.secho("    clear                      Clear all data", fg=theme.MUTED)
        click.secho("    re-encode                  Re-encode all glyphs", fg=theme.MUTED)
        click.secho("    undeploy                   Remove from runtime", fg=theme.MUTED)
        click.secho("    chat [query]               Interactive chat REPL", fg=theme.MUTED)
        click.secho("    query <question>           Single query", fg=theme.MUTED)
    else:
        click.secho("  model commands", fg=theme.TEXT)
        click.echo()
        click.secho("    list                       List deployed models", fg=theme.MUTED)
        click.secho("    deploy [path]              Deploy model to runtime", fg=theme.MUTED)
        click.secho("    status [model-id]          Check deployed status", fg=theme.MUTED)
        click.secho("    undeploy [model-id]        Remove from runtime", fg=theme.MUTED)
        click.secho("    init [name]                Scaffold new model", fg=theme.MUTED)
        click.secho("    package [path]             Create .glyphh file", fg=theme.MUTED)
        click.secho("    test [path]                Run model test suite", fg=theme.MUTED)
    click.echo()
    click.secho("    help                       Show this message", fg=theme.MUTED)
    click.secho("    q                          Exit model shell", fg=theme.MUTED)
    click.echo()


def _show_single_model(model_id: str):
    """Show the model table filtered to a single model."""
    try:
        import httpx
        runtime_url = resolve_runtime_url()
        token = resolve_runtime_token()
        org_id = resolve_org_id(runtime_url)
        if not org_id:
            return
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with httpx.Client(timeout=10) as client:
            res = client.get(f"{runtime_url}/{org_id}/models", headers=headers)
        if res.status_code != 200:
            return
        models = [m for m in res.json().get("models", []) if m.get("model_id") == model_id]
        if not models:
            click.echo()
            click.secho(f"  {model_id} — not deployed", fg=theme.WARNING)
            return

        click.echo()
        click.secho(f"  {runtime_url}", fg=theme.TEXT_DIM)
        click.echo()
        header = f"  {'MODEL ID':<20} {'NAME':<28} {'VERSION':<10} {'GLYPHS':<10} STATUS"
        click.secho(header, fg=theme.TEXT_DIM)
        click.secho("  " + "─" * 80, fg=theme.TEXT_DIM)
        m = models[0]
        mid = m.get("model_id", "?")[:18]
        name = (m.get("name") or mid)[:26]
        ver = (m.get("version") or "—")[:8]
        glyphs = str(m.get("glyphs", 0))
        status = m.get("status", "—")
        click.echo(
            click.style(f"  {mid:<20} ", fg=theme.ACCENT)
            + click.style(f"{name:<28} ", fg=theme.TEXT)
            + click.style(f"{ver:<10} ", fg=theme.MUTED)
            + click.style(f"{glyphs:<10} ", fg=theme.INFO)
            + click.style(status, fg=theme.SUCCESS)
        )
    except Exception:
        pass


def _get_deployed_model_ids():
    """Fetch list of deployed model IDs from the runtime."""
    try:
        import httpx
        runtime_url = resolve_runtime_url()
        token = resolve_runtime_token()
        org_id = resolve_org_id(runtime_url)
        if not org_id:
            return set()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with httpx.Client(timeout=5) as client:
            res = client.get(f"{runtime_url}/{org_id}/models", headers=headers)
        if res.status_code == 200:
            return {m.get("model_id", "") for m in res.json().get("models", [])}
    except Exception:
        pass
    return set()


_UNSCOPED_CMDS = [
    "list", "deploy", "status", "undeploy", "init", "package", "test",
    "help", "q", "quit", "exit",
]

_SCOPED_CMDS = [
    "status", "data", "count", "load", "clear", "re-encode", "undeploy",
    "deploy", "package", "test", "chat", "query",
    "help", "q", "quit", "exit",
]


def _setup_model_completer(model_id: str | None, deployed_ids: set):
    """Install a readline completer for the model REPL."""
    try:
        import readline
    except ImportError:
        return None

    if model_id:
        completions = _SCOPED_CMDS
    else:
        completions = _UNSCOPED_CMDS + sorted(deployed_ids)

    old_completer = readline.get_completer()

    def _completer(text, state):
        options = [c + " " for c in completions if c.startswith(text)]
        try:
            return options[state]
        except IndexError:
            return None

    readline.set_completer(_completer)
    return old_completer


def _restore_completer(old_completer):
    """Restore the previous readline completer."""
    try:
        import readline
        readline.set_completer(old_completer)
    except ImportError:
        pass


def _model_repl(model_id: str | None = None):
    """Interactive model REPL. If model_id is set, commands are scoped to it."""
    if model_id:
        prompt_label = model_id
    else:
        prompt_label = "model"

    # Show model table on REPL entry
    deployed_ids = set()
    if model_id:
        _show_single_model(model_id)
    else:
        _list_remote_models()
        deployed_ids = _get_deployed_model_ids()

    _print_model_help(model_id)

    known_cmds = {
        "list", "deploy", "status", "undeploy", "init", "package",
        "load", "data", "count", "clear", "re-encode", "test",
        "chat", "query",
    }

    old_completer = _setup_model_completer(model_id, deployed_ids)

    while True:
        try:
            prompt = click.style(f"  {prompt_label}", fg=theme.PRIMARY) + click.style("> ", fg=theme.TEXT)
            line = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            click.echo()
            _restore_completer(old_completer)
            return

        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            _restore_completer(old_completer)
            return
        if line.lower() == "help":
            _print_model_help(model_id)
            continue

        parts = line.split(None, 1)
        cmd = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""

        # In unscoped REPL, allow typing a model-id to enter scoped REPL
        if not model_id and cmd not in known_cmds:
            # Could be a model-id — check deployed list or just try it
            if cmd in deployed_ids or cmd.startswith("model-"):
                sub_args = cmd_args.strip()
                if sub_args:
                    # model> model-firewall status → run single scoped command
                    sub_parts = sub_args.split(None, 1)
                    _dispatch_model_cmd(sub_parts[0].lower(), sub_parts[1] if len(sub_parts) > 1 else "", cmd)
                else:
                    # model> model-firewall → enter scoped REPL
                    _model_repl(cmd)
                # Refresh deployed IDs in case something changed
                deployed_ids = _get_deployed_model_ids()
                continue

        if not _dispatch_model_cmd(cmd, cmd_args, model_id):
            click.secho(f"  Unknown: {cmd}. Type 'help' for commands.", fg=theme.MUTED)


# ── Handler for interactive shell ──

def handle_model(func: str | None, args: str = ""):
    """Route model subcommands from the interactive shell.

    Usage:
        model                      Enter model shell
        model <model-id>           Enter model shell scoped to a model
        model <subcommand> [args]  Run a single model command
    """
    if func is None:
        # model → enter unscoped REPL
        _model_repl()
        return

    # Check if func is a known subcommand
    known_cmds = {
        "list", "deploy", "status", "undeploy", "init", "package",
        "load", "data", "count", "clear", "re-encode", "test",
    }

    if func in known_cmds:
        # Direct subcommand: model list, model deploy ./path, etc.
        _dispatch_model_cmd(func, args)
    else:
        # Treat func as a model-id → enter scoped REPL
        model_id = func.strip()
        if args.strip():
            # model model-firewall status → run single scoped command
            parts = args.strip().split(None, 1)
            cmd = parts[0].lower()
            cmd_args = parts[1] if len(parts) > 1 else ""
            _dispatch_model_cmd(cmd, cmd_args, model_id)
        else:
            # model model-firewall → enter scoped REPL
            _model_repl(model_id)
