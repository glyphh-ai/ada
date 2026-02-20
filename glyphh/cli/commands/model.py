"""
CLI model commands — local model management.

model list       List local models found on disk
model deploy     Deploy a model to the runtime
model status     Check deployed model status
model undeploy   Remove a model from the runtime
model init       Scaffold a new model project
model package    Package a model directory into a .glyphh file
"""

import click
from pathlib import Path

from .. import theme
from ..auth import get_token, get_api_url, is_logged_in
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
        click.secho("  Not logged in. Run: auth login", fg=theme.ERROR)
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

    # Upload to runtime
    token = get_token()
    api_url = get_api_url()

    try:
        import httpx

        with httpx.Client(timeout=60) as client:
            with open(glyphh_file, "rb") as f:
                res = client.post(
                    f"{api_url}/runtimes/deploy",
                    files={"file": (glyphh_file.name, f, "application/octet-stream")},
                    headers={"Authorization": f"Bearer {token}"},
                )

            if res.status_code in (200, 201):
                data = res.json()
                click.echo()
                click.secho(f"  ✓ Deployed: {data.get('model_id', glyphh_file.stem)}", fg=theme.SUCCESS)
                click.echo()
            else:
                detail = res.json().get("error", {}).get("message", res.text) if res.headers.get("content-type", "").startswith("application/json") else res.text
                click.secho(f"  Deploy failed: {detail}", fg=theme.ERROR)

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
        click.secho("  Not logged in. Run: auth login", fg=theme.ERROR)
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

    token = get_token()
    api_url = get_api_url()

    try:
        import httpx

        with httpx.Client(timeout=15) as client:
            res = client.get(
                f"{api_url}/models/{model_id}",
                headers={"Authorization": f"Bearer {token}"},
            )

        if res.status_code == 200:
            data = res.json()
            click.echo()
            click.secho(f"  {data.get('name', model_id)}", fg=theme.TEXT)
            click.secho(f"  Status: {data.get('status', 'unknown')}", fg=theme.INFO)
            if data.get("description"):
                click.secho(f"  {data['description']}", fg=theme.MUTED)
            click.echo()
        elif res.status_code == 404:
            click.secho(f"  Model '{model_id}' not found.", fg=theme.WARNING)
        else:
            click.secho(f"  Error: {res.text}", fg=theme.ERROR)

    except Exception as e:
        click.secho(f"  Could not check status: {e}", fg=theme.ERROR)


@model_group.command("undeploy")
@click.argument("model_id", required=False)
def model_undeploy(model_id):
    """Remove a model from the runtime.

    If MODEL_ID is omitted and you're in a model directory, uses that model.
    """
    if not is_logged_in():
        click.secho("  Not logged in. Run: auth login", fg=theme.ERROR)
        return

    if not model_id:
        model_dir = find_model_dir()
        if model_dir:
            manifest = read_manifest(model_dir)
            model_id = manifest.get("model_id", model_dir.name)
        else:
            click.secho("  Provide a model_id or run from a model directory.", fg=theme.MUTED)
            return

    token = get_token()
    api_url = get_api_url()

    try:
        import httpx

        click.secho(f"  Undeploying {model_id}...", fg=theme.MUTED)
        with httpx.Client(timeout=30) as client:
            res = client.post(
                f"{api_url}/models/{model_id}/undeploy",
                headers={"Authorization": f"Bearer {token}"},
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
    elif func == "test":
        ctx = click.Context(model_test)
        model_test.invoke(ctx, path=args.strip() or ".", verbose=True, keyword=None)
    else:
        click.echo()
        click.secho("  usage:", fg=theme.MUTED)
        click.secho("    model list                 List local models", fg=theme.MUTED)
        click.secho("    model deploy [path]        Deploy model to runtime", fg=theme.MUTED)
        click.secho("    model status [model_id]    Check deployed status", fg=theme.MUTED)
        click.secho("    model undeploy [model_id]  Remove from runtime", fg=theme.MUTED)
        click.secho("    model init [name]          Scaffold new model", fg=theme.MUTED)
        click.secho("    model package [path]       Create .glyphh file", fg=theme.MUTED)
        click.secho("    model test [path]          Run model test suite", fg=theme.MUTED)
        click.echo()
