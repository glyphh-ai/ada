"""
CLI docker commands — scaffold Docker files for the runtime.

docker init    Write docker-compose.yml and init.sql to the current directory

When run from a model directory (contains manifest.yaml), the generated
docker-compose.yml will auto-mount the current directory as a custom model
so the runtime deploys it on startup.
"""

import click
import shutil
import yaml
from pathlib import Path
from importlib import resources

from .. import theme

DOCKER_PKG = "glyphh.docker"
FILES = ["docker-compose.yml", "init.sql"]


def _detect_model_dir(directory: Path) -> str | None:
    """If directory contains a manifest.yaml, return the model_id (directory name)."""
    manifest = directory / "manifest.yaml"
    if not manifest.exists():
        manifest = directory / "config.yaml"
    if not manifest.exists():
        return None
    try:
        with open(manifest) as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict) and ("name" in cfg or "encoder" in cfg):
            return directory.name
    except Exception:
        pass
    return None


def _inject_model_volume(compose_text: str, model_id: str) -> str:
    """Add a volume mount for the current model directory to the runtime service."""
    volume_line = f"      - .:/app/custom_models/{model_id}:ro"
    # Insert volumes section into the runtime service (after depends_on block)
    if "volumes:" not in compose_text.split("runtime:")[1].split("volumes:")[0]:
        # No volumes on runtime yet — add before restart
        compose_text = compose_text.replace(
            "    restart: unless-stopped",
            f"    volumes:\n{volume_line}\n    restart: unless-stopped",
        )
    return compose_text


@click.group("docker")
def docker_group():
    """Docker setup helpers."""
    pass


@docker_group.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing files without asking.")
def docker_init(force: bool):
    """Write docker-compose.yml and init.sql to the current directory.

    If the current directory is a Glyphh model (contains manifest.yaml),
    the generated docker-compose.yml will mount it so the runtime
    auto-deploys the model on startup.
    """
    dest = Path.cwd()
    model_id = _detect_model_dir(dest)

    for filename in FILES:
        target = dest / filename
        if target.exists() and not force:
            if not click.confirm(f"  {filename} already exists. Overwrite?"):
                click.secho(f"  Skipped {filename}", fg=theme.MUTED)
                continue

        src = resources.files(DOCKER_PKG).joinpath(filename)
        content = src.read_text()

        # If this is a model directory, inject the volume mount
        if filename == "docker-compose.yml" and model_id:
            content = _inject_model_volume(content, model_id)

        target.write_text(content)
        click.secho(f"  ✓ {filename}", fg=theme.SUCCESS)

    if model_id:
        click.secho(f"  ✓ Detected model: {model_id} (will auto-deploy)", fg=theme.SUCCESS)

    click.echo()
    click.secho("  Ready. Run:", fg=theme.TEXT)
    click.echo()
    click.secho("    docker compose up -d", fg=theme.ACCENT)
    click.echo()
    click.secho("  Verify:", fg=theme.TEXT)
    click.secho("    curl http://localhost:8002/health", fg=theme.MUTED)
