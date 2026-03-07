"""
CLI docker commands — scaffold Docker files for the runtime.

docker init    Write docker-compose.yml and init.sql to the current directory

When run from a model directory (contains manifest.yaml), the generated
docker-compose.yml will auto-mount the current directory as a custom model
so the runtime deploys it on startup.

License is shared via volume mount (~/.glyphh:/root/.glyphh:ro).
"""

import click
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
    # Insert after the existing ~/.glyphh volume line
    compose_text = compose_text.replace(
        "      - ${HOME}/.glyphh:/home/glyphh/.glyphh:ro",
        f"      - ${{HOME}}/.glyphh:/home/glyphh/.glyphh:ro\n{volume_line}",
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
    click.secho("    docker compose up -d --wait", fg=theme.ACCENT)
    click.secho("    # waits until runtime is healthy (models deployed)", fg=theme.TEXT_DIM)
    click.echo()
    click.secho("  Your ~/.glyphh/ directory is shared with Docker.", fg=theme.TEXT_DIM)
    click.secho("  Log in (auth login) to use your plan's full limits.", fg=theme.TEXT_DIM)
    click.echo()
    click.secho("  Verify:", fg=theme.TEXT)
    click.secho("    curl http://localhost:8002/health", fg=theme.MUTED)


# ── Handler for interactive shell ──

def handle_docker(func: str | None, args: str = ""):
    """Route docker subcommands from the interactive shell."""
    if func == "init":
        force = "--force" in args if args else False
        ctx = click.Context(docker_init)
        ctx.invoke(docker_init, force=force)
    else:
        click.secho("  usage: docker init [--force]", fg=theme.MUTED)
