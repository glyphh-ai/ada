"""
CLI docker commands — scaffold Docker files for the runtime.

docker init    Write docker-compose.yml and init.sql to the current directory
"""

import click
import shutil
from pathlib import Path
from importlib import resources

from .. import theme

DOCKER_PKG = "glyphh.docker"
FILES = ["docker-compose.yml", "init.sql"]


@click.group("docker")
def docker_group():
    """Docker setup helpers."""
    pass


@docker_group.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing files without asking.")
def docker_init(force: bool):
    """Write docker-compose.yml and init.sql to the current directory."""
    dest = Path.cwd()

    for filename in FILES:
        target = dest / filename
        if target.exists() and not force:
            if not click.confirm(f"  {filename} already exists. Overwrite?"):
                click.secho(f"  Skipped {filename}", fg=theme.MUTED)
                continue

        src = resources.files(DOCKER_PKG).joinpath(filename)
        target.write_text(src.read_text())
        click.secho(f"  ✓ {filename}", fg=theme.SUCCESS)

    click.echo()
    click.secho("  Ready. Run:", fg=theme.TEXT)
    click.echo()
    click.secho("    docker compose up -d", fg=theme.ACCENT)
    click.echo()
    click.secho("  Verify:", fg=theme.TEXT)
    click.secho("    curl http://localhost:8002/health", fg=theme.MUTED)
