"""
Main CLI entry point using Click framework.

Supports both:
  - Direct subcommands: glyphh auth login, glyphh model list
  - Interactive shell:  glyphh (no args)
"""

import click
from .banner import print_banner
from .commands.auth import auth_group
from .commands.model import model_group
from .commands.catalog import catalog_group
from .commands.dev import dev_group
from .commands.chat import chat_command
from .commands.serve import serve_command
from .commands.docker import docker_group
from .commands.token import token_group
from .commands.query import query_command
from .commands.setup import setup_command
from .commands.config import config_group
from .commands.license import license_group


from importlib.metadata import version as _pkg_version

@click.group(invoke_without_command=True)
@click.version_option(version=_pkg_version("glyphh"), prog_name="glyphh")
@click.pass_context
def cli(ctx):
    """Glyphh — deterministic AI runtime. Run 'glyphh <command> --help' for details."""
    if ctx.invoked_subcommand is None:
        print_banner()
        click.echo(ctx.get_help())


cli.add_command(auth_group)
cli.add_command(model_group)
cli.add_command(catalog_group)
cli.add_command(dev_group)
cli.add_command(chat_command)
cli.add_command(serve_command)
cli.add_command(docker_group)
cli.add_command(token_group)
cli.add_command(query_command)
cli.add_command(setup_command)
cli.add_command(config_group)
cli.add_command(license_group)


if __name__ == "__main__":
    cli()
