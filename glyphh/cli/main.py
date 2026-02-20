"""
Main CLI entry point using Click framework.

Supports both:
  - Direct subcommands: glyphh auth login, glyphh model list
  - Interactive shell:  glyphh (no args)
"""

import click
from .shell import shell
from .commands.auth import auth_group
from .commands.model import model_group
from .commands.catalog import catalog_group
from .commands.serve import serve_command
from .commands.docker import docker_group


@click.group(invoke_without_command=True)
@click.version_option(version="0.4.0", prog_name="glyphh")
@click.pass_context
def cli(ctx):
    """
    Glyphh CLI

    \b
    Run 'glyphh' to start an interactive shell.
    Run 'glyphh <category> <command>' for direct execution.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(shell)


cli.add_command(shell)
cli.add_command(auth_group)
cli.add_command(model_group)
cli.add_command(catalog_group)
cli.add_command(serve_command)
cli.add_command(docker_group)


if __name__ == "__main__":
    cli()
