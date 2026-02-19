"""
Main CLI entry point using Click framework
"""

import click
from .shell import shell


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="glyphh")
@click.pass_context
def cli(ctx):
    """
    Glyphh CLI

    \b
    Run 'glyphh' to start an interactive shell.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(shell)


cli.add_command(shell)


if __name__ == "__main__":
    cli()
