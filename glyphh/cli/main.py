"""
Main CLI entry point.

The interactive shell is the primary interface — run `glyphh` to start.
`glyphh serve` is the only standalone subcommand (used by Docker/production).
"""

import click
from .commands.serve import serve_command


try:
    from importlib.metadata import version as _pkg_version
    _version = _pkg_version("glyphh")
except Exception:
    _version = "0.8.9"

@click.group(invoke_without_command=True)
@click.version_option(version=_version, prog_name="glyphh")
@click.pass_context
def cli(ctx):
    """Glyphh — deterministic AI runtime. Run 'glyphh' to start the interactive shell."""
    if ctx.invoked_subcommand is None:
        from .shell import shell
        ctx.invoke(shell)


# Only keep serve (used by Docker container / production deployments)
cli.add_command(serve_command)


if __name__ == "__main__":
    cli()
