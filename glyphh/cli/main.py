"""
Main CLI entry point using Click framework
"""

import click
from .build import build
from .test import test
from .package import package
from .runtime import runtime
from .query import query
from .procedure import procedure
from .auth import auth
from .config import settings
from .shell import shell
from .discovery import models, hub, docs, demo


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="glyphh")
@click.pass_context
def cli(ctx):
    """
    Glyphh SDK - Hyperdimensional Computing for RAG
    
    A command-line interface for building, testing, and packaging
    hyperdimensional computing models.
    
    \b
    Run 'glyphh' with no arguments to start an interactive shell.
    
    \b
    COMMAND CATEGORIES:
      auth      Authentication (login, logout, signup, whoami)
      build     Model development (init, add, validate, info)
      test      Testing (similarity, encode, visualize)
      package   Packaging (create, inspect, validate)
      runtime   Deployment & management (deploy, status, logs, token)
      query     Execute GQL/NL queries against local or remote models
      procedure Manage stored procedures for deployed models
    
    \b
    QUICK START:
      # Start interactive shell
      glyphh
      
      # Login to your account
      glyphh auth login
      
      # Initialize a new model
      glyphh build init my_model
      
      # Create a .glyphh package
      glyphh package create my_model.glyphh
      
      # Deploy to runtime
      glyphh runtime deploy my_model.glyphh
      
      # Query a model
      glyphh query "FIND SIMILAR TO 'cars'" --model ./my-model.glyphh
    
    For more information, see: https://docs.glyphh.com/cli
    """
    # If no command is given, start the interactive shell
    if ctx.invoked_subcommand is None:
        ctx.invoke(shell)


# Register command groups
cli.add_command(auth)
cli.add_command(build)
cli.add_command(settings)
cli.add_command(test)
cli.add_command(package)
cli.add_command(runtime)
cli.add_command(query)
cli.add_command(procedure)
cli.add_command(shell)

# Register discovery commands
cli.add_command(models)
cli.add_command(hub)
cli.add_command(docs)
cli.add_command(demo)


# TUI command (optional dependency — textual)
@click.command()
def tui():
    """Launch the full-screen Terminal UI (requires textual)."""
    try:
        from .tui import run_tui
        run_tui()
    except ImportError:
        click.secho("TUI requires textual. Install with: pip install glyphh[tui]", fg="red")


cli.add_command(tui)


if __name__ == "__main__":
    cli()
