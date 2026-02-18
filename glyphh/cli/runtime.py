"""
Runtime commands for deploying models and managing Runtime API.

This module provides CLI commands for:
- Deploying .glyphh models to Runtime servers
- Managing webhook tokens
- Checking runtime status
- Viewing runtime logs

Configuration is loaded from .env file with:
- RUNTIME_URL: URL of the Runtime API server
- JWT_TOKEN: Optional JWT token for authentication
"""

import sys
from pathlib import Path
from typing import Optional

import click

from glyphh.cli.config import (
    ConfigurationError,
    generate_env_template,
    load_env_config,
)
from glyphh.cli.runtime_client import (
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    NotFoundError,
    RuntimeAPIClient,
    RuntimeAPIError,
)


@click.group()
def runtime():
    """
    Runtime deployment and management commands.
    
    \b
    CONFIGURATION:
      Configure your runtime connection in a .env file:
      
      # .env
      RUNTIME_URL=http://localhost:8000
      JWT_TOKEN=your_token_here  # optional for local
      RUNTIME_TIMEOUT=30         # optional, default 30s
    
    \b
    DEPLOYMENT SCENARIOS:
      Local Development:
        RUNTIME_URL=http://localhost:8000
        (No JWT_TOKEN needed)
      
      Self-Hosted (Heroku/AWS):
        RUNTIME_URL=https://your-app.herokuapp.com
        JWT_TOKEN=<from Platform UI>
      
      Glyphh Cloud:
        RUNTIME_URL=https://runtime.glyphh.com
        JWT_TOKEN=<from Platform UI>
    
    \b
    QUICK START:
      # Generate .env template
      glyphh runtime init --scenario local
      
      # Deploy a model
      glyphh runtime deploy my_model.glyphh
      
      # Check status
      glyphh runtime status
    
    \b
    TOKEN MANAGEMENT:
      Deployment tokens are obtained from the Platform UI at
      https://platform.glyphh.com → Settings → API Tokens
      
      Consumer tokens for your applications are generated
      in the Platform UI after deployment.
    """
    pass


@runtime.command()
@click.option(
    '--scenario',
    type=click.Choice(['local', 'self-hosted', 'cloud']),
    default='local',
    help='Deployment scenario for the template'
)
@click.option(
    '--output', '-o',
    default='.env',
    help='Output path for the .env file'
)
@click.option(
    '--force', '-f',
    is_flag=True,
    help='Overwrite existing .env file'
)
def init(scenario: str, output: str, force: bool):
    """
    Generate a .env configuration template.
    
    \b
    Examples:
        # Generate local development template
        glyphh runtime init
        
        # Generate self-hosted template
        glyphh runtime init --scenario self-hosted
        
        # Generate cloud template
        glyphh runtime init --scenario cloud -o .env.production
    """
    output_path = Path(output)
    
    if output_path.exists() and not force:
        click.echo(
            click.style("Error: ", fg="red") +
            f"File already exists: {output}\n"
            "Use --force to overwrite."
        )
        sys.exit(1)
    
    try:
        generated_path = generate_env_template(output, scenario)
        click.echo(
            click.style("[OK] ", fg="green") +
            f"Generated {scenario} configuration template: {generated_path}"
        )
        click.echo("\nNext steps:")
        if scenario == "local":
            click.echo("  1. Start your local runtime: runtime start")
            click.echo("  2. Deploy your model: runtime deploy model.glyphh")
        else:
            click.echo("  1. Edit the .env file with your actual values")
            click.echo("  2. Get a JWT token from https://platform.glyphh.com")
            click.echo("  3. Deploy your model: runtime deploy model.glyphh")
    except ConfigurationError as e:
        click.echo(click.style("Error: ", fg="red") + str(e))
        sys.exit(1)



@runtime.command()
@click.argument('package_file', type=click.Path(exists=True))
@click.option(
    '--env-file',
    default=None,
    help='Path to .env configuration file'
)
@click.option(
    '--name',
    default=None,
    help='Deployment name (defaults to model name from package)'
)
def deploy(package_file: str, env_file: Optional[str], name: Optional[str]):
    """
    Deploy a .glyphh model to the Runtime API.
    
    Reads RUNTIME_URL and JWT_TOKEN from .env file (or environment variables).
    
    \b
    Examples:
        # Deploy using default .env
        glyphh runtime deploy my_model.glyphh
        
        # Deploy with custom .env file
        glyphh runtime deploy my_model.glyphh --env-file .env.production
        
        # Deploy with custom name
        glyphh runtime deploy my_model.glyphh --name production-model
    """
    # Load configuration
    try:
        config = load_env_config(env_file)
    except ConfigurationError as e:
        click.echo(click.style("Configuration Error: ", fg="red") + str(e))
        sys.exit(1)
    
    # Validate package file
    package_path = Path(package_file)
    if not package_path.suffix == '.glyphh':
        click.echo(
            click.style("Warning: ", fg="yellow") +
            f"File does not have .glyphh extension: {package_file}"
        )
    
    # Show package info
    click.echo(f"Reading package: {package_file}")
    try:
        package_size_mb = package_path.stat().st_size / (1024 * 1024)
        click.echo(f"   Size: {package_size_mb:.2f} MB")
    except IOError as e:
        click.echo(click.style("Error: ", fg="red") + f"Failed to read package: {e}")
        sys.exit(1)
    
    # Create client and deploy
    client = RuntimeAPIClient(config)
    click.echo(f"Deploying to: {config.runtime_url}")
    
    try:
        with click.progressbar(
            length=100,
            label='   Uploading',
            show_eta=False
        ) as bar:
            bar.update(50)
            result = client.deploy_model(package_file, name=name)
            bar.update(50)
    except ConnectionError as e:
        click.echo(
            click.style("\nConnection Error: ", fg="red") + str(e)
        )
        if config.is_local:
            click.echo("   Start local runtime with: runtime start")
        sys.exit(1)
    except AuthenticationError:
        click.echo(
            click.style("\nAuthentication Error: ", fg="red") +
            "Invalid or missing JWT token.\n"
            "   Get a token from https://platform.glyphh.com and add it to your .env file."
        )
        sys.exit(1)
    except AuthorizationError:
        click.echo(
            click.style("\nAuthorization Error: ", fg="red") +
            "You don't have permission to deploy to this runtime.\n"
            "   Check your token permissions in the Platform UI."
        )
        sys.exit(1)
    except RuntimeAPIError as e:
        click.echo(
            click.style(f"\nDeployment Failed: ", fg="red") +
            str(e)
        )
        if e.details:
            for key, value in e.details.items():
                click.echo(f"   {key}: {value}")
        sys.exit(1)
    
    # Display success
    click.echo(click.style("\n[OK] ", fg="green") + "Model deployed successfully!")
    click.echo()
    
    # Display deployment details
    if result.model_id:
        click.echo(f"Model ID: {result.model_id}")
    if result.org_id:
        click.echo(f"Org ID: {result.org_id}")
    if result.version:
        click.echo(f"Version: {result.version}")
    
    click.echo()
    
    # Display endpoints
    if result.mcp_endpoint or result.listener_endpoint:
        click.echo("Endpoints:")
        if result.mcp_endpoint:
            click.echo(f"   MCP:      {result.mcp_endpoint}")
        if result.listener_endpoint:
            click.echo(f"   Listener: {result.listener_endpoint}")
    
    # Display token if provided
    if result.webhook_token:
        click.echo()
        click.echo(f"Webhook Token: {result.webhook_token}")
        click.echo()
        click.echo(
            click.style("Note: ", fg="cyan") +
            "Save this token securely. Use it to authenticate requests to the endpoints."
        )
    
    # Display next steps
    click.echo()
    click.echo("Next steps:")
    click.echo("  - Generate consumer tokens in the Platform UI for your applications")
    click.echo("  - Use the MCP endpoint for AI assistant integrations")
    click.echo("  - Use the Listener endpoint for webhook-based integrations")



@runtime.command()
@click.option(
    '--env-file',
    default=None,
    help='Path to .env configuration file'
)
def status(env_file: Optional[str]):
    """
    Check the status of the Runtime server.
    
    \b
    Example:
        glyphh runtime status
    """
    try:
        config = load_env_config(env_file)
    except ConfigurationError as e:
        click.echo(click.style("Configuration Error: ", fg="red") + str(e))
        sys.exit(1)
    
    client = RuntimeAPIClient(config)
    click.echo(f"Checking runtime status: {config.runtime_url}")
    
    try:
        result = client.get_status()
        click.echo(click.style("[OK] Online", fg="green"))
        if result.version:
            click.echo(f"  Version: {result.version}")
        if result.models_loaded is not None:
            click.echo(f"  Models loaded: {result.models_loaded}")
        if result.uptime:
            click.echo(f"  Uptime: {result.uptime}")
    except ConnectionError:
        click.echo(
            click.style("Offline: ", fg="red") +
            f"Could not connect to {config.runtime_url}"
        )
        sys.exit(1)
    except RuntimeAPIError as e:
        click.echo(click.style(f"Error: ", fg="red") + str(e))
        sys.exit(1)


@runtime.command()
@click.option('--port', default=8000, help='Port to run the server on')
@click.option('--host', default='127.0.0.1', help='Host to bind to')
def start(port: int, host: str):
    """
    Start a local runtime server (for development).
    
    \b
    Example:
        glyphh runtime start --port 8000
    
    Note: This starts a local development server.
    For production, deploy to Heroku, AWS, or Glyphh Cloud.
    """
    click.echo(
        click.style("Note: ", fg="cyan") +
        "Local runtime server is part of the glyphh-runtime package.\n"
        "Install it with: pip install glyphh-runtime\n"
        "Then run: glyphh-runtime start --port {port} --host {host}"
    )
    click.echo()
    click.echo("For now, you can use the runtime from the glyphh-runtime repository:")
    click.echo(f"  cd glyphh-runtime && python -m uvicorn api.main:app --port {port} --host {host}")


@runtime.command()
@click.option(
    '--env-file',
    default=None,
    help='Path to .env configuration file'
)
@click.option('--lines', '-n', default=100, help='Number of log lines to show')
def logs(env_file: Optional[str], lines: int):
    """
    View runtime server logs.
    
    \b
    Example:
        glyphh runtime logs
        glyphh runtime logs -n 50
    """
    try:
        config = load_env_config(env_file)
    except ConfigurationError as e:
        click.echo(click.style("Configuration Error: ", fg="red") + str(e))
        sys.exit(1)
    
    client = RuntimeAPIClient(config)
    
    try:
        log_lines = client.get_logs(lines=lines)
        for line in log_lines:
            click.echo(line)
    except AuthenticationError:
        click.echo(
            click.style("Error: ", fg="red") +
            "Authentication required to view logs."
        )
        sys.exit(1)
    except RuntimeAPIError as e:
        click.echo(click.style(f"Error: ", fg="red") + str(e))
        sys.exit(1)


# Token management subgroup
@runtime.group()
def token():
    """Manage webhook tokens."""
    pass


@token.command('list')
@click.option(
    '--env-file',
    default=None,
    help='Path to .env configuration file'
)
def token_list(env_file: Optional[str]):
    """
    List all webhook tokens.
    
    \b
    Example:
        glyphh runtime token list
    """
    try:
        config = load_env_config(env_file)
    except ConfigurationError as e:
        click.echo(click.style("Configuration Error: ", fg="red") + str(e))
        sys.exit(1)
    
    client = RuntimeAPIClient(config)
    
    try:
        tokens = client.list_tokens()
        
        if not tokens:
            click.echo("No webhook tokens found.")
            return
        
        # Display tokens in a table format
        click.echo()
        click.echo(f"{'ID':<12} {'Model':<20} {'Created':<20} {'Status':<10}")
        click.echo("-" * 62)
        
        for t in tokens:
            token_id = t.id[:10] if t.id else 'N/A'
            model = (t.model or 'N/A')[:18]
            created = (t.created_at or 'N/A')[:18]
            status = t.status
            
            status_color = 'green' if status == 'Active' else 'red'
            click.echo(
                f"{token_id:<12} {model:<20} {created:<20} " +
                click.style(f"{status:<10}", fg=status_color)
            )
        
        click.echo()
        click.echo(f"Total: {len(tokens)} token(s)")
        
    except AuthenticationError:
        click.echo(
            click.style("Error: ", fg="red") +
            "Authentication required to list tokens."
        )
        sys.exit(1)
    except RuntimeAPIError as e:
        click.echo(click.style(f"Error: ", fg="red") + str(e))
        sys.exit(1)


@token.command('revoke')
@click.argument('token_id')
@click.option(
    '--env-file',
    default=None,
    help='Path to .env configuration file'
)
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompt')
def token_revoke(token_id: str, env_file: Optional[str], yes: bool):
    """
    Revoke a webhook token.
    
    \b
    Example:
        glyphh runtime token revoke abc123
    """
    if not yes:
        click.confirm(
            f"Are you sure you want to revoke token '{token_id}'?",
            abort=True
        )
    
    try:
        config = load_env_config(env_file)
    except ConfigurationError as e:
        click.echo(click.style("Configuration Error: ", fg="red") + str(e))
        sys.exit(1)
    
    client = RuntimeAPIClient(config)
    
    try:
        client.revoke_token(token_id)
        click.echo(click.style("[OK] ", fg="green") + f"Token '{token_id}' revoked successfully")
    except AuthenticationError:
        click.echo(
            click.style("Error: ", fg="red") +
            "Authentication required to revoke tokens."
        )
        sys.exit(1)
    except NotFoundError:
        click.echo(
            click.style("Error: ", fg="red") +
            f"Token '{token_id}' not found."
        )
        sys.exit(1)
    except RuntimeAPIError as e:
        click.echo(click.style(f"Error: ", fg="red") + str(e))
        sys.exit(1)
