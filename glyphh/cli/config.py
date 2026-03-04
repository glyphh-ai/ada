"""
CLI configuration management for Runtime API communication.

This module handles loading configuration from .env files for connecting
to Runtime APIs. It supports:
- RUNTIME_URL: The URL of the Runtime API server
- JWT_TOKEN: Optional JWT token for authentication

Configuration can be loaded from:
1. Default .env file in current directory
2. Custom .env file specified via --env-file flag
3. Environment variables directly
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv


# Default values for local development
DEFAULT_RUNTIME_URL = "http://localhost:8002"
DEFAULT_TIMEOUT = 30


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


@dataclass
class RuntimeConfig:
    """
    Configuration for connecting to a Runtime API.
    
    Attributes:
        runtime_url: URL of the Runtime API server
        jwt_token: Optional JWT token for authentication
        timeout: Request timeout in seconds
    """
    runtime_url: str
    jwt_token: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate the configuration values."""
        # Validate runtime_url
        if not self.runtime_url:
            raise ConfigurationError(
                "RUNTIME_URL is required. "
                "Set it in your .env file or environment variables.\n"
                "Example: RUNTIME_URL=http://localhost:8000"
            )
        
        # Validate URL format
        try:
            parsed = urlparse(self.runtime_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL format")
            if parsed.scheme not in ('http', 'https'):
                raise ValueError("URL scheme must be http or https")
        except Exception as e:
            raise ConfigurationError(
                f"Invalid RUNTIME_URL: {self.runtime_url}\n"
                f"Error: {e}\n"
                "Expected format: http://localhost:8000 or https://runtime.example.com"
            )
        
        # Validate timeout
        if self.timeout <= 0:
            raise ConfigurationError(
                f"Invalid RUNTIME_TIMEOUT: {self.timeout}\n"
                "Timeout must be a positive integer (seconds)"
            )
    
    @property
    def is_local(self) -> bool:
        """Check if this is a local development configuration."""
        parsed = urlparse(self.runtime_url)
        return parsed.hostname in ('localhost', '127.0.0.1', '::1')
    
    @property
    def requires_auth(self) -> bool:
        """Check if authentication is required (non-local deployments)."""
        return not self.is_local
    
    @property
    def has_auth(self) -> bool:
        """Check if authentication token is provided."""
        return self.jwt_token is not None and len(self.jwt_token) > 0
    
    def get_auth_header(self) -> dict:
        """Get the Authorization header for API requests."""
        if self.jwt_token:
            return {"Authorization": f"Bearer {self.jwt_token}"}
        return {}


def _read_dev_port() -> Optional[int]:
    """Read the port from ~/.glyphh/dev.port if a dev server is running."""
    port_file = Path.home() / ".glyphh" / "dev.port"
    pid_file = Path.home() / ".glyphh" / "dev.pid"
    try:
        if not port_file.exists():
            return None
        port = int(port_file.read_text().strip())
        # Validate: if there's a PID file, check the process is alive
        if pid_file.exists():
            import signal
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # check if process exists
        return port
    except (ValueError, ProcessLookupError, OSError):
        return None


def resolve_runtime_url(cli_override: Optional[str] = None) -> str:
    """Resolve the runtime URL using the priority chain:
    CLI flag → RUNTIME_URL env → ~/.glyphh/config.json → dev server port → localhost default.
    """
    if cli_override:
        return cli_override.rstrip("/")

    env_url = os.environ.get("RUNTIME_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")

    from .auth import _load_config
    config = _load_config()
    stored_url = config.get("runtime_url", "").strip()
    if stored_url:
        return stored_url.rstrip("/")

    # Check if a dev server is running on a non-default port
    dev_port = _read_dev_port()
    if dev_port:
        return f"http://localhost:{dev_port}"

    return DEFAULT_RUNTIME_URL


def resolve_runtime_token(cli_override: Optional[str] = None) -> Optional[str]:
    """Resolve the runtime auth token using the priority chain:
    CLI flag → GLYPHH_TOKEN env → config.json runtime_token → config.json access_token → None.
    """
    if cli_override:
        return cli_override

    env_token = os.environ.get("GLYPHH_TOKEN", "").strip()
    if env_token:
        return env_token

    from .auth import _load_config
    config = _load_config()

    runtime_token = config.get("runtime_token", "").strip()
    if runtime_token:
        return runtime_token

    access_token = config.get("access_token", "").strip()
    if access_token:
        return access_token

    return None


def load_env_config(env_file: Optional[str] = None) -> RuntimeConfig:
    """
    Load Runtime configuration from environment variables or .env file.
    
    Args:
        env_file: Optional path to a custom .env file. If not provided,
                  looks for .env in the current directory.
    
    Returns:
        RuntimeConfig with loaded values
    
    Raises:
        ConfigurationError: If required configuration is missing or invalid
    
    Example:
        # Load from default .env
        config = load_env_config()
        
        # Load from custom .env file
        config = load_env_config(".env.production")
    """
    # Load .env file if it exists
    if env_file:
        env_path = Path(env_file)
        if not env_path.exists():
            raise ConfigurationError(
                f"Environment file not found: {env_file}\n"
                "Create a .env file with your configuration or use 'runtime init' "
                "to generate a template."
            )
        load_dotenv(env_path)
    else:
        # Try to load from default .env in current directory
        default_env = Path(".env")
        if default_env.exists():
            load_dotenv(default_env)
    
    # Read configuration from environment
    runtime_url = os.getenv("RUNTIME_URL", DEFAULT_RUNTIME_URL)
    jwt_token = os.getenv("JWT_TOKEN")
    timeout_str = os.getenv("RUNTIME_TIMEOUT", str(DEFAULT_TIMEOUT))
    
    # Parse timeout
    try:
        timeout = int(timeout_str)
    except ValueError:
        raise ConfigurationError(
            f"Invalid RUNTIME_TIMEOUT: {timeout_str}\n"
            "Expected an integer value (seconds)"
        )
    
    # Create and validate config
    config = RuntimeConfig(
        runtime_url=runtime_url,
        jwt_token=jwt_token,
        timeout=timeout
    )
    
    # Warn if auth might be required but not provided
    if config.requires_auth and not config.has_auth:
        import click
        click.echo(
            click.style("Warning: ", fg="yellow") +
            "No JWT_TOKEN provided for remote runtime.\n"
            "   Authentication may be required. Get a token from the Platform UI.\n"
        )
    
    return config


def generate_env_template(output_path: str = ".env", scenario: str = "local") -> str:
    """
    Generate a .env template file for the specified scenario.
    
    Args:
        output_path: Path to write the .env file
        scenario: One of "local", "self-hosted", or "cloud"
    
    Returns:
        Path to the generated file
    
    Raises:
        ConfigurationError: If the scenario is invalid
    """
    templates = {
        "local": '''# Glyphh Runtime Configuration - Local Development
# ================================================
# This configuration is for local development on your machine.

# Runtime API URL (local development server)
RUNTIME_URL=http://localhost:8002

# JWT Token (optional for local development)
# JWT_TOKEN=

# Request timeout in seconds
RUNTIME_TIMEOUT=30
''',
        "self-hosted": '''# Glyphh Runtime Configuration - Self-Hosted
# ============================================
# This configuration is for self-hosted deployments (Heroku, AWS, etc.)

# Runtime API URL (your self-hosted runtime)
# Replace with your actual runtime URL
RUNTIME_URL=https://your-app.herokuapp.com

# JWT Token (required for self-hosted deployments)
# Get this token from the Glyphh Platform UI:
# 1. Log into https://platform.glyphh.com
# 2. Navigate to Settings → API Tokens
# 3. Generate a Runtime Token
JWT_TOKEN=your_jwt_token_here

# Request timeout in seconds
RUNTIME_TIMEOUT=30
''',
        "cloud": '''# Glyphh Runtime Configuration - Glyphh Cloud
# ============================================
# This configuration is for Glyphh Cloud (managed service)

# Runtime API URL (Glyphh Cloud)
RUNTIME_URL=https://runtime.glyphh.com

# JWT Token (required for Glyphh Cloud)
# Get this token from the Glyphh Platform UI:
# 1. Log into https://platform.glyphh.com
# 2. Navigate to Settings → API Tokens
# 3. Generate a Runtime Token
JWT_TOKEN=your_jwt_token_here

# Request timeout in seconds
RUNTIME_TIMEOUT=30
'''
    }
    
    if scenario not in templates:
        raise ConfigurationError(
            f"Invalid scenario: {scenario}\n"
            f"Valid options: {', '.join(templates.keys())}"
        )
    
    # Write the template
    output = Path(output_path)
    output.write_text(templates[scenario])
    
    return str(output)
