"""
Authentication CLI commands - login, logout, signup, whoami
"""

import os
import json
import click
import requests
from pathlib import Path
from typing import Optional

from .ui import print_error, print_success, print_warning, print_connection_error


def _styled_prompt(label: str, **kwargs) -> str:
    """Prompt with cyan label."""
    prompt_text = click.style(f"  {label}", fg="cyan") + click.style(">", fg="white")
    return click.prompt(prompt_text, **kwargs)

# Config directory for storing credentials
GLYPHH_DIR = Path.home() / ".glyphh"
CREDENTIALS_FILE = GLYPHH_DIR / "credentials.json"
CONFIG_FILE = GLYPHH_DIR / "config.json"

# Default API URL - use localhost for development
DEFAULT_API_URL = "http://localhost:8001"


def get_api_url() -> str:
    """Get the API URL from config or environment."""
    # Check environment first
    if url := os.environ.get("GLYPHH_API_URL"):
        return url
    
    # Check config file
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text())
            if url := config.get("api_url"):
                return url
        except:
            pass
    
    return DEFAULT_API_URL


def _get_redis_session_credentials() -> Optional[dict]:
    """
    Get credentials from Redis session (for web terminal).
    
    This is called when GLYPHH_SESSION_ID is set, indicating we're
    running in a web terminal with session stored in Redis.
    """
    session_id = os.environ.get("GLYPHH_SESSION_ID")
    if not session_id:
        return None
    
    # Call the platform API to get session credentials
    # This avoids needing Redis client in the SDK
    api_url = get_api_url()
    try:
        response = requests.get(
            f"{api_url}/api/v1/terminal/session/{session_id}",
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    return None


def save_credentials(access_token: str, refresh_token: str, user: dict):
    """Save credentials to disk."""
    GLYPHH_DIR.mkdir(parents=True, exist_ok=True)
    
    credentials = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": user,
    }
    
    CREDENTIALS_FILE.write_text(json.dumps(credentials, indent=2))
    # Secure the file
    os.chmod(CREDENTIALS_FILE, 0o600)


def load_credentials() -> Optional[dict]:
    """Load credentials from disk or Redis session."""
    # Check for web terminal session first
    if session_creds := _get_redis_session_credentials():
        return session_creds
    
    # Fall back to file-based credentials
    if not CREDENTIALS_FILE.exists():
        return None
    
    try:
        return json.loads(CREDENTIALS_FILE.read_text())
    except:
        return None


def clear_credentials():
    """Remove stored credentials."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()


def get_auth_headers() -> dict:
    """Get authorization headers if logged in."""
    creds = load_credentials()
    if creds and creds.get("access_token"):
        return {"Authorization": f"Bearer {creds['access_token']}"}
    return {}


def is_logged_in() -> bool:
    """Check if user is logged in."""
    creds = load_credentials()
    return creds is not None and creds.get("access_token") is not None


def is_web_session() -> bool:
    """Check if running in a web terminal session."""
    return os.environ.get("GLYPHH_WEB_TERMINAL") == "1"


def get_current_user() -> Optional[dict]:
    """Get current user info."""
    creds = load_credentials()
    if creds:
        return creds.get("user")
    return None


@click.group()
def auth():
    """Authentication commands - login, logout, signup, whoami."""
    pass


@auth.command()
@click.option("--email", "-e", default=None, help="Your email address")
@click.option("--password", "-p", default=None, help="Your password")
def login(email: Optional[str], password: Optional[str]):
    """Login to Glyphh."""
    # Check if already logged in via web session
    if is_web_session() and is_logged_in():
        user = get_current_user()
        if user:
            print_success(
                "Already Logged In",
                f"You're logged in as {user.get('email', 'unknown')} via your browser session."
            )
            click.echo("  To use a different account, log out from the web app first.")
            return
    
    # Prompt for missing values
    if email is None:
        email = _styled_prompt("email")
    if password is None:
        password = _styled_prompt("password", hide_input=True)
    api_url = get_api_url()
    
    click.echo(f"Logging in to {api_url}...")
    
    try:
        response = requests.post(
            f"{api_url}/api/v1/auth/login",
            json={"email": email, "password": password},
            timeout=30,
        )
        
        if response.status_code == 200:
            data = response.json()
            save_credentials(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                user=data["user"],
            )
            user = data["user"]
            print_success("Login Successful", f"Welcome back, {user.get('first_name', user['email'])}!")
            if user.get("org_id"):
                click.echo(f"  Organization: {user['org_id']}")
        elif response.status_code == 401:
            print_error(
                "Login Failed",
                "Invalid email or password.",
                suggestions=[
                    "Check your email address is correct",
                    "Reset your password with 'auth forgot-password'",
                    "Create a new account with 'auth signup'",
                ]
            )
        else:
            error = response.json().get("detail", "Login failed")
            print_error("Login Failed", error)
            
    except requests.exceptions.ConnectionError:
        print_connection_error(api_url)
    except Exception as e:
        print_error("Unexpected Error", str(e))


@auth.command()
def logout():
    """Logout from Glyphh."""
    if not is_logged_in():
        print_warning("Not Logged In", "You are not currently logged in.")
        return
    
    api_url = get_api_url()
    headers = get_auth_headers()
    
    try:
        # Call logout endpoint to invalidate token
        requests.post(
            f"{api_url}/api/v1/auth/logout",
            headers=headers,
            timeout=10,
        )
    except:
        pass  # Ignore errors, still clear local credentials
    
    clear_credentials()
    print_success("Logged Out", "You have been successfully logged out.")


@auth.command()
def whoami():
    """Show current user info."""
    user = get_current_user()
    
    if not user:
        click.echo("Not logged in")
        click.echo("Run 'auth login' to authenticate")
        return
    
    click.echo(f"Email:    {user.get('email', 'N/A')}")
    click.echo(f"Name:     {user.get('first_name', '')} {user.get('last_name', '')}")
    click.echo(f"Role:     {user.get('role', 'N/A')}")
    if user.get("org_id"):
        click.echo(f"Org ID:   {user['org_id']}")


@auth.command()
@click.option("--email", "-e", default=None, help="Your email address")
def signup(email: Optional[str]):
    """Start the signup process."""
    # Check if already logged in via web session
    if is_web_session() and is_logged_in():
        user = get_current_user()
        if user:
            print_warning(
                "Already Logged In",
                f"You're already logged in as {user.get('email', 'unknown')}."
            )
            click.echo("  To create a new account, log out from the web app first.")
            return
    
    # Prompt for missing values
    if email is None:
        email = _styled_prompt("email")
    api_url = get_api_url()
    
    click.echo(f"Starting registration for {email}...")
    
    try:
        # Step 1: Register email
        response = requests.post(
            f"{api_url}/api/v1/auth/register",
            json={"email": email},
            timeout=30,
        )
        
        if response.status_code != 200:
            error = response.json().get("detail", "Registration failed")
            print_error("Registration Failed", error)
            return
        
        print_success("Code Sent", f"Verification code sent to {email}")
        
        # Step 2: Verify code
        code = _styled_prompt("code")
        
        response = requests.post(
            f"{api_url}/api/v1/auth/verify-code",
            json={"email": email, "code": code},
            timeout=30,
        )
        
        if response.status_code != 200:
            error = response.json().get("detail", "Invalid code")
            print_error("Verification Failed", error)
            return
        
        print_success("Email Verified", "Your email has been verified!")
        
        # Step 3: Complete profile
        click.echo()
        first_name = _styled_prompt("first name")
        last_name = _styled_prompt("last name")
        password = _styled_prompt("password", hide_input=True, confirmation_prompt=True)
        org_name = _styled_prompt("org name", default="", show_default=False)
        
        response = requests.post(
            f"{api_url}/api/v1/auth/complete-profile",
            json={
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
                "password": password,
                "org_name": org_name or None,
            },
            timeout=30,
        )
        
        if response.status_code == 200:
            data = response.json()
            save_credentials(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", ""),
                user=data["user"],
            )
            print_success("Welcome to Glyphh!", f"Account created for {first_name} {last_name}")
        else:
            error = response.json().get("detail", "Registration failed")
            print_error("Registration Failed", error)
            
    except requests.exceptions.ConnectionError:
        print_connection_error(api_url)
    except Exception as e:
        print_error("Unexpected Error", str(e))


@auth.command()
@click.argument("url")
def set_api(url: str):
    """Set the API URL (for self-hosted deployments)."""
    GLYPHH_DIR.mkdir(parents=True, exist_ok=True)
    
    config = {}
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text())
        except:
            pass
    
    config["api_url"] = url.rstrip("/")
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    
    click.secho(f"✓ API URL set to {url}", fg="green")


@auth.command()
def status():
    """Show authentication status."""
    api_url = get_api_url()
    user = get_current_user()
    
    click.echo(f"API URL:  {api_url}")
    
    if user:
        click.secho("Status:   Authenticated", fg="green")
        click.echo(f"User:     {user.get('email', 'N/A')}")
    else:
        click.secho("Status:   Not authenticated", fg="yellow")
        click.echo("Run 'auth login' to authenticate")
