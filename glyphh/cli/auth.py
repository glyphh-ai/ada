"""
CLI authentication via device authorization flow.

On first run (no stored token), the CLI:
1. Calls POST /auth/device/start on the platform
2. Opens the browser to the verification URL
3. Polls POST /auth/device/poll until the user approves
4. Stores the token in ~/.glyphh/config.json
"""

import json
import os
import time
import webbrowser
from pathlib import Path

import click
import httpx

from . import theme

GLYPHH_DIR = Path.home() / ".glyphh"
CONFIG_FILE = GLYPHH_DIR / "config.json"

DEFAULT_PLATFORM_URL = os.getenv("GLYPHH_PLATFORM_URL", "http://localhost:8001/api/v1")


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_config(config: dict):
    GLYPHH_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_token() -> str | None:
    """Return stored access token, or None if not logged in."""
    return _load_config().get("access_token")


def get_api_url() -> str:
    """Return the platform API URL."""
    return _load_config().get("platform_url", DEFAULT_PLATFORM_URL)


def is_logged_in() -> bool:
    """Check if the CLI has a stored token."""
    return get_token() is not None


def save_session(access_token: str, refresh_token: str, user: dict):
    """Persist auth tokens and user info."""
    config = _load_config()
    config["access_token"] = access_token
    config["refresh_token"] = refresh_token
    config["user"] = user
    _save_config(config)


def clear_session():
    """Remove stored auth data."""
    config = _load_config()
    for key in ("access_token", "refresh_token", "user"):
        config.pop(key, None)
    _save_config(config)


def get_user() -> dict | None:
    """Return stored user info."""
    return _load_config().get("user")


def device_login() -> bool:
    """Run the device authorization flow. Returns True on success."""
    api_url = get_api_url()

    click.echo()
    click.secho("  Logging in...", fg=theme.MUTED)
    click.echo()

    try:
        # Step 1: Start device auth
        with httpx.Client(timeout=15) as client:
            res = client.post(f"{api_url}/auth/device/start")
            res.raise_for_status()
            data = res.json()

        device_code = data["device_code"]
        user_code = data["user_code"]
        verification_url = data["verification_url"]
        interval = data.get("interval", 5)
        expires_in = data.get("expires_in", 600)

        # Step 2: Show code and open browser
        click.secho("  Your code:", fg=theme.MUTED)
        click.echo()
        click.secho(f"    {user_code}", fg="bright_cyan", bold=True)
        click.echo()
        click.secho("  Opening browser...", fg=theme.MUTED)

        webbrowser.open(verification_url)

        click.secho("  Waiting for approval...", fg=theme.MUTED)
        click.echo()

        # Step 3: Poll until approved or expired
        deadline = time.time() + expires_in

        with httpx.Client(timeout=15) as client:
            while time.time() < deadline:
                time.sleep(interval)

                res = client.post(
                    f"{api_url}/auth/device/poll",
                    json={"device_code": device_code},
                )
                res.raise_for_status()
                poll = res.json()

                status = poll.get("status")

                if status == "approved":
                    save_session(
                        access_token=poll["access_token"],
                        refresh_token=poll["refresh_token"],
                        user=poll.get("user", {}),
                    )
                    user = poll.get("user", {})
                    name = user.get("first_name", user.get("email", ""))
                    click.echo()
                    click.secho(f"  Welcome, {name}!", fg="bright_cyan")
                    click.echo()
                    return True

                if status == "expired":
                    click.secho("  Code expired. Try again.", fg=theme.ERROR)
                    return False

                # still pending — keep polling

        click.secho("  Timed out waiting for approval.", fg=theme.ERROR)
        return False

    except httpx.ConnectError:
        click.echo()
        click.secho("  Could not reach the platform.", fg=theme.ERROR)
        click.secho("  Is the platform running?", fg=theme.MUTED)
        click.echo()
        return False
    except Exception as e:
        click.secho(f"  Login failed: {e}", fg=theme.ERROR)
        return False
