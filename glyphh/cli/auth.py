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
import platform
import time
import uuid
import webbrowser
from pathlib import Path

import click
import httpx

from . import theme

GLYPHH_DIR = Path.home() / ".glyphh"
CONFIG_FILE = GLYPHH_DIR / "config.json"

PLATFORM_URL = os.environ.get("GLYPHH_PLATFORM_URL", "https://api.glyphh.ai/api/v1")


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
    """Return stored access token, refreshing automatically if expired.

    Decodes the JWT exp claim (without verification) to check expiry.
    If expired and a refresh_token exists, calls Platform to get a new
    access token and persists it.
    """
    config = _load_config()
    token = config.get("access_token")
    if not token:
        return None

    # Check if expired
    if _is_token_expired(token):
        refreshed = _try_refresh(config)
        if refreshed:
            return refreshed
        # Refresh failed — return expired token (server will reject it,
        # user will need to re-login)
    return token


def _is_token_expired(token: str) -> bool:
    """Check JWT expiry without signature verification."""
    try:
        import json, base64
        # Decode payload (second segment)
        payload_b64 = token.split(".")[1]
        # Add padding
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp", 0)
        return time.time() > exp
    except Exception:
        return False  # Can't check — assume valid


def _try_refresh(config: dict) -> str | None:
    """Attempt to refresh the access token using the stored refresh token."""
    refresh_token = config.get("refresh_token")
    if not refresh_token:
        return None

    api_url = get_api_url()
    try:
        with httpx.Client(timeout=15) as client:
            res = client.post(
                f"{api_url}/auth/refresh",
                json={"refresh_token": refresh_token},
            )
        if res.status_code != 200:
            return None

        data = res.json()
        new_access = data.get("access_token")
        new_refresh = data.get("refresh_token")
        if not new_access:
            return None

        # Persist the new tokens
        config["access_token"] = new_access
        if new_refresh:
            config["refresh_token"] = new_refresh
        _save_config(config)
        return new_access
    except Exception:
        return None


def get_api_url() -> str:
    """Return the platform API URL (env override, or production default)."""
    return PLATFORM_URL


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
    """Remove stored auth data and runtime token.

    Keeps runtime_id and runtime_name — those are machine-level
    registrations that count toward the org's plan limit. Deleting
    them causes a new registration on every login cycle.
    """
    config = _load_config()
    for key in ("access_token", "refresh_token", "user", "runtime_token"):
        config.pop(key, None)
    _save_config(config)

    # Remove license file so re-login fetches a fresh one
    from glyphh.licensing import LICENSE_FILE
    if LICENSE_FILE.exists():
        try:
            LICENSE_FILE.unlink()
        except Exception:
            pass


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
                    register_runtime()
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
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        click.echo()
        if detail:
            click.secho(f"  {detail}", fg=theme.WARNING)
        else:
            click.secho(f"  Login failed: HTTP {e.response.status_code}", fg=theme.ERROR)
        click.echo()
        return False
    except Exception as e:
        click.secho(f"  Login failed: {e}", fg=theme.ERROR)
        return False


def _get_machine_id() -> str:
    """Generate a stable machine identifier."""
    config = _load_config()
    if mid := config.get("machine_id"):
        return mid
    mid = str(uuid.uuid4())
    config["machine_id"] = mid
    _save_config(config)
    return mid


def register_runtime() -> bool:
    """Register this CLI instance as a runtime on the platform.

    Each runtime counts toward the org's license.
    """
    api_url = get_api_url()
    token = get_token()
    if not token:
        return False

    machine_id = _get_machine_id()
    hostname = platform.node() or "unknown"
    name = f"{hostname}-{machine_id[:8]}"

    # Check if already registered
    config = _load_config()
    existing_id = config.get("runtime_id")

    if existing_id:
        from glyphh.licensing import LICENSE_FILE
        if LICENSE_FILE.exists():
            return True

        # Registered but missing license — fetch it (don't create a new runtime)
        try:
            with httpx.Client(timeout=15) as client:
                res = client.get(
                    f"{api_url}/runtimes/{existing_id}/license",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if res.status_code == 200:
                    license_info = res.json()
                    jwt_token = license_info.get("token")
                    if jwt_token:
                        from glyphh.licensing import save_license_token
                        save_license_token(jwt_token)
                        tier = license_info.get("tier", "free")
                        click.secho(f"  License refreshed ({tier} tier)", fg=theme.MUTED)
                        return True
                # If fetch failed (404 = deleted on platform), fall through to create
        except Exception:
            pass

    try:
        with httpx.Client(timeout=15) as client:
            res = client.post(
                f"{api_url}/runtimes",
                json={"name": name},
                headers={"Authorization": f"Bearer {token}"},
            )
            if res.status_code in (200, 201):
                data = res.json()
                runtime_id = data.get("id")
                config["runtime_id"] = runtime_id
                config["runtime_name"] = name
                _save_config(config)
                click.secho(f"  Runtime registered: {name}", fg=theme.MUTED)

                # Fetch and save the signed JWT license
                license_info = data.get("license")
                if license_info and isinstance(license_info, dict):
                    jwt_token = license_info.get("token")
                    if jwt_token:
                        from glyphh.licensing import save_license_token
                        save_license_token(jwt_token)
                        tier = license_info.get("tier", "free")
                        click.secho(f"  License saved ({tier} tier, signed)", fg=theme.MUTED)

                # Print deployment instructions
                click.echo()
                click.secho("  To deploy this runtime remotely:", fg=theme.TEXT_DIM)
                click.secho(f"    GLYPHH_RUNTIME_ID={runtime_id}", fg=theme.ACCENT)
                click.echo()

                return True
            else:
                click.secho(f"  Runtime registration failed: {res.text}", fg=theme.WARNING)
                return False
    except Exception as e:
        click.secho(f"  Could not register runtime: {e}", fg=theme.WARNING)
        return False


def get_org_id() -> str | None:
    """Get the authenticated user's org_id from the stored session."""
    config = _load_config()
    return config.get("user", {}).get("org_id")


# Backward compat alias
def resolve_org_id(runtime_url: str | None = None) -> str | None:
    """Resolve org_id from stored session. Ignores runtime_url (legacy param)."""
    return get_org_id()


