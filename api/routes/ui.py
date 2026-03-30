"""
Web UI API routes.

Lightweight endpoints for the browser dashboard:
- Session info (auth state)
- Device auth proxy (forwards to Platform to avoid CORS)
- After successful device auth, mints a local database token so the UI
  validates locally (no Platform round-trips on every API call).
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Token
from infrastructure.database import get_db
from shared.auth import get_optional_user, AuthenticatedUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui", tags=["web-ui"])

PLATFORM_URL = "https://api.glyphh.ai/api/v1"


def _extract_org_id_from_jwt(token: str) -> str | None:
    """Extract org_id from a Platform JWT payload without signature verification.

    The JWT was just returned by the Platform, so we trust the payload.
    """
    import base64, json as _json
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = _json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("org_id")
    except Exception:
        return None

try:
    from importlib.metadata import version as _pkg_version
    _RUNTIME_VERSION = _pkg_version("glyphh")
except Exception:
    _RUNTIME_VERSION = "0.0.0"


@router.get("/session")
async def get_session(
    current_user: AuthenticatedUser | None = Depends(get_optional_user),
) -> dict[str, Any]:
    """Return current session state for the UI login gate."""
    if current_user is None:
        return {
            "authenticated": False,
            "version": _RUNTIME_VERSION,
        }
    return {
        "authenticated": True,
        "user_id": str(current_user.user_id),
        "org_id": str(current_user.org_id) if current_user.org_id else None,
        "role": current_user.role,
        "version": _RUNTIME_VERSION,
    }


# ── Device auth proxy ───────────────────────────────────────────────────────
# The browser can't call the Platform directly (CORS), so the runtime
# proxies these two endpoints.  Same flow the CLI uses.


@router.post("/auth/device/start")
async def device_start() -> dict[str, Any]:
    """Proxy to Platform: start device authorization flow."""
    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.post(f"{PLATFORM_URL}/auth/device/start")
        res.raise_for_status()
        return res.json()


class DevicePollRequest(BaseModel):
    device_code: str


@router.post("/auth/device/poll")
async def device_poll(
    request: DevicePollRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Proxy to Platform: poll for device authorization approval.

    On approval, mints a local database token (glyphh_xxxx) and returns it
    as `runtime_token` alongside the Platform JWT.  The UI stores and uses
    the local token for all subsequent API calls — no Platform round-trips.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.post(
            f"{PLATFORM_URL}/auth/device/poll",
            json={"device_code": request.device_code},
        )
        res.raise_for_status()
        data = res.json()

    # On approval, mint a local database token for the UI
    if data.get("status") == "approved":
        org_id = None
        user = data.get("user", {})
        if user:
            org_id = user.get("org_id")

        if org_id:
            try:
                raw_token = f"glyphh_{secrets.token_urlsafe(32)}"
                token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
                db_token = Token(
                    name="ui-session",
                    token_hash=token_hash,
                    token_prefix=raw_token[:12],
                    org_id=org_id,
                    permissions=["read", "write", "admin"],
                    expires_at=datetime.utcnow() + timedelta(days=365),
                )
                db.add(db_token)
                await db.flush()
                data["runtime_token"] = raw_token
            except Exception as e:
                logger.warning("Failed to mint UI runtime token: %s", e)

    return data


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/auth/refresh")
async def refresh_token(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Proxy to Platform: refresh an expired access token.

    Also mints a new local database token so the UI continues using
    fast local validation after refresh.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.post(
            f"{PLATFORM_URL}/auth/refresh",
            json={"refresh_token": request.refresh_token},
        )
        res.raise_for_status()
        data = res.json()

    # Mint a new local token — extract org_id from the refreshed JWT payload
    # (Platform already validated the refresh token, so the JWT is trusted)
    if data.get("access_token"):
        try:
            org_id = _extract_org_id_from_jwt(data["access_token"])
            if org_id:
                raw_token = f"glyphh_{secrets.token_urlsafe(32)}"
                token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
                db_token = Token(
                    name="ui-session",
                    token_hash=token_hash,
                    token_prefix=raw_token[:12],
                    org_id=org_id,
                    permissions=["read", "write", "admin"],
                    expires_at=datetime.utcnow() + timedelta(days=365),
                )
                db.add(db_token)
                await db.flush()
                data["runtime_token"] = raw_token
        except Exception as e:
            logger.warning("Failed to mint runtime token on refresh: %s", e)

    return data
