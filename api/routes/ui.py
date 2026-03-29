"""
Web UI API routes.

Lightweight endpoints for the browser dashboard:
- Session info (deployment mode, auth state)
- Device auth proxy (forwards to Platform to avoid CORS)
"""

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from shared.auth import get_optional_user, AuthenticatedUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui", tags=["web-ui"])

PLATFORM_URL = "https://api.glyphh.ai/api/v1"

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
async def device_poll(request: DevicePollRequest) -> dict[str, Any]:
    """Proxy to Platform: poll for device authorization approval."""
    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.post(
            f"{PLATFORM_URL}/auth/device/poll",
            json={"device_code": request.device_code},
        )
        res.raise_for_status()
        return res.json()


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/auth/refresh")
async def refresh_token(request: RefreshRequest) -> dict[str, Any]:
    """Proxy to Platform: refresh an expired access token."""
    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.post(
            f"{PLATFORM_URL}/auth/refresh",
            json={"refresh_token": request.refresh_token},
        )
        res.raise_for_status()
        return res.json()
