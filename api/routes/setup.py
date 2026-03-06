"""
Runtime setup endpoint — license-authenticated, last-auth-wins.

POST /setup accepts a signed license JWT in the Authorization header,
verifies the Ed25519 signature, extracts org_id from the claims,
revokes old admin tokens for that org, and creates a fresh one.

No body required — org_id comes from the verified license.
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Token
from infrastructure.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["setup"])

security = HTTPBearer(auto_error=False)


class SetupResponse(BaseModel):
    token: str
    token_prefix: str
    org_id: str
    message: str


def _verify_license_jwt(token_str: str) -> dict | None:
    """Verify a license JWT and return claims, or None on failure."""
    try:
        from glyphh.licensing import _verify_token
        return _verify_token(token_str)
    except Exception as e:
        logger.warning(f"License verification failed: {e}")
        return None


@router.post("/setup", response_model=SetupResponse)
async def bootstrap_setup(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> SetupResponse:
    """Create an admin token authenticated by a signed license JWT.

    The license JWT (from Platform) is sent as a Bearer token.
    Last auth wins — old admin tokens for the org are revoked.
    """

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="License JWT required. Run: glyphh auth login",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify the license JWT signature
    claims = _verify_license_jwt(credentials.credentials)
    if claims is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or unverifiable license token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    org_id = claims.get("org_id", "default")

    # Update the runtime's loaded license to match the pushed tier
    from glyphh.licensing import set_current_license, _claims_to_license_info
    license_info = _claims_to_license_info(claims)
    set_current_license(license_info)
    request.app.state.license = license_info
    logger.info(f"License updated via /setup: tier={license_info.tier}, org={org_id}")

    # Revoke existing admin tokens for this org (last auth wins)
    result = await db.execute(
        select(Token).where(Token.org_id == org_id, Token.status == "active")
    )
    for old_token in result.scalars().all():
        if "admin" in (old_token.permissions or []):
            old_token.status = "revoked"
            old_token.revoked_at = datetime.utcnow()

    # Create new admin token
    raw_token = f"glyphh_{secrets.token_urlsafe(32)}"
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    token_prefix = raw_token[:12]

    db_token = Token(
        name="cli-admin",
        token_hash=token_hash,
        token_prefix=token_prefix,
        org_id=org_id,
        permissions=["read", "write", "admin"],
        expires_at=datetime.utcnow() + timedelta(days=365),
    )
    db.add(db_token)
    await db.flush()

    logger.info(f"Setup: created admin token for org '{org_id}' (old admin tokens revoked)")

    return SetupResponse(
        token=raw_token,
        token_prefix=token_prefix,
        org_id=org_id,
        message="Admin token created. Old admin tokens for this org have been revoked.",
    )
