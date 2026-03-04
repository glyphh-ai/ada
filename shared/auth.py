"""
Authentication middleware for Glyphh Runtime.

Validates API tokens (glyphh_xxxx) via SHA-256 hash lookup in the database.
Local mode bypasses all auth for development convenience.

No JWT — all API auth uses database-backed tokens created via POST /setup
or the glyphh token create CLI command.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from infrastructure.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class AuthenticatedUser:
    """Authenticated user context extracted from a database token."""

    user_id: str
    org_id: str
    role: str
    plan: str = "free"


# HTTP Bearer security scheme (auto_error=False to handle missing tokens manually)
security = HTTPBearer(auto_error=False)


async def _validate_db_token(token: str) -> AuthenticatedUser:
    """Validate a token by SHA-256 hash lookup in the database."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    try:
        from infrastructure.database import async_session_maker
        from sqlalchemy import select
        from domains.models.db_models import Token

        async with async_session_maker() as session:
            result = await session.execute(
                select(Token).where(
                    Token.token_hash == token_hash,
                    Token.status == "active",
                )
            )
            db_token = result.scalar_one_or_none()

            if not db_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or revoked token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Check expiry
            if db_token.expires_at and db_token.expires_at < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Determine role from permissions
            permissions = db_token.permissions or ["read"]
            role = "admin" if "admin" in permissions else "service"

            return AuthenticatedUser(
                user_id=f"token:{db_token.id}",
                org_id=db_token.org_id,
                role=role,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AuthenticatedUser:
    """
    Validate token and extract user context.

    In local mode, returns a mock admin user.
    In cloud/self-hosted mode, requires a valid database token.
    """
    settings = get_settings()

    # Local mode: skip auth
    if settings.deployment_mode == "local":
        path_org_id = request.path_params.get("org_id", "local-dev-org")
        return AuthenticatedUser(
            user_id="local-dev-user",
            org_id=path_org_id,
            role="admin",
            plan="pro",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token. Create one with: glyphh token create",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await _validate_db_token(credentials.credentials)


async def require_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AuthenticatedUser:
    """
    Require a valid API token for data and query endpoints.

    In local mode, bypasses token validation.
    In cloud/self-hosted mode, validates via database lookup.
    """
    settings = get_settings()

    if settings.deployment_mode == "local":
        path_org_id = request.path_params.get("org_id", "local-dev-org")
        return AuthenticatedUser(
            user_id="local-dev-user",
            org_id=path_org_id,
            role="admin",
            plan="pro",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Create a token with: glyphh token create",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await _validate_db_token(credentials.credentials)


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[AuthenticatedUser]:
    """Optional authentication - returns None if no valid token."""
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None
