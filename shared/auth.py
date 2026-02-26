"""
JWT authentication middleware for Runtime.

Validates JWT tokens issued by Platform and extracts user context.
Supports local mode bypass for development convenience on model lifecycle
endpoints (deploy, undeploy, status). Data and query endpoints (listener,
MCP) always require a valid token.
"""

from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from infrastructure.config import get_settings


@dataclass
class AuthenticatedUser:
    """Authenticated user context extracted from JWT."""
    
    user_id: str
    org_id: str
    role: str
    plan: str = "free"


# HTTP Bearer security scheme (auto_error=False to handle missing tokens manually)
security = HTTPBearer(auto_error=False)


def _validate_jwt(credentials: HTTPAuthorizationCredentials) -> AuthenticatedUser:
    """Validate a JWT token and return the authenticated user.

    Shared logic used by both get_current_user and require_token.
    """
    settings = get_settings()

    if not settings.jwt_secret_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT_SECRET_KEY not configured. Set it to use token-secured endpoints.",
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )

        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_id = payload.get("sub")
        org_id = payload.get("org_id")
        role = payload.get("role", "user")
        plan = payload.get("plan", "free")

        if not user_id or not org_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing required claims",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return AuthenticatedUser(
            user_id=str(user_id),
            org_id=str(org_id),
            role=str(role),
            plan=str(plan),
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AuthenticatedUser:
    """
    Validate JWT and extract user context.
    
    In local mode, returns a mock user for development convenience.
    In cloud/self-hosted mode, requires valid JWT token.
    
    Used for model lifecycle endpoints (deploy, undeploy, status, re-encode).
    """
    settings = get_settings()
    
    # Local mode: skip auth for development convenience on lifecycle endpoints
    if settings.deployment_mode == "local":
        return AuthenticatedUser(
            user_id="local-dev-user",
            org_id="local-dev-org",
            role="admin",
            plan="free",
        )
    
    # Non-local mode: require valid JWT
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return _validate_jwt(credentials)


async def require_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AuthenticatedUser:
    """
    Require a valid API token for data and query endpoints (listener, MCP).

    In local mode, bypasses token validation for zero-friction development.
    In cloud/self-hosted mode, validates tokens by SHA-256 hash lookup in the
    runtime database, falling back to JWT for platform-issued Studio tokens.
    """
    settings = get_settings()

    # Local mode: skip token requirement for development convenience
    if settings.deployment_mode == "local":
        return AuthenticatedUser(
            user_id="local-dev-user",
            org_id="local-dev-org",
            role="admin",
            plan="free",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Create a token with: glyphh token create --name <name>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Try DB token lookup first (API tokens created via glyphh token create)
    import hashlib
    from datetime import datetime
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

            if db_token:
                # Check expiry
                if db_token.expires_at and db_token.expires_at < datetime.utcnow():
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has expired",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                return AuthenticatedUser(
                    user_id=f"token:{db_token.id}",
                    org_id=db_token.org_id,
                    role="service",
                )
    except HTTPException:
        raise
    except Exception as e:
        # DB not available — fall through to JWT
        import logging
        logging.getLogger(__name__).debug(f"DB token lookup failed: {e}")

    # Fall back to JWT validation (for platform-issued tokens / Studio)
    return _validate_jwt(credentials)


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[AuthenticatedUser]:
    """
    Optional authentication - returns None if no valid token.
    
    Useful for endpoints that work differently for authenticated vs anonymous users.
    """
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None
