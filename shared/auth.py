"""
JWT authentication middleware for Runtime.

Validates JWT tokens issued by Platform and extracts user context.
Supports local mode bypass for development convenience.
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


# HTTP Bearer security scheme (auto_error=False to handle missing tokens manually)
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AuthenticatedUser:
    """
    Validate JWT and extract user context.
    
    In local mode, returns a mock user for development convenience.
    In cloud/self-hosted mode, requires valid JWT token.
    
    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials from Authorization header
        
    Returns:
        AuthenticatedUser with user_id, org_id, and role
        
    Raises:
        HTTPException: 401 if token is missing, expired, or invalid
    """
    settings = get_settings()
    
    # Local mode: skip auth for development
    if settings.deployment_mode == "local":
        return AuthenticatedUser(
            user_id="local-dev-user",
            org_id="local-dev-org",
            role="admin",
        )
    
    # Non-local mode: require valid JWT
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not settings.jwt_secret_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret key not configured",
        )
    
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        
        # Verify token type is "access"
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Extract required claims
        user_id = payload.get("sub")
        org_id = payload.get("org_id")
        role = payload.get("role", "user")
        
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


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[AuthenticatedUser]:
    """
    Optional authentication - returns None if no valid token.
    
    Useful for endpoints that work differently for authenticated vs anonymous users.
    
    Args:
        request: FastAPI request object
        credentials: HTTP Bearer credentials from Authorization header
        
    Returns:
        AuthenticatedUser if valid token present, None otherwise
    """
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None
