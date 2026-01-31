"""
Authentication Service for Glyphh Runtime.

Handles JWT token validation, authorization checks, and security weight computation.
Supports three deployment modes: local (no auth), self-hosted, and cloud.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Token
from infrastructure.config import get_settings
from shared.exceptions import (
    AuthenticationException,
    AuthorizationException,
)

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Permission levels for namespace operations."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class User:
    """Authenticated user with permissions."""
    user_id: str
    namespaces: Dict[str, Set[Permission]]  # namespace -> permissions
    org_id: Optional[str] = None
    email: Optional[str] = None
    token_type: str = "jwt"  # jwt or webhook
    
    def has_permission(self, namespace: str, permission: Permission) -> bool:
        """Check if user has permission for namespace."""
        if namespace not in self.namespaces:
            return False
        return permission in self.namespaces[namespace]
    
    def can_read(self, namespace: str) -> bool:
        """Check if user can read from namespace."""
        return self.has_permission(namespace, Permission.READ)
    
    def can_write(self, namespace: str) -> bool:
        """Check if user can write to namespace."""
        return self.has_permission(namespace, Permission.WRITE)
    
    def is_admin(self, namespace: str) -> bool:
        """Check if user is admin for namespace."""
        return self.has_permission(namespace, Permission.ADMIN)


class AuthService:
    """
    Authentication and authorization service.
    
    Responsibilities:
    - Validate JWT tokens (deployment and consumer tokens)
    - Validate webhook tokens
    - Check namespace-level permissions
    - Compute security weights for glyph filtering
    """
    
    def __init__(self, session: Optional[AsyncSession] = None):
        """
        Initialize AuthService.
        
        Args:
            session: Optional database session for webhook token validation
        """
        self._session = session
        self._settings = get_settings()

    async def validate_token(self, token: str) -> User:
        """
        Validate a token and return the authenticated user.
        
        Supports both JWT tokens and webhook tokens.
        
        Args:
            token: Bearer token (JWT or webhook token)
            
        Returns:
            Authenticated User object
            
        Raises:
            AuthenticationException: If token is invalid or expired
        """
        # Local mode: skip authentication
        if self._settings.deployment_mode == "local":
            logger.debug("Local mode: bypassing authentication")
            return self._create_local_user()
        
        if not token:
            raise AuthenticationException("No token provided")
        
        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]
        
        # Try JWT validation first
        try:
            return await self._validate_jwt_token(token)
        except jwt.InvalidTokenError:
            pass
        
        # Try webhook token validation
        if self._session:
            try:
                return await self._validate_webhook_token(token)
            except AuthenticationException:
                pass
        
        raise AuthenticationException("Invalid or expired token")
    
    async def _validate_jwt_token(self, token: str) -> User:
        """
        Validate a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Authenticated User object
            
        Raises:
            jwt.InvalidTokenError: If token is invalid
            AuthenticationException: If token is expired or malformed
        """
        if not self._settings.jwt_secret_key:
            raise AuthenticationException("JWT authentication not configured")
        
        try:
            payload = jwt.decode(
                token,
                self._settings.jwt_secret_key,
                algorithms=[self._settings.jwt_algorithm],
            )
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationException(f"Invalid token: {e}")
        
        # Extract user info from payload
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationException("Token missing 'sub' claim")
        
        # Parse namespaces and permissions
        namespaces = self._parse_namespace_permissions(payload)
        
        return User(
            user_id=user_id,
            namespaces=namespaces,
            org_id=payload.get("org_id"),
            email=payload.get("email"),
            token_type="jwt",
        )
    
    async def _validate_webhook_token(self, token: str) -> User:
        """
        Validate a webhook token against the database.
        
        Args:
            token: Webhook token string
            
        Returns:
            Authenticated User object
            
        Raises:
            AuthenticationException: If token is invalid or revoked
        """
        if not self._session:
            raise AuthenticationException("Webhook token validation not available")
        
        # Hash the token for lookup
        token_hash = self._hash_token(token)
        
        # Look up token in database
        result = await self._session.execute(
            select(Token).where(
                Token.token_hash == token_hash,
                Token.status == "active",
            )
        )
        db_token = result.scalar_one_or_none()
        
        if not db_token:
            raise AuthenticationException("Invalid webhook token")
        
        # Check expiration
        if db_token.expires_at and db_token.expires_at < datetime.utcnow():
            raise AuthenticationException("Webhook token has expired")
        
        # Build permissions from token
        permissions = {Permission(p) for p in db_token.permissions}
        
        # If token is namespace-scoped, only grant access to that namespace
        if db_token.namespace:
            namespaces = {db_token.namespace: permissions}
        else:
            # Global token - grant access to all namespaces
            namespaces = {"*": permissions}
        
        return User(
            user_id=f"webhook:{db_token.id}",
            namespaces=namespaces,
            token_type="webhook",
        )
    
    def _parse_namespace_permissions(
        self,
        payload: Dict[str, Any]
    ) -> Dict[str, Set[Permission]]:
        """
        Parse namespace permissions from JWT payload.
        
        Expected payload format:
        {
            "namespaces": ["ns1", "ns2"],
            "permissions": {
                "ns1": ["read", "write"],
                "ns2": ["read"]
            }
        }
        
        Or simplified format:
        {
            "namespace": "ns1",
            "permissions": ["read", "write"]
        }
        
        Args:
            payload: JWT payload dict
            
        Returns:
            Dict mapping namespace to set of permissions
        """
        namespaces: Dict[str, Set[Permission]] = {}
        
        # Check for simplified single-namespace format
        if "namespace" in payload and isinstance(payload.get("permissions"), list):
            ns = payload["namespace"]
            perms = {Permission(p) for p in payload["permissions"] if p in Permission.__members__.values()}
            namespaces[ns] = perms
            return namespaces
        
        # Check for multi-namespace format
        ns_list = payload.get("namespaces", [])
        perm_dict = payload.get("permissions", {})
        
        for ns in ns_list:
            if ns in perm_dict:
                perms = {Permission(p) for p in perm_dict[ns] if p in Permission.__members__.values()}
            else:
                # Default to read-only if namespace listed but no permissions specified
                perms = {Permission.READ}
            namespaces[ns] = perms
        
        # If no namespaces specified, check for global permissions
        if not namespaces and "permissions" in payload:
            if isinstance(payload["permissions"], list):
                perms = {Permission(p) for p in payload["permissions"] if p in Permission.__members__.values()}
                namespaces["*"] = perms
        
        return namespaces
    
    def _create_local_user(self) -> User:
        """Create a user with full permissions for local mode."""
        return User(
            user_id="local",
            namespaces={"*": {Permission.READ, Permission.WRITE, Permission.ADMIN}},
            token_type="local",
        )
    
    def _hash_token(self, token: str) -> str:
        """Hash a token for secure storage/lookup."""
        return hashlib.sha256(token.encode()).hexdigest()

    async def check_namespace_access(
        self,
        user: User,
        namespace: str,
        operation: str,
    ) -> bool:
        """
        Check if user has permission for operation on namespace.
        
        Args:
            user: Authenticated user
            namespace: Target namespace
            operation: Operation type (read, write, admin)
            
        Returns:
            True if authorized
            
        Raises:
            AuthorizationException: If not authorized
        """
        # Map operation to permission
        permission_map = {
            "read": Permission.READ,
            "write": Permission.WRITE,
            "admin": Permission.ADMIN,
            "search": Permission.READ,
            "create": Permission.WRITE,
            "update": Permission.WRITE,
            "delete": Permission.WRITE,
            "deploy": Permission.ADMIN,
            "unload": Permission.ADMIN,
            "config": Permission.ADMIN,
        }
        
        required_permission = permission_map.get(operation, Permission.READ)
        
        # Check for wildcard access
        if "*" in user.namespaces:
            if required_permission in user.namespaces["*"]:
                return True
        
        # Check specific namespace access
        if user.has_permission(namespace, required_permission):
            return True
        
        # Admin permission implies all other permissions
        if user.is_admin(namespace):
            return True
        
        # Write permission implies read
        if required_permission == Permission.READ and user.can_write(namespace):
            return True
        
        raise AuthorizationException(
            user_id=user.user_id,
            namespace=namespace,
            operation=operation,
        )
    
    def compute_security_weight(
        self,
        user: User,
        glyph_metadata: Dict[str, Any],
    ) -> float:
        """
        Compute security weight for a glyph based on user permissions.
        
        Security weight is used to filter or de-rank glyphs that the user
        doesn't have full access to.
        
        Args:
            user: Authenticated user
            glyph_metadata: Glyph metadata containing security info
            
        Returns:
            Security weight (0.0 to 1.0)
            - 1.0 = full access
            - 0.0 = no access (should be filtered)
            - 0.5 = partial access (de-ranked in results)
        """
        # Get glyph security level from metadata
        security_level = glyph_metadata.get("security_level", 0)
        required_clearance = glyph_metadata.get("required_clearance", [])
        
        # Local mode: full access
        if user.token_type == "local":
            return 1.0
        
        # Admin users get full access
        if "*" in user.namespaces and Permission.ADMIN in user.namespaces["*"]:
            return 1.0
        
        # Check clearance requirements
        user_clearances = set(user.namespaces.keys())
        if required_clearance:
            if not any(c in user_clearances for c in required_clearance):
                return 0.0  # No access
        
        # Compute weight based on security level
        # Higher security levels require higher permissions
        if security_level == 0:
            return 1.0  # Public
        elif security_level == 1:
            return 0.8 if Permission.READ in user.namespaces.get("*", set()) else 0.5
        elif security_level == 2:
            return 0.6 if Permission.WRITE in user.namespaces.get("*", set()) else 0.3
        else:
            return 0.4 if Permission.ADMIN in user.namespaces.get("*", set()) else 0.1
    
    def require_auth(self) -> bool:
        """Check if authentication is required for current deployment mode."""
        return self._settings.deployment_mode != "local"
