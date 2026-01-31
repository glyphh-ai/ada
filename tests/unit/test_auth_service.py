"""
Unit tests for AuthService.
"""

import pytest
import jwt
from datetime import datetime, timedelta

from domains.auth.service import AuthService
from shared.exceptions import AuthenticationException, AuthorizationException


class TestAuthService:
    """Tests for AuthService authentication and authorization."""
    
    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance for testing."""
        return AuthService(
            jwt_secret_key="test_secret_key",
            jwt_algorithm="HS256",
            deployment_mode="self-hosted",
        )
    
    @pytest.fixture
    def local_auth_service(self):
        """Create AuthService in local mode."""
        return AuthService(
            jwt_secret_key=None,
            jwt_algorithm="HS256",
            deployment_mode="local",
        )
    
    @pytest.fixture
    def valid_token(self):
        """Create a valid JWT token."""
        payload = {
            "sub": "test_user",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=1),
            "namespaces": ["test_namespace"],
            "permissions": ["read", "write"],
        }
        return jwt.encode(payload, "test_secret_key", algorithm="HS256")
    
    @pytest.fixture
    def expired_token(self):
        """Create an expired JWT token."""
        payload = {
            "sub": "test_user",
            "iat": datetime.utcnow() - timedelta(hours=2),
            "exp": datetime.utcnow() - timedelta(hours=1),
            "namespaces": ["test_namespace"],
            "permissions": ["read"],
        }
        return jwt.encode(payload, "test_secret_key", algorithm="HS256")
    
    @pytest.mark.asyncio
    async def test_validate_token_success(self, auth_service, valid_token):
        """Test successful token validation."""
        claims = await auth_service.validate_token(valid_token)
        
        assert claims["sub"] == "test_user"
        assert "test_namespace" in claims["namespaces"]
    
    @pytest.mark.asyncio
    async def test_validate_token_expired(self, auth_service, expired_token):
        """Test that expired token raises AuthenticationException."""
        with pytest.raises(AuthenticationException) as exc_info:
            await auth_service.validate_token(expired_token)
        
        assert "expired" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_validate_token_invalid_signature(self, auth_service):
        """Test that invalid signature raises AuthenticationException."""
        # Token signed with different key
        payload = {
            "sub": "test_user",
            "exp": datetime.utcnow() + timedelta(hours=1),
        }
        bad_token = jwt.encode(payload, "wrong_key", algorithm="HS256")
        
        with pytest.raises(AuthenticationException):
            await auth_service.validate_token(bad_token)
    
    @pytest.mark.asyncio
    async def test_validate_token_malformed(self, auth_service):
        """Test that malformed token raises AuthenticationException."""
        with pytest.raises(AuthenticationException):
            await auth_service.validate_token("not.a.valid.token")
    
    @pytest.mark.asyncio
    async def test_local_mode_bypass(self, local_auth_service):
        """Test that local mode bypasses authentication."""
        # Should not raise even with invalid token
        claims = await local_auth_service.validate_token("any_token")
        
        assert claims is not None
        assert claims.get("local_mode") is True
    
    @pytest.mark.asyncio
    async def test_check_namespace_access_allowed(self, auth_service, valid_token):
        """Test namespace access check when allowed."""
        claims = await auth_service.validate_token(valid_token)
        
        # Should not raise
        await auth_service.check_namespace_access(claims, "test_namespace")
    
    @pytest.mark.asyncio
    async def test_check_namespace_access_denied(self, auth_service, valid_token):
        """Test namespace access check when denied."""
        claims = await auth_service.validate_token(valid_token)
        
        with pytest.raises(AuthorizationException):
            await auth_service.check_namespace_access(claims, "other_namespace")
    
    @pytest.mark.asyncio
    async def test_check_permission_allowed(self, auth_service, valid_token):
        """Test permission check when allowed."""
        claims = await auth_service.validate_token(valid_token)
        
        # Should not raise
        await auth_service.check_permission(claims, "read")
        await auth_service.check_permission(claims, "write")
    
    @pytest.mark.asyncio
    async def test_check_permission_denied(self, auth_service, valid_token):
        """Test permission check when denied."""
        claims = await auth_service.validate_token(valid_token)
        
        with pytest.raises(AuthorizationException):
            await auth_service.check_permission(claims, "admin")
    
    def test_compute_security_weight_full_access(self, auth_service):
        """Test security weight computation with full access."""
        claims = {
            "namespaces": ["test_namespace"],
            "security_level": 1.0,
        }
        glyph_metadata = {"security_level": 0.5}
        
        weight = auth_service.compute_security_weight(
            claims, "test_namespace", glyph_metadata
        )
        
        assert weight == 1.0
    
    def test_compute_security_weight_no_access(self, auth_service):
        """Test security weight computation with no access."""
        claims = {
            "namespaces": ["other_namespace"],
            "security_level": 0.5,
        }
        glyph_metadata = {"security_level": 0.8}
        
        weight = auth_service.compute_security_weight(
            claims, "test_namespace", glyph_metadata
        )
        
        assert weight == 0.0
    
    def test_compute_security_weight_insufficient_level(self, auth_service):
        """Test security weight when user level is insufficient."""
        claims = {
            "namespaces": ["test_namespace"],
            "security_level": 0.3,
        }
        glyph_metadata = {"security_level": 0.8}
        
        weight = auth_service.compute_security_weight(
            claims, "test_namespace", glyph_metadata
        )
        
        assert weight == 0.0
