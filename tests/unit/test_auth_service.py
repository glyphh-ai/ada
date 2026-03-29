"""
Unit tests for AuthService.
"""

import pytest
import jwt
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from domains.auth.service import AuthService, User, Permission
from shared.exceptions import AuthenticationException, AuthorizationException


class TestAuthService:
    """Tests for AuthService authentication and authorization."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.jwt_secret_key = "test_secret_key"
        settings.jwt_algorithm = "HS256"
        return settings
    
    @pytest.fixture
    def auth_service(self, mock_settings):
        """Create AuthService instance for testing (no DB session)."""
        with patch("domains.auth.service.get_settings", return_value=mock_settings):
            return AuthService(session=None)

    @pytest.fixture
    def mock_session(self):
        """Create a mock async session that returns a valid token."""
        session = AsyncMock()
        db_token = MagicMock()
        db_token.id = "test_token_id"
        db_token.org_id = "test_org"
        db_token.permissions = ["read", "write"]
        db_token.expires_at = datetime.utcnow() + timedelta(hours=1)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = db_token
        session.execute = AsyncMock(return_value=mock_result)
        return session

    @pytest.fixture
    def db_auth_service(self, mock_settings, mock_session):
        """Create AuthService with mock database session."""
        with patch("domains.auth.service.get_settings", return_value=mock_settings):
            return AuthService(session=mock_session)

    @pytest.fixture
    def no_secret_settings(self):
        """Settings without JWT secret (Platform API validation path)."""
        settings = MagicMock()
        settings.jwt_secret_key = None
        settings.jwt_algorithm = "HS256"
        return settings
    
    @pytest.fixture
    def valid_token(self):
        """Create a valid JWT token with org_id."""
        payload = {
            "sub": "test_user",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=1),
            "org_id": "test_org",
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
            "org_id": "test_org",
            "permissions": ["read"],
        }
        return jwt.encode(payload, "test_secret_key", algorithm="HS256")
    
    @pytest.mark.asyncio
    async def test_validate_token_success(self, db_auth_service, valid_token):
        """Test successful token validation via database lookup."""
        user = await db_auth_service.validate_token(valid_token)

        assert user.user_id == "token:test_token_id"
        assert user.can_read("test_org")
        assert user.can_write("test_org")
    
    @pytest.mark.asyncio
    async def test_validate_token_expired(self, auth_service, expired_token):
        """Test that expired token raises AuthenticationException."""
        with pytest.raises(AuthenticationException) as exc_info:
            await auth_service.validate_token(expired_token)
        
        assert "expired" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_validate_token_invalid_signature(self, auth_service):
        """Test that invalid signature raises AuthenticationException."""
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
    async def test_no_secret_requires_platform_validation(self, no_secret_settings):
        """Test that without JWT secret, tokens must be validated via Platform API."""
        with patch("domains.auth.service.get_settings", return_value=no_secret_settings):
            service = AuthService(session=None)
        # Without jwt_secret_key, JWT decode will fail — token must go through
        # Platform API validation (tested in integration tests)
        with pytest.raises(AuthenticationException):
            await service.validate_token("some_unsigned_token")
    
    @pytest.mark.asyncio
    async def test_check_access_allowed(self, auth_service):
        """Test access check when allowed."""
        user = User(
            user_id="test_user",
            org_permissions={"test_org": {Permission.READ, Permission.WRITE}},
        )
        result = await auth_service.check_access(user, "test_org", "some_model", "read")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_access_denied(self, auth_service):
        """Test access check when denied (wrong org)."""
        user = User(
            user_id="test_user",
            org_permissions={"test_org": {Permission.READ, Permission.WRITE}},
        )
        with pytest.raises(AuthorizationException):
            await auth_service.check_access(user, "other_org", "some_model", "read")

    @pytest.mark.asyncio
    async def test_check_access_admin_denied(self, auth_service):
        """Test admin access check when user only has read/write."""
        user = User(
            user_id="test_user",
            org_permissions={"test_org": {Permission.READ, Permission.WRITE}},
        )
        with pytest.raises(AuthorizationException):
            await auth_service.check_access(user, "test_org", "some_model", "admin")


class TestUser:
    """Tests for User permission model."""
    
    def test_has_permission_specific_org(self):
        user = User(
            user_id="test",
            org_permissions={"org1": {Permission.READ, Permission.WRITE}},
        )
        assert user.has_permission("org1", Permission.READ)
        assert user.has_permission("org1", Permission.WRITE)
        assert not user.has_permission("org1", Permission.ADMIN)
        assert not user.has_permission("org2", Permission.READ)
    
    def test_has_permission_wildcard(self):
        user = User(
            user_id="test",
            org_permissions={"*": {Permission.READ, Permission.WRITE, Permission.ADMIN}},
        )
        assert user.can_read("any_org")
        assert user.can_write("any_org")
        assert user.is_admin("any_org")
    
    def test_compute_security_weight_admin_user(self):
        """Admin users get full security weight."""
        user = User(
            user_id="admin_user",
            token_type="jwt",
            org_permissions={"*": {Permission.READ, Permission.WRITE, Permission.ADMIN}},
        )

        with patch("domains.auth.service.get_settings") as mock:
            mock.return_value = MagicMock(jwt_secret_key="secret", jwt_algorithm="HS256")
            service = AuthService(session=None)

        weight = service.compute_security_weight(user, {"security_level": 3})
        assert weight == 1.0
