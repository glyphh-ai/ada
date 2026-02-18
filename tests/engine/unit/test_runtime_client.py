"""
Unit tests for Runtime API client.

Tests cover:
- RuntimeAPIClient methods (deploy, status, tokens, logs)
- Error handling (authentication, authorization, connection)
- Response parsing
- JWT token inclusion in headers
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from glyphh.cli.config import RuntimeConfig
from glyphh.cli.runtime_client import (
    RuntimeAPIClient,
    RuntimeAPIError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ConnectionError,
    DeploymentResult,
    RuntimeStatus,
    TokenInfo,
    ModelInfo,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def local_config():
    """Create a local development config."""
    return RuntimeConfig(runtime_url="http://localhost:8000")


@pytest.fixture
def remote_config():
    """Create a remote config with JWT token."""
    return RuntimeConfig(
        runtime_url="https://runtime.glyphh.com",
        jwt_token="test_jwt_token_123"
    )


@pytest.fixture
def client(local_config):
    """Create a RuntimeAPIClient with local config."""
    return RuntimeAPIClient(local_config)


@pytest.fixture
def auth_client(remote_config):
    """Create a RuntimeAPIClient with remote config and auth."""
    return RuntimeAPIClient(remote_config)


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {}
    return response


# ============================================================================
# DeploymentResult Tests
# ============================================================================

class TestDeploymentResult:
    """Test DeploymentResult dataclass."""
    
    def test_from_response_minimal(self):
        """Test creating DeploymentResult from minimal response."""
        data = {"model_id": "model_123"}
        result = DeploymentResult.from_response(data)
        
        assert result.model_id == "model_123"
        assert result.org_id is None
        assert result.version is None
        assert result.mcp_endpoint is None
        assert result.listener_endpoint is None
        assert result.webhook_token is None
    
    def test_from_response_full(self):
        """Test creating DeploymentResult from full response."""
        data = {
            "model_id": "model_123",
            "org_id": "org_456",
            "version": "1.0.0",
            "endpoints": {
                "mcp": "https://runtime.glyphh.com/org_456/model_123/mcp",
                "listener": "https://runtime.glyphh.com/org_456/model_123/listener"
            },
            "webhook_token": "token_789"
        }
        result = DeploymentResult.from_response(data)
        
        assert result.model_id == "model_123"
        assert result.org_id == "org_456"
        assert result.version == "1.0.0"
        assert result.mcp_endpoint == "https://runtime.glyphh.com/org_456/model_123/mcp"
        assert result.listener_endpoint == "https://runtime.glyphh.com/org_456/model_123/listener"
        assert result.webhook_token == "token_789"
    
    def test_from_response_legacy_format(self):
        """Test creating DeploymentResult from legacy response format."""
        data = {
            "model_id": "model_123",
            "mcp_endpoint": "http://localhost:8000/mcp",
            "webhook_url": "http://localhost:8000/listener"
        }
        result = DeploymentResult.from_response(data)
        
        assert result.mcp_endpoint == "http://localhost:8000/mcp"
        assert result.listener_endpoint == "http://localhost:8000/listener"


class TestRuntimeStatus:
    """Test RuntimeStatus dataclass."""
    
    def test_from_response(self):
        """Test creating RuntimeStatus from response."""
        data = {
            "version": "1.2.3",
            "models_loaded": 5,
            "uptime": "2d 3h 45m"
        }
        status = RuntimeStatus.from_response(data)
        
        assert status.online is True
        assert status.version == "1.2.3"
        assert status.models_loaded == 5
        assert status.uptime == "2d 3h 45m"


class TestTokenInfo:
    """Test TokenInfo dataclass."""
    
    def test_from_response(self):
        """Test creating TokenInfo from response."""
        data = {
            "id": "token_123",
            "model": "my_model",
            "created_at": "2024-01-15T10:30:00Z",
            "status": "Active"
        }
        token = TokenInfo.from_response(data)
        
        assert token.id == "token_123"
        assert token.model == "my_model"
        assert token.created_at == "2024-01-15T10:30:00Z"
        assert token.status == "Active"


# ============================================================================
# RuntimeAPIClient Tests
# ============================================================================

class TestRuntimeAPIClientInit:
    """Test RuntimeAPIClient initialization."""
    
    def test_init_with_local_config(self, local_config):
        """Test initializing client with local config."""
        client = RuntimeAPIClient(local_config)
        
        assert client.config == local_config
        assert client.base_url == "http://localhost:8000"
    
    def test_init_with_remote_config(self, remote_config):
        """Test initializing client with remote config."""
        client = RuntimeAPIClient(remote_config)
        
        assert client.config == remote_config
        assert client.base_url == "https://runtime.glyphh.com"
    
    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from URL."""
        config = RuntimeConfig(runtime_url="http://localhost:8000/")
        client = RuntimeAPIClient(config)
        
        assert client.base_url == "http://localhost:8000"


class TestRuntimeAPIClientRequest:
    """Test RuntimeAPIClient._request method."""
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_request_includes_auth_header(self, mock_request, auth_client, mock_response):
        """Test that request includes Authorization header when token is set."""
        mock_request.return_value = mock_response
        
        auth_client._request("GET", "/api/status")
        
        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test_jwt_token_123"
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_request_no_auth_header_without_token(self, mock_request, client, mock_response):
        """Test that request has no Authorization header without token."""
        mock_request.return_value = mock_response
        
        client._request("GET", "/api/status")
        
        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        assert "Authorization" not in call_kwargs["headers"]
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_request_connection_error(self, mock_request, client):
        """Test that connection error is raised properly."""
        import requests
        mock_request.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        with pytest.raises(ConnectionError) as exc_info:
            client._request("GET", "/api/status")
        
        assert "Could not connect" in str(exc_info.value)
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_request_timeout_error(self, mock_request, client):
        """Test that timeout error is raised properly."""
        import requests
        mock_request.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with pytest.raises(RuntimeAPIError) as exc_info:
            client._request("GET", "/api/status")
        
        assert "timed out" in str(exc_info.value)
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_request_401_raises_authentication_error(self, mock_request, client):
        """Test that 401 response raises AuthenticationError."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_request.return_value = mock_response
        
        with pytest.raises(AuthenticationError):
            client._request("GET", "/api/status")
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_request_403_raises_authorization_error(self, mock_request, client):
        """Test that 403 response raises AuthorizationError."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_request.return_value = mock_response
        
        with pytest.raises(AuthorizationError):
            client._request("GET", "/api/status")
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_request_404_raises_not_found_error(self, mock_request, client):
        """Test that 404 response raises NotFoundError."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response
        
        with pytest.raises(NotFoundError):
            client._request("GET", "/api/models/nonexistent")


class TestRuntimeAPIClientDeployModel:
    """Test RuntimeAPIClient.deploy_model method."""
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_deploy_model_success(self, mock_request, client):
        """Test successful model deployment."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "model_id": "model_123",
            "org_id": "org_456",
            "endpoints": {
                "mcp": "http://localhost:8000/mcp",
                "listener": "http://localhost:8000/listener"
            }
        }
        mock_request.return_value = mock_response
        
        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.glyphh') as f:
            f.write(b"test model data")
            temp_path = f.name
        
        try:
            result = client.deploy_model(temp_path)
            
            assert result.model_id == "model_123"
            assert result.org_id == "org_456"
            assert result.mcp_endpoint == "http://localhost:8000/mcp"
        finally:
            Path(temp_path).unlink()
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_deploy_model_with_name(self, mock_request, client):
        """Test deployment with custom name."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model_id": "custom_name"}
        mock_request.return_value = mock_response
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.glyphh') as f:
            f.write(b"test")
            temp_path = f.name
        
        try:
            client.deploy_model(temp_path, name="custom_name")
            
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["params"]["name"] == "custom_name"
        finally:
            Path(temp_path).unlink()
    
    def test_deploy_model_file_not_found(self, client):
        """Test deployment with non-existent file."""
        with pytest.raises(FileNotFoundError):
            client.deploy_model("/nonexistent/model.glyphh")
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_deploy_model_error_response(self, mock_request, client):
        """Test deployment error handling."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Invalid model format"}
        mock_request.return_value = mock_response
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.glyphh') as f:
            f.write(b"invalid")
            temp_path = f.name
        
        try:
            with pytest.raises(RuntimeAPIError) as exc_info:
                client.deploy_model(temp_path)
            
            assert "Invalid model format" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()


class TestRuntimeAPIClientGetStatus:
    """Test RuntimeAPIClient.get_status method."""
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_get_status_success(self, mock_request, client):
        """Test successful status check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "version": "1.0.0",
            "models_loaded": 3,
            "uptime": "1d 2h"
        }
        mock_request.return_value = mock_response
        
        status = client.get_status()
        
        assert status.online is True
        assert status.version == "1.0.0"
        assert status.models_loaded == 3
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_get_status_error(self, mock_request, client):
        """Test status check error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_request.return_value = mock_response
        
        with pytest.raises(RuntimeAPIError):
            client.get_status()


class TestRuntimeAPIClientListTokens:
    """Test RuntimeAPIClient.list_tokens method."""
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_list_tokens_success(self, mock_request, client):
        """Test successful token listing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tokens": [
                {"id": "token_1", "model": "model_a", "status": "Active"},
                {"id": "token_2", "model": "model_b", "status": "Revoked"}
            ]
        }
        mock_request.return_value = mock_response
        
        tokens = client.list_tokens()
        
        assert len(tokens) == 2
        assert tokens[0].id == "token_1"
        assert tokens[0].model == "model_a"
        assert tokens[1].status == "Revoked"
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_list_tokens_empty(self, mock_request, client):
        """Test listing tokens when none exist."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"tokens": []}
        mock_request.return_value = mock_response
        
        tokens = client.list_tokens()
        
        assert tokens == []


class TestRuntimeAPIClientRevokeToken:
    """Test RuntimeAPIClient.revoke_token method."""
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_revoke_token_success(self, mock_request, client):
        """Test successful token revocation."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_request.return_value = mock_response
        
        result = client.revoke_token("token_123")
        
        assert result is True
        mock_request.assert_called_once()
        assert "/api/tokens/token_123" in mock_request.call_args[1]["url"]
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_revoke_token_not_found(self, mock_request, client):
        """Test revoking non-existent token."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response
        
        with pytest.raises(NotFoundError):
            client.revoke_token("nonexistent")


class TestRuntimeAPIClientGetLogs:
    """Test RuntimeAPIClient.get_logs method."""
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_get_logs_success(self, mock_request, client):
        """Test successful log retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "logs": [
                "2024-01-15 10:00:00 INFO Starting server",
                "2024-01-15 10:00:01 INFO Model loaded"
            ]
        }
        mock_request.return_value = mock_response
        
        logs = client.get_logs(lines=50)
        
        assert len(logs) == 2
        assert "Starting server" in logs[0]
        
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["params"]["lines"] == 50


class TestRuntimeAPIClientListModels:
    """Test RuntimeAPIClient.list_models method."""
    
    @patch('glyphh.cli.runtime_client.requests.request')
    def test_list_models_success(self, mock_request, client):
        """Test successful model listing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"model_id": "model_1", "name": "Model One", "version": "1.0.0"},
                {"model_id": "model_2", "name": "Model Two", "version": "2.0.0"}
            ]
        }
        mock_request.return_value = mock_response
        
        models = client.list_models()
        
        assert len(models) == 2
        assert models[0].model_id == "model_1"
        assert models[0].name == "Model One"
        assert models[1].version == "2.0.0"
