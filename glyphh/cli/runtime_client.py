"""
Runtime API client for communicating with Glyphh Runtime servers.

This module provides a clean interface for all Runtime API operations:
- Model deployment
- Status checking
- Token management
- Log retrieval

The client handles authentication, error handling, and response parsing.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from glyphh.cli.config import RuntimeConfig


class RuntimeAPIError(Exception):
    """Base exception for Runtime API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class AuthenticationError(RuntimeAPIError):
    """Raised when authentication fails (401)."""
    pass


class AuthorizationError(RuntimeAPIError):
    """Raised when authorization fails (403)."""
    pass


class NotFoundError(RuntimeAPIError):
    """Raised when a resource is not found (404)."""
    pass


class ConnectionError(RuntimeAPIError):
    """Raised when connection to the runtime fails."""
    pass


@dataclass
class DeploymentResult:
    """Result of a successful model deployment."""
    model_id: str
    org_id: Optional[str] = None
    version: Optional[str] = None
    mcp_endpoint: Optional[str] = None
    listener_endpoint: Optional[str] = None
    webhook_token: Optional[str] = None
    raw_response: Optional[Dict] = None
    
    @classmethod
    def from_response(cls, data: Dict) -> "DeploymentResult":
        """Create DeploymentResult from API response."""
        endpoints = data.get("endpoints", {})
        return cls(
            model_id=data.get("model_id", ""),
            org_id=data.get("org_id"),
            version=data.get("version"),
            mcp_endpoint=endpoints.get("mcp") or data.get("mcp_endpoint"),
            listener_endpoint=endpoints.get("listener") or data.get("webhook_url"),
            webhook_token=data.get("webhook_token"),
            raw_response=data
        )


@dataclass
class RuntimeStatus:
    """Status information from the runtime server."""
    online: bool
    version: Optional[str] = None
    models_loaded: Optional[int] = None
    uptime: Optional[str] = None
    raw_response: Optional[Dict] = None
    
    @classmethod
    def from_response(cls, data: Dict) -> "RuntimeStatus":
        """Create RuntimeStatus from API response."""
        return cls(
            online=True,
            version=data.get("version"),
            models_loaded=data.get("models_loaded"),
            uptime=data.get("uptime"),
            raw_response=data
        )


@dataclass
class TokenInfo:
    """Information about a webhook token."""
    id: str
    model: Optional[str] = None
    created_at: Optional[str] = None
    status: str = "Active"
    
    @classmethod
    def from_response(cls, data: Dict) -> "TokenInfo":
        """Create TokenInfo from API response."""
        return cls(
            id=data.get("id", ""),
            model=data.get("model"),
            created_at=data.get("created_at"),
            status=data.get("status", "Active")
        )


@dataclass
class ModelInfo:
    """Information about a deployed model."""
    model_id: str
    name: Optional[str] = None
    version: Optional[str] = None
    org_id: Optional[str] = None
    deployed_at: Optional[str] = None
    status: str = "Active"
    
    @classmethod
    def from_response(cls, data: Dict) -> "ModelInfo":
        """Create ModelInfo from API response."""
        return cls(
            model_id=data.get("model_id", data.get("id", "")),
            name=data.get("name"),
            version=data.get("version"),
            org_id=data.get("org_id"),
            deployed_at=data.get("deployed_at"),
            status=data.get("status", "Active")
        )


class RuntimeAPIClient:
    """
    Client for communicating with Glyphh Runtime API.
    
    This client provides methods for all Runtime API operations including
    model deployment, status checking, token management, and log retrieval.
    
    Example:
        config = load_env_config()
        client = RuntimeAPIClient(config)
        
        # Deploy a model
        result = client.deploy_model("my_model.glyphh")
        print(f"Deployed to: {result.mcp_endpoint}")
        
        # Check status
        status = client.get_status()
        print(f"Runtime version: {status.version}")
    """
    
    def __init__(self, config: RuntimeConfig):
        """
        Initialize the Runtime API client.
        
        Args:
            config: RuntimeConfig with connection details
        """
        self.config = config
        self.base_url = config.runtime_url.rstrip("/")
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[bytes] = None,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        content_type: Optional[str] = None
    ) -> requests.Response:
        """
        Make an HTTP request to the Runtime API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., "/api/deploy")
            data: Raw bytes data for the request body
            json_data: JSON data for the request body
            params: Query parameters
            content_type: Content-Type header value
        
        Returns:
            Response object
        
        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails (401)
            AuthorizationError: If authorization fails (403)
            NotFoundError: If resource not found (404)
            RuntimeAPIError: For other API errors
        """
        url = f"{self.base_url}{endpoint}"
        headers = self.config.get_auth_header()
        
        if content_type:
            headers["Content-Type"] = content_type
        
        try:
            response = requests.request(
                method=method,
                url=url,
                data=data,
                json=json_data,
                params=params,
                headers=headers,
                timeout=self.config.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to {self.base_url}. "
                "Make sure the runtime server is running.",
                details={"original_error": str(e)}
            )
        except requests.exceptions.Timeout:
            raise RuntimeAPIError(
                f"Request timed out after {self.config.timeout} seconds.",
                details={"timeout": self.config.timeout}
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeAPIError(f"Request failed: {e}")
        
        # Handle error responses
        if response.status_code == 401:
            raise AuthenticationError(
                "Invalid or missing JWT token. "
                "Get a token from https://platform.glyphh.com",
                status_code=401
            )
        elif response.status_code == 403:
            raise AuthorizationError(
                "You don't have permission for this operation. "
                "Check your token permissions in the Platform UI.",
                status_code=403
            )
        elif response.status_code == 404:
            raise NotFoundError(
                f"Resource not found: {endpoint}",
                status_code=404
            )
        
        return response
    
    def _parse_json_response(self, response: requests.Response) -> Dict:
        """Parse JSON response, returning empty dict on failure."""
        try:
            return response.json()
        except json.JSONDecodeError:
            return {}
    
    def deploy_model(
        self,
        package_path: str,
        name: Optional[str] = None
    ) -> DeploymentResult:
        """
        Deploy a .glyphh model to the Runtime.
        
        Args:
            package_path: Path to the .glyphh package file
            name: Optional deployment name (defaults to model name)
        
        Returns:
            DeploymentResult with deployment details and endpoints
        
        Raises:
            RuntimeAPIError: If deployment fails
            FileNotFoundError: If package file doesn't exist
        """
        path = Path(package_path)
        if not path.exists():
            raise FileNotFoundError(f"Package file not found: {package_path}")
        
        # Read package data
        with open(path, "rb") as f:
            package_data = f.read()
        
        # Prepare params
        params = {}
        if name:
            params["name"] = name
        
        # Make request
        response = self._request(
            method="POST",
            endpoint="/api/deploy",
            data=package_data,
            params=params if params else None,
            content_type="application/octet-stream"
        )
        
        if response.status_code not in (200, 201):
            error_data = self._parse_json_response(response)
            raise RuntimeAPIError(
                error_data.get("error", f"Deployment failed: HTTP {response.status_code}"),
                status_code=response.status_code,
                details=error_data
            )
        
        data = self._parse_json_response(response)
        return DeploymentResult.from_response(data)
    
    def get_status(self) -> RuntimeStatus:
        """
        Get the status of the Runtime server.
        
        Returns:
            RuntimeStatus with server information
        
        Raises:
            ConnectionError: If server is offline
            RuntimeAPIError: If status check fails
        """
        response = self._request(method="GET", endpoint="/api/status")
        
        if response.status_code != 200:
            raise RuntimeAPIError(
                f"Status check failed: HTTP {response.status_code}",
                status_code=response.status_code
            )
        
        data = self._parse_json_response(response)
        return RuntimeStatus.from_response(data)
    
    def list_models(self) -> List[ModelInfo]:
        """
        List all deployed models.
        
        Returns:
            List of ModelInfo objects
        
        Raises:
            RuntimeAPIError: If request fails
        """
        response = self._request(method="GET", endpoint="/api/models")
        
        if response.status_code != 200:
            raise RuntimeAPIError(
                f"Failed to list models: HTTP {response.status_code}",
                status_code=response.status_code
            )
        
        data = self._parse_json_response(response)
        models = data.get("models", [])
        return [ModelInfo.from_response(m) for m in models]
    
    def get_logs(self, lines: int = 100) -> List[str]:
        """
        Get runtime server logs.
        
        Args:
            lines: Number of log lines to retrieve
        
        Returns:
            List of log lines
        
        Raises:
            RuntimeAPIError: If request fails
        """
        response = self._request(
            method="GET",
            endpoint="/api/logs",
            params={"lines": lines}
        )
        
        if response.status_code != 200:
            raise RuntimeAPIError(
                f"Failed to get logs: HTTP {response.status_code}",
                status_code=response.status_code
            )
        
        data = self._parse_json_response(response)
        return data.get("logs", [])
    
    def list_tokens(self) -> List[TokenInfo]:
        """
        List all webhook tokens.
        
        Returns:
            List of TokenInfo objects
        
        Raises:
            RuntimeAPIError: If request fails
        """
        response = self._request(method="GET", endpoint="/api/tokens")
        
        if response.status_code != 200:
            raise RuntimeAPIError(
                f"Failed to list tokens: HTTP {response.status_code}",
                status_code=response.status_code
            )
        
        data = self._parse_json_response(response)
        tokens = data.get("tokens", [])
        return [TokenInfo.from_response(t) for t in tokens]
    
    def revoke_token(self, token_id: str) -> bool:
        """
        Revoke a webhook token.
        
        Args:
            token_id: ID of the token to revoke
        
        Returns:
            True if revocation was successful
        
        Raises:
            NotFoundError: If token doesn't exist
            RuntimeAPIError: If revocation fails
        """
        response = self._request(
            method="DELETE",
            endpoint=f"/api/tokens/{token_id}"
        )
        
        if response.status_code in (200, 204):
            return True
        
        raise RuntimeAPIError(
            f"Failed to revoke token: HTTP {response.status_code}",
            status_code=response.status_code
        )
