"""
Unit tests for CLI configuration loading (.env files).

Tests cover:
- RuntimeConfig validation (URL format, timeout, auth)
- load_env_config() function
- generate_env_template() function
- Error handling for missing/invalid configuration
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from glyphh.cli.config import (
    RuntimeConfig,
    ConfigurationError,
    load_env_config,
    generate_env_template,
    DEFAULT_RUNTIME_URL,
    DEFAULT_TIMEOUT,
)


# ============================================================================
# RuntimeConfig Validation Tests
# ============================================================================

class TestRuntimeConfigValidation:
    """Test RuntimeConfig validation logic."""
    
    def test_valid_local_config(self):
        """Test creating config for local development."""
        config = RuntimeConfig(runtime_url="http://localhost:8000")
        
        assert config.runtime_url == "http://localhost:8000"
        assert config.jwt_token is None
        assert config.timeout == DEFAULT_TIMEOUT
        assert config.is_local is True
        assert config.requires_auth is False
        assert config.has_auth is False
    
    def test_valid_remote_config_with_token(self):
        """Test creating config for remote deployment with JWT."""
        config = RuntimeConfig(
            runtime_url="https://runtime.glyphh.com",
            jwt_token="test_token_123"
        )
        
        assert config.runtime_url == "https://runtime.glyphh.com"
        assert config.jwt_token == "test_token_123"
        assert config.is_local is False
        assert config.requires_auth is True
        assert config.has_auth is True
    
    def test_valid_config_with_custom_timeout(self):
        """Test creating config with custom timeout."""
        config = RuntimeConfig(
            runtime_url="http://localhost:8000",
            timeout=60
        )
        
        assert config.timeout == 60
    
    def test_invalid_empty_url(self):
        """Test that empty URL raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            RuntimeConfig(runtime_url="")
        
        assert "RUNTIME_URL is required" in str(exc_info.value)
    
    def test_invalid_url_format_no_scheme(self):
        """Test that URL without scheme raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            RuntimeConfig(runtime_url="localhost:8000")
        
        assert "Invalid RUNTIME_URL" in str(exc_info.value)
    
    def test_invalid_url_format_no_host(self):
        """Test that URL without host raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            RuntimeConfig(runtime_url="http://")
        
        assert "Invalid RUNTIME_URL" in str(exc_info.value)
    
    def test_invalid_url_scheme(self):
        """Test that non-http(s) scheme raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            RuntimeConfig(runtime_url="ftp://localhost:8000")
        
        assert "Invalid RUNTIME_URL" in str(exc_info.value)
        assert "http or https" in str(exc_info.value)
    
    def test_invalid_timeout_zero(self):
        """Test that zero timeout raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            RuntimeConfig(runtime_url="http://localhost:8000", timeout=0)
        
        assert "Invalid RUNTIME_TIMEOUT" in str(exc_info.value)
        assert "positive" in str(exc_info.value)
    
    def test_invalid_timeout_negative(self):
        """Test that negative timeout raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            RuntimeConfig(runtime_url="http://localhost:8000", timeout=-10)
        
        assert "Invalid RUNTIME_TIMEOUT" in str(exc_info.value)


class TestRuntimeConfigProperties:
    """Test RuntimeConfig property methods."""
    
    def test_is_local_localhost(self):
        """Test is_local for localhost."""
        config = RuntimeConfig(runtime_url="http://localhost:8000")
        assert config.is_local is True
    
    def test_is_local_127_0_0_1(self):
        """Test is_local for 127.0.0.1."""
        config = RuntimeConfig(runtime_url="http://127.0.0.1:8000")
        assert config.is_local is True
    
    def test_is_local_ipv6_loopback(self):
        """Test is_local for IPv6 loopback."""
        config = RuntimeConfig(runtime_url="http://[::1]:8000")
        assert config.is_local is True
    
    def test_is_local_remote(self):
        """Test is_local for remote URL."""
        config = RuntimeConfig(runtime_url="https://runtime.glyphh.com")
        assert config.is_local is False
    
    def test_requires_auth_local(self):
        """Test requires_auth for local config."""
        config = RuntimeConfig(runtime_url="http://localhost:8000")
        assert config.requires_auth is False
    
    def test_requires_auth_remote(self):
        """Test requires_auth for remote config."""
        config = RuntimeConfig(runtime_url="https://runtime.glyphh.com")
        assert config.requires_auth is True
    
    def test_has_auth_no_token(self):
        """Test has_auth without token."""
        config = RuntimeConfig(runtime_url="http://localhost:8000")
        assert config.has_auth is False
    
    def test_has_auth_empty_token(self):
        """Test has_auth with empty token."""
        config = RuntimeConfig(runtime_url="http://localhost:8000", jwt_token="")
        assert config.has_auth is False
    
    def test_has_auth_with_token(self):
        """Test has_auth with valid token."""
        config = RuntimeConfig(
            runtime_url="http://localhost:8000",
            jwt_token="test_token"
        )
        assert config.has_auth is True
    
    def test_get_auth_header_no_token(self):
        """Test get_auth_header without token."""
        config = RuntimeConfig(runtime_url="http://localhost:8000")
        assert config.get_auth_header() == {}
    
    def test_get_auth_header_with_token(self):
        """Test get_auth_header with token."""
        config = RuntimeConfig(
            runtime_url="http://localhost:8000",
            jwt_token="test_token_123"
        )
        header = config.get_auth_header()
        assert header == {"Authorization": "Bearer test_token_123"}


# ============================================================================
# load_env_config Tests
# ============================================================================

class TestLoadEnvConfig:
    """Test load_env_config function."""
    
    def test_load_default_without_env_file(self):
        """Test loading config without .env file uses defaults."""
        # Clear any existing env vars
        with patch.dict(os.environ, {}, clear=True):
            config = load_env_config()
            
            assert config.runtime_url == DEFAULT_RUNTIME_URL
            assert config.jwt_token is None
            assert config.timeout == DEFAULT_TIMEOUT
    
    def test_load_from_env_vars(self):
        """Test loading config from environment variables."""
        env_vars = {
            "RUNTIME_URL": "https://test.example.com",
            "JWT_TOKEN": "env_token_123",
            "RUNTIME_TIMEOUT": "45"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_env_config()
            
            assert config.runtime_url == "https://test.example.com"
            assert config.jwt_token == "env_token_123"
            assert config.timeout == 45
    
    def test_load_from_env_file(self):
        """Test loading config from .env file."""
        env_content = """
RUNTIME_URL=https://file.example.com
JWT_TOKEN=file_token_456
RUNTIME_TIMEOUT=60
"""
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.env'
        ) as f:
            f.write(env_content)
            temp_path = f.name
        
        try:
            # Clear env vars to ensure we're reading from file
            with patch.dict(os.environ, {}, clear=True):
                config = load_env_config(temp_path)
                
                assert config.runtime_url == "https://file.example.com"
                assert config.jwt_token == "file_token_456"
                assert config.timeout == 60
        finally:
            os.remove(temp_path)
    
    def test_load_from_nonexistent_file(self):
        """Test that loading from non-existent file raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_env_config("/nonexistent/path/.env")
        
        assert "not found" in str(exc_info.value)
    
    def test_load_invalid_timeout(self):
        """Test that invalid timeout in env raises error."""
        env_vars = {
            "RUNTIME_URL": "http://localhost:8000",
            "RUNTIME_TIMEOUT": "not_a_number"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                load_env_config()
            
            assert "Invalid RUNTIME_TIMEOUT" in str(exc_info.value)
    
    def test_load_partial_config(self):
        """Test loading config with only some values set."""
        env_vars = {
            "RUNTIME_URL": "https://partial.example.com"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_env_config()
            
            assert config.runtime_url == "https://partial.example.com"
            assert config.jwt_token is None
            assert config.timeout == DEFAULT_TIMEOUT


# ============================================================================
# generate_env_template Tests
# ============================================================================

class TestGenerateEnvTemplate:
    """Test generate_env_template function."""
    
    def test_generate_local_template(self):
        """Test generating local development template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, ".env")
            result = generate_env_template(output_path, "local")
            
            assert result == output_path
            assert os.path.exists(output_path)
            
            content = Path(output_path).read_text()
            assert "http://localhost:8000" in content
            assert "Local Development" in content
    
    def test_generate_self_hosted_template(self):
        """Test generating self-hosted template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, ".env")
            result = generate_env_template(output_path, "self-hosted")
            
            assert result == output_path
            
            content = Path(output_path).read_text()
            assert "your-app.herokuapp.com" in content
            assert "Self-Hosted" in content
            assert "JWT_TOKEN" in content
    
    def test_generate_cloud_template(self):
        """Test generating cloud template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, ".env")
            result = generate_env_template(output_path, "cloud")
            
            assert result == output_path
            
            content = Path(output_path).read_text()
            assert "runtime.glyphh.com" in content
            assert "Glyphh Cloud" in content
            assert "JWT_TOKEN" in content
    
    def test_generate_invalid_scenario(self):
        """Test that invalid scenario raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, ".env")
            
            with pytest.raises(ConfigurationError) as exc_info:
                generate_env_template(output_path, "invalid")
            
            assert "Invalid scenario" in str(exc_info.value)
    
    def test_generate_overwrites_existing(self):
        """Test that template overwrites existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, ".env")
            
            # Create existing file
            Path(output_path).write_text("OLD_CONTENT=true")
            
            # Generate new template
            generate_env_template(output_path, "local")
            
            content = Path(output_path).read_text()
            assert "OLD_CONTENT" not in content
            assert "localhost:8000" in content


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfigIntegration:
    """Integration tests for configuration loading."""
    
    def test_env_file_overrides_env_vars(self):
        """Test that .env file values are loaded (env vars take precedence after load)."""
        env_content = """
RUNTIME_URL=https://file.example.com
JWT_TOKEN=file_token
"""
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.env'
        ) as f:
            f.write(env_content)
            temp_path = f.name
        
        try:
            # Clear env vars before loading file
            with patch.dict(os.environ, {}, clear=True):
                config = load_env_config(temp_path)
                
                # File values should be loaded
                assert config.runtime_url == "https://file.example.com"
                assert config.jwt_token == "file_token"
        finally:
            os.remove(temp_path)
    
    def test_generated_template_is_loadable(self):
        """Test that generated templates can be loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, ".env")
            
            # Generate local template
            generate_env_template(output_path, "local")
            
            # Clear env vars and load from file
            with patch.dict(os.environ, {}, clear=True):
                config = load_env_config(output_path)
                
                assert config.runtime_url == "http://localhost:8000"
                assert config.is_local is True
