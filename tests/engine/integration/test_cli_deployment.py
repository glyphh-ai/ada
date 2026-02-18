"""
Integration tests for CLI deployment workflow.

Tests cover:
- End-to-end deployment with mock Runtime API
- .env configuration → API call → response parsing
- CLI command execution
- Error scenarios
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from click.testing import CliRunner

from glyphh.cli.main import cli
from glyphh.cli.config import RuntimeConfig, load_env_config


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        f.write("RUNTIME_URL=http://localhost:8000\n")
        f.write("RUNTIME_TIMEOUT=30\n")
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_model_file():
    """Create a temporary .glyphh model file."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.glyphh') as f:
        # Write minimal valid model data
        model_data = {
            "name": "test_model",
            "version": "1.0.0",
            "glyphs": []
        }
        f.write(json.dumps(model_data).encode())
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.remove(temp_path)


# ============================================================================
# CLI Help Tests
# ============================================================================

class TestCLIHelp:
    """Test CLI help output."""
    
    def test_main_help(self, runner):
        """Test main CLI help."""
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'build' in result.output
        assert 'test' in result.output
        assert 'package' in result.output
        assert 'runtime' in result.output
    
    def test_runtime_help(self, runner):
        """Test runtime command help."""
        result = runner.invoke(cli, ['runtime', '--help'])
        
        assert result.exit_code == 0
        assert 'deploy' in result.output
        assert 'status' in result.output
        assert 'init' in result.output
        assert 'token' in result.output
    
    def test_runtime_init_help(self, runner):
        """Test runtime init command help."""
        result = runner.invoke(cli, ['runtime', 'init', '--help'])
        
        assert result.exit_code == 0
        assert '--scenario' in result.output
        assert 'local' in result.output
        assert 'self-hosted' in result.output
        assert 'cloud' in result.output


# ============================================================================
# Runtime Init Command Tests
# ============================================================================

class TestRuntimeInitCommand:
    """Test glyphh runtime init command."""
    
    def test_init_local_scenario(self, runner):
        """Test generating local .env template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, '.env')
            
            result = runner.invoke(cli, [
                'runtime', 'init',
                '--scenario', 'local',
                '--output', output_path
            ])
            
            assert result.exit_code == 0
            assert 'Generated' in result.output
            assert os.path.exists(output_path)
            
            content = Path(output_path).read_text()
            assert 'localhost:8000' in content
    
    def test_init_self_hosted_scenario(self, runner):
        """Test generating self-hosted .env template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, '.env')
            
            result = runner.invoke(cli, [
                'runtime', 'init',
                '--scenario', 'self-hosted',
                '--output', output_path
            ])
            
            assert result.exit_code == 0
            
            content = Path(output_path).read_text()
            assert 'herokuapp.com' in content
            assert 'JWT_TOKEN' in content
    
    def test_init_cloud_scenario(self, runner):
        """Test generating cloud .env template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, '.env')
            
            result = runner.invoke(cli, [
                'runtime', 'init',
                '--scenario', 'cloud',
                '--output', output_path
            ])
            
            assert result.exit_code == 0
            
            content = Path(output_path).read_text()
            assert 'runtime.glyphh.com' in content
    
    def test_init_refuses_overwrite_without_force(self, runner):
        """Test that init refuses to overwrite existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, '.env')
            
            # Create existing file
            Path(output_path).write_text("EXISTING=true")
            
            result = runner.invoke(cli, [
                'runtime', 'init',
                '--output', output_path
            ])
            
            assert result.exit_code == 1
            assert 'already exists' in result.output
    
    def test_init_force_overwrites(self, runner):
        """Test that init with --force overwrites existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, '.env')
            
            # Create existing file
            Path(output_path).write_text("EXISTING=true")
            
            result = runner.invoke(cli, [
                'runtime', 'init',
                '--output', output_path,
                '--force'
            ])
            
            assert result.exit_code == 0
            
            content = Path(output_path).read_text()
            assert 'EXISTING' not in content


# ============================================================================
# Runtime Deploy Command Tests
# ============================================================================

class TestRuntimeDeployCommand:
    """Test glyphh runtime deploy command."""
    
    @patch('glyphh.cli.runtime.RuntimeAPIClient')
    def test_deploy_success(self, mock_client_class, runner, temp_model_file, temp_env_file):
        """Test successful deployment."""
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_result = Mock()
        mock_result.model_id = "model_123"
        mock_result.org_id = "org_456"
        mock_result.version = "1.0.0"
        mock_result.mcp_endpoint = "http://localhost:8000/mcp"
        mock_result.listener_endpoint = "http://localhost:8000/listener"
        mock_result.webhook_token = "token_789"
        mock_client.deploy_model.return_value = mock_result
        
        result = runner.invoke(cli, [
            'runtime', 'deploy',
            temp_model_file,
            '--env-file', temp_env_file
        ])
        
        assert result.exit_code == 0
        assert 'deployed successfully' in result.output
        assert 'model_123' in result.output
    
    @patch('glyphh.cli.runtime.RuntimeAPIClient')
    def test_deploy_connection_error(self, mock_client_class, runner, temp_model_file, temp_env_file):
        """Test deployment with connection error."""
        from glyphh.cli.runtime_client import ConnectionError
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.deploy_model.side_effect = ConnectionError("Connection refused")
        
        result = runner.invoke(cli, [
            'runtime', 'deploy',
            temp_model_file,
            '--env-file', temp_env_file
        ])
        
        assert result.exit_code == 1
        assert 'Connection Error' in result.output
    
    @patch('glyphh.cli.runtime.RuntimeAPIClient')
    def test_deploy_auth_error(self, mock_client_class, runner, temp_model_file, temp_env_file):
        """Test deployment with authentication error."""
        from glyphh.cli.runtime_client import AuthenticationError
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.deploy_model.side_effect = AuthenticationError("Invalid token")
        
        result = runner.invoke(cli, [
            'runtime', 'deploy',
            temp_model_file,
            '--env-file', temp_env_file
        ])
        
        assert result.exit_code == 1
        assert 'Authentication Error' in result.output
    
    def test_deploy_missing_env_file(self, runner, temp_model_file):
        """Test deployment with missing .env file."""
        result = runner.invoke(cli, [
            'runtime', 'deploy',
            temp_model_file,
            '--env-file', '/nonexistent/.env'
        ])
        
        assert result.exit_code == 1
        assert 'not found' in result.output


# ============================================================================
# Runtime Status Command Tests
# ============================================================================

class TestRuntimeStatusCommand:
    """Test glyphh runtime status command."""
    
    @patch('glyphh.cli.runtime.RuntimeAPIClient')
    def test_status_online(self, mock_client_class, runner, temp_env_file):
        """Test status when runtime is online."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_status = Mock()
        mock_status.version = "1.0.0"
        mock_status.models_loaded = 5
        mock_status.uptime = "2d 3h"
        mock_client.get_status.return_value = mock_status
        
        result = runner.invoke(cli, [
            'runtime', 'status',
            '--env-file', temp_env_file
        ])
        
        assert result.exit_code == 0
        assert 'Online' in result.output
        assert '1.0.0' in result.output
    
    @patch('glyphh.cli.runtime.RuntimeAPIClient')
    def test_status_offline(self, mock_client_class, runner, temp_env_file):
        """Test status when runtime is offline."""
        from glyphh.cli.runtime_client import ConnectionError
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_status.side_effect = ConnectionError("Connection refused")
        
        result = runner.invoke(cli, [
            'runtime', 'status',
            '--env-file', temp_env_file
        ])
        
        assert result.exit_code == 1
        assert 'Offline' in result.output


# ============================================================================
# Runtime Token Commands Tests
# ============================================================================

class TestRuntimeTokenCommands:
    """Test glyphh runtime token commands."""
    
    @patch('glyphh.cli.runtime.RuntimeAPIClient')
    def test_token_list(self, mock_client_class, runner, temp_env_file):
        """Test listing tokens."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_token = Mock()
        mock_token.id = "token_123"
        mock_token.model = "my_model"
        mock_token.created_at = "2024-01-15"
        mock_token.status = "Active"
        mock_client.list_tokens.return_value = [mock_token]
        
        result = runner.invoke(cli, [
            'runtime', 'token', 'list',
            '--env-file', temp_env_file
        ])
        
        assert result.exit_code == 0
        assert 'token_123' in result.output
        assert 'my_model' in result.output
    
    @patch('glyphh.cli.runtime.RuntimeAPIClient')
    def test_token_list_empty(self, mock_client_class, runner, temp_env_file):
        """Test listing tokens when none exist."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.list_tokens.return_value = []
        
        result = runner.invoke(cli, [
            'runtime', 'token', 'list',
            '--env-file', temp_env_file
        ])
        
        assert result.exit_code == 0
        assert 'No webhook tokens found' in result.output
    
    @patch('glyphh.cli.runtime.RuntimeAPIClient')
    def test_token_revoke(self, mock_client_class, runner, temp_env_file):
        """Test revoking a token."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.revoke_token.return_value = True
        
        result = runner.invoke(cli, [
            'runtime', 'token', 'revoke',
            'token_123',
            '--env-file', temp_env_file,
            '-y'  # Skip confirmation
        ])
        
        assert result.exit_code == 0
        assert 'revoked successfully' in result.output
    
    @patch('glyphh.cli.runtime.RuntimeAPIClient')
    def test_token_revoke_not_found(self, mock_client_class, runner, temp_env_file):
        """Test revoking non-existent token."""
        from glyphh.cli.runtime_client import NotFoundError
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.revoke_token.side_effect = NotFoundError("Token not found")
        
        result = runner.invoke(cli, [
            'runtime', 'token', 'revoke',
            'nonexistent',
            '--env-file', temp_env_file,
            '-y'
        ])
        
        assert result.exit_code == 1
        assert 'not found' in result.output


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

class TestEndToEndWorkflow:
    """Test complete deployment workflow."""
    
    @patch('glyphh.cli.runtime.RuntimeAPIClient')
    def test_full_deployment_workflow(self, mock_client_class, runner, temp_model_file):
        """Test complete workflow: init → deploy → status."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Setup deploy mock
        mock_result = Mock()
        mock_result.model_id = "model_123"
        mock_result.org_id = None
        mock_result.version = "1.0.0"
        mock_result.mcp_endpoint = "http://localhost:8000/mcp"
        mock_result.listener_endpoint = None
        mock_result.webhook_token = None
        mock_client.deploy_model.return_value = mock_result
        
        # Setup status mock
        mock_status = Mock()
        mock_status.version = "1.0.0"
        mock_status.models_loaded = 1
        mock_status.uptime = "1m"
        mock_client.get_status.return_value = mock_status
        
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = os.path.join(temp_dir, '.env')
            
            # Step 1: Init
            result = runner.invoke(cli, [
                'runtime', 'init',
                '--scenario', 'local',
                '--output', env_path
            ])
            assert result.exit_code == 0
            
            # Step 2: Deploy
            result = runner.invoke(cli, [
                'runtime', 'deploy',
                temp_model_file,
                '--env-file', env_path
            ])
            assert result.exit_code == 0
            assert 'deployed successfully' in result.output
            
            # Step 3: Status
            result = runner.invoke(cli, [
                'runtime', 'status',
                '--env-file', env_path
            ])
            assert result.exit_code == 0
            assert 'Online' in result.output
