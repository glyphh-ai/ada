"""
Unit tests for CLI commands
"""

import json
import pytest
from pathlib import Path
from click.testing import CliRunner
from glyphh.cli.main import cli


@pytest.fixture
def runner():
    """Create a CLI test runner"""
    return CliRunner()


@pytest.fixture
def temp_model_file(tmp_path):
    """Create a temporary model file"""
    model_file = tmp_path / "test_model.json"
    return str(model_file)


class TestBuildCommands:
    """Test build commands"""
    
    def test_build_init(self, runner, temp_model_file):
        """Test model initialization"""
        result = runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--seed', '42',
            '--output', temp_model_file
        ])
        
        assert result.exit_code == 0
        assert "[OK] Initialized model 'test_model'" in result.output
        assert "Dimension: 1000" in result.output
        assert "Seed: 42" in result.output
        
        # Verify file was created
        assert Path(temp_model_file).exists()
        
        # Verify file content
        with open(temp_model_file, 'r') as f:
            data = json.load(f)
        
        assert data['name'] == 'test_model'
        assert data['config']['dimension'] == 1000
        assert data['config']['seed'] == 42
        assert data['concepts'] == []
    
    def test_build_add(self, runner, temp_model_file):
        """Test adding concepts to model"""
        # Initialize model first
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        # Add concept
        result = runner.invoke(cli, [
            'build', 'add',
            '--concept', 'red car',
            '--attributes', '{"type":"car","color":"red"}',
            '--model', temp_model_file
        ])
        
        assert result.exit_code == 0
        assert "[OK] Added concept 'red car' to model" in result.output
        assert "Total concepts: 1" in result.output
        
        # Verify file content
        with open(temp_model_file, 'r') as f:
            data = json.load(f)
        
        assert len(data['concepts']) == 1
        assert data['concepts'][0]['name'] == 'red car'
        assert data['concepts'][0]['attributes']['type'] == 'car'
        assert data['concepts'][0]['attributes']['color'] == 'red'
    
    def test_build_add_invalid_json(self, runner, temp_model_file):
        """Test adding concept with invalid JSON"""
        # Initialize model first
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        # Try to add concept with invalid JSON
        result = runner.invoke(cli, [
            'build', 'add',
            '--concept', 'red car',
            '--attributes', 'not valid json',
            '--model', temp_model_file
        ])
        
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output
    
    def test_build_list_empty(self, runner, temp_model_file):
        """Test listing concepts in empty model"""
        # Initialize model
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        # List concepts
        result = runner.invoke(cli, [
            'build', 'list',
            '--model', temp_model_file
        ])
        
        assert result.exit_code == 0
        assert "Model: test_model" in result.output
        assert "Total concepts: 0" in result.output
        assert "No concepts added yet" in result.output
    
    def test_build_list_with_concepts(self, runner, temp_model_file):
        """Test listing concepts with data"""
        # Initialize and add concepts
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        runner.invoke(cli, [
            'build', 'add',
            '--concept', 'red car',
            '--attributes', '{"type":"car","color":"red"}',
            '--model', temp_model_file
        ])
        
        runner.invoke(cli, [
            'build', 'add',
            '--concept', 'blue car',
            '--attributes', '{"type":"car","color":"blue"}',
            '--model', temp_model_file
        ])
        
        # List concepts
        result = runner.invoke(cli, [
            'build', 'list',
            '--model', temp_model_file
        ])
        
        assert result.exit_code == 0
        assert "Total concepts: 2" in result.output
        assert "red car" in result.output
        assert "blue car" in result.output
    
    def test_build_list_json_format(self, runner, temp_model_file):
        """Test listing concepts in JSON format"""
        # Initialize and add concept
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        runner.invoke(cli, [
            'build', 'add',
            '--concept', 'red car',
            '--attributes', '{"type":"car","color":"red"}',
            '--model', temp_model_file
        ])
        
        # List in JSON format
        result = runner.invoke(cli, [
            'build', 'list',
            '--model', temp_model_file,
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output = json.loads(result.output)
        assert output['model_name'] == 'test_model'
        assert output['total_concepts'] == 1
        assert len(output['concepts']) == 1
        assert output['concepts'][0]['name'] == 'red car'


class TestTestCommands:
    """Test test commands"""
    
    def test_test_encode(self, runner, temp_model_file):
        """Test encoding a concept"""
        # Initialize model
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        # Test encoding
        result = runner.invoke(cli, [
            'test', 'encode',
            '--concept', 'red car',
            '--attributes', '{"type":"car","color":"red"}',
            '--model', temp_model_file
        ])
        
        assert result.exit_code == 0
        assert "Encoding Test Results" in result.output
        assert "Concept: red car" in result.output
        assert "Dimension: 1000" in result.output
        assert "Space ID:" in result.output
    
    def test_test_similarity(self, runner, temp_model_file):
        """Test similarity computation"""
        # Initialize and add concepts
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        runner.invoke(cli, [
            'build', 'add',
            '--concept', 'red car',
            '--attributes', '{"type":"car","color":"red"}',
            '--model', temp_model_file
        ])
        
        runner.invoke(cli, [
            'build', 'add',
            '--concept', 'blue car',
            '--attributes', '{"type":"car","color":"blue"}',
            '--model', temp_model_file
        ])
        
        # Test similarity
        result = runner.invoke(cli, [
            'test', 'similarity',
            'red car', 'blue car',
            '--model', temp_model_file
        ])
        
        assert result.exit_code == 0
        assert "Similarity Test Results" in result.output
        assert "Glyph 1: red car" in result.output
        assert "Glyph 2: blue car" in result.output
        assert "Similarity Score:" in result.output
    
    def test_test_similarity_concept_not_found(self, runner, temp_model_file):
        """Test similarity with non-existent concept"""
        # Initialize model
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        # Test similarity with non-existent concept
        result = runner.invoke(cli, [
            'test', 'similarity',
            'red car', 'blue car',
            '--model', temp_model_file
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.output


class TestPackageCommands:
    """Test package commands"""
    
    def test_package_create(self, runner, temp_model_file, tmp_path):
        """Test creating a package"""
        # Initialize and add concepts
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        runner.invoke(cli, [
            'build', 'add',
            '--concept', 'red car',
            '--attributes', '{"type":"car","color":"red"}',
            '--model', temp_model_file
        ])
        
        # Create package
        package_file = tmp_path / "test_model.glyphh"
        result = runner.invoke(cli, [
            'package', 'create',
            '--model', temp_model_file,
            '--output', str(package_file),
            '--version', '1.0.0'
        ])
        
        assert result.exit_code == 0
        assert "[OK] Created package:" in result.output
        assert "Model: test_model" in result.output
        assert "Version: 1.0.0" in result.output
        assert "Glyphs: 1" in result.output
        
        # Verify package file exists
        assert package_file.exists()
    
    def test_package_validate(self, runner, temp_model_file, tmp_path):
        """Test validating a package"""
        # Create a package first
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        runner.invoke(cli, [
            'build', 'add',
            '--concept', 'red car',
            '--attributes', '{"type":"car","color":"red"}',
            '--model', temp_model_file
        ])
        
        package_file = tmp_path / "test_model.glyphh"
        runner.invoke(cli, [
            'package', 'create',
            '--model', temp_model_file,
            '--output', str(package_file)
        ])
        
        # Validate package
        result = runner.invoke(cli, [
            'package', 'validate',
            str(package_file)
        ])
        
        assert result.exit_code == 0
        assert "Package Validation Results" in result.output
        assert "[OK] Validation PASSED" in result.output
    
    def test_package_info(self, runner, temp_model_file, tmp_path):
        """Test displaying package info"""
        # Create a package first
        runner.invoke(cli, [
            'build', 'init',
            '--name', 'test_model',
            '--dimension', '1000',
            '--output', temp_model_file
        ])
        
        runner.invoke(cli, [
            'build', 'add',
            '--concept', 'red car',
            '--attributes', '{"type":"car","color":"red"}',
            '--model', temp_model_file
        ])
        
        package_file = tmp_path / "test_model.glyphh"
        runner.invoke(cli, [
            'package', 'create',
            '--model', temp_model_file,
            '--output', str(package_file),
            '--version', '1.2.3'
        ])
        
        # Get package info
        result = runner.invoke(cli, [
            'package', 'info',
            str(package_file)
        ])
        
        assert result.exit_code == 0
        assert "Package Information" in result.output
        assert "Name: test_model" in result.output
        assert "Version: 1.2.3" in result.output
        assert "Dimension: 1000" in result.output
        assert "Glyphs: 1" in result.output


class TestRuntimeCommands:
    """Test runtime commands"""
    
    def test_runtime_init_local(self, runner, tmp_path):
        """Test runtime init command for local scenario"""
        env_file = tmp_path / ".env"
        result = runner.invoke(cli, [
            'runtime', 'init',
            '--scenario', 'local',
            '--output', str(env_file)
        ])
        
        assert result.exit_code == 0
        assert "[OK]" in result.output
        assert "Generated local configuration template" in result.output
        assert env_file.exists()
    
    def test_runtime_status_no_config(self, runner):
        """Test runtime status command without config"""
        result = runner.invoke(cli, [
            'runtime', 'status'
        ])
        
        # Should fail gracefully when no runtime is configured
        # The command will try to connect to default localhost
        assert result.exit_code != 0 or "Offline" in result.output or "Error" in result.output


