"""
Unit tests for model packaging and deployment.

Tests cover:
- GlyphhModel data structure creation
- Model serialization to .glyphh files
- Model deserialization from .glyphh files
- Model validation (completeness, version format, vector space consistency)
- Round-trip serialization/deserialization
- Custom encoder preservation
"""

import pytest
import numpy as np
import tempfile
import os
from datetime import datetime
from pathlib import Path

from glyphh import (
    GlyphhModel,
    EncoderConfig,
    Glyph,
    Vector,
    Layer,
    Segment,
    ModelValidationException
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def encoder_config():
    """Create a test encoder configuration."""
    return EncoderConfig(
        dimension=100,
        seed=42,
        similarity_weight=1.0,
        security_weight=0.9,
    )


@pytest.fixture
def sample_vector(encoder_config):
    """Create a sample bipolar vector."""
    from glyphh.core.types import compute_space_id
    
    space_id = compute_space_id(
        encoder_config.dimension,
        encoder_config.seed,
        encoder_config.to_json()
    )
    
    data = np.random.choice([-1, 1], size=encoder_config.dimension, replace=True).astype(np.int8)
    return Vector(data=data, dimension=encoder_config.dimension, space_id=space_id)


@pytest.fixture
def sample_glyph(encoder_config, sample_vector):
    """Create a sample glyph for testing."""
    from glyphh.core.types import compute_space_id
    
    space_id = compute_space_id(
        encoder_config.dimension,
        encoder_config.seed,
        encoder_config.to_json()
    )
    
    # Create role vectors
    role1 = Vector(
        data=np.random.choice([-1, 1], size=encoder_config.dimension).astype(np.int8),
        dimension=encoder_config.dimension,
        space_id=space_id
    )
    role2 = Vector(
        data=np.random.choice([-1, 1], size=encoder_config.dimension).astype(np.int8),
        dimension=encoder_config.dimension,
        space_id=space_id
    )
    
    # Create segment
    segment = Segment(
        name="attributes",
        cortex=sample_vector,
        roles={"type": role1, "color": role2},
        role_values={"type": "car", "color": "red"},
        weights={"segment": 0.9, "type": 1.0, "color": 0.8}
    )
    
    # Create layer
    layer = Layer(
        name="semantic",
        cortex=sample_vector,
        segments={"attributes": segment},
        weights={"layer": 0.8}
    )
    
    # Create glyph
    return Glyph(
        identifier="car_red@2024-01-15T10:30:00Z#v1",
        name="red car",
        space_id=space_id,
        global_cortex=sample_vector,
        layers={"semantic": layer},
        security_levels={"cortex": 0.9, "layer": 0.8},
        metadata={"domain": "automotive"},
        timestamp=datetime.now(),
        version="v1"
    )


@pytest.fixture
def valid_model(encoder_config, sample_glyph):
    """Create a valid model for testing."""
    return GlyphhModel(
        name="test_model",
        version="1.0.0",
        encoder_config=encoder_config,
        glyphs=[sample_glyph],
        custom_encoders={},
        metadata={"domain": "test", "description": "Test model"}
    )


# ============================================================================
# Test GlyphhModel Creation
# ============================================================================

def test_glyphh_model_creation(encoder_config, sample_glyph):
    """Test creating a valid GlyphhModel."""
    model = GlyphhModel(
        name="test_model",
        version="1.0.0",
        encoder_config=encoder_config,
        glyphs=[sample_glyph],
        custom_encoders={},
        metadata={"domain": "test"}
    )
    
    assert model.name == "test_model"
    assert model.version == "1.0.0"
    assert model.encoder_config == encoder_config
    assert len(model.glyphs) == 1
    assert model.glyphs[0] == sample_glyph
    assert model.metadata["domain"] == "test"
    assert isinstance(model.created_at, datetime)


def test_glyphh_model_with_defaults(encoder_config, sample_glyph):
    """Test creating a model with default values."""
    model = GlyphhModel(
        name="minimal_model",
        version="1.0.0",
        encoder_config=encoder_config,
        glyphs=[sample_glyph]
    )
    
    assert model.custom_encoders == {}
    assert model.metadata == {}
    assert isinstance(model.created_at, datetime)


# ============================================================================
# Test Model Validation
# ============================================================================

def test_validate_completeness_valid_model(valid_model):
    """Test validation passes for a valid model."""
    errors = valid_model.validate_completeness()
    assert errors == []


def test_validate_completeness_empty_glyphs_allowed(encoder_config):
    """Test that empty glyphs list is allowed — data loading happens later."""
    model = GlyphhModel(
        name="empty_model",
        version="1.0.0",
        encoder_config=encoder_config,
        glyphs=[]
    )
    assert model.name == "empty_model"
    assert model.glyphs == []


def test_validate_completeness_no_encoder_config(sample_glyph):
    """Test validation fails when encoder config is missing."""
    with pytest.raises(ModelValidationException) as exc_info:
        GlyphhModel(
            name="no_config_model",
            version="1.0.0",
            encoder_config=None,
            glyphs=[sample_glyph]
        )
    
    assert "Missing encoder configuration" in str(exc_info.value)


def test_validate_version_format_valid():
    """Test validation passes for valid semantic versions."""
    valid_versions = ["1.0.0", "0.1.0", "10.20.30", "999.999.999"]
    
    for version in valid_versions:
        # Should not raise exception
        assert re.match(r'^\d+\.\d+\.\d+$', version)


def test_validate_version_format_invalid(encoder_config, sample_glyph):
    """Test validation fails for invalid version formats."""
    invalid_versions = ["1.0", "1", "v1.0.0", "1.0.0-beta", "1.0.0.0"]
    
    for version in invalid_versions:
        with pytest.raises(ModelValidationException) as exc_info:
            GlyphhModel(
                name="invalid_version_model",
                version=version,
                encoder_config=encoder_config,
                glyphs=[sample_glyph]
            )
        
        assert "Invalid version format" in str(exc_info.value)
        assert version in str(exc_info.value)


def test_validate_vector_space_consistency(encoder_config, sample_glyph):
    """Test validation fails when glyphs have different space_ids."""
    # Create a second glyph with different space_id
    different_config = EncoderConfig(dimension=100, seed=999)
    from glyphh.core.types import compute_space_id
    
    different_space_id = compute_space_id(
        different_config.dimension,
        different_config.seed,
        different_config.to_json()
    )
    
    different_vector = Vector(
        data=np.random.choice([-1, 1], size=100).astype(np.int8),
        dimension=100,
        space_id=different_space_id
    )
    
    segment = Segment(
        name="attributes",
        cortex=different_vector,
        roles={},
        role_values={},
        weights={}
    )
    
    layer = Layer(
        name="semantic",
        cortex=different_vector,
        segments={"attributes": segment},
        weights={}
    )
    
    different_glyph = Glyph(
        identifier="different@2024-01-15T10:30:00Z#v1",
        name="different",
        space_id=different_space_id,
        global_cortex=different_vector,
        layers={"semantic": layer},
        security_levels={},
        metadata={},
        timestamp=datetime.now(),
        version="v1"
    )
    
    # Try to create model with glyphs from different spaces
    with pytest.raises(ModelValidationException) as exc_info:
        GlyphhModel(
            name="mixed_spaces_model",
            version="1.0.0",
            encoder_config=encoder_config,
            glyphs=[sample_glyph, different_glyph]
        )
    
    assert "Multiple vector spaces detected" in str(exc_info.value)


# ============================================================================
# Test Model Serialization
# ============================================================================

def test_to_file_creates_glyphh_file(valid_model):
    """Test that to_file creates a .glyphh file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.glyphh")
        valid_model.to_file(filepath)
        
        assert os.path.exists(filepath)
        assert Path(filepath).suffix == ".glyphh"


def test_to_file_adds_glyphh_extension(valid_model):
    """Test that to_file adds .glyphh extension if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model")
        valid_model.to_file(filepath)
        
        expected_path = os.path.join(tmpdir, "test_model.glyphh")
        assert os.path.exists(expected_path)


def test_to_file_with_empty_glyphs(encoder_config):
    """Test that to_file works with empty glyphs list."""
    model = GlyphhModel(
        name="empty_model",
        version="1.0.0",
        encoder_config=encoder_config,
        glyphs=[],
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "empty_model.glyphh")
        model.to_file(filepath)
        assert os.path.exists(filepath)
        
        # Verify it can be loaded back
        loaded = GlyphhModel.from_file(filepath)
        assert loaded.name == "empty_model"
        assert loaded.glyphs == []


# ============================================================================
# Test Model Deserialization
# ============================================================================

def test_from_file_loads_model(valid_model):
    """Test that from_file correctly loads a saved model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.glyphh")
        valid_model.to_file(filepath)
        
        loaded_model = GlyphhModel.from_file(filepath)
        
        assert loaded_model.name == valid_model.name
        assert loaded_model.version == valid_model.version
        assert loaded_model.encoder_config.dimension == valid_model.encoder_config.dimension
        assert loaded_model.encoder_config.seed == valid_model.encoder_config.seed
        assert len(loaded_model.glyphs) == len(valid_model.glyphs)
        assert loaded_model.metadata == valid_model.metadata


def test_from_file_nonexistent_file():
    """Test that from_file raises FileNotFoundError for nonexistent file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        GlyphhModel.from_file("nonexistent_model.glyphh")
    
    assert "Model file not found" in str(exc_info.value)


def test_from_file_invalid_json():
    """Test that from_file raises ValueError for invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "invalid.glyphh")
        
        # Write invalid JSON
        import gzip
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(ValueError) as exc_info:
            GlyphhModel.from_file(filepath)
        
        assert "Invalid JSON" in str(exc_info.value)


# ============================================================================
# Test Round-Trip Serialization
# ============================================================================

def test_round_trip_serialization(valid_model):
    """Test that model survives round-trip serialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.glyphh")
        
        # Save model
        valid_model.to_file(filepath)
        
        # Load model
        loaded_model = GlyphhModel.from_file(filepath)
        
        # Verify all fields match
        assert loaded_model.name == valid_model.name
        assert loaded_model.version == valid_model.version
        assert loaded_model.metadata == valid_model.metadata
        
        # Verify encoder config
        assert loaded_model.encoder_config.dimension == valid_model.encoder_config.dimension
        assert loaded_model.encoder_config.seed == valid_model.encoder_config.seed
        assert loaded_model.encoder_config.similarity_weight == valid_model.encoder_config.similarity_weight
        
        # Verify glyphs
        assert len(loaded_model.glyphs) == len(valid_model.glyphs)
        
        for orig_glyph, loaded_glyph in zip(valid_model.glyphs, loaded_model.glyphs):
            assert loaded_glyph.identifier == orig_glyph.identifier
            assert loaded_glyph.name == orig_glyph.name
            assert loaded_glyph.space_id == orig_glyph.space_id
            assert loaded_glyph.version == orig_glyph.version
            
            # Verify vectors match
            assert np.array_equal(
                loaded_glyph.global_cortex.data,
                orig_glyph.global_cortex.data
            )


def test_round_trip_with_multiple_glyphs(encoder_config, sample_glyph):
    """Test round-trip with multiple glyphs."""
    # Create second glyph (same space_id)
    from glyphh.core.types import compute_space_id
    
    space_id = compute_space_id(
        encoder_config.dimension,
        encoder_config.seed,
        encoder_config.to_json()
    )
    
    vector2 = Vector(
        data=np.random.choice([-1, 1], size=encoder_config.dimension).astype(np.int8),
        dimension=encoder_config.dimension,
        space_id=space_id
    )
    
    segment2 = Segment(
        name="attributes",
        cortex=vector2,
        roles={},
        role_values={},
        weights={}
    )
    
    layer2 = Layer(
        name="semantic",
        cortex=vector2,
        segments={"attributes": segment2},
        weights={}
    )
    
    glyph2 = Glyph(
        identifier="car_blue@2024-01-15T10:35:00Z#v1",
        name="blue car",
        space_id=space_id,
        global_cortex=vector2,
        layers={"semantic": layer2},
        security_levels={},
        metadata={},
        timestamp=datetime.now(),
        version="v1"
    )
    
    model = GlyphhModel(
        name="multi_glyph_model",
        version="2.1.3",
        encoder_config=encoder_config,
        glyphs=[sample_glyph, glyph2],
        metadata={"count": 2}
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "multi_glyph.glyphh")
        
        model.to_file(filepath)
        loaded_model = GlyphhModel.from_file(filepath)
        
        assert len(loaded_model.glyphs) == 2
        assert loaded_model.glyphs[0].name == "red car"
        assert loaded_model.glyphs[1].name == "blue car"


# ============================================================================
# Module-level class for custom encoder testing (must be at module level for pickling)
# ============================================================================

class CustomEncoderForTesting:
    """Custom encoder class for testing (module-level for pickling)."""
    def __init__(self, name):
        self.name = name
    
    def encode(self, data):
        return f"encoded_{data}"


# ============================================================================
# Test Custom Encoder Preservation
# ============================================================================

def test_custom_encoder_serialization(encoder_config, sample_glyph):
    """Test that custom encoders are preserved in serialization."""
    custom_encoder = CustomEncoderForTesting("test_encoder")
    
    model = GlyphhModel(
        name="custom_encoder_model",
        version="1.0.0",
        encoder_config=encoder_config,
        glyphs=[sample_glyph],
        custom_encoders={"test_encoder": custom_encoder}
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "custom_encoder.glyphh")
        
        model.to_file(filepath)
        loaded_model = GlyphhModel.from_file(filepath)
        
        assert "test_encoder" in loaded_model.custom_encoders
        loaded_encoder = loaded_model.custom_encoders["test_encoder"]
        assert loaded_encoder.name == "test_encoder"
        assert loaded_encoder.encode("test") == "encoded_test"


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_model_with_complex_metadata(encoder_config, sample_glyph):
    """Test model with complex metadata structures."""
    complex_metadata = {
        "domain": "automotive",
        "description": "Vehicle classification model",
        "tags": ["vehicles", "classification", "test"],
        "stats": {
            "num_concepts": 100,
            "accuracy": 0.95
        },
        "created_by": "test_user"
    }
    
    model = GlyphhModel(
        name="complex_metadata_model",
        version="1.0.0",
        encoder_config=encoder_config,
        glyphs=[sample_glyph],
        metadata=complex_metadata
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "complex_metadata.glyphh")
        
        model.to_file(filepath)
        loaded_model = GlyphhModel.from_file(filepath)
        
        assert loaded_model.metadata == complex_metadata
        assert loaded_model.metadata["stats"]["accuracy"] == 0.95


def test_model_with_hierarchical_glyph(encoder_config):
    """Test model with glyph containing multiple layers and segments."""
    from glyphh.core.types import compute_space_id
    
    space_id = compute_space_id(
        encoder_config.dimension,
        encoder_config.seed,
        encoder_config.to_json()
    )
    
    # Create multiple vectors
    vectors = [
        Vector(
            data=np.random.choice([-1, 1], size=encoder_config.dimension).astype(np.int8),
            dimension=encoder_config.dimension,
            space_id=space_id
        )
        for _ in range(5)
    ]
    
    # Create segments
    segment1 = Segment(
        name="attributes",
        cortex=vectors[0],
        roles={"type": vectors[1], "color": vectors[2]},
        role_values={"type": "car", "color": "red"},
        weights={"segment": 0.9}
    )
    
    segment2 = Segment(
        name="relations",
        cortex=vectors[3],
        roles={"has_part": vectors[4]},
        role_values={"has_part": "wheels"},
        weights={"segment": 0.85}
    )
    
    # Create layer
    layer = Layer(
        name="semantic",
        cortex=vectors[0],
        segments={"attributes": segment1, "relations": segment2},
        weights={"layer": 0.8}
    )
    
    # Create glyph
    glyph = Glyph(
        identifier="complex@2024-01-15T10:30:00Z#v1",
        name="complex glyph",
        space_id=space_id,
        global_cortex=vectors[0],
        layers={"semantic": layer},
        security_levels={"cortex": 0.9, "layer": 0.8},
        metadata={},
        timestamp=datetime.now(),
        version="v1"
    )
    
    model = GlyphhModel(
        name="hierarchical_model",
        version="1.0.0",
        encoder_config=encoder_config,
        glyphs=[glyph]
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "hierarchical.glyphh")
        
        model.to_file(filepath)
        loaded_model = GlyphhModel.from_file(filepath)
        
        loaded_glyph = loaded_model.glyphs[0]
        assert len(loaded_glyph.layers) == 1
        assert len(loaded_glyph.layers["semantic"].segments) == 2
        assert "attributes" in loaded_glyph.layers["semantic"].segments
        assert "relations" in loaded_glyph.layers["semantic"].segments


# ============================================================================
# Helper for version validation test
# ============================================================================

import re
