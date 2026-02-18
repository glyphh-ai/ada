"""
Unit tests for the SchemaVectorizer class.

Tests cover:
- SchemaVectorizer initialization
- Encoder and config storage
- Schema vectors cache initialization
- Type validation

Validates: Requirement 1.3 - WHEN generating schema vectors, THE SDK SHALL
use the same encoder configuration as the model
"""

import pytest

from glyphh.encoder.base import Encoder
from glyphh.core.config import EncoderConfig, Layer, Segment, Role
from glyphh.nl.schema_vectorizer import SchemaVectorizer, SchemaVector


class TestSchemaVectorizerInit:
    """Test SchemaVectorizer.__init__() method."""
    
    def test_init_stores_encoder_reference(self):
        """Test that __init__ stores the encoder reference.
        
        **Validates: Requirement 1.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        assert vectorizer.encoder is encoder
    
    def test_init_stores_config_reference(self):
        """Test that __init__ stores the config reference.
        
        **Validates: Requirement 1.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        assert vectorizer.config is config
    
    def test_init_creates_empty_schema_vectors_dict(self):
        """Test that __init__ initializes empty _schema_vectors dict.
        
        **Validates: Requirement 1.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        assert isinstance(vectorizer._schema_vectors, dict)
        assert len(vectorizer._schema_vectors) == 0
    
    def test_init_with_config_containing_layers(self):
        """Test initialization with a config containing layers, segments, and roles.
        
        **Validates: Requirement 1.3**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[
                                Role(name="make"),
                                Role(name="model"),
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        assert vectorizer.encoder is encoder
        assert vectorizer.config is config
        assert len(vectorizer._schema_vectors) == 0
    
    def test_init_raises_type_error_for_invalid_encoder(self):
        """Test that __init__ raises TypeError for non-Encoder encoder."""
        config = EncoderConfig(dimension=1000, seed=42)
        
        with pytest.raises(TypeError) as exc_info:
            SchemaVectorizer("not_an_encoder", config)
        
        assert "encoder must be an Encoder instance" in str(exc_info.value)
    
    def test_init_raises_type_error_for_invalid_config(self):
        """Test that __init__ raises TypeError for non-EncoderConfig config."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        with pytest.raises(TypeError) as exc_info:
            SchemaVectorizer(encoder, "not_a_config")
        
        assert "config must be an EncoderConfig instance" in str(exc_info.value)
    
    def test_init_raises_type_error_for_none_encoder(self):
        """Test that __init__ raises TypeError for None encoder."""
        config = EncoderConfig(dimension=1000, seed=42)
        
        with pytest.raises(TypeError) as exc_info:
            SchemaVectorizer(None, config)
        
        assert "encoder must be an Encoder instance" in str(exc_info.value)
    
    def test_init_raises_type_error_for_none_config(self):
        """Test that __init__ raises TypeError for None config."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        with pytest.raises(TypeError) as exc_info:
            SchemaVectorizer(encoder, None)
        
        assert "config must be an EncoderConfig instance" in str(exc_info.value)


class TestSchemaVectorizerGetSchemaVectors:
    """Test SchemaVectorizer.get_schema_vectors() method."""
    
    def test_get_schema_vectors_returns_empty_dict_initially(self):
        """Test that get_schema_vectors returns empty dict after init."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectors = vectorizer.get_schema_vectors()
        
        assert isinstance(vectors, dict)
        assert len(vectors) == 0
    
    def test_get_schema_vectors_returns_same_dict_reference(self):
        """Test that get_schema_vectors returns the internal dict reference."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectors1 = vectorizer.get_schema_vectors()
        vectors2 = vectorizer.get_schema_vectors()
        
        assert vectors1 is vectors2
        assert vectors1 is vectorizer._schema_vectors


class TestSchemaVectorizerEncoderConsistency:
    """Test that SchemaVectorizer uses the same encoder as the model.
    
    **Validates: Requirement 1.3** - WHEN generating schema vectors, THE SDK SHALL
    use the same encoder configuration as the model
    """
    
    def test_encoder_space_id_accessible(self):
        """Test that the encoder's space_id is accessible through vectorizer."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # The vectorizer should use the same encoder, so space_id should match
        assert vectorizer.encoder.space_id == encoder.space_id
    
    def test_encoder_dimension_accessible(self):
        """Test that the encoder's dimension is accessible through vectorizer."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        assert vectorizer.encoder.dimension == 1000
    
    def test_encoder_seed_accessible(self):
        """Test that the encoder's seed is accessible through vectorizer."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        assert vectorizer.encoder.seed == 42


class TestSchemaVectorizerVectorizeRole:
    """Test SchemaVectorizer.vectorize_role() method.
    
    **Validates: Requirement 1.1** - THE SDK SHALL generate a bipolar vector
    for each role name in the model config
    """
    
    def test_vectorize_role_returns_schema_vector(self):
        """Test that vectorize_role returns a SchemaVector instance.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("vehicle.identity.make")
        
        assert isinstance(result, SchemaVector)
    
    def test_vectorize_role_extracts_role_name_from_path(self):
        """Test that vectorize_role extracts the role name from the path.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("vehicle.identity.make")
        
        # Key should be the role name (last component of path)
        assert result.key == "make"
    
    def test_vectorize_role_preserves_full_path(self):
        """Test that vectorize_role preserves the full role_path.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("vehicle.identity.make")
        
        assert result.role_path == "vehicle.identity.make"
    
    def test_vectorize_role_sets_element_type_to_role(self):
        """Test that vectorize_role sets element_type to 'role'.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("vehicle.identity.make")
        
        assert result.element_type == "role"
    
    def test_vectorize_role_sets_original_value_to_none(self):
        """Test that vectorize_role sets original_value to None for roles.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("vehicle.identity.make")
        
        assert result.original_value is None
    
    def test_vectorize_role_generates_bipolar_vector(self):
        """Test that vectorize_role generates a bipolar vector {-1, +1}.
        
        **Validates: Requirement 1.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("vehicle.identity.make")
        
        # All values should be in {-1, +1}
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_vectorize_role_vector_has_correct_dimension(self):
        """Test that vectorize_role generates vector with correct dimension.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("vehicle.identity.make")
        
        assert result.vector.dimension == 1000
    
    def test_vectorize_role_vector_has_correct_space_id(self):
        """Test that vectorize_role generates vector with encoder's space_id.
        
        **Validates: Requirement 1.1, 1.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("vehicle.identity.make")
        
        assert result.vector.space_id == encoder.space_id
    
    def test_vectorize_role_with_simple_role_name(self):
        """Test vectorize_role with a simple role name (no path).
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("make")
        
        assert result.key == "make"
        assert result.role_path == "make"
        assert result.element_type == "role"
    
    def test_vectorize_role_with_two_level_path(self):
        """Test vectorize_role with a two-level path.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("identity.make")
        
        assert result.key == "make"
        assert result.role_path == "identity.make"
    
    def test_vectorize_role_deterministic(self):
        """Test that vectorize_role is deterministic (same input → same output).
        
        **Validates: Requirement 1.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result1 = vectorizer.vectorize_role("vehicle.identity.make")
        result2 = vectorizer.vectorize_role("vehicle.identity.make")
        
        # Same role path should produce same vector
        assert np.array_equal(result1.vector.data, result2.vector.data)
    
    def test_vectorize_role_different_roles_produce_different_vectors(self):
        """Test that different role names produce different vectors.
        
        **Validates: Requirement 1.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        make_vector = vectorizer.vectorize_role("vehicle.identity.make")
        model_vector = vectorizer.vectorize_role("vehicle.identity.model")
        
        # Different role names should produce different vectors
        assert not np.array_equal(make_vector.vector.data, model_vector.vector.data)
    
    def test_vectorize_role_raises_value_error_for_empty_path(self):
        """Test that vectorize_role raises ValueError for empty path.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.vectorize_role("")
        
        assert "role_path must be a non-empty string" in str(exc_info.value)
    
    def test_vectorize_role_raises_value_error_for_whitespace_only_path(self):
        """Test that vectorize_role raises ValueError for whitespace-only path.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.vectorize_role("   ")
        
        assert "role_path must be a non-empty string" in str(exc_info.value)
    
    def test_vectorize_role_with_different_dimensions(self):
        """Test vectorize_role with different encoder dimensions.
        
        **Validates: Requirement 1.1, 1.3**
        """
        import numpy as np
        
        # Test with smaller dimension
        config_small = EncoderConfig(dimension=500, seed=42)
        encoder_small = Encoder(config_small)
        vectorizer_small = SchemaVectorizer(encoder_small, config_small)
        
        result_small = vectorizer_small.vectorize_role("make")
        assert result_small.vector.dimension == 500
        assert np.all(np.isin(result_small.vector.data, [-1, 1]))
        
        # Test with larger dimension
        config_large = EncoderConfig(dimension=5000, seed=42)
        encoder_large = Encoder(config_large)
        vectorizer_large = SchemaVectorizer(encoder_large, config_large)
        
        result_large = vectorizer_large.vectorize_role("make")
        assert result_large.vector.dimension == 5000
        assert np.all(np.isin(result_large.vector.data, [-1, 1]))
    
    def test_vectorize_role_same_role_name_different_paths_different_vectors(self):
        """Test that same role name in different paths produces different vectors.
        
        The vector is generated from the full path to ensure distinct vectors
        for nested role paths.
        
        **Validates: Requirements 1.1, 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Same role name "make" in different paths
        result1 = vectorizer.vectorize_role("vehicle.identity.make")
        result2 = vectorizer.vectorize_role("product.details.make")
        
        # Different paths should produce different vectors (Requirement 1.6)
        assert not np.array_equal(result1.vector.data, result2.vector.data)
        
        # Different role_path values
        assert result1.role_path != result2.role_path
        
        # But same key (role name)
        assert result1.key == result2.key == "make"


class TestSchemaVectorizerVectorizeValue:
    """Test SchemaVectorizer.vectorize_value() method.
    
    **Validates: Requirement 1.2** - THE SDK SHALL generate a bipolar vector
    for each unique value defined in the model config
    """
    
    def test_vectorize_value_returns_schema_vector(self):
        """Test that vectorize_value returns a SchemaVector instance.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        assert isinstance(result, SchemaVector)
    
    def test_vectorize_value_creates_role_equals_value_key(self):
        """Test that vectorize_value creates key in 'role=value' format.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        # Key should be "role=value" format
        assert result.key == "make=Toyota"
    
    def test_vectorize_value_preserves_full_path(self):
        """Test that vectorize_value preserves the full role_path.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        assert result.role_path == "vehicle.identity.make"
    
    def test_vectorize_value_sets_element_type_to_value(self):
        """Test that vectorize_value sets element_type to 'value'.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        assert result.element_type == "value"
    
    def test_vectorize_value_sets_original_value(self):
        """Test that vectorize_value sets original_value to the provided value.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        assert result.original_value == "Toyota"
    
    def test_vectorize_value_generates_bipolar_vector(self):
        """Test that vectorize_value generates a bipolar vector {-1, +1}.
        
        **Validates: Requirement 1.2**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        # All values should be in {-1, +1}
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_vectorize_value_vector_has_correct_dimension(self):
        """Test that vectorize_value generates vector with correct dimension.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        assert result.vector.dimension == 1000
    
    def test_vectorize_value_vector_has_correct_space_id(self):
        """Test that vectorize_value generates vector with encoder's space_id.
        
        **Validates: Requirement 1.2, 1.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        assert result.vector.space_id == encoder.space_id
    
    def test_vectorize_value_with_simple_role_name(self):
        """Test vectorize_value with a simple role name (no path).
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("make", "Toyota")
        
        assert result.key == "make=Toyota"
        assert result.role_path == "make"
        assert result.element_type == "value"
        assert result.original_value == "Toyota"
    
    def test_vectorize_value_with_multi_word_value(self):
        """Test vectorize_value with a multi-word value.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("parts.category", "Brake Pads")
        
        assert result.key == "category=Brake Pads"
        assert result.original_value == "Brake Pads"
    
    def test_vectorize_value_deterministic(self):
        """Test that vectorize_value is deterministic (same input → same output).
        
        **Validates: Requirement 1.2**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result1 = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        result2 = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        # Same inputs should produce same vector
        assert np.array_equal(result1.vector.data, result2.vector.data)
    
    def test_vectorize_value_different_values_produce_different_vectors(self):
        """Test that different values produce different vectors.
        
        **Validates: Requirement 1.2**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        toyota_vector = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        honda_vector = vectorizer.vectorize_value("vehicle.identity.make", "Honda")
        
        # Different values should produce different vectors
        assert not np.array_equal(toyota_vector.vector.data, honda_vector.vector.data)
    
    def test_vectorize_value_same_value_different_roles_same_vector(self):
        """Test that same value in different roles produces same vector.
        
        The vector is generated from the value, not the role.
        
        **Validates: Requirement 1.2**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Same value "Premium" in different roles
        result1 = vectorizer.vectorize_value("vehicle.identity.trim", "Premium")
        result2 = vectorizer.vectorize_value("product.quality.grade", "Premium")
        
        # Same value should produce same vector
        assert np.array_equal(result1.vector.data, result2.vector.data)
        
        # But different keys and role_paths
        assert result1.key != result2.key
        assert result1.role_path != result2.role_path
    
    def test_vectorize_value_raises_value_error_for_empty_role_path(self):
        """Test that vectorize_value raises ValueError for empty role_path.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.vectorize_value("", "Toyota")
        
        assert "role_path must be a non-empty string" in str(exc_info.value)
    
    def test_vectorize_value_raises_value_error_for_whitespace_only_role_path(self):
        """Test that vectorize_value raises ValueError for whitespace-only role_path.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.vectorize_value("   ", "Toyota")
        
        assert "role_path must be a non-empty string" in str(exc_info.value)
    
    def test_vectorize_value_raises_value_error_for_empty_value(self):
        """Test that vectorize_value raises ValueError for empty value.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.vectorize_value("vehicle.identity.make", "")
        
        assert "value must be a non-empty string" in str(exc_info.value)
    
    def test_vectorize_value_raises_value_error_for_whitespace_only_value(self):
        """Test that vectorize_value raises ValueError for whitespace-only value.
        
        **Validates: Requirement 1.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.vectorize_value("vehicle.identity.make", "   ")
        
        assert "value must be a non-empty string" in str(exc_info.value)
    
    def test_vectorize_value_with_different_dimensions(self):
        """Test vectorize_value with different encoder dimensions.
        
        **Validates: Requirement 1.2, 1.3**
        """
        import numpy as np
        
        # Test with smaller dimension
        config_small = EncoderConfig(dimension=500, seed=42)
        encoder_small = Encoder(config_small)
        vectorizer_small = SchemaVectorizer(encoder_small, config_small)
        
        result_small = vectorizer_small.vectorize_value("make", "Toyota")
        assert result_small.vector.dimension == 500
        assert np.all(np.isin(result_small.vector.data, [-1, 1]))
        
        # Test with larger dimension
        config_large = EncoderConfig(dimension=5000, seed=42)
        encoder_large = Encoder(config_large)
        vectorizer_large = SchemaVectorizer(encoder_large, config_large)
        
        result_large = vectorizer_large.vectorize_value("make", "Toyota")
        assert result_large.vector.dimension == 5000
        assert np.all(np.isin(result_large.vector.data, [-1, 1]))
    
    def test_vectorize_value_with_special_characters(self):
        """Test vectorize_value with values containing special characters.
        
        **Validates: Requirement 1.2**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Value with special characters
        result = vectorizer.vectorize_value("product.name", "O'Reilly Auto Parts")
        
        assert result.key == "name=O'Reilly Auto Parts"
        assert result.original_value == "O'Reilly Auto Parts"
        assert result.element_type == "value"
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_vectorize_value_with_numeric_string(self):
        """Test vectorize_value with numeric string values.
        
        **Validates: Requirement 1.2**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_value("vehicle.year", "2024")
        
        assert result.key == "year=2024"
        assert result.original_value == "2024"
        assert result.element_type == "value"
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_vectorize_value_vector_differs_from_role_vector(self):
        """Test that value vector differs from role vector for same role.
        
        **Validates: Requirement 1.2**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        role_vector = vectorizer.vectorize_role("vehicle.identity.make")
        value_vector = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        # Role vector (for "make") should differ from value vector (for "Toyota")
        assert not np.array_equal(role_vector.vector.data, value_vector.vector.data)
        
        # Different element types
        assert role_vector.element_type == "role"
        assert value_vector.element_type == "value"


class TestSchemaVectorizerVectorizeSchema:
    """Test SchemaVectorizer.vectorize_schema() method.
    
    **Validates: Requirements 1.1, 1.2, 1.4**
    - 1.1: THE SDK SHALL generate a bipolar vector for each role name
    - 1.2: THE SDK SHALL generate a bipolar vector for each unique value
    - 1.4: THE SDK SHALL cache schema vectors to avoid regeneration
    """
    
    def test_vectorize_schema_returns_dict(self):
        """Test that vectorize_schema returns a dictionary.
        
        **Validates: Requirements 1.1, 1.2**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="make")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        assert isinstance(result, dict)
    
    def test_vectorize_schema_generates_role_vectors(self):
        """Test that vectorize_schema generates vectors for all roles.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[
                                Role(name="make"),
                                Role(name="model"),
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Both role vectors should be generated
        assert "make" in result
        assert "model" in result
        assert result["make"].element_type == "role"
        assert result["model"].element_type == "role"
    
    def test_vectorize_schema_preserves_full_role_paths(self):
        """Test that vectorize_schema preserves full role paths.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[
                                Role(name="make"),
                                Role(name="model"),
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Full paths should be preserved
        assert result["make"].role_path == "vehicle.identity.make"
        assert result["model"].role_path == "vehicle.identity.model"
    
    def test_vectorize_schema_generates_bipolar_vectors(self):
        """Test that vectorize_schema generates bipolar vectors.
        
        **Validates: Requirement 1.1**
        """
        import numpy as np
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="make")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Vector should be bipolar
        assert np.all(np.isin(result["make"].vector.data, [-1, 1]))
    
    def test_vectorize_schema_vectors_have_correct_dimension(self):
        """Test that vectorize_schema generates vectors with correct dimension.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="make")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        assert result["make"].vector.dimension == 1000
    
    def test_vectorize_schema_vectors_have_correct_space_id(self):
        """Test that vectorize_schema generates vectors with encoder's space_id.
        
        **Validates: Requirements 1.1, 1.3**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="make")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        assert result["make"].vector.space_id == encoder.space_id
    
    def test_vectorize_schema_caches_results(self):
        """Test that vectorize_schema caches results.
        
        **Validates: Requirement 1.4**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="make")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result1 = vectorizer.vectorize_schema()
        result2 = vectorizer.vectorize_schema()
        
        # Should return the same dict reference (cached)
        assert result1 is result2
    
    def test_vectorize_schema_populates_internal_cache(self):
        """Test that vectorize_schema populates _schema_vectors cache.
        
        **Validates: Requirement 1.4**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="make")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Cache should be empty initially
        assert len(vectorizer._schema_vectors) == 0
        
        result = vectorizer.vectorize_schema()
        
        # Cache should be populated
        assert len(vectorizer._schema_vectors) > 0
        assert result is vectorizer._schema_vectors
    
    def test_vectorize_schema_with_multiple_layers(self):
        """Test vectorize_schema with multiple layers.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="make")]
                        )
                    ]
                ),
                Layer(
                    name="parts",
                    segments=[
                        Segment(
                            name="catalog",
                            roles=[Role(name="category")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Both roles from different layers should be present
        assert "make" in result
        assert "category" in result
        assert result["make"].role_path == "vehicle.identity.make"
        assert result["category"].role_path == "parts.catalog.category"
    
    def test_vectorize_schema_with_multiple_segments(self):
        """Test vectorize_schema with multiple segments in a layer.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="make")]
                        ),
                        Segment(
                            name="specs",
                            roles=[Role(name="engine")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Both roles from different segments should be present
        assert "make" in result
        assert "engine" in result
        assert result["make"].role_path == "vehicle.identity.make"
        assert result["engine"].role_path == "vehicle.specs.engine"
    
    def test_vectorize_schema_with_multiple_roles_in_segment(self):
        """Test vectorize_schema with multiple roles in a segment.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[
                                Role(name="make"),
                                Role(name="model"),
                                Role(name="year"),
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # All three roles should be present
        assert "make" in result
        assert "model" in result
        assert "year" in result
        assert len(result) == 3
    
    def test_vectorize_schema_with_empty_layers(self):
        """Test vectorize_schema with no layers in config.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_vectorize_schema_with_empty_segments(self):
        """Test vectorize_schema with layer containing no segments.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_vectorize_schema_with_empty_roles(self):
        """Test vectorize_schema with segment containing no roles.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_vectorize_schema_deterministic(self):
        """Test that vectorize_schema is deterministic.
        
        **Validates: Requirement 1.1**
        """
        import numpy as np
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="make")]
                        )
                    ]
                )
            ]
        )
        
        # Create two separate vectorizers with same config
        encoder1 = Encoder(config)
        vectorizer1 = SchemaVectorizer(encoder1, config)
        
        encoder2 = Encoder(config)
        vectorizer2 = SchemaVectorizer(encoder2, config)
        
        result1 = vectorizer1.vectorize_schema()
        result2 = vectorizer2.vectorize_schema()
        
        # Same config should produce same vectors
        assert np.array_equal(result1["make"].vector.data, result2["make"].vector.data)
    
    def test_vectorize_schema_different_roles_produce_different_vectors(self):
        """Test that different roles produce different vectors.
        
        **Validates: Requirement 1.1**
        """
        import numpy as np
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[
                                Role(name="make"),
                                Role(name="model"),
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Different roles should have different vectors
        assert not np.array_equal(
            result["make"].vector.data,
            result["model"].vector.data
        )
    
    def test_vectorize_schema_same_role_name_in_different_paths(self):
        """Test that same role name in different paths produces same vector.
        
        The vector is generated from the role name, not the full path.
        
        **Validates: Requirement 1.1**
        """
        import numpy as np
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="name")]
                        )
                    ]
                ),
                Layer(
                    name="parts",
                    segments=[
                        Segment(
                            name="catalog",
                            roles=[Role(name="name")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Only one "name" key should exist (last one wins)
        assert "name" in result
        # The role_path will be from the last occurrence
        assert result["name"].role_path == "parts.catalog.name"
    
    def test_vectorize_schema_complex_hierarchy(self):
        """Test vectorize_schema with complex hierarchy.
        
        **Validates: Requirement 1.1**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[
                                Role(name="make"),
                                Role(name="model"),
                            ]
                        ),
                        Segment(
                            name="specs",
                            roles=[
                                Role(name="engine"),
                                Role(name="transmission"),
                            ]
                        )
                    ]
                ),
                Layer(
                    name="parts",
                    segments=[
                        Segment(
                            name="catalog",
                            roles=[
                                Role(name="category"),
                                Role(name="price"),
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # All 6 roles should be present
        assert len(result) == 6
        assert "make" in result
        assert "model" in result
        assert "engine" in result
        assert "transmission" in result
        assert "category" in result
        assert "price" in result
        
        # Verify paths
        assert result["make"].role_path == "vehicle.identity.make"
        assert result["model"].role_path == "vehicle.identity.model"
        assert result["engine"].role_path == "vehicle.specs.engine"
        assert result["transmission"].role_path == "vehicle.specs.transmission"
        assert result["category"].role_path == "parts.catalog.category"
        assert result["price"].role_path == "parts.catalog.price"
    
    def test_vectorize_schema_get_schema_vectors_returns_same_cache(self):
        """Test that get_schema_vectors returns the same cache after vectorize_schema.
        
        **Validates: Requirement 1.4**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[Role(name="make")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Call vectorize_schema first
        schema_result = vectorizer.vectorize_schema()
        
        # get_schema_vectors should return the same cache
        get_result = vectorizer.get_schema_vectors()
        
        assert schema_result is get_result


class TestNestedRolePathSupport:
    """Test nested role path support for SchemaVectorizer.
    
    **Validates: Requirement 1.6** - THE SDK SHALL support generating vectors
    for nested role paths (e.g., "vehicle.identity.make")
    """
    
    def test_nested_path_three_levels(self):
        """Test vectorize_role with standard three-level path.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("vehicle.identity.make")
        
        assert result.key == "make"
        assert result.role_path == "vehicle.identity.make"
        assert result.element_type == "role"
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_nested_path_four_levels(self):
        """Test vectorize_role with four-level path.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("domain.layer.segment.role")
        
        assert result.key == "role"
        assert result.role_path == "domain.layer.segment.role"
        assert result.element_type == "role"
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_nested_path_five_levels(self):
        """Test vectorize_role with five-level path (a.b.c.d.e).
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("a.b.c.d.e")
        
        assert result.key == "e"
        assert result.role_path == "a.b.c.d.e"
        assert result.element_type == "role"
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_nested_path_deep_hierarchy(self):
        """Test vectorize_role with very deep path (10 levels).
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        deep_path = "level1.level2.level3.level4.level5.level6.level7.level8.level9.role"
        result = vectorizer.vectorize_role(deep_path)
        
        assert result.key == "role"
        assert result.role_path == deep_path
        assert result.element_type == "role"
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_nested_paths_produce_distinct_vectors(self):
        """Test that different nested paths produce distinct vectors.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Different paths with same role name
        path1 = vectorizer.vectorize_role("vehicle.identity.make")
        path2 = vectorizer.vectorize_role("product.details.make")
        path3 = vectorizer.vectorize_role("inventory.item.make")
        
        # All should have the same key (role name)
        assert path1.key == path2.key == path3.key == "make"
        
        # But different vectors (distinct for full paths)
        assert not np.array_equal(path1.vector.data, path2.vector.data)
        assert not np.array_equal(path1.vector.data, path3.vector.data)
        assert not np.array_equal(path2.vector.data, path3.vector.data)
    
    def test_nested_path_vector_distinct_from_components(self):
        """Test that full path vector is distinct from individual component vectors.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Full path
        full_path = vectorizer.vectorize_role("vehicle.identity.make")
        
        # Individual components
        vehicle_only = vectorizer.vectorize_role("vehicle")
        identity_only = vectorizer.vectorize_role("identity")
        make_only = vectorizer.vectorize_role("make")
        
        # Full path vector should be distinct from all component vectors
        assert not np.array_equal(full_path.vector.data, vehicle_only.vector.data)
        assert not np.array_equal(full_path.vector.data, identity_only.vector.data)
        assert not np.array_equal(full_path.vector.data, make_only.vector.data)
    
    def test_nested_path_deterministic(self):
        """Test that same nested path always produces same vector.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        path = "vehicle.identity.make"
        result1 = vectorizer.vectorize_role(path)
        result2 = vectorizer.vectorize_role(path)
        
        # Same path should produce identical vectors
        assert np.array_equal(result1.vector.data, result2.vector.data)
    
    def test_nested_path_different_depths_different_vectors(self):
        """Test that paths of different depths produce different vectors.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Paths of different depths ending in same role name
        depth1 = vectorizer.vectorize_role("make")
        depth2 = vectorizer.vectorize_role("identity.make")
        depth3 = vectorizer.vectorize_role("vehicle.identity.make")
        depth4 = vectorizer.vectorize_role("domain.vehicle.identity.make")
        
        # All have same key
        assert depth1.key == depth2.key == depth3.key == depth4.key == "make"
        
        # But different vectors
        vectors = [depth1.vector.data, depth2.vector.data, depth3.vector.data, depth4.vector.data]
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                assert not np.array_equal(vectors[i], vectors[j])
    
    def test_nested_path_value_vectorization(self):
        """Test that nested paths work correctly for value vectorization.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Value with nested path
        result = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        
        assert result.key == "make=Toyota"
        assert result.role_path == "vehicle.identity.make"
        assert result.element_type == "value"
        assert result.original_value == "Toyota"
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_nested_path_value_same_value_different_paths(self):
        """Test that same value in different nested paths has same vector.
        
        Value vectors are generated from the value, not the path.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Same value "Toyota" in different paths
        result1 = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
        result2 = vectorizer.vectorize_value("product.details.brand", "Toyota")
        
        # Same value produces same vector
        assert np.array_equal(result1.vector.data, result2.vector.data)
        
        # But different keys and paths
        assert result1.key != result2.key
        assert result1.role_path != result2.role_path
    
    def test_vectorize_schema_with_nested_paths(self):
        """Test that vectorize_schema correctly handles nested paths.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(
                            name="identity",
                            roles=[
                                Role(name="make"),
                                Role(name="model"),
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_schema()
        
        # Verify nested paths are preserved
        assert result["make"].role_path == "vehicle.identity.make"
        assert result["model"].role_path == "vehicle.identity.model"
        
        # Verify vectors are distinct
        assert not np.array_equal(
            result["make"].vector.data,
            result["model"].vector.data
        )
    
    def test_nested_path_with_underscores(self):
        """Test nested paths with underscores in component names.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("vehicle_data.identity_info.make_name")
        
        assert result.key == "make_name"
        assert result.role_path == "vehicle_data.identity_info.make_name"
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_nested_path_with_numbers(self):
        """Test nested paths with numbers in component names.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_role("layer1.segment2.role3")
        
        assert result.key == "role3"
        assert result.role_path == "layer1.segment2.role3"
        assert np.all(np.isin(result.vector.data, [-1, 1]))
    
    def test_nested_path_preserves_case(self):
        """Test that nested paths preserve case sensitivity.
        
        **Validates: Requirement 1.6**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Different cases should produce different vectors
        lower = vectorizer.vectorize_role("vehicle.identity.make")
        upper = vectorizer.vectorize_role("Vehicle.Identity.Make")
        mixed = vectorizer.vectorize_role("VEHICLE.identity.Make")
        
        # Different cases produce different vectors
        assert not np.array_equal(lower.vector.data, upper.vector.data)
        assert not np.array_equal(lower.vector.data, mixed.vector.data)
        assert not np.array_equal(upper.vector.data, mixed.vector.data)


class TestSchemaVectorizerVectorizeCompound:
    """Test SchemaVectorizer.vectorize_compound() method.
    
    **Validates: Requirement 10.1** - THE SDK SHALL generate vectors for
    multi-word values as single units
    """
    
    def test_vectorize_compound_returns_vector(self):
        """Test that vectorize_compound returns a Vector instance.
        
        **Validates: Requirement 10.1**
        """
        from glyphh.core.types import Vector
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_compound(["Brake", "Pads"])
        
        assert isinstance(result, Vector)
    
    def test_vectorize_compound_generates_bipolar_vector(self):
        """Test that vectorize_compound generates a bipolar vector {-1, +1}.
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_compound(["Brake", "Pads"])
        
        # All values should be in {-1, +1}
        assert np.all(np.isin(result.data, [-1, 1]))
    
    def test_vectorize_compound_vector_has_correct_dimension(self):
        """Test that vectorize_compound generates vector with correct dimension.
        
        **Validates: Requirement 10.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_compound(["Brake", "Pads"])
        
        assert result.dimension == 1000
    
    def test_vectorize_compound_vector_has_correct_space_id(self):
        """Test that vectorize_compound generates vector with encoder's space_id.
        
        **Validates: Requirement 10.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_compound(["Brake", "Pads"])
        
        assert result.space_id == encoder.space_id
    
    def test_vectorize_compound_with_two_words(self):
        """Test vectorize_compound with two words.
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_compound(["Brake", "Pads"])
        
        assert result.dimension == 1000
        assert np.all(np.isin(result.data, [-1, 1]))
    
    def test_vectorize_compound_with_three_words(self):
        """Test vectorize_compound with three words.
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_compound(["Toyota", "Camry", "XLE"])
        
        assert result.dimension == 1000
        assert np.all(np.isin(result.data, [-1, 1]))
    
    def test_vectorize_compound_with_single_word(self):
        """Test vectorize_compound with a single word.
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_compound(["Toyota"])
        
        assert result.dimension == 1000
        assert np.all(np.isin(result.data, [-1, 1]))
        
        # Single word compound should be equivalent to generate_symbol
        single_symbol = encoder.generate_symbol("Toyota")
        assert np.array_equal(result.data, single_symbol.data)
    
    def test_vectorize_compound_is_deterministic(self):
        """Test that vectorize_compound is deterministic (same input → same output).
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result1 = vectorizer.vectorize_compound(["Brake", "Pads"])
        result2 = vectorizer.vectorize_compound(["Brake", "Pads"])
        
        # Same inputs should produce same vector
        assert np.array_equal(result1.data, result2.data)
    
    def test_vectorize_compound_word_order_invariant(self):
        """Test that vectorize_compound is word order invariant (bundling is commutative).
        
        **Validates: Requirement 10.1, 10.5**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Different word orders should produce same vector
        result1 = vectorizer.vectorize_compound(["Brake", "Pads"])
        result2 = vectorizer.vectorize_compound(["Pads", "Brake"])
        
        # Bundling is commutative, so word order doesn't matter
        assert np.array_equal(result1.data, result2.data)
    
    def test_vectorize_compound_different_words_produce_different_vectors(self):
        """Test that different word combinations produce different vectors.
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        brake_pads = vectorizer.vectorize_compound(["Brake", "Pads"])
        oil_filter = vectorizer.vectorize_compound(["Oil", "Filter"])
        
        # Different word combinations should produce different vectors
        assert not np.array_equal(brake_pads.data, oil_filter.data)
    
    def test_vectorize_compound_differs_from_individual_words(self):
        """Test that compound vector differs from individual word vectors.
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        compound = vectorizer.vectorize_compound(["Brake", "Pads"])
        brake_only = encoder.generate_symbol("Brake")
        pads_only = encoder.generate_symbol("Pads")
        
        # Compound vector should differ from individual word vectors
        assert not np.array_equal(compound.data, brake_only.data)
        assert not np.array_equal(compound.data, pads_only.data)
    
    def test_vectorize_compound_raises_value_error_for_empty_list(self):
        """Test that vectorize_compound raises ValueError for empty list.
        
        **Validates: Requirement 10.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.vectorize_compound([])
        
        assert "words list cannot be empty" in str(exc_info.value)
    
    def test_vectorize_compound_raises_value_error_for_empty_word(self):
        """Test that vectorize_compound raises ValueError for empty word in list.
        
        **Validates: Requirement 10.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.vectorize_compound(["Brake", "", "Pads"])
        
        assert "word at index 1 must be a non-empty string" in str(exc_info.value)
    
    def test_vectorize_compound_raises_value_error_for_whitespace_only_word(self):
        """Test that vectorize_compound raises ValueError for whitespace-only word.
        
        **Validates: Requirement 10.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.vectorize_compound(["Brake", "   ", "Pads"])
        
        assert "word at index 1 must be a non-empty string" in str(exc_info.value)
    
    def test_vectorize_compound_with_different_dimensions(self):
        """Test vectorize_compound with different encoder dimensions.
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        # Test with smaller dimension
        config_small = EncoderConfig(dimension=500, seed=42)
        encoder_small = Encoder(config_small)
        vectorizer_small = SchemaVectorizer(encoder_small, config_small)
        
        result_small = vectorizer_small.vectorize_compound(["Brake", "Pads"])
        assert result_small.dimension == 500
        assert np.all(np.isin(result_small.data, [-1, 1]))
        
        # Test with larger dimension
        config_large = EncoderConfig(dimension=5000, seed=42)
        encoder_large = Encoder(config_large)
        vectorizer_large = SchemaVectorizer(encoder_large, config_large)
        
        result_large = vectorizer_large.vectorize_compound(["Brake", "Pads"])
        assert result_large.dimension == 5000
        assert np.all(np.isin(result_large.data, [-1, 1]))
    
    def test_vectorize_compound_with_special_characters(self):
        """Test vectorize_compound with words containing special characters.
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_compound(["O'Reilly", "Auto", "Parts"])
        
        assert result.dimension == 1000
        assert np.all(np.isin(result.data, [-1, 1]))
    
    def test_vectorize_compound_with_numeric_strings(self):
        """Test vectorize_compound with numeric string words.
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_compound(["Model", "2024"])
        
        assert result.dimension == 1000
        assert np.all(np.isin(result.data, [-1, 1]))
    
    def test_vectorize_compound_preserves_case(self):
        """Test that vectorize_compound preserves case sensitivity.
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Different cases should produce different vectors
        lower = vectorizer.vectorize_compound(["brake", "pads"])
        upper = vectorizer.vectorize_compound(["BRAKE", "PADS"])
        mixed = vectorizer.vectorize_compound(["Brake", "Pads"])
        
        # Different cases produce different vectors
        assert not np.array_equal(lower.data, upper.data)
        assert not np.array_equal(lower.data, mixed.data)
        assert not np.array_equal(upper.data, mixed.data)
    
    def test_vectorize_compound_with_many_words(self):
        """Test vectorize_compound with many words (5+).
        
        **Validates: Requirement 10.1**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        result = vectorizer.vectorize_compound([
            "Toyota", "Camry", "XLE", "V6", "Premium"
        ])
        
        assert result.dimension == 1000
        assert np.all(np.isin(result.data, [-1, 1]))
    
    def test_vectorize_compound_three_word_order_invariant(self):
        """Test that three-word compound is order invariant.
        
        **Validates: Requirement 10.1, 10.5**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # All permutations should produce same vector
        result1 = vectorizer.vectorize_compound(["A", "B", "C"])
        result2 = vectorizer.vectorize_compound(["B", "C", "A"])
        result3 = vectorizer.vectorize_compound(["C", "A", "B"])
        result4 = vectorizer.vectorize_compound(["A", "C", "B"])
        result5 = vectorizer.vectorize_compound(["B", "A", "C"])
        result6 = vectorizer.vectorize_compound(["C", "B", "A"])
        
        # All permutations should be equal
        assert np.array_equal(result1.data, result2.data)
        assert np.array_equal(result1.data, result3.data)
        assert np.array_equal(result1.data, result4.data)
        assert np.array_equal(result1.data, result5.data)
        assert np.array_equal(result1.data, result6.data)


class TestSchemaVectorizerConfigChangeDetection:
    """Test SchemaVectorizer config change detection for cache invalidation.
    
    **Validates: Requirements 1.4, 1.5**
    - 1.4: THE SDK SHALL cache schema vectors to avoid regeneration on each query
    - 1.5: THE SDK SHALL regenerate schema vectors when the model config changes
    """
    
    def test_compute_config_hash_returns_string(self):
        """Test that _compute_config_hash returns a string.
        
        **Validates: Requirements 1.4, 1.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        hash_value = vectorizer._compute_config_hash()
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex digest length
    
    def test_compute_config_hash_is_deterministic(self):
        """Test that _compute_config_hash is deterministic.
        
        Same config should always produce the same hash.
        
        **Validates: Requirements 1.4, 1.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        hash1 = vectorizer._compute_config_hash()
        hash2 = vectorizer._compute_config_hash()
        
        assert hash1 == hash2
    
    def test_compute_config_hash_different_for_different_dimension(self):
        """Test that different dimensions produce different hashes.
        
        **Validates: Requirements 1.4, 1.5**
        """
        config1 = EncoderConfig(dimension=1000, seed=42)
        encoder1 = Encoder(config1)
        vectorizer1 = SchemaVectorizer(encoder1, config1)
        
        config2 = EncoderConfig(dimension=2000, seed=42)
        encoder2 = Encoder(config2)
        vectorizer2 = SchemaVectorizer(encoder2, config2)
        
        hash1 = vectorizer1._compute_config_hash()
        hash2 = vectorizer2._compute_config_hash()
        
        assert hash1 != hash2
    
    def test_compute_config_hash_different_for_different_seed(self):
        """Test that different seeds produce different hashes.
        
        **Validates: Requirements 1.4, 1.5**
        """
        config1 = EncoderConfig(dimension=1000, seed=42)
        encoder1 = Encoder(config1)
        vectorizer1 = SchemaVectorizer(encoder1, config1)
        
        config2 = EncoderConfig(dimension=1000, seed=123)
        encoder2 = Encoder(config2)
        vectorizer2 = SchemaVectorizer(encoder2, config2)
        
        hash1 = vectorizer1._compute_config_hash()
        hash2 = vectorizer2._compute_config_hash()
        
        assert hash1 != hash2
    
    def test_compute_config_hash_different_for_different_layers(self):
        """Test that different layer structures produce different hashes.
        
        **Validates: Requirements 1.4, 1.5**
        """
        config1 = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="layer1",
                    segments=[
                        Segment(name="seg1", roles=[Role(name="role1")])
                    ]
                )
            ]
        )
        encoder1 = Encoder(config1)
        vectorizer1 = SchemaVectorizer(encoder1, config1)
        
        config2 = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="layer1",
                    segments=[
                        Segment(name="seg1", roles=[Role(name="role2")])
                    ]
                )
            ]
        )
        encoder2 = Encoder(config2)
        vectorizer2 = SchemaVectorizer(encoder2, config2)
        
        hash1 = vectorizer1._compute_config_hash()
        hash2 = vectorizer2._compute_config_hash()
        
        assert hash1 != hash2
    
    def test_get_config_hash_returns_none_before_vectorize_schema(self):
        """Test that get_config_hash returns None before vectorize_schema is called.
        
        **Validates: Requirements 1.4, 1.5**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(name="identity", roles=[Role(name="make")])
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        assert vectorizer.get_config_hash() is None
    
    def test_get_config_hash_returns_hash_after_vectorize_schema(self):
        """Test that get_config_hash returns hash after vectorize_schema is called.
        
        **Validates: Requirements 1.4, 1.5**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(name="identity", roles=[Role(name="make")])
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.vectorize_schema()
        
        hash_value = vectorizer.get_config_hash()
        assert hash_value is not None
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64
    
    def test_vectorize_schema_caches_vectors(self):
        """Test that vectorize_schema caches vectors on subsequent calls.
        
        **Validates: Requirement 1.4**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(name="identity", roles=[Role(name="make")])
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectors1 = vectorizer.vectorize_schema()
        vectors2 = vectorizer.vectorize_schema()
        
        # Should return the same dictionary reference (cached)
        assert vectors1 is vectors2
    
    def test_vectorize_schema_regenerates_on_config_change(self):
        """Test that vectorize_schema regenerates vectors when config changes.
        
        This test simulates a config change by modifying the config object
        after initial vectorization.
        
        **Validates: Requirement 1.5**
        """
        import numpy as np
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(name="identity", roles=[Role(name="make")])
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # First vectorization
        vectors1 = vectorizer.vectorize_schema()
        hash1 = vectorizer.get_config_hash()
        original_make_vector = vectors1["make"].vector.data.copy()
        
        # Modify the config by adding a new layer
        config.layers.append(
            Layer(
                name="product",
                segments=[
                    Segment(name="details", roles=[Role(name="category")])
                ]
            )
        )
        
        # Second vectorization should detect config change and regenerate
        vectors2 = vectorizer.vectorize_schema()
        hash2 = vectorizer.get_config_hash()
        
        # Hash should be different after config change
        assert hash1 != hash2
        
        # New role should be present
        assert "category" in vectors2
        
        # Original role should still be present with same vector
        assert "make" in vectors2
        assert np.array_equal(vectors2["make"].vector.data, original_make_vector)
    
    def test_vectorize_schema_regenerates_on_role_addition(self):
        """Test that adding a role triggers regeneration.
        
        **Validates: Requirement 1.5**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(name="identity", roles=[Role(name="make")])
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # First vectorization
        vectors1 = vectorizer.vectorize_schema()
        assert "make" in vectors1
        assert "model" not in vectors1
        
        # Add a new role
        config.layers[0].segments[0].roles.append(Role(name="model"))
        
        # Second vectorization should detect change and regenerate
        vectors2 = vectorizer.vectorize_schema()
        
        # Both roles should now be present
        assert "make" in vectors2
        assert "model" in vectors2
    
    def test_vectorize_schema_regenerates_on_dimension_change(self):
        """Test that changing dimension triggers regeneration.
        
        Note: This test creates a new vectorizer with a different config
        to simulate what would happen if the config dimension changed.
        
        **Validates: Requirement 1.5**
        """
        import numpy as np
        
        config1 = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(name="identity", roles=[Role(name="make")])
                    ]
                )
            ]
        )
        encoder1 = Encoder(config1)
        vectorizer1 = SchemaVectorizer(encoder1, config1)
        
        vectors1 = vectorizer1.vectorize_schema()
        assert vectors1["make"].vector.dimension == 1000
        
        # Create new config with different dimension
        config2 = EncoderConfig(
            dimension=2000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(name="identity", roles=[Role(name="make")])
                    ]
                )
            ]
        )
        encoder2 = Encoder(config2)
        vectorizer2 = SchemaVectorizer(encoder2, config2)
        
        vectors2 = vectorizer2.vectorize_schema()
        assert vectors2["make"].vector.dimension == 2000
        
        # Hashes should be different
        assert vectorizer1.get_config_hash() != vectorizer2.get_config_hash()
    
    def test_vectorize_schema_no_regeneration_without_change(self):
        """Test that vectorize_schema doesn't regenerate without config change.
        
        **Validates: Requirement 1.4**
        """
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="vehicle",
                    segments=[
                        Segment(name="identity", roles=[Role(name="make")])
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # First vectorization
        vectors1 = vectorizer.vectorize_schema()
        hash1 = vectorizer.get_config_hash()
        
        # Multiple subsequent calls
        vectors2 = vectorizer.vectorize_schema()
        hash2 = vectorizer.get_config_hash()
        
        vectors3 = vectorizer.vectorize_schema()
        hash3 = vectorizer.get_config_hash()
        
        # All should return the same cached dictionary
        assert vectors1 is vectors2
        assert vectors2 is vectors3
        
        # Hash should remain the same
        assert hash1 == hash2 == hash3
    
    def test_config_hash_includes_similarity_weight(self):
        """Test that config hash includes similarity_weight.
        
        **Validates: Requirements 1.4, 1.5**
        """
        config1 = EncoderConfig(dimension=1000, seed=42, similarity_weight=0.5)
        encoder1 = Encoder(config1)
        vectorizer1 = SchemaVectorizer(encoder1, config1)
        
        config2 = EncoderConfig(dimension=1000, seed=42, similarity_weight=0.8)
        encoder2 = Encoder(config2)
        vectorizer2 = SchemaVectorizer(encoder2, config2)
        
        hash1 = vectorizer1._compute_config_hash()
        hash2 = vectorizer2._compute_config_hash()
        
        assert hash1 != hash2
    
    def test_config_hash_includes_security_weight(self):
        """Test that config hash includes security_weight.
        
        **Validates: Requirements 1.4, 1.5**
        """
        config1 = EncoderConfig(dimension=1000, seed=42, security_weight=0.5)
        encoder1 = Encoder(config1)
        vectorizer1 = SchemaVectorizer(encoder1, config1)
        
        config2 = EncoderConfig(dimension=1000, seed=42, security_weight=0.8)
        encoder2 = Encoder(config2)
        vectorizer2 = SchemaVectorizer(encoder2, config2)
        
        hash1 = vectorizer1._compute_config_hash()
        hash2 = vectorizer2._compute_config_hash()
        
        assert hash1 != hash2
    
    def test_config_hash_includes_role_weights(self):
        """Test that config hash includes role weights.
        
        **Validates: Requirements 1.4, 1.5**
        """
        config1 = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="layer1",
                    segments=[
                        Segment(
                            name="seg1",
                            roles=[Role(name="role1", similarity_weight=0.5)]
                        )
                    ]
                )
            ]
        )
        encoder1 = Encoder(config1)
        vectorizer1 = SchemaVectorizer(encoder1, config1)
        
        config2 = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="layer1",
                    segments=[
                        Segment(
                            name="seg1",
                            roles=[Role(name="role1", similarity_weight=0.9)]
                        )
                    ]
                )
            ]
        )
        encoder2 = Encoder(config2)
        vectorizer2 = SchemaVectorizer(encoder2, config2)
        
        hash1 = vectorizer1._compute_config_hash()
        hash2 = vectorizer2._compute_config_hash()
        
        assert hash1 != hash2


class TestSchemaVectorizerAddSynonym:
    """Test SchemaVectorizer.add_synonym() method.
    
    **Validates: Requirements 11.2, 11.3**
    - 11.2: THE SDK SHALL support defining synonyms for values
    - 11.3: WHEN generating schema vectors, THE SDK SHALL include alias/synonym vectors
    """
    
    def test_add_synonym_stores_synonym_vector(self):
        """Test that add_synonym stores a synonym vector in schema vectors.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        
        vectors = vectorizer.get_schema_vectors()
        assert "synonym:TYT" in vectors
    
    def test_add_synonym_links_to_primary_value(self):
        """Test that synonym vector's original_value is the primary value.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        
        vectors = vectorizer.get_schema_vectors()
        synonym_vector = vectors["synonym:TYT"]
        
        # original_value should be the primary value, not the synonym
        assert synonym_vector.original_value == "Toyota"
    
    def test_add_synonym_sets_element_type_to_value(self):
        """Test that synonym vector has element_type 'value'.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        
        vectors = vectorizer.get_schema_vectors()
        synonym_vector = vectors["synonym:TYT"]
        
        assert synonym_vector.element_type == "value"
    
    def test_add_synonym_sets_role_path_to_synonym(self):
        """Test that synonym vector has role_path 'synonym'.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        
        vectors = vectorizer.get_schema_vectors()
        synonym_vector = vectors["synonym:TYT"]
        
        assert synonym_vector.role_path == "synonym"
    
    def test_add_synonym_generates_bipolar_vector(self):
        """Test that add_synonym generates a bipolar vector {-1, +1}.
        
        **Validates: Requirements 11.2, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        
        vectors = vectorizer.get_schema_vectors()
        synonym_vector = vectors["synonym:TYT"]
        
        # All values should be in {-1, +1}
        assert np.all(np.isin(synonym_vector.vector.data, [-1, 1]))
    
    def test_add_synonym_vector_has_correct_dimension(self):
        """Test that synonym vector has correct dimension.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        
        vectors = vectorizer.get_schema_vectors()
        synonym_vector = vectors["synonym:TYT"]
        
        assert synonym_vector.vector.dimension == 1000
    
    def test_add_synonym_vector_has_correct_space_id(self):
        """Test that synonym vector has encoder's space_id.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        
        vectors = vectorizer.get_schema_vectors()
        synonym_vector = vectors["synonym:TYT"]
        
        assert synonym_vector.vector.space_id == encoder.space_id
    
    def test_add_synonym_tracks_synonym_mapping(self):
        """Test that add_synonym tracks the synonym → primary mapping.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        
        synonyms = vectorizer.get_synonyms()
        assert "TYT" in synonyms
        assert synonyms["TYT"] == "Toyota"
    
    def test_add_synonym_multiple_synonyms_for_same_primary(self):
        """Test adding multiple synonyms for the same primary value.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        vectorizer.add_synonym("Toyota", "Toy")
        vectorizer.add_synonym("Toyota", "TOYOTA")
        
        vectors = vectorizer.get_schema_vectors()
        assert "synonym:TYT" in vectors
        assert "synonym:Toy" in vectors
        assert "synonym:TOYOTA" in vectors
        
        # All should link to the same primary value
        assert vectors["synonym:TYT"].original_value == "Toyota"
        assert vectors["synonym:Toy"].original_value == "Toyota"
        assert vectors["synonym:TOYOTA"].original_value == "Toyota"
        
        synonyms = vectorizer.get_synonyms()
        assert synonyms["TYT"] == "Toyota"
        assert synonyms["Toy"] == "Toyota"
        assert synonyms["TOYOTA"] == "Toyota"
    
    def test_add_synonym_different_primary_values(self):
        """Test adding synonyms for different primary values.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        vectorizer.add_synonym("Honda", "HND")
        vectorizer.add_synonym("Ford", "FRD")
        
        vectors = vectorizer.get_schema_vectors()
        assert vectors["synonym:TYT"].original_value == "Toyota"
        assert vectors["synonym:HND"].original_value == "Honda"
        assert vectors["synonym:FRD"].original_value == "Ford"
        
        synonyms = vectorizer.get_synonyms()
        assert synonyms["TYT"] == "Toyota"
        assert synonyms["HND"] == "Honda"
        assert synonyms["FRD"] == "Ford"
    
    def test_add_synonym_overwrites_existing_synonym(self):
        """Test that adding the same synonym twice overwrites the previous entry.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add synonym mapping TYT → Toyota
        vectorizer.add_synonym("Toyota", "TYT")
        assert vectorizer.get_synonyms()["TYT"] == "Toyota"
        
        # Overwrite with TYT → Honda
        vectorizer.add_synonym("Honda", "TYT")
        assert vectorizer.get_synonyms()["TYT"] == "Honda"
        
        # Vector should now link to Honda
        vectors = vectorizer.get_schema_vectors()
        assert vectors["synonym:TYT"].original_value == "Honda"
    
    def test_add_synonym_raises_value_error_for_empty_primary(self):
        """Test that add_synonym raises ValueError for empty primary.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.add_synonym("", "TYT")
        
        assert "primary must be a non-empty string" in str(exc_info.value)
    
    def test_add_synonym_raises_value_error_for_whitespace_only_primary(self):
        """Test that add_synonym raises ValueError for whitespace-only primary.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.add_synonym("   ", "TYT")
        
        assert "primary must be a non-empty string" in str(exc_info.value)
    
    def test_add_synonym_raises_value_error_for_empty_synonym(self):
        """Test that add_synonym raises ValueError for empty synonym.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.add_synonym("Toyota", "")
        
        assert "synonym must be a non-empty string" in str(exc_info.value)
    
    def test_add_synonym_raises_value_error_for_whitespace_only_synonym(self):
        """Test that add_synonym raises ValueError for whitespace-only synonym.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.add_synonym("Toyota", "   ")
        
        assert "synonym must be a non-empty string" in str(exc_info.value)
    
    def test_add_synonym_vector_generated_from_synonym_text(self):
        """Test that the vector is generated from the synonym text, not primary.
        
        This is important because queries containing the synonym should match
        against the synonym vector.
        
        **Validates: Requirements 11.2, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        
        # Get the synonym vector
        synonym_vector = vectorizer.get_schema_vectors()["synonym:TYT"]
        
        # Generate a vector directly from "TYT"
        expected_vector = encoder.generate_symbol("TYT")
        
        # The synonym vector should match the vector generated from "TYT"
        assert np.array_equal(synonym_vector.vector.data, expected_vector.data)
        
        # It should NOT match the vector generated from "Toyota"
        toyota_vector = encoder.generate_symbol("Toyota")
        assert not np.array_equal(synonym_vector.vector.data, toyota_vector.data)
    
    def test_add_synonym_with_multi_word_synonym(self):
        """Test add_synonym with a multi-word synonym.
        
        **Validates: Requirements 11.2, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Brake Pads", "brake shoes")
        
        vectors = vectorizer.get_schema_vectors()
        assert "synonym:brake shoes" in vectors
        
        synonym_vector = vectors["synonym:brake shoes"]
        assert synonym_vector.original_value == "Brake Pads"
        assert np.all(np.isin(synonym_vector.vector.data, [-1, 1]))
    
    def test_add_synonym_with_special_characters(self):
        """Test add_synonym with special characters in synonym.
        
        **Validates: Requirements 11.2, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("O'Reilly Auto Parts", "O'Reilly")
        
        vectors = vectorizer.get_schema_vectors()
        assert "synonym:O'Reilly" in vectors
        
        synonym_vector = vectors["synonym:O'Reilly"]
        assert synonym_vector.original_value == "O'Reilly Auto Parts"
        assert np.all(np.isin(synonym_vector.vector.data, [-1, 1]))
    
    def test_add_synonym_with_numeric_synonym(self):
        """Test add_synonym with numeric string synonym.
        
        **Validates: Requirements 11.2, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Model Year 2024", "2024")
        
        vectors = vectorizer.get_schema_vectors()
        assert "synonym:2024" in vectors
        
        synonym_vector = vectors["synonym:2024"]
        assert synonym_vector.original_value == "Model Year 2024"
        assert np.all(np.isin(synonym_vector.vector.data, [-1, 1]))
    
    def test_add_synonym_deterministic(self):
        """Test that add_synonym is deterministic (same input → same output).
        
        **Validates: Requirements 11.2, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create two vectorizers with same config
        vectorizer1 = SchemaVectorizer(encoder, config)
        vectorizer2 = SchemaVectorizer(encoder, config)
        
        vectorizer1.add_synonym("Toyota", "TYT")
        vectorizer2.add_synonym("Toyota", "TYT")
        
        vector1 = vectorizer1.get_schema_vectors()["synonym:TYT"]
        vector2 = vectorizer2.get_schema_vectors()["synonym:TYT"]
        
        # Same inputs should produce same vectors
        assert np.array_equal(vector1.vector.data, vector2.vector.data)
    
    def test_add_synonym_different_synonyms_produce_different_vectors(self):
        """Test that different synonyms produce different vectors.
        
        **Validates: Requirements 11.2, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_synonym("Toyota", "TYT")
        vectorizer.add_synonym("Toyota", "Toy")
        
        vectors = vectorizer.get_schema_vectors()
        tyt_vector = vectors["synonym:TYT"]
        toy_vector = vectors["synonym:Toy"]
        
        # Different synonyms should produce different vectors
        assert not np.array_equal(tyt_vector.vector.data, toy_vector.vector.data)
    
    def test_add_synonym_with_different_dimensions(self):
        """Test add_synonym with different encoder dimensions.
        
        **Validates: Requirements 11.2, 11.3**
        """
        import numpy as np
        
        # Test with smaller dimension
        config_small = EncoderConfig(dimension=500, seed=42)
        encoder_small = Encoder(config_small)
        vectorizer_small = SchemaVectorizer(encoder_small, config_small)
        
        vectorizer_small.add_synonym("Toyota", "TYT")
        vector_small = vectorizer_small.get_schema_vectors()["synonym:TYT"]
        assert vector_small.vector.dimension == 500
        assert np.all(np.isin(vector_small.vector.data, [-1, 1]))
        
        # Test with larger dimension
        config_large = EncoderConfig(dimension=5000, seed=42)
        encoder_large = Encoder(config_large)
        vectorizer_large = SchemaVectorizer(encoder_large, config_large)
        
        vectorizer_large.add_synonym("Toyota", "TYT")
        vector_large = vectorizer_large.get_schema_vectors()["synonym:TYT"]
        assert vector_large.vector.dimension == 5000
        assert np.all(np.isin(vector_large.vector.data, [-1, 1]))


class TestSchemaVectorizerGetSynonyms:
    """Test SchemaVectorizer.get_synonyms() method.
    
    **Validates: Requirements 11.2, 11.3**
    """
    
    def test_get_synonyms_returns_empty_dict_initially(self):
        """Test that get_synonyms returns empty dict after init.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        synonyms = vectorizer.get_synonyms()
        
        assert isinstance(synonyms, dict)
        assert len(synonyms) == 0
    
    def test_get_synonyms_returns_same_dict_reference(self):
        """Test that get_synonyms returns the internal dict reference.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        synonyms1 = vectorizer.get_synonyms()
        synonyms2 = vectorizer.get_synonyms()
        
        assert synonyms1 is synonyms2
        assert synonyms1 is vectorizer._synonyms
    
    def test_get_synonyms_reflects_added_synonyms(self):
        """Test that get_synonyms reflects synonyms added via add_synonym.
        
        **Validates: Requirements 11.2, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Initially empty
        assert len(vectorizer.get_synonyms()) == 0
        
        # Add synonyms
        vectorizer.add_synonym("Toyota", "TYT")
        assert len(vectorizer.get_synonyms()) == 1
        assert vectorizer.get_synonyms()["TYT"] == "Toyota"
        
        vectorizer.add_synonym("Honda", "HND")
        assert len(vectorizer.get_synonyms()) == 2
        assert vectorizer.get_synonyms()["HND"] == "Honda"


class TestSchemaVectorizerAddAlias:
    """Test SchemaVectorizer.add_alias() method.
    
    **Validates: Requirements 11.1, 11.3**
    - 11.1: THE SDK SHALL support defining aliases for role names
    - 11.3: WHEN generating schema vectors, THE SDK SHALL include alias/synonym vectors
    """
    
    def test_add_alias_stores_alias_vector(self):
        """Test that add_alias stores an alias vector in schema vectors.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        
        vectors = vectorizer.get_schema_vectors()
        assert "alias:manufacturer" in vectors
    
    def test_add_alias_links_to_primary_role(self):
        """Test that alias vector's original_value is the primary role.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        
        vectors = vectorizer.get_schema_vectors()
        alias_vector = vectors["alias:manufacturer"]
        
        # original_value should be the primary role, not the alias
        assert alias_vector.original_value == "make"
    
    def test_add_alias_sets_element_type_to_role(self):
        """Test that alias vector has element_type 'role'.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        
        vectors = vectorizer.get_schema_vectors()
        alias_vector = vectors["alias:manufacturer"]
        
        assert alias_vector.element_type == "role"
    
    def test_add_alias_sets_role_path_to_alias(self):
        """Test that alias vector has role_path 'alias'.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        
        vectors = vectorizer.get_schema_vectors()
        alias_vector = vectors["alias:manufacturer"]
        
        assert alias_vector.role_path == "alias"
    
    def test_add_alias_generates_bipolar_vector(self):
        """Test that add_alias generates a bipolar vector {-1, +1}.
        
        **Validates: Requirements 11.1, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        
        vectors = vectorizer.get_schema_vectors()
        alias_vector = vectors["alias:manufacturer"]
        
        # All values should be in {-1, +1}
        assert np.all(np.isin(alias_vector.vector.data, [-1, 1]))
    
    def test_add_alias_vector_has_correct_dimension(self):
        """Test that alias vector has correct dimension.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        
        vectors = vectorizer.get_schema_vectors()
        alias_vector = vectors["alias:manufacturer"]
        
        assert alias_vector.vector.dimension == 1000
    
    def test_add_alias_vector_has_correct_space_id(self):
        """Test that alias vector has encoder's space_id.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        
        vectors = vectorizer.get_schema_vectors()
        alias_vector = vectors["alias:manufacturer"]
        
        assert alias_vector.vector.space_id == encoder.space_id
    
    def test_add_alias_tracks_alias_mapping(self):
        """Test that add_alias tracks the alias → primary role mapping.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        
        aliases = vectorizer.get_aliases()
        assert "manufacturer" in aliases
        assert aliases["manufacturer"] == "make"
    
    def test_add_alias_multiple_aliases_for_same_primary(self):
        """Test adding multiple aliases for the same primary role.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        vectorizer.add_alias("make", "brand")
        vectorizer.add_alias("make", "automaker")
        
        vectors = vectorizer.get_schema_vectors()
        assert "alias:manufacturer" in vectors
        assert "alias:brand" in vectors
        assert "alias:automaker" in vectors
        
        # All should link to the same primary role
        assert vectors["alias:manufacturer"].original_value == "make"
        assert vectors["alias:brand"].original_value == "make"
        assert vectors["alias:automaker"].original_value == "make"
        
        aliases = vectorizer.get_aliases()
        assert aliases["manufacturer"] == "make"
        assert aliases["brand"] == "make"
        assert aliases["automaker"] == "make"
    
    def test_add_alias_different_primary_roles(self):
        """Test adding aliases for different primary roles.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        vectorizer.add_alias("model", "product_name")
        vectorizer.add_alias("year", "model_year")
        
        vectors = vectorizer.get_schema_vectors()
        assert vectors["alias:manufacturer"].original_value == "make"
        assert vectors["alias:product_name"].original_value == "model"
        assert vectors["alias:model_year"].original_value == "year"
        
        aliases = vectorizer.get_aliases()
        assert aliases["manufacturer"] == "make"
        assert aliases["product_name"] == "model"
        assert aliases["model_year"] == "year"
    
    def test_add_alias_overwrites_existing_alias(self):
        """Test that adding the same alias twice overwrites the previous entry.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add alias mapping manufacturer → make
        vectorizer.add_alias("make", "manufacturer")
        assert vectorizer.get_aliases()["manufacturer"] == "make"
        
        # Overwrite with manufacturer → brand
        vectorizer.add_alias("brand", "manufacturer")
        assert vectorizer.get_aliases()["manufacturer"] == "brand"
        
        # Vector should now link to brand
        vectors = vectorizer.get_schema_vectors()
        assert vectors["alias:manufacturer"].original_value == "brand"
    
    def test_add_alias_raises_value_error_for_empty_role(self):
        """Test that add_alias raises ValueError for empty role.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.add_alias("", "manufacturer")
        
        assert "role must be a non-empty string" in str(exc_info.value)
    
    def test_add_alias_raises_value_error_for_whitespace_only_role(self):
        """Test that add_alias raises ValueError for whitespace-only role.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.add_alias("   ", "manufacturer")
        
        assert "role must be a non-empty string" in str(exc_info.value)
    
    def test_add_alias_raises_value_error_for_empty_alias(self):
        """Test that add_alias raises ValueError for empty alias.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.add_alias("make", "")
        
        assert "alias must be a non-empty string" in str(exc_info.value)
    
    def test_add_alias_raises_value_error_for_whitespace_only_alias(self):
        """Test that add_alias raises ValueError for whitespace-only alias.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.add_alias("make", "   ")
        
        assert "alias must be a non-empty string" in str(exc_info.value)
    
    def test_add_alias_vector_generated_from_alias_text(self):
        """Test that the vector is generated from the alias text, not primary role.
        
        This is important because queries containing the alias should match
        against the alias vector.
        
        **Validates: Requirements 11.1, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        
        # Get the alias vector
        alias_vector = vectorizer.get_schema_vectors()["alias:manufacturer"]
        
        # Generate a vector directly from "manufacturer"
        expected_vector = encoder.generate_symbol("manufacturer")
        
        # The alias vector should match the vector generated from "manufacturer"
        assert np.array_equal(alias_vector.vector.data, expected_vector.data)
        
        # It should NOT match the vector generated from "make"
        make_vector = encoder.generate_symbol("make")
        assert not np.array_equal(alias_vector.vector.data, make_vector.data)
    
    def test_add_alias_with_multi_word_alias(self):
        """Test add_alias with a multi-word alias.
        
        **Validates: Requirements 11.1, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "car manufacturer")
        
        vectors = vectorizer.get_schema_vectors()
        assert "alias:car manufacturer" in vectors
        
        alias_vector = vectors["alias:car manufacturer"]
        assert alias_vector.original_value == "make"
        assert np.all(np.isin(alias_vector.vector.data, [-1, 1]))
    
    def test_add_alias_with_special_characters(self):
        """Test add_alias with special characters in alias.
        
        **Validates: Requirements 11.1, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("category", "part-type")
        
        vectors = vectorizer.get_schema_vectors()
        assert "alias:part-type" in vectors
        
        alias_vector = vectors["alias:part-type"]
        assert alias_vector.original_value == "category"
        assert np.all(np.isin(alias_vector.vector.data, [-1, 1]))
    
    def test_add_alias_deterministic(self):
        """Test that add_alias is deterministic (same input → same output).
        
        **Validates: Requirements 11.1, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create two vectorizers with same config
        vectorizer1 = SchemaVectorizer(encoder, config)
        vectorizer2 = SchemaVectorizer(encoder, config)
        
        vectorizer1.add_alias("make", "manufacturer")
        vectorizer2.add_alias("make", "manufacturer")
        
        vector1 = vectorizer1.get_schema_vectors()["alias:manufacturer"]
        vector2 = vectorizer2.get_schema_vectors()["alias:manufacturer"]
        
        # Same inputs should produce same vectors
        assert np.array_equal(vector1.vector.data, vector2.vector.data)
    
    def test_add_alias_different_aliases_produce_different_vectors(self):
        """Test that different aliases produce different vectors.
        
        **Validates: Requirements 11.1, 11.3**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        vectorizer.add_alias("make", "manufacturer")
        vectorizer.add_alias("make", "brand")
        
        vectors = vectorizer.get_schema_vectors()
        manufacturer_vector = vectors["alias:manufacturer"]
        brand_vector = vectors["alias:brand"]
        
        # Different aliases should produce different vectors
        assert not np.array_equal(manufacturer_vector.vector.data, brand_vector.vector.data)
    
    def test_add_alias_with_different_dimensions(self):
        """Test add_alias with different encoder dimensions.
        
        **Validates: Requirements 11.1, 11.3**
        """
        import numpy as np
        
        # Test with smaller dimension
        config_small = EncoderConfig(dimension=500, seed=42)
        encoder_small = Encoder(config_small)
        vectorizer_small = SchemaVectorizer(encoder_small, config_small)
        
        vectorizer_small.add_alias("make", "manufacturer")
        vector_small = vectorizer_small.get_schema_vectors()["alias:manufacturer"]
        assert vector_small.vector.dimension == 500
        assert np.all(np.isin(vector_small.vector.data, [-1, 1]))
        
        # Test with larger dimension
        config_large = EncoderConfig(dimension=5000, seed=42)
        encoder_large = Encoder(config_large)
        vectorizer_large = SchemaVectorizer(encoder_large, config_large)
        
        vectorizer_large.add_alias("make", "manufacturer")
        vector_large = vectorizer_large.get_schema_vectors()["alias:manufacturer"]
        assert vector_large.vector.dimension == 5000
        assert np.all(np.isin(vector_large.vector.data, [-1, 1]))


class TestSchemaVectorizerGetAliases:
    """Test SchemaVectorizer.get_aliases() method.
    
    **Validates: Requirements 11.1, 11.3**
    """
    
    def test_get_aliases_returns_empty_dict_initially(self):
        """Test that get_aliases returns empty dict after init.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        aliases = vectorizer.get_aliases()
        
        assert isinstance(aliases, dict)
        assert len(aliases) == 0
    
    def test_get_aliases_returns_same_dict_reference(self):
        """Test that get_aliases returns the internal dict reference.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        aliases1 = vectorizer.get_aliases()
        aliases2 = vectorizer.get_aliases()
        
        assert aliases1 is aliases2
        assert aliases1 is vectorizer._aliases
    
    def test_get_aliases_reflects_added_aliases(self):
        """Test that get_aliases reflects aliases added via add_alias.
        
        **Validates: Requirements 11.1, 11.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Initially empty
        assert len(vectorizer.get_aliases()) == 0
        
        # Add aliases
        vectorizer.add_alias("make", "manufacturer")
        assert len(vectorizer.get_aliases()) == 1
        assert vectorizer.get_aliases()["manufacturer"] == "make"
        
        vectorizer.add_alias("model", "product_name")
        assert len(vectorizer.get_aliases()) == 2
        assert vectorizer.get_aliases()["product_name"] == "model"


class TestSchemaVectorizerImportSynonymsFromCSV:
    """Test SchemaVectorizer.import_synonyms_from_csv() method.
    
    **Validates: Requirement 11.6** - THE SDK SHALL support importing synonyms
    from external sources (CSV, JSON)
    """
    
    def test_import_synonyms_from_csv_file_like_object(self):
        """Test importing synonyms from a file-like object.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("primary,synonym\nToyota,TYT\nToyota,Toy\nHonda,HND")
        count = vectorizer.import_synonyms_from_csv(csv_data)
        
        assert count == 3
        synonyms = vectorizer.get_synonyms()
        assert synonyms["TYT"] == "Toyota"
        assert synonyms["Toy"] == "Toyota"
        assert synonyms["HND"] == "Honda"
    
    def test_import_synonyms_from_csv_creates_vectors(self):
        """Test that imported synonyms create schema vectors.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("primary,synonym\nToyota,TYT")
        vectorizer.import_synonyms_from_csv(csv_data)
        
        vectors = vectorizer.get_schema_vectors()
        assert "synonym:TYT" in vectors
        assert vectors["synonym:TYT"].original_value == "Toyota"
        assert vectors["synonym:TYT"].element_type == "value"

    def test_import_synonyms_from_csv_case_insensitive_columns(self):
        """Test that column names are case-insensitive.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("PRIMARY,SYNONYM\nToyota,TYT")
        count = vectorizer.import_synonyms_from_csv(csv_data)
        
        assert count == 1
        assert vectorizer.get_synonyms()["TYT"] == "Toyota"
    
    def test_import_synonyms_from_csv_skips_empty_rows(self):
        """Test that empty primary or synonym values are skipped.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("primary,synonym\nToyota,TYT\n,empty_primary\nHonda,")
        count = vectorizer.import_synonyms_from_csv(csv_data)
        
        assert count == 1
        assert "TYT" in vectorizer.get_synonyms()
        assert "empty_primary" not in vectorizer.get_synonyms()

    def test_import_synonyms_from_csv_raises_on_missing_columns(self):
        """Test that ValueError is raised when required columns are missing.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("name,value\nToyota,TYT")
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_synonyms_from_csv(csv_data)
        
        assert "missing required columns" in str(exc_info.value).lower()
    
    def test_import_synonyms_from_csv_raises_on_empty_csv(self):
        """Test that ValueError is raised when CSV has no data rows.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("primary,synonym")
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_synonyms_from_csv(csv_data)
        
        assert "no data rows" in str(exc_info.value).lower()

    def test_import_synonyms_from_csv_raises_on_no_header(self):
        """Test that ValueError is raised when CSV has no header.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("")
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_synonyms_from_csv(csv_data)
        
        assert "empty" in str(exc_info.value).lower() or "header" in str(exc_info.value).lower()


class TestSchemaVectorizerImportSynonymsFromJSON:
    """Test SchemaVectorizer.import_synonyms_from_json() method.
    
    **Validates: Requirement 11.6** - THE SDK SHALL support importing synonyms
    from external sources (CSV, JSON)
    """
    
    def test_import_synonyms_from_json_file_like_object(self):
        """Test importing synonyms from a file-like object.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('{"Toyota": ["TYT", "Toy"], "Honda": ["HND"]}')
        count = vectorizer.import_synonyms_from_json(json_data)
        
        assert count == 3
        synonyms = vectorizer.get_synonyms()
        assert synonyms["TYT"] == "Toyota"
        assert synonyms["Toy"] == "Toyota"
        assert synonyms["HND"] == "Honda"

    def test_import_synonyms_from_json_creates_vectors(self):
        """Test that imported synonyms create schema vectors.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('{"Toyota": ["TYT"]}')
        vectorizer.import_synonyms_from_json(json_data)
        
        vectors = vectorizer.get_schema_vectors()
        assert "synonym:TYT" in vectors
        assert vectors["synonym:TYT"].original_value == "Toyota"
        assert vectors["synonym:TYT"].element_type == "value"
    
    def test_import_synonyms_from_json_skips_empty_synonyms(self):
        """Test that empty string synonyms are skipped.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('{"Toyota": ["TYT", "", "  "]}')
        count = vectorizer.import_synonyms_from_json(json_data)
        
        assert count == 1
        assert "TYT" in vectorizer.get_synonyms()

    def test_import_synonyms_from_json_raises_on_non_dict(self):
        """Test that ValueError is raised when JSON is not a dict.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('["Toyota", "TYT"]')
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_synonyms_from_json(json_data)
        
        assert "object/dict" in str(exc_info.value).lower()
    
    def test_import_synonyms_from_json_raises_on_non_list_synonyms(self):
        """Test that ValueError is raised when synonyms are not a list.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('{"Toyota": "TYT"}')
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_synonyms_from_json(json_data)
        
        assert "array/list" in str(exc_info.value).lower()

    def test_import_synonyms_from_json_raises_on_empty_json(self):
        """Test that ValueError is raised when JSON is empty.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('{}')
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_synonyms_from_json(json_data)
        
        assert "empty" in str(exc_info.value).lower()


class TestSchemaVectorizerImportAliasesFromCSV:
    """Test SchemaVectorizer.import_aliases_from_csv() method.
    
    **Validates: Requirement 11.6** - THE SDK SHALL support importing synonyms
    from external sources (CSV, JSON)
    """
    
    def test_import_aliases_from_csv_file_like_object(self):
        """Test importing aliases from a file-like object.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("role,alias\nmake,manufacturer\nmake,brand\nmodel,product_name")
        count = vectorizer.import_aliases_from_csv(csv_data)
        
        assert count == 3
        aliases = vectorizer.get_aliases()
        assert aliases["manufacturer"] == "make"
        assert aliases["brand"] == "make"
        assert aliases["product_name"] == "model"

    def test_import_aliases_from_csv_creates_vectors(self):
        """Test that imported aliases create schema vectors.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("role,alias\nmake,manufacturer")
        vectorizer.import_aliases_from_csv(csv_data)
        
        vectors = vectorizer.get_schema_vectors()
        assert "alias:manufacturer" in vectors
        assert vectors["alias:manufacturer"].original_value == "make"
        assert vectors["alias:manufacturer"].element_type == "role"
    
    def test_import_aliases_from_csv_case_insensitive_columns(self):
        """Test that column names are case-insensitive.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("ROLE,ALIAS\nmake,manufacturer")
        count = vectorizer.import_aliases_from_csv(csv_data)
        
        assert count == 1
        assert vectorizer.get_aliases()["manufacturer"] == "make"

    def test_import_aliases_from_csv_skips_empty_rows(self):
        """Test that empty role or alias values are skipped.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("role,alias\nmake,manufacturer\n,empty_role\nmodel,")
        count = vectorizer.import_aliases_from_csv(csv_data)
        
        assert count == 1
        assert "manufacturer" in vectorizer.get_aliases()
        assert "empty_role" not in vectorizer.get_aliases()
    
    def test_import_aliases_from_csv_raises_on_missing_columns(self):
        """Test that ValueError is raised when required columns are missing.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("name,value\nmake,manufacturer")
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_aliases_from_csv(csv_data)
        
        assert "missing required columns" in str(exc_info.value).lower()

    def test_import_aliases_from_csv_raises_on_empty_csv(self):
        """Test that ValueError is raised when CSV has no data rows.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        csv_data = io.StringIO("role,alias")
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_aliases_from_csv(csv_data)
        
        assert "no data rows" in str(exc_info.value).lower()


class TestSchemaVectorizerImportAliasesFromJSON:
    """Test SchemaVectorizer.import_aliases_from_json() method.
    
    **Validates: Requirement 11.6** - THE SDK SHALL support importing synonyms
    from external sources (CSV, JSON)
    """
    
    def test_import_aliases_from_json_file_like_object(self):
        """Test importing aliases from a file-like object.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('{"make": ["manufacturer", "brand"], "model": ["product_name"]}')
        count = vectorizer.import_aliases_from_json(json_data)
        
        assert count == 3
        aliases = vectorizer.get_aliases()
        assert aliases["manufacturer"] == "make"
        assert aliases["brand"] == "make"
        assert aliases["product_name"] == "model"

    def test_import_aliases_from_json_creates_vectors(self):
        """Test that imported aliases create schema vectors.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('{"make": ["manufacturer"]}')
        vectorizer.import_aliases_from_json(json_data)
        
        vectors = vectorizer.get_schema_vectors()
        assert "alias:manufacturer" in vectors
        assert vectors["alias:manufacturer"].original_value == "make"
        assert vectors["alias:manufacturer"].element_type == "role"
    
    def test_import_aliases_from_json_skips_empty_aliases(self):
        """Test that empty string aliases are skipped.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('{"make": ["manufacturer", "", "  "]}')
        count = vectorizer.import_aliases_from_json(json_data)
        
        assert count == 1
        assert "manufacturer" in vectorizer.get_aliases()

    def test_import_aliases_from_json_raises_on_non_dict(self):
        """Test that ValueError is raised when JSON is not a dict.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('["make", "manufacturer"]')
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_aliases_from_json(json_data)
        
        assert "object/dict" in str(exc_info.value).lower()
    
    def test_import_aliases_from_json_raises_on_non_list_aliases(self):
        """Test that ValueError is raised when aliases are not a list.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('{"make": "manufacturer"}')
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_aliases_from_json(json_data)
        
        assert "array/list" in str(exc_info.value).lower()
    
    def test_import_aliases_from_json_raises_on_empty_json(self):
        """Test that ValueError is raised when JSON is empty.
        
        **Validates: Requirement 11.6**
        """
        import io
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        json_data = io.StringIO('{}')
        
        with pytest.raises(ValueError) as exc_info:
            vectorizer.import_aliases_from_json(json_data)
        
        assert "empty" in str(exc_info.value).lower()
