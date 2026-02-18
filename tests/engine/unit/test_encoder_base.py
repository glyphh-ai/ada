"""
Unit tests for the base Encoder class.

Tests cover:
- Encoder initialization
- Symbol generation and caching
- Vector validation
- Space ID computation
- Custom encoder extension
"""

import pytest
import numpy as np

from glyphh.encoder.base import Encoder
from glyphh.core.types import Vector
from glyphh.core.config import EncoderConfig
from glyphh.exceptions import (
    BipolarConstraintException,
    DimensionMismatchException,
    VectorSpaceException,
)


class TestEncoderInitialization:
    """Test encoder initialization."""
    
    def test_encoder_init_with_valid_config(self):
        """Test encoder initializes correctly with valid config."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        assert encoder.dimension == 1000
        assert encoder.seed == 42
        assert encoder.config == config
        assert encoder.space_id is not None
        assert len(encoder.space_id) == 16
        assert len(encoder.symbol_cache) == 0
    
    def test_encoder_space_id_deterministic(self):
        """Test that same config produces same space_id."""
        config1 = EncoderConfig(dimension=1000, seed=42)
        config2 = EncoderConfig(dimension=1000, seed=42)
        
        encoder1 = Encoder(config1)
        encoder2 = Encoder(config2)
        
        assert encoder1.space_id == encoder2.space_id
    
    def test_encoder_space_id_different_for_different_configs(self):
        """Test that different configs produce different space_ids."""
        config1 = EncoderConfig(dimension=1000, seed=42)
        config2 = EncoderConfig(dimension=1000, seed=43)
        config3 = EncoderConfig(dimension=2000, seed=42)
        
        encoder1 = Encoder(config1)
        encoder2 = Encoder(config2)
        encoder3 = Encoder(config3)
        
        assert encoder1.space_id != encoder2.space_id
        assert encoder1.space_id != encoder3.space_id
        assert encoder2.space_id != encoder3.space_id


class TestSymbolGeneration:
    """Test symbol generation and caching."""
    
    def test_generate_symbol_returns_vector(self):
        """Test that generate_symbol returns a valid Vector."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        symbol = encoder.generate_symbol("color")
        
        assert isinstance(symbol, Vector)
        assert symbol.dimension == 1000
        assert symbol.space_id == encoder.space_id
        assert len(symbol.data) == 1000
    
    def test_generate_symbol_is_bipolar(self):
        """Test that generated symbols are bipolar."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        symbol = encoder.generate_symbol("color")
        
        # All values should be in {-1, +1}
        assert np.all(np.isin(symbol.data, [-1, 1]))
    
    def test_generate_symbol_is_deterministic(self):
        """Test that same key produces same vector."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        symbol1 = encoder.generate_symbol("color")
        symbol2 = encoder.generate_symbol("color")
        
        assert np.array_equal(symbol1.data, symbol2.data)
        assert symbol1.space_id == symbol2.space_id
    
    def test_generate_symbol_different_keys_produce_different_vectors(self):
        """Test that different keys produce different vectors."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        symbol1 = encoder.generate_symbol("color")
        symbol2 = encoder.generate_symbol("size")
        
        # Vectors should be different (with very high probability)
        assert not np.array_equal(symbol1.data, symbol2.data)
    
    def test_generate_symbol_caching(self):
        """Test that symbols are cached."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        assert encoder.get_cache_size() == 0
        
        encoder.generate_symbol("color")
        assert encoder.get_cache_size() == 1
        
        encoder.generate_symbol("red")
        assert encoder.get_cache_size() == 2
        
        # Generating same key again doesn't increase cache size
        encoder.generate_symbol("color")
        assert encoder.get_cache_size() == 2
    
    def test_generate_symbol_cache_keys(self):
        """Test that cached keys are tracked correctly."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        encoder.generate_symbol("color")
        encoder.generate_symbol("red")
        encoder.generate_symbol("size")
        
        cached_keys = encoder.get_cached_keys()
        assert set(cached_keys) == {"color", "red", "size"}
    
    def test_clear_cache(self):
        """Test that cache can be cleared."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        encoder.generate_symbol("color")
        encoder.generate_symbol("red")
        assert encoder.get_cache_size() == 2
        
        encoder.clear_cache()
        assert encoder.get_cache_size() == 0
        assert encoder.get_cached_keys() == []
    
    def test_generate_symbol_after_clear_cache(self):
        """Test that symbols can be regenerated after cache clear."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        symbol1 = encoder.generate_symbol("color")
        encoder.clear_cache()
        symbol2 = encoder.generate_symbol("color")
        
        # Should produce same vector (deterministic)
        assert np.array_equal(symbol1.data, symbol2.data)


class TestVectorValidation:
    """Test vector validation."""
    
    def test_validate_vector_valid(self):
        """Test that valid vectors pass validation."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        symbol = encoder.generate_symbol("color")
        
        # Should not raise
        encoder._validate_vector(symbol)
    
    def test_validate_vector_wrong_dimension(self):
        """Test that wrong dimension raises exception."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        bad_vector = Vector(
            data=np.array([-1, 1, -1], dtype=np.int8),
            dimension=3,
            space_id=encoder.space_id
        )
        
        with pytest.raises(DimensionMismatchException) as exc_info:
            encoder._validate_vector(bad_vector)
        
        assert "1000" in str(exc_info.value)
        assert "3" in str(exc_info.value)
    
    def test_validate_vector_wrong_space_id(self):
        """Test that wrong space_id raises exception."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        bad_vector = Vector(
            data=np.array([-1, 1] * 500, dtype=np.int8),
            dimension=1000,
            space_id="different_space"
        )
        
        with pytest.raises(VectorSpaceException) as exc_info:
            encoder._validate_vector(bad_vector)
        
        assert encoder.space_id in str(exc_info.value)
        assert "different_space" in str(exc_info.value)
    
    def test_validate_vector_non_bipolar(self):
        """Test that non-bipolar values raise exception."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        # Create vector with invalid values (0, 2)
        bad_data = np.array([0, 2, -1, 1] * 250, dtype=np.int8)
        
        # Vector constructor should raise on non-bipolar values
        with pytest.raises(ValueError):
            Vector(
                data=bad_data,
                dimension=1000,
                space_id=encoder.space_id
            )


class TestSpaceIdComputation:
    """Test space_id computation."""
    
    def test_compute_space_id_deterministic(self):
        """Test that space_id computation is deterministic."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        space_id1 = encoder._compute_space_id(1000, 42, '{"dimension":1000,"seed":42}')
        space_id2 = encoder._compute_space_id(1000, 42, '{"dimension":1000,"seed":42}')
        
        assert space_id1 == space_id2
    
    def test_compute_space_id_different_for_different_inputs(self):
        """Test that different inputs produce different space_ids."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        space_id1 = encoder._compute_space_id(1000, 42, '{"dimension":1000,"seed":42}')
        space_id2 = encoder._compute_space_id(1000, 43, '{"dimension":1000,"seed":43}')
        space_id3 = encoder._compute_space_id(2000, 42, '{"dimension":2000,"seed":42}')
        
        assert space_id1 != space_id2
        assert space_id1 != space_id3
        assert space_id2 != space_id3


class TestCustomEncoderExtension:
    """Test custom encoder extension framework."""
    
    def test_custom_encoder_inherits_base_functionality(self):
        """Test that custom encoder can use base class methods."""
        
        class CustomEncoder(Encoder):
            pass
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = CustomEncoder(config)
        
        # Should have all base functionality
        assert encoder.dimension == 1000
        assert encoder.seed == 42
        assert encoder.space_id is not None
        
        # Should be able to generate symbols
        symbol = encoder.generate_symbol("test")
        assert isinstance(symbol, Vector)
        assert symbol.dimension == 1000
    
    def test_custom_encoder_can_validate_vectors(self):
        """Test that custom encoder can use validation."""
        
        class CustomEncoder(Encoder):
            def custom_method(self, key: str) -> Vector:
                # Generate and validate
                vector = self.generate_symbol(key)
                self._validate_vector(vector)
                return vector
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = CustomEncoder(config)
        
        vector = encoder.custom_method("test")
        assert isinstance(vector, Vector)
    
    def test_custom_encoder_same_space_as_base(self):
        """Test that custom encoder shares space_id with base encoder."""
        
        class CustomEncoder(Encoder):
            pass
        
        config = EncoderConfig(dimension=1000, seed=42)
        base_encoder = Encoder(config)
        custom_encoder = CustomEncoder(config)
        
        # Should have same space_id
        assert base_encoder.space_id == custom_encoder.space_id
        
        # Vectors should be in same space
        base_symbol = base_encoder.generate_symbol("test")
        custom_symbol = custom_encoder.generate_symbol("test")
        
        assert base_symbol.space_id == custom_symbol.space_id


class TestEncoderRepr:
    """Test encoder string representation."""
    
    def test_repr(self):
        """Test that repr provides useful information."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        encoder.generate_symbol("color")
        encoder.generate_symbol("red")
        
        repr_str = repr(encoder)
        
        assert "Encoder" in repr_str
        assert "dimension=1000" in repr_str
        assert "seed=42" in repr_str
        assert "space_id=" in repr_str
        assert "cached_symbols=2" in repr_str


class TestWeightedBundle:
    """Test weighted_bundle method."""
    
    def test_weighted_bundle_returns_vector(self):
        """Test that weighted_bundle returns a valid Vector."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        v1 = encoder.generate_symbol("a")
        v2 = encoder.generate_symbol("b")
        
        result = encoder.weighted_bundle([(v1, 1.0), (v2, 0.5)])
        
        assert isinstance(result, Vector)
        assert result.dimension == 1000
        assert result.space_id == encoder.space_id
    
    def test_weighted_bundle_is_bipolar(self):
        """Test that weighted_bundle output is always bipolar."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        v1 = encoder.generate_symbol("a")
        v2 = encoder.generate_symbol("b")
        v3 = encoder.generate_symbol("c")
        
        result = encoder.weighted_bundle([
            (v1, 1.0),
            (v2, 0.5),
            (v3, 0.3)
        ])
        
        # All values should be in {-1, +1}
        assert np.all(np.isin(result.data, [-1, 1]))
    
    def test_weighted_bundle_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        with pytest.raises(ValueError) as exc_info:
            encoder.weighted_bundle([])
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_weighted_bundle_single_vector(self):
        """Test weighted_bundle with single vector returns that vector."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        v1 = encoder.generate_symbol("a")
        
        result = encoder.weighted_bundle([(v1, 1.0)])
        
        # Single vector with positive weight should return same vector
        assert np.array_equal(result.data, v1.data)
    
    def test_weighted_bundle_higher_weight_dominates(self):
        """Test that higher weighted vectors have more influence."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        # Create two opposite vectors
        v1 = encoder.generate_symbol("a")
        v2_data = -v1.data  # Opposite of v1
        v2 = Vector(data=v2_data, dimension=1000, space_id=encoder.space_id)
        
        # With equal weights, result depends on tie-breaking (>= 0 -> +1)
        result_equal = encoder.weighted_bundle([(v1, 1.0), (v2, 1.0)])
        
        # With v1 having much higher weight, result should be closer to v1
        result_v1_heavy = encoder.weighted_bundle([(v1, 10.0), (v2, 1.0)])
        
        # v1 should dominate when heavily weighted
        similarity_to_v1 = np.sum(result_v1_heavy.data == v1.data) / 1000
        assert similarity_to_v1 > 0.8  # Should be very similar to v1
    
    def test_weighted_bundle_zero_weight_ignored(self):
        """Test that zero-weighted vectors don't contribute."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        v1 = encoder.generate_symbol("a")
        v2 = encoder.generate_symbol("b")
        
        # v2 with zero weight should not affect result
        result = encoder.weighted_bundle([(v1, 1.0), (v2, 0.0)])
        
        # Result should be same as v1 alone
        assert np.array_equal(result.data, v1.data)
    
    def test_weighted_bundle_validates_vectors(self):
        """Test that weighted_bundle validates all input vectors."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        v1 = encoder.generate_symbol("a")
        
        # Create vector with wrong space_id
        bad_vector = Vector(
            data=np.array([-1, 1] * 500, dtype=np.int8),
            dimension=1000,
            space_id="different_space"
        )
        
        with pytest.raises(VectorSpaceException):
            encoder.weighted_bundle([(v1, 1.0), (bad_vector, 0.5)])
    
    def test_bundle_with_weights_uses_weighted_bundle(self):
        """Test that bundle() with weights delegates to weighted_bundle()."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        v1 = encoder.generate_symbol("a")
        v2 = encoder.generate_symbol("b")
        
        # Using bundle with weights
        result1 = encoder.bundle([v1, v2], weights=[1.0, 0.5])
        
        # Using weighted_bundle directly
        result2 = encoder.weighted_bundle([(v1, 1.0), (v2, 0.5)])
        
        # Should produce same result
        assert np.array_equal(result1.data, result2.data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
