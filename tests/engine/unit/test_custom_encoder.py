"""
Unit tests for custom encoder extension framework.

Tests cover:
- Custom encoder validation
- Vector space consistency
- Dimension consistency
- Proper initialization
- Error handling
"""

import pytest
import numpy as np

from glyphh.encoder.base import Encoder
from glyphh.core.config import EncoderConfig
from glyphh.exceptions import (
    DimensionMismatchException,
    VectorSpaceException,
)
from glyphh.core.types import Concept


class CustomTestEncoder(Encoder):
    """Test custom encoder that extends base encoder."""
    
    def __init__(self, config: EncoderConfig, custom_param: str = "default"):
        super().__init__(config)
        self.custom_param = custom_param


class BadCustomEncoder:
    """Improperly initialized custom encoder (doesn't call super().__init__)."""
    
    def __init__(self, config: EncoderConfig):
        # Missing super().__init__(config) call
        self.config = config


class TestCustomEncoderValidation:
    """Test custom encoder validation functionality."""
    
    def test_validate_compatible_custom_encoder(self):
        """Test that compatible custom encoder passes validation."""
        config = EncoderConfig(dimension=10000, seed=42)
        base_encoder = Encoder(config)
        custom_encoder = CustomTestEncoder(config)
        
        # Should not raise any exception
        base_encoder._validate_custom_encoder(custom_encoder)
    
    def test_validate_custom_encoder_different_dimension(self):
        """Test that custom encoder with different dimension fails validation."""
        base_config = EncoderConfig(dimension=10000, seed=42)
        custom_config = EncoderConfig(dimension=5000, seed=42)
        
        base_encoder = Encoder(base_config)
        custom_encoder = CustomTestEncoder(custom_config)
        
        with pytest.raises(DimensionMismatchException) as exc_info:
            base_encoder._validate_custom_encoder(custom_encoder)
        
        assert "10000" in str(exc_info.value)
        assert "5000" in str(exc_info.value)
    
    def test_validate_custom_encoder_different_seed(self):
        """Test that custom encoder with different seed fails validation."""
        base_config = EncoderConfig(dimension=10000, seed=42)
        custom_config = EncoderConfig(dimension=10000, seed=99)
        
        base_encoder = Encoder(base_config)
        custom_encoder = CustomTestEncoder(custom_config)
        
        # Different seed results in different space_id
        with pytest.raises(VectorSpaceException) as exc_info:
            base_encoder._validate_custom_encoder(custom_encoder)
        
        assert "space_id" in str(exc_info.value).lower()
    
    def test_validate_custom_encoder_not_initialized(self):
        """Test that improperly initialized custom encoder fails validation."""
        config = EncoderConfig(dimension=10000, seed=42)
        base_encoder = Encoder(config)
        bad_encoder = BadCustomEncoder(config)
        
        with pytest.raises(ValueError) as exc_info:
            base_encoder._validate_custom_encoder(bad_encoder)
        
        assert "not properly initialized" in str(exc_info.value)
    
    def test_custom_encoder_inherits_base_methods(self):
        """Test that custom encoder inherits all base encoder methods."""
        config = EncoderConfig(dimension=10000, seed=42)
        custom_encoder = CustomTestEncoder(config, custom_param="test")
        
        # Test inherited methods
        assert hasattr(custom_encoder, 'generate_symbol')
        assert hasattr(custom_encoder, 'bind')
        assert hasattr(custom_encoder, 'bundle')
        assert hasattr(custom_encoder, '_validate_vector')
        assert hasattr(custom_encoder, 'encode')
        
        # Test custom parameter
        assert custom_encoder.custom_param == "test"
    
    def test_custom_encoder_generates_compatible_vectors(self):
        """Test that custom encoder generates vectors compatible with base encoder."""
        config = EncoderConfig(dimension=10000, seed=42)
        base_encoder = Encoder(config)
        custom_encoder = CustomTestEncoder(config)
        
        # Generate same symbol from both encoders
        base_vector = base_encoder.generate_symbol("test")
        custom_vector = custom_encoder.generate_symbol("test")
        
        # Vectors should be identical
        assert np.array_equal(base_vector.data, custom_vector.data)
        assert base_vector.space_id == custom_vector.space_id
        assert base_vector.dimension == custom_vector.dimension
    
    def test_custom_encoder_can_use_base_operations(self):
        """Test that custom encoder can use base encoder operations."""
        config = EncoderConfig(dimension=10000, seed=42)
        custom_encoder = CustomTestEncoder(config)
        
        # Generate vectors
        role = custom_encoder.generate_symbol("color")
        value = custom_encoder.generate_symbol("red")
        
        # Bind operation
        bound = custom_encoder.bind(role, value)
        assert bound.dimension == 10000
        assert bound.space_id == custom_encoder.space_id
        
        # Bundle operation
        bundled = custom_encoder.bundle([role, value, bound])
        assert bundled.dimension == 10000
        assert bundled.space_id == custom_encoder.space_id
    
    def test_custom_encoder_validation_in_operations(self):
        """Test that validation works in custom encoder operations."""
        config = EncoderConfig(dimension=10000, seed=42)
        custom_encoder = CustomTestEncoder(config)
        
        # Generate valid vector
        vector = custom_encoder.generate_symbol("test")
        
        # Validation should pass
        custom_encoder._validate_vector(vector)
    
    def test_multiple_custom_encoders_same_config(self):
        """Test that multiple custom encoders with same config are compatible."""
        config = EncoderConfig(dimension=10000, seed=42)
        
        encoder1 = CustomTestEncoder(config, custom_param="encoder1")
        encoder2 = CustomTestEncoder(config, custom_param="encoder2")
        
        # Both should be compatible with each other
        encoder1._validate_custom_encoder(encoder2)
        encoder2._validate_custom_encoder(encoder1)
        
        # Vectors should be compatible
        vector1 = encoder1.generate_symbol("test")
        vector2 = encoder2.generate_symbol("test")
        
        assert vector1.space_id == vector2.space_id
        assert np.array_equal(vector1.data, vector2.data)
    
    def test_custom_encoder_encode_concept(self):
        """Test that custom encoder can encode concepts."""
        config = EncoderConfig(dimension=10000, seed=42)
        custom_encoder = CustomTestEncoder(config)
        
        concept = Concept(
            name="test_concept",
            attributes={"type": "test", "color": "red"},
            relationships=[("related_to", "other")],
            metadata={"domain": "test"}
        )
        
        glyph = custom_encoder.encode(concept)
        
        # Verify glyph structure
        assert glyph.name == "test_concept"
        assert glyph.space_id == custom_encoder.space_id
        assert glyph.global_cortex.dimension == 10000
        assert "semantic" in glyph.layers
    
    def test_custom_encoder_cache_independence(self):
        """Test that custom encoders have independent caches."""
        config = EncoderConfig(dimension=10000, seed=42)
        
        encoder1 = CustomTestEncoder(config)
        encoder2 = CustomTestEncoder(config)
        
        # Generate symbol in encoder1
        encoder1.generate_symbol("test")
        
        # encoder2 should have empty cache
        assert encoder1.get_cache_size() == 1
        assert encoder2.get_cache_size() == 0
        
        # Generate same symbol in encoder2
        encoder2.generate_symbol("test")
        
        # Now both should have it cached
        assert encoder1.get_cache_size() == 1
        assert encoder2.get_cache_size() == 1


class TestCustomEncoderContracts:
    """Test that custom encoders maintain all base encoder contracts."""
    
    def test_custom_encoder_maintains_bipolar_constraint(self):
        """Test that custom encoder maintains bipolar constraint."""
        config = EncoderConfig(dimension=10000, seed=42)
        custom_encoder = CustomTestEncoder(config)
        
        # Generate multiple vectors
        for key in ["test1", "test2", "test3"]:
            vector = custom_encoder.generate_symbol(key)
            assert np.all(np.isin(vector.data, [-1, 1]))
    
    def test_custom_encoder_maintains_dimension_consistency(self):
        """Test that custom encoder maintains dimension consistency."""
        config = EncoderConfig(dimension=10000, seed=42)
        custom_encoder = CustomTestEncoder(config)
        
        # Generate multiple vectors
        for key in ["test1", "test2", "test3"]:
            vector = custom_encoder.generate_symbol(key)
            assert vector.dimension == 10000
            assert len(vector.data) == 10000
    
    def test_custom_encoder_maintains_space_id_consistency(self):
        """Test that custom encoder maintains space_id consistency."""
        config = EncoderConfig(dimension=10000, seed=42)
        custom_encoder = CustomTestEncoder(config)
        
        # Generate multiple vectors
        vectors = [custom_encoder.generate_symbol(f"test{i}") for i in range(5)]
        
        # All should have same space_id
        space_ids = {v.space_id for v in vectors}
        assert len(space_ids) == 1
        assert space_ids.pop() == custom_encoder.space_id
    
    def test_custom_encoder_deterministic_encoding(self):
        """Test that custom encoder maintains deterministic encoding."""
        config = EncoderConfig(dimension=10000, seed=42)
        
        # Create two encoders with same config
        encoder1 = CustomTestEncoder(config)
        encoder2 = CustomTestEncoder(config)
        
        # Generate same symbols
        vector1 = encoder1.generate_symbol("test")
        vector2 = encoder2.generate_symbol("test")
        
        # Should be identical
        assert np.array_equal(vector1.data, vector2.data)
    
    def test_custom_encoder_bind_inverse_property(self):
        """Test that custom encoder bind operation maintains inverse property."""
        config = EncoderConfig(dimension=10000, seed=42)
        custom_encoder = CustomTestEncoder(config)
        
        role = custom_encoder.generate_symbol("role")
        value = custom_encoder.generate_symbol("value")
        
        # Bind role and value
        bound = custom_encoder.bind(role, value)
        
        # Unbind to retrieve value
        retrieved = custom_encoder.bind(bound, role)
        
        # Should get original value back
        assert np.array_equal(retrieved.data, value.data)
    
    def test_custom_encoder_bundle_commutativity(self):
        """Test that custom encoder bundle operation is commutative."""
        config = EncoderConfig(dimension=10000, seed=42)
        custom_encoder = CustomTestEncoder(config)
        
        v1 = custom_encoder.generate_symbol("v1")
        v2 = custom_encoder.generate_symbol("v2")
        v3 = custom_encoder.generate_symbol("v3")
        
        # Bundle in different orders
        bundle1 = custom_encoder.bundle([v1, v2, v3])
        bundle2 = custom_encoder.bundle([v3, v1, v2])
        bundle3 = custom_encoder.bundle([v2, v3, v1])
        
        # All should be identical
        assert np.array_equal(bundle1.data, bundle2.data)
        assert np.array_equal(bundle1.data, bundle3.data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
