"""
Unit tests for SimilarityCalculator.

Tests cover:
- Cosine similarity computation
- Hamming similarity computation
- Metric conversion functions
- Dual weighting system (similarity + security)
- Visibility decisions
- Edge type support
- Error handling
"""

import pytest
import numpy as np
from datetime import datetime

from glyphh.similarity import SimilarityCalculator, SimilarityResult
from glyphh.core.types import Glyph, Layer, Segment, Vector, Concept
from glyphh.core.config import EncoderConfig
from glyphh.encoder.base import Encoder


@pytest.fixture
def encoder():
    """Create a test encoder."""
    config = EncoderConfig(dimension=100, seed=42)
    return Encoder(config)


@pytest.fixture
def simple_glyph1(encoder):
    """Create a simple test glyph."""
    concept = Concept(
        name="red car",
        attributes={"type": "car", "color": "red"},
        relationships=[],
        metadata={}
    )
    glyph = encoder.encode(concept)
    # Add security levels
    glyph.security_levels = {
        "cortex": 0.9,
        "layer": 0.8,
        "segment": 0.7,
        "role": 0.5
    }
    return glyph


@pytest.fixture
def simple_glyph2(encoder):
    """Create another simple test glyph."""
    concept = Concept(
        name="blue car",
        attributes={"type": "car", "color": "blue"},
        relationships=[],
        metadata={}
    )
    glyph = encoder.encode(concept)
    # Add security levels
    glyph.security_levels = {
        "cortex": 0.9,
        "layer": 0.8,
        "segment": 0.7,
        "role": 0.5
    }
    return glyph


class TestSimilarityCalculatorInit:
    """Test SimilarityCalculator initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        calc = SimilarityCalculator()
        assert calc.threshold == 0.5
        assert calc.default_metric == "cosine"
    
    def test_custom_threshold(self):
        """Test custom threshold."""
        calc = SimilarityCalculator(threshold=0.7)
        assert calc.threshold == 0.7
    
    def test_custom_metric(self):
        """Test custom default metric."""
        calc = SimilarityCalculator(default_metric="hamming")
        assert calc.default_metric == "hamming"
    
    def test_invalid_threshold(self):
        """Test invalid threshold raises error."""
        with pytest.raises(ValueError, match="Threshold must be in"):
            SimilarityCalculator(threshold=1.5)
        
        with pytest.raises(ValueError, match="Threshold must be in"):
            SimilarityCalculator(threshold=-0.1)
    
    def test_invalid_metric(self):
        """Test invalid metric raises error."""
        with pytest.raises(ValueError, match="Invalid metric"):
            SimilarityCalculator(default_metric="euclidean")


class TestCosineSimilarity:
    """Test cosine similarity computation."""
    
    def test_identical_vectors(self, simple_glyph1):
        """Test cosine similarity of identical vectors."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph1,
            edge_type="neural_cortex",
            metric="cosine"
        )
        # Identical vectors should have similarity 1.0
        assert result.raw_score == pytest.approx(1.0, abs=0.01)
    
    def test_similar_vectors(self, simple_glyph1, simple_glyph2):
        """Test cosine similarity of similar vectors."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            metric="cosine"
        )
        # Similar vectors (both cars) should have positive similarity
        assert -1.0 <= result.raw_score <= 1.0
        # Should be somewhat similar (both are cars)
        assert result.raw_score > 0.0
    
    def test_cosine_range(self, simple_glyph1, simple_glyph2):
        """Test cosine similarity is in valid range."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            metric="cosine"
        )
        assert -1.0 <= result.raw_score <= 1.0


class TestHammingSimilarity:
    """Test hamming similarity computation."""
    
    def test_identical_vectors(self, simple_glyph1):
        """Test hamming similarity of identical vectors."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph1,
            edge_type="neural_cortex",
            metric="hamming"
        )
        # Identical vectors should have similarity 1.0
        assert result.raw_score == pytest.approx(1.0, abs=0.01)
    
    def test_similar_vectors(self, simple_glyph1, simple_glyph2):
        """Test hamming similarity of similar vectors."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            metric="hamming"
        )
        # Similar vectors should have positive similarity
        assert 0.0 <= result.raw_score <= 1.0
        # Should be somewhat similar (both are cars)
        assert result.raw_score > 0.5
    
    def test_hamming_range(self, simple_glyph1, simple_glyph2):
        """Test hamming similarity is in valid range."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            metric="hamming"
        )
        assert 0.0 <= result.raw_score <= 1.0


class TestMetricConversion:
    """Test metric conversion functions."""
    
    def test_cosine_to_hamming(self):
        """Test cosine to hamming conversion."""
        calc = SimilarityCalculator()
        
        # Test known conversions
        assert calc.convert_cosine_to_hamming(1.0) == pytest.approx(1.0)
        assert calc.convert_cosine_to_hamming(0.0) == pytest.approx(0.5)
        assert calc.convert_cosine_to_hamming(-1.0) == pytest.approx(0.0)
        assert calc.convert_cosine_to_hamming(0.4) == pytest.approx(0.7)
    
    def test_hamming_to_cosine(self):
        """Test hamming to cosine conversion."""
        calc = SimilarityCalculator()
        
        # Test known conversions
        assert calc.convert_hamming_to_cosine(1.0) == pytest.approx(1.0)
        assert calc.convert_hamming_to_cosine(0.5) == pytest.approx(0.0)
        assert calc.convert_hamming_to_cosine(0.0) == pytest.approx(-1.0)
        assert calc.convert_hamming_to_cosine(0.7) == pytest.approx(0.4)
    
    def test_round_trip_conversion(self):
        """Test round-trip conversion preserves values."""
        calc = SimilarityCalculator()
        
        # Test multiple values
        for cosine_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            hamming_val = calc.convert_cosine_to_hamming(cosine_val)
            recovered_cosine = calc.convert_hamming_to_cosine(hamming_val)
            assert recovered_cosine == pytest.approx(cosine_val, abs=0.001)
    
    def test_metric_relationship(self, simple_glyph1, simple_glyph2):
        """Test that cosine and hamming metrics have correct relationship."""
        calc = SimilarityCalculator()
        
        # Compute with both metrics
        cosine_result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            metric="cosine"
        )
        hamming_result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            metric="hamming"
        )
        
        # Verify relationship: hamming = (cosine + 1) / 2
        expected_hamming = (cosine_result.raw_score + 1.0) / 2.0
        assert hamming_result.raw_score == pytest.approx(expected_hamming, abs=0.01)


class TestSimilarityWeighting:
    """Test similarity weighting system."""
    
    def test_no_weights(self, simple_glyph1, simple_glyph2):
        """Test similarity with no weights (should be 1.0)."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex"
        )
        # With no weights, similarity_weight should be 1.0
        assert result.similarity_weight == pytest.approx(1.0)
    
    def test_with_weights(self, encoder):
        """Test similarity with custom weights."""
        # Create glyphs with custom weights
        concept1 = Concept(
            name="test1",
            attributes={"type": "car"},
            relationships=[],
            metadata={}
        )
        glyph1 = encoder.encode(concept1)
        
        # Add weights to first layer
        first_layer = next(iter(glyph1.layers.values()))
        first_layer.weights = {"layer": 0.8}
        
        concept2 = Concept(
            name="test2",
            attributes={"type": "car"},
            relationships=[],
            metadata={}
        )
        glyph2 = encoder.encode(concept2)
        
        # Add weights to first layer
        first_layer2 = next(iter(glyph2.layers.values()))
        first_layer2.weights = {"layer": 0.6}
        
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            glyph1,
            glyph2,
            edge_type="neural_layer"
        )
        
        # Similarity weight should be average of both weights
        expected_weight = (0.8 + 0.6) / 2.0
        assert result.similarity_weight == pytest.approx(expected_weight)


class TestSecurityWeighting:
    """Test security weighting system."""
    
    def test_full_clearance(self, simple_glyph1, simple_glyph2):
        """Test with full user clearance."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            user_clearance=1.0
        )
        # With full clearance, security_weight should be 1.0
        assert result.security_weight == pytest.approx(1.0)
    
    def test_partial_clearance(self, simple_glyph1, simple_glyph2):
        """Test with partial user clearance."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            user_clearance=0.5
        )
        # With partial clearance (0.5) and required clearance (0.9),
        # security_weight should be 0.5 / 0.9
        expected_weight = 0.5 / 0.9
        assert result.security_weight == pytest.approx(expected_weight, abs=0.01)
    
    def test_no_security_requirement(self, encoder):
        """Test with no security requirements."""
        concept = Concept(
            name="test",
            attributes={"type": "car"},
            relationships=[],
            metadata={}
        )
        glyph1 = encoder.encode(concept)
        glyph2 = encoder.encode(concept)
        
        # No security levels set (defaults to 0.0)
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            glyph1,
            glyph2,
            edge_type="neural_cortex",
            user_clearance=0.5
        )
        # With no security requirement, security_weight should be 1.0
        assert result.security_weight == pytest.approx(1.0)


class TestVisibilityDecision:
    """Test visibility decision logic."""
    
    def test_visible_above_threshold(self, simple_glyph1, simple_glyph2):
        """Test visibility when score is above threshold."""
        calc = SimilarityCalculator(threshold=0.0)  # Very low threshold
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            user_clearance=1.0
        )
        # Should be visible with full clearance and low threshold
        assert result.visible is True
    
    def test_hidden_below_threshold(self, simple_glyph1, simple_glyph2):
        """Test visibility when score is below threshold."""
        calc = SimilarityCalculator(threshold=1.0)  # Very high threshold
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            user_clearance=1.0
        )
        # Should be hidden with very high threshold (unless identical)
        if simple_glyph1.identifier != simple_glyph2.identifier:
            assert result.visible is False
    
    def test_hidden_insufficient_clearance(self, simple_glyph1, simple_glyph2):
        """Test visibility with insufficient clearance."""
        calc = SimilarityCalculator(threshold=0.0)  # Low threshold
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            user_clearance=0.1  # Very low clearance
        )
        # Should be hidden due to insufficient clearance
        # (required clearance is 0.9 from cortex level)
        assert result.visible is False


class TestEdgeTypes:
    """Test support for different edge types."""
    
    def test_neural_cortex(self, simple_glyph1, simple_glyph2):
        """Test neural_cortex edge type."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex"
        )
        assert result.edge_type == "neural_cortex"
        assert result.score is not None
    
    def test_neural_layer(self, simple_glyph1, simple_glyph2):
        """Test neural_layer edge type."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_layer"
        )
        assert result.edge_type == "neural_layer"
        assert result.score is not None
    
    def test_neural_segment(self, simple_glyph1, simple_glyph2):
        """Test neural_segment edge type."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_segment"
        )
        assert result.edge_type == "neural_segment"
        assert result.score is not None
    
    def test_neural_role(self, simple_glyph1, simple_glyph2):
        """Test neural_role edge type."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_role"
        )
        assert result.edge_type == "neural_role"
        assert result.score is not None
    
    def test_invalid_edge_type(self, simple_glyph1, simple_glyph2):
        """Test invalid edge type raises error."""
        calc = SimilarityCalculator()
        with pytest.raises(ValueError, match="Invalid edge type"):
            calc.compute_similarity(
                simple_glyph1,
                simple_glyph2,
                edge_type="invalid_type"
            )


class TestErrorHandling:
    """Test error handling."""
    
    def test_different_vector_spaces(self, encoder):
        """Test error when comparing glyphs from different spaces."""
        # Create two encoders with different configs
        config1 = EncoderConfig(dimension=100, seed=42)
        config2 = EncoderConfig(dimension=100, seed=43)  # Different seed
        encoder1 = Encoder(config1)
        encoder2 = Encoder(config2)
        
        concept = Concept(
            name="test",
            attributes={"type": "car"},
            relationships=[],
            metadata={}
        )
        
        glyph1 = encoder1.encode(concept)
        glyph2 = encoder2.encode(concept)
        
        calc = SimilarityCalculator()
        with pytest.raises(ValueError, match="different vector spaces"):
            calc.compute_similarity(
                glyph1,
                glyph2,
                edge_type="neural_cortex"
            )
    
    def test_invalid_user_clearance(self, simple_glyph1, simple_glyph2):
        """Test invalid user clearance raises error."""
        calc = SimilarityCalculator()
        
        with pytest.raises(ValueError, match="User clearance must be in"):
            calc.compute_similarity(
                simple_glyph1,
                simple_glyph2,
                edge_type="neural_cortex",
                user_clearance=1.5
            )
        
        with pytest.raises(ValueError, match="User clearance must be in"):
            calc.compute_similarity(
                simple_glyph1,
                simple_glyph2,
                edge_type="neural_cortex",
                user_clearance=-0.1
            )
    
    def test_invalid_metric(self, simple_glyph1, simple_glyph2):
        """Test invalid metric raises error."""
        calc = SimilarityCalculator()
        with pytest.raises(ValueError, match="Invalid metric"):
            calc.compute_similarity(
                simple_glyph1,
                simple_glyph2,
                edge_type="neural_cortex",
                metric="euclidean"
            )


class TestSimilarityResult:
    """Test SimilarityResult dataclass."""
    
    def test_result_structure(self, simple_glyph1, simple_glyph2):
        """Test result has all expected fields."""
        calc = SimilarityCalculator()
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex"
        )
        
        assert result.glyph1 == simple_glyph1
        assert result.glyph2 == simple_glyph2
        assert result.edge_type == "neural_cortex"
        assert isinstance(result.score, float)
        assert isinstance(result.visible, bool)
        assert isinstance(result.raw_score, float)
        assert isinstance(result.similarity_weight, float)
        assert isinstance(result.security_weight, float)
        assert result.metric in ["cosine", "hamming"]
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.metadata, dict)
    
    def test_result_metadata(self, simple_glyph1, simple_glyph2):
        """Test result metadata contains expected information."""
        calc = SimilarityCalculator(threshold=0.7)
        result = calc.compute_similarity(
            simple_glyph1,
            simple_glyph2,
            edge_type="neural_cortex",
            user_clearance=0.8
        )
        
        assert "user_clearance" in result.metadata
        assert result.metadata["user_clearance"] == 0.8
        assert "required_clearance" in result.metadata
        assert "threshold" in result.metadata
        assert result.metadata["threshold"] == 0.7
        assert "dimensions" in result.metadata


class TestCalculatorRepr:
    """Test calculator string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        calc = SimilarityCalculator(threshold=0.7, default_metric="hamming")
        repr_str = repr(calc)
        assert "SimilarityCalculator" in repr_str
        assert "0.7" in repr_str
        assert "hamming" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
