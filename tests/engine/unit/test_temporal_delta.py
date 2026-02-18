"""
Unit tests for temporal delta computation.

Tests the TemporalEncoder class for computing and applying temporal deltas.
"""

import pytest
import numpy as np
from glyphh.temporal.delta import TemporalEncoder
from glyphh.core.types import Vector


class TestTemporalEncoder:
    """Test suite for TemporalEncoder class."""
    
    def test_compute_temporal_delta_basic(self):
        """Test basic temporal delta computation."""
        encoder = TemporalEncoder()
        
        # Create two versions
        v1 = Vector(
            data=np.array([-1, 1, -1, 1, 1, -1, 1, -1], dtype=np.int8),
            dimension=8,
            space_id="test"
        )
        v2 = Vector(
            data=np.array([1, 1, -1, -1, 1, -1, -1, 1], dtype=np.int8),
            dimension=8,
            space_id="test"
        )
        
        # Compute delta
        delta = encoder.compute_temporal_delta(v1, v2)
        
        # Verify delta is a valid vector
        assert isinstance(delta, Vector)
        assert delta.dimension == 8
        assert delta.space_id == "test"
        assert np.all(np.isin(delta.data, [-1, 1]))
    
    def test_apply_temporal_delta_basic(self):
        """Test basic temporal delta application."""
        encoder = TemporalEncoder()
        
        # Create base and delta
        base = Vector(
            data=np.array([-1, 1, -1, 1, 1, -1, 1, -1], dtype=np.int8),
            dimension=8,
            space_id="test"
        )
        delta = Vector(
            data=np.array([-1, 1, 1, -1, 1, 1, -1, -1], dtype=np.int8),
            dimension=8,
            space_id="test"
        )
        
        # Apply delta
        new_version = encoder.apply_temporal_delta(base, delta)
        
        # Verify result is a valid vector
        assert isinstance(new_version, Vector)
        assert new_version.dimension == 8
        assert new_version.space_id == "test"
        assert np.all(np.isin(new_version.data, [-1, 1]))
    
    def test_delta_roundtrip(self):
        """Test that applying a computed delta reconstructs the target vector."""
        encoder = TemporalEncoder()
        
        # Create two versions
        v1 = Vector(
            data=np.array([-1, 1, -1, 1, 1, -1, 1, -1, 1, 1], dtype=np.int8),
            dimension=10,
            space_id="test"
        )
        v2 = Vector(
            data=np.array([1, 1, -1, -1, 1, -1, -1, 1, -1, 1], dtype=np.int8),
            dimension=10,
            space_id="test"
        )
        
        # Compute delta
        delta = encoder.compute_temporal_delta(v1, v2)
        
        # Apply delta to v1
        v2_reconstructed = encoder.apply_temporal_delta(v1, delta)
        
        # Verify reconstruction matches v2
        assert np.array_equal(v2.data, v2_reconstructed.data)
        assert v2_reconstructed.space_id == v2.space_id
        assert v2_reconstructed.dimension == v2.dimension
    
    def test_compute_delta_different_spaces(self):
        """Test that computing delta between different spaces raises error."""
        encoder = TemporalEncoder()
        
        v1 = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="space1"
        )
        v2 = Vector(
            data=np.array([1, 1, -1, -1], dtype=np.int8),
            dimension=4,
            space_id="space2"
        )
        
        with pytest.raises(ValueError, match="different spaces"):
            encoder.compute_temporal_delta(v1, v2)
    
    def test_compute_delta_different_dimensions(self):
        """Test that computing delta between different dimensions raises error."""
        encoder = TemporalEncoder()
        
        v1 = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        v2 = Vector(
            data=np.array([1, 1, -1, -1, 1, -1], dtype=np.int8),
            dimension=6,
            space_id="test"
        )
        
        with pytest.raises(ValueError, match="different dimensions"):
            encoder.compute_temporal_delta(v1, v2)
    
    def test_apply_delta_different_spaces(self):
        """Test that applying delta from different space raises error."""
        encoder = TemporalEncoder()
        
        base = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="space1"
        )
        delta = Vector(
            data=np.array([1, 1, -1, -1], dtype=np.int8),
            dimension=4,
            space_id="space2"
        )
        
        with pytest.raises(ValueError, match="different space"):
            encoder.apply_temporal_delta(base, delta)
    
    def test_apply_delta_different_dimensions(self):
        """Test that applying delta with different dimension raises error."""
        encoder = TemporalEncoder()
        
        base = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        delta = Vector(
            data=np.array([1, 1, -1, -1, 1, -1], dtype=np.int8),
            dimension=6,
            space_id="test"
        )
        
        with pytest.raises(ValueError, match="different dimension"):
            encoder.apply_temporal_delta(base, delta)
    
    def test_inverse_helper(self):
        """Test the _inverse helper method."""
        encoder = TemporalEncoder()
        
        v = np.array([-1, 1, -1, 1, 1, -1], dtype=np.int8)
        v_inv = encoder._inverse(v)
        
        # Verify inverse is negation
        expected = np.array([1, -1, 1, -1, -1, 1], dtype=np.int8)
        assert np.array_equal(v_inv, expected)
        
        # Verify double inverse returns original
        v_inv_inv = encoder._inverse(v_inv)
        assert np.array_equal(v_inv_inv, v)
    
    def test_merge_segments_equal_weights(self):
        """Test merging segments with equal weights."""
        encoder = TemporalEncoder()
        
        seg1 = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        seg2 = Vector(
            data=np.array([1, 1, -1, -1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        
        # Merge with equal weights
        merged = encoder.merge_segments([seg1, seg2])
        
        # Verify result
        assert isinstance(merged, Vector)
        assert merged.dimension == 4
        assert merged.space_id == "test"
        assert np.all(np.isin(merged.data, [-1, 1]))
    
    def test_merge_segments_custom_weights(self):
        """Test merging segments with custom weights."""
        encoder = TemporalEncoder()
        
        seg1 = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        seg2 = Vector(
            data=np.array([1, 1, -1, -1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        
        # Merge with custom weights (seg1 has more influence)
        merged = encoder.merge_segments([seg1, seg2], weights=[0.7, 0.3])
        
        # Verify result
        assert isinstance(merged, Vector)
        assert merged.dimension == 4
        assert merged.space_id == "test"
        assert np.all(np.isin(merged.data, [-1, 1]))
    
    def test_merge_segments_empty_list(self):
        """Test that merging empty segment list raises error."""
        encoder = TemporalEncoder()
        
        with pytest.raises(ValueError, match="empty segment list"):
            encoder.merge_segments([])
    
    def test_merge_segments_dimension_mismatch(self):
        """Test that merging segments with different dimensions raises error."""
        encoder = TemporalEncoder()
        
        seg1 = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        seg2 = Vector(
            data=np.array([1, 1, -1, -1, 1, -1], dtype=np.int8),
            dimension=6,
            space_id="test"
        )
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            encoder.merge_segments([seg1, seg2])
    
    def test_merge_segments_space_mismatch(self):
        """Test that merging segments from different spaces raises error."""
        encoder = TemporalEncoder()
        
        seg1 = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="space1"
        )
        seg2 = Vector(
            data=np.array([1, 1, -1, -1], dtype=np.int8),
            dimension=4,
            space_id="space2"
        )
        
        with pytest.raises(ValueError, match="Space ID mismatch"):
            encoder.merge_segments([seg1, seg2])
    
    def test_merge_segments_invalid_weights(self):
        """Test that invalid weights raise error."""
        encoder = TemporalEncoder()
        
        seg1 = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        seg2 = Vector(
            data=np.array([1, 1, -1, -1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        
        # Weights don't sum to 1.0
        with pytest.raises(ValueError, match="must sum to 1.0"):
            encoder.merge_segments([seg1, seg2], weights=[0.5, 0.3])
        
        # Wrong number of weights
        with pytest.raises(ValueError, match="must match"):
            encoder.merge_segments([seg1, seg2], weights=[1.0])
    
    def test_encode_segment_basic(self):
        """Test basic segment encoding."""
        encoder = TemporalEncoder()
        
        segment_data = {"type": "car", "color": "red"}
        segment = encoder.encode_segment(segment_data, "test", 100)
        
        # Verify result
        assert isinstance(segment, Vector)
        assert segment.dimension == 100
        assert segment.space_id == "test"
        assert np.all(np.isin(segment.data, [-1, 1]))
    
    def test_encode_segment_deterministic(self):
        """Test that segment encoding is deterministic."""
        encoder = TemporalEncoder()
        
        segment_data = {"type": "car", "color": "red"}
        
        # Encode twice
        segment1 = encoder.encode_segment(segment_data, "test", 100)
        segment2 = encoder.encode_segment(segment_data, "test", 100)
        
        # Verify they're identical
        assert np.array_equal(segment1.data, segment2.data)
