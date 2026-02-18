"""
Unit tests for core data structures.

Tests cover:
- Vector bipolar constraint validation
- Vector dimension consistency
- Space ID computation and validation
- Glyph hierarchical structure validation
- Concept creation and serialization
- Edge type validation
"""

import pytest
import numpy as np
from datetime import datetime

from glyphh.core import (
    Vector,
    Concept,
    Edge,
    Glyph,
    Layer,
    Segment,
    compute_space_id
)


class TestVector:
    """Test Vector data structure."""
    
    def test_valid_bipolar_vector(self):
        """Test creating a valid bipolar vector."""
        data = np.array([-1, 1, -1, 1, 1, -1], dtype=np.int8)
        vec = Vector(data=data, dimension=6, space_id="test_space")
        
        assert np.array_equal(vec.data, data)
        assert vec.dimension == 6
        assert vec.space_id == "test_space"
    
    def test_bipolar_constraint_violation(self):
        """Test that non-bipolar values raise ValueError."""
        data = np.array([0, 1, -1, 2], dtype=np.int8)
        
        with pytest.raises(ValueError, match="Vector must be bipolar"):
            Vector(data=data, dimension=4, space_id="test_space")
    
    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        data = np.array([-1, 1, -1, 1], dtype=np.int8)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            Vector(data=data, dimension=10, space_id="test_space")
    
    def test_vector_equality(self):
        """Test vector equality comparison."""
        data = np.array([-1, 1, -1, 1], dtype=np.int8)
        vec1 = Vector(data=data, dimension=4, space_id="test_space")
        vec2 = Vector(data=data.copy(), dimension=4, space_id="test_space")
        vec3 = Vector(data=data, dimension=4, space_id="other_space")
        
        assert vec1 == vec2
        assert vec1 != vec3
    
    def test_vector_from_list(self):
        """Test creating vector from list."""
        data = [-1, 1, -1, 1]
        vec = Vector(data=data, dimension=4, space_id="test_space")
        
        assert isinstance(vec.data, np.ndarray)
        assert np.array_equal(vec.data, np.array(data, dtype=np.int8))


class TestConcept:
    """Test Concept data structure."""
    
    def test_simple_concept(self):
        """Test creating a simple concept."""
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"}
        )
        
        assert concept.name == "red car"
        assert concept.attributes["type"] == "car"
        assert concept.attributes["color"] == "red"
        assert concept.relationships == []
        assert concept.metadata == {}
    
    def test_concept_with_relationships(self):
        """Test concept with relationships."""
        concept = Concept(
            name="car",
            attributes={"type": "vehicle"},
            relationships=[("has_part", "wheels"), ("used_for", "transportation")]
        )
        
        assert len(concept.relationships) == 2
        assert ("has_part", "wheels") in concept.relationships
    
    def test_concept_to_dict(self):
        """Test concept serialization to dict."""
        concept = Concept(
            name="test",
            attributes={"key": "value"},
            relationships=[("rel", "target")],
            metadata={"meta": "data"}
        )
        
        d = concept.to_dict()
        assert d["name"] == "test"
        assert d["attributes"] == {"key": "value"}
        assert d["relationships"] == [("rel", "target")]
        assert d["metadata"] == {"meta": "data"}


class TestSpaceId:
    """Test space_id computation."""
    
    def test_deterministic_space_id(self):
        """Test that space_id is deterministic."""
        space_id1 = compute_space_id(10000, 42, '{"dimension":10000,"seed":42}')
        space_id2 = compute_space_id(10000, 42, '{"dimension":10000,"seed":42}')
        
        assert space_id1 == space_id2
        assert len(space_id1) == 16
    
    def test_different_configs_different_space_ids(self):
        """Test that different configs produce different space_ids."""
        space_id1 = compute_space_id(10000, 42, '{"dimension":10000,"seed":42}')
        space_id2 = compute_space_id(10000, 43, '{"dimension":10000,"seed":43}')
        space_id3 = compute_space_id(5000, 42, '{"dimension":5000,"seed":42}')
        
        assert space_id1 != space_id2
        assert space_id1 != space_id3
        assert space_id2 != space_id3


class TestSegment:
    """Test Segment data structure."""
    
    def test_segment_creation(self):
        """Test creating a segment."""
        cortex = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test_space"
        )
        role_vec = Vector(
            data=np.array([1, -1, 1, -1], dtype=np.int8),
            dimension=4,
            space_id="test_space"
        )
        
        segment = Segment(
            name="attributes",
            cortex=cortex,
            roles={"color": role_vec},
            role_values={"color": "red"},
            weights={"segment": 0.9}
        )
        
        assert segment.name == "attributes"
        assert segment.cortex == cortex
        assert "color" in segment.roles
        assert segment.role_values["color"] == "red"


class TestLayer:
    """Test Layer data structure."""
    
    def test_layer_creation(self):
        """Test creating a layer."""
        cortex = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test_space"
        )
        segment_cortex = Vector(
            data=np.array([1, -1, 1, -1], dtype=np.int8),
            dimension=4,
            space_id="test_space"
        )
        segment = Segment(
            name="attributes",
            cortex=segment_cortex,
            roles={},
            role_values={},
            weights={}
        )
        
        layer = Layer(
            name="semantic",
            cortex=cortex,
            segments={"attributes": segment},
            weights={"layer": 0.8}
        )
        
        assert layer.name == "semantic"
        assert layer.cortex == cortex
        assert "attributes" in layer.segments


class TestGlyph:
    """Test Glyph data structure."""
    
    def test_valid_glyph(self):
        """Test creating a valid glyph."""
        space_id = "test_space"
        global_cortex = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id=space_id
        )
        
        glyph = Glyph(
            identifier="car_red@2024-01-15T10:30:00Z#v1",
            name="red car",
            space_id=space_id,
            global_cortex=global_cortex,
            layers={},
            security_levels={"cortex": 0.9},
            metadata={"domain": "automotive"}
        )
        
        assert glyph.identifier == "car_red@2024-01-15T10:30:00Z#v1"
        assert glyph.name == "red car"
        assert glyph.space_id == space_id
        assert glyph.global_cortex == global_cortex
    
    def test_invalid_identifier_format(self):
        """Test that invalid identifier format raises ValueError."""
        global_cortex = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test_space"
        )
        
        with pytest.raises(ValueError, match="Invalid identifier format"):
            Glyph(
                identifier="invalid_identifier",
                name="test",
                space_id="test_space",
                global_cortex=global_cortex
            )
    
    def test_space_id_mismatch_in_cortex(self):
        """Test that space_id mismatch in cortex raises ValueError."""
        global_cortex = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="other_space"
        )
        
        with pytest.raises(ValueError, match="Space ID mismatch"):
            Glyph(
                identifier="test@2024-01-15T10:30:00Z#v1",
                name="test",
                space_id="test_space",
                global_cortex=global_cortex
            )
    
    def test_space_id_consistency_in_layers(self):
        """Test that space_id is validated across all layers."""
        space_id = "test_space"
        global_cortex = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id=space_id
        )
        layer_cortex = Vector(
            data=np.array([1, -1, 1, -1], dtype=np.int8),
            dimension=4,
            space_id="wrong_space"
        )
        layer = Layer(
            name="semantic",
            cortex=layer_cortex,
            segments={},
            weights={}
        )
        
        with pytest.raises(ValueError, match="Space ID mismatch in layer"):
            Glyph(
                identifier="test@2024-01-15T10:30:00Z#v1",
                name="test",
                space_id=space_id,
                global_cortex=global_cortex,
                layers={"semantic": layer}
            )


class TestEdge:
    """Test Edge data structure."""
    
    def test_valid_spatial_edge(self):
        """Test creating a valid spatial edge."""
        vector = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test_space"
        )
        
        edge = Edge(
            type="neural_cortex",
            source="glyph1@2024-01-15T10:30:00Z#v1",
            target=None,
            vector=vector,
            weights={"cortex": 1.0},
            security_level=0.9
        )
        
        assert edge.type == "neural_cortex"
        assert edge.source == "glyph1@2024-01-15T10:30:00Z#v1"
        assert edge.target is None
        assert edge.vector == vector
    
    def test_valid_temporal_edge(self):
        """Test creating a valid temporal edge."""
        vector = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test_space"
        )
        
        edge = Edge(
            type="temporal_cortex",
            source="glyph1@2024-01-15T10:30:00Z#v1",
            target="glyph1@2024-01-15T10:35:00Z#v2",
            vector=vector,
            weights={"cortex": 1.0},
            security_level=0.9
        )
        
        assert edge.type == "temporal_cortex"
        assert edge.target is not None
    
    def test_invalid_edge_type(self):
        """Test that invalid edge type raises ValueError."""
        vector = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test_space"
        )
        
        with pytest.raises(ValueError, match="Invalid edge type"):
            Edge(
                type="invalid_type",
                source="glyph1",
                target=None,
                vector=vector,
                weights={},
                security_level=0.9
            )
    
    def test_edge_with_hierarchy_info(self):
        """Test edge with layer/segment/role information."""
        vector = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test_space"
        )
        
        edge = Edge(
            type="neural_role",
            source="glyph1@2024-01-15T10:30:00Z#v1",
            target=None,
            vector=vector,
            weights={"role": 0.8},
            security_level=0.5,
            layer=0,
            segment=0,
            role="color"
        )
        
        assert edge.layer == 0
        assert edge.segment == 0
        assert edge.role == "color"
