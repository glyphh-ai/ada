"""
Unit tests for EdgeGenerator.

This module tests the EdgeGenerator class for creating spatial and temporal edges
at all hierarchy levels (cortex, layer, segment, role).
"""

import pytest
import numpy as np
from datetime import datetime

from glyphh.core.types import Concept, Glyph, Layer, Segment, Vector, Edge
from glyphh.core.config import EncoderConfig
from glyphh.encoder.base import Encoder
from glyphh.edges.generator import EdgeGenerator


@pytest.fixture
def encoder():
    """Create a test encoder."""
    config = EncoderConfig(dimension=1000, seed=42)
    return Encoder(config)


@pytest.fixture
def simple_concept():
    """Create a simple test concept."""
    return Concept(
        name="red car",
        attributes={"type": "car", "color": "red", "size": "medium"},
        relationships=[("has_part", "wheels"), ("used_for", "transportation")],
        metadata={"domain": "automotive"}
    )


@pytest.fixture
def simple_glyph(encoder, simple_concept):
    """Create a simple test glyph."""
    return encoder.encode(simple_concept)


@pytest.fixture
def edge_generator():
    """Create an EdgeGenerator instance."""
    return EdgeGenerator()


class TestEdgeGeneratorInit:
    """Test EdgeGenerator initialization."""
    
    def test_init(self, edge_generator):
        """Test EdgeGenerator can be initialized."""
        assert edge_generator is not None
        assert isinstance(edge_generator, EdgeGenerator)
    
    def test_repr(self, edge_generator):
        """Test EdgeGenerator string representation."""
        repr_str = repr(edge_generator)
        assert "EdgeGenerator" in repr_str
        assert "spatial_types=4" in repr_str
        assert "temporal_types=4" in repr_str


class TestSpatialEdgeGeneration:
    """Test spatial edge generation."""
    
    def test_generate_spatial_edges_returns_list(self, edge_generator, simple_glyph):
        """Test that generate_spatial_edges returns a list."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        assert isinstance(edges, list)
        assert len(edges) > 0
    
    def test_spatial_edges_have_correct_types(self, edge_generator, simple_glyph):
        """Test that spatial edges have correct types."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        edge_types = {edge.type for edge in edges}
        expected_types = {"neural_cortex", "neural_layer", "neural_segment", "neural_role"}
        
        # All generated edge types should be in expected types
        assert edge_types.issubset(expected_types)
        
        # Should have at least cortex edge
        assert "neural_cortex" in edge_types
    
    def test_neural_cortex_edge(self, edge_generator, simple_glyph):
        """Test neural_cortex edge generation."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        cortex_edges = [e for e in edges if e.type == "neural_cortex"]
        assert len(cortex_edges) == 1
        
        cortex_edge = cortex_edges[0]
        assert cortex_edge.source == simple_glyph.identifier
        assert cortex_edge.target is None
        assert cortex_edge.layer is None
        assert cortex_edge.segment is None
        assert cortex_edge.role is None
        assert np.array_equal(cortex_edge.vector.data, simple_glyph.global_cortex.data)
        assert cortex_edge.vector.space_id == simple_glyph.space_id
    
    def test_neural_layer_edges(self, edge_generator, simple_glyph):
        """Test neural_layer edge generation."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        layer_edges = [e for e in edges if e.type == "neural_layer"]
        assert len(layer_edges) == len(simple_glyph.layers)
        
        for edge in layer_edges:
            assert edge.source == simple_glyph.identifier
            assert edge.target is None
            assert edge.layer is not None
            assert edge.segment is None
            assert edge.role is None
            assert edge.vector.space_id == simple_glyph.space_id
    
    def test_neural_segment_edges(self, edge_generator, simple_glyph):
        """Test neural_segment edge generation."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        segment_edges = [e for e in edges if e.type == "neural_segment"]
        
        # Count expected segments
        expected_segments = sum(
            len(layer.segments) for layer in simple_glyph.layers.values()
        )
        assert len(segment_edges) == expected_segments
        
        for edge in segment_edges:
            assert edge.source == simple_glyph.identifier
            assert edge.target is None
            assert edge.layer is not None
            assert edge.segment is not None
            assert edge.role is None
            assert edge.vector.space_id == simple_glyph.space_id
    
    def test_neural_role_edges(self, edge_generator, simple_glyph):
        """Test neural_role edge generation."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        role_edges = [e for e in edges if e.type == "neural_role"]
        
        # Count expected roles
        expected_roles = sum(
            len(segment.roles)
            for layer in simple_glyph.layers.values()
            for segment in layer.segments.values()
        )
        assert len(role_edges) == expected_roles
        
        for edge in role_edges:
            assert edge.source == simple_glyph.identifier
            assert edge.target is None
            assert edge.layer is not None
            assert edge.segment is not None
            assert edge.role is not None
            assert edge.vector.space_id == simple_glyph.space_id
    
    def test_spatial_edges_have_weights(self, edge_generator, simple_glyph):
        """Test that spatial edges have weights."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        for edge in edges:
            assert isinstance(edge.weights, dict)
            assert len(edge.weights) > 0
    
    def test_spatial_edges_have_security_levels(self, edge_generator, simple_glyph):
        """Test that spatial edges have security levels."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        for edge in edges:
            assert isinstance(edge.security_level, (int, float))
            assert edge.security_level >= 0.0


class TestTemporalEdgeGeneration:
    """Test temporal edge generation."""
    
    def test_generate_temporal_edges_returns_list(self, edge_generator, encoder, simple_concept):
        """Test that generate_temporal_edges returns a list."""
        glyph_v1 = encoder.encode(simple_concept)
        
        # Create v2 with modified attributes
        concept_v2 = Concept(
            name="blue car",
            attributes={"type": "car", "color": "blue", "size": "medium"},
            relationships=[("has_part", "wheels"), ("used_for", "transportation")],
            metadata={"domain": "automotive"}
        )
        glyph_v2 = encoder.encode(concept_v2)
        
        edges = edge_generator.generate_temporal_edges(glyph_v1, glyph_v2)
        assert isinstance(edges, list)
        assert len(edges) > 0
    
    def test_temporal_edges_have_correct_types(self, edge_generator, encoder, simple_concept):
        """Test that temporal edges have correct types."""
        glyph_v1 = encoder.encode(simple_concept)
        
        concept_v2 = Concept(
            name="blue car",
            attributes={"type": "car", "color": "blue", "size": "medium"},
            relationships=[("has_part", "wheels"), ("used_for", "transportation")],
            metadata={"domain": "automotive"}
        )
        glyph_v2 = encoder.encode(concept_v2)
        
        edges = edge_generator.generate_temporal_edges(glyph_v1, glyph_v2)
        
        edge_types = {edge.type for edge in edges}
        expected_types = {"temporal_cortex", "temporal_layer", "temporal_segment", "temporal_role"}
        
        # All generated edge types should be in expected types
        assert edge_types.issubset(expected_types)
        
        # Should have at least cortex edge
        assert "temporal_cortex" in edge_types
    
    def test_temporal_cortex_edge(self, edge_generator, encoder, simple_concept):
        """Test temporal_cortex edge generation."""
        glyph_v1 = encoder.encode(simple_concept)
        
        concept_v2 = Concept(
            name="blue car",
            attributes={"type": "car", "color": "blue", "size": "medium"},
            relationships=[("has_part", "wheels"), ("used_for", "transportation")],
            metadata={"domain": "automotive"}
        )
        glyph_v2 = encoder.encode(concept_v2)
        
        edges = edge_generator.generate_temporal_edges(glyph_v1, glyph_v2)
        
        cortex_edges = [e for e in edges if e.type == "temporal_cortex"]
        assert len(cortex_edges) == 1
        
        cortex_edge = cortex_edges[0]
        assert cortex_edge.source == glyph_v1.identifier
        assert cortex_edge.target == glyph_v2.identifier
        assert cortex_edge.layer is None
        assert cortex_edge.segment is None
        assert cortex_edge.role is None
        assert cortex_edge.vector.space_id == glyph_v1.space_id
    
    def test_temporal_edges_different_space_raises_error(self, edge_generator, simple_concept):
        """Test that temporal edges from different spaces raise error."""
        # Create two encoders with different configs
        encoder1 = Encoder(EncoderConfig(dimension=1000, seed=42))
        encoder2 = Encoder(EncoderConfig(dimension=1000, seed=99))
        
        glyph_v1 = encoder1.encode(simple_concept)
        glyph_v2 = encoder2.encode(simple_concept)
        
        with pytest.raises(ValueError, match="different vector spaces"):
            edge_generator.generate_temporal_edges(glyph_v1, glyph_v2)
    
    def test_temporal_edges_have_source_and_target(self, edge_generator, encoder, simple_concept):
        """Test that temporal edges have both source and target."""
        glyph_v1 = encoder.encode(simple_concept)
        
        concept_v2 = Concept(
            name="blue car",
            attributes={"type": "car", "color": "blue", "size": "medium"},
            relationships=[("has_part", "wheels"), ("used_for", "transportation")],
            metadata={"domain": "automotive"}
        )
        glyph_v2 = encoder.encode(concept_v2)
        
        edges = edge_generator.generate_temporal_edges(glyph_v1, glyph_v2)
        
        for edge in edges:
            assert edge.source == glyph_v1.identifier
            assert edge.target == glyph_v2.identifier


class TestTemporalDeltaComputation:
    """Test temporal delta computation."""
    
    def test_compute_temporal_delta(self, edge_generator, encoder):
        """Test temporal delta computation."""
        v1 = encoder.generate_symbol("red")
        v2 = encoder.generate_symbol("blue")
        
        delta = edge_generator._compute_temporal_delta(v1, v2)
        
        assert isinstance(delta, Vector)
        assert delta.dimension == v1.dimension
        assert delta.space_id == v1.space_id
        assert np.all(np.isin(delta.data, [-1, 1]))
    
    def test_temporal_delta_different_dimensions_raises_error(self, edge_generator):
        """Test that temporal delta with different dimensions raises error."""
        v1 = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        v2 = Vector(
            data=np.array([-1, 1, -1], dtype=np.int8),
            dimension=3,
            space_id="test"
        )
        
        with pytest.raises(ValueError, match="different dimensions"):
            edge_generator._compute_temporal_delta(v1, v2)
    
    def test_temporal_delta_different_spaces_raises_error(self, edge_generator):
        """Test that temporal delta with different spaces raises error."""
        v1 = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="space1"
        )
        v2 = Vector(
            data=np.array([-1, 1, -1, 1], dtype=np.int8),
            dimension=4,
            space_id="space2"
        )
        
        with pytest.raises(ValueError, match="different spaces"):
            edge_generator._compute_temporal_delta(v1, v2)
    
    def test_temporal_delta_is_bipolar(self, edge_generator, encoder):
        """Test that temporal delta is bipolar."""
        v1 = encoder.generate_symbol("test1")
        v2 = encoder.generate_symbol("test2")
        
        delta = edge_generator._compute_temporal_delta(v1, v2)
        
        # Delta should be bipolar
        assert np.all(np.isin(delta.data, [-1, 1]))


class TestEdgeFiltering:
    """Test edge filtering methods."""
    
    def test_get_edges_by_type(self, edge_generator, simple_glyph):
        """Test filtering edges by type."""
        all_edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        cortex_edges = edge_generator.get_edges_by_type(all_edges, "neural_cortex")
        assert len(cortex_edges) == 1
        assert all(e.type == "neural_cortex" for e in cortex_edges)
        
        layer_edges = edge_generator.get_edges_by_type(all_edges, "neural_layer")
        assert all(e.type == "neural_layer" for e in layer_edges)
        
        segment_edges = edge_generator.get_edges_by_type(all_edges, "neural_segment")
        assert all(e.type == "neural_segment" for e in segment_edges)
        
        role_edges = edge_generator.get_edges_by_type(all_edges, "neural_role")
        assert all(e.type == "neural_role" for e in role_edges)
    
    def test_get_edges_by_hierarchy_layer(self, edge_generator, simple_glyph):
        """Test filtering edges by layer."""
        all_edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        layer_0_edges = edge_generator.get_edges_by_hierarchy(all_edges, layer=0)
        assert all(e.layer == 0 for e in layer_0_edges)
    
    def test_get_edges_by_hierarchy_segment(self, edge_generator, simple_glyph):
        """Test filtering edges by segment."""
        all_edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        segment_0_edges = edge_generator.get_edges_by_hierarchy(all_edges, layer=0, segment=0)
        assert all(e.layer == 0 and e.segment == 0 for e in segment_0_edges)
    
    def test_get_edges_by_hierarchy_role(self, edge_generator, simple_glyph):
        """Test filtering edges by role."""
        all_edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        # Get all color role edges
        color_edges = edge_generator.get_edges_by_hierarchy(all_edges, role="color")
        assert all(e.role == "color" for e in color_edges)
        assert len(color_edges) > 0
    
    def test_get_edges_by_hierarchy_combined(self, edge_generator, simple_glyph):
        """Test filtering edges by multiple hierarchy levels."""
        all_edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        # Get specific role in specific segment in specific layer
        specific_edges = edge_generator.get_edges_by_hierarchy(
            all_edges,
            layer=0,
            segment=0,
            role="color"
        )
        
        for edge in specific_edges:
            assert edge.layer == 0
            assert edge.segment == 0
            assert edge.role == "color"


class TestEdgeVectorProperties:
    """Test properties of edge vectors."""
    
    def test_all_edge_vectors_are_bipolar(self, edge_generator, simple_glyph):
        """Test that all edge vectors are bipolar."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        for edge in edges:
            assert np.all(np.isin(edge.vector.data, [-1, 1]))
    
    def test_all_edge_vectors_have_correct_dimension(self, edge_generator, simple_glyph):
        """Test that all edge vectors have correct dimension."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        expected_dimension = simple_glyph.global_cortex.dimension
        
        for edge in edges:
            assert edge.vector.dimension == expected_dimension
    
    def test_all_edge_vectors_have_correct_space_id(self, edge_generator, simple_glyph):
        """Test that all edge vectors have correct space_id."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        expected_space_id = simple_glyph.space_id
        
        for edge in edges:
            assert edge.vector.space_id == expected_space_id


class TestEdgeCount:
    """Test edge count calculations."""
    
    def test_spatial_edge_count_matches_structure(self, edge_generator, simple_glyph):
        """Test that spatial edge count matches glyph structure."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        # Count expected edges
        expected_count = 1  # cortex
        expected_count += len(simple_glyph.layers)  # layers
        
        for layer in simple_glyph.layers.values():
            expected_count += len(layer.segments)  # segments
            for segment in layer.segments.values():
                expected_count += len(segment.roles)  # roles
        
        assert len(edges) == expected_count
    
    def test_temporal_edge_count_matches_common_structure(self, edge_generator, encoder):
        """Test that temporal edge count matches common structure."""
        concept1 = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
            relationships=[("has_part", "wheels")],
            metadata={}
        )
        concept2 = Concept(
            name="blue car",
            attributes={"type": "car", "color": "blue"},
            relationships=[("has_part", "wheels")],
            metadata={}
        )
        
        glyph_v1 = encoder.encode(concept1)
        glyph_v2 = encoder.encode(concept2)
        
        edges = edge_generator.generate_temporal_edges(glyph_v1, glyph_v2)
        
        # Count expected edges (only common structure)
        expected_count = 1  # cortex
        
        # Common layers
        common_layers = set(glyph_v1.layers.keys()) & set(glyph_v2.layers.keys())
        expected_count += len(common_layers)
        
        for layer_name in common_layers:
            layer_v1 = glyph_v1.layers[layer_name]
            layer_v2 = glyph_v2.layers[layer_name]
            
            # Common segments
            common_segments = set(layer_v1.segments.keys()) & set(layer_v2.segments.keys())
            expected_count += len(common_segments)
            
            for segment_name in common_segments:
                segment_v1 = layer_v1.segments[segment_name]
                segment_v2 = layer_v2.segments[segment_name]
                
                # Common roles
                common_roles = set(segment_v1.roles.keys()) & set(segment_v2.roles.keys())
                expected_count += len(common_roles)
        
        assert len(edges) == expected_count


class TestEdgeValidation:
    """Test edge validation."""
    
    def test_all_edges_are_valid_edge_objects(self, edge_generator, simple_glyph):
        """Test that all generated edges are valid Edge objects."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        for edge in edges:
            assert isinstance(edge, Edge)
            assert isinstance(edge.type, str)
            assert isinstance(edge.source, str)
            assert isinstance(edge.vector, Vector)
            assert isinstance(edge.weights, dict)
            assert isinstance(edge.security_level, (int, float))
    
    def test_edge_types_are_valid(self, edge_generator, simple_glyph):
        """Test that all edge types are valid."""
        edges = edge_generator.generate_spatial_edges(simple_glyph)
        
        valid_types = {
            "neural_cortex", "neural_layer", "neural_segment", "neural_role",
            "temporal_cortex", "temporal_layer", "temporal_segment", "temporal_role"
        }
        
        for edge in edges:
            assert edge.type in valid_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
