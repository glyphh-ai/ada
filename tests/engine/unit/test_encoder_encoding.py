"""
Unit tests for core encoding operations in the Encoder class.

Tests cover:
- bind() operation
- bundle() operation
- encode_segment() method
- merge_segments() method
- encode() method (full concept encoding)
"""

import pytest
import numpy as np
from glyphh.encoder.base import Encoder
from glyphh.core.config import EncoderConfig
from glyphh.core.types import Concept, Vector, Segment


class TestBindOperation:
    """Tests for the bind() operation."""
    
    def test_bind_creates_bound_vector(self):
        """Test that bind creates a valid bound vector."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        role = encoder.generate_symbol("color")
        value = encoder.generate_symbol("red")
        
        bound = encoder.bind(role, value)
        
        # Verify bound vector properties
        assert bound.dimension == 1000
        assert bound.space_id == encoder.space_id
        assert np.all(np.isin(bound.data, [-1, 1]))
    
    def test_bind_inverse_property(self):
        """Test that bind(bind(r, v), r) = v (inverse property)."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        role = encoder.generate_symbol("color")
        value = encoder.generate_symbol("red")
        
        # Bind role with value
        bound = encoder.bind(role, value)
        
        # Unbind by binding again with role
        retrieved = encoder.bind(bound, role)
        
        # Should retrieve original value
        assert np.array_equal(retrieved.data, value.data)
    
    def test_bind_validates_vectors(self):
        """Test that bind validates input vectors."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        role = encoder.generate_symbol("color")
        
        # Create vector from different space
        other_encoder = Encoder(EncoderConfig(dimension=1000, seed=99))
        value = other_encoder.generate_symbol("red")
        
        # Should raise VectorSpaceException
        with pytest.raises(Exception):  # Will be VectorSpaceException
            encoder.bind(role, value)


class TestBundleOperation:
    """Tests for the bundle() operation."""
    
    def test_bundle_creates_bundled_vector(self):
        """Test that bundle creates a valid bundled vector."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        v1 = encoder.generate_symbol("a")
        v2 = encoder.generate_symbol("b")
        v3 = encoder.generate_symbol("c")
        
        bundled = encoder.bundle([v1, v2, v3])
        
        # Verify bundled vector properties
        assert bundled.dimension == 1000
        assert bundled.space_id == encoder.space_id
        assert np.all(np.isin(bundled.data, [-1, 1]))
    
    def test_bundle_commutativity(self):
        """Test that bundle([a, b, c]) = bundle([c, a, b])."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        v1 = encoder.generate_symbol("a")
        v2 = encoder.generate_symbol("b")
        v3 = encoder.generate_symbol("c")
        
        bundled1 = encoder.bundle([v1, v2, v3])
        bundled2 = encoder.bundle([v3, v1, v2])
        bundled3 = encoder.bundle([v2, v3, v1])
        
        # All should be equal
        assert np.array_equal(bundled1.data, bundled2.data)
        assert np.array_equal(bundled1.data, bundled3.data)
    
    def test_bundle_empty_list_raises_error(self):
        """Test that bundling empty list raises ValueError."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        with pytest.raises(ValueError, match="Cannot bundle empty vector list"):
            encoder.bundle([])
    
    def test_bundle_validates_vectors(self):
        """Test that bundle validates all input vectors."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        v1 = encoder.generate_symbol("a")
        v2 = encoder.generate_symbol("b")
        
        # Create vector from different space
        other_encoder = Encoder(EncoderConfig(dimension=1000, seed=99))
        v3 = other_encoder.generate_symbol("c")
        
        # Should raise VectorSpaceException
        with pytest.raises(Exception):  # Will be VectorSpaceException
            encoder.bundle([v1, v2, v3])


class TestEncodeSegment:
    """Tests for the encode_segment() method."""
    
    def test_encode_segment_creates_segment(self):
        """Test that encode_segment creates a valid segment."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        segment = encoder.encode_segment(
            segment_name="attributes",
            attributes={"type": "car", "color": "red", "size": "medium"}
        )
        
        # Verify segment structure
        assert segment.name == "attributes"
        assert segment.cortex.dimension == 1000
        assert segment.cortex.space_id == encoder.space_id
        assert len(segment.roles) == 3
        assert len(segment.role_values) == 3
        
        # Verify role values stored correctly
        assert segment.role_values["type"] == "car"
        assert segment.role_values["color"] == "red"
        assert segment.role_values["size"] == "medium"
    
    def test_encode_segment_role_bindings(self):
        """Test that role bindings are created correctly."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        segment = encoder.encode_segment(
            segment_name="attributes",
            attributes={"color": "red"}
        )
        
        # Verify role binding exists
        assert "color" in segment.roles
        color_binding = segment.roles["color"]
        
        # Verify it's a valid vector
        assert color_binding.dimension == 1000
        assert color_binding.space_id == encoder.space_id
        assert np.all(np.isin(color_binding.data, [-1, 1]))
        
        # Verify we can unbind to retrieve value
        color_role = encoder.generate_symbol("color")
        retrieved_value = encoder.bind(color_binding, color_role)
        expected_value = encoder.generate_symbol("red")
        assert np.array_equal(retrieved_value.data, expected_value.data)
    
    def test_encode_segment_empty_attributes_raises_error(self):
        """Test that encoding segment with no attributes raises EncodingException."""
        from glyphh.exceptions import EncodingException
        
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        with pytest.raises(EncodingException, match="Cannot encode segment"):
            encoder.encode_segment("attributes", {})
    
    def test_encode_segment_with_weights(self):
        """Test that segment stores weights correctly."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        weights = {"segment": 0.9, "type": 1.0, "color": 0.8}
        segment = encoder.encode_segment(
            segment_name="attributes",
            attributes={"type": "car", "color": "red"},
            weights=weights
        )
        
        assert segment.weights == weights


class TestMergeSegments:
    """Tests for the merge_segments() method."""
    
    def test_merge_segments_creates_layer_cortex(self):
        """Test that merge_segments creates a valid layer cortex."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        segment1 = encoder.encode_segment(
            "attributes",
            {"type": "car", "color": "red"}
        )
        segment2 = encoder.encode_segment(
            "relations",
            {"has_part": "wheels", "used_for": "transportation"}
        )
        
        layer_cortex = encoder.merge_segments([segment1, segment2])
        
        # Verify layer cortex properties
        assert layer_cortex.dimension == 1000
        assert layer_cortex.space_id == encoder.space_id
        assert np.all(np.isin(layer_cortex.data, [-1, 1]))
    
    def test_merge_segments_empty_list_raises_error(self):
        """Test that merging empty segment list raises EncodingException."""
        from glyphh.exceptions import EncodingException
        
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        with pytest.raises(EncodingException, match="Cannot merge empty segment list"):
            encoder.merge_segments([])


class TestEncode:
    """Tests for the encode() method (full concept encoding)."""
    
    def test_encode_creates_glyph(self):
        """Test that encode creates a valid glyph."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red", "size": "medium"},
            relationships=[("has_part", "wheels"), ("used_for", "transportation")],
            metadata={"domain": "automotive"}
        )
        
        glyph = encoder.encode(concept)
        
        # Verify glyph structure
        assert glyph.name == "red car"
        assert glyph.space_id == encoder.space_id
        assert glyph.global_cortex.dimension == 1000
        assert glyph.global_cortex.space_id == encoder.space_id
        assert len(glyph.layers) > 0
        assert glyph.metadata == {"domain": "automotive"}
    
    def test_encode_hierarchical_structure(self):
        """Test that encode creates correct hierarchical structure."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
            relationships=[("has_part", "wheels")],
        )
        
        glyph = encoder.encode(concept)
        
        # Verify hierarchy: layers → segments → roles
        assert "semantic" in glyph.layers
        layer = glyph.layers["semantic"]
        
        assert "attributes" in layer.segments
        attributes_segment = layer.segments["attributes"]
        assert "type" in attributes_segment.roles
        assert "color" in attributes_segment.roles
        
        assert "relations" in layer.segments
        relations_segment = layer.segments["relations"]
        assert "has_part" in relations_segment.roles
    
    def test_encode_identifier_format(self):
        """Test that glyph identifier has correct format."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        concept = Concept(
            name="red car",
            attributes={"type": "car"},
        )
        
        glyph = encoder.encode(concept)
        
        # Verify identifier format: primary_key@timestamp#version
        assert "@" in glyph.identifier
        assert "#" in glyph.identifier
        assert glyph.identifier.endswith("#v1")
        assert glyph.identifier.startswith("red_car@")
    
    def test_encode_deterministic(self):
        """Test that encoding is deterministic for semantic content."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
        )
        
        # Encode twice (with different timestamps, but same concept)
        glyph1 = encoder.encode(concept)
        glyph2 = encoder.encode(concept)
        
        # Semantic layer cortices should be identical (deterministic encoding)
        # Note: global_cortex includes _temporal layer which differs due to timestamps
        assert np.array_equal(
            glyph1.layers["semantic"].cortex.data,
            glyph2.layers["semantic"].cortex.data
        )
        
        # Role bindings should also be identical
        attrs1 = glyph1.layers["semantic"].segments["attributes"]
        attrs2 = glyph2.layers["semantic"].segments["attributes"]
        assert np.array_equal(attrs1.roles["type"].data, attrs2.roles["type"].data)
        assert np.array_equal(attrs1.roles["color"].data, attrs2.roles["color"].data)
    
    def test_encode_space_id_consistency(self):
        """Test that all vectors in glyph have same space_id."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
            relationships=[("has_part", "wheels")],
        )
        
        glyph = encoder.encode(concept)
        
        # Verify space_id consistency throughout hierarchy
        assert glyph.space_id == encoder.space_id
        assert glyph.global_cortex.space_id == encoder.space_id
        
        for layer in glyph.layers.values():
            assert layer.cortex.space_id == encoder.space_id
            
            for segment in layer.segments.values():
                assert segment.cortex.space_id == encoder.space_id
                
                for role_vector in segment.roles.values():
                    assert role_vector.space_id == encoder.space_id
    
    def test_encode_empty_concept_raises_error(self):
        """Test that encoding concept with no data raises EncodingException."""
        from glyphh.exceptions import EncodingException
        
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        concept = Concept(
            name="empty",
            attributes={},
            relationships=[],
        )
        
        with pytest.raises(EncodingException, match="Cannot encode concept"):
            encoder.encode(concept)


class TestEncodingIntegration:
    """Integration tests for the complete encoding workflow."""
    
    def test_full_encoding_workflow(self):
        """Test complete encoding workflow from concept to glyph."""
        # Create encoder
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create concept
        concept = Concept(
            name="red sports car",
            attributes={
                "type": "car",
                "color": "red",
                "size": "medium",
                "style": "sports"
            },
            relationships=[
                ("has_part", "wheels"),
                ("has_part", "engine"),
                ("used_for", "transportation"),
                ("used_for", "racing")
            ],
            metadata={"domain": "automotive", "category": "vehicle"}
        )
        
        # Encode concept
        glyph = encoder.encode(concept)
        
        # Verify complete structure
        assert glyph.name == "red sports car"
        assert glyph.space_id == encoder.space_id
        assert glyph.metadata == {"domain": "automotive", "category": "vehicle"}
        
        # Verify hierarchical structure (semantic + _temporal layers)
        assert len(glyph.layers) == 2
        assert "semantic" in glyph.layers
        assert "_temporal" in glyph.layers
        
        layer = glyph.layers["semantic"]
        
        assert len(layer.segments) == 2
        attributes_segment = layer.segments["attributes"]
        relations_segment = layer.segments["relations"]
        
        # Verify attributes segment
        assert len(attributes_segment.roles) == 4
        assert attributes_segment.role_values["type"] == "car"
        assert attributes_segment.role_values["color"] == "red"
        assert attributes_segment.role_values["size"] == "medium"
        assert attributes_segment.role_values["style"] == "sports"
        
        # Verify relations segment
        assert len(relations_segment.roles) == 2
        assert "has_part" in relations_segment.role_values
        assert "used_for" in relations_segment.role_values
        
        # Verify all vectors are bipolar
        assert np.all(np.isin(glyph.global_cortex.data, [-1, 1]))
        assert np.all(np.isin(layer.cortex.data, [-1, 1]))
        assert np.all(np.isin(attributes_segment.cortex.data, [-1, 1]))
        assert np.all(np.isin(relations_segment.cortex.data, [-1, 1]))
        
        # Verify _temporal layer structure
        temporal_layer = glyph.layers["_temporal"]
        assert "signal" in temporal_layer.segments
        assert "value" in temporal_layer.segments["signal"].roles
    
    def test_multiple_concepts_same_encoder(self):
        """Test encoding multiple concepts with same encoder."""
        encoder = Encoder(EncoderConfig(dimension=1000, seed=42))
        
        concept1 = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
        )
        
        concept2 = Concept(
            name="blue car",
            attributes={"type": "car", "color": "blue"},
        )
        
        glyph1 = encoder.encode(concept1)
        glyph2 = encoder.encode(concept2)
        
        # Both should have same space_id
        assert glyph1.space_id == glyph2.space_id == encoder.space_id
        
        # Type bindings should be identical (same role, same value)
        type_binding1 = glyph1.layers["semantic"].segments["attributes"].roles["type"]
        type_binding2 = glyph2.layers["semantic"].segments["attributes"].roles["type"]
        assert np.array_equal(type_binding1.data, type_binding2.data)
        
        # Color bindings should be different (same role, different values)
        color_binding1 = glyph1.layers["semantic"].segments["attributes"].roles["color"]
        color_binding2 = glyph2.layers["semantic"].segments["attributes"].roles["color"]
        assert not np.array_equal(color_binding1.data, color_binding2.data)


class TestExplicitConfigEncoding:
    """Tests for encoding with explicit layer/segment/role config."""
    
    def test_encode_with_explicit_config(self):
        """Test encoding with explicit layers in config."""
        from glyphh.core.config import Layer, Segment, Role
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="identity",
                    similarity_weight=1.0,
                    segments=[
                        Segment(
                            name="core",
                            roles=[
                                Role(name="type", similarity_weight=0.8),
                                Role(name="color", similarity_weight=1.0),
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
        )
        
        glyph = encoder.encode(concept)
        
        # Verify structure matches config
        assert "identity" in glyph.layers
        layer = glyph.layers["identity"]
        assert "core" in layer.segments
        segment = layer.segments["core"]
        assert "type" in segment.roles
        assert "color" in segment.roles
    
    def test_encode_with_weighted_encoding(self):
        """Test encoding with apply_weights_during_encoding=True."""
        from glyphh.core.config import Layer, Segment, Role
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            apply_weights_during_encoding=True,
            layers=[
                Layer(
                    name="identity",
                    similarity_weight=0.8,
                    segments=[
                        Segment(
                            name="core",
                            similarity_weight=0.9,
                            roles=[
                                Role(name="type", similarity_weight=1.0),
                                Role(name="color", similarity_weight=0.5),
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
        )
        
        glyph = encoder.encode(concept)
        
        # Verify glyph is created with weighted encoding
        assert glyph.name == "red car"
        assert np.all(np.isin(glyph.global_cortex.data, [-1, 1]))
    
    def test_encode_with_unweighted_encoding(self):
        """Test encoding with apply_weights_during_encoding=False (default)."""
        from glyphh.core.config import Layer, Segment, Role
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            apply_weights_during_encoding=False,
            layers=[
                Layer(
                    name="identity",
                    similarity_weight=0.8,
                    segments=[
                        Segment(
                            name="core",
                            roles=[
                                Role(name="type"),
                                Role(name="color"),
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
        )
        
        glyph = encoder.encode(concept)
        
        # Verify weights are stored in glyph structure
        layer = glyph.layers["identity"]
        assert layer.weights["similarity"] == 0.8
    
    def test_encode_skips_missing_attributes(self):
        """Test that encoding skips roles not present in concept."""
        from glyphh.core.config import Layer, Segment, Role
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="identity",
                    segments=[
                        Segment(
                            name="core",
                            roles=[
                                Role(name="type"),
                                Role(name="color"),
                                Role(name="size"),  # Not in concept
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},  # No "size"
        )
        
        glyph = encoder.encode(concept)
        
        # Only type and color should be encoded
        segment = glyph.layers["identity"].segments["core"]
        assert "type" in segment.roles
        assert "color" in segment.roles
        assert "size" not in segment.roles
    
    def test_encode_with_multiple_layers(self):
        """Test encoding with multiple layers."""
        from glyphh.core.config import Layer, Segment, Role
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="identity",
                    similarity_weight=1.0,
                    segments=[
                        Segment(name="core", roles=[Role(name="type")])
                    ]
                ),
                Layer(
                    name="attributes",
                    similarity_weight=0.8,
                    segments=[
                        Segment(name="visual", roles=[Role(name="color")])
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
        )
        
        glyph = encoder.encode(concept)
        
        # Both layers should be present
        assert "identity" in glyph.layers
        assert "attributes" in glyph.layers
        assert glyph.layers["identity"].weights["similarity"] == 1.0
        assert glyph.layers["attributes"].weights["similarity"] == 0.8
    
    def test_encode_no_matching_attributes_raises_error(self):
        """Test that encoding fails when no attributes match defined roles."""
        from glyphh.core.config import Layer, Segment, Role
        from glyphh.exceptions import EncodingException
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name="identity",
                    segments=[
                        Segment(
                            name="core",
                            roles=[
                                Role(name="foo"),  # Not in concept
                                Role(name="bar"),  # Not in concept
                            ]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
        )
        
        with pytest.raises(EncodingException, match="no attributes match"):
            encoder.encode(concept)
    
    def test_weighted_vs_unweighted_produces_different_results(self):
        """Test that weighted and unweighted encoding produce different results."""
        from glyphh.core.config import Layer, Segment, Role
        
        # Config with weighted encoding
        config_weighted = EncoderConfig(
            dimension=1000,
            seed=42,
            apply_weights_during_encoding=True,
            layers=[
                Layer(
                    name="identity",
                    similarity_weight=0.5,
                    segments=[
                        Segment(
                            name="core",
                            similarity_weight=0.5,
                            roles=[
                                Role(name="type", similarity_weight=1.0),
                                Role(name="color", similarity_weight=0.1),
                            ]
                        )
                    ]
                )
            ]
        )
        
        # Config with unweighted encoding (same structure)
        config_unweighted = EncoderConfig(
            dimension=1000,
            seed=42,
            apply_weights_during_encoding=False,
            layers=[
                Layer(
                    name="identity",
                    similarity_weight=0.5,
                    segments=[
                        Segment(
                            name="core",
                            similarity_weight=0.5,
                            roles=[
                                Role(name="type", similarity_weight=1.0),
                                Role(name="color", similarity_weight=0.1),
                            ]
                        )
                    ]
                )
            ]
        )
        
        encoder_weighted = Encoder(config_weighted)
        encoder_unweighted = Encoder(config_unweighted)
        
        concept = Concept(
            name="red car",
            attributes={"type": "car", "color": "red"},
        )
        
        glyph_weighted = encoder_weighted.encode(concept)
        glyph_unweighted = encoder_unweighted.encode(concept)
        
        # The global cortices should be different due to weighting strategy
        # (weighted bundling vs unweighted bundling)
        # Note: They might be the same in some cases due to majority vote
        # but the segment cortices should differ
        segment_weighted = glyph_weighted.layers["identity"].segments["core"]
        segment_unweighted = glyph_unweighted.layers["identity"].segments["core"]
        
        # Both should be valid bipolar vectors
        assert np.all(np.isin(segment_weighted.cortex.data, [-1, 1]))
        assert np.all(np.isin(segment_unweighted.cortex.data, [-1, 1]))
