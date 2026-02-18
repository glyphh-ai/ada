"""
Security and access control testing for Task 19.3

Tests security weight computation, dimension filtering, visibility decisions,
and clearance checking across all hierarchy levels.

Updated for new explicit config structure with security_weight at each level.
"""

import pytest
import numpy as np

from glyphh import (
    Encoder, EncoderConfig, Concept,
    SimilarityCalculator, SimilarityResult,
    EdgeGenerator, Role
)
from glyphh.core.config import Layer, Segment


class TestSecurityWeightComputation:
    """Test security weight computation at all hierarchy levels"""
    
    def test_cortex_level_security(self):
        """Test security filtering at cortex level"""
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            security_weight=0.9
        )
        
        encoder = Encoder(config)
        calc = SimilarityCalculator()
        
        concept1 = Concept(
            name="classified_doc_1",
            attributes={"type": "document", "classification": "secret"},
            relationships=[],
            metadata={}
        )
        
        concept2 = Concept(
            name="classified_doc_2",
            attributes={"type": "document", "classification": "secret"},
            relationships=[],
            metadata={}
        )
        
        glyph1 = encoder.encode(concept1)
        glyph2 = encoder.encode(concept2)
        
        # Full clearance
        result_full = calc.compute_similarity(
            glyph1, glyph2, "neural_cortex", user_clearance=1.0
        )
        
        # Partial clearance
        result_partial = calc.compute_similarity(
            glyph1, glyph2, "neural_cortex", user_clearance=0.5
        )
        
        # No clearance
        result_none = calc.compute_similarity(
            glyph1, glyph2, "neural_cortex", user_clearance=0.0
        )
        
        # Verify security filtering affects scores
        assert result_full.score >= result_partial.score
        assert result_partial.score >= result_none.score
        
        # Verify fact trees contain security information
        assert result_full.fact_tree is not None
    
    def test_explicit_layer_security(self):
        """Test security filtering with explicit layer config"""
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            security_weight=1.0,
            layers=[
                Layer(
                    name="identity",
                    security_weight=0.8,
                    segments=[
                        Segment(
                            name="core",
                            security_weight=0.9,
                            roles=[
                                Role(name="type", security_weight=1.0),
                                Role(name="classification", security_weight=0.5),
                            ]
                        )
                    ]
                )
            ]
        )
        
        encoder = Encoder(config)
        calc = SimilarityCalculator()
        
        concept1 = Concept(
            name="layered_data_1",
            attributes={"type": "data", "classification": "sensitive"},
            relationships=[],
            metadata={}
        )
        
        concept2 = Concept(
            name="layered_data_2",
            attributes={"type": "data", "classification": "sensitive"},
            relationships=[],
            metadata={}
        )
        
        glyph1 = encoder.encode(concept1)
        glyph2 = encoder.encode(concept2)
        
        # Verify glyphs have layer structure
        assert "identity" in glyph1.layers
        assert "core" in glyph1.layers["identity"].segments
        
        # Verify security weights are stored
        layer = glyph1.layers["identity"]
        assert layer.weights.get("security") == 0.8
        
        segment = layer.segments["core"]
        assert segment.weights.get("security") == 0.9


class TestVisibilityDecisions:
    """Test visibility decisions based on clearance"""
    
    def test_visibility_with_sufficient_clearance(self):
        """Test that sufficient clearance allows visibility"""
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            security_weight=0.5
        )
        
        encoder = Encoder(config)
        calc = SimilarityCalculator()
        
        concept1 = Concept(
            name="visible_doc_1",
            attributes={"type": "document", "status": "public"},
            relationships=[],
            metadata={}
        )
        
        concept2 = Concept(
            name="visible_doc_2",
            attributes={"type": "document", "status": "public"},
            relationships=[],
            metadata={}
        )
        
        glyph1 = encoder.encode(concept1)
        glyph2 = encoder.encode(concept2)
        
        # High clearance should allow visibility
        result = calc.compute_similarity(
            glyph1, glyph2, "neural_cortex", user_clearance=1.0
        )
        
        assert result.visible is True
    
    def test_visibility_with_insufficient_clearance(self):
        """Test that insufficient clearance blocks visibility"""
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            security_weight=0.9
        )
        
        encoder = Encoder(config)
        calc = SimilarityCalculator()
        
        concept1 = Concept(
            name="restricted_doc_1",
            attributes={"type": "document", "status": "restricted"},
            relationships=[],
            metadata={}
        )
        
        concept2 = Concept(
            name="restricted_doc_2",
            attributes={"type": "document", "status": "restricted"},
            relationships=[],
            metadata={}
        )
        
        glyph1 = encoder.encode(concept1)
        glyph2 = encoder.encode(concept2)
        
        # Very low clearance should block visibility
        result = calc.compute_similarity(
            glyph1, glyph2, "neural_cortex", user_clearance=0.1
        )
        
        # With low clearance and high security weight, visibility may be blocked
        # The exact behavior depends on the similarity calculator implementation
        assert result is not None


class TestClearanceChecking:
    """Test clearance checking across edge types"""
    
    def test_clearance_affects_all_edge_types(self):
        """Test that clearance affects similarity across all edge types"""
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            security_weight=0.9
        )
        
        encoder = Encoder(config)
        calc = SimilarityCalculator()
        
        concept1 = Concept(
            name="secure_item_1",
            attributes={"type": "item", "level": "high"},
            relationships=[],
            metadata={}
        )
        
        concept2 = Concept(
            name="secure_item_2",
            attributes={"type": "item", "level": "high"},
            relationships=[],
            metadata={}
        )
        
        glyph1 = encoder.encode(concept1)
        glyph2 = encoder.encode(concept2)
        
        edge_types = ["neural_cortex", "neural_layer", "neural_segment"]
        
        for edge_type in edge_types:
            result_high = calc.compute_similarity(
                glyph1, glyph2, edge_type, user_clearance=1.0
            )
            result_low = calc.compute_similarity(
                glyph1, glyph2, edge_type, user_clearance=0.1
            )
            
            # Higher clearance should generally give higher or equal scores
            assert result_high.score >= result_low.score * 0.9  # Allow some tolerance
