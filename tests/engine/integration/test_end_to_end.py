"""
End-to-end integration tests for the Glyphh SDK.

These tests verify complete workflows from concept encoding through similarity
computation, fact tree generation, and model packaging.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from glyphh.core.types import Concept
from glyphh.core.config import EncoderConfig
from glyphh.encoder.base import Encoder
from glyphh.similarity.calculator import SimilarityCalculator
from glyphh.edges.generator import EdgeGenerator
from glyphh.temporal.delta import TemporalEncoder
from glyphh.temporal.predictor import BeamSearchPredictor
from glyphh.model.package import GlyphhModel
from tests.fixtures.loader import load_sample_concepts, get_config_by_name


class TestEndToEndEncoding:
    """Test complete encoding workflow from concept to glyph."""

    def test_encode_single_concept(self):
        """Test encoding a single concept end-to-end."""
        # Load fixtures
        config = get_config_by_name("standard_config")
        concepts = load_sample_concepts()
        red_car = concepts[0]

        # Create encoder
        encoder = Encoder(config)

        # Encode concept
        glyph = encoder.encode(red_car)

        # Verify glyph structure
        assert glyph is not None
        assert glyph.name == red_car.name
        assert glyph.space_id == encoder.space_id
        assert glyph.global_cortex is not None
        assert len(glyph.layers) >= 1  # At least one layer

        # Verify hierarchical structure
        for layer_name, layer in glyph.layers.items():
            assert layer.cortex is not None
            assert len(layer.segments) >= 1  # At least one segment
            for segment_name, segment in layer.segments.items():
                assert segment.cortex is not None
                assert len(segment.roles) > 0

    def test_encode_multiple_concepts(self):
        """Test encoding multiple concepts with same encoder."""
        config = get_config_by_name("standard_config")
        concepts = load_sample_concepts()[:3]  # red_car, blue_car, red_truck

        encoder = Encoder(config)
        glyphs = [encoder.encode(concept) for concept in concepts]

        # Verify all glyphs have same space_id
        space_ids = {glyph.space_id for glyph in glyphs}
        assert len(space_ids) == 1

        # Verify glyphs are different
        for i, glyph1 in enumerate(glyphs):
            for j, glyph2 in enumerate(glyphs):
                if i != j:
                    assert not np.array_equal(glyph1.global_cortex.data, glyph2.global_cortex.data)

    def test_deterministic_encoding_across_instances(self):
        """Test that encoding is deterministic across encoder instances for semantic content."""
        config = get_config_by_name("standard_config")
        concept = load_sample_concepts()[0]

        # Create two separate encoders with same config
        encoder1 = Encoder(config)
        encoder2 = Encoder(config)

        # Encode same concept with both encoders
        glyph1 = encoder1.encode(concept)
        glyph2 = encoder2.encode(concept)

        # Verify identical space_id
        assert glyph1.space_id == glyph2.space_id
        
        # Verify semantic layer cortices are identical (deterministic encoding)
        # Note: global_cortex includes _temporal layer which differs due to timestamps
        assert np.array_equal(
            glyph1.layers["semantic"].cortex.data,
            glyph2.layers["semantic"].cortex.data
        )


class TestEndToEndSimilarity:
    """Test complete similarity computation workflow."""

    def test_similarity_computation_workflow(self):
        """Test complete workflow: encode → generate edges → compute similarity."""
        # Setup
        config = get_config_by_name("standard_config")
        concepts = load_sample_concepts()
        red_car = concepts[0]
        blue_car = concepts[1]

        # Encode concepts
        encoder = Encoder(config)
        glyph1 = encoder.encode(red_car)
        glyph2 = encoder.encode(blue_car)

        # Generate edges
        edge_gen = EdgeGenerator()
        edges1 = edge_gen.generate_spatial_edges(glyph1)
        edges2 = edge_gen.generate_spatial_edges(glyph2)

        # Verify edges generated
        assert len(edges1) > 0
        assert len(edges2) > 0

        # Compute similarity
        calc = SimilarityCalculator()
        result = calc.compute_similarity(glyph1, glyph2, "neural_cortex")

        # Verify result
        assert result is not None
        assert -1.0 <= result.score <= 1.0
        assert result.glyph1 == glyph1
        assert result.glyph2 == glyph2
        assert result.fact_tree is not None

    def test_similarity_across_edge_types(self):
        """Test similarity computation across all edge types."""
        config = get_config_by_name("standard_config")
        concepts = load_sample_concepts()[:2]

        encoder = Encoder(config)
        glyph1 = encoder.encode(concepts[0])
        glyph2 = encoder.encode(concepts[1])

        calc = SimilarityCalculator()

        # Test all spatial edge types
        edge_types = ["neural_cortex", "neural_layer", "neural_segment", "neural_role"]

        for edge_type in edge_types:
            result = calc.compute_similarity(glyph1, glyph2, edge_type)
            assert result is not None
            assert -1.0 <= result.score <= 1.0

    def test_similarity_with_security_filtering(self):
        """Test similarity computation with security filtering."""
        config = get_config_by_name("standard_config")
        concepts = load_sample_concepts()[:2]

        encoder = Encoder(config)
        glyph1 = encoder.encode(concepts[0])
        glyph2 = encoder.encode(concepts[1])

        calc = SimilarityCalculator()

        # Test with different clearance levels
        clearances = [0.0, 0.5, 1.0]

        for clearance in clearances:
            result = calc.compute_similarity(
                glyph1, glyph2, "neural_cortex", user_clearance=clearance
            )
            assert result is not None
            # Lower clearance should generally result in lower scores
            assert -1.0 <= result.score <= 1.0


class TestEndToEndTemporal:
    """Test complete temporal reasoning workflow."""

    def test_temporal_delta_workflow(self):
        """Test complete temporal delta computation workflow."""
        config = get_config_by_name("standard_config")
        concepts = load_sample_concepts()[:2]

        # Encode two versions
        encoder = Encoder(config)
        glyph_v1 = encoder.encode(concepts[0])
        glyph_v2 = encoder.encode(concepts[1])

        # Compute temporal delta
        edge_gen = EdgeGenerator()
        temporal_edges = edge_gen.generate_temporal_edges(glyph_v1, glyph_v2)

        # Verify temporal edges
        assert len(temporal_edges) > 0
        for edge in temporal_edges:
            assert edge.type.startswith("temporal_")
            assert edge.source == glyph_v1.identifier
            assert edge.target == glyph_v2.identifier

    def test_temporal_prediction_workflow(self):
        """Test complete temporal prediction workflow."""
        config = get_config_by_name("minimal_config")  # Use smaller config for speed
        concepts = load_sample_concepts()[:4]

        # Encode historical versions
        encoder = Encoder(config)
        history = [encoder.encode(concept) for concept in concepts]

        # Predict future state
        predictor = BeamSearchPredictor(beam_width=3, drift_reduction=True)
        result = predictor.predict(history, time_intervals=2, hierarchy_level="cortex")

        # Verify predictions
        assert result is not None
        assert len(result.predictions) <= 3  # beam_width
        assert result.trends is not None
        assert result.fact_tree is not None

        # Verify all predictions are valid
        for prediction in result.predictions:
            assert prediction.vector is not None
            assert 0.0 <= prediction.confidence <= 1.0
            assert len(prediction.path) == 2  # time_intervals


class TestEndToEndModelPackaging:
    """Test complete model packaging workflow."""

    def test_model_packaging_round_trip(self):
        """Test complete workflow: encode → package → save → load."""
        config = get_config_by_name("standard_config")
        concepts = load_sample_concepts()[:3]

        # Encode concepts
        encoder = Encoder(config)
        glyphs = [encoder.encode(concept) for concept in concepts]

        # Create model
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=config,
            glyphs=glyphs,
            metadata={"description": "Test model for integration testing"},
        )

        # Validate model
        errors = model.validate_completeness()
        assert len(errors) == 0

        # Save to file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.glyphh"
            model.to_file(str(model_path))

            # Verify file exists
            assert model_path.exists()

            # Load from file
            loaded_model = GlyphhModel.from_file(str(model_path))

            # Verify loaded model matches original
            assert loaded_model.name == model.name
            assert loaded_model.version == model.version
            assert loaded_model.encoder_config.dimension == model.encoder_config.dimension
            assert len(loaded_model.glyphs) == len(model.glyphs)

    def test_model_with_metadata(self):
        """Test model packaging with complex metadata."""
        config = get_config_by_name("standard_config")

        # Encode concepts
        encoder = Encoder(config)
        concepts = load_sample_concepts()[:2]
        glyphs = [encoder.encode(concept) for concept in concepts]

        # Create model with metadata
        model = GlyphhModel(
            name="metadata_model",
            version="1.0.0",
            encoder_config=config,
            glyphs=glyphs,
            metadata={"encoder_type": "standard", "domain": "automotive"},
        )

        # Validate and save
        errors = model.validate_completeness()
        assert len(errors) == 0

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "metadata_model.glyphh"
            model.to_file(str(model_path))

            # Load and verify
            loaded_model = GlyphhModel.from_file(str(model_path))
            assert loaded_model.name == model.name
            assert loaded_model.metadata["domain"] == "automotive"


class TestEndToEndCLIWorkflow:
    """Test complete CLI workflow integration."""

    def test_cli_build_test_package_workflow(self):
        """Test complete CLI workflow: build → test → package."""
        # This test would use Click's testing utilities
        # For now, we verify the components work together
        config = get_config_by_name("standard_config")
        concepts = load_sample_concepts()[:2]

        # Simulate build phase
        encoder = Encoder(config)
        glyphs = [encoder.encode(concept) for concept in concepts]

        # Simulate test phase
        calc = SimilarityCalculator()
        result = calc.compute_similarity(glyphs[0], glyphs[1], "neural_cortex")
        assert result is not None

        # Simulate package phase
        model = GlyphhModel(
            name="cli_test_model",
            version="1.0.0",
            encoder_config=config,
            glyphs=glyphs,
            metadata={"source": "cli_workflow"},
        )

        errors = model.validate_completeness()
        assert len(errors) == 0


class TestEndToEndFactTree:
    """Test complete fact tree generation workflow."""

    def test_fact_tree_generation_workflow(self):
        """Test complete fact tree generation in similarity computation."""
        config = get_config_by_name("standard_config")
        concepts = load_sample_concepts()[:2]

        # Encode concepts
        encoder = Encoder(config)
        glyph1 = encoder.encode(concepts[0])
        glyph2 = encoder.encode(concepts[1])

        # Compute similarity with fact tree
        calc = SimilarityCalculator()
        result = calc.compute_similarity(glyph1, glyph2, "neural_cortex")

        # Verify fact tree structure
        fact_tree = result.fact_tree
        assert fact_tree is not None

        # Verify fact tree can be serialized
        fact_tree_json = fact_tree.to_json()
        assert fact_tree_json is not None
        assert "description" in fact_tree_json  # Root node has description

        # Verify fact tree can be rendered as text
        fact_tree_text = fact_tree.to_text()
        assert fact_tree_text is not None
        assert len(fact_tree_text) > 0

    def test_fact_tree_contains_citations(self):
        """Test that fact tree contains proper citations."""
        config = get_config_by_name("standard_config")
        concepts = load_sample_concepts()[:2]

        encoder = Encoder(config)
        glyph1 = encoder.encode(concepts[0])
        glyph2 = encoder.encode(concepts[1])

        calc = SimilarityCalculator()
        result = calc.compute_similarity(glyph1, glyph2, "neural_cortex")

        # Verify fact tree has citations
        fact_tree_json = result.fact_tree.to_json()
        # The fact tree should contain references to the glyphs
        fact_tree_str = str(fact_tree_json)
        assert glyph1.identifier in fact_tree_str or glyph1.name in fact_tree_str


class TestEndToEndPerformance:
    """Test end-to-end performance with realistic data volumes."""

    def test_encode_many_concepts(self):
        """Test encoding many concepts efficiently."""
        config = get_config_by_name("minimal_config")  # Use smaller config for speed
        concepts = load_sample_concepts()

        encoder = Encoder(config)

        # Encode all concepts
        glyphs = []
        for concept in concepts:
            glyph = encoder.encode(concept)
            glyphs.append(glyph)

        # Verify all encoded successfully
        assert len(glyphs) == len(concepts)

        # Verify symbol caching is working (cache should have entries)
        assert len(encoder.symbol_cache) > 0

    def test_compute_many_similarities(self):
        """Test computing similarities between many glyphs."""
        config = get_config_by_name("minimal_config")
        concepts = load_sample_concepts()[:5]

        encoder = Encoder(config)
        glyphs = [encoder.encode(concept) for concept in concepts]

        calc = SimilarityCalculator()

        # Compute all pairwise similarities
        results = []
        for i, glyph1 in enumerate(glyphs):
            for j, glyph2 in enumerate(glyphs):
                if i < j:  # Only compute upper triangle
                    result = calc.compute_similarity(glyph1, glyph2, "neural_cortex")
                    results.append(result)

        # Verify all computed successfully
        expected_count = len(glyphs) * (len(glyphs) - 1) // 2
        assert len(results) == expected_count


class TestEndToEndErrorHandling:
    """Test error handling across integrated components."""

    def test_cross_space_comparison_error(self):
        """Test that comparing glyphs from different spaces raises error."""
        config1 = get_config_by_name("minimal_config")
        config2 = get_config_by_name("standard_config")

        concepts = load_sample_concepts()[:2]

        # Encode with different configs (different spaces)
        encoder1 = Encoder(config1)
        encoder2 = Encoder(config2)

        glyph1 = encoder1.encode(concepts[0])
        glyph2 = encoder2.encode(concepts[1])

        # Attempt similarity computation should raise error
        calc = SimilarityCalculator()
        with pytest.raises(ValueError, match="different vector spaces"):
            calc.compute_similarity(glyph1, glyph2, "neural_cortex")

    def test_empty_glyphs_model_allowed(self):
        """Test that models with empty glyphs are valid — data loads later."""
        config = get_config_by_name("standard_config")

        # Empty glyphs should be allowed
        model = GlyphhModel(
            name="empty_model",
            version="1.0.0",
            encoder_config=config,
            glyphs=[],
            metadata={},
        )
        assert model.glyphs == []
