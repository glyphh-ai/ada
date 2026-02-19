"""
Complete integration tests for Task 19.1 - Wire all components together

Tests that all SDK components integrate correctly and work together
in complete workflows.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from glyphh import (
    # Core types
    Vector, Concept, Edge, Glyph, Layer, Segment, compute_space_id,
    # Configuration
    EncoderConfig,
    # Encoder
    Encoder,
    # Edge Generator
    EdgeGenerator,
    # Similarity
    SimilarityCalculator, SimilarityResult,
    # Fact Tree
    FactTree, FactNode, Citation,
    # Temporal
    TemporalEncoder, BeamSearchPredictor, PredictionResult, Prediction, TrendStatistics,
    # Model Packaging
    GlyphhModel,
    # Visualization
    visualize_similarity_graph, create_similarity_matrix,
    visualize_vector_heatmap, visualize_vector_comparison,
    visualize_fact_tree, print_fact_tree,
    # Exceptions
    GlyphhException, ConfigurationException, VectorSpaceException,
    DimensionMismatchException, SpaceIdMismatchException,
    BipolarConstraintException, EncodingException,
    ModelValidationException, ValidationException, OperationException,
    log_encoding_failure, log_validation_failure,
    # HDC operations
    bind, bundle, cosine_similarity, hamming_similarity,
    generate_symbol,
)


class TestCompleteIntegration:
    """Test complete integration of all SDK components"""
    
    def test_all_imports_available(self):
        """Verify all public API components are importable"""
        # Core types
        assert Vector is not None
        assert Concept is not None
        assert Edge is not None
        assert Glyph is not None
        assert Layer is not None
        assert Segment is not None
        assert compute_space_id is not None
        
        # Configuration
        assert EncoderConfig is not None
        
        # Encoder
        assert Encoder is not None
        
        # Edge Generator
        assert EdgeGenerator is not None
        
        # Similarity
        assert SimilarityCalculator is not None
        assert SimilarityResult is not None
        
        # Fact Tree
        assert FactTree is not None
        assert FactNode is not None
        assert Citation is not None
        
        # Temporal
        assert TemporalEncoder is not None
        assert BeamSearchPredictor is not None
        assert PredictionResult is not None
        assert Prediction is not None
        assert TrendStatistics is not None
        
        # Model Packaging
        assert GlyphhModel is not None
        
        # Visualization
        assert visualize_similarity_graph is not None
        assert create_similarity_matrix is not None
        assert visualize_vector_heatmap is not None
        assert visualize_vector_comparison is not None
        assert visualize_fact_tree is not None
        assert print_fact_tree is not None
        
        # Exceptions
        assert GlyphhException is not None
        assert ConfigurationException is not None
        assert VectorSpaceException is not None
        assert DimensionMismatchException is not None
        assert SpaceIdMismatchException is not None
        assert BipolarConstraintException is not None
        assert EncodingException is not None
        assert ModelValidationException is not None
        assert ValidationException is not None
        assert OperationException is not None
        assert log_encoding_failure is not None
        assert log_validation_failure is not None
        
        # HDC operations
        assert bind is not None
        assert bundle is not None
        assert cosine_similarity is not None
        assert hamming_similarity is not None
        assert generate_symbol is not None
    
    def test_complete_workflow_encode_to_similarity(self):
        """Test complete workflow: encode → edges → similarity → fact tree"""
        # 1. Create configuration
        config = EncoderConfig(dimension=1000, seed=42)
        
        # 2. Create encoder
        encoder = Encoder(config)
        
        # 3. Encode concepts
        concept1 = Concept(
            name="red car",
            attributes={"type": "car", "color": "red", "size": "medium"},
            relationships=[("has_part", "wheels"), ("used_for", "transportation")],
            metadata={"domain": "automotive"}
        )
        
        concept2 = Concept(
            name="blue car",
            attributes={"type": "car", "color": "blue", "size": "medium"},
            relationships=[("has_part", "wheels"), ("used_for", "transportation")],
            metadata={"domain": "automotive"}
        )
        
        glyph1 = encoder.encode(concept1)
        glyph2 = encoder.encode(concept2)
        
        # 4. Generate edges
        edge_gen = EdgeGenerator()
        edges1 = edge_gen.generate_spatial_edges(glyph1)
        edges2 = edge_gen.generate_spatial_edges(glyph2)
        
        # Verify edges generated
        assert len(edges1) > 0
        assert len(edges2) > 0
        
        # 5. Compute similarity
        calc = SimilarityCalculator()
        result = calc.compute_similarity(glyph1, glyph2, "neural_cortex")
        
        # Verify similarity result
        assert isinstance(result, SimilarityResult)
        assert 0.0 <= result.score <= 1.0
        assert result.visible is True
        assert result.fact_tree is not None
        
        # 6. Verify fact tree
        fact_tree_text = result.fact_tree.to_text()
        assert "Similarity Computation" in fact_tree_text
        assert "Raw Similarity" in fact_tree_text
        assert "Final Score" in fact_tree_text
    
    def test_complete_workflow_temporal_prediction(self):
        """Test complete workflow: encode → temporal deltas → prediction"""
        # 1. Create configuration
        config = EncoderConfig(dimension=1000, seed=42)
        
        # 2. Create encoder
        encoder = Encoder(config)
        
        # 3. Create temporal encoder (no arguments needed)
        temporal_encoder = TemporalEncoder()
        
        # 4. Create historical versions
        history = []
        for i in range(5):
            concept = Concept(
                name=f"evolving_concept_v{i}",
                attributes={"type": "data", "version": i, "value": i * 10},
                relationships=[],
                metadata={"timestamp": f"2024-01-{i+1:02d}"}
            )
            glyph = encoder.encode(concept)
            history.append(glyph)
        
        # 5. Predict future state
        predictor = BeamSearchPredictor(beam_width=3, drift_reduction=True)
        result = predictor.predict(history, time_intervals=2, hierarchy_level="cortex")
        
        # Verify prediction result
        assert isinstance(result, PredictionResult)
        assert len(result.predictions) <= 3  # beam_width
        assert all(isinstance(p, Prediction) for p in result.predictions)
        assert all(0.0 <= p.confidence <= 1.0 for p in result.predictions)
        assert result.fact_tree is not None
        assert result.trends is not None
    
    def test_complete_workflow_model_packaging(self):
        """Test complete workflow: encode → package → save → load → validate"""
        # 1. Create configuration
        config = EncoderConfig(dimension=1000, seed=42)
        
        # 2. Create encoder
        encoder = Encoder(config)
        
        # 3. Encode concepts
        concepts = [
            Concept(
                name=f"concept_{i}",
                attributes={"type": "test", "id": i},
                relationships=[],
                metadata={}
            )
            for i in range(5)
        ]
        
        glyphs = [encoder.encode(c) for c in concepts]
        
        # 4. Create model
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=config,
            glyphs=glyphs,
            custom_encoders={},
            metadata={"description": "Integration test model"}
        )
        
        # 5. Validate model
        errors = model.validate_completeness()
        assert len(errors) == 0
        
        # 6. Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.glyphh"
            model.to_file(str(model_path))
            
            loaded_model = GlyphhModel.from_file(str(model_path))
            
            # Verify loaded model
            assert loaded_model.name == model.name
            assert loaded_model.version == model.version
            assert len(loaded_model.glyphs) == len(model.glyphs)
            assert loaded_model.encoder_config.dimension == config.dimension
            assert loaded_model.encoder_config.seed == config.seed
    
    def test_complete_workflow_with_visualization(self):
        """Test complete workflow with visualization components"""
        # 1. Create configuration
        config = EncoderConfig(dimension=1000, seed=42)
        
        # 2. Create encoder
        encoder = Encoder(config)
        
        # 3. Encode concepts
        concepts = [
            Concept(
                name=f"concept_{i}",
                attributes={"type": "test", "id": i},
                relationships=[],
                metadata={}
            )
            for i in range(3)
        ]
        
        glyphs = [encoder.encode(c) for c in concepts]
        
        # 4. Compute similarities
        calc = SimilarityCalculator()
        results = []
        for i in range(len(glyphs)):
            for j in range(i + 1, len(glyphs)):
                result = calc.compute_similarity(glyphs[i], glyphs[j], "neural_cortex")
                results.append(result)
        
        # 5. Create similarity matrix
        matrix, names = create_similarity_matrix(glyphs, edge_type="neural_cortex")
        assert matrix.shape == (len(glyphs), len(glyphs))
        assert np.allclose(matrix, matrix.T)  # Symmetric
        assert np.allclose(np.diag(matrix), 1.0)  # Diagonal is 1.0
        
        # 6. Test fact tree printing
        if results:
            fact_tree_text = print_fact_tree(results[0].fact_tree)
            assert fact_tree_text is not None
            assert len(fact_tree_text) > 0
    
    def test_hdc_operations_integration(self):
        """Test HDC vector operations integration"""
        dim = 1000
        seed = 42

        # Generate symbols
        v1 = generate_symbol(seed, "test1", dim)
        v2 = generate_symbol(seed, "test2", dim)

        assert len(v1) == dim
        assert len(v2) == dim
        assert np.all(np.isin(v1, [-1, 1]))
        assert np.all(np.isin(v2, [-1, 1]))

        # Test bind
        bound = bind(v1, v2)
        assert len(bound) == dim
        assert np.all(np.isin(bound, [-1, 1]))

        # Test bundle
        bundled = bundle([v1, v2])
        assert len(bundled) == dim
        assert np.all(np.isin(bundled, [-1, 1]))

        # Test similarity
        cos_sim = cosine_similarity(v1, v2)
        ham_sim = hamming_similarity(v1, v2)

        assert -1.0 <= cos_sim <= 1.0
        assert 0.0 <= ham_sim <= 1.0

        # Verify relationship between cosine and hamming
        expected_ham = (cos_sim + 1.0) / 2.0
        assert abs(ham_sim - expected_ham) < 0.01
    
    def test_error_handling_integration(self):
        """Test error handling across components"""
        # 1. Test configuration validation
        with pytest.raises(ConfigurationException):
            EncoderConfig(dimension=-1, seed=42)
        
        # 2. Test vector space mismatch
        config1 = EncoderConfig(dimension=1000, seed=42)
        config2 = EncoderConfig(dimension=1000, seed=43)  # Different seed
        
        encoder1 = Encoder(config1)
        encoder2 = Encoder(config2)
        
        concept = Concept(
            name="test",
            attributes={"type": "test"},
            relationships=[],
            metadata={}
        )
        
        glyph1 = encoder1.encode(concept)
        glyph2 = encoder2.encode(concept)
        
        # Should raise exception for cross-space comparison
        calc = SimilarityCalculator()
        with pytest.raises((SpaceIdMismatchException, ValueError)):
            calc.compute_similarity(glyph1, glyph2, "neural_cortex")
        
        # 3. Test dimension mismatch
        v1 = Vector(
            data=np.array([1, -1, 1, -1], dtype=np.int8),
            dimension=4,
            space_id="test"
        )
        v2 = Vector(
            data=np.array([1, -1, 1, -1, 1, -1], dtype=np.int8),
            dimension=6,
            space_id="test"
        )
        
        with pytest.raises((DimensionMismatchException, ValueError)):
            bind(v1.data, v2.data)
        
        # 4. Test bipolar constraint
        with pytest.raises((BipolarConstraintException, ValueError)):
            Vector(
                data=np.array([1, 0, -1, 2], dtype=np.int8),  # Invalid values
                dimension=4,
                space_id="test"
            )
    
    def test_edge_types_integration(self):
        """Test all 8 edge types work correctly"""
        # 1. Create configuration
        config = EncoderConfig(dimension=1000, seed=42)
        
        # 2. Create encoder
        encoder = Encoder(config)
        
        # 3. Encode concept
        concept = Concept(
            name="test_concept",
            attributes={"type": "test", "value": "data"},
            relationships=[],
            metadata={}
        )
        
        glyph = encoder.encode(concept)
        
        # 4. Generate spatial edges
        edge_gen = EdgeGenerator()
        spatial_edges = edge_gen.generate_spatial_edges(glyph)
        
        # Verify spatial edge types
        edge_types = {e.type for e in spatial_edges}
        assert "neural_cortex" in edge_types
        assert "neural_layer" in edge_types
        assert "neural_segment" in edge_types
        assert "neural_role" in edge_types
        
        # 5. Create temporal version
        concept_v2 = Concept(
            name="test_concept_v2",
            attributes={"type": "test", "value": "updated_data"},
            relationships=[],
            metadata={}
        )
        glyph_v2 = encoder.encode(concept_v2)
        
        # 6. Generate temporal edges
        temporal_edges = edge_gen.generate_temporal_edges(glyph, glyph_v2)
        
        # Verify temporal edge types
        temporal_types = {e.type for e in temporal_edges}
        assert "temporal_cortex" in temporal_types
        assert "temporal_layer" in temporal_types
        assert "temporal_segment" in temporal_types
        assert "temporal_role" in temporal_types
        
        # 7. Test similarity computation for each edge type
        calc = SimilarityCalculator()
        
        for edge_type in ["neural_cortex", "neural_layer", "neural_segment", "neural_role"]:
            result = calc.compute_similarity(glyph, glyph_v2, edge_type)
            assert isinstance(result, SimilarityResult)
            assert 0.0 <= result.score <= 1.0
    
    def test_security_weighting_integration(self):
        """Test security weighting across components"""
        # 1. Create configuration with security weights (new explicit structure)
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            security_weight=0.9
        )
        
        # 2. Create encoder
        encoder = Encoder(config)
        
        # 3. Encode concepts
        concept1 = Concept(
            name="secure_concept_1",
            attributes={"type": "classified", "level": "high"},
            relationships=[],
            metadata={}
        )
        
        concept2 = Concept(
            name="secure_concept_2",
            attributes={"type": "classified", "level": "medium"},
            relationships=[],
            metadata={}
        )
        
        glyph1 = encoder.encode(concept1)
        glyph2 = encoder.encode(concept2)
        
        # 4. Compute similarity with different clearance levels
        calc = SimilarityCalculator()
        
        # Full clearance
        result_full = calc.compute_similarity(
            glyph1, glyph2, "neural_cortex", user_clearance=1.0
        )
        
        # Partial clearance
        result_partial = calc.compute_similarity(
            glyph1, glyph2, "neural_cortex", user_clearance=0.5
        )
        
        # Verify security filtering affects score
        assert result_full.score >= result_partial.score
        
        # Verify fact tree contains security information
        fact_tree_text = result_full.fact_tree.to_text()
        assert "Security Weights" in fact_tree_text or "security" in fact_tree_text.lower()


class TestComponentInteroperability:
    """Test that components work together seamlessly"""
    
    def test_encoder_to_edge_generator(self):
        """Test encoder output works with edge generator"""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        edge_gen = EdgeGenerator()
        
        concept = Concept(
            name="test",
            attributes={"type": "test"},
            relationships=[],
            metadata={}
        )
        
        glyph = encoder.encode(concept)
        edges = edge_gen.generate_spatial_edges(glyph)
        
        assert len(edges) > 0
        assert all(isinstance(e, Edge) for e in edges)
    
    def test_edge_generator_to_similarity_calculator(self):
        """Test edge generator output works with similarity calculator"""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        edge_gen = EdgeGenerator()
        calc = SimilarityCalculator()
        
        concept1 = Concept(name="test1", attributes={"type": "test"}, relationships=[], metadata={})
        concept2 = Concept(name="test2", attributes={"type": "test"}, relationships=[], metadata={})
        
        glyph1 = encoder.encode(concept1)
        glyph2 = encoder.encode(concept2)
        
        edges1 = edge_gen.generate_spatial_edges(glyph1)
        edges2 = edge_gen.generate_spatial_edges(glyph2)
        
        result = calc.compute_similarity(glyph1, glyph2, "neural_cortex")
        
        assert isinstance(result, SimilarityResult)
        assert result.fact_tree is not None
    
    def test_encoder_to_temporal_encoder(self):
        """Test encoder output works with temporal encoder"""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        temporal_encoder = TemporalEncoder()  # No arguments needed
        
        concept1 = Concept(name="test_v1", attributes={"version": 1}, relationships=[], metadata={})
        concept2 = Concept(name="test_v2", attributes={"version": 2}, relationships=[], metadata={})
        
        glyph1 = encoder.encode(concept1)
        glyph2 = encoder.encode(concept2)
        
        delta = temporal_encoder.compute_temporal_delta(
            glyph1.global_cortex,
            glyph2.global_cortex
        )
        
        assert isinstance(delta, Vector)
        assert delta.dimension == config.dimension
    
    def test_temporal_encoder_to_predictor(self):
        """Test temporal encoder output works with predictor"""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        predictor = BeamSearchPredictor(beam_width=3)
        
        history = []
        for i in range(5):
            concept = Concept(
                name=f"version_{i}",
                attributes={"version": i},
                relationships=[],
                metadata={}
            )
            glyph = encoder.encode(concept)
            history.append(glyph)
        
        result = predictor.predict(history, time_intervals=2, hierarchy_level="cortex")
        
        assert isinstance(result, PredictionResult)
        assert len(result.predictions) > 0
    
    def test_encoder_to_model_packaging(self):
        """Test encoder output works with model packaging"""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        concepts = [
            Concept(name=f"concept_{i}", attributes={"id": i}, relationships=[], metadata={})
            for i in range(3)
        ]
        
        glyphs = [encoder.encode(c) for c in concepts]
        
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=config,
            glyphs=glyphs,
            custom_encoders={},
            metadata={}
        )
        
        errors = model.validate_completeness()
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
