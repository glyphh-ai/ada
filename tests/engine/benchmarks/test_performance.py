"""
Performance benchmarks for the Glyphh SDK.

These benchmarks measure the performance of core operations and can be used
to track performance over time and identify optimization opportunities.

Run with: pytest tests/benchmarks/ -v --tb=short
"""

import pytest
import time
import numpy as np
from typing import List

from glyphh.core.types import Concept, Vector
from glyphh.core.config import EncoderConfig
from glyphh.encoder.base import Encoder
from glyphh.similarity.calculator import SimilarityCalculator
from glyphh.edges.generator import EdgeGenerator
from glyphh.temporal.predictor import BeamSearchPredictor
from glyphh.core.ops import bind, bundle, cosine_similarity, generate_symbol
from tests.fixtures.loader import load_sample_concepts, get_config_by_name


class BenchmarkTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"\n{self.name}: {self.duration:.6f} seconds")


@pytest.mark.benchmark
class TestCoreOperationsBenchmarks:
    """Benchmarks for core vector operations."""

    def test_bind_operation_performance(self):
        """Benchmark bind operation performance."""
        dimension = 10000
        seed = 42

        # Generate test vectors
        v1 = generate_symbol(seed, "role", dimension)
        v2 = generate_symbol(seed + 1, "value", dimension)

        # Benchmark single bind
        iterations = 1000
        with BenchmarkTimer(f"Bind operation ({iterations} iterations, {dimension}D)"):
            for _ in range(iterations):
                result = bind(v1, v2)

        assert result is not None

    def test_bundle_operation_performance(self):
        """Benchmark bundle operation performance."""
        dimension = 10000
        seed = 42
        num_vectors = 10

        # Generate test vectors
        vectors = [generate_symbol(seed + i, f"v{i}", dimension) for i in range(num_vectors)]

        # Benchmark bundle
        iterations = 100
        with BenchmarkTimer(
            f"Bundle operation ({iterations} iterations, {num_vectors} vectors, {dimension}D)"
        ):
            for _ in range(iterations):
                result = bundle(vectors)

        assert result is not None

    def test_cosine_similarity_performance(self):
        """Benchmark cosine similarity computation."""
        dimension = 10000
        seed = 42

        # Generate test vectors
        v1 = generate_symbol(seed, "v1", dimension)
        v2 = generate_symbol(seed + 1, "v2", dimension)

        # Benchmark similarity
        iterations = 1000
        with BenchmarkTimer(f"Cosine similarity ({iterations} iterations, {dimension}D)"):
            for _ in range(iterations):
                score = cosine_similarity(v1, v2)

        assert -1.0 <= score <= 1.0

    def test_symbol_generation_performance(self):
        """Benchmark symbol vector generation."""
        dimension = 10000
        seed = 42

        # Benchmark generation
        iterations = 1000
        with BenchmarkTimer(f"Symbol generation ({iterations} iterations, {dimension}D)"):
            for i in range(iterations):
                vector = generate_symbol(seed, f"key_{i}", dimension)

        assert vector is not None

    def test_symbol_generation_with_caching(self):
        """Benchmark symbol generation with caching."""
        config = EncoderConfig(dimension=10000, seed=42)
        encoder = Encoder(config)

        # First pass - populate cache
        keys = [f"key_{i}" for i in range(100)]
        for key in keys:
            encoder.generate_symbol(key)

        # Second pass - use cache
        iterations = 1000
        with BenchmarkTimer(
            f"Symbol generation with cache ({iterations} iterations, 100 keys)"
        ):
            for _ in range(iterations):
                for key in keys:
                    vector = encoder.generate_symbol(key)

        assert len(encoder.symbol_cache) == 100


@pytest.mark.benchmark
class TestEncodingBenchmarks:
    """Benchmarks for encoding operations."""

    def test_encode_single_concept_performance(self):
        """Benchmark encoding a single concept."""
        config = get_config_by_name("standard_config")
        encoder = Encoder(config)
        concept = load_sample_concepts()[0]

        # Benchmark encoding
        iterations = 100
        with BenchmarkTimer(f"Encode single concept ({iterations} iterations)"):
            for _ in range(iterations):
                glyph = encoder.encode(concept)

        assert glyph is not None

    def test_encode_batch_concepts_performance(self):
        """Benchmark encoding multiple concepts."""
        config = get_config_by_name("standard_config")
        encoder = Encoder(config)
        concepts = load_sample_concepts()

        # Benchmark batch encoding
        with BenchmarkTimer(f"Encode batch of {len(concepts)} concepts"):
            glyphs = [encoder.encode(concept) for concept in concepts]

        assert len(glyphs) == len(concepts)

    def test_encode_with_different_dimensions(self):
        """Benchmark encoding with different vector dimensions."""
        concept = load_sample_concepts()[0]
        dimensions = [1000, 5000, 10000]

        for dim in dimensions:
            config = EncoderConfig(dimension=dim, seed=42)
            encoder = Encoder(config)

            iterations = 50
            with BenchmarkTimer(f"Encode concept ({iterations} iterations, {dim}D)"):
                for _ in range(iterations):
                    glyph = encoder.encode(concept)

            assert glyph is not None


@pytest.mark.benchmark
class TestSimilarityBenchmarks:
    """Benchmarks for similarity computation."""

    def test_similarity_computation_performance(self):
        """Benchmark similarity computation."""
        config = get_config_by_name("standard_config")
        encoder = Encoder(config)
        concepts = load_sample_concepts()[:2]

        glyph1 = encoder.encode(concepts[0])
        glyph2 = encoder.encode(concepts[1])

        calc = SimilarityCalculator()

        # Benchmark similarity
        iterations = 100
        with BenchmarkTimer(f"Similarity computation ({iterations} iterations)"):
            for _ in range(iterations):
                result = calc.compute_similarity(glyph1, glyph2, "neural_cortex")

        assert result is not None

    def test_pairwise_similarity_performance(self):
        """Benchmark computing all pairwise similarities."""
        config = get_config_by_name("minimal_config")  # Use smaller config for speed
        encoder = Encoder(config)
        concepts = load_sample_concepts()[:5]

        glyphs = [encoder.encode(concept) for concept in concepts]
        calc = SimilarityCalculator()

        # Benchmark pairwise similarities
        num_comparisons = len(glyphs) * (len(glyphs) - 1) // 2
        with BenchmarkTimer(f"Pairwise similarities ({num_comparisons} comparisons)"):
            results = []
            for i, glyph1 in enumerate(glyphs):
                for j, glyph2 in enumerate(glyphs):
                    if i < j:
                        result = calc.compute_similarity(glyph1, glyph2, "neural_cortex")
                        results.append(result)

        assert len(results) == num_comparisons

    def test_similarity_across_edge_types_performance(self):
        """Benchmark similarity computation across different edge types."""
        config = get_config_by_name("standard_config")
        encoder = Encoder(config)
        concepts = load_sample_concepts()[:2]

        glyph1 = encoder.encode(concepts[0])
        glyph2 = encoder.encode(concepts[1])

        calc = SimilarityCalculator()
        edge_types = ["neural_cortex", "neural_layer", "neural_segment", "neural_role"]

        # Benchmark each edge type
        for edge_type in edge_types:
            iterations = 50
            with BenchmarkTimer(f"Similarity {edge_type} ({iterations} iterations)"):
                for _ in range(iterations):
                    result = calc.compute_similarity(glyph1, glyph2, edge_type)

            assert result is not None


@pytest.mark.benchmark
class TestEdgeGenerationBenchmarks:
    """Benchmarks for edge generation."""

    def test_spatial_edge_generation_performance(self):
        """Benchmark spatial edge generation."""
        config = get_config_by_name("standard_config")
        encoder = Encoder(config)
        concept = load_sample_concepts()[0]

        glyph = encoder.encode(concept)
        edge_gen = EdgeGenerator()

        # Benchmark edge generation
        iterations = 100
        with BenchmarkTimer(f"Spatial edge generation ({iterations} iterations)"):
            for _ in range(iterations):
                edges = edge_gen.generate_spatial_edges(glyph)

        assert len(edges) > 0

    def test_temporal_edge_generation_performance(self):
        """Benchmark temporal edge generation."""
        config = get_config_by_name("standard_config")
        encoder = Encoder(config)
        concepts = load_sample_concepts()[:2]

        glyph1 = encoder.encode(concepts[0])
        glyph2 = encoder.encode(concepts[1])

        edge_gen = EdgeGenerator()

        # Benchmark temporal edge generation
        iterations = 100
        with BenchmarkTimer(f"Temporal edge generation ({iterations} iterations)"):
            for _ in range(iterations):
                edges = edge_gen.generate_temporal_edges(glyph1, glyph2)

        assert len(edges) > 0


@pytest.mark.benchmark
class TestTemporalBenchmarks:
    """Benchmarks for temporal operations."""

    def test_temporal_prediction_performance(self):
        """Benchmark temporal prediction."""
        config = get_config_by_name("minimal_config")  # Use smaller config for speed
        encoder = Encoder(config)
        concepts = load_sample_concepts()[:4]

        history = [encoder.encode(concept) for concept in concepts]
        predictor = BeamSearchPredictor(beam_width=3, drift_reduction=True)

        # Benchmark prediction
        with BenchmarkTimer("Temporal prediction (2 time steps, beam_width=3)"):
            result = predictor.predict(history, time_intervals=2, hierarchy_level="cortex")

        assert result is not None
        assert len(result.predictions) <= 3


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Benchmarks for scalability testing."""

    def test_encode_many_concepts(self):
        """Benchmark encoding many concepts."""
        config = get_config_by_name("minimal_config")
        encoder = Encoder(config)

        # Create many concepts
        num_concepts = 100
        concepts = []
        for i in range(num_concepts):
            concepts.append(
                Concept(
                    name=f"concept_{i}",
                    attributes={"type": f"type_{i % 10}", "value": f"value_{i}"},
                    relationships=[],
                    metadata={},
                )
            )

        # Benchmark encoding
        with BenchmarkTimer(f"Encode {num_concepts} concepts"):
            glyphs = [encoder.encode(concept) for concept in concepts]

        assert len(glyphs) == num_concepts

    def test_large_dimension_performance(self):
        """Benchmark operations with large dimensions."""
        dimension = 20000
        config = EncoderConfig(dimension=dimension, seed=42)
        encoder = Encoder(config)
        concept = load_sample_concepts()[0]

        # Benchmark encoding with large dimension
        iterations = 10
        with BenchmarkTimer(f"Encode with {dimension}D ({iterations} iterations)"):
            for _ in range(iterations):
                glyph = encoder.encode(concept)

        assert glyph is not None

    def test_memory_efficiency(self):
        """Test memory efficiency with many glyphs."""
        config = get_config_by_name("minimal_config")
        encoder = Encoder(config)
        concepts = load_sample_concepts()

        # Encode many times to test memory
        num_iterations = 100
        with BenchmarkTimer(f"Memory test: encode {len(concepts)} concepts {num_iterations} times"):
            for _ in range(num_iterations):
                glyphs = [encoder.encode(concept) for concept in concepts]

        # Verify cache size is reasonable
        assert len(encoder.symbol_cache) < 1000  # Should not grow unbounded


@pytest.mark.benchmark
class TestBenchmarkSummary:
    """Summary of all benchmarks."""

    def test_benchmark_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print("\nCore Operations:")
        print("  - Bind: Element-wise multiplication of bipolar vectors")
        print("  - Bundle: Majority-vote aggregation of multiple vectors")
        print("  - Cosine Similarity: Normalized dot product")
        print("  - Symbol Generation: Deterministic vector generation")
        print("\nEncoding:")
        print("  - Single concept encoding with hierarchical structure")
        print("  - Batch encoding with symbol caching")
        print("  - Variable dimension performance")
        print("\nSimilarity:")
        print("  - Neural cortex (global) similarity")
        print("  - Pairwise similarity computation")
        print("  - Multi-level edge type comparison")
        print("\nTemporal:")
        print("  - Beam search prediction with drift reduction")
        print("\nScalability:")
        print("  - Large batch encoding (100+ concepts)")
        print("  - High-dimensional vectors (20000D)")
        print("  - Memory efficiency testing")
        print("\n" + "=" * 80)
        print("Run individual benchmarks with: pytest tests/benchmarks/ -v -k <test_name>")
        print("=" * 80 + "\n")
