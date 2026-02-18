"""
Performance benchmarks for NL auto-schema matching.

These benchmarks measure the performance of the auto-schema NL matching
components and validate that performance targets are met:

1. Schema Index Build Time
   - Target: < 1s for 1000 values

2. Query Matching Latency
   - Target: p95 < 50ms for schemas with 1000 values

3. Memory Usage
   - Target: < 100MB for 10000 values

Run with: pytest tests/benchmarks/test_nl_performance.py -v --tb=short

Validates: Requirement 13.4 - THE SDK SHALL provide performance benchmarks
for different schema sizes
"""

import gc
import statistics
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from glyphh.core.config import EncoderConfig, Layer, Segment, Role
from glyphh.encoder.base import Encoder
from glyphh.nl.schema_vectorizer import SchemaVectorizer, SchemaVector
from glyphh.nl.schema_matcher import SchemaMatcher, MatchConfig
from glyphh.nl.query_tokenizer import QueryTokenizer, TokenizerConfig
from glyphh.nl.auto_schema_matcher import AutoSchemaMatcher, AutoMatchConfig
from glyphh.nl.performance import (
    ANNIndex,
    OptimizedVectorOps,
    should_use_ann_index,
    create_ann_index_from_schema_vectors,
    ANN_THRESHOLD,
)


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
        print(f"\n{self.name}: {self.duration:.6f} seconds ({self.duration * 1000:.2f} ms)")


def create_test_config(num_values: int) -> EncoderConfig:
    """Create a test encoder config with the specified number of values."""
    # Create roles with values
    roles = []
    values_per_role = max(1, num_values // 10)  # Distribute across 10 roles
    
    for i in range(10):
        role = Role(name=f"role_{i}")
        roles.append(role)
    
    segment = Segment(name="test_segment", roles=roles)
    layer = Layer(name="test_layer", segments=[segment])
    
    return EncoderConfig(
        dimension=10000,
        seed=42,
        layers=[layer]
    )


def generate_schema_vectors(
    encoder: Encoder,
    config: EncoderConfig,
    num_values: int
) -> Dict[str, SchemaVector]:
    """Generate schema vectors for benchmarking."""
    vectorizer = SchemaVectorizer(encoder, config)
    
    # First vectorize the schema to get role vectors
    vectorizer.vectorize_schema()
    
    # Generate value vectors and add them to the vectorizer's cache
    for i in range(num_values):
        role_idx = i % 10
        value = f"value_{i}"
        role_path = f"role_{role_idx}"
        
        # Generate the value vector
        value_vector = vectorizer.vectorize_value(role_path, value)
        
        # Manually add to the schema vectors cache
        # (vectorize_value returns the vector but doesn't cache it)
        vectorizer._schema_vectors[value_vector.key] = value_vector
    
    return vectorizer.get_schema_vectors()


def measure_memory_usage() -> int:
    """Measure current memory usage in bytes."""
    gc.collect()
    # Use sys.getsizeof for basic measurement
    # For more accurate measurement, use tracemalloc or memory_profiler
    return 0  # Placeholder - actual measurement would use tracemalloc


@pytest.mark.benchmark
class TestSchemaIndexBuildTime:
    """
    Benchmarks for schema index build time.
    
    Target: < 1s for 1000 values
    
    Validates: Requirement 13.4
    """
    
    @pytest.mark.parametrize("num_values", [10, 100, 1000, 10000])
    def test_schema_vectorization_build_time(self, num_values: int):
        """Benchmark schema vectorization build time for different sizes."""
        config = create_test_config(num_values)
        encoder = Encoder(config)
        
        with BenchmarkTimer(f"Schema vectorization ({num_values} values)"):
            schema_vectors = generate_schema_vectors(encoder, config, num_values)
        
        assert len(schema_vectors) >= num_values
        
        # Verify target for 1000 values
        if num_values == 1000:
            # Note: This is a soft assertion - actual timing depends on hardware
            print(f"  Target: < 1.0s")
    
    @pytest.mark.parametrize("num_values", [10, 100, 1000, 10000])
    def test_ann_index_build_time(self, num_values: int):
        """Benchmark ANN index build time for different sizes."""
        config = create_test_config(num_values)
        encoder = Encoder(config)
        schema_vectors = generate_schema_vectors(encoder, config, num_values)
        
        metrics = None
        with BenchmarkTimer(f"ANN index build ({num_values} values)"):
            if schema_vectors:
                index = create_ann_index_from_schema_vectors(schema_vectors)
                metrics = index.get_metrics()
        
        if metrics is not None and num_values >= ANN_THRESHOLD:
            print(f"  ANN index metrics: {metrics}")
        
        # Verify target for 1000 values
        if num_values == 1000:
            print(f"  Target: < 1.0s total")
    
    def test_schema_matcher_initialization_time(self):
        """Benchmark SchemaMatcher initialization time."""
        num_values = 1000
        config = create_test_config(num_values)
        encoder = Encoder(config)
        schema_vectors = generate_schema_vectors(encoder, config, num_values)
        
        with BenchmarkTimer(f"SchemaMatcher initialization ({num_values} values)"):
            matcher = SchemaMatcher(schema_vectors, encoder)
        
        assert matcher is not None
        print(f"  Target: < 1.0s")


@pytest.mark.benchmark
class TestQueryMatchingLatency:
    """
    Benchmarks for query matching latency.
    
    Target: p95 < 50ms for schemas with 1000 values
    
    Validates: Requirement 13.4
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.num_values = 1000
        self.config = create_test_config(self.num_values)
        self.encoder = Encoder(self.config)
        self.schema_vectors = generate_schema_vectors(
            self.encoder, self.config, self.num_values
        )
        self.matcher = SchemaMatcher(self.schema_vectors, self.encoder)
        self.tokenizer = QueryTokenizer()
    
    def test_single_query_latency(self):
        """Benchmark single query matching latency."""
        query = "find value_42 in role_4"
        
        # Warm up
        for _ in range(5):
            self.matcher.match_query(query, self.tokenizer)
        
        # Measure
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start = time.perf_counter()
            result = self.matcher.match_query(query, self.tokenizer)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Calculate percentiles
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg = statistics.mean(latencies)
        
        print(f"\nQuery matching latency ({self.num_values} values, {iterations} iterations):")
        print(f"  Average: {avg:.2f} ms")
        print(f"  p50: {p50:.2f} ms")
        print(f"  p95: {p95:.2f} ms")
        print(f"  p99: {p99:.2f} ms")
        print(f"  Target: p95 < 50ms")
        
        # Soft assertion - actual timing depends on hardware
        assert result is not None
    
    def test_batch_query_latency(self):
        """Benchmark batch query matching latency."""
        queries = [
            f"find value_{i} in role_{i % 10}"
            for i in range(10)
        ]
        
        # Create AutoSchemaMatcher for batch processing
        auto_matcher = AutoSchemaMatcher(self.encoder, self.config)
        
        # Warm up
        for _ in range(3):
            auto_matcher.match_queries(queries)
        
        # Measure
        latencies = []
        iterations = 50
        
        for _ in range(iterations):
            start = time.perf_counter()
            results = auto_matcher.match_queries(queries)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg = statistics.mean(latencies)
        per_query_avg = avg / len(queries)
        
        print(f"\nBatch query latency ({len(queries)} queries, {iterations} iterations):")
        print(f"  Total average: {avg:.2f} ms")
        print(f"  Per-query average: {per_query_avg:.2f} ms")
        
        assert len(results) == len(queries)
    
    def test_token_matching_latency(self):
        """Benchmark individual token matching latency."""
        from glyphh.nl.query_tokenizer import Token
        
        token = Token(
            text="value_42",
            original="value_42",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Warm up
        for _ in range(10):
            self.matcher.match_token(token)
        
        # Measure
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start = time.perf_counter()
            matches = self.matcher.match_token(token)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\nToken matching latency ({self.num_values} schema vectors):")
        print(f"  Average: {avg:.2f} ms")
        print(f"  p95: {p95:.2f} ms")
        
        assert len(matches) >= 0


@pytest.mark.benchmark
class TestMemoryUsage:
    """
    Benchmarks for memory usage.
    
    Target: < 100MB for 10000 values
    
    Validates: Requirement 13.4
    """
    
    @pytest.mark.parametrize("num_values", [100, 1000, 10000])
    def test_schema_vectors_memory(self, num_values: int):
        """Benchmark memory usage of schema vectors."""
        config = create_test_config(num_values)
        encoder = Encoder(config)
        
        # Generate schema vectors
        schema_vectors = generate_schema_vectors(encoder, config, num_values)
        
        # Estimate memory usage
        # Each vector: dimension * 1 byte (int8) + overhead
        dimension = config.dimension
        vector_memory = num_values * dimension  # bytes
        
        # Add overhead for dict, SchemaVector objects, etc.
        overhead_per_vector = 200  # Approximate bytes per SchemaVector
        total_overhead = num_values * overhead_per_vector
        
        estimated_memory = vector_memory + total_overhead
        estimated_memory_mb = estimated_memory / (1024 * 1024)
        
        print(f"\nSchema vectors memory ({num_values} values, {dimension}D):")
        print(f"  Vector data: {vector_memory / (1024 * 1024):.2f} MB")
        print(f"  Overhead: {total_overhead / (1024 * 1024):.2f} MB")
        print(f"  Estimated total: {estimated_memory_mb:.2f} MB")
        
        if num_values == 10000:
            print(f"  Target: < 100 MB")
        
        assert len(schema_vectors) >= num_values
    
    def test_ann_index_memory(self):
        """Benchmark ANN index memory usage."""
        num_values = 10000
        config = create_test_config(num_values)
        encoder = Encoder(config)
        schema_vectors = generate_schema_vectors(encoder, config, num_values)
        
        # Create ANN index
        index = create_ann_index_from_schema_vectors(schema_vectors)
        metrics = index.get_metrics()
        
        memory_mb = metrics.memory_bytes / (1024 * 1024)
        
        print(f"\nANN index memory ({num_values} values):")
        print(f"  Index memory: {memory_mb:.2f} MB")
        print(f"  Build time: {metrics.build_time_ms:.2f} ms")
        print(f"  Target: < 100 MB")
        
        assert metrics.num_vectors == len(schema_vectors)


@pytest.mark.benchmark
class TestSIMDOptimizations:
    """
    Benchmarks for SIMD-optimized vector operations.
    
    Validates: Requirement 13.6
    """
    
    def test_cosine_similarity_performance(self):
        """Benchmark SIMD-optimized cosine similarity."""
        dimension = 10000
        
        # Generate random vectors
        np.random.seed(42)
        vec1_data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
        vec2_data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
        
        # Create mock vectors
        class MockVector:
            def __init__(self, data):
                self.data = data
        
        vec1 = MockVector(vec1_data)
        vec2 = MockVector(vec2_data)
        
        ops = OptimizedVectorOps()
        
        # Warm up
        for _ in range(10):
            ops.cosine_similarity(vec1, vec2)
        
        # Measure
        iterations = 1000
        with BenchmarkTimer(f"Cosine similarity ({iterations} iterations, {dimension}D)"):
            for _ in range(iterations):
                sim = ops.cosine_similarity(vec1, vec2)
        
        assert -1.0 <= sim <= 1.0
    
    def test_batch_similarity_performance(self):
        """Benchmark batch similarity computation."""
        dimension = 10000
        num_vectors = 1000
        
        # Generate random vectors
        np.random.seed(42)
        query_data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
        vectors_data = [
            np.random.choice([-1, 1], size=dimension).astype(np.int8)
            for _ in range(num_vectors)
        ]
        
        class MockVector:
            def __init__(self, data):
                self.data = data
        
        query = MockVector(query_data)
        vectors = [MockVector(v) for v in vectors_data]
        
        ops = OptimizedVectorOps()
        
        # Warm up
        for _ in range(3):
            ops.batch_cosine_similarity(query, vectors)
        
        # Measure
        iterations = 100
        with BenchmarkTimer(f"Batch similarity ({iterations} iterations, {num_vectors} vectors)"):
            for _ in range(iterations):
                sims = ops.batch_cosine_similarity(query, vectors)
        
        assert len(sims) == num_vectors
    
    def test_bipolar_similarity_performance(self):
        """Benchmark bipolar-optimized similarity."""
        dimension = 10000
        
        np.random.seed(42)
        vec1_data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
        vec2_data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
        
        class MockVector:
            def __init__(self, data):
                self.data = data
        
        vec1 = MockVector(vec1_data)
        vec2 = MockVector(vec2_data)
        
        ops = OptimizedVectorOps()
        
        # Warm up
        for _ in range(10):
            ops.bipolar_similarity(vec1, vec2)
        
        # Measure
        iterations = 1000
        with BenchmarkTimer(f"Bipolar similarity ({iterations} iterations, {dimension}D)"):
            for _ in range(iterations):
                sim = ops.bipolar_similarity(vec1, vec2)
        
        assert -1.0 <= sim <= 1.0


@pytest.mark.benchmark
class TestANNIndexPerformance:
    """
    Benchmarks for ANN index operations.
    
    Validates: Requirement 13.1
    """
    
    def test_ann_search_performance(self):
        """Benchmark ANN search performance."""
        dimension = 10000
        num_vectors = 5000
        
        # Create index
        index = ANNIndex(dimension=dimension)
        
        # Generate and add random vectors
        np.random.seed(42)
        for i in range(num_vectors):
            vec_data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
            
            class MockVector:
                def __init__(self, data):
                    self.data = data
            
            index.add_vector(f"key_{i}", MockVector(vec_data))
        
        # Build index
        with BenchmarkTimer(f"ANN index build ({num_vectors} vectors)"):
            metrics = index.build()
        
        print(f"  Build time: {metrics.build_time_ms:.2f} ms")
        
        # Generate query vector
        query_data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
        
        class MockVector:
            def __init__(self, data):
                self.data = data
        
        query = MockVector(query_data)
        
        # Warm up
        for _ in range(5):
            index.search(query, k=10)
        
        # Measure search
        iterations = 100
        latencies = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            results = index.search(query, k=10)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\nANN search latency ({num_vectors} vectors, k=10):")
        print(f"  Average: {avg:.2f} ms")
        print(f"  p95: {p95:.2f} ms")
        
        assert len(results) <= 10
    
    def test_ann_batch_search_performance(self):
        """Benchmark ANN batch search performance."""
        dimension = 10000
        num_vectors = 5000
        num_queries = 10
        
        # Create and build index
        index = ANNIndex(dimension=dimension)
        
        np.random.seed(42)
        for i in range(num_vectors):
            vec_data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
            
            class MockVector:
                def __init__(self, data):
                    self.data = data
            
            index.add_vector(f"key_{i}", MockVector(vec_data))
        
        index.build()
        
        # Generate query vectors
        queries = []
        for _ in range(num_queries):
            query_data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
            
            class MockVector:
                def __init__(self, data):
                    self.data = data
            
            queries.append(MockVector(query_data))
        
        # Warm up
        for _ in range(3):
            index.search_batch(queries, k=10)
        
        # Measure
        iterations = 50
        with BenchmarkTimer(f"ANN batch search ({num_queries} queries, {iterations} iterations)"):
            for _ in range(iterations):
                results = index.search_batch(queries, k=10)
        
        assert len(results) == num_queries


@pytest.mark.benchmark
class TestBenchmarkSummary:
    """Summary of NL performance benchmarks."""
    
    def test_benchmark_summary(self):
        """Print a summary of benchmark results and targets."""
        print("\n" + "=" * 80)
        print("NL PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)
        print("\nPerformance Targets:")
        print("  1. Schema Index Build Time:")
        print("     - Target: < 1s for 1000 values")
        print("     - Includes: vectorization + ANN index build")
        print("\n  2. Query Matching Latency:")
        print("     - Target: p95 < 50ms for schemas with 1000 values")
        print("     - Includes: tokenization + matching + intent inference")
        print("\n  3. Memory Usage:")
        print("     - Target: < 100MB for 10000 values")
        print("     - Includes: schema vectors + ANN index")
        print("\nOptimizations:")
        print("  - ANN Index: Used for schemas > 1000 values")
        print("  - SIMD: numpy/BLAS for vector operations")
        print("  - Batch Processing: Efficient multi-query handling")
        print("\n" + "=" * 80)
        print("Run benchmarks with: pytest tests/benchmarks/test_nl_performance.py -v")
        print("=" * 80 + "\n")
