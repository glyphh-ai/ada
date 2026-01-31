"""
Performance benchmarks for Glyphh Runtime.

Run with: pytest tests/performance/ -v --benchmark-only
"""

import asyncio
import time
from typing import List
from uuid import uuid4

import numpy as np
import pytest


class TestEmbeddingPerformance:
    """Benchmarks for embedding operations."""
    
    def test_embedding_generation_speed(self):
        """Benchmark embedding vector generation."""
        iterations = 1000
        
        start = time.perf_counter()
        for _ in range(iterations):
            embedding = np.random.randn(768).astype(float)
            embedding = embedding / np.linalg.norm(embedding)
        elapsed = time.perf_counter() - start
        
        ops_per_second = iterations / elapsed
        print(f"\nEmbedding generation: {ops_per_second:.0f} ops/sec")
        
        # Should be able to generate at least 10000 embeddings per second
        assert ops_per_second > 10000
    
    def test_cosine_similarity_speed(self):
        """Benchmark cosine similarity computation."""
        iterations = 10000
        
        # Pre-generate embeddings
        embeddings = [np.random.randn(768).astype(float) for _ in range(100)]
        query = np.random.randn(768).astype(float)
        
        start = time.perf_counter()
        for _ in range(iterations):
            for emb in embeddings:
                similarity = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb))
        elapsed = time.perf_counter() - start
        
        comparisons_per_second = (iterations * len(embeddings)) / elapsed
        print(f"\nCosine similarity: {comparisons_per_second:.0f} comparisons/sec")
        
        # Should be able to do at least 1M comparisons per second
        assert comparisons_per_second > 1000000


class TestSerializationPerformance:
    """Benchmarks for serialization operations."""
    
    def test_json_serialization_speed(self):
        """Benchmark JSON serialization."""
        import json
        
        iterations = 10000
        data = {
            "id": str(uuid4()),
            "namespace": "test_namespace",
            "concept_text": "This is a test concept with some text",
            "metadata": {"key1": "value1", "key2": 123, "key3": [1, 2, 3]},
            "embedding": list(np.random.randn(768).astype(float)),
        }
        
        start = time.perf_counter()
        for _ in range(iterations):
            json.dumps(data)
        elapsed = time.perf_counter() - start
        
        ops_per_second = iterations / elapsed
        print(f"\nJSON serialization: {ops_per_second:.0f} ops/sec")
        
        # Should be able to serialize at least 1000 glyphs per second
        assert ops_per_second > 1000
    
    def test_json_deserialization_speed(self):
        """Benchmark JSON deserialization."""
        import json
        
        iterations = 10000
        data = {
            "id": str(uuid4()),
            "namespace": "test_namespace",
            "concept_text": "This is a test concept with some text",
            "metadata": {"key1": "value1", "key2": 123, "key3": [1, 2, 3]},
            "embedding": list(np.random.randn(768).astype(float)),
        }
        json_str = json.dumps(data)
        
        start = time.perf_counter()
        for _ in range(iterations):
            json.loads(json_str)
        elapsed = time.perf_counter() - start
        
        ops_per_second = iterations / elapsed
        print(f"\nJSON deserialization: {ops_per_second:.0f} ops/sec")
        
        # Should be able to deserialize at least 1000 glyphs per second
        assert ops_per_second > 1000


class TestConcurrencyPerformance:
    """Benchmarks for concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_throughput(self):
        """Benchmark concurrent async task execution."""
        iterations = 1000
        
        async def dummy_task():
            await asyncio.sleep(0.001)
            return True
        
        start = time.perf_counter()
        tasks = [dummy_task() for _ in range(iterations)]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        
        tasks_per_second = iterations / elapsed
        print(f"\nConcurrent tasks: {tasks_per_second:.0f} tasks/sec")
        
        # Should be able to handle at least 100 concurrent tasks per second
        assert tasks_per_second > 100


class TestMemoryPerformance:
    """Benchmarks for memory usage."""
    
    def test_embedding_memory_usage(self):
        """Measure memory usage for embeddings."""
        import sys
        
        # Single embedding
        embedding = np.random.randn(768).astype(np.float32)
        single_size = sys.getsizeof(embedding) + embedding.nbytes
        
        # 1000 embeddings
        embeddings = [np.random.randn(768).astype(np.float32) for _ in range(1000)]
        total_size = sum(sys.getsizeof(e) + e.nbytes for e in embeddings)
        
        print(f"\nSingle embedding: {single_size / 1024:.2f} KB")
        print(f"1000 embeddings: {total_size / (1024 * 1024):.2f} MB")
        
        # 1000 embeddings should use less than 10MB
        assert total_size < 10 * 1024 * 1024
    
    def test_glyph_data_memory_usage(self):
        """Measure memory usage for glyph data structures."""
        import sys
        
        glyph = {
            "id": str(uuid4()),
            "namespace": "test_namespace",
            "concept_text": "This is a test concept",
            "metadata": {"key": "value"},
            "embedding": list(np.random.randn(768).astype(float)),
        }
        
        # Approximate size (Python dicts are complex)
        size = sys.getsizeof(glyph)
        for key, value in glyph.items():
            size += sys.getsizeof(key) + sys.getsizeof(value)
            if isinstance(value, list):
                size += sum(sys.getsizeof(v) for v in value)
        
        print(f"\nSingle glyph dict: {size / 1024:.2f} KB")
        
        # Single glyph should use less than 100KB
        assert size < 100 * 1024
