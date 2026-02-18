"""
Unit tests for Rust performance operations.

These tests verify both the Rust implementation (if available) and the
Python fallback implementation.
"""

import numpy as np
import pytest
from glyphh.core.rust_ops import (
    bind,
    bundle,
    cosine_similarity,
    hamming_similarity,
    generate_symbol,
    is_rust_available,
    get_backend_info,
)


class TestBindOperation:
    """Tests for bind operation"""
    
    def test_bind_basic(self):
        """Test basic bind operation"""
        r = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        v = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        
        result = bind(r, v)
        expected = np.array([-1, -1, -1, -1, -1], dtype=np.int8)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_bind_inverse_property(self):
        """Test that bind(bind(r, v), r) = v"""
        r = np.array([1, -1, 1, -1, 1, 1, -1, 1], dtype=np.int8)
        v = np.array([-1, 1, -1, 1, -1, 1, 1, -1], dtype=np.int8)
        
        # First bind
        bound = bind(r, v)
        
        # Second bind should recover v
        recovered = bind(bound, r)
        
        np.testing.assert_array_equal(recovered, v)
    
    def test_bind_commutative(self):
        """Test that bind(r, v) = bind(v, r)"""
        r = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        v = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        
        result1 = bind(r, v)
        result2 = bind(v, r)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_bind_self_inverse(self):
        """Test that bind(v, v) = identity (all 1s)"""
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        
        result = bind(v, v)
        expected = np.array([1, 1, 1, 1, 1], dtype=np.int8)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_bind_dimension_mismatch(self):
        """Test that bind raises error on dimension mismatch"""
        r = np.array([1, -1, 1], dtype=np.int8)
        v = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            bind(r, v)
    
    def test_bind_python_fallback(self):
        """Test Python fallback implementation"""
        r = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        v = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        
        result = bind(r, v, use_rust=False)
        expected = np.array([-1, -1, -1, -1, -1], dtype=np.int8)
        
        np.testing.assert_array_equal(result, expected)


class TestBundleOperation:
    """Tests for bundle operation"""
    
    def test_bundle_basic(self):
        """Test basic bundle operation"""
        vectors = [
            np.array([1, -1, 1, -1, 1], dtype=np.int8),
            np.array([-1, 1, -1, 1, -1], dtype=np.int8),
            np.array([1, 1, -1, -1, 1], dtype=np.int8),
        ]
        
        result = bundle(vectors)
        # Sums: [1, 1, -1, -1, 1] -> all >= 0 except positions 2 and 3
        expected = np.array([1, 1, -1, -1, 1], dtype=np.int8)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_bundle_commutativity(self):
        """Test that bundle([a, b, c]) = bundle([c, a, b])"""
        a = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        b = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        c = np.array([1, 1, -1, -1, 1], dtype=np.int8)
        
        result1 = bundle([a, b, c])
        result2 = bundle([c, a, b])
        result3 = bundle([b, c, a])
        
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result1, result3)
    
    def test_bundle_single_vector(self):
        """Test bundling a single vector returns the vector"""
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        
        result = bundle([v])
        
        np.testing.assert_array_equal(result, v)
    
    def test_bundle_identical_vectors(self):
        """Test bundling identical vectors returns the vector"""
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        
        result = bundle([v, v, v])
        
        np.testing.assert_array_equal(result, v)
    
    def test_bundle_empty_list(self):
        """Test that bundle raises error on empty list"""
        with pytest.raises(ValueError, match="Cannot bundle empty vector list"):
            bundle([])
    
    def test_bundle_dimension_mismatch(self):
        """Test that bundle raises error on dimension mismatch"""
        vectors = [
            np.array([1, -1, 1], dtype=np.int8),
            np.array([-1, 1, -1, 1, -1], dtype=np.int8),
        ]
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            bundle(vectors)
    
    def test_bundle_python_fallback(self):
        """Test Python fallback implementation"""
        vectors = [
            np.array([1, -1, 1, -1, 1], dtype=np.int8),
            np.array([-1, 1, -1, 1, -1], dtype=np.int8),
            np.array([1, 1, -1, -1, 1], dtype=np.int8),
        ]
        
        result = bundle(vectors, use_rust=False)
        expected = np.array([1, 1, -1, -1, 1], dtype=np.int8)
        
        np.testing.assert_array_equal(result, expected)


class TestCosineSimilarity:
    """Tests for cosine similarity"""
    
    def test_cosine_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0"""
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        
        result = cosine_similarity(v, v)
        
        assert abs(result - 1.0) < 1e-6
    
    def test_cosine_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1.0"""
        v1 = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        v2 = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        
        result = cosine_similarity(v1, v2)
        
        assert abs(result - (-1.0)) < 1e-6
    
    def test_cosine_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is ~0.0"""
        # For bipolar vectors, orthogonal means 50% agreement
        v1 = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int8)
        v2 = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int8)
        
        result = cosine_similarity(v1, v2)
        
        # 4 agree, 4 disagree -> 0.0
        assert abs(result - 0.0) < 1e-6
    
    def test_cosine_dimension_mismatch(self):
        """Test that cosine_similarity raises error on dimension mismatch"""
        v1 = np.array([1, -1, 1], dtype=np.int8)
        v2 = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cosine_similarity(v1, v2)
    
    def test_cosine_python_fallback(self):
        """Test Python fallback implementation"""
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        
        result = cosine_similarity(v, v, use_rust=False)
        
        assert abs(result - 1.0) < 1e-6


class TestHammingSimilarity:
    """Tests for Hamming similarity"""
    
    def test_hamming_identical_vectors(self):
        """Test Hamming similarity of identical vectors is 1.0"""
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        
        result = hamming_similarity(v, v)
        
        assert abs(result - 1.0) < 1e-6
    
    def test_hamming_opposite_vectors(self):
        """Test Hamming similarity of opposite vectors is 0.0"""
        v1 = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        v2 = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        
        result = hamming_similarity(v1, v2)
        
        assert abs(result - 0.0) < 1e-6
    
    def test_hamming_orthogonal_vectors(self):
        """Test Hamming similarity of orthogonal vectors is 0.5"""
        v1 = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int8)
        v2 = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int8)
        
        result = hamming_similarity(v1, v2)
        
        # 4 agree, 4 disagree -> 0.5
        assert abs(result - 0.5) < 1e-6
    
    def test_hamming_cosine_relationship(self):
        """Test that hamming = (cosine + 1) / 2"""
        v1 = np.array([1, -1, 1, -1, 1, 1, -1, 1], dtype=np.int8)
        v2 = np.array([-1, 1, -1, 1, -1, 1, 1, -1], dtype=np.int8)
        
        cosine = cosine_similarity(v1, v2)
        hamming = hamming_similarity(v1, v2)
        
        expected_hamming = (cosine + 1) / 2
        
        assert abs(hamming - expected_hamming) < 1e-6
    
    def test_hamming_dimension_mismatch(self):
        """Test that hamming_similarity raises error on dimension mismatch"""
        v1 = np.array([1, -1, 1], dtype=np.int8)
        v2 = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            hamming_similarity(v1, v2)
    
    def test_hamming_python_fallback(self):
        """Test Python fallback implementation"""
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        
        result = hamming_similarity(v, v, use_rust=False)
        
        assert abs(result - 1.0) < 1e-6


class TestGenerateSymbol:
    """Tests for symbol generation"""
    
    def test_generate_symbol_deterministic(self):
        """Test that same seed/key produces same vector"""
        seed = 42
        key = "test_key"
        dimension = 100
        
        vec1 = generate_symbol(seed, key, dimension)
        vec2 = generate_symbol(seed, key, dimension)
        
        np.testing.assert_array_equal(vec1, vec2)
    
    def test_generate_symbol_different_keys(self):
        """Test that different keys produce different vectors"""
        seed = 42
        dimension = 100
        
        vec1 = generate_symbol(seed, "key1", dimension)
        vec2 = generate_symbol(seed, "key2", dimension)
        
        # Vectors should be different
        assert not np.array_equal(vec1, vec2)
    
    def test_generate_symbol_different_seeds(self):
        """Test that different seeds produce different vectors"""
        key = "test_key"
        dimension = 100
        
        vec1 = generate_symbol(42, key, dimension)
        vec2 = generate_symbol(43, key, dimension)
        
        # Vectors should be different
        assert not np.array_equal(vec1, vec2)
    
    def test_generate_symbol_bipolar(self):
        """Test that generated vectors are bipolar {-1, +1}"""
        vec = generate_symbol(42, "test", 100)
        
        # All values should be -1 or 1
        assert np.all(np.isin(vec, [-1, 1]))
    
    def test_generate_symbol_correct_dimension(self):
        """Test that generated vectors have correct dimension"""
        dimension = 1000
        vec = generate_symbol(42, "test", dimension)
        
        assert len(vec) == dimension
    
    def test_generate_symbol_python_fallback(self):
        """Test Python fallback implementation"""
        seed = 42
        key = "test_key"
        dimension = 100
        
        vec1 = generate_symbol(seed, key, dimension, use_rust=False)
        vec2 = generate_symbol(seed, key, dimension, use_rust=False)
        
        # Should be deterministic
        np.testing.assert_array_equal(vec1, vec2)
        
        # Should be bipolar
        assert np.all(np.isin(vec1, [-1, 1]))


class TestBackendInfo:
    """Tests for backend information"""
    
    def test_is_rust_available(self):
        """Test that is_rust_available returns a boolean"""
        result = is_rust_available()
        assert isinstance(result, bool)
    
    def test_get_backend_info(self):
        """Test that get_backend_info returns expected structure"""
        info = get_backend_info()
        
        assert "rust_available" in info
        assert "backend" in info
        assert "operations" in info
        
        assert isinstance(info["rust_available"], bool)
        assert info["backend"] in ["rust", "python"]
        
        operations = info["operations"]
        assert "bind" in operations
        assert "bundle" in operations
        assert "cosine_similarity" in operations
        assert "hamming_similarity" in operations
        assert "generate_symbol" in operations


class TestLargeVectors:
    """Tests with large vectors (performance-critical)"""
    
    def test_bind_large_vectors(self):
        """Test bind with large vectors"""
        dimension = 10000
        r = generate_symbol(42, "role", dimension)
        v = generate_symbol(42, "value", dimension)
        
        result = bind(r, v)
        
        assert len(result) == dimension
        assert np.all(np.isin(result, [-1, 1]))
    
    def test_bundle_many_large_vectors(self):
        """Test bundle with many large vectors"""
        dimension = 10000
        num_vectors = 10
        
        vectors = [
            generate_symbol(42, f"vec_{i}", dimension)
            for i in range(num_vectors)
        ]
        
        result = bundle(vectors)
        
        assert len(result) == dimension
        assert np.all(np.isin(result, [-1, 1]))
    
    def test_similarity_large_vectors(self):
        """Test similarity with large vectors"""
        dimension = 10000
        v1 = generate_symbol(42, "vec1", dimension)
        v2 = generate_symbol(42, "vec2", dimension)
        
        cosine = cosine_similarity(v1, v2)
        hamming = hamming_similarity(v1, v2)
        
        assert -1.0 <= cosine <= 1.0
        assert 0.0 <= hamming <= 1.0
        
        # Verify relationship
        expected_hamming = (cosine + 1) / 2
        assert abs(hamming - expected_hamming) < 1e-6
