"""
Unit tests for HDC vector operations.

Tests verify the core mathematical operations used throughout the
Glyphh engine: bind, bundle, cosine similarity, hamming similarity,
and deterministic symbol generation.
"""

import numpy as np
import pytest
from glyphh.core.ops import (
    bind,
    bundle,
    cosine_similarity,
    hamming_similarity,
    generate_symbol,
)


class TestBindOperation:
    """Tests for bind operation"""

    def test_bind_basic(self):
        r = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        v = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        result = bind(r, v)
        expected = np.array([-1, -1, -1, -1, -1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_bind_inverse_property(self):
        r = np.array([1, -1, 1, -1, 1, 1, -1, 1], dtype=np.int8)
        v = np.array([-1, 1, -1, 1, -1, 1, 1, -1], dtype=np.int8)
        bound = bind(r, v)
        recovered = bind(bound, r)
        np.testing.assert_array_equal(recovered, v)

    def test_bind_commutative(self):
        r = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        v = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        np.testing.assert_array_equal(bind(r, v), bind(v, r))

    def test_bind_self_inverse(self):
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        result = bind(v, v)
        expected = np.array([1, 1, 1, 1, 1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_bind_dimension_mismatch(self):
        r = np.array([1, -1, 1], dtype=np.int8)
        v = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            bind(r, v)


class TestBundleOperation:
    """Tests for bundle operation"""

    def test_bundle_basic(self):
        vectors = [
            np.array([1, -1, 1, -1, 1], dtype=np.int8),
            np.array([-1, 1, -1, 1, -1], dtype=np.int8),
            np.array([1, 1, -1, -1, 1], dtype=np.int8),
        ]
        result = bundle(vectors)
        expected = np.array([1, 1, -1, -1, 1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_bundle_commutativity(self):
        a = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        b = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        c = np.array([1, 1, -1, -1, 1], dtype=np.int8)
        np.testing.assert_array_equal(bundle([a, b, c]), bundle([c, a, b]))
        np.testing.assert_array_equal(bundle([a, b, c]), bundle([b, c, a]))

    def test_bundle_single_vector(self):
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        np.testing.assert_array_equal(bundle([v]), v)

    def test_bundle_identical_vectors(self):
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        np.testing.assert_array_equal(bundle([v, v, v]), v)

    def test_bundle_empty_list(self):
        with pytest.raises(ValueError, match="Cannot bundle empty vector list"):
            bundle([])

    def test_bundle_dimension_mismatch(self):
        vectors = [
            np.array([1, -1, 1], dtype=np.int8),
            np.array([-1, 1, -1, 1, -1], dtype=np.int8),
        ]
        with pytest.raises(ValueError, match="Dimension mismatch"):
            bundle(vectors)


class TestCosineSimilarity:
    """Tests for cosine similarity"""

    def test_cosine_identical_vectors(self):
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_opposite_vectors(self):
        v1 = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        v2 = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        assert abs(cosine_similarity(v1, v2) - (-1.0)) < 1e-6

    def test_cosine_orthogonal_vectors(self):
        v1 = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int8)
        v2 = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int8)
        assert abs(cosine_similarity(v1, v2) - 0.0) < 1e-6

    def test_cosine_dimension_mismatch(self):
        v1 = np.array([1, -1, 1], dtype=np.int8)
        v2 = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cosine_similarity(v1, v2)


class TestHammingSimilarity:
    """Tests for Hamming similarity"""

    def test_hamming_identical_vectors(self):
        v = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        assert abs(hamming_similarity(v, v) - 1.0) < 1e-6

    def test_hamming_opposite_vectors(self):
        v1 = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        v2 = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        assert abs(hamming_similarity(v1, v2) - 0.0) < 1e-6

    def test_hamming_orthogonal_vectors(self):
        v1 = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int8)
        v2 = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int8)
        assert abs(hamming_similarity(v1, v2) - 0.5) < 1e-6

    def test_hamming_cosine_relationship(self):
        v1 = np.array([1, -1, 1, -1, 1, 1, -1, 1], dtype=np.int8)
        v2 = np.array([-1, 1, -1, 1, -1, 1, 1, -1], dtype=np.int8)
        cosine = cosine_similarity(v1, v2)
        hamming = hamming_similarity(v1, v2)
        assert abs(hamming - (cosine + 1) / 2) < 1e-6

    def test_hamming_dimension_mismatch(self):
        v1 = np.array([1, -1, 1], dtype=np.int8)
        v2 = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            hamming_similarity(v1, v2)


class TestGenerateSymbol:
    """Tests for symbol generation"""

    def test_generate_symbol_deterministic(self):
        vec1 = generate_symbol(42, "test_key", 100)
        vec2 = generate_symbol(42, "test_key", 100)
        np.testing.assert_array_equal(vec1, vec2)

    def test_generate_symbol_different_keys(self):
        vec1 = generate_symbol(42, "key1", 100)
        vec2 = generate_symbol(42, "key2", 100)
        assert not np.array_equal(vec1, vec2)

    def test_generate_symbol_different_seeds(self):
        vec1 = generate_symbol(42, "test_key", 100)
        vec2 = generate_symbol(43, "test_key", 100)
        assert not np.array_equal(vec1, vec2)

    def test_generate_symbol_bipolar(self):
        vec = generate_symbol(42, "test", 100)
        assert np.all(np.isin(vec, [-1, 1]))

    def test_generate_symbol_correct_dimension(self):
        vec = generate_symbol(42, "test", 1000)
        assert len(vec) == 1000


class TestLargeVectors:
    """Tests with large vectors"""

    def test_bind_large_vectors(self):
        dim = 10000
        r = generate_symbol(42, "role", dim)
        v = generate_symbol(42, "value", dim)
        result = bind(r, v)
        assert len(result) == dim
        assert np.all(np.isin(result, [-1, 1]))

    def test_bundle_many_large_vectors(self):
        dim = 10000
        vectors = [generate_symbol(42, f"vec_{i}", dim) for i in range(10)]
        result = bundle(vectors)
        assert len(result) == dim
        assert np.all(np.isin(result, [-1, 1]))

    def test_similarity_large_vectors(self):
        dim = 10000
        v1 = generate_symbol(42, "vec1", dim)
        v2 = generate_symbol(42, "vec2", dim)
        cosine = cosine_similarity(v1, v2)
        hamming = hamming_similarity(v1, v2)
        assert -1.0 <= cosine <= 1.0
        assert 0.0 <= hamming <= 1.0
        assert abs(hamming - (cosine + 1) / 2) < 1e-6
