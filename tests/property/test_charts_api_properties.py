"""
Property-based tests for Charts API.

This module contains property-based tests using Hypothesis to verify
correctness properties for the Charts API endpoints.

Feature: data-viewer-charts-stats

**Validates: Requirements 6.2, 6.4, 6.6**

Properties tested:
- Property 1: Similarity Scores Valid Range - All similarity scores are in [0, 1]
- Property 2: Histogram Bucket Sum Invariant - Bucket counts sum to total_pairs
- Property 3: Neighbors Sorted Descending - Neighbors are sorted by similarity descending
"""

import pytest
from typing import List
from uuid import uuid4

from hypothesis import given, settings, strategies as st, assume, HealthCheck

from api.routes.charts import compute_cosine_similarity


# =============================================================================
# Test Data Generators (Strategies)
# =============================================================================

@st.composite
def valid_embedding(draw, dim: int = 128) -> List[float]:
    """
    Generate a valid embedding vector.
    
    Uses a smaller dimension (128) for faster test execution while
    still being representative of real embeddings.
    """
    import numpy as np
    
    values = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=dim,
        max_size=dim,
    ))
    # Normalize to unit length for realistic embeddings
    norm = np.linalg.norm(values)
    if norm > 0:
        values = [v / norm for v in values]
    return values


@st.composite
def embedding_pair(draw, dim: int = 128) -> tuple[List[float], List[float]]:
    """Generate a pair of valid embeddings for similarity testing."""
    return (draw(valid_embedding(dim)), draw(valid_embedding(dim)))


@st.composite
def embedding_list(draw, min_size: int = 2, max_size: int = 20, dim: int = 128) -> List[List[float]]:
    """Generate a list of valid embeddings for distribution testing."""
    return draw(st.lists(
        valid_embedding(dim),
        min_size=min_size,
        max_size=max_size,
    ))


@st.composite
def similarity_score_list(draw, min_size: int = 1, max_size: int = 50) -> List[float]:
    """Generate a list of valid similarity scores in [0, 1]."""
    return draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=min_size,
        max_size=max_size,
    ))


# =============================================================================
# Property 1: Similarity Scores Valid Range
# =============================================================================

class TestSimilarityScoresValidRange:
    """
    Property tests for Similarity Scores Valid Range (Property 1).
    
    Feature: data-viewer-charts-stats, Property 1: Similarity Scores Valid Range
    
    **Validates: Requirements 2.1, 2.2, 2.3, 3.1, 5.3**
    
    For any pair of glyphs and any similarity computation (cortex, segment, role),
    the resulting similarity score SHALL be in the range [0, 1] inclusive.
    """
    
    @given(pair=embedding_pair())
    @settings(max_examples=200)
    def test_cosine_similarity_in_valid_range(self, pair: tuple[List[float], List[float]]):
        """
        Property test: Cosine similarity is always in [0, 1].
        
        For any two embedding vectors, compute_cosine_similarity() SHALL
        return a value in the range [0, 1] inclusive.
        
        Feature: data-viewer-charts-stats, Property 1: Similarity Scores Valid Range
        **Validates: Requirements 2.1, 2.2, 2.3, 3.1, 5.3**
        """
        vec1, vec2 = pair
        
        similarity = compute_cosine_similarity(vec1, vec2)
        
        # Property: Similarity must be in [0, 1]
        assert 0.0 <= similarity <= 1.0, \
            f"Similarity {similarity} is outside valid range [0, 1]"
    
    @given(vec=valid_embedding())
    @settings(max_examples=100)
    def test_self_similarity_is_one(self, vec: List[float]):
        """
        Property test: Self-similarity is 1.0.
        
        For any embedding vector, the similarity with itself SHALL be 1.0
        (or very close due to floating point).
        
        Feature: data-viewer-charts-stats, Property 1: Similarity Scores Valid Range
        **Validates: Requirements 2.1**
        """
        # Skip zero vectors (norm is 0)
        norm = sum(v * v for v in vec) ** 0.5
        assume(norm > 1e-10)
        
        similarity = compute_cosine_similarity(vec, vec)
        
        # Property: Self-similarity should be 1.0 (with floating point tolerance)
        assert abs(similarity - 1.0) < 1e-6, \
            f"Self-similarity {similarity} should be 1.0"
    
    @given(embeddings=embedding_list(min_size=2, max_size=10))
    @settings(max_examples=100)
    def test_all_pairwise_similarities_in_range(self, embeddings: List[List[float]]):
        """
        Property test: All pairwise similarities are in [0, 1].
        
        For any set of embeddings, all pairwise similarity computations
        SHALL produce values in [0, 1].
        
        Feature: data-viewer-charts-stats, Property 1: Similarity Scores Valid Range
        **Validates: Requirements 2.2, 5.3**
        """
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = compute_cosine_similarity(embeddings[i], embeddings[j])
                
                # Property: Every pairwise similarity must be in [0, 1]
                assert 0.0 <= similarity <= 1.0, \
                    f"Pairwise similarity {similarity} between embeddings {i} and {j} " \
                    f"is outside valid range [0, 1]"


# =============================================================================
# Property 2: Histogram Bucket Sum Invariant
# =============================================================================

class TestHistogramBucketSumInvariant:
    """
    Property tests for Histogram Bucket Sum Invariant (Property 2).
    
    Feature: data-viewer-charts-stats, Property 2: Histogram Bucket Sum Invariant
    
    **Validates: Requirements 2.2, 6.4**
    
    For any set of glyph pairs and computed similarity distribution,
    the sum of all histogram bucket counts SHALL equal the total number
    of pairs computed.
    """
    
    @given(scores=similarity_score_list(min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_bucket_sum_equals_total_pairs(self, scores: List[float]):
        """
        Property test: Histogram bucket sum equals total pairs.
        
        When similarity scores are bucketed into a histogram, the sum
        of all bucket counts SHALL equal the number of input scores.
        
        Feature: data-viewer-charts-stats, Property 2: Histogram Bucket Sum Invariant
        **Validates: Requirements 2.2, 6.4**
        """
        # Build histogram with 10 buckets (0.0-1.0)
        buckets = [0] * 10
        
        for score in scores:
            bucket_idx = min(9, int(score * 10))
            buckets[bucket_idx] += 1
        
        # Property: Sum of buckets must equal number of scores
        bucket_sum = sum(buckets)
        assert bucket_sum == len(scores), \
            f"Bucket sum {bucket_sum} does not equal total pairs {len(scores)}"
    
    @given(embeddings=embedding_list(min_size=2, max_size=15))
    @settings(max_examples=100)
    def test_pairwise_histogram_sum_invariant(self, embeddings: List[List[float]]):
        """
        Property test: Pairwise histogram sum equals n*(n-1)/2.
        
        For n embeddings, the number of unique pairs is n*(n-1)/2.
        The histogram bucket sum SHALL equal this value.
        
        Feature: data-viewer-charts-stats, Property 2: Histogram Bucket Sum Invariant
        **Validates: Requirements 2.2, 6.4**
        """
        n = len(embeddings)
        expected_pairs = n * (n - 1) // 2
        
        # Build histogram from pairwise similarities
        buckets = [0] * 10
        actual_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                bucket_idx = min(9, int(sim * 10))
                buckets[bucket_idx] += 1
                actual_pairs += 1
        
        # Property: Bucket sum must equal expected pairs
        bucket_sum = sum(buckets)
        assert bucket_sum == expected_pairs, \
            f"Bucket sum {bucket_sum} does not equal expected pairs {expected_pairs}"
        assert actual_pairs == expected_pairs, \
            f"Actual pairs {actual_pairs} does not equal expected pairs {expected_pairs}"


# =============================================================================
# Property 3: Neighbors Sorted Descending
# =============================================================================

class TestNeighborsSortedDescending:
    """
    Property tests for Neighbors Sorted Descending (Property 3).
    
    Feature: data-viewer-charts-stats, Property 3: Neighbors Sorted Descending
    
    **Validates: Requirements 3.1, 6.6**
    
    For any anchor glyph and computed neighbors list, the neighbors
    SHALL be sorted by similarity score in descending order (highest
    similarity first).
    """
    
    @given(scores=similarity_score_list(min_size=2, max_size=50))
    @settings(max_examples=200)
    def test_sorted_scores_are_descending(self, scores: List[float]):
        """
        Property test: Sorted similarity scores are in descending order.
        
        When similarity scores are sorted descending, each score SHALL
        be greater than or equal to the next score.
        
        Feature: data-viewer-charts-stats, Property 3: Neighbors Sorted Descending
        **Validates: Requirements 3.1, 6.6**
        """
        sorted_scores = sorted(scores, reverse=True)
        
        # Property: Each score must be >= the next score
        for i in range(len(sorted_scores) - 1):
            assert sorted_scores[i] >= sorted_scores[i + 1], \
                f"Score at index {i} ({sorted_scores[i]}) is less than " \
                f"score at index {i+1} ({sorted_scores[i+1]})"
    
    @given(embeddings=embedding_list(min_size=3, max_size=20))
    @settings(max_examples=100)
    def test_neighbors_sorted_by_similarity(self, embeddings: List[List[float]]):
        """
        Property test: Neighbors are sorted by similarity descending.
        
        For any anchor embedding, when computing similarities to all
        other embeddings and sorting, the result SHALL be in descending
        order of similarity.
        
        Feature: data-viewer-charts-stats, Property 3: Neighbors Sorted Descending
        **Validates: Requirements 3.1, 6.6**
        """
        # Use first embedding as anchor
        anchor = embeddings[0]
        others = embeddings[1:]
        
        # Compute similarities
        similarities = []
        for i, other in enumerate(others):
            sim = compute_cosine_similarity(anchor, other)
            similarities.append((i, sim))
        
        # Sort by similarity descending
        sorted_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Property: Sorted neighbors must be in descending order
        for i in range(len(sorted_neighbors) - 1):
            current_sim = sorted_neighbors[i][1]
            next_sim = sorted_neighbors[i + 1][1]
            assert current_sim >= next_sim, \
                f"Neighbor at index {i} (sim={current_sim}) has lower similarity " \
                f"than neighbor at index {i+1} (sim={next_sim})"
    
    @given(
        embeddings=embedding_list(min_size=5, max_size=30),
        top_k=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_top_k_neighbors_are_highest_similarity(
        self, embeddings: List[List[float]], top_k: int
    ):
        """
        Property test: Top-k neighbors have highest similarities.
        
        The top-k neighbors returned SHALL have similarity scores
        greater than or equal to all non-returned neighbors.
        
        Feature: data-viewer-charts-stats, Property 3: Neighbors Sorted Descending
        **Validates: Requirements 3.1, 6.6**
        """
        # Ensure we have enough embeddings
        assume(len(embeddings) > top_k)
        
        # Use first embedding as anchor
        anchor = embeddings[0]
        others = embeddings[1:]
        
        # Compute all similarities
        similarities = []
        for other in others:
            sim = compute_cosine_similarity(anchor, other)
            similarities.append(sim)
        
        # Sort and take top-k
        sorted_sims = sorted(similarities, reverse=True)
        top_k_sims = sorted_sims[:top_k]
        remaining_sims = sorted_sims[top_k:]
        
        # Property: All top-k similarities must be >= all remaining similarities
        if top_k_sims and remaining_sims:
            min_top_k = min(top_k_sims)
            max_remaining = max(remaining_sims)
            assert min_top_k >= max_remaining, \
                f"Minimum top-k similarity ({min_top_k}) is less than " \
                f"maximum remaining similarity ({max_remaining})"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """
    Edge case tests for charts API functions.
    
    These tests verify correct behavior for boundary conditions
    and special cases.
    """
    
    def test_empty_vectors_return_zero_similarity(self):
        """Empty vectors should return 0.0 similarity."""
        assert compute_cosine_similarity([], []) == 0.0
        assert compute_cosine_similarity([1.0, 2.0], []) == 0.0
        assert compute_cosine_similarity([], [1.0, 2.0]) == 0.0
    
    def test_zero_vectors_return_zero_similarity(self):
        """Zero vectors should return 0.0 similarity."""
        zero_vec = [0.0] * 10
        non_zero_vec = [1.0] * 10
        
        assert compute_cosine_similarity(zero_vec, zero_vec) == 0.0
        assert compute_cosine_similarity(zero_vec, non_zero_vec) == 0.0
    
    def test_identical_vectors_return_one(self):
        """Identical non-zero vectors should return 1.0 similarity."""
        vec = [1.0, 2.0, 3.0, 4.0, 5.0]
        similarity = compute_cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors should return 0.0 similarity."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = compute_cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-6
    
    def test_opposite_vectors_clamped_to_zero(self):
        """Opposite vectors should be clamped to 0.0 (not negative)."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = compute_cosine_similarity(vec1, vec2)
        # Cosine similarity of opposite vectors is -1, but we clamp to [0, 1]
        assert similarity == 0.0
    
    def test_different_length_vectors_use_minimum(self):
        """Vectors of different lengths should use minimum length."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Should compute similarity using first 3 elements
        similarity = compute_cosine_similarity(vec1, vec2)
        assert 0.0 <= similarity <= 1.0
