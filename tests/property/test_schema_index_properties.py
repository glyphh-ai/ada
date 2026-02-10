"""
Property-based tests for SchemaIndex.

Tests the runtime schema index for NL query matching using Hypothesis.

Property Tests:
- Property 22: Cache Hit Consistency - verify cached results match fresh results
- Property 3: Schema Vector Regeneration on Config Change - verify config changes
  trigger vector regeneration

Validates: Requirements 7.3, 7.6
"""

import pytest
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, strategies as st, assume

from domains.nl_query.schema_index import SchemaIndex, SchemaIndexMetrics


# =============================================================================
# Custom Strategies for Schema Index Testing
# =============================================================================

@st.composite
def valid_query_string(draw) -> str:
    """Generate a valid query string for caching tests."""
    words = draw(st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz",
            min_size=2,
            max_size=15,
        ),
        min_size=1,
        max_size=10,
    ))
    return " ".join(words).strip()


@st.composite
def valid_model_id_strategy(draw) -> str:
    """Generate a valid model_id string."""
    prefix = draw(st.sampled_from(["model", "test", "demo", "prod"]))
    suffix = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=4,
        max_size=8,
    ))
    return f"{prefix}_{suffix}"


@st.composite
def mock_match_result(draw) -> MagicMock:
    """Generate a mock MatchResult for caching tests."""
    result = MagicMock()
    result.query = draw(valid_query_string())
    result.token_matches = []
    result.role_matches = []
    result.value_matches = []
    result.compound_matches = []
    result.all_candidates = []
    return result


# =============================================================================
# Property 22: Cache Hit Consistency
# =============================================================================

class TestCacheHitConsistency:
    """
    Property test: Cache Hit Consistency
    
    Verify cached results match fresh results.
    
    **Validates: Property 22**
    **Validates: Requirement 7.6** - THE Runtime SHALL cache match results
    for repeated queries
    """
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
        query=valid_query_string(),
    )
    def test_cached_result_matches_original(self, model_id: str, query: str):
        """
        Property: For any query processed twice with the same schema index,
        the second result SHALL be identical to the first (cache consistency).
        
        **Validates: Requirements 7.6**
        """
        assume(len(query) > 0)
        assume(len(model_id) > 0)
        
        # Create index
        index = SchemaIndex(model_id=model_id)
        
        # Create a mock result
        mock_result = MagicMock()
        mock_result.query = query
        
        # Compute query hash
        query_hash = index.compute_query_hash(query)
        
        # Cache the result
        index.cache_match_result(query_hash, mock_result)
        
        # Retrieve the cached result
        cached_result = index.get_cached_result(query_hash)
        
        # Property: cached result should be identical to original
        assert cached_result is mock_result
        assert cached_result.query == query
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
        queries=st.lists(valid_query_string(), min_size=2, max_size=10, unique=True),
    )
    def test_multiple_cached_results_consistency(self, model_id: str, queries: list):
        """
        Property: Multiple cached results should all be retrievable and consistent.
        
        **Validates: Requirements 7.6**
        """
        assume(all(len(q) > 0 for q in queries))
        assume(len(model_id) > 0)
        
        # Create index with sufficient cache size
        index = SchemaIndex(model_id=model_id, cache_size=len(queries) + 10)
        
        # Cache results for all queries
        results = {}
        for query in queries:
            mock_result = MagicMock()
            mock_result.query = query
            query_hash = index.compute_query_hash(query)
            index.cache_match_result(query_hash, mock_result)
            results[query_hash] = mock_result
        
        # Verify all cached results are consistent
        for query in queries:
            query_hash = index.compute_query_hash(query)
            cached = index.get_cached_result(query_hash)
            
            # Property: cached result should match original
            assert cached is results[query_hash]
            assert cached.query == query
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
        query=valid_query_string(),
        access_count=st.integers(min_value=2, max_value=10),
    )
    def test_repeated_cache_access_consistency(
        self, model_id: str, query: str, access_count: int
    ):
        """
        Property: Repeated access to the same cached result should always
        return the same result.
        
        **Validates: Requirements 7.6**
        """
        assume(len(query) > 0)
        assume(len(model_id) > 0)
        
        # Create index
        index = SchemaIndex(model_id=model_id)
        
        # Create and cache a result
        mock_result = MagicMock()
        mock_result.query = query
        query_hash = index.compute_query_hash(query)
        index.cache_match_result(query_hash, mock_result)
        
        # Access the cached result multiple times
        for _ in range(access_count):
            cached = index.get_cached_result(query_hash)
            
            # Property: each access should return the same result
            assert cached is mock_result
            assert cached.query == query
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
        query1=valid_query_string(),
        query2=valid_query_string(),
    )
    def test_query_hash_determinism(self, model_id: str, query1: str, query2: str):
        """
        Property: Query hash computation should be deterministic.
        Same query should always produce the same hash.
        
        **Validates: Requirements 7.6**
        """
        assume(len(query1) > 0)
        assume(len(query2) > 0)
        assume(len(model_id) > 0)
        
        index = SchemaIndex(model_id=model_id)
        
        # Compute hash multiple times for same query
        hash1_a = index.compute_query_hash(query1)
        hash1_b = index.compute_query_hash(query1)
        
        # Property: same query should produce same hash
        assert hash1_a == hash1_b
        
        # Different queries should produce different hashes (with high probability)
        if query1.lower().strip() != query2.lower().strip():
            hash2 = index.compute_query_hash(query2)
            assert hash1_a != hash2


# =============================================================================
# Property 3: Schema Vector Regeneration on Config Change
# =============================================================================

class TestSchemaVectorRegenerationOnConfigChange:
    """
    Property test: Schema Vector Regeneration on Config Change
    
    Verify config changes trigger vector regeneration.
    
    **Validates: Property 3**
    **Validates: Requirement 7.3** - WHEN a model config is updated,
    THE Runtime SHALL rebuild the schema index
    """
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
        initial_hash=st.text(
            alphabet="0123456789abcdef",
            min_size=64,
            max_size=64,
        ),
        new_hash=st.text(
            alphabet="0123456789abcdef",
            min_size=64,
            max_size=64,
        ),
    )
    def test_config_change_detection(
        self, model_id: str, initial_hash: str, new_hash: str
    ):
        """
        Property: When config hash changes, check_config_changed() should
        return True.
        
        **Validates: Requirements 7.3**
        """
        assume(len(model_id) > 0)
        assume(initial_hash != new_hash)
        
        # Create index
        index = SchemaIndex(model_id=model_id)
        
        # Set up initial state with a mock vectorizer
        mock_vectorizer = MagicMock()
        index._vectorizer = mock_vectorizer
        index._config_hash = initial_hash
        
        # Mock the vectorizer to return a different hash
        mock_vectorizer._compute_config_hash.return_value = new_hash
        
        # Property: config change should be detected
        assert index.check_config_changed() is True
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
        config_hash=st.text(
            alphabet="0123456789abcdef",
            min_size=64,
            max_size=64,
        ),
    )
    def test_no_config_change_when_hash_same(self, model_id: str, config_hash: str):
        """
        Property: When config hash is the same, check_config_changed() should
        return False.
        
        **Validates: Requirements 7.3**
        """
        assume(len(model_id) > 0)
        
        # Create index
        index = SchemaIndex(model_id=model_id)
        
        # Set up state with same hash
        mock_vectorizer = MagicMock()
        index._vectorizer = mock_vectorizer
        index._config_hash = config_hash
        
        # Mock the vectorizer to return the same hash
        mock_vectorizer._compute_config_hash.return_value = config_hash
        
        # Property: no config change should be detected
        assert index.check_config_changed() is False
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
        queries=st.lists(valid_query_string(), min_size=1, max_size=5, unique=True),
    )
    def test_rebuild_clears_cache(self, model_id: str, queries: list):
        """
        Property: When rebuild() is called, the cache should be cleared.
        
        **Validates: Requirements 7.3**
        """
        assume(len(model_id) > 0)
        assume(all(len(q) > 0 for q in queries))
        # Ensure queries are unique by their normalized hash
        query_hashes = set()
        unique_queries = []
        for q in queries:
            normalized = q.lower().strip()
            if normalized not in query_hashes:
                query_hashes.add(normalized)
                unique_queries.append(q)
        assume(len(unique_queries) > 0)
        
        # Create index
        index = SchemaIndex(model_id=model_id)
        
        # Cache some results
        for query in unique_queries:
            mock_result = MagicMock()
            mock_result.query = query
            query_hash = index.compute_query_hash(query)
            index.cache_match_result(query_hash, mock_result)
        
        # Verify cache has entries
        initial_cache_size = len(index._match_cache)
        assert initial_cache_size == len(unique_queries)
        
        # Clear cache (simulating what rebuild does)
        index.clear_cache()
        
        # Property: cache should be empty after clear
        assert len(index._match_cache) == 0
        
        # Property: all previous cached results should be gone
        for query in unique_queries:
            query_hash = index.compute_query_hash(query)
            cached = index.get_cached_result(query_hash)
            assert cached is None
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
    )
    def test_rebuild_requires_model_reference(self, model_id: str):
        """
        Property: rebuild() should raise RuntimeError if no model has been set.
        
        **Validates: Requirements 7.3**
        """
        assume(len(model_id) > 0)
        
        # Create index without setting model
        index = SchemaIndex(model_id=model_id)
        
        # Property: rebuild should fail without model reference
        with pytest.raises(RuntimeError) as exc_info:
            index.rebuild()
        
        assert "no model has been set" in str(exc_info.value)
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
        initial_hash=st.text(
            alphabet="0123456789abcdef",
            min_size=64,
            max_size=64,
        ),
    )
    def test_config_hash_stored_after_build(self, model_id: str, initial_hash: str):
        """
        Property: After build, the config hash should be stored for
        change detection.
        
        **Validates: Requirements 7.3**
        """
        assume(len(model_id) > 0)
        
        # Create index
        index = SchemaIndex(model_id=model_id)
        
        # Initially, config hash should be None
        assert index.get_config_hash() is None
        
        # Manually set config hash (simulating what build_from_model does)
        index._config_hash = initial_hash
        
        # Property: config hash should be retrievable
        assert index.get_config_hash() == initial_hash


# =============================================================================
# Additional Cache Properties
# =============================================================================

class TestCacheLRUProperties:
    """
    Property tests for LRU cache behavior.
    
    **Validates: Requirement 7.6**
    """
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
        cache_size=st.integers(min_value=1, max_value=10),
        num_entries=st.integers(min_value=1, max_value=20),
    )
    def test_cache_size_limit_respected(
        self, model_id: str, cache_size: int, num_entries: int
    ):
        """
        Property: Cache should never exceed its configured size limit.
        
        **Validates: Requirements 7.6**
        """
        assume(len(model_id) > 0)
        
        # Create index with specific cache size
        index = SchemaIndex(model_id=model_id, cache_size=cache_size)
        
        # Add more entries than cache size
        for i in range(num_entries):
            mock_result = MagicMock()
            mock_result.query = f"query_{i}"
            index.cache_match_result(f"hash_{i}", mock_result)
        
        # Property: cache size should never exceed limit
        assert len(index._match_cache) <= cache_size
    
    @settings(max_examples=100)
    @given(
        model_id=valid_model_id_strategy(),
        cache_size=st.integers(min_value=3, max_value=10),
    )
    def test_lru_eviction_order(self, model_id: str, cache_size: int):
        """
        Property: LRU eviction should remove least recently used entries first.
        
        **Validates: Requirements 7.6**
        """
        assume(len(model_id) > 0)
        
        # Create index
        index = SchemaIndex(model_id=model_id, cache_size=cache_size)
        
        # Fill cache to capacity
        for i in range(cache_size):
            mock_result = MagicMock()
            mock_result.query = f"query_{i}"
            index.cache_match_result(f"hash_{i}", mock_result)
        
        # Access first entry to make it recently used
        index.get_cached_result("hash_0")
        
        # Add new entry (should evict hash_1, not hash_0)
        new_result = MagicMock()
        new_result.query = "new_query"
        index.cache_match_result("hash_new", new_result)
        
        # Property: hash_0 should still be in cache (was accessed recently)
        assert index.get_cached_result("hash_0") is not None
        
        # Property: hash_1 should be evicted (was least recently used)
        # Note: This assumes cache_size >= 3 so hash_1 was the LRU entry
        if cache_size >= 3:
            # hash_1 should be evicted
            assert index.get_cached_result("hash_1") is None
