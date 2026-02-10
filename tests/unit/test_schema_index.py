"""
Unit tests for SchemaIndex.

Tests the runtime schema index for NL query matching.
Validates: Requirement 7 - Runtime Schema Index
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from domains.nl_query.schema_index import SchemaIndex, SchemaIndexMetrics


class TestSchemaIndexMetrics:
    """Tests for SchemaIndexMetrics dataclass."""
    
    def test_default_values(self):
        """Test that metrics have correct default values."""
        metrics = SchemaIndexMetrics()
        
        assert metrics.vector_count == 0
        assert metrics.role_count == 0
        assert metrics.value_count == 0
        assert metrics.memory_bytes == 0
        assert metrics.build_time_ms == 0.0
        assert metrics.last_rebuild is None
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
    
    def test_custom_values(self):
        """Test metrics with custom values."""
        now = datetime.now()
        metrics = SchemaIndexMetrics(
            vector_count=150,
            role_count=10,
            value_count=140,
            memory_bytes=1500000,
            build_time_ms=45.5,
            last_rebuild=now,
            cache_hits=100,
            cache_misses=20,
        )
        
        assert metrics.vector_count == 150
        assert metrics.role_count == 10
        assert metrics.value_count == 140
        assert metrics.memory_bytes == 1500000
        assert metrics.build_time_ms == 45.5
        assert metrics.last_rebuild == now
        assert metrics.cache_hits == 100
        assert metrics.cache_misses == 20
    
    def test_to_dict(self):
        """Test metrics serialization to dictionary."""
        now = datetime.now()
        metrics = SchemaIndexMetrics(
            vector_count=100,
            role_count=10,
            value_count=90,
            memory_bytes=1000000,
            build_time_ms=30.0,
            last_rebuild=now,
            cache_hits=50,
            cache_misses=10,
        )
        
        result = metrics.to_dict()
        
        assert result["vector_count"] == 100
        assert result["role_count"] == 10
        assert result["value_count"] == 90
        assert result["memory_bytes"] == 1000000
        assert result["build_time_ms"] == 30.0
        assert result["last_rebuild"] == now.isoformat()
        assert result["cache_hits"] == 50
        assert result["cache_misses"] == 10
        assert result["cache_hit_rate"] == pytest.approx(50 / 60)
    
    def test_to_dict_no_cache_activity(self):
        """Test cache_hit_rate is 0 when no cache activity."""
        metrics = SchemaIndexMetrics()
        result = metrics.to_dict()
        
        assert result["cache_hit_rate"] == 0.0
    
    def test_to_dict_no_last_rebuild(self):
        """Test last_rebuild is None in dict when not set."""
        metrics = SchemaIndexMetrics()
        result = metrics.to_dict()
        
        assert result["last_rebuild"] is None


class TestSchemaIndex:
    """Tests for SchemaIndex class."""
    
    def test_init_valid_model_id(self):
        """Test initialization with valid model_id."""
        index = SchemaIndex(model_id="my-model-123")
        
        assert index.model_id == "my-model-123"
        assert index.is_built is False
        assert index.is_building is False
        assert len(index.get_vectors()) == 0
    
    def test_init_empty_model_id_raises(self):
        """Test that empty model_id raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SchemaIndex(model_id="")
        
        assert "model_id must be a non-empty string" in str(exc_info.value)
    
    def test_init_whitespace_model_id_raises(self):
        """Test that whitespace-only model_id raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SchemaIndex(model_id="   ")
        
        assert "model_id must be a non-empty string" in str(exc_info.value)
    
    def test_init_custom_cache_size(self):
        """Test initialization with custom cache size."""
        index = SchemaIndex(model_id="my-model", cache_size=500)
        
        assert index._cache_size == 500
    
    def test_init_lazy_load_enabled(self):
        """Test initialization with lazy loading enabled."""
        index = SchemaIndex(model_id="my-model", lazy_load=True)
        
        assert index._lazy_load_enabled is True
    
    def test_get_vectors_empty(self):
        """Test get_vectors returns empty dict initially."""
        index = SchemaIndex(model_id="my-model")
        
        vectors = index.get_vectors()
        
        assert isinstance(vectors, dict)
        assert len(vectors) == 0
    
    def test_get_metrics_initial(self):
        """Test get_metrics returns initial metrics."""
        index = SchemaIndex(model_id="my-model")
        
        metrics = index.get_metrics()
        
        assert isinstance(metrics, SchemaIndexMetrics)
        assert metrics.vector_count == 0
        assert metrics.role_count == 0
        assert metrics.value_count == 0
    
    def test_compute_query_hash_deterministic(self):
        """Test that query hash is deterministic."""
        index = SchemaIndex(model_id="my-model")
        
        hash1 = index.compute_query_hash("find Toyota brake pads")
        hash2 = index.compute_query_hash("find Toyota brake pads")
        
        assert hash1 == hash2
    
    def test_compute_query_hash_normalized(self):
        """Test that query hash normalizes case and whitespace."""
        index = SchemaIndex(model_id="my-model")
        
        hash1 = index.compute_query_hash("Find Toyota")
        hash2 = index.compute_query_hash("find toyota")
        hash3 = index.compute_query_hash("  FIND TOYOTA  ")
        
        assert hash1 == hash2
        assert hash2 == hash3
    
    def test_compute_query_hash_different_queries(self):
        """Test that different queries produce different hashes."""
        index = SchemaIndex(model_id="my-model")
        
        hash1 = index.compute_query_hash("find Toyota")
        hash2 = index.compute_query_hash("find Honda")
        
        assert hash1 != hash2
    
    def test_cache_match_result_and_retrieve(self):
        """Test caching and retrieving match results."""
        index = SchemaIndex(model_id="my-model")
        
        # Create a mock match result
        mock_result = MagicMock()
        mock_result.query = "find Toyota"
        
        query_hash = index.compute_query_hash("find Toyota")
        
        # Cache the result
        index.cache_match_result(query_hash, mock_result)
        
        # Retrieve the result
        cached = index.get_cached_result(query_hash)
        
        assert cached is mock_result
        assert index.get_metrics().cache_hits == 1
    
    def test_cache_miss_increments_counter(self):
        """Test that cache miss increments counter."""
        index = SchemaIndex(model_id="my-model")
        
        query_hash = index.compute_query_hash("find Toyota")
        
        # Try to get non-existent result
        cached = index.get_cached_result(query_hash)
        
        assert cached is None
        assert index.get_metrics().cache_misses == 1
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Create index with small cache
        index = SchemaIndex(model_id="my-model", cache_size=3)
        
        # Add 3 entries
        for i in range(3):
            mock_result = MagicMock()
            mock_result.query = f"query{i}"
            index.cache_match_result(f"hash{i}", mock_result)
        
        # Add 4th entry - should evict first
        mock_result = MagicMock()
        mock_result.query = "query3"
        index.cache_match_result("hash3", mock_result)
        
        # First entry should be evicted
        assert index.get_cached_result("hash0") is None
        
        # Other entries should still exist
        assert index.get_cached_result("hash1") is not None
        assert index.get_cached_result("hash2") is not None
        assert index.get_cached_result("hash3") is not None
    
    def test_cache_update_moves_to_end(self):
        """Test that updating a cached entry moves it to end (most recent)."""
        index = SchemaIndex(model_id="my-model", cache_size=3)
        
        # Add 3 entries
        for i in range(3):
            mock_result = MagicMock()
            mock_result.query = f"query{i}"
            index.cache_match_result(f"hash{i}", mock_result)
        
        # Access first entry (moves to end)
        index.get_cached_result("hash0")
        
        # Add new entry - should evict hash1 (now oldest)
        mock_result = MagicMock()
        mock_result.query = "query3"
        index.cache_match_result("hash3", mock_result)
        
        # hash0 should still exist (was accessed recently)
        assert index.get_cached_result("hash0") is not None
        
        # hash1 should be evicted
        assert index.get_cached_result("hash1") is None
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        index = SchemaIndex(model_id="my-model")
        
        # Add some entries
        for i in range(5):
            mock_result = MagicMock()
            index.cache_match_result(f"hash{i}", mock_result)
        
        # Access some entries to build up stats
        index.get_cached_result("hash0")
        index.get_cached_result("hash1")
        index.get_cached_result("nonexistent")
        
        # Clear cache
        index.clear_cache()
        
        # Cache should be empty
        assert index.get_cached_result("hash0") is None
        
        # Stats should be reset
        metrics = index.get_metrics()
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 1  # The miss from checking hash0 after clear
    
    def test_build_from_model_missing_encoder(self):
        """Test that build_from_model raises ValueError if model lacks encoder."""
        index = SchemaIndex(model_id="my-model")
        
        mock_model = MagicMock(spec=[])  # No attributes
        
        with pytest.raises(ValueError) as exc_info:
            index.build_from_model(mock_model)
        
        assert "encoder" in str(exc_info.value)
    
    def test_build_from_model_missing_config(self):
        """Test that build_from_model raises ValueError if model lacks config."""
        index = SchemaIndex(model_id="my-model")
        
        mock_model = MagicMock(spec=['encoder'])  # Has encoder but no config
        
        with pytest.raises(ValueError) as exc_info:
            index.build_from_model(mock_model)
        
        assert "config" in str(exc_info.value)
    
    def test_rebuild_without_model_raises(self):
        """Test that rebuild raises RuntimeError if no model has been set."""
        index = SchemaIndex(model_id="my-model")
        
        with pytest.raises(RuntimeError) as exc_info:
            index.rebuild()
        
        assert "no model has been set" in str(exc_info.value)
        assert "build_from_model()" in str(exc_info.value)
    
    def test_is_built_property(self):
        """Test is_built property."""
        index = SchemaIndex(model_id="my-model")
        
        assert index.is_built is False
        
        # Manually set for testing
        index._is_built = True
        assert index.is_built is True
    
    def test_is_building_property(self):
        """Test is_building property."""
        index = SchemaIndex(model_id="my-model")
        
        assert index.is_building is False
        
        # Manually set for testing
        index._is_building = True
        assert index.is_building is True


class TestSchemaIndexMemoryEstimation:
    """Tests for memory estimation functionality."""
    
    def test_estimate_memory_empty_index(self):
        """Test memory estimation for empty index."""
        index = SchemaIndex(model_id="my-model")
        
        memory = index._estimate_memory_usage()
        
        # Should be minimal (just cache overhead)
        assert memory >= 0
    
    def test_update_metrics(self):
        """Test _update_metrics method."""
        index = SchemaIndex(model_id="my-model")
        
        # Create mock vectors
        mock_role_vector = MagicMock()
        mock_role_vector.element_type = "role"
        mock_role_vector.key = "make"
        mock_role_vector.role_path = "vehicle.identity.make"
        mock_role_vector.original_value = None
        mock_role_vector.vector = MagicMock()
        mock_role_vector.vector.data = MagicMock()
        mock_role_vector.vector.data.nbytes = 10000
        
        mock_value_vector = MagicMock()
        mock_value_vector.element_type = "value"
        mock_value_vector.key = "make=Toyota"
        mock_value_vector.role_path = "vehicle.identity.make"
        mock_value_vector.original_value = "Toyota"
        mock_value_vector.vector = MagicMock()
        mock_value_vector.vector.data = MagicMock()
        mock_value_vector.vector.data.nbytes = 10000
        
        # Add vectors to index
        index._vectors = {
            "make": mock_role_vector,
            "make=Toyota": mock_value_vector,
        }
        
        # Update metrics
        index._update_metrics(build_time_ms=50.0)
        
        metrics = index.get_metrics()
        
        assert metrics.vector_count == 2
        assert metrics.role_count == 1
        assert metrics.value_count == 1
        assert metrics.build_time_ms == 50.0
        assert metrics.last_rebuild is not None
        assert metrics.memory_bytes > 0


class TestSchemaIndexBuildFromModel:
    """Tests for build_from_model() method."""
    
    def test_build_from_model_already_building_raises(self):
        """Test that build_from_model raises if already building."""
        index = SchemaIndex(model_id="my-model")
        index._is_building = True
        
        mock_model = MagicMock()
        mock_model.encoder = MagicMock()
        mock_model.config = MagicMock()
        
        with pytest.raises(RuntimeError) as exc_info:
            index.build_from_model(mock_model)
        
        assert "already being built" in str(exc_info.value)


class TestSchemaIndexRebuild:
    """Tests for rebuild() method."""
    
    def test_rebuild_already_building_raises(self):
        """Test that rebuild raises if already building."""
        index = SchemaIndex(model_id="my-model")
        index._model_reference = MagicMock()  # Set model reference
        index._is_building = True
        
        with pytest.raises(RuntimeError) as exc_info:
            index.rebuild()
        
        assert "already being built" in str(exc_info.value)


class TestSchemaIndexLazyLoading:
    """Tests for lazy loading functionality."""
    
    def test_lazy_load_pending_flag(self):
        """Test that lazy_load_pending flag is set correctly."""
        index = SchemaIndex(model_id="my-model", lazy_load=True)
        
        assert index._lazy_load_enabled is True
        assert index._lazy_load_pending is False  # Not pending until build
    
    def test_get_vectors_triggers_lazy_load(self):
        """Test that get_vectors triggers lazy loading when pending."""
        index = SchemaIndex(model_id="my-model", lazy_load=True)
        
        # Simulate lazy load pending state
        index._lazy_load_pending = True
        index._vectorizer = None  # No vectorizer, so lazy load will warn
        
        # get_vectors should attempt lazy load
        vectors = index.get_vectors()
        
        # Should return empty dict since vectorizer is None
        assert vectors == {}


class TestSchemaIndexConfigHash:
    """Tests for config hash functionality."""
    
    def test_get_config_hash_initial(self):
        """Test that config hash is None initially."""
        index = SchemaIndex(model_id="my-model")
        
        assert index.get_config_hash() is None
    
    def test_check_config_changed_no_vectorizer(self):
        """Test check_config_changed returns False without vectorizer."""
        index = SchemaIndex(model_id="my-model")
        
        assert index.check_config_changed() is False
    
    def test_check_config_changed_no_hash(self):
        """Test check_config_changed returns False without stored hash."""
        index = SchemaIndex(model_id="my-model")
        index._vectorizer = MagicMock()
        
        assert index.check_config_changed() is False


class TestSchemaIndexCacheStats:
    """Tests for cache statistics functionality."""
    
    def test_get_cache_stats_initial(self):
        """Test cache stats are correct initially."""
        index = SchemaIndex(model_id="my-model", cache_size=500)
        
        stats = index.get_cache_stats()
        
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["size"] == 0
        assert stats["capacity"] == 500
    
    def test_get_cache_stats_after_activity(self):
        """Test cache stats after cache activity."""
        index = SchemaIndex(model_id="my-model", cache_size=100)
        
        # Add some cache entries
        for i in range(5):
            mock_result = MagicMock()
            index.cache_match_result(f"hash{i}", mock_result)
        
        # Generate some hits and misses
        index.get_cached_result("hash0")  # Hit
        index.get_cached_result("hash1")  # Hit
        index.get_cached_result("nonexistent")  # Miss
        
        stats = index.get_cache_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2/3)
        assert stats["size"] == 5
        assert stats["capacity"] == 100
