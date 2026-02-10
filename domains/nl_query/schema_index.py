"""
Schema Index for Runtime NL Query Matching.

This module provides the SchemaIndex class for maintaining schema vectors
for deployed models in the runtime. The SchemaIndex enables fast NL query
matching by storing pre-computed schema vectors in memory.

The SchemaIndex integrates with the SDK's SchemaVectorizer to generate
vectors when a model is deployed, and provides caching for match results
to improve query performance.

Design Principle: "When your LLM can't afford to be wrong, sidecar it with Glyphh"

Validates: Requirement 7 - Runtime Schema Index
"""

import hashlib
import logging
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Import types for type hints only to avoid circular imports
    from glyphh.nl.schema_vectorizer import SchemaVector
    from glyphh.nl.schema_matcher import MatchResult

logger = logging.getLogger(__name__)


@dataclass
class SchemaIndexMetrics:
    """
    Metrics for schema index.
    
    Tracks performance and resource usage metrics for the schema index,
    enabling monitoring and optimization of NL query matching.
    
    Attributes:
        vector_count: Total number of schema vectors in the index.
            Includes both role vectors and value vectors.
        role_count: Number of role vectors in the index.
            These represent schema role names (e.g., "make", "model").
        value_count: Number of value vectors in the index.
            These represent known values (e.g., "Toyota", "Brake Pads").
        memory_bytes: Estimated memory usage of the index in bytes.
            Includes vector data and metadata overhead.
        build_time_ms: Time taken to build the index in milliseconds.
            Measured from start to completion of build_from_model().
        last_rebuild: Timestamp of the last index rebuild.
            Updated whenever rebuild() or build_from_model() completes.
        cache_hits: Number of cache hits for match results.
            Incremented when a cached result is returned.
        cache_misses: Number of cache misses for match results.
            Incremented when a fresh match is computed.
    
    Example:
        >>> metrics = SchemaIndexMetrics(
        ...     vector_count=150,
        ...     role_count=10,
        ...     value_count=140,
        ...     memory_bytes=1500000,
        ...     build_time_ms=45.5,
        ...     last_rebuild=datetime.now()
        ... )
        >>> print(f"Index has {metrics.vector_count} vectors")
        Index has 150 vectors
        >>> print(f"Memory usage: {metrics.memory_bytes / 1024:.1f} KB")
        Memory usage: 1464.8 KB
    
    Validates: Requirement 7.5 - THE Runtime SHALL provide metrics on schema
    index size and match performance
    """
    vector_count: int = 0
    role_count: int = 0
    value_count: int = 0
    memory_bytes: int = 0
    build_time_ms: float = 0.0
    last_rebuild: Optional[datetime] = None
    cache_hits: int = 0
    cache_misses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to a dictionary for API responses.
        
        Returns:
            Dictionary containing all metrics with serializable values.
        """
        return {
            "vector_count": self.vector_count,
            "role_count": self.role_count,
            "value_count": self.value_count,
            "memory_bytes": self.memory_bytes,
            "build_time_ms": self.build_time_ms,
            "last_rebuild": self.last_rebuild.isoformat() if self.last_rebuild else None,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
        }


class SchemaIndex:
    """
    Runtime schema index for fast NL query matching.
    
    The SchemaIndex maintains pre-computed schema vectors for a deployed model,
    enabling efficient matching of NL query tokens against the schema. It
    integrates with the SDK's SchemaVectorizer to generate vectors and provides
    caching for match results to improve query performance.
    
    Key Features:
    - Stores schema vectors in memory for fast matching
    - Tracks metrics (vector count, memory usage, build time)
    - Caches match results for repeated queries
    - Supports lazy loading for memory efficiency
    - Rebuilds automatically when model config changes
    
    Attributes:
        model_id: Unique identifier for the deployed model.
        _vectors: Dictionary mapping schema element keys to SchemaVector instances.
        _metrics: SchemaIndexMetrics tracking index performance.
        _match_cache: LRU cache for match results.
        _lazy_load_enabled: Whether lazy loading is enabled.
        _vectorizer: Reference to the SchemaVectorizer (set during build).
        _config_hash: Hash of the model config for change detection.
    
    Example:
        >>> from domains.nl_query.schema_index import SchemaIndex
        >>> 
        >>> # Create index for a deployed model
        >>> index = SchemaIndex(model_id="my-model-123")
        >>> 
        >>> # Build index from model (called during deployment)
        >>> # index.build_from_model(model)
        >>> 
        >>> # Get metrics
        >>> metrics = index.get_metrics()
        >>> print(f"Index has {metrics.vector_count} vectors")
        >>> 
        >>> # Get vectors for matching
        >>> vectors = index.get_vectors()
    
    Validates: Requirement 7 - Runtime Schema Index
    - 7.1: THE Runtime SHALL build a schema index when a model is deployed
    - 7.2: THE Runtime SHALL store schema vectors in memory for fast matching
    - 7.3: WHEN a model config is updated, THE Runtime SHALL rebuild the schema index
    - 7.4: THE Runtime SHALL support lazy loading of schema indexes for memory efficiency
    - 7.5: THE Runtime SHALL provide metrics on schema index size and match performance
    - 7.6: THE Runtime SHALL cache match results for repeated queries
    """
    
    # Default cache size limit (number of entries)
    DEFAULT_CACHE_SIZE = 1000
    
    def __init__(
        self,
        model_id: str,
        cache_size: int = DEFAULT_CACHE_SIZE,
        lazy_load: bool = False
    ):
        """
        Initialize the SchemaIndex for a deployed model.
        
        Creates an empty schema index that will be populated when
        build_from_model() is called during model deployment.
        
        Args:
            model_id: Unique identifier for the deployed model.
                Used for logging and metrics tracking.
            cache_size: Maximum number of match results to cache.
                Defaults to 1000 entries. Uses LRU eviction.
            lazy_load: Whether to enable lazy loading of vectors.
                When True, vectors are loaded on demand.
                Defaults to False (eager loading).
        
        Raises:
            ValueError: If model_id is empty or contains only whitespace.
        
        Example:
            >>> index = SchemaIndex(model_id="my-model-123")
            >>> print(index.model_id)
            'my-model-123'
            >>> 
            >>> # With custom cache size
            >>> index = SchemaIndex(model_id="my-model", cache_size=500)
            >>> 
            >>> # With lazy loading enabled
            >>> index = SchemaIndex(model_id="my-model", lazy_load=True)
        
        Validates: Requirement 7.4 - THE Runtime SHALL support lazy loading
        of schema indexes for memory efficiency
        """
        # Validate model_id
        if not model_id or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        
        # Store model identifier
        self.model_id = model_id
        
        # Initialize empty vectors dictionary
        # Key: schema element key (e.g., "make" or "make=Toyota")
        # Value: SchemaVector instance
        # Validates: Requirement 7.2
        self._vectors: Dict[str, "SchemaVector"] = {}
        
        # Initialize metrics
        # Validates: Requirement 7.5
        self._metrics: SchemaIndexMetrics = SchemaIndexMetrics()
        
        # Initialize LRU cache for match results
        # Uses OrderedDict for LRU eviction
        # Validates: Requirement 7.6
        self._cache_size = cache_size
        self._match_cache: OrderedDict[str, "MatchResult"] = OrderedDict()
        
        # Lazy loading configuration
        # Validates: Requirement 7.4
        self._lazy_load_enabled = lazy_load
        self._lazy_load_pending = False
        self._model_reference: Optional[Any] = None
        
        # Reference to vectorizer (set during build)
        self._vectorizer: Optional[Any] = None
        
        # Config hash for change detection
        # Validates: Requirement 7.3
        self._config_hash: Optional[str] = None
        
        # Index state
        self._is_built = False
        self._is_building = False
        
        logger.info(
            f"SchemaIndex initialized for model '{model_id}' "
            f"(cache_size={cache_size}, lazy_load={lazy_load})"
        )
    
    @property
    def is_built(self) -> bool:
        """Check if the index has been built."""
        return self._is_built
    
    @property
    def is_building(self) -> bool:
        """Check if the index is currently being built."""
        return self._is_building
    
    def get_vectors(self) -> Dict[str, "SchemaVector"]:
        """
        Get all schema vectors.
        
        Returns the dictionary of schema vectors stored in the index.
        If lazy loading is enabled and vectors haven't been loaded yet,
        this will trigger loading.
        
        Returns:
            Dictionary mapping schema element keys to SchemaVector instances.
            Keys are either role names (e.g., "make") or role=value pairs
            (e.g., "make=Toyota").
        
        Example:
            >>> index = SchemaIndex(model_id="my-model")
            >>> # After build_from_model() is called:
            >>> vectors = index.get_vectors()
            >>> print(f"Index has {len(vectors)} vectors")
        
        Validates: Requirement 7.2 - THE Runtime SHALL store schema vectors
        in memory for fast matching
        """
        # Handle lazy loading if enabled and pending
        if self._lazy_load_enabled and self._lazy_load_pending:
            self._load_vectors_lazy()
        
        return self._vectors
    
    def get_metrics(self) -> SchemaIndexMetrics:
        """
        Get index metrics.
        
        Returns the current metrics for the schema index, including
        vector counts, memory usage, build time, and cache statistics.
        
        Returns:
            SchemaIndexMetrics instance with current metrics.
        
        Example:
            >>> index = SchemaIndex(model_id="my-model")
            >>> metrics = index.get_metrics()
            >>> print(f"Vector count: {metrics.vector_count}")
            >>> print(f"Memory usage: {metrics.memory_bytes} bytes")
            >>> print(f"Build time: {metrics.build_time_ms} ms")
        
        Validates: Requirement 7.5 - THE Runtime SHALL provide metrics on
        schema index size and match performance
        """
        return self._metrics
    
    def cache_match_result(self, query_hash: str, result: "MatchResult") -> None:
        """
        Cache a match result.
        
        Stores a match result in the LRU cache for future retrieval.
        If the cache is full, the least recently used entry is evicted.
        
        Args:
            query_hash: Hash of the query string used as cache key.
                Should be computed using compute_query_hash().
            result: The MatchResult to cache.
        
        Example:
            >>> index = SchemaIndex(model_id="my-model")
            >>> query = "find Toyota brake pads"
            >>> query_hash = index.compute_query_hash(query)
            >>> # After matching:
            >>> index.cache_match_result(query_hash, match_result)
        
        Validates: Requirement 7.6 - THE Runtime SHALL cache match results
        for repeated queries
        """
        # If key already exists, move to end (most recently used)
        if query_hash in self._match_cache:
            self._match_cache.move_to_end(query_hash)
            self._match_cache[query_hash] = result
            return
        
        # Evict LRU entry if cache is full
        if len(self._match_cache) >= self._cache_size:
            # Remove oldest entry (first item)
            self._match_cache.popitem(last=False)
            logger.debug(f"Cache eviction for model '{self.model_id}'")
        
        # Add new entry
        self._match_cache[query_hash] = result
    
    def get_cached_result(self, query_hash: str) -> Optional["MatchResult"]:
        """
        Get cached match result.
        
        Retrieves a previously cached match result if available.
        Updates cache statistics (hits/misses) for metrics.
        
        Args:
            query_hash: Hash of the query string used as cache key.
                Should be computed using compute_query_hash().
        
        Returns:
            The cached MatchResult if found, None otherwise.
        
        Example:
            >>> index = SchemaIndex(model_id="my-model")
            >>> query = "find Toyota brake pads"
            >>> query_hash = index.compute_query_hash(query)
            >>> cached = index.get_cached_result(query_hash)
            >>> if cached:
            ...     print("Cache hit!")
            ... else:
            ...     print("Cache miss, computing fresh result")
        
        Validates: Requirement 7.6 - THE Runtime SHALL cache match results
        for repeated queries
        """
        if query_hash in self._match_cache:
            # Move to end (most recently used)
            self._match_cache.move_to_end(query_hash)
            self._metrics.cache_hits += 1
            return self._match_cache[query_hash]
        
        self._metrics.cache_misses += 1
        return None
    
    def compute_query_hash(self, query: str) -> str:
        """
        Compute hash for a query string.
        
        Generates a SHA-256 hash of the query string for use as a cache key.
        The query is normalized (lowercased, stripped) before hashing.
        
        Args:
            query: The query string to hash.
        
        Returns:
            Hexadecimal string representing the SHA-256 hash.
        
        Example:
            >>> index = SchemaIndex(model_id="my-model")
            >>> hash1 = index.compute_query_hash("Find Toyota")
            >>> hash2 = index.compute_query_hash("find toyota")
            >>> assert hash1 == hash2  # Normalized to same hash
        """
        # Normalize query before hashing
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def clear_cache(self) -> None:
        """
        Clear the match result cache.
        
        Removes all cached match results and resets cache statistics.
        Useful when the schema index is rebuilt.
        """
        self._match_cache.clear()
        self._metrics.cache_hits = 0
        self._metrics.cache_misses = 0
        logger.info(f"Cache cleared for model '{self.model_id}'")
    
    def build_from_model(self, model: Any) -> None:
        """
        Build index from deployed model.
        
        Generates schema vectors for all roles and values in the model's
        config using the SDK's SchemaVectorizer. This method is called
        when a model is deployed to the runtime.
        
        Args:
            model: The deployed model object. Must have:
                - encoder: The Encoder instance
                - config: The EncoderConfig with schema definition
        
        Raises:
            RuntimeError: If the index is already being built.
            ValueError: If the model is missing required attributes.
        
        Example:
            >>> index = SchemaIndex(model_id="my-model")
            >>> index.build_from_model(deployed_model)
            >>> print(f"Built index with {index.get_metrics().vector_count} vectors")
        
        Validates: Requirement 7.1 - THE Runtime SHALL build a schema index
        when a model is deployed
        """
        # Check if already building
        if self._is_building:
            raise RuntimeError(
                f"Schema index for model '{self.model_id}' is already being built"
            )
        
        # Validate model has required attributes
        if not hasattr(model, 'encoder'):
            raise ValueError("model must have an 'encoder' attribute")
        if not hasattr(model, 'config'):
            raise ValueError("model must have a 'config' attribute")
        
        # Import SchemaVectorizer from SDK
        # This is done here to avoid circular imports at module level
        from glyphh.nl.schema_vectorizer import SchemaVectorizer
        
        # Mark as building
        self._is_building = True
        start_time = time.time()
        
        try:
            # Store model reference for lazy loading and rebuild
            self._model_reference = model
            
            # Create vectorizer using model's encoder and config
            self._vectorizer = SchemaVectorizer(model.encoder, model.config)
            
            # If lazy loading is enabled, defer actual vectorization
            if self._lazy_load_enabled:
                self._lazy_load_pending = True
                self._is_built = True
                self._is_building = False
                
                # Update metrics with minimal info (vectors not loaded yet)
                self._metrics = SchemaIndexMetrics(
                    vector_count=0,
                    role_count=0,
                    value_count=0,
                    memory_bytes=0,
                    build_time_ms=(time.time() - start_time) * 1000,
                    last_rebuild=datetime.now(),
                    cache_hits=self._metrics.cache_hits,
                    cache_misses=self._metrics.cache_misses,
                )
                
                logger.info(
                    f"SchemaIndex for model '{self.model_id}' initialized with "
                    f"lazy loading enabled (vectors will be loaded on demand)"
                )
                return
            
            # Generate schema vectors using the SDK's SchemaVectorizer
            schema_vectors = self._vectorizer.vectorize_schema()
            
            # Store vectors in the index
            self._vectors = schema_vectors.copy()
            
            # Store config hash for change detection
            self._config_hash = self._vectorizer.get_config_hash()
            
            # Calculate build time
            build_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self._update_metrics(build_time_ms)
            
            # Mark as built
            self._is_built = True
            
            logger.info(
                f"SchemaIndex built for model '{self.model_id}': "
                f"{len(self._vectors)} vectors in {build_time_ms:.2f}ms"
            )
            
        finally:
            self._is_building = False
    
    def rebuild(self) -> None:
        """
        Rebuild index (e.g., after config change).
        
        Regenerates all schema vectors using the stored model reference.
        Clears the match cache since cached results may be invalid.
        
        Raises:
            RuntimeError: If no model has been set (build_from_model not called).
            RuntimeError: If the index is already being built.
        
        Example:
            >>> index = SchemaIndex(model_id="my-model")
            >>> index.build_from_model(model)
            >>> # After config change:
            >>> index.rebuild()
        
        Validates: Requirement 7.3 - WHEN a model config is updated,
        THE Runtime SHALL rebuild the schema index
        """
        # Check if model reference exists
        if self._model_reference is None:
            raise RuntimeError(
                f"Cannot rebuild schema index for model '{self.model_id}': "
                f"no model has been set. Call build_from_model() first."
            )
        
        # Check if already building
        if self._is_building:
            raise RuntimeError(
                f"Schema index for model '{self.model_id}' is already being built"
            )
        
        logger.info(f"Rebuilding schema index for model '{self.model_id}'")
        
        # Clear the cache since cached results may be invalid after rebuild
        self.clear_cache()
        
        # Clear existing vectors
        self._vectors.clear()
        
        # Reset lazy load state
        self._lazy_load_pending = False
        
        # Mark as not built during rebuild
        self._is_built = False
        
        # Rebuild using the stored model reference
        self.build_from_model(self._model_reference)
    
    def _load_vectors_lazy(self) -> None:
        """
        Load vectors on demand (lazy loading).
        
        Called when get_vectors() is accessed and lazy loading is enabled.
        Loads vectors from the stored model reference.
        
        Validates: Requirement 7.4 - THE Runtime SHALL support lazy loading
        of schema indexes for memory efficiency
        """
        # Check if lazy loading is pending
        if not self._lazy_load_pending:
            return
        
        # Check if vectorizer exists
        if self._vectorizer is None:
            logger.warning(
                f"Cannot lazy load vectors for model '{self.model_id}': "
                f"vectorizer not initialized"
            )
            return
        
        logger.info(f"Lazy loading schema vectors for model '{self.model_id}'")
        
        start_time = time.time()
        
        # Generate schema vectors using the SDK's SchemaVectorizer
        schema_vectors = self._vectorizer.vectorize_schema()
        
        # Store vectors in the index
        self._vectors = schema_vectors.copy()
        
        # Store config hash for change detection
        self._config_hash = self._vectorizer.get_config_hash()
        
        # Mark lazy loading as complete
        self._lazy_load_pending = False
        
        # Calculate load time
        load_time_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        self._update_metrics(load_time_ms)
        
        logger.info(
            f"Lazy loaded {len(self._vectors)} vectors for model '{self.model_id}' "
            f"in {load_time_ms:.2f}ms"
        )
    
    def _estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of the index in bytes.
        
        Calculates an estimate of the memory used by the schema vectors
        and associated metadata. This is used for metrics reporting.
        
        Returns:
            Estimated memory usage in bytes.
        
        Validates: Requirement 7.5 - THE Runtime SHALL provide metrics on
        schema index size and match performance
        """
        total_bytes = 0
        
        for key, schema_vector in self._vectors.items():
            # Key string size
            total_bytes += sys.getsizeof(key)
            
            # SchemaVector object overhead
            total_bytes += sys.getsizeof(schema_vector)
            
            # Vector data (numpy array)
            if hasattr(schema_vector, 'vector') and schema_vector.vector is not None:
                if hasattr(schema_vector.vector, 'data'):
                    total_bytes += schema_vector.vector.data.nbytes
                elif hasattr(schema_vector.vector, 'nbytes'):
                    total_bytes += schema_vector.vector.nbytes
            
            # String fields
            if schema_vector.key:
                total_bytes += sys.getsizeof(schema_vector.key)
            if schema_vector.role_path:
                total_bytes += sys.getsizeof(schema_vector.role_path)
            if schema_vector.original_value:
                total_bytes += sys.getsizeof(schema_vector.original_value)
        
        # Cache overhead
        total_bytes += sys.getsizeof(self._match_cache)
        
        return total_bytes
    
    def _update_metrics(self, build_time_ms: float) -> None:
        """
        Update metrics after building or rebuilding the index.
        
        Args:
            build_time_ms: Time taken to build the index in milliseconds.
        """
        role_count = sum(
            1 for v in self._vectors.values()
            if v.element_type == "role"
        )
        value_count = sum(
            1 for v in self._vectors.values()
            if v.element_type == "value"
        )
        
        self._metrics = SchemaIndexMetrics(
            vector_count=len(self._vectors),
            role_count=role_count,
            value_count=value_count,
            memory_bytes=self._estimate_memory_usage(),
            build_time_ms=build_time_ms,
            last_rebuild=datetime.now(),
            cache_hits=self._metrics.cache_hits,
            cache_misses=self._metrics.cache_misses,
        )
        
        logger.info(
            f"SchemaIndex metrics updated for model '{self.model_id}': "
            f"vectors={self._metrics.vector_count}, "
            f"roles={self._metrics.role_count}, "
            f"values={self._metrics.value_count}, "
            f"memory={self._metrics.memory_bytes} bytes, "
            f"build_time={self._metrics.build_time_ms:.2f} ms"
        )

    def get_config_hash(self) -> Optional[str]:
        """
        Get the stored config hash used for change detection.
        
        Returns the hash that was computed when the index was built.
        This can be used to detect if the model config has changed.
        
        Returns:
            The stored config hash string, or None if the index
            has not been built yet.
        
        Example:
            >>> index = SchemaIndex(model_id="my-model")
            >>> index.build_from_model(model)
            >>> hash_value = index.get_config_hash()
            >>> print(f"Config hash: {hash_value}")
        
        Validates: Requirement 7.3 - WHEN a model config is updated,
        THE Runtime SHALL rebuild the schema index
        """
        return self._config_hash
    
    def check_config_changed(self) -> bool:
        """
        Check if the model config has changed since the index was built.
        
        Compares the current config hash with the stored hash to detect
        if the config has been modified. If the config has changed,
        the index should be rebuilt.
        
        Returns:
            True if the config has changed, False otherwise.
            Returns False if the index has not been built yet.
        
        Example:
            >>> index = SchemaIndex(model_id="my-model")
            >>> index.build_from_model(model)
            >>> 
            >>> # Modify the model config...
            >>> 
            >>> if index.check_config_changed():
            ...     index.rebuild()
        
        Validates: Requirement 7.3 - WHEN a model config is updated,
        THE Runtime SHALL rebuild the schema index
        """
        # If no vectorizer, can't check for changes
        if self._vectorizer is None:
            return False
        
        # If no stored hash, index hasn't been built
        if self._config_hash is None:
            return False
        
        # Compute current hash and compare
        current_hash = self._vectorizer._compute_config_hash()
        return current_hash != self._config_hash
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.
        
        Returns statistics about the match result cache, including
        hit rate, size, and capacity.
        
        Returns:
            Dictionary containing cache statistics:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - size: Current number of cached entries
            - capacity: Maximum cache capacity
        
        Example:
            >>> index = SchemaIndex(model_id="my-model", cache_size=1000)
            >>> # After some queries...
            >>> stats = index.get_cache_stats()
            >>> print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        
        Validates: Requirement 7.5 - THE Runtime SHALL provide metrics on
        schema index size and match performance
        """
        total_requests = self._metrics.cache_hits + self._metrics.cache_misses
        hit_rate = (
            self._metrics.cache_hits / total_requests
            if total_requests > 0
            else 0.0
        )
        
        return {
            "hits": self._metrics.cache_hits,
            "misses": self._metrics.cache_misses,
            "hit_rate": hit_rate,
            "size": len(self._match_cache),
            "capacity": self._cache_size,
        }
