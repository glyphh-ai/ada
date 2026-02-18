"""
GQL Query Cache - HDC-based semantic query caching.

The query cache stores query results and uses HDC similarity to find
semantically equivalent queries, enabling cache hits even when queries
are not textually identical.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from glyphh.core import Vector
from glyphh.gql.encoder import QueryEncoder


@dataclass
class CacheEntry:
    """
    A cached query result.
    
    Attributes:
        query_vector: The encoded query vector
        query_text: Original query text (for debugging)
        result: The cached result (typically a FactTree)
        created_at: When the entry was created
        hit_count: Number of cache hits
        glyph_refs: Set of glyph IDs referenced in the result (for invalidation)
    """
    query_vector: Vector
    query_text: str
    result: Any
    created_at: datetime = field(default_factory=datetime.now)
    hit_count: int = 0
    glyph_refs: Set[str] = field(default_factory=set)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "hit_rate": self.hit_rate
        }


class QueryCache:
    """
    HDC-based semantic query cache.
    
    The cache uses HDC similarity to match queries, allowing cache hits
    for semantically equivalent queries even if they differ textually.
    
    Features:
    - Semantic matching using HDC similarity
    - LRU eviction when cache is full
    - Glyph-based invalidation
    - Cache statistics tracking
    
    Example:
        >>> from glyphh.gql import parse, QueryEncoder, QueryCache
        >>> 
        >>> encoder = QueryEncoder()
        >>> cache = QueryCache(encoder, max_size=100, similarity_threshold=0.95)
        >>> 
        >>> # Cache a result
        >>> ast = parse('FIND SIMILAR TO "red car" LIMIT 10')
        >>> query_vec = encoder.encode(ast)
        >>> cache.put(query_vec, "FIND SIMILAR...", result, glyph_refs={"g1", "g2"})
        >>> 
        >>> # Look up (may hit even for slightly different query)
        >>> cached = cache.get(query_vec)
        >>> if cached:
        ...     result, similarity = cached
    """
    
    def __init__(
        self,
        encoder: QueryEncoder,
        max_size: int = 1000,
        similarity_threshold: float = 0.95
    ):
        """
        Initialize the query cache.
        
        Args:
            encoder: QueryEncoder for computing similarity
            max_size: Maximum number of cached entries
            similarity_threshold: Minimum similarity for cache hit (0.0-1.0)
        """
        self.encoder = encoder
        self.max_size = max_size
        self.threshold = similarity_threshold
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._glyph_index: Dict[str, Set[str]] = {}  # glyph_id -> cache_keys
    
    def get(self, query_vector: Vector) -> Optional[Tuple[Any, float]]:
        """
        Look up a query in the cache using HDC similarity.
        
        Args:
            query_vector: The encoded query vector
        
        Returns:
            Tuple of (result, similarity_score) if found, None otherwise
        """
        best_match: Optional[CacheEntry] = None
        best_key: Optional[str] = None
        best_score = 0.0
        
        for key, entry in self._cache.items():
            score = self.encoder.similarity(query_vector, entry.query_vector)
            if score > best_score and score >= self.threshold:
                best_score = score
                best_match = entry
                best_key = key
        
        if best_match and best_key:
            # Update hit count and move to end (most recently used)
            best_match.hit_count += 1
            self._cache.move_to_end(best_key)
            self._stats.hits += 1
            return (best_match.result, best_score)
        
        self._stats.misses += 1
        return None
    
    def put(
        self,
        query_vector: Vector,
        query_text: str,
        result: Any,
        glyph_refs: Optional[Set[str]] = None
    ) -> str:
        """
        Store a query result in the cache.
        
        Args:
            query_vector: The encoded query vector
            query_text: Original query text (for debugging)
            result: The result to cache
            glyph_refs: Set of glyph IDs referenced in the result
        
        Returns:
            Cache key for the entry
        """
        # Evict if at capacity (LRU - remove oldest)
        while len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        # Generate cache key from vector hash
        key = self._generate_key(query_vector)
        
        # Create entry
        entry = CacheEntry(
            query_vector=query_vector,
            query_text=query_text,
            result=result,
            glyph_refs=glyph_refs or set()
        )
        
        # Store entry
        self._cache[key] = entry
        
        # Update glyph index for invalidation
        for glyph_id in entry.glyph_refs:
            if glyph_id not in self._glyph_index:
                self._glyph_index[glyph_id] = set()
            self._glyph_index[glyph_id].add(key)
        
        return key
    
    def invalidate_for_glyph(self, glyph_id: str) -> int:
        """
        Invalidate cache entries that reference a specific glyph.
        
        Call this when a glyph is updated or deleted to ensure
        stale results are not returned.
        
        Args:
            glyph_id: The glyph ID to invalidate
        
        Returns:
            Number of entries invalidated
        """
        if glyph_id not in self._glyph_index:
            return 0
        
        keys_to_remove = self._glyph_index[glyph_id].copy()
        count = 0
        
        for key in keys_to_remove:
            if key in self._cache:
                self._remove_entry(key)
                count += 1
                self._stats.invalidations += 1
        
        return count
    
    def invalidate_all(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        self._glyph_index.clear()
        return count
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    def get_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def _generate_key(self, vector: Vector) -> str:
        """Generate a cache key from a vector."""
        return str(hash(vector.data.tobytes()))
    
    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if not self._cache:
            return
        
        # Get oldest key (first item in OrderedDict)
        oldest_key = next(iter(self._cache))
        self._remove_entry(oldest_key)
        self._stats.evictions += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry and update indices."""
        if key not in self._cache:
            return
        
        entry = self._cache[key]
        
        # Remove from glyph index
        for glyph_id in entry.glyph_refs:
            if glyph_id in self._glyph_index:
                self._glyph_index[glyph_id].discard(key)
                if not self._glyph_index[glyph_id]:
                    del self._glyph_index[glyph_id]
        
        # Remove entry
        del self._cache[key]
