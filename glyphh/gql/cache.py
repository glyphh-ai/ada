"""
GQL Query Cache - HDC-based semantic query caching.

The query cache stores query results and uses HDC similarity to find
semantically equivalent queries, enabling cache hits even when queries
are not textually identical.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from glyphh.core import Vector
from glyphh.gql.encoder import QueryEncoder

# Hebbian reinforcement bounds
_MIN_STRENGTH = 0.3
_MAX_STRENGTH = 3.0


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
    strength: float = 1.0  # Hebbian reinforcement strength


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
        encoder: Optional[QueryEncoder] = None,
        max_size: int = 1000,
        similarity_threshold: float = 0.95,
        similarity_fn: Optional[Callable[[Vector, Vector], float]] = None,
    ):
        """
        Initialize the query cache.

        Args:
            encoder: QueryEncoder for computing similarity (optional if similarity_fn provided)
            max_size: Maximum number of cached entries
            similarity_threshold: Minimum similarity for cache hit (0.0-1.0)
            similarity_fn: Optional custom similarity function. When provided,
                          used instead of encoder.similarity(). Allows GlyphSpace
                          to create a cache without a GQL QueryEncoder.
        """
        self.encoder = encoder
        self.max_size = max_size
        self.threshold = similarity_threshold
        self._similarity_fn = similarity_fn
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._glyph_index: Dict[str, Set[str]] = {}  # glyph_id -> cache_keys
        self._last_hit_key: Optional[str] = None  # for Hebbian reinforce()
    
    def _compute_similarity(self, v1: Vector, v2: Vector) -> float:
        """Compute similarity using custom function or encoder."""
        if self._similarity_fn is not None:
            return self._similarity_fn(v1, v2)
        if self.encoder is not None:
            return self.encoder.similarity(v1, v2)
        return 0.0

    def get(self, query_vector: Vector) -> Optional[Tuple[Any, float]]:
        """
        Look up a query in the cache using HDC similarity.

        Similarity is weighted by Hebbian strength: entries that have been
        reinforced positively score higher, weakened entries score lower.

        Args:
            query_vector: The encoded query vector

        Returns:
            Tuple of (result, similarity_score) if found, None otherwise
        """
        best_match: Optional[CacheEntry] = None
        best_key: Optional[str] = None
        best_score = 0.0

        for key, entry in self._cache.items():
            raw_score = self._compute_similarity(query_vector, entry.query_vector)
            # Weight by Hebbian strength (capped at 1.0 for scoring)
            weighted_score = raw_score * min(entry.strength, 1.0)
            if weighted_score > best_score and weighted_score >= self.threshold:
                best_score = weighted_score
                best_match = entry
                best_key = key

        if best_match and best_key:
            # Update hit count and move to end (most recently used)
            best_match.hit_count += 1
            self._cache.move_to_end(best_key)
            self._stats.hits += 1
            self._last_hit_key = best_key
            return (best_match.result, best_score)

        self._stats.misses += 1
        self._last_hit_key = None
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
    
    def reinforce(self, correct: bool) -> None:
        """
        Hebbian reinforcement on the most recently retrieved cache entry.

        Strengthens correct results (+0.2), weakens incorrect ones (-0.3).
        Entries that decay below _MIN_STRENGTH are evicted automatically.

        Args:
            correct: Whether the cached result was correct
        """
        if self._last_hit_key is None:
            return
        if self._last_hit_key not in self._cache:
            self._last_hit_key = None
            return

        entry = self._cache[self._last_hit_key]
        if correct:
            entry.strength = min(_MAX_STRENGTH, entry.strength + 0.2)
        else:
            entry.strength = max(0.0, entry.strength - 0.3)
            if entry.strength < _MIN_STRENGTH:
                self._remove_entry(self._last_hit_key)
                self._last_hit_key = None

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
