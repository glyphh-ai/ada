"""
GQL Storage Abstraction - Protocol and implementations for glyph storage.

This module defines the GlyphStorageProtocol that abstracts glyph storage
operations, enabling the GQL executor to work with any storage backend
(in-memory SDK glyphs, database-stored runtime glyphs, etc.).

The design follows the Strategy pattern:
- GlyphStorageProtocol defines the interface
- InMemoryGlyphStorage implements it for SDK Glyph objects
- DatabaseGlyphStorage (in runtime) implements it for database records
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Union


# Type aliases for clarity
VectorLike = Union[Any, List[float]]  # SDK Vector, numpy array, or list
GlyphId = str


@runtime_checkable
class GlyphStorageProtocol(Protocol):
    """
    Protocol defining glyph storage operations for GQL execution.
    
    This protocol abstracts the storage layer, allowing the GQL executor
    to work with different backends without modification. Implementations
    must provide all methods defined here.
    
    The @runtime_checkable decorator enables isinstance() checks against
    this protocol, useful for validation.
    
    Example:
        >>> storage = InMemoryGlyphStorage(glyphs=my_glyphs)
        >>> assert isinstance(storage, GlyphStorageProtocol)
        >>> context = ExecutionContext(storage=storage)
    """
    
    def list_glyphs(self) -> Dict[str, Any]:
        """
        List all glyphs in storage.
        
        Returns:
            Dictionary mapping glyph_id (str) to glyph object.
            The glyph object type depends on the implementation
            (SDK Glyph, GlyphResponse, etc.).
        
        Note:
            For large datasets, implementations may want to add
            pagination support in the future.
        """
        ...
    
    def get_glyph(self, glyph_id: GlyphId) -> Any:
        """
        Get a glyph by ID.
        
        Args:
            glyph_id: The glyph identifier (string)
            
        Returns:
            The glyph object. Type depends on implementation.
            
        Raises:
            KeyError: If glyph not found
        """
        ...
    
    def has_glyph(self, glyph_id: GlyphId) -> bool:
        """
        Check if a glyph exists in storage.
        
        Args:
            glyph_id: The glyph identifier (string)
            
        Returns:
            True if glyph exists, False otherwise
        
        Note:
            This should be consistent with get_glyph():
            has_glyph(id) == True implies get_glyph(id) succeeds
        """
        ...
    
    def get_embedding(self, glyph_id: GlyphId) -> Optional[VectorLike]:
        """
        Get the primary embedding vector for a glyph.
        
        This returns the main/cortex embedding for the glyph,
        used for similarity computations.
        
        Args:
            glyph_id: The glyph identifier
            
        Returns:
            The embedding vector (SDK Vector, numpy array, or list),
            or None if embedding is not available.
            
        Raises:
            KeyError: If glyph not found
        """
        ...
    
    def get_embedding_for_scope(
        self,
        glyph_id: GlyphId,
        layer: Optional[str] = None,
        segment: Optional[str] = None
    ) -> Optional[VectorLike]:
        """
        Get embedding for a specific scope (layer/segment).
        
        SDK glyphs have hierarchical structure with layers and segments,
        each with their own embeddings. This method retrieves the
        embedding at a specific level.
        
        Args:
            glyph_id: The glyph identifier
            layer: Optional layer name (e.g., "identity", "attributes")
            segment: Optional segment name within the layer
            
        Returns:
            The scoped embedding vector, or None if:
            - The scope is not supported by this storage
            - The layer/segment doesn't exist
            - No embedding at that scope
            
        Note:
            Database storage typically returns None for scoped queries
            since runtime glyphs don't have hierarchical structure.
        """
        ...
    
    def compute_similarity(self, v1: VectorLike, v2: VectorLike) -> float:
        """
        Compute similarity between two vectors.
        
        Args:
            v1: First vector (SDK Vector, numpy array, or list)
            v2: Second vector (SDK Vector, numpy array, or list)
            
        Returns:
            Similarity score, typically in range [0, 1] where:
            - 1.0 means identical vectors
            - 0.0 means orthogonal/dissimilar vectors
            
        Note:
            Implementations should handle different vector types
            (SDK Vector with .data attribute, numpy arrays, lists).
        """
        ...
    
    def get_glyph_attribute(self, glyph_id: GlyphId, attribute: str) -> Any:
        """
        Get an attribute value from a glyph.
        
        This retrieves metadata or properties from a glyph,
        used for filtering and predicate evaluation.
        
        Args:
            glyph_id: The glyph identifier
            attribute: The attribute name to retrieve
            
        Returns:
            The attribute value, or None if not found.
            
        Note:
            Implementations should check:
            1. Direct object attributes (getattr)
            2. attributes dict (SDK glyphs)
            3. metadata dict (runtime glyphs)
        """
        ...


class InMemoryGlyphStorage:
    """
    Storage implementation for in-memory SDK Glyph objects.
    
    This is the default storage used when ExecutionContext is created
    with a glyphs dict. It supports the full SDK Glyph structure
    including layers, segments, and hierarchical embeddings.
    
    Example:
        >>> from glyphh.gql.storage import InMemoryGlyphStorage
        >>> storage = InMemoryGlyphStorage(
        ...     glyphs={"g1": glyph1, "g2": glyph2},
        ...     similarity_calculator=calc
        ... )
        >>> embedding = storage.get_embedding("g1")
    """
    
    def __init__(
        self,
        glyphs: Dict[str, Any],
        similarity_calculator: Optional[Any] = None
    ):
        """
        Initialize in-memory storage.
        
        Args:
            glyphs: Dictionary mapping glyph_id to SDK Glyph objects
            similarity_calculator: Optional SimilarityCalculator for
                computing similarity. Falls back to cosine similarity.
        """
        self._glyphs = glyphs
        self._similarity_calculator = similarity_calculator
    
    def list_glyphs(self) -> Dict[str, Any]:
        """List all glyphs."""
        return self._glyphs
    
    def get_glyph(self, glyph_id: GlyphId) -> Any:
        """Get a glyph by ID."""
        if glyph_id not in self._glyphs:
            raise KeyError(f"Glyph not found: {glyph_id}")
        return self._glyphs[glyph_id]
    
    def has_glyph(self, glyph_id: GlyphId) -> bool:
        """Check if a glyph exists."""
        return glyph_id in self._glyphs
    
    def get_embedding(self, glyph_id: GlyphId) -> Optional[VectorLike]:
        """Get the primary embedding for a glyph."""
        glyph = self.get_glyph(glyph_id)
        
        # SDK glyphs use cortex or global_cortex
        if hasattr(glyph, 'cortex') and glyph.cortex is not None:
            return glyph.cortex
        if hasattr(glyph, 'global_cortex') and glyph.global_cortex is not None:
            return glyph.global_cortex
        
        return None
    
    def get_embedding_for_scope(
        self,
        glyph_id: GlyphId,
        layer: Optional[str] = None,
        segment: Optional[str] = None
    ) -> Optional[VectorLike]:
        """Get embedding for a specific scope."""
        # No scope = primary embedding
        if layer is None:
            return self.get_embedding(glyph_id)
        
        glyph = self.get_glyph(glyph_id)
        
        # Navigate SDK glyph hierarchy
        if not hasattr(glyph, 'layers'):
            return None
        
        # Find the layer
        layer_obj = None
        for l in glyph.layers:
            if hasattr(l, 'name') and l.name == layer:
                layer_obj = l
                break
        
        if layer_obj is None:
            return None
        
        # No segment = layer embedding
        if segment is None:
            return layer_obj.cortex if hasattr(layer_obj, 'cortex') else None
        
        # Find the segment
        if not hasattr(layer_obj, 'segments'):
            return None
        
        for seg in layer_obj.segments:
            if hasattr(seg, 'name') and seg.name == segment:
                return seg.cortex if hasattr(seg, 'cortex') else None
        
        return None
    
    def compute_similarity(self, v1: VectorLike, v2: VectorLike) -> float:
        """Compute similarity between two vectors."""
        if self._similarity_calculator:
            return self._similarity_calculator.compute_similarity(v1, v2)
        
        # Fallback to cosine similarity
        try:
            from glyphh.core.ops import cosine_similarity
            
            # Handle SDK Vector objects
            data1 = v1.data if hasattr(v1, 'data') else v1
            data2 = v2.data if hasattr(v2, 'data') else v2
            
            return cosine_similarity(data1, data2)
        except (ImportError, Exception):
            # Ultimate fallback using numpy
            try:
                import numpy as np
                
                arr1 = np.array(v1.data if hasattr(v1, 'data') else v1)
                arr2 = np.array(v2.data if hasattr(v2, 'data') else v2)
                
                norm1 = np.linalg.norm(arr1)
                norm2 = np.linalg.norm(arr2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return float(np.dot(arr1, arr2) / (norm1 * norm2))
            except Exception:
                return 0.0
    
    def get_glyph_attribute(self, glyph_id: GlyphId, attribute: str) -> Any:
        """
        Get an attribute value from a glyph.
        
        Supports dotted path notation for nested attributes (e.g., "vehicle.identity.make").
        """
        glyph = self.get_glyph(glyph_id)
        
        # Check direct object attribute (non-dotted only)
        if '.' not in attribute and hasattr(glyph, attribute):
            return getattr(glyph, attribute)
        
        # Check attributes dict (SDK glyphs) with dotted path support
        if hasattr(glyph, 'attributes') and isinstance(glyph.attributes, dict):
            result = self._get_nested_value(glyph.attributes, attribute)
            if result is not None:
                return result
        
        # Check metadata dict with dotted path support
        if hasattr(glyph, 'metadata') and isinstance(glyph.metadata, dict):
            result = self._get_nested_value(glyph.metadata, attribute)
            if result is not None:
                return result
        
        return None
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """
        Get a value from a nested dict using dotted path notation.
        
        Args:
            data: The dictionary to search
            path: Dotted path like "vehicle.identity.make"
            
        Returns:
            The value at the path, or None if not found.
        """
        parts = path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
