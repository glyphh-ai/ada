"""
Database Glyph Storage for Glyphh Runtime.

Implements the GlyphStorageProtocol for database-stored glyphs, enabling
the GQL executor to work with runtime glyphs stored in PostgreSQL.

This implementation differs from InMemoryGlyphStorage in that:
- Glyphs are pre-fetched GlyphResponse objects (not SDK Glyph objects)
- Embeddings are stored separately in a dict (not in glyph.cortex)
- Layer/segment hierarchical structure is not supported
"""

import logging
from typing import Any, Dict, List, Optional

from domains.models.schemas import GlyphResponse
from shared.similarity_service import SimilarityService

logger = logging.getLogger(__name__)


class DatabaseGlyphStorage:
    """
    Storage implementation for database-stored glyphs in the runtime.
    
    This class implements the GlyphStorageProtocol interface, allowing the
    GQL executor to work with glyphs stored in the PostgreSQL database.
    
    Unlike SDK glyphs which have hierarchical structure (layers, segments),
    database glyphs are flat records with concept_text, metadata, and
    embeddings stored separately.
    
    Example:
        >>> from domains.gql.storage import DatabaseGlyphStorage
        >>> storage = DatabaseGlyphStorage(
        ...     org_id="org-123",
        ...     model_id="model-456",
        ...     glyphs=glyph_responses,
        ...     embeddings={"glyph-id": [0.1, 0.2, ...]},
        ...     similarity_service=sim_service
        ... )
        >>> embedding = storage.get_embedding("glyph-id")
    
    Requirements:
        - 5.1: Implements GlyphStorageProtocol interface
        - 5.3: get_embedding() returns from embeddings dict
        - 5.4: compute_similarity() uses SimilarityService
        - 5.6: get_embedding_for_scope() returns None for layer/segment
    """
    
    def __init__(
        self,
        org_id: str,
        model_id: str,
        glyphs: List[GlyphResponse],
        embeddings: Dict[str, List[float]],
        similarity_service: Optional[SimilarityService] = None,
    ):
        """
        Initialize database glyph storage.
        
        Args:
            org_id: Organization ID for scoping
            model_id: Model ID for scoping
            glyphs: List of GlyphResponse objects from the database
            embeddings: Dictionary mapping glyph_id (str) to embedding vectors
            similarity_service: Optional SimilarityService for computing similarity.
                               Falls back to numpy cosine similarity if not provided.
        """
        self._org_id = org_id
        self._model_id = model_id
        # Convert list to dict keyed by glyph ID (as string)
        self._glyphs: Dict[str, GlyphResponse] = {str(g.id): g for g in glyphs}
        self._embeddings = embeddings
        self._similarity_service = similarity_service
        
        logger.debug(
            f"DatabaseGlyphStorage initialized: org={org_id}, model={model_id}, "
            f"glyphs={len(self._glyphs)}, embeddings={len(embeddings)}"
        )
    
    @property
    def org_id(self) -> str:
        """Get the organization ID."""
        return self._org_id
    
    @property
    def model_id(self) -> str:
        """Get the model ID."""
        return self._model_id
    
    def list_glyphs(self) -> Dict[str, Any]:
        """
        List all glyphs in storage.
        
        Returns:
            Dictionary mapping glyph_id (str) to GlyphResponse object.
        """
        return self._glyphs
    
    def get_glyph(self, glyph_id: str) -> GlyphResponse:
        """
        Get a glyph by ID.
        
        Args:
            glyph_id: The glyph identifier (string)
            
        Returns:
            The GlyphResponse object
            
        Raises:
            KeyError: If glyph not found
        """
        glyph_id_str = str(glyph_id)
        if glyph_id_str not in self._glyphs:
            raise KeyError(f"Glyph not found: {glyph_id}")
        return self._glyphs[glyph_id_str]
    
    def has_glyph(self, glyph_id: str) -> bool:
        """
        Check if a glyph exists in storage.
        
        Args:
            glyph_id: The glyph identifier (string)
            
        Returns:
            True if glyph exists, False otherwise
        """
        return str(glyph_id) in self._glyphs
    
    def get_embedding(self, glyph_id: str) -> Optional[List[float]]:
        """
        Get the primary embedding vector for a glyph.
        
        Retrieves the embedding from the embeddings dict passed at construction.
        
        Args:
            glyph_id: The glyph identifier
            
        Returns:
            The embedding vector as a list of floats, or None if not available.
        
        Requirements:
            - 5.3: Returns embedding from the embeddings dict
        """
        return self._embeddings.get(str(glyph_id))
    
    def get_embedding_for_scope(
        self,
        glyph_id: str,
        layer: Optional[str] = None,
        segment: Optional[str] = None,
    ) -> Optional[List[float]]:
        """
        Get embedding for a specific scope (layer/segment).
        
        Database glyphs don't have hierarchical structure (layers/segments),
        so this method returns None when layer or segment is specified.
        When no scope is specified, returns the primary embedding.
        
        Args:
            glyph_id: The glyph identifier
            layer: Optional layer name (not supported for database glyphs)
            segment: Optional segment name (not supported for database glyphs)
            
        Returns:
            The primary embedding if no scope specified, None otherwise.
        
        Requirements:
            - 5.6: Returns None for layer/segment since runtime glyphs
                   don't have hierarchical structure
        """
        # Database glyphs don't have hierarchical structure
        # Return None if layer or segment is specified
        if layer is not None or segment is not None:
            return None
        
        # No scope = primary embedding
        return self.get_embedding(glyph_id)
    
    def compute_similarity(self, v1: Any, v2: Any) -> float:
        """
        Compute similarity between two vectors.
        
        Uses SimilarityService if available, otherwise falls back to
        numpy cosine similarity.
        
        Args:
            v1: First vector (list of floats or numpy array)
            v2: Second vector (list of floats or numpy array)
            
        Returns:
            Similarity score, typically in range [-1, 1] for cosine similarity.
        
        Requirements:
            - 5.4: Uses SimilarityService for similarity computation
        """
        # Convert to list if needed (handle SDK Vector objects)
        v1_list = v1.data if hasattr(v1, 'data') else v1
        v2_list = v2.data if hasattr(v2, 'data') else v2
        
        if self._similarity_service:
            return self._similarity_service.compute_similarity(v1_list, v2_list)
        
        # Fallback to numpy cosine similarity
        try:
            import numpy as np
            
            v1_arr = np.array(v1_list) if not isinstance(v1_list, np.ndarray) else v1_list
            v2_arr = np.array(v2_list) if not isinstance(v2_list, np.ndarray) else v2_list
            
            norm1 = np.linalg.norm(v1_arr)
            norm2 = np.linalg.norm(v2_arr)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(np.dot(v1_arr, v2_arr) / (norm1 * norm2))
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.0
    
    def get_glyph_attribute(self, glyph_id: str, attribute: str) -> Any:
        """
        Get an attribute value from a glyph.
        
        Checks direct GlyphResponse attributes first, then metadata dict.
        
        Args:
            glyph_id: The glyph identifier
            attribute: The attribute name to retrieve
            
        Returns:
            The attribute value, or None if not found.
        """
        glyph = self.get_glyph(glyph_id)
        
        # Check direct GlyphResponse attributes
        if hasattr(glyph, attribute):
            return getattr(glyph, attribute)
        
        # Check metadata dict
        if hasattr(glyph, 'metadata') and isinstance(glyph.metadata, dict):
            if attribute in glyph.metadata:
                return glyph.metadata[attribute]
        
        return None
