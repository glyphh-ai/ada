"""
Database Glyph Storage for Glyphh Runtime.

Implements the GlyphStorageProtocol for database-stored glyphs, enabling
the GQL executor to work with runtime glyphs stored in PostgreSQL.

This implementation differs from InMemoryGlyphStorage in that:
- Glyphs are pre-fetched GlyphResponse objects (not SDK Glyph objects)
- Embeddings are stored separately in a dict (not in glyph.cortex)
- Layer/segment/role hierarchical vectors are stored in glyph_vectors table
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from domains.models.schemas import GlyphResponse
from shared.similarity_service import SimilarityService

logger = logging.getLogger(__name__)


class DatabaseGlyphStorage:
    """
    Storage implementation for database-stored glyphs in the runtime.
    
    This class implements the GlyphStorageProtocol interface, allowing the
    GQL executor to work with glyphs stored in the PostgreSQL database.
    
    Supports hierarchical vector queries via glyph_vectors table for
    layer, segment, and role level similarity searches.
    
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
        - 5.6: get_embedding_for_scope() queries hierarchical vectors
    """
    
    def __init__(
        self,
        org_id: str,
        model_id: str,
        glyphs: List[GlyphResponse],
        embeddings: Dict[str, List[float]],
        similarity_service: Optional[SimilarityService] = None,
        hierarchical_embeddings: Optional[Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        vector_fetcher: Optional[Callable] = None,
        session_factory: Optional[Callable] = None,
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
            hierarchical_embeddings: Optional pre-fetched hierarchical embeddings
                                    {glyph_id: {level: {path: embedding}}}
            vector_fetcher: Optional async callable to fetch hierarchical vectors on demand
            session_factory: Optional async context manager for DB sessions.
                            When provided, find_similar() delegates to pgvector
                            instead of iterating in-memory.
        """
        self._org_id = org_id
        self._model_id = model_id
        # Convert list to dict keyed by glyph ID (as string)
        self._glyphs: Dict[str, GlyphResponse] = {str(g.id): g for g in glyphs}
        self._embeddings = embeddings
        self._similarity_service = similarity_service
        self._hierarchical_embeddings = hierarchical_embeddings or {}
        self._vector_fetcher = vector_fetcher
        self._session_factory = session_factory

        logger.debug(
            f"DatabaseGlyphStorage initialized: org={org_id}, model={model_id}, "
            f"glyphs={len(self._glyphs)}, embeddings={len(embeddings)}, "
            f"hierarchical={len(self._hierarchical_embeddings)}"
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

        Retrieves from the in-memory cache first.  When session_factory is
        available (pgvector mode), fetches from DB on cache miss — this
        avoids pre-loading all embeddings while still supporting
        glyph("uuid") references in GQL.
        """
        cached = self._embeddings.get(str(glyph_id))
        if cached is not None:
            return cached

        # pgvector mode: fetch single embedding from DB on demand
        if self._session_factory is not None:
            try:
                import asyncio
                from concurrent.futures import ThreadPoolExecutor

                async def _fetch():
                    from domains.models.storage import GlyphStorage
                    async with self._session_factory() as session:
                        storage = GlyphStorage(session)
                        return await storage.get_glyph_embedding(
                            org_id=self._org_id,
                            model_id=self._model_id,
                            glyph_id=glyph_id,
                        )

                with ThreadPoolExecutor(max_workers=1) as pool:
                    embedding = pool.submit(asyncio.run, _fetch()).result(timeout=10)
                if embedding is not None:
                    # Cache for future lookups
                    self._embeddings[str(glyph_id)] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Failed to fetch embedding for {glyph_id}: {e}")

        return None
    
    def get_embedding_for_scope(
        self,
        glyph_id: str,
        layer: Optional[str] = None,
        segment: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Optional[List[float]]:
        """
        Get embedding for a specific scope (layer/segment/role).
        
        Queries hierarchical vectors from the glyph_vectors table.
        When no scope is specified, returns the primary (cortex) embedding.
        
        Args:
            glyph_id: The glyph identifier
            layer: Optional layer name (e.g., 'semantic')
            segment: Optional segment name (e.g., 'attributes')
            role: Optional role name (e.g., 'color')
            
        Returns:
            The embedding for the specified scope, or None if not found.
        
        Requirements:
            - 5.6: Queries hierarchical vectors from glyph_vectors table
        """
        glyph_id_str = str(glyph_id)
        
        # No scope = primary (cortex) embedding
        if layer is None and segment is None and role is None:
            return self.get_embedding(glyph_id)
        
        # Build the path based on scope
        if role is not None and segment is not None and layer is not None:
            # Role level: layer.segment.role
            level = 'role'
            path = f"{layer}.{segment}.{role}"
        elif segment is not None and layer is not None:
            # Segment level: layer.segment
            level = 'segment'
            path = f"{layer}.{segment}"
        elif layer is not None:
            # Layer level: layer
            level = 'layer'
            path = layer
        else:
            # Invalid scope combination
            return None
        
        # Check pre-fetched hierarchical embeddings
        if glyph_id_str in self._hierarchical_embeddings:
            glyph_hier = self._hierarchical_embeddings[glyph_id_str]
            if level in glyph_hier and path in glyph_hier[level]:
                return glyph_hier[level][path]

        # pgvector mode: fetch from glyph_vectors table on cache miss
        if self._session_factory is not None:
            try:
                import asyncio
                from concurrent.futures import ThreadPoolExecutor

                async def _fetch():
                    from domains.models.storage import GlyphStorage
                    from uuid import UUID as _UUID
                    gid = glyph_id if isinstance(glyph_id, _UUID) else _UUID(str(glyph_id))
                    async with self._session_factory() as session:
                        storage = GlyphStorage(session)
                        vectors = await storage.get_glyph_vectors_by_level(
                            org_id=self._org_id,
                            model_id=self._model_id,
                            glyph_id=gid,
                            level=level,
                        )
                        return vectors.get(path)

                with ThreadPoolExecutor(max_workers=1) as pool:
                    embedding = pool.submit(asyncio.run, _fetch()).result(timeout=10)
                if embedding is not None:
                    # Cache for future lookups
                    if glyph_id_str not in self._hierarchical_embeddings:
                        self._hierarchical_embeddings[glyph_id_str] = {}
                    if level not in self._hierarchical_embeddings[glyph_id_str]:
                        self._hierarchical_embeddings[glyph_id_str][level] = {}
                    self._hierarchical_embeddings[glyph_id_str][level][path] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Failed to fetch scoped embedding for {glyph_id}/{level}/{path}: {e}")

        return None
    
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
    
    def find_similar(
        self,
        query_vector,
        limit: int = 10,
        threshold: float = 0.0,
        scope_layer: Optional[str] = None,
        scope_segment: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Native pgvector similarity search — avoids loading all vectors into RAM.

        Returns None if no session_factory (falls back to Python loop).
        """
        if self._session_factory is None:
            return None  # signal: not supported, use Python fallback

        import asyncio

        async def _search():
            from domains.models.storage import GlyphStorage
            from infrastructure.config import get_settings

            max_dim = get_settings().resolved_max_vector_dimension
            vec = query_vector.data if hasattr(query_vector, 'data') else list(query_vector)
            if len(vec) < max_dim:
                vec = vec + [0.0] * (max_dim - len(vec))

            async with self._session_factory() as session:
                storage = GlyphStorage(session)

                if scope_layer:
                    raw = await storage.similarity_search_by_level(
                        org_id=self._org_id,
                        model_id=self._model_id,
                        query_embedding=vec,
                        level="layer",
                        path=scope_layer,
                        top_k=limit,
                    )
                    # (glyph_id, path, score) → look up glyph from our metadata cache
                    results = []
                    for glyph_id, _path, score in raw:
                        gid = str(glyph_id)
                        if score < threshold:
                            continue
                        glyph = self._glyphs.get(gid)
                        if glyph is not None:
                            results.append({"glyph_id": gid, "score": score, "glyph": glyph})
                    return results
                else:
                    raw = await storage.similarity_search(
                        org_id=self._org_id,
                        model_id=self._model_id,
                        query_embedding=vec,
                        top_k=limit,
                    )
                    results = []
                    for glyph_resp, score in raw:
                        if score < threshold:
                            continue
                        gid = str(glyph_resp.id)
                        # Use cached glyph if available (has full metadata)
                        glyph = self._glyphs.get(gid, glyph_resp)
                        results.append({"glyph_id": gid, "score": score, "glyph": glyph})
                    return results

        # The GQL executor is sync but DB access is async.
        # Run in a new thread with its own event loop to avoid deadlocking
        # the caller's event loop.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _search())
            return future.result(timeout=30)

    def get_glyph_attribute(self, glyph_id: str, attribute: str) -> Any:
        """
        Get an attribute value from a glyph.
        
        Supports dotted path notation for nested attributes (e.g., "vehicle.identity.make").
        Checks direct GlyphResponse attributes first, then metadata dict.
        
        Args:
            glyph_id: The glyph identifier
            attribute: The attribute name or dotted path to retrieve
            
        Returns:
            The attribute value, or None if not found.
        """
        glyph = self.get_glyph(glyph_id)
        
        # Check direct GlyphResponse attributes (non-dotted only)
        if '.' not in attribute and hasattr(glyph, attribute):
            return getattr(glyph, attribute)
        
        # Check metadata dict with dotted path support
        if hasattr(glyph, 'metadata') and isinstance(glyph.metadata, dict):
            return self._get_nested_value(glyph.metadata, attribute)
        
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
