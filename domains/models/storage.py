"""
Glyph Storage for Glyphh Runtime.

Handles persistent storage of glyphs in PostgreSQL with pgvector for efficient
similarity search. Provides CRUD operations and namespace-filtered queries.
"""

import base64
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Edge, Glyph, ModelConfig
from domains.models.schemas import (
    CreateGlyphResponse,
    GlyphResponse,
    ScoredGlyph,
)
from shared.exceptions import (
    GlyphNotFoundException,
    NamespaceNotFoundException,
    ValidationException,
)

logger = logging.getLogger(__name__)


class GlyphStorage:
    """
    Persistent storage for glyphs using PostgreSQL with pgvector.
    
    Responsibilities:
    - CRUD operations for glyphs
    - Similarity search using pgvector operators
    - Edge management
    - Namespace filtering for multi-model isolation
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize GlyphStorage with database session.
        
        Args:
            session: Async SQLAlchemy session
        """
        self._session = session
    
    async def create_glyph(
        self,
        namespace: str,
        concept_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        glyph_id: Optional[UUID] = None,
    ) -> CreateGlyphResponse:
        """
        Store a new glyph in the database.
        
        Args:
            namespace: Model namespace
            concept_text: Original concept text
            embedding: Vector embedding (768-dim)
            metadata: Optional metadata dict
            glyph_id: Optional UUID (generated if not provided)
            
        Returns:
            CreateGlyphResponse with glyph ID
            
        Raises:
            ValidationException: If embedding dimension is wrong
        """
        # Validate embedding dimension
        if len(embedding) != 768:
            raise ValidationException(
                field="embedding",
                reason=f"Expected 768 dimensions, got {len(embedding)}"
            )
        
        # Generate ID if not provided
        if glyph_id is None:
            glyph_id = uuid4()
        
        # Create glyph record
        glyph = Glyph(
            id=glyph_id,
            namespace=namespace,
            concept_text=concept_text,
            embedding=embedding,
            metadata=metadata or {},
        )
        
        self._session.add(glyph)
        await self._session.flush()
        
        logger.debug(f"Created glyph {glyph_id} in namespace '{namespace}'")
        
        return CreateGlyphResponse(
            glyph_id=glyph_id,
            namespace=namespace,
            created_at=glyph.created_at,
        )
    
    async def get_glyph(
        self,
        namespace: str,
        glyph_id: UUID,
    ) -> GlyphResponse:
        """
        Retrieve a glyph by ID within a namespace.
        
        Args:
            namespace: Model namespace
            glyph_id: Glyph UUID
            
        Returns:
            GlyphResponse
            
        Raises:
            GlyphNotFoundException: If glyph not found
        """
        result = await self._session.execute(
            select(Glyph).where(
                Glyph.id == glyph_id,
                Glyph.namespace == namespace,
            )
        )
        glyph = result.scalar_one_or_none()
        
        if glyph is None:
            raise GlyphNotFoundException(str(glyph_id), namespace)
        
        return GlyphResponse(
            id=glyph.id,
            namespace=glyph.namespace,
            concept_text=glyph.concept_text,
            metadata=glyph.metadata,
            created_at=glyph.created_at,
            updated_at=glyph.updated_at,
        )
    
    async def update_glyph(
        self,
        namespace: str,
        glyph_id: UUID,
        concept_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GlyphResponse:
        """
        Update an existing glyph.
        
        Args:
            namespace: Model namespace
            glyph_id: Glyph UUID
            concept_text: New concept text (optional)
            embedding: New embedding (optional)
            metadata: New metadata (optional)
            
        Returns:
            Updated GlyphResponse
            
        Raises:
            GlyphNotFoundException: If glyph not found
        """
        # Build update values
        values = {"updated_at": datetime.utcnow()}
        
        if concept_text is not None:
            values["concept_text"] = concept_text
        
        if embedding is not None:
            if len(embedding) != 768:
                raise ValidationException(
                    field="embedding",
                    reason=f"Expected 768 dimensions, got {len(embedding)}"
                )
            values["embedding"] = embedding
        
        if metadata is not None:
            values["metadata"] = metadata
        
        # Execute update
        result = await self._session.execute(
            update(Glyph)
            .where(Glyph.id == glyph_id, Glyph.namespace == namespace)
            .values(**values)
            .returning(Glyph)
        )
        glyph = result.scalar_one_or_none()
        
        if glyph is None:
            raise GlyphNotFoundException(str(glyph_id), namespace)
        
        return GlyphResponse(
            id=glyph.id,
            namespace=glyph.namespace,
            concept_text=glyph.concept_text,
            metadata=glyph.metadata,
            created_at=glyph.created_at,
            updated_at=glyph.updated_at,
        )
    
    async def delete_glyph(
        self,
        namespace: str,
        glyph_id: UUID,
    ) -> bool:
        """
        Delete a glyph and its edges.
        
        Args:
            namespace: Model namespace
            glyph_id: Glyph UUID
            
        Returns:
            True if deleted, False if not found
        """
        # Edges are deleted via CASCADE
        result = await self._session.execute(
            delete(Glyph).where(
                Glyph.id == glyph_id,
                Glyph.namespace == namespace,
            )
        )
        
        deleted = result.rowcount > 0
        if deleted:
            logger.debug(f"Deleted glyph {glyph_id} from namespace '{namespace}'")
        
        return deleted
    
    async def similarity_search(
        self,
        namespace: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[GlyphResponse, float]]:
        """
        Find top-k most similar glyphs using pgvector.
        
        Args:
            namespace: Model namespace
            query_embedding: Query vector (768-dim)
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of (GlyphResponse, similarity_score) tuples
        """
        # Validate embedding dimension
        if len(query_embedding) != 768:
            raise ValidationException(
                field="query_embedding",
                reason=f"Expected 768 dimensions, got {len(query_embedding)}"
            )
        
        # Build query with cosine distance
        # pgvector uses <=> for cosine distance (1 - similarity)
        query = (
            select(
                Glyph,
                (1 - Glyph.embedding.cosine_distance(query_embedding)).label("similarity")
            )
            .where(Glyph.namespace == namespace)
            .order_by(Glyph.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        
        # Apply metadata filters if provided
        if filters:
            for key, value in filters.items():
                query = query.where(Glyph.metadata[key].astext == str(value))
        
        result = await self._session.execute(query)
        rows = result.all()
        
        return [
            (
                GlyphResponse(
                    id=row.Glyph.id,
                    namespace=row.Glyph.namespace,
                    concept_text=row.Glyph.concept_text,
                    metadata=row.Glyph.metadata,
                    created_at=row.Glyph.created_at,
                    updated_at=row.Glyph.updated_at,
                ),
                float(row.similarity)
            )
            for row in rows
        ]

    
    async def list_glyphs(
        self,
        namespace: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[GlyphResponse]:
        """
        List glyphs in a namespace with pagination.
        
        Args:
            namespace: Model namespace
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of GlyphResponse
        """
        result = await self._session.execute(
            select(Glyph)
            .where(Glyph.namespace == namespace)
            .order_by(Glyph.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        glyphs = result.scalars().all()
        
        return [
            GlyphResponse(
                id=g.id,
                namespace=g.namespace,
                concept_text=g.concept_text,
                metadata=g.metadata,
                created_at=g.created_at,
                updated_at=g.updated_at,
            )
            for g in glyphs
        ]
    
    async def count_glyphs(self, namespace: str) -> int:
        """Count glyphs in a namespace."""
        result = await self._session.execute(
            select(func.count(Glyph.id)).where(Glyph.namespace == namespace)
        )
        return result.scalar() or 0
    
    # =========================================================================
    # Edge Operations
    # =========================================================================
    
    async def create_edge(
        self,
        namespace: str,
        source_glyph_id: UUID,
        target_glyph_id: UUID,
        edge_type: str,
        weight: float,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> UUID:
        """
        Store an edge between glyphs.
        
        Args:
            namespace: Model namespace
            source_glyph_id: Source glyph UUID
            target_glyph_id: Target glyph UUID
            edge_type: Edge type (similarity, contrast, etc.)
            weight: Edge weight
            metadata: Optional metadata
            expires_at: Optional TTL expiration
            
        Returns:
            Edge UUID
        """
        edge_id = uuid4()
        
        edge = Edge(
            id=edge_id,
            namespace=namespace,
            source_glyph_id=source_glyph_id,
            target_glyph_id=target_glyph_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
            expires_at=expires_at,
        )
        
        self._session.add(edge)
        await self._session.flush()
        
        return edge_id
    
    async def get_edges(
        self,
        namespace: str,
        glyph_id: UUID,
        edge_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve edges for a glyph.
        
        Args:
            namespace: Model namespace
            glyph_id: Glyph UUID
            edge_type: Optional filter by edge type
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            List of edge dicts
        """
        queries = []
        
        if direction in ("outgoing", "both"):
            q = select(Edge).where(
                Edge.namespace == namespace,
                Edge.source_glyph_id == glyph_id,
            )
            if edge_type:
                q = q.where(Edge.edge_type == edge_type)
            queries.append(q)
        
        if direction in ("incoming", "both"):
            q = select(Edge).where(
                Edge.namespace == namespace,
                Edge.target_glyph_id == glyph_id,
            )
            if edge_type:
                q = q.where(Edge.edge_type == edge_type)
            queries.append(q)
        
        edges = []
        for query in queries:
            result = await self._session.execute(query)
            for edge in result.scalars().all():
                edges.append({
                    "id": edge.id,
                    "source_glyph_id": edge.source_glyph_id,
                    "target_glyph_id": edge.target_glyph_id,
                    "edge_type": edge.edge_type,
                    "weight": edge.weight,
                    "metadata": edge.metadata,
                    "created_at": edge.created_at,
                    "expires_at": edge.expires_at,
                })
        
        return edges
    
    async def delete_edges_for_glyph(
        self,
        namespace: str,
        glyph_id: UUID,
    ) -> int:
        """
        Delete all edges connected to a glyph.
        
        Args:
            namespace: Model namespace
            glyph_id: Glyph UUID
            
        Returns:
            Number of edges deleted
        """
        # Delete outgoing edges
        result1 = await self._session.execute(
            delete(Edge).where(
                Edge.namespace == namespace,
                Edge.source_glyph_id == glyph_id,
            )
        )
        
        # Delete incoming edges
        result2 = await self._session.execute(
            delete(Edge).where(
                Edge.namespace == namespace,
                Edge.target_glyph_id == glyph_id,
            )
        )
        
        return result1.rowcount + result2.rowcount
    
    async def delete_stale_edges(
        self,
        namespace: str,
        before: datetime,
    ) -> int:
        """
        Delete edges that have expired.
        
        Args:
            namespace: Model namespace
            before: Delete edges with expires_at before this time
            
        Returns:
            Number of edges deleted
        """
        result = await self._session.execute(
            delete(Edge).where(
                Edge.namespace == namespace,
                Edge.expires_at.isnot(None),
                Edge.expires_at < before,
            )
        )
        return result.rowcount
    
    # =========================================================================
    # Namespace Operations
    # =========================================================================
    
    async def delete_namespace(self, namespace: str) -> Tuple[int, int]:
        """
        Delete all glyphs and edges in a namespace.
        
        Args:
            namespace: Namespace to delete
            
        Returns:
            Tuple of (glyphs_deleted, edges_deleted)
        """
        # Delete edges first (foreign key constraint)
        edge_result = await self._session.execute(
            delete(Edge).where(Edge.namespace == namespace)
        )
        edges_deleted = edge_result.rowcount
        
        # Delete glyphs
        glyph_result = await self._session.execute(
            delete(Glyph).where(Glyph.namespace == namespace)
        )
        glyphs_deleted = glyph_result.rowcount
        
        logger.info(
            f"Deleted namespace '{namespace}': "
            f"{glyphs_deleted} glyphs, {edges_deleted} edges"
        )
        
        return glyphs_deleted, edges_deleted
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_json(self, glyph: Glyph, include_embedding: bool = False) -> Dict[str, Any]:
        """
        Serialize a glyph to JSON format.
        
        Args:
            glyph: Glyph database model
            include_embedding: Whether to include embedding
            
        Returns:
            JSON-serializable dict
        """
        data = {
            "id": str(glyph.id),
            "namespace": glyph.namespace,
            "concept_text": glyph.concept_text,
            "metadata": glyph.metadata,
            "created_at": glyph.created_at.isoformat() + "Z",
            "updated_at": glyph.updated_at.isoformat() + "Z",
        }
        
        if include_embedding:
            # Encode embedding as base64 for efficiency
            embedding_array = np.array(glyph.embedding, dtype=np.float32)
            data["embedding"] = base64.b64encode(embedding_array.tobytes()).decode()
            data["embedding_format"] = "base64_float32"
        
        return data
    
    def from_json(self, data: Dict[str, Any]) -> Tuple[str, List[float], Dict[str, Any]]:
        """
        Deserialize a glyph from JSON format.
        
        Args:
            data: JSON dict with glyph data
            
        Returns:
            Tuple of (concept_text, embedding, metadata)
            
        Raises:
            ValidationException: If JSON is invalid
        """
        # Validate required fields
        if "concept_text" not in data:
            raise ValidationException(
                field="concept_text",
                reason="Missing required field"
            )
        
        concept_text = data["concept_text"]
        metadata = data.get("metadata", {})
        
        # Parse embedding
        embedding = None
        if "embedding" in data:
            embedding_format = data.get("embedding_format", "array")
            
            if embedding_format == "base64_float32":
                # Decode base64
                try:
                    embedding_bytes = base64.b64decode(data["embedding"])
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
                except Exception as e:
                    raise ValidationException(
                        field="embedding",
                        reason=f"Invalid base64 encoding: {e}"
                    )
            elif embedding_format == "array":
                embedding = data["embedding"]
            else:
                raise ValidationException(
                    field="embedding_format",
                    reason=f"Unknown format: {embedding_format}"
                )
            
            # Validate dimension
            if len(embedding) != 768:
                raise ValidationException(
                    field="embedding",
                    reason=f"Expected 768 dimensions, got {len(embedding)}"
                )
        
        return concept_text, embedding, metadata
