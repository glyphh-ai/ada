"""
Glyph Storage for Glyphh Runtime.

Handles persistent storage of glyphs in PostgreSQL with pgvector for efficient
similarity search. Provides CRUD operations filtered by org_id and model_id.

Every query filters on org_id as a mandatory security boundary.
model_id scopes the data/vector space within an org.
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
from infrastructure.config import get_settings
from shared.exceptions import (
    GlyphNotFoundException,
    QuotaExceededException,
    ValidationException,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class GlyphStorage:
    """
    Persistent storage for glyphs using PostgreSQL with pgvector.
    
    Every method requires org_id and model_id for tenant isolation.
    org_id is the security boundary — cross-org access is structurally impossible.
    """
    
    def __init__(self, session: AsyncSession):
        self._session = session
    
    async def create_glyph(
        self,
        org_id: str,
        model_id: str,
        concept_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        glyph_id: Optional[UUID] = None,
    ) -> CreateGlyphResponse:
        """
        Store a new glyph in the database.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            concept_text: Original concept text
            embedding: Vector embedding (768-dim)
            metadata: Optional metadata dict
            glyph_id: Optional UUID (generated if not provided)
        """
        # Check local mode glyph limit
        if settings.deployment_mode == "local":
            current_count = await self.count_glyphs(org_id, model_id)
            if current_count >= settings.local_mode_max_glyphs:
                raise QuotaExceededException(
                    org_id=org_id,
                    model_id=model_id,
                    resource="glyphs",
                    limit=settings.local_mode_max_glyphs,
                    current=current_count,
                    message=f"Local mode limit: maximum {settings.local_mode_max_glyphs} glyphs. "
                            f"Upgrade to a production license for unlimited glyphs."
                )

        # Validate embedding dimension against runtime max
        max_dim = settings.max_vector_dimension
        if len(embedding) > max_dim:
            raise ValidationException(
                field="embedding",
                reason=f"Embedding dimension {len(embedding)} exceeds runtime limit of {max_dim}"
            )
        
        if glyph_id is None:
            glyph_id = uuid4()
        
        glyph = Glyph(
            id=glyph_id,
            org_id=org_id,
            model_id=model_id,
            concept_text=concept_text,
            embedding=embedding,
            glyph_metadata=metadata or {},
        )
        
        self._session.add(glyph)
        await self._session.flush()
        
        logger.debug(f"Created glyph {glyph_id} in org={org_id}, model={model_id}")
        
        return CreateGlyphResponse(
            glyph_id=glyph_id,
            org_id=org_id,
            model_id=model_id,
            created_at=glyph.created_at,
        )
    
    async def get_glyph(
        self,
        org_id: str,
        model_id: str,
        glyph_id: UUID,
    ) -> GlyphResponse:
        """Retrieve a glyph by ID, scoped to org and model."""
        result = await self._session.execute(
            select(Glyph).where(
                Glyph.id == glyph_id,
                Glyph.org_id == org_id,
                Glyph.model_id == model_id,
            )
        )
        glyph = result.scalar_one_or_none()
        
        if glyph is None:
            raise GlyphNotFoundException(str(glyph_id), org_id, model_id)
        
        return GlyphResponse(
            id=glyph.id,
            org_id=glyph.org_id,
            model_id=glyph.model_id,
            concept_text=glyph.concept_text,
            metadata=glyph.glyph_metadata,
            created_at=glyph.created_at,
            updated_at=glyph.updated_at,
        )
    
    async def update_glyph(
        self,
        org_id: str,
        model_id: str,
        glyph_id: UUID,
        concept_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GlyphResponse:
        """Update an existing glyph, scoped to org and model."""
        values = {"updated_at": datetime.utcnow()}
        
        if concept_text is not None:
            values["concept_text"] = concept_text
        
        if embedding is not None:
            max_dim = settings.max_vector_dimension
            if len(embedding) > max_dim:
                raise ValidationException(
                    field="embedding",
                    reason=f"Embedding dimension {len(embedding)} exceeds runtime limit of {max_dim}"
                )
            values["embedding"] = embedding
        
        if metadata is not None:
            values["glyph_metadata"] = metadata
        
        result = await self._session.execute(
            update(Glyph)
            .where(
                Glyph.id == glyph_id,
                Glyph.org_id == org_id,
                Glyph.model_id == model_id,
            )
            .values(**values)
            .returning(Glyph)
        )
        glyph = result.scalar_one_or_none()
        
        if glyph is None:
            raise GlyphNotFoundException(str(glyph_id), org_id, model_id)
        
        return GlyphResponse(
            id=glyph.id,
            org_id=glyph.org_id,
            model_id=glyph.model_id,
            concept_text=glyph.concept_text,
            metadata=glyph.glyph_metadata,
            created_at=glyph.created_at,
            updated_at=glyph.updated_at,
        )
    
    async def delete_glyph(
        self,
        org_id: str,
        model_id: str,
        glyph_id: UUID,
    ) -> bool:
        """Delete a glyph and its edges (CASCADE), scoped to org and model."""
        result = await self._session.execute(
            delete(Glyph).where(
                Glyph.id == glyph_id,
                Glyph.org_id == org_id,
                Glyph.model_id == model_id,
            )
        )
        deleted = result.rowcount > 0
        if deleted:
            logger.debug(f"Deleted glyph {glyph_id} from org={org_id}, model={model_id}")
        return deleted
    
    async def similarity_search(
        self,
        org_id: str,
        model_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[GlyphResponse, float]]:
        """Find top-k most similar glyphs using pgvector, scoped to org and model."""
        max_dim = settings.max_vector_dimension
        if len(query_embedding) > max_dim:
            raise ValidationException(
                field="query_embedding",
                reason=f"Query embedding dimension {len(query_embedding)} exceeds runtime limit of {max_dim}"
            )
        
        query = (
            select(
                Glyph,
                (1 - Glyph.embedding.cosine_distance(query_embedding)).label("similarity")
            )
            .where(
                Glyph.org_id == org_id,
                Glyph.model_id == model_id,
            )
            .order_by(Glyph.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        
        if filters:
            for key, value in filters.items():
                query = query.where(Glyph.glyph_metadata[key].astext == str(value))
        
        result = await self._session.execute(query)
        rows = result.all()
        
        return [
            (
                GlyphResponse(
                    id=row.Glyph.id,
                    org_id=row.Glyph.org_id,
                    model_id=row.Glyph.model_id,
                    concept_text=row.Glyph.concept_text,
                    metadata=row.Glyph.glyph_metadata,
                    created_at=row.Glyph.created_at,
                    updated_at=row.Glyph.updated_at,
                ),
                float(row.similarity)
            )
            for row in rows
        ]
    
    async def list_glyphs(
        self,
        org_id: str,
        model_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[GlyphResponse]:
        """List glyphs with pagination, scoped to org and model."""
        result = await self._session.execute(
            select(Glyph)
            .where(
                Glyph.org_id == org_id,
                Glyph.model_id == model_id,
            )
            .order_by(Glyph.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        glyphs = result.scalars().all()
        
        return [
            GlyphResponse(
                id=g.id,
                org_id=g.org_id,
                model_id=g.model_id,
                concept_text=g.concept_text,
                metadata=g.glyph_metadata,
                created_at=g.created_at,
                updated_at=g.updated_at,
            )
            for g in glyphs
        ]
    
    async def count_glyphs(self, org_id: str, model_id: str) -> int:
        """Count glyphs scoped to org and model."""
        result = await self._session.execute(
            select(func.count(Glyph.id)).where(
                Glyph.org_id == org_id,
                Glyph.model_id == model_id,
            )
        )
        return result.scalar() or 0
    
    # =========================================================================
    # Edge Operations
    # =========================================================================
    
    async def create_edge(
        self,
        org_id: str,
        model_id: str,
        source_glyph_id: UUID,
        target_glyph_id: UUID,
        edge_type: str,
        weight: float,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> UUID:
        """Store an edge between glyphs, scoped to org and model."""
        edge_id = uuid4()
        
        edge = Edge(
            id=edge_id,
            org_id=org_id,
            model_id=model_id,
            source_glyph_id=source_glyph_id,
            target_glyph_id=target_glyph_id,
            edge_type=edge_type,
            weight=weight,
            edge_metadata=metadata or {},
            expires_at=expires_at,
        )
        
        self._session.add(edge)
        await self._session.flush()
        
        return edge_id
    
    async def get_edges(
        self,
        org_id: str,
        model_id: str,
        glyph_id: UUID,
        edge_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[Dict[str, Any]]:
        """Retrieve edges for a glyph, scoped to org and model."""
        queries = []
        
        if direction in ("outgoing", "both"):
            q = select(Edge).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
                Edge.source_glyph_id == glyph_id,
            )
            if edge_type:
                q = q.where(Edge.edge_type == edge_type)
            queries.append(q)
        
        if direction in ("incoming", "both"):
            q = select(Edge).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
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
                    "metadata": edge.edge_metadata,
                    "created_at": edge.created_at,
                    "expires_at": edge.expires_at,
                })
        
        return edges
    
    async def delete_edges_for_glyph(
        self,
        org_id: str,
        model_id: str,
        glyph_id: UUID,
    ) -> int:
        """Delete all edges connected to a glyph, scoped to org and model."""
        result1 = await self._session.execute(
            delete(Edge).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
                Edge.source_glyph_id == glyph_id,
            )
        )
        result2 = await self._session.execute(
            delete(Edge).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
                Edge.target_glyph_id == glyph_id,
            )
        )
        return result1.rowcount + result2.rowcount
    
    async def delete_stale_edges(
        self,
        org_id: str,
        model_id: str,
        before: datetime,
    ) -> int:
        """Delete edges that have expired, scoped to org and model."""
        result = await self._session.execute(
            delete(Edge).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
                Edge.expires_at.isnot(None),
                Edge.expires_at < before,
            )
        )
        return result.rowcount
    
    # =========================================================================
    # Model Data Operations
    # =========================================================================
    
    async def delete_model_data(self, org_id: str, model_id: str) -> Tuple[int, int]:
        """
        Delete all glyphs and edges for an org/model.
        
        Returns:
            Tuple of (glyphs_deleted, edges_deleted)
        """
        # Delete edges first (foreign key constraint)
        edge_result = await self._session.execute(
            delete(Edge).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
            )
        )
        edges_deleted = edge_result.rowcount
        
        glyph_result = await self._session.execute(
            delete(Glyph).where(
                Glyph.org_id == org_id,
                Glyph.model_id == model_id,
            )
        )
        glyphs_deleted = glyph_result.rowcount
        
        logger.info(
            f"Deleted model data org={org_id}, model={model_id}: "
            f"{glyphs_deleted} glyphs, {edges_deleted} edges"
        )
        
        return glyphs_deleted, edges_deleted
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_json(self, glyph: Glyph, include_embedding: bool = False) -> Dict[str, Any]:
        """Serialize a glyph to JSON format."""
        data = {
            "id": str(glyph.id),
            "org_id": glyph.org_id,
            "model_id": glyph.model_id,
            "concept_text": glyph.concept_text,
            "metadata": glyph.glyph_metadata,
            "created_at": glyph.created_at.isoformat() + "Z",
            "updated_at": glyph.updated_at.isoformat() + "Z",
        }
        
        if include_embedding:
            embedding_array = np.array(glyph.embedding, dtype=np.float32)
            data["embedding"] = base64.b64encode(embedding_array.tobytes()).decode()
            data["embedding_format"] = "base64_float32"
        
        return data
    
    def from_json(self, data: Dict[str, Any]) -> Tuple[str, List[float], Dict[str, Any]]:
        """
        Deserialize a glyph from JSON format.
        
        Returns:
            Tuple of (concept_text, embedding, metadata)
        """
        if "concept_text" not in data:
            raise ValidationException(
                field="concept_text",
                reason="Missing required field"
            )
        
        concept_text = data["concept_text"]
        metadata = data.get("metadata", {})
        
        embedding = None
        if "embedding" in data:
            embedding_format = data.get("embedding_format", "array")
            
            if embedding_format == "base64_float32":
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
            
            max_dim = settings.max_vector_dimension
            if len(embedding) > max_dim:
                raise ValidationException(
                    field="embedding",
                    reason=f"Embedding dimension {len(embedding)} exceeds runtime limit of {max_dim}"
                )
        
        return concept_text, embedding, metadata
