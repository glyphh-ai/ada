"""
Edge Generator Service for Glyphh Runtime.

Generates and caches edges between glyphs using the SDK's EdgeGenerator.
Supports eager, lazy, and on-demand edge generation strategies.

Updated to use SimilarityService for consistent similarity calculations.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Edge, Glyph
from shared.exceptions import GlyphNotFoundException
from shared.similarity_service import SimilarityService

logger = logging.getLogger(__name__)


class EdgeGenerationStrategy(str, Enum):
    """Strategy for edge generation."""
    EAGER = "eager"      # Generate all edges immediately on glyph creation
    LAZY = "lazy"        # Generate edges on first query that needs them
    ON_DEMAND = "on_demand"  # Only generate edges when explicitly requested


class EdgeGeneratorService:
    """
    Service for generating and caching edges between glyphs.
    
    Uses the SDK's EdgeGenerator to compute edges and stores them in
    PostgreSQL for efficient retrieval during queries.
    
    Responsibilities:
    - Generate spatial edges (4 types) for new glyphs
    - Generate temporal edges (4 types) between glyph versions
    - Cache edges in database with TTL tracking
    - Refresh stale edges based on TTL
    - Delete edges when glyphs are removed
    """
    
    # Default TTL for cached edges (7 days)
    DEFAULT_EDGE_TTL = timedelta(days=7)
    
    # Edge types from SDK
    SPATIAL_EDGE_TYPES = [
        "neural_cortex",
        "neural_layer", 
        "neural_segment",
        "neural_role"
    ]
    
    TEMPORAL_EDGE_TYPES = [
        "temporal_cortex",
        "temporal_layer",
        "temporal_segment",
        "temporal_role"
    ]
    
    def __init__(
        self,
        session: AsyncSession,
        sdk_edge_generator: Optional[Any] = None,
        similarity_service: Optional[SimilarityService] = None,
    ):
        """
        Initialize EdgeGeneratorService.
        
        Args:
            session: Async SQLAlchemy session
            sdk_edge_generator: SDK's EdgeGenerator instance (optional, for testing)
            similarity_service: SimilarityService for computing edge weights
        """
        self._session = session
        self._sdk_generator = sdk_edge_generator
        self._similarity_service = similarity_service or SimilarityService()

    def set_sdk_generator(self, sdk_edge_generator: Any) -> None:
        """
        Set the SDK EdgeGenerator instance.
        
        Args:
            sdk_edge_generator: SDK's EdgeGenerator instance
        """
        self._sdk_generator = sdk_edge_generator
    
    def set_similarity_service(self, similarity_service: SimilarityService) -> None:
        """
        Set the SimilarityService instance.
        
        Args:
            similarity_service: SimilarityService for computing edge weights
        """
        self._similarity_service = similarity_service
    
    async def generate_edges_for_glyph(
        self,
        org_id: str,
        model_id: str,
        glyph_id: UUID,
        glyph_data: Dict[str, Any],
        strategy: EdgeGenerationStrategy = EdgeGenerationStrategy.EAGER,
        ttl: Optional[timedelta] = None,
    ) -> List[UUID]:
        """Generate edges for a glyph based on the specified strategy."""
        if ttl is None:
            ttl = self.DEFAULT_EDGE_TTL
        
        expires_at = datetime.utcnow() + ttl
        
        if strategy == EdgeGenerationStrategy.LAZY:
            logger.debug(f"Lazy edge generation for glyph {glyph_id}")
            return []
        
        if strategy == EdgeGenerationStrategy.ON_DEMAND:
            logger.debug(f"On-demand edge generation for glyph {glyph_id}")
            return []
        
        return await self._generate_spatial_edges(
            org_id=org_id,
            model_id=model_id,
            source_glyph_id=glyph_id,
            source_embedding=glyph_data.get("embedding", []),
            expires_at=expires_at,
        )
    
    async def _generate_spatial_edges(
        self,
        org_id: str,
        model_id: str,
        source_glyph_id: UUID,
        source_embedding: List[float],
        expires_at: datetime,
        target_glyph_ids: Optional[List[UUID]] = None,
    ) -> List[UUID]:
        """Generate spatial edges from source glyph to target glyphs."""
        if target_glyph_ids is None:
            result = await self._session.execute(
                select(Glyph).where(
                    Glyph.org_id == org_id,
                    Glyph.model_id == model_id,
                    Glyph.id != source_glyph_id,
                )
            )
            target_glyphs = result.scalars().all()
        else:
            result = await self._session.execute(
                select(Glyph).where(
                    Glyph.org_id == org_id,
                    Glyph.model_id == model_id,
                    Glyph.id.in_(target_glyph_ids),
                )
            )
            target_glyphs = result.scalars().all()
        
        if not target_glyphs:
            logger.debug(f"No target glyphs found for edge generation in org={org_id}, model={model_id}")
            return []
        
        created_edge_ids = []
        
        for target_glyph in target_glyphs:
            similarity = self._similarity_service.compute_similarity(
                source_embedding,
                target_glyph.embedding
            )
            
            edge = Edge(
                org_id=org_id,
                model_id=model_id,
                source_glyph_id=source_glyph_id,
                target_glyph_id=target_glyph.id,
                edge_type="neural_cortex",
                weight=similarity,
                metadata={"generated_at": datetime.utcnow().isoformat()},
                expires_at=expires_at,
            )
            self._session.add(edge)
            created_edge_ids.append(edge.id)
        
        await self._session.flush()
        
        logger.info(
            f"Generated {len(created_edge_ids)} spatial edges for glyph {source_glyph_id} "
            f"in org={org_id}, model={model_id}"
        )
        
        return created_edge_ids
    
    async def generate_temporal_edges(
        self,
        org_id: str,
        model_id: str,
        glyph_v1_id: UUID,
        glyph_v2_id: UUID,
        ttl: Optional[timedelta] = None,
    ) -> List[UUID]:
        """Generate temporal edges between two versions of a glyph."""
        if ttl is None:
            ttl = self.DEFAULT_EDGE_TTL
        
        expires_at = datetime.utcnow() + ttl
        
        result = await self._session.execute(
            select(Glyph).where(
                Glyph.org_id == org_id,
                Glyph.model_id == model_id,
                Glyph.id.in_([glyph_v1_id, glyph_v2_id]),
            )
        )
        glyphs = {g.id: g for g in result.scalars().all()}
        
        if glyph_v1_id not in glyphs:
            raise GlyphNotFoundException(str(glyph_v1_id), org_id, model_id)
        if glyph_v2_id not in glyphs:
            raise GlyphNotFoundException(str(glyph_v2_id), org_id, model_id)
        
        glyph_v1 = glyphs[glyph_v1_id]
        glyph_v2 = glyphs[glyph_v2_id]
        
        # Compute temporal delta (change magnitude)
        delta_magnitude = self._compute_temporal_delta_magnitude(
            glyph_v1.embedding,
            glyph_v2.embedding
        )
        
        # Create temporal_cortex edge
        edge = Edge(
            org_id=org_id,
            model_id=model_id,
            source_glyph_id=glyph_v1_id,
            target_glyph_id=glyph_v2_id,
            edge_type="temporal_cortex",
            weight=delta_magnitude,
            metadata={
                "generated_at": datetime.utcnow().isoformat(),
                "v1_created_at": glyph_v1.created_at.isoformat(),
                "v2_created_at": glyph_v2.created_at.isoformat(),
            },
            expires_at=expires_at,
        )
        self._session.add(edge)
        await self._session.flush()
        
        logger.info(
            f"Generated temporal edge from {glyph_v1_id} to {glyph_v2_id} "
            f"in org={org_id}, model={model_id} (delta: {delta_magnitude:.4f})"
        )
        
        return [edge.id]

    async def refresh_stale_edges(
        self,
        org_id: str,
        model_id: str,
        ttl: Optional[timedelta] = None,
    ) -> int:
        """
        Refresh edges that have expired or are about to expire.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            ttl: New TTL for refreshed edges
            
        Returns:
            Number of edges refreshed
        """
        if ttl is None:
            ttl = self.DEFAULT_EDGE_TTL
        
        now = datetime.utcnow()
        new_expires_at = now + ttl
        
        # Find stale edges (expired or expiring within 1 hour)
        stale_threshold = now + timedelta(hours=1)
        
        result = await self._session.execute(
            select(Edge).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
                Edge.expires_at.isnot(None),
                Edge.expires_at < stale_threshold,
            )
        )
        stale_edges = result.scalars().all()
        
        if not stale_edges:
            logger.debug(f"No stale edges found in org={org_id}, model={model_id}")
            return 0
        
        # Group edges by source glyph for batch processing
        edges_by_source: Dict[UUID, List[Edge]] = {}
        for edge in stale_edges:
            if edge.source_glyph_id not in edges_by_source:
                edges_by_source[edge.source_glyph_id] = []
            edges_by_source[edge.source_glyph_id].append(edge)
        
        refreshed_count = 0
        
        for source_id, edges in edges_by_source.items():
            # Get source glyph
            source_result = await self._session.execute(
                select(Glyph).where(Glyph.id == source_id)
            )
            source_glyph = source_result.scalar_one_or_none()
            
            if source_glyph is None:
                # Source glyph was deleted, delete orphan edges
                for edge in edges:
                    await self._session.delete(edge)
                continue
            
            # Refresh each edge
            for edge in edges:
                # Get target glyph
                target_result = await self._session.execute(
                    select(Glyph).where(Glyph.id == edge.target_glyph_id)
                )
                target_glyph = target_result.scalar_one_or_none()
                
                if target_glyph is None:
                    # Target glyph was deleted, delete edge
                    await self._session.delete(edge)
                    continue
                
                # Recompute edge weight using SimilarityService
                if edge.edge_type.startswith("neural_"):
                    new_weight = self._similarity_service.compute_similarity(
                        source_glyph.embedding,
                        target_glyph.embedding
                    )
                else:
                    new_weight = self._compute_temporal_delta_magnitude(
                        source_glyph.embedding,
                        target_glyph.embedding
                    )
                
                # Update edge
                edge.weight = new_weight
                edge.expires_at = new_expires_at
                edge.edge_metadata = {
                    **edge.edge_metadata,
                    "refreshed_at": now.isoformat(),
                }
                refreshed_count += 1
        
        await self._session.flush()
        
        logger.info(f"Refreshed {refreshed_count} stale edges in org={org_id}, model={model_id}")
        
        return refreshed_count
    
    async def delete_edges_for_glyph(
        self,
        org_id: str,
        model_id: str,
        glyph_id: UUID,
    ) -> int:
        """
        Delete all edges connected to a glyph (as source or target).
        
        This is called when a glyph is deleted to maintain referential integrity.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            glyph_id: Glyph UUID
            
        Returns:
            Number of edges deleted
        """
        # Delete outgoing edges
        result1 = await self._session.execute(
            delete(Edge).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
                Edge.source_glyph_id == glyph_id,
            )
        )
        
        # Delete incoming edges
        result2 = await self._session.execute(
            delete(Edge).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
                Edge.target_glyph_id == glyph_id,
            )
        )
        
        total_deleted = result1.rowcount + result2.rowcount
        
        if total_deleted > 0:
            logger.debug(
                f"Deleted {total_deleted} edges for glyph {glyph_id} "
                f"in org={org_id}, model={model_id}"
            )
        
        return total_deleted
    
    async def delete_expired_edges(
        self,
        org_id: str,
        model_id: str,
    ) -> int:
        """
        Delete all expired edges for a model.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            
        Returns:
            Number of edges deleted
        """
        now = datetime.utcnow()
        
        result = await self._session.execute(
            delete(Edge).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
                Edge.expires_at.isnot(None),
                Edge.expires_at < now,
            )
        )
        
        deleted_count = result.rowcount
        
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} expired edges in org={org_id}, model={model_id}")
        
        return deleted_count
    
    async def get_edges_for_query(
        self,
        org_id: str,
        model_id: str,
        glyph_id: UUID,
        edge_types: Optional[List[str]] = None,
        generate_if_missing: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get edges for a glyph, optionally generating them if missing (lazy strategy).
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            glyph_id: Glyph UUID
            edge_types: Optional filter by edge types
            generate_if_missing: Whether to generate edges if none exist
            
        Returns:
            List of edge dicts
        """
        # Build query
        query = select(Edge).where(
            Edge.org_id == org_id,
            Edge.model_id == model_id,
            Edge.source_glyph_id == glyph_id,
        )
        
        if edge_types:
            query = query.where(Edge.edge_type.in_(edge_types))
        
        result = await self._session.execute(query)
        edges = result.scalars().all()
        
        # If no edges and lazy generation enabled, generate them
        if not edges and generate_if_missing:
            # Get source glyph
            glyph_result = await self._session.execute(
                select(Glyph).where(
                    Glyph.org_id == org_id,
                    Glyph.model_id == model_id,
                    Glyph.id == glyph_id,
                )
            )
            glyph = glyph_result.scalar_one_or_none()
            
            if glyph:
                await self._generate_spatial_edges(
                    org_id=org_id,
                    model_id=model_id,
                    source_glyph_id=glyph_id,
                    source_embedding=glyph.embedding,
                    expires_at=datetime.utcnow() + self.DEFAULT_EDGE_TTL,
                )
                
                # Re-fetch edges
                result = await self._session.execute(query)
                edges = result.scalars().all()
        
        return [
            {
                "id": edge.id,
                "source_glyph_id": edge.source_glyph_id,
                "target_glyph_id": edge.target_glyph_id,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "metadata": edge.edge_metadata,
                "created_at": edge.created_at,
                "expires_at": edge.expires_at,
            }
            for edge in edges
        ]
    
    async def count_edges(self, org_id: str, model_id: str) -> Dict[str, int]:
        """
        Count edges by type for a model.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            
        Returns:
            Dict mapping edge type to count
        """
        from sqlalchemy import func
        
        result = await self._session.execute(
            select(Edge.edge_type, func.count(Edge.id))
            .where(Edge.org_id == org_id, Edge.model_id == model_id)
            .group_by(Edge.edge_type)
        )
        
        counts = {row[0]: row[1] for row in result.all()}
        
        # Ensure all edge types are represented
        for edge_type in self.SPATIAL_EDGE_TYPES + self.TEMPORAL_EDGE_TYPES:
            if edge_type not in counts:
                counts[edge_type] = 0
        
        return counts

    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _compute_temporal_delta_magnitude(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute the magnitude of change between two embeddings.
        
        Uses Euclidean distance normalized to [0, 1] range.
        
        Args:
            embedding1: Earlier version embedding
            embedding2: Later version embedding
            
        Returns:
            Change magnitude (0.0 = no change, 1.0 = maximum change)
        """
        import numpy as np
        
        v1 = np.array(embedding1, dtype=np.float32)
        v2 = np.array(embedding2, dtype=np.float32)
        
        # Euclidean distance
        distance = np.linalg.norm(v2 - v1)
        
        # Normalize by maximum possible distance (2 * sqrt(dim) for unit vectors)
        # For 768-dim vectors, max distance ≈ 55.4
        max_distance = 2 * np.sqrt(len(embedding1))
        
        # Normalize to [0, 1]
        magnitude = min(1.0, distance / max_distance)
        
        return float(magnitude)
