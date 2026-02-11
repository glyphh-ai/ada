"""
Charts API Routes for Glyphh Runtime.

Endpoints for retrieving aggregated chart and statistics data.
All routes scoped by /{org_id}/{model_id}/charts/...
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Edge, Glyph
from domains.models.storage import GlyphStorage
from infrastructure.database import get_db
from shared.exceptions import GlyphNotFoundException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/charts", tags=["charts"])


# =============================================================================
# Response Models
# =============================================================================

class ChartStatsResponse(BaseModel):
    """Aggregated model statistics for charts."""
    total_glyphs: int = Field(..., description="Total number of glyphs")
    total_edges: int = Field(..., description="Total number of edges")
    avg_similarity: float = Field(..., description="Average pairwise similarity (0-1)")
    avg_layer_similarity: float = Field(0.0, description="Average layer-level similarity (0-1)")
    avg_segment_similarity: float = Field(0.0, description="Average segment-level similarity (0-1)")
    avg_role_similarity: float = Field(0.0, description="Average role-level similarity (0-1)")
    role_coverage: Dict[str, float] = Field(
        default_factory=dict,
        description="Coverage percentage per role"
    )
    computed_at: str = Field(..., description="ISO timestamp of computation")


class DistributionResponse(BaseModel):
    """Similarity distribution histogram."""
    buckets: List[int] = Field(..., description="Count per 0.1 bucket (10 buckets)")
    labels: List[str] = Field(..., description="Bucket labels")
    total_pairs: int = Field(..., description="Total pairs computed")
    layer_index: Optional[int] = Field(None, description="Filter applied")
    segment_id: Optional[str] = Field(None, description="Filter applied")


class NeighborResult(BaseModel):
    """Single neighbor result."""
    glyph_id: str = Field(..., description="Glyph UUID")
    name: str = Field(..., description="Glyph concept text")
    similarity_score: float = Field(..., description="Similarity score (0-1)")


class NeighborsResponse(BaseModel):
    """Nearest neighbors for a glyph."""
    anchor_id: str = Field(..., description="Anchor glyph UUID")
    anchor_name: str = Field(..., description="Anchor glyph concept text")
    neighbors: List[NeighborResult] = Field(..., description="List of neighbors")
    layer_index: Optional[int] = Field(None, description="Layer used for similarity")


# =============================================================================
# Dependency Injection
# =============================================================================

async def get_storage(db: AsyncSession = Depends(get_db)) -> GlyphStorage:
    return GlyphStorage(db)


async def get_db_session(db: AsyncSession = Depends(get_db)) -> AsyncSession:
    return db


# =============================================================================
# Helper Functions
# =============================================================================

def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0
    
    # Ensure same length
    min_len = min(len(vec1), len(vec2))
    v1 = vec1[:min_len]
    v2 = vec2[:min_len]
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    # Clamp to [0, 1] range
    return max(0.0, min(1.0, similarity))


def _compute_level_similarities(
    hierarchical: Dict[str, Dict[str, Dict[str, List[float]]]],
    level: str
) -> List[float]:
    """
    Compute pairwise similarities at a specific hierarchy level.
    
    For each unique path at the given level, computes pairwise similarities
    between all glyphs that have that path, then averages across paths.
    
    Args:
        hierarchical: Nested dict {glyph_id: {level: {path: embedding}}}
        level: 'layer', 'segment', or 'role'
        
    Returns:
        List of similarity scores
    """
    # Group embeddings by path
    path_embeddings: Dict[str, Dict[str, List[float]]] = {}
    
    for glyph_id, levels in hierarchical.items():
        if level not in levels:
            continue
        for path, embedding in levels[level].items():
            if path not in path_embeddings:
                path_embeddings[path] = {}
            path_embeddings[path][glyph_id] = embedding
    
    # Compute pairwise similarities for each path
    all_similarities = []
    
    for path, glyph_embeddings in path_embeddings.items():
        glyph_ids = list(glyph_embeddings.keys())
        if len(glyph_ids) < 2:
            continue
        
        for i in range(len(glyph_ids)):
            for j in range(i + 1, len(glyph_ids)):
                sim = compute_cosine_similarity(
                    glyph_embeddings[glyph_ids[i]],
                    glyph_embeddings[glyph_ids[j]]
                )
                all_similarities.append(sim)
    
    return all_similarities


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/stats", response_model=ChartStatsResponse)
async def get_chart_stats(
    org_id: str,
    model_id: str,
    storage: GlyphStorage = Depends(get_storage),
    db: AsyncSession = Depends(get_db_session),
) -> ChartStatsResponse:
    """
    Get aggregated statistics for charts.
    
    Returns total glyph count, edge count, average similarity at all levels
    (cortex, layer, segment, role), and role coverage.
    """
    try:
        # Count glyphs
        total_glyphs = await storage.count_glyphs(org_id, model_id)
        
        # Count edges
        edge_result = await db.execute(
            select(func.count(Edge.id)).where(
                Edge.org_id == org_id,
                Edge.model_id == model_id,
            )
        )
        total_edges = edge_result.scalar() or 0
        
        # Initialize similarity metrics
        avg_similarity = 0.0
        avg_layer_similarity = 0.0
        avg_segment_similarity = 0.0
        avg_role_similarity = 0.0
        role_coverage: Dict[str, float] = {}
        
        if total_glyphs >= 2:
            # Get glyphs with embeddings for cortex-level similarity
            glyphs, embeddings = await storage.list_glyphs_with_embeddings(
                org_id, model_id, limit=100
            )
            
            if len(glyphs) >= 2:
                # Compute pairwise cortex similarities
                similarities = []
                glyph_ids = list(embeddings.keys())
                
                for i in range(len(glyph_ids)):
                    for j in range(i + 1, len(glyph_ids)):
                        sim = compute_cosine_similarity(
                            embeddings[glyph_ids[i]],
                            embeddings[glyph_ids[j]]
                        )
                        similarities.append(sim)
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                
                # Get hierarchical embeddings for layer/segment/role similarities
                glyph_uuids = [g.id for g in glyphs]
                hierarchical = await storage.get_hierarchical_embeddings(
                    org_id, model_id, glyph_uuids
                )
                
                # Compute layer-level similarities
                layer_sims = _compute_level_similarities(hierarchical, 'layer')
                if layer_sims:
                    avg_layer_similarity = sum(layer_sims) / len(layer_sims)
                
                # Compute segment-level similarities
                segment_sims = _compute_level_similarities(hierarchical, 'segment')
                if segment_sims:
                    avg_segment_similarity = sum(segment_sims) / len(segment_sims)
                
                # Compute role-level similarities
                role_sims = _compute_level_similarities(hierarchical, 'role')
                if role_sims:
                    avg_role_similarity = sum(role_sims) / len(role_sims)
            
            # Compute role coverage from metadata
            role_counts: Dict[str, int] = {}
            for glyph in glyphs:
                if glyph.metadata:
                    for key in glyph.metadata.keys():
                        if not key.startswith("_"):
                            role_counts[key] = role_counts.get(key, 0) + 1
            
            if total_glyphs > 0:
                role_coverage = {
                    role: count / total_glyphs
                    for role, count in role_counts.items()
                }
        
        return ChartStatsResponse(
            total_glyphs=total_glyphs,
            total_edges=total_edges,
            avg_similarity=round(avg_similarity, 4),
            avg_layer_similarity=round(avg_layer_similarity, 4),
            avg_segment_similarity=round(avg_segment_similarity, 4),
            avg_role_similarity=round(avg_role_similarity, 4),
            role_coverage=role_coverage,
            computed_at=datetime.utcnow().isoformat() + "Z",
        )
        
    except Exception as e:
        logger.error(f"Failed to get chart stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distribution", response_model=DistributionResponse)
async def get_distribution(
    org_id: str,
    model_id: str,
    layer_index: Optional[int] = Query(None, description="Filter by layer index"),
    segment_id: Optional[str] = Query(None, description="Filter by segment ID"),
    storage: GlyphStorage = Depends(get_storage),
) -> DistributionResponse:
    """
    Get similarity distribution histogram.
    
    Returns histogram bucket counts for similarity scores in 10 buckets (0.0-1.0).
    """
    try:
        # Get glyphs with embeddings
        glyphs, embeddings = await storage.list_glyphs_with_embeddings(
            org_id, model_id, limit=200
        )
        
        # Initialize histogram buckets (10 buckets: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
        buckets = [0] * 10
        labels = [f"{i/10:.1f}" for i in range(10)]
        total_pairs = 0
        
        if len(glyphs) >= 2:
            glyph_ids = list(embeddings.keys())
            
            # Compute pairwise similarities
            for i in range(len(glyph_ids)):
                for j in range(i + 1, len(glyph_ids)):
                    sim = compute_cosine_similarity(
                        embeddings[glyph_ids[i]],
                        embeddings[glyph_ids[j]]
                    )
                    
                    # Assign to bucket
                    bucket_idx = min(9, int(sim * 10))
                    buckets[bucket_idx] += 1
                    total_pairs += 1
        
        return DistributionResponse(
            buckets=buckets,
            labels=labels,
            total_pairs=total_pairs,
            layer_index=layer_index,
            segment_id=segment_id,
        )
        
    except Exception as e:
        logger.error(f"Failed to get distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neighbors/{glyph_id}", response_model=NeighborsResponse)
async def get_neighbors(
    org_id: str,
    model_id: str,
    glyph_id: str,
    top_k: int = Query(8, ge=1, le=50, description="Number of neighbors to return"),
    layer_index: Optional[int] = Query(None, description="Layer index for similarity"),
    storage: GlyphStorage = Depends(get_storage),
) -> NeighborsResponse:
    """
    Get nearest neighbors for a glyph.
    
    Returns top-k most similar glyphs sorted by similarity score descending.
    """
    try:
        # Get all glyphs with embeddings
        glyphs, embeddings = await storage.list_glyphs_with_embeddings(
            org_id, model_id, limit=500
        )
        
        # Find anchor glyph
        anchor_glyph = None
        anchor_embedding = None
        
        for glyph in glyphs:
            if str(glyph.id) == glyph_id:
                anchor_glyph = glyph
                anchor_embedding = embeddings.get(str(glyph.id))
                break
        
        if anchor_glyph is None:
            raise HTTPException(
                status_code=404,
                detail=f"Glyph not found: {glyph_id}"
            )
        
        if anchor_embedding is None:
            raise HTTPException(
                status_code=404,
                detail=f"Embedding not found for glyph: {glyph_id}"
            )
        
        # Compute similarities to all other glyphs
        neighbors_with_scores = []
        
        for glyph in glyphs:
            if str(glyph.id) == glyph_id:
                continue
            
            other_embedding = embeddings.get(str(glyph.id))
            if other_embedding is None:
                continue
            
            sim = compute_cosine_similarity(anchor_embedding, other_embedding)
            neighbors_with_scores.append((glyph, sim))
        
        # Sort by similarity descending and take top_k
        neighbors_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = neighbors_with_scores[:top_k]
        
        return NeighborsResponse(
            anchor_id=str(anchor_glyph.id),
            anchor_name=anchor_glyph.concept_text,
            neighbors=[
                NeighborResult(
                    glyph_id=str(glyph.id),
                    name=glyph.concept_text,
                    similarity_score=round(score, 4),
                )
                for glyph, score in top_neighbors
            ],
            layer_index=layer_index,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get neighbors: {e}")
        raise HTTPException(status_code=500, detail=str(e))
