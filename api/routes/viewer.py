"""
Viewer API Routes for Glyphh Runtime.

Endpoints for 3D visualization data.
All routes scoped by /{org_id}/{model_id}/viewer/...
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.storage import GlyphStorage
from infrastructure.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/viewer", tags=["viewer"])


# =============================================================================
# Response Models
# =============================================================================

class ViewerLayerData(BaseModel):
    """Layer data for viewer."""
    index: int = Field(..., description="Layer index")
    cortex: List[bool] = Field(..., description="Cortex bit vector")


class ViewerGlyphData(BaseModel):
    """Glyph data for 3D viewer."""
    name: str = Field(..., description="Glyph concept text (display name)")
    glyph_id: str = Field(..., description="Glyph UUID")
    node_type: str = Field(default="concept", description="Node type")
    layers: List[ViewerLayerData] = Field(default_factory=list, description="Layer cortex data")
    semantic: Dict[str, Any] = Field(default_factory=dict, description="Semantic metadata")


class ViewerEdge(BaseModel):
    """Edge data for viewer."""
    source: str = Field(..., description="Source glyph name")
    target: str = Field(..., description="Target glyph name")
    weight: float = Field(default=1.0, description="Edge weight")


class ViewerEdges(BaseModel):
    """Grouped edges by type."""
    semantic: List[ViewerEdge] = Field(default_factory=list)
    neural: List[ViewerEdge] = Field(default_factory=list)
    hierarchy: List[ViewerEdge] = Field(default_factory=list)


class ViewerDataResponse(BaseModel):
    """Complete viewer data response."""
    glyphs: List[ViewerGlyphData] = Field(..., description="List of glyphs")
    edges: ViewerEdges = Field(default_factory=ViewerEdges, description="Edges by type")
    total: int = Field(..., description="Total glyph count")


# =============================================================================
# Dependency Injection
# =============================================================================

async def get_storage(db: AsyncSession = Depends(get_db)) -> GlyphStorage:
    return GlyphStorage(db)


# =============================================================================
# Helper Functions
# =============================================================================

def embedding_to_cortex_bits(embedding: List[float], threshold: float = 0.0) -> List[bool]:
    """
    Convert embedding vector to boolean cortex bits.
    
    Uses threshold to determine which dimensions are "active".
    """
    if not embedding:
        return []
    return [v > threshold for v in embedding]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/data", response_model=ViewerDataResponse)
async def get_viewer_data(
    org_id: str,
    model_id: str,
    limit: int = Query(100, ge=1, le=500, description="Max glyphs to return"),
    storage: GlyphStorage = Depends(get_storage),
) -> ViewerDataResponse:
    """
    Get glyph data formatted for 3D viewer visualization.
    
    Returns glyphs with cortex bit vectors. Edges are computed client-side
    based on cortex similarity.
    """
    try:
        # Get glyphs with embeddings
        glyphs, embeddings = await storage.list_glyphs_with_embeddings(
            org_id, model_id, limit=limit
        )
        
        if not glyphs:
            return ViewerDataResponse(glyphs=[], edges=ViewerEdges(), total=0)
        
        # Get hierarchical embeddings for layer data
        glyph_uuids = [g.id for g in glyphs]
        hierarchical = await storage.get_hierarchical_embeddings(
            org_id, model_id, glyph_uuids
        )
        
        # Build viewer glyph data
        viewer_glyphs: List[ViewerGlyphData] = []
        
        for glyph in glyphs:
            glyph_id_str = str(glyph.id)
            
            # Get cortex embedding and convert to bits
            cortex_embedding = embeddings.get(glyph_id_str, [])
            cortex_bits = embedding_to_cortex_bits(cortex_embedding)
            
            # Build layer data from hierarchical embeddings
            layers: List[ViewerLayerData] = []
            if glyph_id_str in hierarchical:
                glyph_hier = hierarchical[glyph_id_str]
                if 'layer' in glyph_hier:
                    for layer_path, layer_embedding in glyph_hier['layer'].items():
                        # Extract layer index from path (e.g., "layer0" -> 0)
                        try:
                            layer_idx = int(layer_path.replace('layer', ''))
                        except ValueError:
                            layer_idx = 0
                        
                        layer_bits = embedding_to_cortex_bits(layer_embedding)
                        layers.append(ViewerLayerData(index=layer_idx, cortex=layer_bits))
            
            # If no layer data, use cortex as layer 0
            if not layers and cortex_bits:
                layers.append(ViewerLayerData(index=0, cortex=cortex_bits))
            
            # Extract semantic metadata
            semantic: Dict[str, Any] = {}
            if glyph.metadata:
                for key, value in glyph.metadata.items():
                    if not key.startswith('_'):
                        semantic[key] = value
            
            viewer_glyphs.append(ViewerGlyphData(
                name=glyph.concept_text,
                glyph_id=glyph_id_str,
                node_type="concept",
                layers=layers,
                semantic=semantic,
            ))
        
        # Edges are computed client-side based on cortex similarity
        edges = ViewerEdges()
        
        total = await storage.count_glyphs(org_id, model_id)
        
        return ViewerDataResponse(
            glyphs=viewer_glyphs,
            edges=edges,
            total=total,
        )
        
    except Exception as e:
        logger.error(f"Failed to get viewer data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
