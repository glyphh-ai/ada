"""
Glyph CRUD API Routes for Glyphh Runtime.

Endpoints for creating, reading, updating, and deleting glyphs.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.storage import GlyphStorage
from domains.models.schemas import CreateGlyphResponse, GlyphResponse
from domains.resources.manager import ResourceManager
from infrastructure.database import get_db
from shared.exceptions import GlyphNotFoundException, NamespaceQuotaExceededException, ValidationException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/{namespace}/glyphs", tags=["glyphs"])


# Request/Response Models
class CreateGlyphRequest(BaseModel):
    """Request to create a glyph."""
    concept: str = Field(..., description="Concept text to encode")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class BatchCreateGlyphRequest(BaseModel):
    """Request to create multiple glyphs."""
    concepts: List[str] = Field(..., description="List of concept texts")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Shared metadata")


class UpdateGlyphRequest(BaseModel):
    """Request to update a glyph."""
    concept: Optional[str] = Field(None, description="New concept text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="New metadata")


class GlyphListResponse(BaseModel):
    """Response for glyph listing."""
    glyphs: List[GlyphResponse]
    total: int
    limit: int
    offset: int


# Dependency injection
async def get_storage(db: AsyncSession = Depends(get_db)) -> GlyphStorage:
    """Get glyph storage instance."""
    return GlyphStorage(db)


async def get_resource_manager() -> ResourceManager:
    """Get resource manager instance."""
    from main import resource_manager
    if resource_manager is None:
        raise HTTPException(status_code=503, detail="Resource manager not initialized")
    return resource_manager


async def get_encoder(namespace: str):
    """Get encoder for namespace."""
    from main import model_manager
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    model = await model_manager.get_model(namespace)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {namespace}")
    
    return model.encoder


# Endpoints
@router.post("", response_model=CreateGlyphResponse)
async def create_glyph(
    namespace: str,
    request: CreateGlyphRequest,
    storage: GlyphStorage = Depends(get_storage),
    resource_mgr: ResourceManager = Depends(get_resource_manager),
) -> CreateGlyphResponse:
    """
    Create a new glyph.
    
    Encodes the concept text and stores the resulting glyph.
    """
    try:
        # Check quota before creating
        await resource_mgr.enforce_quota(namespace, additional_glyphs=1)
        
        # Get encoder for namespace
        encoder = await get_encoder(namespace)
        
        # Encode concept
        embedding = encoder.encode_text(request.concept)
        
        # Store glyph
        result = await storage.create_glyph(
            namespace=namespace,
            concept_text=request.concept,
            embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
            metadata=request.metadata,
        )
        
        # Invalidate resource cache
        resource_mgr.invalidate_cache(namespace)
        
        return result
        
    except NamespaceQuotaExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create glyph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=Dict[str, Any])
async def create_glyphs_batch(
    namespace: str,
    request: BatchCreateGlyphRequest,
    storage: GlyphStorage = Depends(get_storage),
    resource_mgr: ResourceManager = Depends(get_resource_manager),
) -> Dict[str, Any]:
    """
    Create multiple glyphs in a batch.
    
    Encodes all concepts and stores them in a single transaction.
    """
    try:
        # Check quota before creating batch
        await resource_mgr.enforce_quota(namespace, additional_glyphs=len(request.concepts))
        
        encoder = await get_encoder(namespace)
        
        results = []
        errors = []
        
        for i, concept in enumerate(request.concepts):
            try:
                embedding = encoder.encode_text(concept)
                result = await storage.create_glyph(
                    namespace=namespace,
                    concept_text=concept,
                    embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                    metadata=request.metadata,
                )
                results.append({
                    "index": i,
                    "glyph_id": str(result.glyph_id),
                    "status": "created"
                })
            except Exception as e:
                errors.append({
                    "index": i,
                    "concept": concept[:50],
                    "error": str(e)
                })
        
        # Invalidate resource cache
        resource_mgr.invalidate_cache(namespace)
        
        return {
            "created": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }
        
    except NamespaceQuotaExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Batch create failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{glyph_id}", response_model=GlyphResponse)
async def get_glyph(
    namespace: str,
    glyph_id: UUID,
    storage: GlyphStorage = Depends(get_storage),
) -> GlyphResponse:
    """
    Get a glyph by ID.
    
    Returns the glyph details including concept text and metadata.
    """
    try:
        return await storage.get_glyph(namespace, glyph_id)
    except GlyphNotFoundException:
        raise HTTPException(status_code=404, detail=f"Glyph not found: {glyph_id}")


@router.put("/{glyph_id}", response_model=GlyphResponse)
async def update_glyph(
    namespace: str,
    glyph_id: UUID,
    request: UpdateGlyphRequest,
    storage: GlyphStorage = Depends(get_storage),
) -> GlyphResponse:
    """
    Update a glyph.
    
    Updates concept text and/or metadata. If concept is updated,
    the embedding is re-encoded.
    """
    try:
        embedding = None
        if request.concept:
            encoder = await get_encoder(namespace)
            embedding = encoder.encode_text(request.concept)
            embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        
        return await storage.update_glyph(
            namespace=namespace,
            glyph_id=glyph_id,
            concept_text=request.concept,
            embedding=embedding,
            metadata=request.metadata,
        )
    except GlyphNotFoundException:
        raise HTTPException(status_code=404, detail=f"Glyph not found: {glyph_id}")
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{glyph_id}")
async def delete_glyph(
    namespace: str,
    glyph_id: UUID,
    storage: GlyphStorage = Depends(get_storage),
) -> Dict[str, str]:
    """
    Delete a glyph.
    
    Removes the glyph and all associated edges.
    """
    deleted = await storage.delete_glyph(namespace, glyph_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Glyph not found: {glyph_id}")
    
    return {"status": "deleted", "glyph_id": str(glyph_id)}


@router.get("", response_model=GlyphListResponse)
async def list_glyphs(
    namespace: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    storage: GlyphStorage = Depends(get_storage),
) -> GlyphListResponse:
    """
    List glyphs in a namespace.
    
    Returns paginated list of glyphs.
    """
    glyphs = await storage.list_glyphs(namespace, limit=limit, offset=offset)
    total = await storage.count_glyphs(namespace)
    
    return GlyphListResponse(
        glyphs=glyphs,
        total=total,
        limit=limit,
        offset=offset,
    )
