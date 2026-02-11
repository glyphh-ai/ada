"""
Glyph CRUD API Routes for Glyphh Runtime.

Endpoints for creating, reading, updating, and deleting glyphs.
All routes scoped by /{org_id}/{model_id}/glyphs/...
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.storage import GlyphStorage
from domains.models.schemas import CreateGlyphResponse, GlyphResponse
from domains.resources.manager import ResourceManager
from infrastructure.database import get_db
from shared.exceptions import GlyphNotFoundException, QuotaExceededException, ValidationException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/glyphs", tags=["glyphs"])


# Request/Response Models
class CreateGlyphRequest(BaseModel):
    concept: str = Field(..., description="Concept text to encode")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class BatchCreateGlyphRequest(BaseModel):
    concepts: List[str] = Field(..., description="List of concept texts")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Shared metadata")


class UpdateGlyphRequest(BaseModel):
    concept: Optional[str] = Field(None, description="New concept text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="New metadata")


class GlyphListResponse(BaseModel):
    glyphs: List[GlyphResponse]
    total: int
    limit: int
    offset: int


# Dependency injection
async def get_storage(db: AsyncSession = Depends(get_db)) -> GlyphStorage:
    return GlyphStorage(db)


async def get_resource_manager() -> ResourceManager:
    from main import resource_manager
    if resource_manager is None:
        raise HTTPException(status_code=503, detail="Resource manager not initialized")
    return resource_manager


async def get_encoder(org_id: str, model_id: str):
    from main import model_manager
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    model = await model_manager.get_model(org_id, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: org={org_id}, model={model_id}")
    return model.encoder


# Endpoints
@router.post("", response_model=CreateGlyphResponse)
async def create_glyph(
    org_id: str,
    model_id: str,
    request: CreateGlyphRequest,
    storage: GlyphStorage = Depends(get_storage),
    resource_mgr: ResourceManager = Depends(get_resource_manager),
) -> CreateGlyphResponse:
    """Create a new glyph."""
    try:
        await resource_mgr.enforce_quota(org_id, model_id, additional_glyphs=1)
        encoder = await get_encoder(org_id, model_id)
        embedding = encoder.encode_text(request.concept)
        
        result = await storage.create_glyph(
            org_id=org_id,
            model_id=model_id,
            concept_text=request.concept,
            embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
            metadata=request.metadata,
        )
        
        resource_mgr.invalidate_cache(org_id, model_id)
        return result
        
    except QuotaExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create glyph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=Dict[str, Any])
async def create_glyphs_batch(
    org_id: str,
    model_id: str,
    request: BatchCreateGlyphRequest,
    storage: GlyphStorage = Depends(get_storage),
    resource_mgr: ResourceManager = Depends(get_resource_manager),
) -> Dict[str, Any]:
    """Create multiple glyphs in a batch."""
    try:
        await resource_mgr.enforce_quota(org_id, model_id, additional_glyphs=len(request.concepts))
        encoder = await get_encoder(org_id, model_id)
        
        results = []
        errors = []
        
        for i, concept in enumerate(request.concepts):
            try:
                embedding = encoder.encode_text(concept)
                result = await storage.create_glyph(
                    org_id=org_id,
                    model_id=model_id,
                    concept_text=concept,
                    embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                    metadata=request.metadata,
                )
                results.append({"index": i, "glyph_id": str(result.glyph_id), "status": "created"})
            except Exception as e:
                errors.append({"index": i, "concept": concept[:50], "error": str(e)})
        
        resource_mgr.invalidate_cache(org_id, model_id)
        
        return {"created": len(results), "failed": len(errors), "results": results, "errors": errors}
        
    except QuotaExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Batch create failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{glyph_id}", response_model=GlyphResponse)
async def get_glyph(
    org_id: str,
    model_id: str,
    glyph_id: UUID,
    storage: GlyphStorage = Depends(get_storage),
) -> GlyphResponse:
    """Get a glyph by ID."""
    try:
        return await storage.get_glyph(org_id, model_id, glyph_id)
    except GlyphNotFoundException:
        raise HTTPException(status_code=404, detail=f"Glyph not found: {glyph_id}")


@router.put("/{glyph_id}", response_model=GlyphResponse)
async def update_glyph(
    org_id: str,
    model_id: str,
    glyph_id: UUID,
    request: UpdateGlyphRequest,
    storage: GlyphStorage = Depends(get_storage),
) -> GlyphResponse:
    """Update a glyph."""
    try:
        embedding = None
        if request.concept:
            encoder = await get_encoder(org_id, model_id)
            embedding = encoder.encode_text(request.concept)
            embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        
        return await storage.update_glyph(
            org_id=org_id,
            model_id=model_id,
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
    org_id: str,
    model_id: str,
    glyph_id: UUID,
    storage: GlyphStorage = Depends(get_storage),
) -> Dict[str, str]:
    """Delete a glyph and its edges."""
    deleted = await storage.delete_glyph(org_id, model_id, glyph_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Glyph not found: {glyph_id}")
    return {"status": "deleted", "glyph_id": str(glyph_id)}


@router.get("", response_model=GlyphListResponse)
async def list_glyphs(
    org_id: str,
    model_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    storage: GlyphStorage = Depends(get_storage),
) -> GlyphListResponse:
    """List glyphs with pagination."""
    glyphs = await storage.list_glyphs(org_id, model_id, limit=limit, offset=offset)
    total = await storage.count_glyphs(org_id, model_id)
    
    return GlyphListResponse(glyphs=glyphs, total=total, limit=limit, offset=offset)


class GlyphStatsResponse(BaseModel):
    """Response model for glyph statistics."""
    total_glyphs: int = Field(..., description="Total number of glyphs")
    org_id: str = Field(..., description="Organization ID")
    model_id: str = Field(..., description="Model ID")


@router.get("/stats", response_model=GlyphStatsResponse)
async def get_glyph_stats(
    org_id: str,
    model_id: str,
    storage: GlyphStorage = Depends(get_storage),
) -> GlyphStatsResponse:
    """Get glyph statistics for a model."""
    total = await storage.count_glyphs(org_id, model_id)
    return GlyphStatsResponse(
        total_glyphs=total,
        org_id=org_id,
        model_id=model_id,
    )
