"""
Deployment API Routes for Glyphh Runtime.

Endpoints for model management: status, config, re-encode, version history.
All models identified by (org_id, model_id) — no namespace concept.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from domains.auth.service import AuthService, User
from domains.models.manager import ModelManager
from domains.models.schemas import ModelMetadataResponse
from infrastructure.config import get_settings
from shared.exceptions import ModelNotFoundException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["deployment"])
settings = get_settings()


# Request/Response Models
class ModelConfigUpdate(BaseModel):
    """Model configuration update request."""
    similarity_weights: Optional[Dict[str, float]] = None
    beam_width: Optional[int] = Field(None, ge=1, le=20)
    max_tree_depth: Optional[int] = Field(None, ge=1, le=10)






class StatusResponse(BaseModel):
    """Runtime status response."""
    version: str
    models_loaded: int
    uptime: str
    deployment_mode: str
    license_status: str = "valid"


class ConfigUpdateRequest(BaseModel):
    """Full config update request from Platform."""
    config: Dict[str, Any]
    change_type: str  # "nl_only", "encoder_only", "mixed"


class ConfigUpdateResponse(BaseModel):
    """Response from config update."""
    status: str  # "applied", "re_encoding"
    change_type: str
    job_id: Optional[str] = None
    message: str


# Dependency injection
async def get_model_manager() -> ModelManager:
    from main import model_manager
    return model_manager


async def get_auth_service() -> AuthService:
    return AuthService()


async def get_current_user(
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[User]:
    if settings.deployment_mode == "local":
        return None
    return None
















@router.get("/status", response_model=StatusResponse)
async def get_status(
    manager: ModelManager = Depends(get_model_manager),
) -> StatusResponse:
    """Get runtime status."""
    from main import _start_time
    
    uptime = datetime.utcnow() - _start_time
    models = await manager.list_models() if manager else []
    
    return StatusResponse(
        version="1.0.0",
        models_loaded=len(models),
        uptime=str(uptime),
        deployment_mode=settings.deployment_mode,
    )


@router.get("/models")
async def list_models(
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, Any]:
    """List all deployed models."""
    models = await manager.list_models()
    return {"models": [m.model_dump() for m in models]}


@router.delete("/models/{org_id}/{model_id}")
async def delete_model(
    org_id: str,
    model_id: str,
    delete_data: bool = Query(True, description="Also delete glyphs and edges"),
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, str]:
    """Remove a deployed model identified by org_id and model_id."""
    try:
        await manager.unload_model(org_id, model_id, delete_data=delete_data)
        return {"status": "deleted", "org_id": org_id, "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: org={org_id}, model={model_id}")


@router.patch("/models/{org_id}/{model_id}/config")
async def update_model_config(
    org_id: str,
    model_id: str,
    config: ModelConfigUpdate,
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, Any]:
    """Update model configuration (weights, beam width, max tree depth)."""
    try:
        await manager.update_config(
            org_id=org_id,
            model_id=model_id,
            similarity_weights=config.similarity_weights,
            beam_width=config.beam_width,
            max_tree_depth=config.max_tree_depth,
        )
        return {"status": "updated", "org_id": org_id, "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: org={org_id}, model={model_id}")


@router.post("/models/{org_id}/{model_id}/re-encode")
async def re_encode_model(
    org_id: str,
    model_id: str,
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, str]:
    """Re-encode all glyphs in a model."""
    try:
        await manager.re_encode_model(org_id, model_id)
        return {"status": "re-encoding", "org_id": org_id, "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: org={org_id}, model={model_id}")


@router.delete("/models/{org_id}/{model_id}/data")
async def clear_model_data(
    org_id: str,
    model_id: str,
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, Any]:
    """Clear all glyphs and edges for a model, preserving config."""
    try:
        result = await manager.clear_model_data(org_id, model_id)
        return {
            "status": "cleared",
            "org_id": org_id,
            "model_id": model_id,
            "glyphs_deleted": result.glyphs_deleted,
            "edges_deleted": result.edges_deleted,
        }
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: org={org_id}, model={model_id}")


@router.get("/logs")
async def get_logs(
    lines: int = Query(100, ge=1, le=1000, description="Number of log lines"),
) -> Dict[str, Any]:
    return {"logs": [], "lines_requested": lines, "message": "Log retrieval not yet implemented"}


@router.get("/tokens")
async def list_tokens() -> Dict[str, List[Dict[str, Any]]]:
    return {"tokens": []}


@router.delete("/tokens/{token_id}")
async def revoke_token(token_id: str) -> Dict[str, str]:
    return {"status": "revoked", "token_id": token_id}


class VersionHistoryEntry(BaseModel):
    """Version history entry."""
    version: str
    deployed_at: datetime
    deployed_by: Optional[str] = None
    is_current: bool
    metadata: Optional[Dict[str, Any]] = None


class VersionHistoryResponse(BaseModel):
    """Version history response."""
    org_id: str
    model_id: str
    versions: List[VersionHistoryEntry]


@router.get("/models/{org_id}/{model_id}/versions", response_model=VersionHistoryResponse)
async def get_version_history(
    org_id: str,
    model_id: str,
) -> VersionHistoryResponse:
    """
    Get version history for a model.
    
    Returns all deployed versions of the model, ordered by deployment time (newest first).
    History is retained even after model deletion.
    
    Validates: Requirements 31.3, 31.6 - Version history API and retention
    """
    try:
        from infrastructure.database import async_session_maker
        from domains.models.db_models import ModelVersionHistory
        from sqlalchemy import select
        
        async with async_session_maker() as session:
            result = await session.execute(
                select(ModelVersionHistory)
                .where(ModelVersionHistory.org_id == org_id)
                .where(ModelVersionHistory.model_id == model_id)
                .order_by(ModelVersionHistory.deployed_at.desc())
            )
            history_entries = result.scalars().all()
            
            versions = [
                VersionHistoryEntry(
                    version=entry.version,
                    deployed_at=entry.deployed_at,
                    deployed_by=entry.deployed_by,
                    is_current=entry.is_current == 1,
                    metadata=entry.model_metadata,
                )
                for entry in history_entries
            ]
            
            return VersionHistoryResponse(
                org_id=org_id,
                model_id=model_id,
                versions=versions,
            )
            
    except ImportError as e:
        logger.warning(f"Database components not available: {e}")
        return VersionHistoryResponse(org_id=org_id, model_id=model_id, versions=[])
    except Exception as e:
        logger.error(f"Failed to get version history for {org_id}/{model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get version history: {str(e)}")


@router.get("/models/{org_id}/{model_id}/metadata", response_model=ModelMetadataResponse)
async def get_model_metadata(
    org_id: str,
    model_id: str,
    manager: ModelManager = Depends(get_model_manager),
) -> ModelMetadataResponse:
    """Get model metadata for marketplace display."""
    try:
        loaded_model = await manager.get_model(org_id, model_id)
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)
        
        return ModelMetadataResponse(
            org_id=org_id,
            model_id=model_id,
            meta_name=loaded_model.meta_name,
            short_description=loaded_model.short_description,
            long_description=loaded_model.long_description,
            model_version=getattr(loaded_model.sdk_model, 'version', None),
            sdk_version=await manager._get_sdk_version(),
        )
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: org={org_id}, model={model_id}")


@router.get("/models/{org_id}/{model_id}/config")
async def get_model_active_config(
    org_id: str,
    model_id: str,
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, Any]:
    """
    Get the currently active configuration for a deployed model.
    
    Returns the encoder config as stored in the loaded model,
    used by Platform to compute config diffs for hot updates.
    """
    try:
        return await manager.get_active_config(org_id, model_id)
    except ModelNotFoundException:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: org={org_id}, model={model_id}"
        )


@router.post("/models/{org_id}/{model_id}/config/update", response_model=ConfigUpdateResponse)
async def update_model_config_full(
    org_id: str,
    model_id: str,
    request: ConfigUpdateRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> ConfigUpdateResponse:
    """
    Apply a full config update to a deployed model.
    
    For NL-only changes: hot-reload query patterns immediately.
    For encoder changes: update encoder and trigger background re-encode.
    
    This endpoint is called by Platform's push-update flow.
    """
    try:
        result = await manager.apply_config_update(
            org_id=org_id,
            model_id=model_id,
            new_config=request.config,
            change_type=request.change_type,
        )
        return ConfigUpdateResponse(**result)
    except ModelNotFoundException:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: org={org_id}, model={model_id}"
        )
    except Exception as e:
        logger.error(f"Config update failed for org={org_id}, model={model_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Config update failed: {str(e)}"
        )
