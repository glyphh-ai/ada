"""
Deployment API Routes for Glyphh Runtime.

CLI-facing endpoints for model deployment and management.
All models identified by (org_id, model_id) — no namespace concept.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
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


class DeployResponse(BaseModel):
    """Deployment response."""
    org_id: str
    model_id: str
    mcp_endpoint: str
    listener_endpoint: str
    status: str = "deployed"


class StatusResponse(BaseModel):
    """Runtime status response."""
    version: str
    models_loaded: int
    uptime: str
    deployment_mode: str
    license_status: str = "valid"


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


# Endpoints
@router.post("/deploy", response_model=DeployResponse)
async def deploy_model(
    file: UploadFile = File(..., description="The .glyphh model file"),
    org_id: str = Query(..., description="Organization ID"),
    model_id: str = Query(..., description="Model ID"),
    manager: ModelManager = Depends(get_model_manager),
) -> DeployResponse:
    """
    Deploy a .glyphh model file.
    
    Accepts a binary .glyphh file and deploys it to the runtime.
    Requires org_id and model_id as separate query params.
    """
    if not file.filename.endswith(".glyphh"):
        raise HTTPException(status_code=400, detail="File must have .glyphh extension")
    
    content = await file.read()
    
    try:
        model_info = await manager.load_model_from_bytes(content, org_id, model_id)
        
        base_url = f"{settings.host}:{settings.port}"
        
        return DeployResponse(
            org_id=org_id,
            model_id=model_id,
            mcp_endpoint=f"http://{base_url}/{org_id}/{model_id}/mcp",
            listener_endpoint=f"http://{base_url}/{org_id}/{model_id}/listener",
        )
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        raise HTTPException(status_code=400, detail=str(e))


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
