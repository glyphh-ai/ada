"""
Deployment API Routes for Glyphh Runtime.

CLI-facing endpoints for model deployment and management.
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
    namespace: str
    model_id: str
    org_id: Optional[str] = None
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
    """Get model manager from app state."""
    from main import model_manager
    return model_manager


async def get_auth_service() -> AuthService:
    """Get auth service."""
    return AuthService()


async def get_current_user(
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[User]:
    """Get current user (optional in local mode)."""
    if settings.deployment_mode == "local":
        return None
    # In production, extract token from header and validate
    # For now, return None (will be implemented with middleware)
    return None


# Endpoints
@router.post("/deploy", response_model=DeployResponse)
async def deploy_model(
    file: UploadFile = File(..., description="The .glyphh model file"),
    namespace: Optional[str] = Query(None, description="Namespace to assign (e.g. org_id/model_id). Generated from filename if not provided."),
    manager: ModelManager = Depends(get_model_manager),
) -> DeployResponse:
    """
    Deploy a .glyphh model file.
    
    Accepts a binary .glyphh file and deploys it to the runtime.
    Optionally accepts a namespace query param for org-scoped deployments.
    Returns the namespace and endpoint URLs for accessing the model.
    """
    if not file.filename.endswith(".glyphh"):
        raise HTTPException(
            status_code=400,
            detail="File must have .glyphh extension"
        )
    
    # Read file content
    content = await file.read()
    
    # Use provided namespace or generate from filename
    if namespace is None:
        namespace = file.filename.replace(".glyphh", "")
    
    try:
        # Load model
        model_info = await manager.load_model_from_bytes(content, namespace)
        
        # Build endpoint URLs
        base_url = f"{settings.host}:{settings.port}"
        
        # Extract org_id if namespace is org-scoped (org_id/model_id)
        org_id = None
        if "/" in namespace:
            org_id = namespace.split("/")[0]
        
        return DeployResponse(
            namespace=namespace,
            model_id=namespace,
            org_id=org_id,
            mcp_endpoint=f"http://{base_url}/{namespace}/mcp",
            listener_endpoint=f"http://{base_url}/{namespace}/listener",
        )
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status", response_model=StatusResponse)
async def get_status(
    manager: ModelManager = Depends(get_model_manager),
) -> StatusResponse:
    """
    Get runtime status.
    
    Returns version, loaded models count, uptime, and deployment mode.
    """
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
    """
    List all deployed models.
    
    Returns an array of model info objects.
    """
    models = await manager.list_models()
    return {"models": [m.model_dump() for m in models]}


@router.delete("/models/{model_id:path}")
async def delete_model(
    model_id: str,
    delete_data: bool = Query(True, description="Also delete glyphs and edges"),
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, str]:
    """
    Remove a deployed model.
    
    Unloads the model and optionally deletes all associated data.
    model_id can be a simple name or an org-scoped path like org_id/model_id.
    """
    try:
        await manager.unload_model(model_id, delete_data=delete_data)
        return {"status": "deleted", "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")


@router.patch("/models/{model_id:path}/config")
async def update_model_config(
    model_id: str,
    config: ModelConfigUpdate,
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, Any]:
    """
    Update model configuration.
    
    Updates similarity weights, beam width, or max tree depth.
    Changes take effect immediately without re-encoding.
    """
    try:
        await manager.update_config(
            namespace=model_id,
            similarity_weights=config.similarity_weights,
            beam_width=config.beam_width,
            max_tree_depth=config.max_tree_depth,
        )
        return {"status": "updated", "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")


@router.post("/models/{model_id:path}/re-encode")
async def re_encode_model(
    model_id: str,
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, str]:
    """
    Re-encode all glyphs in a model.
    
    Triggers re-encoding of all glyphs using the current encoder.
    This is a long-running operation for large models.
    """
    try:
        await manager.re_encode_namespace(model_id)
        return {"status": "re-encoding", "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")


@router.delete("/models/{model_id:path}/data")
async def clear_model_data(
    model_id: str,
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, Any]:
    """
    Clear all glyphs and edges for a model.
    
    Preserves the model configuration but removes all data.
    """
    try:
        result = await manager.clear_namespace_data(model_id)
        return {
            "status": "cleared",
            "model_id": model_id,
            "glyphs_deleted": result.get("glyphs_deleted", 0),
            "edges_deleted": result.get("edges_deleted", 0),
        }
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")


@router.get("/logs")
async def get_logs(
    lines: int = Query(100, ge=1, le=1000, description="Number of log lines"),
) -> Dict[str, Any]:
    """
    Get runtime logs.
    
    Returns the most recent log lines.
    """
    # TODO: Implement actual log retrieval
    return {
        "logs": [],
        "lines_requested": lines,
        "message": "Log retrieval not yet implemented"
    }


@router.get("/tokens")
async def list_tokens() -> Dict[str, List[Dict[str, Any]]]:
    """
    List webhook tokens.
    
    Returns all active webhook tokens.
    """
    # TODO: Implement token listing from database
    return {"tokens": []}


@router.delete("/tokens/{token_id}")
async def revoke_token(token_id: str) -> Dict[str, str]:
    """
    Revoke a webhook token.
    
    Marks the token as revoked, preventing further use.
    """
    # TODO: Implement token revocation
    return {"status": "revoked", "token_id": token_id}


@router.get("/models/{model_id:path}/metadata", response_model=ModelMetadataResponse)
async def get_model_metadata(
    model_id: str,
    manager: ModelManager = Depends(get_model_manager),
) -> ModelMetadataResponse:
    """
    Get model metadata for marketplace display.
    
    Returns meta_name, short_description, and long_description
    for rendering in the Studio marketplace.
    """
    try:
        loaded_model = await manager.get_model(model_id)
        if loaded_model is None:
            raise ModelNotFoundException(model_id)
        
        return ModelMetadataResponse(
            namespace=model_id,
            meta_name=loaded_model.meta_name,
            short_description=loaded_model.short_description,
            long_description=loaded_model.long_description,
            model_version=getattr(loaded_model.sdk_model, 'version', None),
            sdk_version=await manager._get_sdk_version(),
        )
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
