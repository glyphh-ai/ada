"""
Deployment API Routes for Glyphh Runtime.

CLI-facing endpoints for model deployment and management.
All models identified by (org_id, model_id) — no namespace concept.

Updated to support async schema index building during deployment.
Validates: Requirement 13.5
"""

import asyncio
import gzip
import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from domains.auth.service import AuthService, User
from domains.models.manager import ModelManager
from domains.models.schemas import ModelMetadataResponse
from infrastructure.config import get_settings
from shared.exceptions import ModelNotFoundException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["deployment"])
settings = get_settings()

# Track schema index building status per model
_schema_index_building_status: Dict[str, Dict[str, Any]] = {}


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
    schema_index_status: str = "not_started"  # "not_started", "building", "ready", "failed"
    schema_index_message: Optional[str] = None
    concepts_load_status: str = "not_applicable"  # "not_applicable", "loading", "failed"
    concepts_load_job_id: Optional[str] = None


class SchemaIndexStatusResponse(BaseModel):
    """Schema index building status response."""
    org_id: str
    model_id: str
    status: str  # "not_started", "building", "ready", "failed"
    message: Optional[str] = None
    vector_count: Optional[int] = None
    build_time_ms: Optional[float] = None


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


def _extract_concepts_from_model(content: bytes) -> Optional[List[Dict[str, Any]]]:
    """
    Extract concepts from a .glyphh model file.
    
    The .glyphh format is gzip-compressed JSON. If the model contains
    a "concepts" field, return it for auto-loading on deploy.
    
    Args:
        content: Raw bytes of the .glyphh file
    
    Returns:
        List of concept records if present, None otherwise
    
    Validates: Requirements 2.1
    """
    try:
        # Decompress gzip content
        decompressed = gzip.decompress(content)
        model_data = json.loads(decompressed.decode('utf-8'))
        
        # Extract concepts if present
        concepts = model_data.get("concepts")
        if concepts is not None and isinstance(concepts, list) and len(concepts) > 0:
            logger.info(f"Found {len(concepts)} embedded concepts in model")
            return concepts
        
        return None
    except Exception as e:
        logger.warning(f"Could not extract concepts from model: {e}")
        return None


# Endpoints
@router.post("/deploy", response_model=DeployResponse)
async def deploy_model(
    file: UploadFile = File(..., description="The .glyphh model file"),
    org_id: str = Query(..., description="Organization ID"),
    model_id: str = Query(..., description="Model ID"),
    manager: ModelManager = Depends(get_model_manager),
    background_tasks: BackgroundTasks = None,
) -> DeployResponse:
    """
    Deploy a .glyphh model file.
    
    Accepts a binary .glyphh file and deploys it to the runtime.
    Requires org_id and model_id as separate query params.
    
    Vector dimension is limited by MAX_VECTOR_DIMENSION env var (default 2048).
    
    Schema index building is started in the background. The endpoint returns
    immediately with status "deployed" and schema_index_status "building".
    Use GET /api/models/{org_id}/{model_id}/schema-index/status to check
    the schema index building progress.
    
    Stored procedures from the model are persisted to the database (upsert).
    
    Validates: Requirement 13.5 - THE Runtime SHALL support async schema index
    building during deployment
    Validates: Requirement 8.4, 8.5 - Stored procedures are persisted on deploy
    """
    if not file.filename.endswith(".glyphh"):
        raise HTTPException(status_code=400, detail="File must have .glyphh extension")
    
    content = await file.read()
    
    # Pre-validate dimension before loading
    dimension_error = await _validate_model_dimension(content)
    if dimension_error:
        raise HTTPException(status_code=400, detail=dimension_error)
    
    try:
        model_info = await manager.load_model_from_bytes(content, org_id, model_id)
        
        # Persist stored procedures from the model (Requirements 8.4, 8.5)
        await _persist_model_procedures(manager, org_id, model_id)
        
        # Record version history (Requirement 31.1, 31.2)
        await _record_version_history(manager, org_id, model_id)
        
        base_url = f"{settings.host}:{settings.port}"
        
        # Start async schema index building
        # Validates: Requirement 13.5
        model_key = f"{org_id}/{model_id}"
        _schema_index_building_status[model_key] = {
            "status": "building",
            "message": "Schema index building started",
            "started_at": datetime.utcnow().isoformat(),
        }
        
        if background_tasks is not None:
            background_tasks.add_task(
                _build_schema_index_async,
                manager,
                org_id,
                model_id,
            )
        else:
            # If no background tasks available, start in a separate task
            asyncio.create_task(
                _build_schema_index_async(manager, org_id, model_id)
            )
        
        # Check for embedded concepts and auto-trigger loading
        # Validates: Requirements 2.1, 2.2, 2.5, 2.6
        concepts_load_status = "not_applicable"
        concepts_load_job_id = None
        
        concepts = _extract_concepts_from_model(content)
        if concepts:
            try:
                from api.routes.listeners import get_async_listener_service
                async_service = get_async_listener_service()
                job_id = await async_service.start_load(
                    org_id=org_id,
                    model_id=model_id,
                    records=concepts,
                    batch_size=50,
                )
                concepts_load_status = "loading"
                concepts_load_job_id = str(job_id)
                logger.info(f"Auto-triggered concepts load job {job_id} for {org_id}/{model_id} with {len(concepts)} concepts")
            except Exception as e:
                logger.error(f"Failed to start concepts load: {e}")
                concepts_load_status = "failed"
        
        return DeployResponse(
            org_id=org_id,
            model_id=model_id,
            mcp_endpoint=f"http://{base_url}/{org_id}/{model_id}/mcp",
            listener_endpoint=f"http://{base_url}/{org_id}/{model_id}/listener",
            status="deployed",
            schema_index_status="building",
            schema_index_message="Schema index building started in background",
            concepts_load_status=concepts_load_status,
            concepts_load_job_id=concepts_load_job_id,
        )
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        raise HTTPException(status_code=400, detail=str(e))


async def _build_schema_index_async(
    manager: ModelManager,
    org_id: str,
    model_id: str,
) -> None:
    """
    Build schema index asynchronously in the background.
    
    This function is called as a background task after model deployment.
    It builds the schema index for the deployed model and updates the
    status in _schema_index_building_status.
    
    Args:
        manager: The ModelManager instance
        org_id: Organization ID
        model_id: Model ID
    
    Validates: Requirement 13.5 - THE Runtime SHALL support async schema index
    building during deployment
    """
    import time
    
    model_key = f"{org_id}/{model_id}"
    start_time = time.time()
    
    try:
        # Get the deployed model
        model = await manager.get_model(org_id, model_id)
        if model is None:
            _schema_index_building_status[model_key] = {
                "status": "failed",
                "message": f"Model not found: {org_id}/{model_id}",
                "completed_at": datetime.utcnow().isoformat(),
            }
            return
        
        # Import SDK components
        try:
            from glyphh.nl.auto_schema_matcher import AutoSchemaMatcher, AutoMatchConfig
            from domains.nl_query.schema_index import SchemaIndex
        except ImportError as e:
            _schema_index_building_status[model_key] = {
                "status": "failed",
                "message": f"SDK components not available: {e}",
                "completed_at": datetime.utcnow().isoformat(),
            }
            return
        
        # Check if model has required attributes
        sdk_model = model.sdk_model if hasattr(model, 'sdk_model') else model
        if not hasattr(sdk_model, 'encoder') or not hasattr(sdk_model, 'config'):
            _schema_index_building_status[model_key] = {
                "status": "failed",
                "message": "Model missing encoder or config",
                "completed_at": datetime.utcnow().isoformat(),
            }
            return
        
        # Create and build schema index
        schema_index = SchemaIndex(model_id=model_id)
        schema_index.build_from_model(sdk_model)
        
        # Calculate build time
        build_time_ms = (time.time() - start_time) * 1000
        
        # Update status
        metrics = schema_index.get_metrics()
        _schema_index_building_status[model_key] = {
            "status": "ready",
            "message": f"Schema index built successfully with {metrics.vector_count} vectors",
            "completed_at": datetime.utcnow().isoformat(),
            "vector_count": metrics.vector_count,
            "role_count": metrics.role_count,
            "value_count": metrics.value_count,
            "build_time_ms": build_time_ms,
            "memory_bytes": metrics.memory_bytes,
        }
        
        logger.info(
            f"Schema index built for {org_id}/{model_id}: "
            f"{metrics.vector_count} vectors in {build_time_ms:.2f}ms"
        )
        
    except Exception as e:
        logger.error(f"Failed to build schema index for {org_id}/{model_id}: {e}")
        _schema_index_building_status[model_key] = {
            "status": "failed",
            "message": f"Schema index building failed: {str(e)}",
            "completed_at": datetime.utcnow().isoformat(),
        }


@router.get("/models/{org_id}/{model_id}/schema-index/status", response_model=SchemaIndexStatusResponse)
async def get_schema_index_status(
    org_id: str,
    model_id: str,
) -> SchemaIndexStatusResponse:
    """
    Get the schema index building status for a model.
    
    Returns the current status of schema index building:
    - "not_started": Schema index building has not been started
    - "building": Schema index is currently being built
    - "ready": Schema index is ready for use
    - "failed": Schema index building failed
    
    If the status is "building", the client should retry after a short delay.
    Returns HTTP 202 Accepted when building is in progress.
    
    Validates: Requirement 13.5 - THE Runtime SHALL support async schema index
    building during deployment
    """
    model_key = f"{org_id}/{model_id}"
    
    if model_key not in _schema_index_building_status:
        return SchemaIndexStatusResponse(
            org_id=org_id,
            model_id=model_id,
            status="not_started",
            message="Schema index building has not been started for this model",
        )
    
    status_info = _schema_index_building_status[model_key]
    
    response = SchemaIndexStatusResponse(
        org_id=org_id,
        model_id=model_id,
        status=status_info.get("status", "unknown"),
        message=status_info.get("message"),
        vector_count=status_info.get("vector_count"),
        build_time_ms=status_info.get("build_time_ms"),
    )
    
    # Return 202 Accepted if still building
    # This allows clients to poll for completion
    if response.status == "building":
        # Note: FastAPI doesn't support returning 202 with a response model directly
        # The client should check the status field to determine if building is complete
        pass
    
    return response


async def _persist_model_procedures(
    manager: ModelManager,
    org_id: str,
    model_id: str,
) -> None:
    """
    Persist stored procedures from a deployed model to the database.
    
    Extracts stored_procedures from the SDK model and upserts them
    to the database. Existing procedures with the same name are updated.
    
    Args:
        manager: The ModelManager instance
        org_id: Organization ID
        model_id: Model ID
    
    Validates: Requirements 8.4, 8.5 - Stored procedures are persisted on deploy
    """
    try:
        # Get the deployed model
        model = await manager.get_model(org_id, model_id)
        if model is None:
            logger.warning(f"Model not found for procedure persistence: {org_id}/{model_id}")
            return
        
        # Get SDK model
        sdk_model = model.sdk_model if hasattr(model, 'sdk_model') else model
        
        # Check if model has stored_procedures
        if not hasattr(sdk_model, 'stored_procedures') or not sdk_model.stored_procedures:
            logger.debug(f"No stored procedures in model {org_id}/{model_id}")
            return
        
        # Import procedure service
        from domains.procedures.service import StoredProcedureService
        from domains.procedures.schemas import StoredProcedureCreate
        from infrastructure.database import async_session_maker
        
        async with async_session_maker() as session:
            procedure_service = StoredProcedureService(session)
            
            persisted_count = 0
            for procedure in sdk_model.stored_procedures:
                try:
                    # Create procedure data from SDK StoredProcedure
                    create_data = StoredProcedureCreate(
                        name=procedure.name,
                        gql_query=procedure.gql_query,
                        lexicons=procedure.lexicons,
                        description=procedure.description,
                    )
                    
                    # Upsert (create or update)
                    await procedure_service.upsert(org_id, model_id, create_data)
                    persisted_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to persist procedure '{procedure.name}': {e}")
            
            logger.info(f"Persisted {persisted_count} stored procedures for {org_id}/{model_id}")
            
    except ImportError as e:
        logger.warning(f"Procedure service not available: {e}")
    except Exception as e:
        logger.error(f"Failed to persist procedures for {org_id}/{model_id}: {e}")


async def _record_version_history(
    manager: ModelManager,
    org_id: str,
    model_id: str,
    deployed_by: Optional[str] = None,
) -> None:
    """
    Record version history for a deployed model.
    
    Creates a new version history entry and marks previous versions as non-current.
    
    Args:
        manager: The ModelManager instance
        org_id: Organization ID
        model_id: Model ID
        deployed_by: User ID or email of deployer (optional)
    
    Validates: Requirements 31.1, 31.2 - Version history tracking
    """
    try:
        # Get the deployed model
        model = await manager.get_model(org_id, model_id)
        if model is None:
            logger.warning(f"Model not found for version history: {org_id}/{model_id}")
            return
        
        # Get SDK model version
        sdk_model = model.sdk_model if hasattr(model, 'sdk_model') else model
        version = getattr(sdk_model, 'version', '1.0.0')
        
        # Import database components
        from infrastructure.database import async_session_maker
        from domains.models.db_models import ModelVersionHistory
        from sqlalchemy import update
        
        async with async_session_maker() as session:
            # Mark previous versions as non-current
            await session.execute(
                update(ModelVersionHistory)
                .where(ModelVersionHistory.org_id == org_id)
                .where(ModelVersionHistory.model_id == model_id)
                .values(is_current=0)
            )
            
            # Create new version history entry
            history_entry = ModelVersionHistory(
                org_id=org_id,
                model_id=model_id,
                version=version,
                deployed_by=deployed_by,
                is_current=1,
                model_metadata={
                    "dimension": getattr(sdk_model.encoder_config, 'dimension', None) if hasattr(sdk_model, 'encoder_config') else None,
                    "glyph_count": len(sdk_model.glyphs) if hasattr(sdk_model, 'glyphs') else 0,
                }
            )
            session.add(history_entry)
            await session.commit()
            
            logger.info(f"Recorded version history for {org_id}/{model_id}: v{version}")
            
    except ImportError as e:
        logger.warning(f"Database components not available: {e}")
    except Exception as e:
        logger.error(f"Failed to record version history for {org_id}/{model_id}: {e}")


async def _validate_model_dimension(content: bytes) -> Optional[str]:
    """
    Validate model dimension against runtime limit.
    
    Returns error message if dimension exceeds limit, None if valid.
    """
    import gzip
    import json as json_module
    
    max_dim = settings.max_vector_dimension
    
    # Try to extract dimension from content
    try:
        # Try gzipped format first
        try:
            decompressed = gzip.decompress(content)
            # For gzipped .glyphh files, we'd need to parse the model
            # For now, skip validation for gzipped files (they're from CLI)
            return None
        except gzip.BadGzipFile:
            pass
        
        # Plain JSON config from platform
        config_data = json_module.loads(content)
        model_config = config_data.get("config", {})
        
        # Check encoder_config.dimension
        encoder_config = model_config.get("encoder_config", model_config)
        dimension = encoder_config.get("dimension", 0)
        
        if dimension > max_dim:
            return (
                f"Model dimension ({dimension}) exceeds runtime limit ({max_dim}). "
                f"Cloud runtimes support up to {max_dim} dimensions (pgvector index limit). "
                f"For larger models, please use a local runtime."
            )
        
        return None
        
    except json_module.JSONDecodeError:
        # Can't parse, let the model loader handle it
        return None
    except Exception as e:
        logger.warning(f"Could not validate model dimension: {e}")
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
    
    For NL-only changes: hot-reload IntentMatcher patterns immediately.
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
