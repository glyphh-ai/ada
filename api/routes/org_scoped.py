"""
Org-Scoped API Routes for Glyphh Runtime (Cloud Mode).

Endpoints scoped by org_id and model_id for multi-tenant cloud deployments.
URL pattern: /{org_id}/{model_id}/...

No namespace concept — org_id and model_id are passed directly to services.
"""

import logging
import shutil
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from domains.auth.service import AuthService
from domains.mcp.server import MCPServer
from domains.query.service import QueryService
from infrastructure.config import get_settings
from shared.auth import AuthenticatedUser, get_current_user, require_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}", tags=["org-scoped"])
settings = get_settings()


# Dependency injection
async def get_auth_service() -> AuthService:
    return AuthService()


async def validate_org_access(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """
    Validate org access with local mode bypass.

    Used for model lifecycle endpoints (deploy, undeploy, status, re-encode)
    where the user is operating their own CLI.
    """
    if current_user.org_id == "local-dev-org":
        return current_user

    if current_user.org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail="Organization mismatch - you don't have access to this organization"
        )

    return current_user


async def validate_token_access(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(require_token),
) -> AuthenticatedUser:
    """
    Always require a valid JWT token — no local mode bypass.

    Used for data and query endpoints (listener, MCP) that external
    services like Boomi, Make.com, or agents call with API tokens.
    """
    if current_user.org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail="Organization mismatch - you don't have access to this organization"
        )

    return current_user


async def get_mcp_server(
    org_id: str,
    model_id: str,
) -> MCPServer:
    """Get MCP server for org/model."""
    from main import model_manager
    from infrastructure.database import async_session_maker

    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    query_service = QueryService(model_manager, async_session_maker)
    auth_service = AuthService()

    return MCPServer(query_service, auth_service)



class UndeployRequest(BaseModel):
    delete_data: bool = False


async def get_model_manager():
    """Get the ModelManager instance from main."""
    from main import model_manager

    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return model_manager




# MCP Endpoint
@router.post("/mcp")
async def mcp_endpoint(
    org_id: str,
    model_id: str,
    request: Dict[str, Any],
    mcp_server: MCPServer = Depends(get_mcp_server),
    current_user: AuthenticatedUser = Depends(validate_token_access),
) -> Dict[str, Any]:
    """
    MCP endpoint for org-scoped model access.

    Requires a valid JWT token (even in local mode). External services
    like Boomi, Make.com, and agents use API tokens to query models.
    """
    tool_name = request.get("tool")
    arguments = request.get("arguments", {})

    if not tool_name:
        raise HTTPException(status_code=400, detail="Missing 'tool' field")

    arguments["org_id"] = org_id
    arguments["model_id"] = model_id

    response = await mcp_server.handle_tool_call(
        tool_name=tool_name,
        arguments=arguments,
        auth_token="",
    )

    return response.to_dict()


@router.get("/mcp/tools")
async def list_mcp_tools(
    org_id: str,
    model_id: str,
    mcp_server: MCPServer = Depends(get_mcp_server),
    current_user: AuthenticatedUser = Depends(validate_token_access),
) -> Dict[str, Any]:
    return {"tools": mcp_server.get_tools_list()}


# Model Lifecycle Endpoints

@router.get("/ready")
async def readiness_check(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """Check if a model is deployed and ready to serve queries."""
    from main import model_manager

    if model_manager is None:
        return {"ready": False, "status": "model_manager_not_initialized", "model_id": model_id}

    loaded_model = await model_manager.get_model(org_id, model_id)
    if loaded_model is None:
        return {"ready": False, "status": "not_deployed", "model_id": model_id}

    is_locked = model_manager.is_model_locked(org_id, model_id)
    if is_locked:
        return {"ready": False, "status": "locked", "model_id": model_id}

    return {
        "ready": True,
        "status": "ready",
        "model_id": model_id,
        "meta_name": loaded_model.meta_name,
    }


@router.post("/model/deploy")
async def deploy_model(
    org_id: str,
    model_id: str,
    file: Optional[UploadFile] = File(None),
    model_path: Optional[str] = None,
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """
    Deploy a .glyphh model.

    Accepts either:
    - A multipart file upload of a .glyphh file (field name: file)
    - A form field model_path pointing to a filesystem path

    Supports two .glyphh formats:
    - ZIP archive (from CLI packaging): unpacked and loaded via directory path
    - Gzip JSON (from SDK GlyphhModel.to_file): loaded via GlyphhModel.from_file
    """
    from shared.exceptions import ModelLoadException, ModelIncompatibleException

    resolved_path: Optional[str] = None
    tmp_path: Optional[str] = None
    tmp_dir: Optional[str] = None

    try:
        if file is not None:
            # Multipart file upload — save to temp file
            suffix = ".glyphh"
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            try:
                content = await file.read()
                os.write(tmp_fd, content)
            finally:
                os.close(tmp_fd)
            resolved_path = tmp_path
        elif model_path is not None:
            resolved_path = model_path
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a .glyphh file upload or a model_path.",
            )

        # Detect file format by magic bytes
        with open(resolved_path, "rb") as f:
            magic = f.read(2)

        if magic == b"PK":
            # ZIP format (CLI packaging) — unpack and load from directory
            from glyphh.cli.packaging import unpack_model

            tmp_dir = tempfile.mkdtemp(prefix="glyphh_deploy_")
            model_dir = unpack_model(Path(resolved_path), dest=Path(tmp_dir) / model_id)

            loaded_model = await model_manager.load_model_from_directory(
                model_dir=model_dir,
                org_id=org_id,
                model_id=model_id,
            )
        elif magic == b"\x1f\x8b":
            # Gzip JSON format (SDK GlyphhModel.to_file)
            loaded_model = await model_manager.load_model(
                model_path=resolved_path,
                org_id=org_id,
                model_id=model_id,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unrecognized .glyphh file format (magic bytes: {magic!r}). "
                       f"Expected ZIP (CLI package) or gzip (SDK model).",
            )

        return {
            "status": "deployed",
            "model_id": model_id,
            "version": getattr(loaded_model.sdk_model, "version", "unknown"),
            "meta_name": loaded_model.meta_name,
        }

    except ModelLoadException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelIncompatibleException as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp file if we created one
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)




@router.post("/model/undeploy")
async def undeploy_model(
    org_id: str,
    model_id: str,
    request: UndeployRequest = UndeployRequest(),
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """Unload a model from memory, optionally deleting its data."""
    from shared.exceptions import ModelNotFoundException

    try:
        await model_manager.unload_model(
            org_id=org_id,
            model_id=model_id,
            delete_data=request.delete_data,
        )
        return {"status": "undeployed", "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")


@router.post("/model/re-encode")
async def re_encode_model(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """Re-encode all glyphs for a deployed model."""
    from shared.exceptions import ModelNotFoundException

    # Check if model is loaded
    loaded_model = await model_manager.get_model(org_id, model_id)
    if loaded_model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")

    # Check if re-encode is already in progress
    if model_manager.is_model_locked(org_id, model_id):
        raise HTTPException(status_code=409, detail="Re-encode already in progress for this model")

    try:
        result = await model_manager.re_encode_model(
            org_id=org_id,
            model_id=model_id,
        )
        return {"job_id": result.job_id, "status": result.status}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")


@router.delete("/model")
async def delete_model(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """Full delete: unload model + purge all data (config, glyphs, vectors, edges, procedures)."""
    from shared.exceptions import ModelNotFoundException
    from infrastructure.database import async_session_maker
    from sqlalchemy import delete as sql_delete
    from domains.procedures.models import StoredProcedureModel

    # Check if model exists
    loaded_model = await model_manager.get_model(org_id, model_id)
    if loaded_model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")

    # Check if re-encode is in progress
    if model_manager.is_model_locked(org_id, model_id):
        raise HTTPException(status_code=409, detail="Cannot delete model while re-encode is in progress")

    try:
        # Unload model and delete config + glyph data
        await model_manager.unload_model(
            org_id=org_id,
            model_id=model_id,
            delete_data=True,
        )

        # Purge stored procedures
        async with async_session_maker() as session:
            await session.execute(
                sql_delete(StoredProcedureModel).where(
                    StoredProcedureModel.org_id == org_id,
                    StoredProcedureModel.model_id == model_id,
                )
            )
            await session.commit()

        return {"status": "deleted", "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")


# ── Data Management Endpoints ──

@router.get("/data")
async def list_data(
    org_id: str,
    model_id: str,
    limit: int = 20,
    offset: int = 0,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """List glyphs stored for this model with pagination."""
    from infrastructure.database import async_session_maker
    from domains.models.storage import GlyphStorage

    async with async_session_maker() as session:
        storage = GlyphStorage(session)
        count = await storage.count_glyphs(org_id, model_id)
        glyphs = await storage.list_glyphs(org_id, model_id, limit=limit, offset=offset)

    return {
        "total": count,
        "limit": limit,
        "offset": offset,
        "glyphs": [
            {
                "id": str(g.id),
                "concept_text": g.concept_text[:200] if g.concept_text else "",
                "metadata": g.metadata,
                "created_at": g.created_at.isoformat() + "Z" if g.created_at else None,
            }
            for g in glyphs
        ],
    }


@router.get("/data/count")
async def count_data(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """Count glyphs and edges for this model."""
    from infrastructure.database import async_session_maker
    from domains.models.storage import GlyphStorage

    async with async_session_maker() as session:
        storage = GlyphStorage(session)
        glyph_count = await storage.count_glyphs(org_id, model_id)
        vector_count = await storage.count_glyph_vectors(org_id, model_id)

    return {
        "glyphs": glyph_count,
        "vectors": vector_count,
        "model_id": model_id,
    }


@router.delete("/data")
async def clear_data(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """Clear all glyphs and edges for this model without unloading it."""
    from infrastructure.database import async_session_maker
    from domains.models.storage import GlyphStorage

    async with async_session_maker() as session:
        storage = GlyphStorage(session)
        glyphs_deleted, edges_deleted = await storage.delete_model_data(org_id, model_id)
        await session.commit()

    return {
        "status": "cleared",
        "model_id": model_id,
        "glyphs_deleted": glyphs_deleted,
        "edges_deleted": edges_deleted,
    }

