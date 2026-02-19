"""
Org-Scoped API Routes for Glyphh Runtime (Cloud Mode).

Endpoints scoped by org_id and model_id for multi-tenant cloud deployments.
URL pattern: /{org_id}/{model_id}/...

No namespace concept — org_id and model_id are passed directly to services.
"""

import logging
import tempfile
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from domains.auth.service import AuthService
from domains.mcp.server import MCPServer
from domains.query.service import QueryService
from infrastructure.config import get_settings
from shared.auth import AuthenticatedUser, get_current_user

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
    Validate that the authenticated user has access to the org and model.

    Uses JWT authentication from shared/auth.py which handles:
    - Local mode bypass (returns mock user)
    - JWT token validation and claim extraction
    - Token expiry and signature verification

    Additionally validates that the JWT org_id matches the URL org_id.
    """
    if current_user.org_id == "local-dev-org":
        return current_user

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
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """
    MCP endpoint for org-scoped model access.

    Accepts JWT authentication from Studio for direct queries.
    Validates org_id in URL matches org_id in JWT token.
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
    """
    from shared.exceptions import ModelLoadException, ModelIncompatibleException

    resolved_path: Optional[str] = None
    tmp_path: Optional[str] = None

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

        loaded_model = await model_manager.load_model(
            model_path=resolved_path,
            org_id=org_id,
            model_id=model_id,
        )

        return {
            "status": "deployed",
            "model_id": model_id,
            "version": loaded_model.sdk_model.version,
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

