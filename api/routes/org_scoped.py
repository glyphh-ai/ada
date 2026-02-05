"""
Org-Scoped API Routes for Glyphh Runtime (Cloud Mode).

Endpoints scoped by organization ID and model ID for multi-tenant cloud deployments.
The URL pattern /{org_id}/{model_id}/... ensures proper isolation between
organizations and models.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from domains.auth.service import AuthService, User
from domains.mcp.server import MCPServer
from domains.models.storage import GlyphStorage
from domains.query.service import QueryService
from infrastructure.config import get_settings
from infrastructure.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}", tags=["org-scoped"])
settings = get_settings()


def build_namespace(org_id: str, model_id: str) -> str:
    """Build namespace from org_id and model_id.
    
    Format: {org_id}/{model_id}
    
    This matches the URL pattern and ensures:
    - Multi-tenant isolation (different orgs can't access each other's data)
    - Model-level isolation (each model has its own vector space)
    """
    return f"{org_id}/{model_id}"


# Dependency injection
async def get_auth_service() -> AuthService:
    """Get auth service."""
    return AuthService()


async def validate_org_access(
    org_id: str,
    model_id: str,
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """
    Validate that the user has access to the org and model.
    
    In cloud mode, extracts org_id from JWT claims and validates.
    """
    if settings.deployment_mode == "local":
        # Local mode: allow all access
        return User(
            user_id="local",
            namespaces={"*": {"read", "write", "admin"}},
            org_id=org_id,
        )
    
    # TODO: Extract token from request and validate org access
    raise HTTPException(
        status_code=501,
        detail="Org-scoped authentication not yet implemented"
    )


async def get_mcp_server(
    org_id: str,
    model_id: str,
    db: AsyncSession = Depends(get_db),
) -> MCPServer:
    """Get MCP server for org/model."""
    from main import model_manager
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    storage = GlyphStorage(db)
    query_service = QueryService(storage, model_manager)
    auth_service = AuthService()
    
    return MCPServer(query_service, auth_service)


# MCP Endpoint
@router.post("/mcp")
async def mcp_endpoint(
    org_id: str,
    model_id: str,
    request: Dict[str, Any],
    mcp_server: MCPServer = Depends(get_mcp_server),
    user: User = Depends(validate_org_access),
) -> Dict[str, Any]:
    """
    MCP endpoint for org-scoped model access.
    
    Handles MCP tool calls with org-level authentication.
    The namespace is automatically set to {org_id}_{model_id} for proper isolation.
    """
    tool_name = request.get("tool")
    arguments = request.get("arguments", {})
    
    if not tool_name:
        raise HTTPException(status_code=400, detail="Missing 'tool' field")
    
    # Build namespace from org_id and model_id for proper isolation
    arguments["namespace"] = build_namespace(org_id, model_id)
    
    # Handle tool call (using user's token for auth)
    response = await mcp_server.handle_tool_call(
        tool_name=tool_name,
        arguments=arguments,
        auth_token="",  # Already validated via validate_org_access
    )
    
    return response.to_dict()


@router.get("/mcp/tools")
async def list_mcp_tools(
    org_id: str,
    model_id: str,
    mcp_server: MCPServer = Depends(get_mcp_server),
) -> Dict[str, Any]:
    """
    List available MCP tools.
    
    Returns tool schemas for the model.
    """
    return {"tools": mcp_server.get_tools_list()}


# Listener Endpoint (WebSocket)
@router.websocket("/listener")
async def listener_websocket(
    websocket: WebSocket,
    org_id: str,
    model_id: str,
):
    """
    WebSocket listener for real-time glyph ingestion.
    
    Accepts WebSocket connections for streaming glyph creation.
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "create_glyph":
                # TODO: Implement glyph creation via listener service
                await websocket.send_json({
                    "type": "glyph_created",
                    "status": "not_implemented",
                    "message": "Listener service not yet implemented"
                })
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {org_id}/{model_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


# HTTP Listener Endpoint (batch ingestion)
@router.post("/listener")
async def listener_batch(
    org_id: str,
    model_id: str,
    request: Dict[str, Any],
    user: User = Depends(validate_org_access),
) -> Dict[str, Any]:
    """
    HTTP endpoint for batch glyph ingestion.
    
    Accepts an array of concepts and creates glyphs in batch.
    The namespace is automatically set to {org_id}_{model_id} for proper isolation.
    """
    concepts = request.get("concepts", [])
    metadata = request.get("metadata", {})
    
    if not concepts:
        raise HTTPException(status_code=400, detail="No concepts provided")
    
    # Build namespace for proper isolation
    namespace = build_namespace(org_id, model_id)
    
    # TODO: Implement batch creation via listener service
    return {
        "status": "not_implemented",
        "message": "Batch listener not yet implemented",
        "namespace": namespace,
        "concepts_received": len(concepts)
    }
