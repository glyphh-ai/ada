from __future__ import annotations

from typing import Any

from fastapi import Body, HTTPException, Request
from pydantic import BaseModel

from server.mcp_server import MCPServer
from ..services.audit_log import log_tool_access_decision
from ..services.auth_runtime import enforce_model_access, require_scopes
from .router import api_router


router = api_router(tags=["mcp"])
_server = MCPServer()


class MCPRequest(BaseModel):
    tool: str
    payload: dict[str, Any] = {}


@router.post("/mcp")
def run_mcp_tool(request: Request, payload: MCPRequest = Body(...)) -> dict[str, Any]:
    if not payload.tool:
        raise HTTPException(status_code=400, detail="Tool is required")
    claims = getattr(request.state, "runtime_claims", {}) or {}
    model_id = payload.payload.get("model_id") if isinstance(payload.payload, dict) else None
    try:
        require_scopes(claims, ["mcp:execute"])
        if model_id:
            enforce_model_access(claims, model_id)
    except HTTPException as exc:
        log_tool_access_decision(
            request=request,
            tool=payload.tool,
            decision="deny",
            reason=str(exc.detail),
            model_id=model_id,
        )
        raise
    log_tool_access_decision(
        request=request,
        tool=payload.tool,
        decision="allow",
        model_id=model_id,
    )
    return _server.handle_tool(payload.tool, payload.payload or {})
