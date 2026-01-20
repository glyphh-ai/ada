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
    governance_policies: list[dict] = []
    try:
        scopes = require_scopes(claims, ["mcp:execute"])
        governance_policies.append(
            {
                "policy_id": "scope_check",
                "required": ["mcp:execute"],
                "granted": scopes,
                "decision": "allow",
            }
        )
        if model_id:
            governance_policies.append(enforce_model_access(claims, model_id) | {"decision": "allow"})
    except HTTPException as exc:
        governance_policies.append(
            {
                "policy_id": "runtime_auth",
                "required": ["mcp:execute"],
                "decision": "deny",
                "reason": str(exc.detail),
            }
        )
        log_tool_access_decision(
            request=request,
            tool=payload.tool,
            decision="deny",
            reason=str(exc.detail),
            model_id=model_id,
        )
        raise
    request.state.governance_policies = governance_policies
    log_tool_access_decision(
        request=request,
        tool=payload.tool,
        decision="allow",
        model_id=model_id,
    )
    call_payload = dict(payload.payload or {})
    if governance_policies:
        call_payload["governance"] = {
            "policies": governance_policies,
            "subject": claims.get("sub") or claims.get("user_id"),
        }
    return _server.handle_tool(payload.tool, call_payload)
