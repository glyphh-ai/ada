from __future__ import annotations

from typing import Any

from fastapi import Body, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from server.mcp_server import MCPServer
from ..services.audit_log import log_tool_access_decision
from ..services.auth_runtime import enforce_model_access, require_scopes
from ..services.mcp_config_helpers import resolve_mcp_config
from ..core.db import get_db
from .router import api_router


router = api_router(tags=["mcp"])
_server = MCPServer()


class MCPRequest(BaseModel):
    tool: str
    payload: dict[str, Any] = {}


def _run_mcp_tool(
    request: Request,
    payload: MCPRequest,
    *,
    model_id_override: str | None = None,
    org_id_override: str | None = None,
    endpoint_name: str | None = None,
) -> dict[str, Any]:
    if not payload.tool:
        raise HTTPException(status_code=400, detail="Tool is required")
    claims = getattr(request.state, "runtime_claims", {}) or {}
    model_id = payload.payload.get("model_id") if isinstance(payload.payload, dict) else None
    if model_id_override:
        model_id = model_id_override
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
        if org_id_override:
            claim_org = claims.get("org_id")
            if claim_org and claim_org != org_id_override:
                raise HTTPException(status_code=403, detail="Org access denied")
            governance_policies.append(
                {"policy_id": "org_access", "allowed_org_id": org_id_override, "decision": "allow"}
            )
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
    if endpoint_name:
        call_payload["endpoint_name"] = endpoint_name
    if governance_policies:
        call_payload["governance"] = {
            "policies": governance_policies,
            "subject": claims.get("sub") or claims.get("user_id"),
        }
    return _server.handle_tool(payload.tool, call_payload)


@router.post("/mcp")
def run_mcp_tool(request: Request, payload: MCPRequest = Body(...)) -> dict[str, Any]:
    return _run_mcp_tool(request, payload)


@router.post("/org/{org_id}/model/{model_id}/mcp/{endpoint_name}")
def run_scoped_mcp_tool(
    request: Request,
    org_id: str,
    model_id: str,
    endpoint_name: str,
    db: Session = Depends(get_db),
    payload: MCPRequest = Body(...),
) -> dict[str, Any]:
    cfg = resolve_mcp_config(db, model_id=model_id, endpoint_name=endpoint_name)
    if cfg is None:
        raise HTTPException(status_code=404, detail="MCP endpoint not found")
    return _run_mcp_tool(
        request,
        payload,
        model_id_override=model_id,
        org_id_override=org_id,
        endpoint_name=endpoint_name,
    )
