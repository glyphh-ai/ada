from __future__ import annotations

from typing import Any

from fastapi import Body, HTTPException
from pydantic import BaseModel

from ...server.mcp_server import MCPServer
from .router import api_router


router = api_router(tags=["mcp"])
_server = MCPServer()


class MCPRequest(BaseModel):
    tool: str
    payload: dict[str, Any] = {}


@router.post("/mcp")
def run_mcp_tool(payload: MCPRequest = Body(...)) -> dict[str, Any]:
    if not payload.tool:
        raise HTTPException(status_code=400, detail="Tool is required")
    return _server.handle_tool(payload.tool, payload.payload or {})
