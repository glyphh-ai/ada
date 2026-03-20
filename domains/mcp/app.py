"""
MCP ASGI Application for Glyphh Runtime.

Creates dual MCP session managers (JSON + SSE) using the official mcp SDK
and provides an ASGI middleware that routes /{org_id}/{model_id}/mcp to the
correct handler based on the client's Accept header.

- Accept: application/json (no SSE) → JSON transport (Studio, Platform)
- Accept: text/event-stream         → SSE transport (Claude Code, CLI agents)
"""

import logging
import re
from typing import Callable, Optional, Tuple

from starlette.types import ASGIApp, Receive, Scope, Send

from domains.mcp.server import current_model_id, current_org_id

logger = logging.getLogger(__name__)

# Regex to match /{org_id}/{model_id}/mcp paths (with optional trailing path)
_MCP_PATH_RE = re.compile(r"^/([^/]+)/([^/]+)/mcp(/.*)?$")


def _wants_sse(scope: Scope) -> bool:
    """Return True if the client Accept header includes text/event-stream."""
    for key, value in scope.get("headers", []):
        if key == b"accept":
            return b"text/event-stream" in value
    return False


class MCPRoutingMiddleware:
    """
    ASGI middleware that intercepts /{org_id}/{model_id}/mcp requests
    and forwards them to the appropriate MCP session manager based on
    the client's Accept header.

    Non-MCP requests are passed through to the main FastAPI app unchanged.
    """

    def __init__(
        self,
        app: ASGIApp,
        mcp_app_getter: Callable[[], Optional[Tuple]],
    ):
        self.app = app
        self._mcp_app_getter = mcp_app_getter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            match = _MCP_PATH_RE.match(path)
            if match:
                managers = self._mcp_app_getter()
                if managers is not None:
                    json_manager, sse_manager = managers
                    org_id = match.group(1)
                    model_id = match.group(2)

                    # Pick transport based on Accept header
                    manager = sse_manager if _wants_sse(scope) else json_manager

                    # Set context variables for the tool handlers
                    org_token = current_org_id.set(org_id)
                    model_token = current_model_id.set(model_id)

                    try:
                        await manager.handle_request(scope, receive, send)
                    finally:
                        current_org_id.reset(org_token)
                        current_model_id.reset(model_token)
                    return

        # Not an MCP request — pass through to FastAPI
        await self.app(scope, receive, send)


def create_mcp_session_managers(query_service, auth_service):
    """
    Create dual MCP session managers: one for JSON, one for SSE.

    Returns (json_manager, sse_manager). Both must be started via
    `async with manager.run():` in the app lifespan.
    """
    from domains.mcp.server import create_mcp_server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings

    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    server = create_mcp_server(query_service, auth_service)

    json_manager = StreamableHTTPSessionManager(
        app=server,
        stateless=True,
        json_response=True,
        security_settings=security,
    )

    sse_manager = StreamableHTTPSessionManager(
        app=server,
        stateless=True,
        json_response=False,
        security_settings=security,
    )

    return json_manager, sse_manager
