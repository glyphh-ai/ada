"""
MCP ASGI Application for Glyphh Runtime.

Creates the MCP session manager using the official mcp SDK and provides
an ASGI middleware that routes /{org_id}/{model_id}/mcp to the SDK handler,
injecting org_id and model_id into context variables.
"""

import logging
import re
from typing import Callable, Optional

from starlette.types import ASGIApp, Receive, Scope, Send

from domains.mcp.server import current_model_id, current_org_id

logger = logging.getLogger(__name__)

# Regex to match /{org_id}/{model_id}/mcp paths (with optional trailing path)
_MCP_PATH_RE = re.compile(r"^/([^/]+)/([^/]+)/mcp(/.*)?$")


class MCPRoutingMiddleware:
    """
    ASGI middleware that intercepts /{org_id}/{model_id}/mcp requests
    and forwards them to the MCP SDK's session manager.

    Non-MCP requests are passed through to the main FastAPI app unchanged.

    Uses a getter function for the session manager since it's created during
    lifespan (after DB init), not at import time.
    """

    def __init__(self, app: ASGIApp, mcp_app_getter: Callable[[], Optional["StreamableHTTPSessionManager"]]):
        self.app = app
        self._mcp_app_getter = mcp_app_getter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            match = _MCP_PATH_RE.match(path)
            if match:
                session_manager = self._mcp_app_getter()
                if session_manager is not None:
                    org_id = match.group(1)
                    model_id = match.group(2)

                    # Set context variables for the tool handlers
                    org_token = current_org_id.set(org_id)
                    model_token = current_model_id.set(model_id)

                    try:
                        await session_manager.handle_request(scope, receive, send)
                    finally:
                        current_org_id.reset(org_token)
                        current_model_id.reset(model_token)
                    return

        # Not an MCP request — pass through to FastAPI
        await self.app(scope, receive, send)


def create_mcp_session_manager(query_service, auth_service):
    """
    Create the MCP StreamableHTTPSessionManager.

    Returns a session manager that handles JSON-RPC 2.0 MCP protocol.
    Must be started via `async with manager.run():` in the app lifespan.
    The middleware calls `manager.handle_request()` for each request.
    """
    from domains.mcp.server import create_mcp_server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings

    server = create_mcp_server(query_service, auth_service)

    session_manager = StreamableHTTPSessionManager(
        app=server,
        stateless=True,
        json_response=False,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
    )

    return session_manager
