"""
MCP ASGI Application for Glyphh Runtime.

Single /mcp endpoint. No org_id, no model_id. One door into the brain.

Transport selection:
- Accept: application/json → JSON transport (Studio, Platform)
- Default / text/event-stream → SSE transport (Claude Code, CLI agents)
"""

import logging
import re
from typing import Callable, Optional, Tuple

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Single MCP path — no org/model in the URL
_MCP_PATH_RE = re.compile(r"^/mcp(/.*)?$")


def _wants_json(scope: Scope) -> bool:
    """Return True if the client explicitly accepts application/json without SSE."""
    for key, value in scope.get("headers", []):
        if key == b"accept":
            return (
                b"application/json" in value
                and b"text/event-stream" not in value
            )
    return False


class MCPRoutingMiddleware:
    """ASGI middleware that intercepts /mcp requests and forwards to
    the MCP session manager. No org/model scoping — Ada is the brain."""

    def __init__(
        self,
        app: ASGIApp,
        mcp_app_getter: Callable[[], Optional[Tuple]],
    ):
        self.app = app
        self._mcp_app_getter = mcp_app_getter

    def _get_origin(self, scope: Scope) -> Optional[str]:
        for key, value in scope.get("headers", []):
            if key == b"origin":
                return value.decode("latin-1")
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            match = _MCP_PATH_RE.match(path)
            if match:
                origin = self._get_origin(scope)

                # CORS preflight
                if scope.get("method") == "OPTIONS" and origin:
                    await self.app(scope, receive, send)
                    return

                managers = self._mcp_app_getter()
                if managers is not None:
                    json_manager, sse_manager = managers
                    manager = json_manager if _wants_json(scope) else sse_manager

                    # Inject CORS headers (MCP SDK runs outside FastAPI's CORSMiddleware)
                    async def cors_send(message) -> None:
                        if message["type"] == "http.response.start" and origin:
                            headers = list(message.get("headers", []))
                            headers.append((b"access-control-allow-origin", origin.encode()))
                            headers.append((b"access-control-allow-headers", b"*"))
                            headers.append((b"access-control-allow-methods", b"GET, POST, OPTIONS"))
                            message = {**message, "headers": headers}
                        await send(message)

                    await manager.handle_request(scope, receive, cors_send)
                    return

        await self.app(scope, receive, send)


def create_mcp_session_managers(brain, auth_service):
    """Create dual MCP session managers (JSON + SSE).

    Args:
        brain: The Brain instance (domains.brain.think.Brain)
        auth_service: AuthService for token validation
    """
    from domains.mcp.server import create_mcp_server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings

    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    server = create_mcp_server(brain, auth_service)

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
