"""
MCP (Model Context Protocol) Domain.

Single /mcp endpoint exposing Ada's think tool.
Provides Streamable HTTP transport for agent integration.
"""

from domains.mcp.app import MCPRoutingMiddleware, create_mcp_session_managers
from domains.mcp.server import create_mcp_server

__all__ = [
    "create_mcp_server",
    "create_mcp_session_managers",
    "MCPRoutingMiddleware",
]
