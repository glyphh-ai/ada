"""
MCP (Model Context Protocol) Domain.

Proper MCP server implementation using the official mcp Python SDK.
Provides Streamable HTTP transport for agent integration.
"""

from domains.mcp.app import MCPRoutingMiddleware, create_mcp_session_managers
from domains.mcp.server import ToolHandler, create_mcp_server

__all__ = [
    "create_mcp_server",
    "create_mcp_session_managers",
    "MCPRoutingMiddleware",
    "ToolHandler",
]
