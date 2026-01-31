"""
MCP (Model Context Protocol) Domain.

Implements MCP server for agent integration with Glyphh queries.
"""

from domains.mcp.server import MCPResponse, MCPServer, MCPToolSchema

__all__ = ["MCPServer", "MCPToolSchema", "MCPResponse"]
