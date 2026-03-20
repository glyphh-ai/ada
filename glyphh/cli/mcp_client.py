"""
Shared MCP client for CLI commands.

Handles JSON-RPC 2.0 protocol and SSE response parsing for the Glyphh runtime
MCP endpoint. Used by chat.py, query.py, and any future CLI commands that call
the runtime.

Usage:
    from ..mcp_client import call_tool, list_tools

    # Discover model tools
    tools = list_tools(url, headers)

    # Call a tool
    result = call_tool(url, headers, "nl_query", {"query": "hello"})
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_jsonrpc_id = 0


def _next_id() -> int:
    global _jsonrpc_id
    _jsonrpc_id += 1
    return _jsonrpc_id


def _parse_sse_response(text: str) -> dict:
    """Parse an SSE response that wraps a JSON-RPC message.

    The runtime returns:
        event: message
        data: {"jsonrpc": "2.0", "id": 1, "result": {...}}

    We need to extract the data line and parse the JSON-RPC envelope.
    """
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                continue
    # Fallback: try parsing the whole thing as JSON (non-SSE response)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": {"code": -32700, "message": f"Could not parse response: {text[:200]}"}}


def _unwrap_jsonrpc(raw: dict) -> dict:
    """Unwrap a JSON-RPC 2.0 response into the tool result.

    Success: {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "..."}]}}
    Error:   {"jsonrpc": "2.0", "id": 1, "error": {"code": ..., "message": "..."}}

    Returns the parsed tool result dict, or an error dict.
    """
    if "error" in raw:
        err = raw["error"]
        return {"isError": True, "error": err.get("message", str(err))}

    result = raw.get("result", {})
    content = result.get("content", [])

    if not content:
        return result

    # The tool result is JSON-serialized inside content[0].text
    first = content[0]
    if first.get("type") == "text":
        try:
            parsed = json.loads(first["text"])
            # Wrap in the structure _print_result expects
            return {
                "result": parsed.get("fact_tree"),
                "content": [{"type": "json", "data": parsed}],
                "query_time_ms": parsed.get("query_time_ms"),
                "match_method": parsed.get("match_method"),
                "query_type": parsed.get("query_type"),
            }
        except (json.JSONDecodeError, AttributeError):
            return {"result": first["text"]}

    return result


def call_tool(
    url: str,
    headers: dict[str, str],
    tool_name: str,
    arguments: dict[str, Any],
    timeout: float = 30.0,
) -> dict:
    """Call an MCP tool via JSON-RPC 2.0.

    Args:
        url: Full MCP endpoint URL (e.g. http://localhost:8002/org/model/mcp)
        headers: Request headers (Content-Type, Authorization, Accept)
        tool_name: MCP tool name (e.g. "nl_query", "glyphh_search")
        arguments: Tool arguments dict
        timeout: Request timeout in seconds

    Returns:
        Parsed tool result dict ready for _print_result().
    """
    import httpx

    payload = {
        "jsonrpc": "2.0",
        "id": _next_id(),
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    with httpx.Client(timeout=timeout) as client:
        res = client.post(url, json=payload, headers=headers)

    if res.status_code != 200:
        return {
            "isError": True,
            "error": f"HTTP {res.status_code}: {res.text[:500]}",
            "_status_code": res.status_code,
        }

    # Response may be SSE-wrapped or plain JSON
    content_type = res.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        raw = _parse_sse_response(res.text)
    else:
        raw = res.json()

    return _unwrap_jsonrpc(raw)


def list_tools(
    url: str,
    headers: dict[str, str],
    timeout: float = 10.0,
) -> list[dict]:
    """List available MCP tools via JSON-RPC 2.0.

    Returns a list of tool dicts with name, description, inputSchema.
    """
    import httpx

    payload = {
        "jsonrpc": "2.0",
        "id": _next_id(),
        "method": "tools/list",
        "params": {},
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            res = client.post(url, json=payload, headers=headers)

        if res.status_code != 200:
            logger.warning("tools/list returned %d", res.status_code)
            return []

        content_type = res.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            raw = _parse_sse_response(res.text)
        else:
            raw = res.json()

        if "error" in raw:
            logger.warning("tools/list error: %s", raw["error"])
            return []

        result = raw.get("result", {})
        return result.get("tools", [])

    except Exception as e:
        logger.warning("tools/list failed: %s", e)
        return []
