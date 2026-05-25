"""
Shared MCP client for Glyphh hooks.

Hooks call the running Glyphh runtime's MCP endpoint (the same one the shell
uses) to remember() decisions and recall() relevant memory. No SDK dependency
— just a JSON-RPC POST over HTTP, so the hook scripts stay tiny and fast.

Endpoint defaults to http://localhost:8002/mcp; override with GLYPHH_MCP_URL.
All calls are best-effort: if the runtime is down, hooks degrade to no-ops
rather than ever blocking the agent.
"""

from __future__ import annotations

import json
import os
import urllib.request

MCP_URL = os.environ.get("GLYPHH_MCP_URL", "http://localhost:8002/mcp")
TIMEOUT = float(os.environ.get("GLYPHH_HOOK_TIMEOUT", "4"))


def _call(tool: str, arguments: dict) -> dict | None:
    """Call an MCP tool; return the parsed result dict, or None on any failure."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }).encode()
    req = urllib.request.Request(
        MCP_URL, data=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode())
    except Exception:
        return None  # runtime down / unreachable → degrade gracefully

    try:
        text = body["result"]["content"][0]["text"]
        return json.loads(text)
    except Exception:
        return None


def recall(query: str, top_k: int = 5, min_confidence: float = 0.0,
           exclude_speakers: list | None = None) -> list[dict]:
    """Return [{content, confidence}, ...] relevant to query (empty on failure)."""
    args = {"query": query, "top_k": top_k, "min_confidence": min_confidence}
    if exclude_speakers:
        args["exclude_speakers"] = exclude_speakers
    res = _call("recall", args)
    if not res:
        return []
    return res.get("facts", [])


def remember(text: str, speaker: str = "incoming") -> bool:
    """Store a decision/fact. Returns True if stored."""
    res = _call("remember", {"text": text, "speaker": speaker})
    return bool(res and res.get("stored"))
