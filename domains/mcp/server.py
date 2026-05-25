"""
MCP Server — Ada's brain exposed as a single tool.

One tool: think. Natural language in, structured response out.
Ada routes internally, processes, and returns.
"""

import json
import logging
from datetime import datetime
from typing import Any

from mcp.server.lowlevel.server import Server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from domains.auth.service import AuthService

logger = logging.getLogger(__name__)


def create_mcp_server(brain: Any, auth_service: AuthService) -> Server:
    """Create the MCP server with a single `think` tool.

    Args:
        brain: The Brain instance (domains.brain.think.Brain)
        auth_service: AuthService for token validation
    """
    app = Server("glyphh")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="think",
                description=(
                    "Send a natural language request to Ada's brain. "
                    "Ada routes internally to the right capability "
                    "(firewall, voice, memory, etc.) and returns a "
                    "structured response. Use this for ALL interactions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Natural language input to process",
                        },
                        "tool": {
                            "type": "string",
                            "description": "Which tool is calling (e.g. claude-code, claude-desktop, gemini). Used to tag memory threads.",
                        },
                    },
                    "required": ["input"],
                },
            ),
            Tool(
                name="remember",
                description=(
                    "Store a decision/fact/constraint in persistent memory. "
                    "No LLM call — pure capture. Use from a harness hook (e.g. "
                    "PreCompact) so a decision survives the context window."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The decision or fact to remember"},
                        "speaker": {"type": "string", "description": "Source tag (default 'incoming')"},
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="recall",
                description=(
                    "Retrieve relevant facts/decisions from memory. No LLM call "
                    "— pure retrieval, zero tokens. Use from a hook (e.g. "
                    "UserPromptSubmit) to re-ground the agent every turn. "
                    "Returns grounded facts for the caller to inject into context."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to recall about"},
                        "top_k": {"type": "integer", "description": "Max facts (default 5)"},
                        "min_confidence": {"type": "number", "description": "Drop facts below this score (default 0)"},
                    },
                    "required": ["query"],
                },
            ),
        ]

    def _ok(payload: dict) -> CallToolResult:
        return CallToolResult(content=[TextContent(
            type="text", text=json.dumps(payload, ensure_ascii=False),
        )])

    def _err(msg: str) -> CallToolResult:
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": msg}))],
            isError=True,
        )

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> CallToolResult:
        args = arguments or {}

        # ── LLM-free memory tools (for harness hooks) ──
        if name == "remember":
            text = (args.get("text") or args.get("input") or "").strip()
            if not text:
                return _err("Missing 'text'")
            try:
                return _ok(brain.remember(text, speaker=args.get("speaker", "incoming")))
            except Exception as e:
                logger.error(f"remember() failed: {e}", exc_info=True)
                return _err(str(e))

        if name == "recall":
            query = (args.get("query") or args.get("input") or "").strip()
            if not query:
                return _err("Missing 'query'")
            try:
                facts = brain.recall(
                    query,
                    top_k=int(args.get("top_k", 5)),
                    min_confidence=float(args.get("min_confidence", 0.0)),
                    exclude_speakers=args.get("exclude_speakers"),
                )
                return _ok({"facts": facts, "count": len(facts)})
            except Exception as e:
                logger.error(f"recall() failed: {e}", exc_info=True)
                return _err(str(e))

        if name != "think":
            return _err(f"Unknown tool: {name}")

        input_text = (arguments or {}).get("input", "")
        tool_name = (arguments or {}).get("tool", "unknown")
        if not input_text:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({"error": "Missing 'input' parameter"}),
                )],
                isError=True,
            )

        # ── Metering: count every MCP operation ──
        try:
            from glyphh.metering import get_meter
            from glyphh.licensing import get_current_license

            meter = get_meter()
            count = meter.record("ada")
            license_info = get_current_license()

            if not license_info.is_unlimited:
                limit = license_info.max_encodings_per_month
                threshold = license_info.encoding_warning_threshold()
                if count >= limit:
                    logger.warning(
                        f"Monthly operation limit exceeded "
                        f"({count:,}/{limit:,}). Tier: {license_info.tier}"
                    )
                elif threshold and count >= threshold:
                    logger.warning(
                        f"Approaching monthly limit "
                        f"({count:,}/{limit:,}, {count/limit*100:.0f}%)"
                    )
        except Exception:
            pass  # Metering is best-effort

        try:
            start = datetime.utcnow()
            result = await brain.think(input_text, tool=tool_name)
            elapsed = (datetime.utcnow() - start).total_seconds() * 1000

            response = {
                "response": result.response,
                "capability": result.capability,
                "confidence": round(result.confidence, 3),
                "gate": result.gate,
                "llm_assisted": result.llm_assisted,
                "elapsed_ms": round(result.elapsed_ms, 1),
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "tokens_saved": result.tokens_saved,
            }

            logger.info(f"think() completed in {elapsed:.1f}ms")

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, ensure_ascii=False),
                )],
            )

        except Exception as e:
            logger.error(f"think() failed: {e}", exc_info=True)
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}),
                )],
                isError=True,
            )

    return app
