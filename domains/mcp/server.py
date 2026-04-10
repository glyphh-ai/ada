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
                    },
                    "required": ["input"],
                },
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> CallToolResult:
        if name != "think":
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )],
                isError=True,
            )

        input_text = (arguments or {}).get("input", "")
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
            result = await brain.think(input_text)
            elapsed = (datetime.utcnow() - start).total_seconds() * 1000

            response = {
                "response": result.response,
                "capability": result.capability,
                "confidence": round(result.confidence, 3),
                "cognitive_state": result.cognitive_state,
                "gate": result.gate,
                "llm_fallback": result.llm_fallback,
                "elapsed_ms": round(result.elapsed_ms, 1),
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
