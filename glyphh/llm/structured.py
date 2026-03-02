"""Structured output schemas and parsing for LLM responses.

Qwen3 supports function calling (tool_use) but not json_schema mode.
This module defines tool schemas for each integration point and robust
parsing for the model's output.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResult:
    """Result from a structured LLM call."""

    data: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0


# ── Tool schemas for each integration point ──

INTENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_intent",
            "description": "Classify the user's intent from query text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The primary action verb (e.g. get, create, delete, search)",
                    },
                    "target": {
                        "type": "string",
                        "description": "The primary target noun (e.g. file, user, message)",
                    },
                    "domain": {
                        "type": "string",
                        "description": "The inferred domain (e.g. filesystem, messaging, general)",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0-1.0",
                    },
                },
                "required": ["action", "target", "domain", "confidence"],
            },
        },
    },
]

SLOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fill_slots",
            "description": "Extract function arguments from the user query and context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arguments": {
                        "type": "object",
                        "description": "Map of parameter_name -> extracted_value",
                    },
                },
                "required": ["arguments"],
            },
        },
    },
]

ARBITRATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "arbitrate",
            "description": (
                "Decide whether the pipeline result is correct "
                "given the full context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "accept": {
                        "type": "boolean",
                        "description": "Accept the pipeline result?",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for accept/reject",
                    },
                    "override_action": {
                        "type": "string",
                        "description": "Corrected action verb if rejecting",
                    },
                    "override_target": {
                        "type": "string",
                        "description": "Corrected target noun if rejecting",
                    },
                },
                "required": ["accept", "reason"],
            },
        },
    },
]


def build_classify_tools(function_names: list[str]) -> list[dict[str, Any]]:
    """Build a tool schema that constrains LLM output to available functions.

    The function names are passed as an enum so the LLM can only select
    from the registered function set.

    Args:
        function_names: List of available function names (from begin() schemas).

    Returns:
        Tool schema list for structured_generate().
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "classify_and_call",
                "description": (
                    "Classify the user's intent and select function(s) to call "
                    "with their arguments."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "functions": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": function_names,
                            },
                            "description": "Ordered list of function(s) to call",
                        },
                        "arguments": {
                            "type": "object",
                            "description": (
                                "Map of function_name to {param_name: value}. "
                                "Only include arguments that can be extracted "
                                "from the query."
                            ),
                        },
                        "confidence": {
                            "type": "number",
                            "description": (
                                "Classification confidence from 0.0 to 1.0. "
                                "1.0 = clearly matches a function. "
                                "Below 0.3 = ambiguous or no match."
                            ),
                        },
                    },
                    "required": ["functions", "arguments", "confidence"],
                },
            },
        },
    ]


def format_function_descriptions(functions: list[dict[str, Any]]) -> str:
    """Format function schemas into a human-readable description block.

    Used to build the SCHEMA_CLASSIFY_SYSTEM prompt at configure() time.

    Args:
        functions: List of function schema dicts [{name, description, parameters}].

    Returns:
        Multi-line string listing each function with its signature.
    """
    lines = []
    for func in functions:
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        required = set(func.get("parameters", {}).get("required", []))

        # Build param signature
        param_parts = []
        for pname, pdef in params.items():
            ptype = pdef.get("type", "string")
            req_mark = "" if pname in required else "?"
            param_parts.append(f"{pname}{req_mark}: {ptype}")

        sig = ", ".join(param_parts)
        line = f"- {name}({sig})"
        if desc:
            line += f" — {desc}"
        lines.append(line)

    return "\n".join(lines)


def parse_tool_call_response(raw_text: str) -> dict[str, Any]:
    """Parse structured output from LLM response text.

    Handles multiple formats:
      1. Direct JSON object
      2. Markdown-fenced JSON
      3. Embedded JSON within surrounding text
    """
    text = raw_text.strip()
    if not text:
        return {}

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner = "\n".join(lines[1:])
        if inner.rstrip().endswith("```"):
            inner = inner.rstrip()[:-3]
        inner = inner.strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    # Extract first JSON object from text
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}
