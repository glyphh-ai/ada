from __future__ import annotations

from typing import Any, Dict

from jsonschema import ValidationError, validate


MCP_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["jsonrpc", "id"],
    "properties": {
        "jsonrpc": {"const": "2.0"},
        "id": {"type": ["string", "number"]},
        "result": {
            "type": "object",
            "required": ["content", "isError"],
            "properties": {
                "content": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                },
                "isError": {"type": "boolean"},
                "structuredContent": {"type": "object"},
                "_meta": {"type": "object"},
            },
            "additionalProperties": True,
        },
        "error": {
            "type": "object",
            "required": ["code", "message"],
            "properties": {
                "code": {"type": "integer"},
                "message": {"type": "string"},
                "data": {},
            },
            "additionalProperties": True,
        },
    },
    "oneOf": [
        {"required": ["result"]},
        {"required": ["error"]},
    ],
    "additionalProperties": True,
}


def validate_mcp_response(payload: Dict[str, Any]) -> None:
    validate(instance=payload, schema=MCP_RESPONSE_SCHEMA)


def format_validation_error(exc: ValidationError) -> str:
    location = ".".join(str(part) for part in exc.path) or "root"
    return f"{location}: {exc.message}"
