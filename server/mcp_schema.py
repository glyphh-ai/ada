from __future__ import annotations

from typing import Any, Dict

from jsonschema import ValidationError, validate


MCP_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "version",
        "status",
        "answer",
        "facts",
        "reasons",
        "grounding",
        "citations",
        "freshness",
        "constraints_applied",
        "data_provenance",
        "redactions",
        "traceability",
    ],
    "properties": {
        "version": {"type": "string"},
        "status": {"type": "string", "enum": ["ok", "error"]},
        "answer": {
            "type": "object",
            "required": ["text", "format"],
            "properties": {
                "text": {"type": "string"},
                "format": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "facts": {"type": "array", "items": {"type": "object"}},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "grounding": {
            "type": "object",
            "required": ["mode", "score", "strict"],
            "properties": {
                "mode": {"type": "string"},
                "score": {"type": "number"},
                "strict": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        "citations": {"type": "array", "items": {"type": "object"}},
        "freshness": {
            "type": "object",
            "required": ["as_of", "max_age_seconds", "policy"],
            "properties": {
                "as_of": {"type": ["string", "null"]},
                "max_age_seconds": {"type": "number"},
                "policy": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "constraints_applied": {
            "type": "object",
            "required": ["roles", "segments", "time_window", "allowlist_sources"],
            "properties": {
                "roles": {"type": "array", "items": {"type": "string"}},
                "segments": {"type": "array", "items": {"type": "string"}},
                "time_window": {"type": ["string", "null"]},
                "allowlist_sources": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        },
        "data_provenance": {
            "type": "object",
            "required": ["origin", "pipeline_id", "dataset_version"],
            "properties": {
                "origin": {"type": ["string", "null"]},
                "pipeline_id": {"type": ["string", "null"]},
                "dataset_version": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        },
        "redactions": {
            "type": "object",
            "required": ["pii_removed", "fields"],
            "properties": {
                "pii_removed": {"type": "boolean"},
                "fields": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        },
        "traceability": {
            "type": "object",
            "required": [
                "request_id",
                "runtime_id",
                "model_id",
                "pipeline_id",
                "glyph_ids",
                "filters",
                "temporal_edges",
                "path_edges",
                "sources",
            ],
            "properties": {
                "request_id": {"type": ["string", "null"]},
                "runtime_id": {"type": ["string", "null"]},
                "model_id": {"type": ["string", "null"]},
                "pipeline_id": {"type": ["string", "null"]},
                "glyph_ids": {"type": "array", "items": {"type": "string"}},
                "filters": {"type": "object"},
                "temporal_edges": {"type": "array", "items": {"type": "object"}},
                "path_edges": {"type": "array", "items": {"type": "object"}},
                "sources": {"type": "array", "items": {"type": "object"}},
            },
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}


def validate_mcp_response(payload: Dict[str, Any]) -> None:
    validate(instance=payload, schema=MCP_RESPONSE_SCHEMA)


def format_validation_error(exc: ValidationError) -> str:
    location = ".".join(str(part) for part in exc.path) or "root"
    return f"{location}: {exc.message}"
