"""
Natural Language Query Domain.

Routes NL queries to the appropriate operation via model encoders.

Key Components:
- NLQueryService: Orchestrates query routing (auto-schema → similarity search)
- SchemaIndex: Runtime schema index for fast NL query matching
- SchemaIndexMetrics: Metrics for schema index performance

Design Principle: "When your LLM can't afford to be wrong, sidecar it with Glyphh"
"""

# Use lazy imports to avoid circular dependencies and missing deps during testing
__all__ = [
    "NLQueryService",
    "NLQueryResult",
    "SchemaIndex",
    "SchemaIndexMetrics",
]


def __getattr__(name):
    """Lazy import attributes to avoid import errors when deps are missing."""
    if name == "NLQueryService":
        from domains.nl_query.service import NLQueryService
        return NLQueryService
    elif name == "NLQueryResult":
        from domains.nl_query.service import NLQueryResult
        return NLQueryResult
    elif name == "SchemaIndex":
        from domains.nl_query.schema_index import SchemaIndex
        return SchemaIndex
    elif name == "SchemaIndexMetrics":
        from domains.nl_query.schema_index import SchemaIndexMetrics
        return SchemaIndexMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
