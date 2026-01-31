"""
Natural Language Query Domain.

Provides hybrid rules-first + LLM-fallback query translation.

Key Components:
- IntentMatcher: Rules-based matching using HDC similarity
- NLQueryService: Orchestrates rules + LLM fallback
- LLMFallback: Optional LLM-based translation (Phi-3.5-mini-instruct)

Design Principle: "When your LLM can't be wrong, sidecar it with Glyphh"
"""

# Use lazy imports to avoid circular dependencies and missing deps during testing
__all__ = [
    "IntentMatcher",
    "IntentMatch",
    "NLQueryService",
    "NLQueryResult",
    "LLMFallback",
]


def __getattr__(name):
    """Lazy import attributes to avoid import errors when deps are missing."""
    if name == "IntentMatcher":
        from domains.nl_query.intent_matcher import IntentMatcher
        return IntentMatcher
    elif name == "IntentMatch":
        from domains.nl_query.intent_matcher import IntentMatch
        return IntentMatch
    elif name == "NLQueryService":
        from domains.nl_query.service import NLQueryService
        return NLQueryService
    elif name == "NLQueryResult":
        from domains.nl_query.service import NLQueryResult
        return NLQueryResult
    elif name == "LLMFallback":
        from domains.nl_query.llm_fallback import LLMFallback
        return LLMFallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
