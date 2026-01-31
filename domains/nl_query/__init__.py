"""
Natural Language Query Domain.

Provides hybrid rules-first + LLM-fallback query translation.

Key Components:
- IntentMatcher: Rules-based matching using HDC similarity
- NLQueryService: Orchestrates rules + LLM fallback
- LLMFallback: Optional LLM-based translation (Phi-3.5-mini-instruct)

Design Principle: "When your LLM can't be wrong, sidecar it with Glyphh"
"""

from domains.nl_query.intent_matcher import IntentMatcher, IntentMatch
from domains.nl_query.service import NLQueryService, NLQueryResult

__all__ = [
    "IntentMatcher",
    "IntentMatch",
    "NLQueryService",
    "NLQueryResult",
]

# Optional LLM fallback - only import if available
try:
    from domains.nl_query.llm_fallback import LLMFallback
    __all__.append("LLMFallback")
except ImportError:
    pass
