"""
Natural Language Query Domain.

Provides hybrid rules-first + LLM-fallback query translation.
"""

from domains.nl_query.intent_matcher import IntentMatcher
from domains.nl_query.service import NLQueryService

__all__ = ["IntentMatcher", "NLQueryService"]
