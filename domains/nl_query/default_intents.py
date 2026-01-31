"""
Default Intent Patterns for the Runtime.

These patterns are used when the SDK's IntentEncoder is not available
or as a fallback. They provide basic keyword-based matching for
common Glyphh operations.

Note: The SDK's default_intents.py contains the authoritative patterns
that use HDC similarity. This file provides a simpler fallback.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RuntimeIntentPattern:
    """
    A simple intent pattern for keyword-based matching.
    
    Used as fallback when SDK IntentEncoder is unavailable.
    """
    intent_type: str
    keywords: List[str]
    query_template: Dict[str, Any]
    description: str


# Default patterns for common Glyphh operations
DEFAULT_RUNTIME_PATTERNS = [
    # Similarity Search - find similar concepts
    RuntimeIntentPattern(
        intent_type="similarity_search",
        keywords=[
            "find",
            "search",
            "similar",
            "like",
            "matching",
            "lookup",
            "show me",
        ],
        query_template={
            "operation": "similarity_search",
            "top_k": 10,
        },
        description="Find concepts similar to the query",
    ),
    
    # Fact Tree - verify and explain relationships
    RuntimeIntentPattern(
        intent_type="fact_tree",
        keywords=[
            "verify",
            "explain",
            "prove",
            "evidence",
            "why",
            "how",
            "is it true",
            "show proof",
        ],
        query_template={
            "operation": "fact_tree",
            "max_depth": 3,
        },
        description="Verify a claim and show supporting evidence",
    ),
    
    # Temporal Prediction - predict future states
    RuntimeIntentPattern(
        intent_type="temporal_predict",
        keywords=[
            "predict",
            "forecast",
            "next",
            "after",
            "future",
            "what will",
            "what comes",
        ],
        query_template={
            "operation": "temporal_predict",
            "steps_ahead": 1,
            "beam_width": 3,
        },
        description="Predict future states based on current state",
    ),
    
    # List All - enumerate concepts
    RuntimeIntentPattern(
        intent_type="list_all",
        keywords=[
            "list all",
            "show all",
            "get all",
            "enumerate",
            "all items",
            "everything",
        ],
        query_template={
            "operation": "list",
            "limit": 100,
        },
        description="List all concepts in the namespace",
    ),
    
    # Count - count concepts
    RuntimeIntentPattern(
        intent_type="count",
        keywords=[
            "how many",
            "count",
            "total",
            "number of",
        ],
        query_template={
            "operation": "count",
        },
        description="Count the number of concepts",
    ),
    
    # Compare - compare two concepts
    RuntimeIntentPattern(
        intent_type="compare",
        keywords=[
            "compare",
            "difference",
            "versus",
            "vs",
            "contrast",
            "between",
        ],
        query_template={
            "operation": "compare",
        },
        description="Compare two concepts",
    ),
    
    # Edge Query - find relationships
    RuntimeIntentPattern(
        intent_type="edge_query",
        keywords=[
            "related",
            "relationship",
            "connection",
            "edge",
            "link",
            "connected to",
        ],
        query_template={
            "operation": "get_edges",
        },
        description="Find relationships between concepts",
    ),
]


def get_default_patterns() -> List[RuntimeIntentPattern]:
    """
    Get a copy of the default runtime intent patterns.
    
    Returns:
        List of RuntimeIntentPattern objects
    """
    return DEFAULT_RUNTIME_PATTERNS.copy()


def get_pattern_by_type(intent_type: str) -> RuntimeIntentPattern:
    """
    Get a specific default pattern by intent type.
    
    Args:
        intent_type: The intent type to look up
    
    Returns:
        RuntimeIntentPattern if found
    
    Raises:
        KeyError: If intent_type not found in defaults
    """
    for pattern in DEFAULT_RUNTIME_PATTERNS:
        if pattern.intent_type == intent_type:
            return pattern
    raise KeyError(f"No default pattern with intent_type '{intent_type}'")


def match_keywords(query: str, patterns: List[RuntimeIntentPattern] = None) -> tuple:
    """
    Simple keyword-based intent matching.
    
    Args:
        query: Natural language query
        patterns: Patterns to match against (defaults to DEFAULT_RUNTIME_PATTERNS)
    
    Returns:
        Tuple of (intent_type, confidence, template) or (None, 0.0, None)
    """
    if patterns is None:
        patterns = DEFAULT_RUNTIME_PATTERNS
    
    query_lower = query.lower()
    best_match = None
    best_score = 0.0
    best_template = None
    
    for pattern in patterns:
        matches = sum(1 for kw in pattern.keywords if kw in query_lower)
        if matches > 0:
            # Score based on number of keyword matches
            score = 0.5 + (matches * 0.1)
            score = min(score, 0.95)  # Cap at 0.95
            
            if score > best_score:
                best_score = score
                best_match = pattern.intent_type
                best_template = pattern.query_template.copy()
    
    return best_match, best_score, best_template
