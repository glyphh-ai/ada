"""
Default Intent Patterns for Natural Language Query.

Built-in patterns for common query types. Models can override
or extend these patterns.
"""

from typing import Dict, List


# Intent categories
INTENT_SIMILARITY_SEARCH = "similarity_search"
INTENT_FACT_TREE = "fact_tree"
INTENT_TEMPORAL_PREDICTION = "temporal_prediction"
INTENT_GLYPH_LOOKUP = "glyph_lookup"
INTENT_EDGE_QUERY = "edge_query"


# Default patterns for each intent
DEFAULT_INTENT_PATTERNS: Dict[str, List[str]] = {
    INTENT_SIMILARITY_SEARCH: [
        "find similar to {query}",
        "what's like {query}",
        "search for {query}",
        "find {query}",
        "look up {query}",
        "similar to {query}",
        "related to {query}",
        "concepts like {query}",
        "things similar to {query}",
        "what is similar to {query}",
        "show me things like {query}",
        "find concepts related to {query}",
    ],
    
    INTENT_FACT_TREE: [
        "verify {claim}",
        "explain {claim}",
        "prove {claim}",
        "is it true that {claim}",
        "check if {claim}",
        "validate {claim}",
        "confirm {claim}",
        "evidence for {claim}",
        "support for {claim}",
        "why is {claim}",
        "how do we know {claim}",
        "what supports {claim}",
    ],
    
    INTENT_TEMPORAL_PREDICTION: [
        "predict {state}",
        "what comes after {state}",
        "what happens next after {state}",
        "forecast {state}",
        "future of {state}",
        "what follows {state}",
        "next step after {state}",
        "what will happen to {state}",
        "predict the future of {state}",
        "what comes before {state}",
        "what led to {state}",
    ],
    
    INTENT_GLYPH_LOOKUP: [
        "get glyph {id}",
        "show me {id}",
        "retrieve {id}",
        "fetch glyph {id}",
        "display {id}",
        "what is glyph {id}",
        "details of {id}",
        "info about {id}",
    ],
    
    INTENT_EDGE_QUERY: [
        "how is {source} related to {target}",
        "connections to {source}",
        "relationships of {source}",
        "edges from {source}",
        "what connects {source} and {target}",
        "link between {source} and {target}",
        "how does {source} connect to {target}",
        "neighbors of {source}",
    ],
}


# Pattern templates for extracting parameters
PARAMETER_PATTERNS = {
    "{query}": r"(.+)",
    "{claim}": r"(.+)",
    "{state}": r"(.+)",
    "{id}": r"([a-f0-9-]+)",
    "{source}": r"(.+?)",
    "{target}": r"(.+)",
}


def get_default_patterns() -> Dict[str, List[str]]:
    """Get default intent patterns."""
    return DEFAULT_INTENT_PATTERNS.copy()


def get_all_intents() -> List[str]:
    """Get list of all intent categories."""
    return [
        INTENT_SIMILARITY_SEARCH,
        INTENT_FACT_TREE,
        INTENT_TEMPORAL_PREDICTION,
        INTENT_GLYPH_LOOKUP,
        INTENT_EDGE_QUERY,
    ]
