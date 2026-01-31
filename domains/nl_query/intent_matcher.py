"""
Intent Matcher for Rules-Based NL Query Matching.

Uses HDC similarity from the SDK's IntentEncoder to match natural language
queries against registered intent patterns. This is the deterministic,
rules-first approach - "when your LLM can't be wrong, sidecar it with Glyphh."
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class IntentMatch:
    """Result of matching a query against intent patterns."""
    intent: str
    confidence: float
    parameters: Dict[str, str]
    pattern_matched: Optional[str]
    structured_query: Dict[str, Any]


class IntentMatcher:
    """
    Rules-based intent matcher using HDC similarity.
    
    Matches natural language queries against registered patterns
    using the SDK's IntentEncoder for deterministic matching.
    """
    
    def __init__(self, confidence_threshold: float = 0.85):
        """
        Initialize the IntentMatcher.
        
        Args:
            confidence_threshold: Minimum confidence for a match
        """
        self.confidence_threshold = confidence_threshold
        self._encoder = None
        self._patterns_loaded = False
        
    def _get_encoder(self):
        """Lazy-load the SDK IntentEncoder."""
        if self._encoder is None:
            try:
                from glyphh.encoder.intent import IntentEncoder
                from glyphh.core.config import EncoderConfig
                
                config = EncoderConfig(dimension=10000, seed=42)
                self._encoder = IntentEncoder(config)
                self._encoder.add_defaults()
                self._patterns_loaded = True
                logger.info(f"IntentEncoder initialized with {len(self._encoder.get_patterns())} patterns")
            except ImportError as e:
                logger.warning(f"SDK IntentEncoder not available: {e}")
                self._encoder = None
        return self._encoder
    
    async def match_intent(self, query: str) -> Optional[IntentMatch]:
        """
        Match a query against registered intent patterns.
        
        Args:
            query: Natural language query to match
            
        Returns:
            IntentMatch if confidence >= threshold, None otherwise
        """
        encoder = self._get_encoder()
        
        if encoder is None:
            # Fallback to simple keyword matching
            return self._fallback_match(query)
        
        try:
            # Use SDK's IntentEncoder for HDC similarity matching
            match = encoder.match_intent(query, self.confidence_threshold)
            
            # Extract parameters from query
            parameters = self._extract_parameters(query, match.intent_type)
            
            # Build structured query with parameters
            structured_query = self._build_structured_query(
                match.structured_query,
                parameters,
                query
            )
            
            return IntentMatch(
                intent=match.intent_type,
                confidence=match.confidence,
                parameters=parameters,
                pattern_matched=match.matched_phrase,
                structured_query=structured_query,
            )
        except Exception as e:
            logger.warning(f"Intent matching failed: {e}")
            return self._fallback_match(query)
    
    def _fallback_match(self, query: str) -> Optional[IntentMatch]:
        """
        Simple keyword-based fallback when SDK is unavailable.
        
        Args:
            query: Natural language query
            
        Returns:
            IntentMatch based on keyword matching
        """
        query_lower = query.lower()
        
        # Simple keyword patterns
        patterns = [
            (["find", "search", "similar", "like"], "similarity_search", 0.7),
            (["verify", "explain", "prove", "evidence"], "fact_tree", 0.7),
            (["predict", "forecast", "next", "after"], "temporal_predict", 0.7),
            (["list", "show all", "get all"], "list_all", 0.7),
            (["count", "how many"], "count", 0.7),
            (["compare", "difference", "versus", "vs"], "compare", 0.7),
        ]
        
        best_match = None
        best_score = 0.0
        
        for keywords, intent, base_score in patterns:
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > 0:
                score = base_score + (matches * 0.05)
                if score > best_score:
                    best_score = score
                    best_match = intent
        
        if best_match and best_score >= self.confidence_threshold:
            parameters = self._extract_parameters(query, best_match)
            structured_query = self._build_structured_query(
                self._get_default_template(best_match),
                parameters,
                query
            )
            
            return IntentMatch(
                intent=best_match,
                confidence=best_score,
                parameters=parameters,
                pattern_matched=None,
                structured_query=structured_query,
            )
        
        return None
    
    def _extract_parameters(self, query: str, intent: str) -> Dict[str, str]:
        """
        Extract parameters from the query based on intent type.
        
        Args:
            query: Original query
            intent: Matched intent type
            
        Returns:
            Dictionary of extracted parameters
        """
        params = {}
        
        # Remove common intent keywords to get the concept/subject
        query_clean = query.lower()
        
        # Remove intent-specific keywords
        remove_patterns = {
            "similarity_search": ["find", "search", "similar to", "like", "what's like", "show me"],
            "fact_tree": ["verify", "explain", "prove", "is it true that", "evidence for"],
            "temporal_predict": ["predict", "forecast", "what comes after", "next state", "what will happen"],
            "list_all": ["list all", "show all", "get all", "enumerate"],
            "count": ["how many", "count", "total number of"],
            "compare": ["compare", "difference between", "versus", "vs"],
        }
        
        for pattern in remove_patterns.get(intent, []):
            query_clean = query_clean.replace(pattern, "").strip()
        
        # Clean up extra whitespace
        query_clean = " ".join(query_clean.split())
        
        if query_clean:
            params["query"] = query_clean
            params["concept"] = query_clean
        
        return params
    
    def _build_structured_query(
        self,
        template: Dict[str, Any],
        parameters: Dict[str, str],
        original_query: str
    ) -> Dict[str, Any]:
        """
        Build a structured query from template and parameters.
        
        Args:
            template: Query template from pattern
            parameters: Extracted parameters
            original_query: Original NL query
            
        Returns:
            Structured query ready for execution
        """
        query = template.copy()
        
        # Add the query/concept parameter
        if "query" in parameters:
            query["query"] = parameters["query"]
        elif "concept" in parameters:
            query["query"] = parameters["concept"]
        else:
            query["query"] = original_query
        
        return query
    
    def _get_default_template(self, intent: str) -> Dict[str, Any]:
        """Get default query template for an intent."""
        templates = {
            "similarity_search": {
                "operation": "similarity_search",
                "top_k": 10,
            },
            "fact_tree": {
                "operation": "fact_tree",
                "max_depth": 3,
            },
            "temporal_predict": {
                "operation": "temporal_predict",
                "steps_ahead": 1,
                "beam_width": 3,
            },
            "list_all": {
                "operation": "list",
                "limit": 100,
            },
            "count": {
                "operation": "count",
            },
            "compare": {
                "operation": "compare",
            },
        }
        return templates.get(intent, {"operation": intent})
    
    def get_intents(self) -> Dict[str, Any]:
        """
        Get available intents and their patterns.
        
        Returns:
            Dictionary with intent names and example patterns
        """
        encoder = self._get_encoder()
        
        if encoder is not None:
            patterns = encoder.get_patterns()
            return {
                "intents": [p.intent_type for p in patterns],
                "patterns": {
                    p.intent_type: p.example_phrases
                    for p in patterns
                }
            }
        
        # Fallback patterns
        return {
            "intents": [
                "similarity_search",
                "fact_tree",
                "temporal_predict",
                "list_all",
                "count",
                "compare",
            ],
            "patterns": {
                "similarity_search": ["find similar to", "search for", "what's like"],
                "fact_tree": ["verify", "explain", "prove"],
                "temporal_predict": ["predict", "what comes after", "forecast"],
                "list_all": ["list all", "show all"],
                "count": ["how many", "count"],
                "compare": ["compare", "difference between"],
            }
        }
