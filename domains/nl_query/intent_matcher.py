"""
Intent Matcher for Natural Language Query.

Rules-based intent matching using pattern matching and
optional HDC similarity for fuzzy matching.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from domains.nl_query.default_intents import (
    DEFAULT_INTENT_PATTERNS,
    INTENT_SIMILARITY_SEARCH,
    INTENT_FACT_TREE,
    INTENT_TEMPORAL_PREDICTION,
    INTENT_GLYPH_LOOKUP,
    INTENT_EDGE_QUERY,
    PARAMETER_PATTERNS,
    get_default_patterns,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of intent matching."""
    intent: str
    confidence: float
    parameters: Dict[str, str]
    pattern_matched: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "parameters": self.parameters,
            "pattern_matched": self.pattern_matched,
        }


class IntentMatcher:
    """
    Rules-based intent matcher for natural language queries.
    
    Uses pattern matching with optional HDC similarity for
    fuzzy matching when exact patterns don't match.
    
    Responsibilities:
    - Load intent patterns from model or defaults
    - Match queries against patterns
    - Extract parameters from matched patterns
    - Return confidence scores
    """
    
    def __init__(
        self,
        patterns: Optional[Dict[str, List[str]]] = None,
        encoder: Optional[Any] = None,
        confidence_threshold: float = 0.85,
    ):
        """
        Initialize IntentMatcher.
        
        Args:
            patterns: Custom intent patterns (uses defaults if None)
            encoder: Optional SDK encoder for HDC similarity
            confidence_threshold: Minimum confidence for a match
        """
        self._patterns = patterns or get_default_patterns()
        self._encoder = encoder
        self._confidence_threshold = confidence_threshold
        self._compiled_patterns: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        self._encoded_patterns: Dict[str, List[Tuple[str, Any]]] = {}
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Encode patterns if encoder available
        if encoder:
            self._encode_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile pattern strings to regex."""
        for intent, patterns in self._patterns.items():
            self._compiled_patterns[intent] = []
            
            for pattern in patterns:
                # Convert pattern template to regex
                regex_pattern = pattern
                for param, param_regex in PARAMETER_PATTERNS.items():
                    regex_pattern = regex_pattern.replace(param, param_regex)
                
                # Make it case-insensitive and match full string
                regex_pattern = f"^{regex_pattern}$"
                
                try:
                    compiled = re.compile(regex_pattern, re.IGNORECASE)
                    self._compiled_patterns[intent].append((pattern, compiled))
                except re.error as e:
                    logger.warning(f"Failed to compile pattern '{pattern}': {e}")
    
    def _encode_patterns(self) -> None:
        """Encode patterns using HDC encoder for similarity matching."""
        if not self._encoder:
            return
        
        for intent, patterns in self._patterns.items():
            self._encoded_patterns[intent] = []
            
            for pattern in patterns:
                try:
                    # Remove parameter placeholders for encoding
                    clean_pattern = pattern
                    for param in PARAMETER_PATTERNS:
                        clean_pattern = clean_pattern.replace(param, "something")
                    
                    # Encode using SDK encoder
                    embedding = self._encoder.encode_text(clean_pattern)
                    self._encoded_patterns[intent].append((pattern, embedding))
                except Exception as e:
                    logger.warning(f"Failed to encode pattern '{pattern}': {e}")
    
    def match_intent(self, query: str) -> Optional[MatchResult]:
        """
        Match a query against intent patterns.
        
        Args:
            query: Natural language query
            
        Returns:
            MatchResult if match found above threshold, None otherwise
        """
        query = query.strip()
        
        # Try exact pattern matching first
        result = self._match_exact(query)
        if result and result.confidence >= self._confidence_threshold:
            return result
        
        # Try HDC similarity matching if encoder available
        if self._encoder and self._encoded_patterns:
            result = self._match_similarity(query)
            if result and result.confidence >= self._confidence_threshold:
                return result
        
        # Try fuzzy pattern matching
        result = self._match_fuzzy(query)
        if result and result.confidence >= self._confidence_threshold:
            return result
        
        return None
    
    def _match_exact(self, query: str) -> Optional[MatchResult]:
        """Try exact regex pattern matching."""
        for intent, patterns in self._compiled_patterns.items():
            for pattern_str, compiled in patterns:
                match = compiled.match(query)
                if match:
                    # Extract parameters
                    params = self._extract_parameters(pattern_str, match)
                    
                    return MatchResult(
                        intent=intent,
                        confidence=1.0,
                        parameters=params,
                        pattern_matched=pattern_str,
                    )
        
        return None
    
    def _match_similarity(self, query: str) -> Optional[MatchResult]:
        """Match using HDC similarity."""
        if not self._encoder:
            return None
        
        try:
            import numpy as np
            
            # Encode query
            query_embedding = self._encoder.encode_text(query)
            
            best_match = None
            best_score = 0.0
            best_intent = None
            best_pattern = None
            
            for intent, patterns in self._encoded_patterns.items():
                for pattern_str, pattern_embedding in patterns:
                    # Compute cosine similarity
                    similarity = np.dot(query_embedding, pattern_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(pattern_embedding)
                    )
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_intent = intent
                        best_pattern = pattern_str
            
            if best_intent and best_score >= self._confidence_threshold:
                # Try to extract parameters using fuzzy matching
                params = self._extract_parameters_fuzzy(best_pattern, query)
                
                return MatchResult(
                    intent=best_intent,
                    confidence=float(best_score),
                    parameters=params,
                    pattern_matched=best_pattern,
                )
                
        except Exception as e:
            logger.warning(f"Similarity matching failed: {e}")
        
        return None
    
    def _match_fuzzy(self, query: str) -> Optional[MatchResult]:
        """Try fuzzy pattern matching using keyword detection."""
        query_lower = query.lower()
        
        # Keyword-based intent detection
        intent_keywords = {
            INTENT_SIMILARITY_SEARCH: ["similar", "like", "find", "search", "related"],
            INTENT_FACT_TREE: ["verify", "explain", "prove", "true", "evidence", "support"],
            INTENT_TEMPORAL_PREDICTION: ["predict", "next", "future", "forecast", "after", "before"],
            INTENT_GLYPH_LOOKUP: ["glyph", "get", "show", "retrieve", "fetch"],
            INTENT_EDGE_QUERY: ["related", "connection", "edge", "link", "connect"],
        }
        
        best_intent = None
        best_score = 0.0
        
        for intent, keywords in intent_keywords.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            score = matches / len(keywords)
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        if best_intent and best_score > 0:
            # Extract the main content as parameter
            params = {"query": query}
            
            # Adjust confidence based on keyword matches
            confidence = min(0.9, 0.5 + best_score * 0.4)
            
            return MatchResult(
                intent=best_intent,
                confidence=confidence,
                parameters=params,
                pattern_matched=None,
            )
        
        return None
    
    def _extract_parameters(
        self,
        pattern: str,
        match: re.Match,
    ) -> Dict[str, str]:
        """Extract parameters from regex match."""
        params = {}
        groups = match.groups()
        
        # Find parameter names in pattern
        param_names = []
        for param in PARAMETER_PATTERNS:
            if param in pattern:
                param_names.append(param.strip("{}"))
        
        # Map groups to parameter names
        for i, name in enumerate(param_names):
            if i < len(groups):
                params[name] = groups[i].strip()
        
        return params
    
    def _extract_parameters_fuzzy(
        self,
        pattern: str,
        query: str,
    ) -> Dict[str, str]:
        """Extract parameters using fuzzy matching."""
        params = {}
        
        # Find parameter placeholders
        for param in PARAMETER_PATTERNS:
            if param in pattern:
                param_name = param.strip("{}")
                # Use the whole query as the parameter value
                params[param_name] = query
                break
        
        return params
    
    def get_patterns(self) -> Dict[str, List[str]]:
        """Get current intent patterns."""
        return self._patterns.copy()
    
    def add_patterns(self, intent: str, patterns: List[str]) -> None:
        """Add patterns for an intent."""
        if intent not in self._patterns:
            self._patterns[intent] = []
        
        self._patterns[intent].extend(patterns)
        
        # Recompile patterns
        self._compile_patterns()
        
        # Re-encode if encoder available
        if self._encoder:
            self._encode_patterns()
