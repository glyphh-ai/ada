"""
Intent Matcher for Rules-Based NL Query Matching.

Uses HDC similarity from the SDK's IntentEncoder to match natural language
queries against registered intent patterns. This is the deterministic,
rules-first approach - "When your LLM can't afford to be wrong, sidecar it with Glyphh."

Updated to use PatternMerger for comprehensive default patterns (Requirement 4.3, 4.9).
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from shared.encoder_config_factory import EncoderConfigFactory, ConfigurationError
from shared.sdk_adapter import get_sdk_adapter, SDKNotAvailableError

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
    
    Supports loading patterns from:
    1. Model's NL encoder config (preferred)
    2. SDK default patterns (fallback)
    """
    
    def __init__(
        self, 
        confidence_threshold: float = 0.85,
        model_nl_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the IntentMatcher.
        
        Args:
            confidence_threshold: Minimum confidence for a match
            model_nl_config: Optional NL encoder config from deployed model
        """
        self.confidence_threshold = confidence_threshold
        self._model_nl_config = model_nl_config
        self._encoder = None
        self._patterns_loaded = False
        self._using_model_patterns = False
    
    def set_model_config(self, model_nl_config: Optional[Dict[str, Any]]) -> None:
        """
        Set or update the model NL config.
        
        This will reset the encoder so it reloads with new patterns.
        
        Args:
            model_nl_config: NL encoder config from model, or None to use defaults
        """
        self._model_nl_config = model_nl_config
        self._encoder = None
        self._patterns_loaded = False
        self._using_model_patterns = False
        
    def _get_encoder(self):
        """
        Lazy-load the SDK IntentEncoder.
        
        If model_nl_config is provided, loads patterns from the model.
        Otherwise, falls back to SDK default patterns.
        """
        if self._encoder is None:
            adapter = get_sdk_adapter()
            
            if not adapter.is_available:
                logger.warning("SDK not available, IntentEncoder will use fallback matching")
                return None
            
            try:
                # Create intent-optimized config using the factory
                config = EncoderConfigFactory.create_intent_config(
                    dimension=10000,
                    seed=42
                )
                
                # Use SDK adapter to create the intent encoder
                self._encoder = adapter.create_intent_encoder(config)
                
                if self._encoder is not None:
                    # Load patterns from model config if provided
                    if self._model_nl_config and "patterns" in self._model_nl_config:
                        self._load_model_patterns()
                    else:
                        # Fall back to SDK defaults
                        logger.info("No model NL config provided, using SDK default patterns")
                        self._encoder.add_defaults()
                    
                    self._patterns_loaded = True
                    pattern_count = len(self._encoder.get_patterns()) if hasattr(self._encoder, 'get_patterns') else 0
                    source = "model" if self._using_model_patterns else "SDK defaults"
                    logger.info(f"IntentEncoder initialized with {pattern_count} patterns from {source}")
                else:
                    logger.warning("IntentEncoder not available in SDK, using fallback matching")
                    
            except SDKNotAvailableError as e:
                logger.warning(f"SDK not available for IntentEncoder: {e}")
                self._encoder = None
            except ConfigurationError as e:
                logger.error(f"Failed to create intent config: {e}")
                self._encoder = None
            except ImportError as e:
                logger.warning(f"SDK IntentEncoder not available: {e}")
                self._encoder = None
            except Exception as e:
                logger.error(f"Unexpected error initializing IntentEncoder: {e}")
                self._encoder = None
                
        return self._encoder
    
    def _load_model_patterns(self) -> None:
        """
        Load intent patterns from model NL config using PatternMerger.
        
        Uses PatternMerger to merge custom patterns with defaults.
        Logs pattern source for debugging (Requirement 4.9).
        """
        if not self._encoder:
            return
        
        try:
            # Import PatternMerger from SDK
            from glyphh.gql.pattern_merger import PatternMerger
            from glyphh.gql.patterns import GQLPattern
            
            # Build config dict for PatternMerger
            config = {}
            if self._model_nl_config:
                config["nl_encoder_config"] = self._model_nl_config
            
            # Merge patterns (custom + defaults)
            merged_patterns = PatternMerger.from_config(config, log_source=True)
            
            # Import IntentPattern from SDK
            adapter = get_sdk_adapter()
            IntentPattern = adapter.import_intent_pattern()
            
            if IntentPattern is None:
                logger.warning("IntentPattern not available, cannot load patterns")
                return
            
            # Convert GQLPatterns to IntentPatterns and add to encoder
            loaded_count = 0
            for gql_pattern in merged_patterns:
                try:
                    # Normalize phrases for intent matching
                    normalized_phrases = [
                        re.sub(r'\{[^}]+\}', 'SLOT', phrase)
                        for phrase in gql_pattern.phrases
                    ]
                    
                    pattern = IntentPattern(
                        intent_type=gql_pattern.name,
                        example_phrases=normalized_phrases,
                        query_template={"operation": gql_pattern.name, "gql_template": gql_pattern.gql_template}
                    )
                    self._encoder.add_pattern(pattern)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to add pattern '{gql_pattern.name}': {e}")
            
            if loaded_count > 0:
                self._using_model_patterns = bool(self._model_nl_config and self._model_nl_config.get("patterns"))
                logger.info(f"Loaded {loaded_count} patterns (merged defaults + custom)")
            else:
                logger.warning("No patterns loaded, using SDK defaults")
                
        except ImportError as e:
            logger.warning(f"PatternMerger not available: {e}")
            # Fall back to old behavior
            self._load_model_patterns_legacy()
        except Exception as e:
            logger.error(f"Failed to load patterns via PatternMerger: {e}")
            self._load_model_patterns_legacy()
    
    def _load_model_patterns_legacy(self) -> None:
        """
        Legacy pattern loading (fallback if PatternMerger unavailable).
        """
        if not self._encoder or not self._model_nl_config:
            return
        
        patterns = self._model_nl_config.get("patterns", [])
        if not patterns:
            logger.info("Model NL config has no patterns, using SDK defaults")
            return
        
        try:
            adapter = get_sdk_adapter()
            IntentPattern = adapter.import_intent_pattern()
            
            if IntentPattern is None:
                logger.warning("IntentPattern not available")
                return
            
            loaded_count = 0
            for pattern_data in patterns:
                try:
                    pattern = IntentPattern(
                        intent_type=pattern_data.get("intent_type", ""),
                        example_phrases=pattern_data.get("example_phrases", []),
                        query_template=pattern_data.get("query_template", {})
                    )
                    self._encoder.add_pattern(pattern)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Skipping invalid pattern: {e}")
            
            if loaded_count > 0:
                self._using_model_patterns = True
                logger.info(f"Loaded {loaded_count} patterns from model NL config (legacy)")
                
        except Exception as e:
            logger.error(f"Failed to load model patterns (legacy): {e}")
    
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
        
        # Simple keyword patterns - higher base scores for clear matches
        patterns = [
            (["find", "search", "similar", "like", "show me"], "similarity_search", 0.85),
            (["verify", "explain", "prove", "evidence", "why", "how"], "fact_tree", 0.85),
            (["predict", "forecast", "next", "after", "future", "will"], "temporal_predict", 0.85),
            (["list all", "show all", "get all", "list everything"], "list_all", 0.90),
            (["count", "how many", "total", "number of"], "count", 0.90),
            (["compare", "difference", "versus", "vs", "contrast"], "compare", 0.85),
        ]
        
        best_match = None
        best_score = 0.0
        
        for keywords, intent, base_score in patterns:
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > 0:
                # Higher score for more keyword matches
                score = base_score + (matches * 0.02)
                score = min(score, 0.98)  # Cap at 0.98
                if score > best_score:
                    best_score = score
                    best_match = intent
        
        if best_match:
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
