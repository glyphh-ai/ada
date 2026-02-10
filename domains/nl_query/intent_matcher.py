"""
Intent Matcher for Rules-Based NL Query Matching.

Uses HDC similarity from the SDK's IntentEncoder to match natural language
queries against registered intent patterns. This is the deterministic,
rules-first approach - "When your LLM can't afford to be wrong, sidecar it with Glyphh."

Updated to use PatternMerger for comprehensive default patterns (Requirement 4.3, 4.9).
Extended to support stored procedure matching (Requirement 4.1-4.6).
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from shared.encoder_config_factory import EncoderConfigFactory, ConfigurationError
from shared.sdk_adapter import get_sdk_adapter, SDKNotAvailableError

if TYPE_CHECKING:
    from domains.procedures.service import StoredProcedureService

logger = logging.getLogger(__name__)


@dataclass
class IntentMatch:
    """Result of matching a query against intent patterns."""
    intent: str
    confidence: float
    parameters: Dict[str, str]
    pattern_matched: Optional[str]
    structured_query: Dict[str, Any]
    match_method: str = "default"  # "default", "stored_procedure", or "fallback"
    procedure_name: Optional[str] = None  # Name of matched stored procedure


class IntentMatcher:
    """
    Rules-based intent matcher using HDC similarity.
    
    Matches natural language queries against registered patterns
    using the SDK's IntentEncoder for deterministic matching.
    
    Supports loading patterns from:
    1. Stored procedures (highest priority)
    2. Model's NL encoder config (preferred)
    3. SDK default patterns (fallback)
    """
    
    def __init__(
        self, 
        confidence_threshold: float = 0.85,
        model_nl_config: Optional[Dict[str, Any]] = None,
        procedure_service: Optional['StoredProcedureService'] = None,
    ):
        """
        Initialize the IntentMatcher.
        
        Args:
            confidence_threshold: Minimum confidence for a match
            model_nl_config: Optional NL encoder config from deployed model
            procedure_service: Optional service for stored procedure matching
        """
        self.confidence_threshold = confidence_threshold
        self._model_nl_config = model_nl_config
        self._procedure_service = procedure_service
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
    
    def set_procedure_service(self, procedure_service: Optional['StoredProcedureService']) -> None:
        """
        Set or update the procedure service for stored procedure matching.
        
        Args:
            procedure_service: Service for stored procedure CRUD operations
        """
        self._procedure_service = procedure_service
        
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
    
    async def match_intent(
        self, 
        query: str,
        org_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Optional[IntentMatch]:
        """
        Match a query against stored procedures first, then default patterns.
        
        Args:
            query: Natural language query to match
            org_id: Organization ID for stored procedure lookup
            model_id: Model ID for stored procedure lookup
            
        Returns:
            IntentMatch if confidence >= threshold, None otherwise
        """
        # Check stored procedures first (highest priority)
        if self._procedure_service and org_id and model_id:
            procedure_match = await self._match_stored_procedure(query, org_id, model_id)
            if procedure_match and procedure_match.confidence >= self.confidence_threshold:
                logger.info(
                    f"Matched stored procedure '{procedure_match.procedure_name}' "
                    f"with confidence {procedure_match.confidence:.3f}"
                )
                return procedure_match
        
        # Fall back to default pattern matching
        encoder = self._get_encoder()
        
        if encoder is None:
            # Fallback to simple keyword matching
            logger.info("No SDK encoder available, using fallback matcher")
            return self._fallback_match(query)
        
        try:
            # Use SDK's IntentEncoder for HDC similarity matching
            match = encoder.match_intent(query, self.confidence_threshold)
            
            if match is None:
                logger.info(f"SDK encoder returned no match for query: '{query}', using fallback")
                return self._fallback_match(query)
            
            logger.info(f"SDK encoder matched: intent={match.intent_type}, confidence={match.confidence:.3f}")
            
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
                match_method="default",
            )
        except Exception as e:
            logger.warning(f"Intent matching failed: {e}, using fallback")
            return self._fallback_match(query)
    
    async def _match_stored_procedure(
        self,
        query: str,
        org_id: str,
        model_id: str,
    ) -> Optional[IntentMatch]:
        """
        Match query against stored procedure lexicons using HDC similarity.
        
        Args:
            query: Natural language query
            org_id: Organization ID
            model_id: Model ID
            
        Returns:
            IntentMatch if a procedure matches above threshold, None otherwise
        """
        try:
            procedures = await self._procedure_service.list(org_id, model_id)
            if not procedures:
                logger.debug(f"No stored procedures found for {org_id}/{model_id}")
                return None
            
            logger.debug(f"Checking {len(procedures)} stored procedures for match")
            
            encoder = self._get_encoder()
            if encoder is None:
                return self._fallback_procedure_match(query, procedures)
            
            best_match = None
            best_score = 0.0
            
            # Encode query
            query_lower = query.lower()
            
            for procedure in procedures:
                # Compute similarity against each lexicon
                for lexicon in procedure.lexicons:
                    try:
                        # Use encoder's text similarity if available
                        if hasattr(encoder, 'compute_text_similarity'):
                            score = encoder.compute_text_similarity(query_lower, lexicon.lower())
                        else:
                            # Fallback to simple word overlap
                            score = self._compute_lexicon_similarity(query_lower, lexicon.lower())
                        
                        if score > best_score:
                            best_score = score
                            best_match = procedure
                    except Exception as e:
                        logger.warning(f"Error computing similarity for lexicon '{lexicon}': {e}")
                        continue
            
            if best_match and best_score >= self.confidence_threshold:
                return IntentMatch(
                    intent="stored_procedure",
                    confidence=best_score,
                    parameters={"procedure_name": best_match.name},
                    pattern_matched=None,
                    structured_query={
                        "operation": "execute_procedure",
                        "procedure_name": best_match.name,
                        "gql_query": best_match.gql_query,
                    },
                    match_method="stored_procedure",
                    procedure_name=best_match.name,
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error matching stored procedures: {e}")
            return None
    
    def _compute_lexicon_similarity(self, query: str, lexicon: str) -> float:
        """
        Compute simple word-based similarity between query and lexicon.
        
        Args:
            query: Query string (lowercase)
            lexicon: Lexicon string (lowercase)
            
        Returns:
            Similarity score between 0 and 1
        """
        query_words = set(query.split())
        lexicon_words = set(lexicon.split())
        
        if not lexicon_words:
            return 0.0
        
        # Check if lexicon is contained in query
        if lexicon in query:
            return 0.95
        
        # Word overlap score
        overlap = len(query_words & lexicon_words)
        if overlap == 0:
            return 0.0
        
        # Jaccard-like similarity with boost for lexicon coverage
        lexicon_coverage = overlap / len(lexicon_words)
        query_coverage = overlap / len(query_words) if query_words else 0
        
        # Weight lexicon coverage more heavily
        return 0.7 * lexicon_coverage + 0.3 * query_coverage
    
    def _fallback_procedure_match(
        self,
        query: str,
        procedures: List[Any],
    ) -> Optional[IntentMatch]:
        """
        Simple keyword-based procedure matching when encoder unavailable.
        
        Args:
            query: Natural language query
            procedures: List of stored procedures
            
        Returns:
            IntentMatch if a procedure matches, None otherwise
        """
        query_lower = query.lower()
        best_match = None
        best_score = 0.0
        
        for procedure in procedures:
            for lexicon in procedure.lexicons:
                score = self._compute_lexicon_similarity(query_lower, lexicon.lower())
                if score > best_score:
                    best_score = score
                    best_match = procedure
        
        if best_match and best_score >= self.confidence_threshold:
            return IntentMatch(
                intent="stored_procedure",
                confidence=best_score,
                parameters={"procedure_name": best_match.name},
                pattern_matched=None,
                structured_query={
                    "operation": "execute_procedure",
                    "procedure_name": best_match.name,
                    "gql_query": best_match.gql_query,
                },
                match_method="stored_procedure",
                procedure_name=best_match.name,
            )
        
        return None
    
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
                match_method="fallback",
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
