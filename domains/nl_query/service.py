"""
Natural Language Query Service.

Provides hybrid rules-first + LLM-fallback query translation and execution.
The core principle: "When your LLM can't afford to be wrong, sidecar it with Glyphh."

Updated to support AutoSchemaMatcher for automatic schema-based NL query matching.
Updated to return SDK FactTree in all responses for unified response format.
Validates: Requirements 3, 4, 5, 7.1, 7.2, 7.3, 12.2, 12.5

Design Principle: "When your LLM can't afford to be wrong, sidecar it with Glyphh"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from glyphh.fact_tree.builder import FactTree

from domains.nl_query.intent_matcher import IntentMatcher, IntentMatch
from domains.query.service import QueryService
from domains.query.fact_tree_builder import FactTreeBuilder

if TYPE_CHECKING:
    from domains.nl_query.schema_index import SchemaIndex
    from glyphh.nl.auto_schema_matcher import AutoSchemaMatcher, AutoMatchResult

logger = logging.getLogger(__name__)


@dataclass
class NLQueryResult:
    """
    Result of executing a natural language query.
    
    Updated to contain a FactTree instead of raw result data.
    The fact_tree field contains the SDK FactTree which can be
    serialized via to_json() for transport.
    
    Validates: Requirements 7.1, 7.2, 7.3
    """
    fact_tree: FactTree  # SDK FactTree with full structure
    query_type: str
    match_method: str  # "rules", "llm", "auto", "hybrid", or "none"
    confidence: float
    translated_query: Optional[Dict[str, Any]]
    query_time_ms: float
    disambiguation_needed: bool = False
    disambiguation_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for API response.
        
        Returns both the FactTree JSON and legacy fields for
        backward compatibility.
        
        Validates: Requirements 7.1, 11.1
        """
        return {
            "result": self.fact_tree.to_json(),
            "query_type": self.query_type,
            "match_method": self.match_method,
            "confidence": self.confidence,
            "translated_query": self.translated_query,
            "query_time_ms": self.query_time_ms,
            "disambiguation_needed": self.disambiguation_needed,
            "disambiguation_suggestions": self.disambiguation_suggestions,
        }


class NLQueryService:
    """
    Natural Language Query Service with hybrid rules + LLM approach.
    
    Uses rules-first matching via IntentMatcher (HDC similarity).
    Falls back to LLM only when rules can't confidently match.
    
    Now supports AutoSchemaMatcher for automatic schema-based matching:
    - When a SchemaIndex is provided, uses AutoSchemaMatcher for query processing
    - AutoSchemaMatcher provides token-to-schema matching, intent inference, and parameter extraction
    - Supports disambiguation when multiple intents have similar confidence
    
    Validates: Requirements 3, 4, 5, 12.2, 12.5
    """
    
    def __init__(
        self,
        query_service: QueryService,
        intent_matcher: IntentMatcher,
        llm_fallback: Optional[Any] = None,
        confidence_threshold: float = 0.85,
        schema_index: Optional['SchemaIndex'] = None,
        auto_schema_matcher: Optional['AutoSchemaMatcher'] = None,
    ):
        """
        Initialize the NL Query Service.
        
        Args:
            query_service: QueryService for executing structured queries
            intent_matcher: IntentMatcher for rules-based matching
            llm_fallback: Optional LLMFallback for low-confidence queries
            confidence_threshold: Minimum confidence for rules-based match
            schema_index: Optional SchemaIndex for auto-schema matching
            auto_schema_matcher: Optional AutoSchemaMatcher for auto-schema matching
        
        Validates: Requirements 3, 4, 5
        """
        self.query_service = query_service
        self.intent_matcher = intent_matcher
        self.llm_fallback = llm_fallback
        self.confidence_threshold = confidence_threshold
        self._schema_index = schema_index
        self._auto_schema_matcher = auto_schema_matcher
    
    def set_schema_index(self, schema_index: 'SchemaIndex') -> None:
        """
        Set the schema index for auto-schema matching.
        
        Args:
            schema_index: The SchemaIndex to use for matching
        
        Validates: Requirement 7
        """
        self._schema_index = schema_index
        logger.info(f"Schema index set for model '{schema_index.model_id}'")
    
    def set_auto_schema_matcher(self, matcher: 'AutoSchemaMatcher') -> None:
        """
        Set the AutoSchemaMatcher for auto-schema matching.
        
        Args:
            matcher: The AutoSchemaMatcher to use for matching
        
        Validates: Requirements 3, 4, 5
        """
        self._auto_schema_matcher = matcher
        logger.info("AutoSchemaMatcher set for NL query service")
    
    async def execute_nl_query(
        self,
        org_id: str,
        model_id: str,
        query: str,
        debug: bool = False,
    ) -> NLQueryResult:
        """
        Execute a natural language query.
        
        Flow:
        1. Try auto-schema matching if AutoSchemaMatcher is available
        2. Try rules-based matching (IntentMatcher)
        3. If confidence >= threshold: execute directly
        4. If confidence < threshold AND LLM enabled: use LLM fallback
        5. If confidence < threshold AND no LLM: execute with lower confidence
        
        Validates: Requirements 3, 4, 5, 12.2, 12.5
        """
        start_time = time.time()
        
        logger.info(f"NL query received: '{query}' for org={org_id}, model={model_id}")
        
        # Step 0: Try auto-schema matching if available
        # Validates: Requirements 3, 4, 5
        if self._auto_schema_matcher is not None:
            try:
                auto_result = await self._execute_auto_schema_query(
                    org_id=org_id,
                    model_id=model_id,
                    query=query,
                    debug=debug,
                    start_time=start_time,
                )
                if auto_result is not None:
                    return auto_result
            except Exception as e:
                logger.warning(f"Auto-schema matching failed: {e}, falling back to rules")
        
        # Step 1: Try rules-based matching
        match_result = await self.intent_matcher.match_intent(query)
        
        # Accept any match with confidence > 0.3 (SDK HDC similarity is more conservative)
        min_acceptable_confidence = 0.3
        
        if match_result and match_result.confidence >= min_acceptable_confidence:
            logger.info(
                f"Rules match: intent={match_result.intent}, "
                f"confidence={match_result.confidence:.3f}"
            )
            
            result = await self._execute_structured_query(
                org_id,
                model_id,
                match_result.intent,
                match_result.structured_query,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return NLQueryResult(
                fact_tree=result,
                query_type=match_result.intent,
                match_method="rules",
                confidence=match_result.confidence,
                translated_query=match_result.structured_query if debug else None,
                query_time_ms=elapsed_ms,
                disambiguation_needed=False,
                disambiguation_suggestions=[],
            )
        
        # Step 2: Try LLM fallback if available
        if self.llm_fallback is not None:
            logger.info("Rules confidence too low, trying LLM fallback")
            
            try:
                llm_result = await self.llm_fallback.translate_query(query, f"org={org_id}, model={model_id}")
                
                if llm_result:
                    result = await self._execute_structured_query(
                        org_id,
                        model_id,
                        llm_result.get("operation", "similarity_search"),
                        llm_result,
                    )
                    
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    return NLQueryResult(
                        fact_tree=result,
                        query_type=llm_result.get("operation", "unknown"),
                        match_method="llm",
                        confidence=0.0,
                        translated_query=llm_result if debug else None,
                        query_time_ms=elapsed_ms,
                        disambiguation_needed=False,
                        disambiguation_suggestions=[],
                    )
            except Exception as e:
                logger.warning(f"LLM fallback failed: {e}")
        
        # Step 3: No match - return error FactTree
        logger.info(f"No intent match for query: '{query}'")
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Build error FactTree for no match case
        error_fact_tree = FactTreeBuilder.build_error(
            error_message="No intent match found for query",
            error_type="NoMatchError",
            query=query,
        )
        
        return NLQueryResult(
            fact_tree=error_fact_tree,
            query_type="unknown",
            match_method="none",
            confidence=match_result.confidence if match_result else 0.0,
            translated_query=None,
            query_time_ms=elapsed_ms,
            disambiguation_needed=False,
            disambiguation_suggestions=[],
        )
    
    async def _execute_auto_schema_query(
        self,
        org_id: str,
        model_id: str,
        query: str,
        debug: bool,
        start_time: float,
    ) -> Optional[NLQueryResult]:
        """
        Execute a query using auto-schema matching.
        
        Uses the AutoSchemaMatcher to process the query through the
        auto-matching pipeline:
        1. Tokenize query
        2. Match tokens against schema vectors
        3. Infer intent from matches and keywords
        4. Extract parameters
        5. Handle disambiguation if needed
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            query: Natural language query
            debug: Whether to include debug info
            start_time: Query start time for timing
        
        Returns:
            NLQueryResult if auto-matching succeeds, None otherwise
        
        Validates: Requirements 3, 4, 5, 12.2, 12.5
        """
        if self._auto_schema_matcher is None:
            return None
        
        # Check cache if schema index is available
        cache_key = None
        if self._schema_index is not None:
            cache_key = self._schema_index.compute_query_hash(query)
            cached_match = self._schema_index.get_cached_result(cache_key)
            if cached_match is not None:
                logger.info(f"Cache hit for query: '{query}'")
                # Use cached match result to build response
                # Note: We still need to execute the query, but we can skip matching
        
        # Use AutoSchemaMatcher to match the query
        auto_result = self._auto_schema_matcher.match_query(query)
        
        # Cache the match result if schema index is available
        if self._schema_index is not None and cache_key is not None:
            self._schema_index.cache_match_result(cache_key, auto_result.match_result)
        
        # Check if disambiguation is needed
        # Validates: Requirements 12.2, 12.5
        if auto_result.disambiguation_needed:
            logger.info(
                f"Disambiguation needed for query: '{query}', "
                f"options: {[opt.intent_type for opt in auto_result.disambiguation_options]}"
            )
            
            # Build disambiguation suggestions
            suggestions = self._build_disambiguation_suggestions(auto_result)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Build disambiguation FactTree
            disambiguation_fact_tree = FactTreeBuilder.build_error(
                error_message="Disambiguation needed - multiple intents matched",
                error_type="DisambiguationNeeded",
                query=query,
            )
            
            return NLQueryResult(
                fact_tree=disambiguation_fact_tree,
                query_type=auto_result.intent.intent_type,
                match_method=auto_result.match_method,
                confidence=auto_result.confidence,
                translated_query=auto_result.to_dict() if debug else None,
                query_time_ms=elapsed_ms,
                disambiguation_needed=True,
                disambiguation_suggestions=suggestions,
            )
        
        # Check confidence threshold
        min_acceptable_confidence = 0.3
        if auto_result.confidence < min_acceptable_confidence:
            logger.info(
                f"Auto-schema confidence too low: {auto_result.confidence:.3f}, "
                f"falling back to rules"
            )
            return None
        
        logger.info(
            f"Auto-schema match: intent={auto_result.intent.intent_type}, "
            f"confidence={auto_result.confidence:.3f}, "
            f"method={auto_result.match_method}"
        )
        
        # Map auto-schema intent to operation
        operation = self._map_intent_to_operation(auto_result.intent.intent_type)
        
        # Build structured query from extracted parameters
        structured_query = self._build_structured_query_from_auto_result(
            auto_result, operation, query
        )
        
        # Execute the query
        result = await self._execute_structured_query(
            org_id,
            model_id,
            operation,
            structured_query,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return NLQueryResult(
            fact_tree=result,
            query_type=auto_result.intent.intent_type,
            match_method=auto_result.match_method,
            confidence=auto_result.confidence,
            translated_query=auto_result.to_dict() if debug else None,
            query_time_ms=elapsed_ms,
            disambiguation_needed=False,
            disambiguation_suggestions=[],
        )
    
    def _build_disambiguation_suggestions(
        self,
        auto_result: 'AutoMatchResult'
    ) -> List[str]:
        """
        Build disambiguation suggestions from auto-match result.
        
        Creates human-readable suggestions for each disambiguation option
        to help the user clarify their query.
        
        Args:
            auto_result: The AutoMatchResult with disambiguation options
        
        Returns:
            List of suggestion strings
        
        Validates: Requirements 12.2, 12.5
        """
        suggestions = []
        
        for option in auto_result.disambiguation_options:
            intent_type = option.intent_type
            confidence = option.confidence
            
            # Build suggestion based on intent type
            if intent_type == "find":
                suggestions.append(
                    f"Did you mean to find/search for something? "
                    f"(confidence: {confidence:.0%})"
                )
            elif intent_type == "count":
                suggestions.append(
                    f"Did you mean to count items? "
                    f"(confidence: {confidence:.0%})"
                )
            elif intent_type == "filter":
                suggestions.append(
                    f"Did you mean to filter by a condition? "
                    f"(confidence: {confidence:.0%})"
                )
            elif intent_type == "similar":
                suggestions.append(
                    f"Did you mean to find similar items? "
                    f"(confidence: {confidence:.0%})"
                )
            else:
                suggestions.append(
                    f"Did you mean '{intent_type}'? "
                    f"(confidence: {confidence:.0%})"
                )
        
        return suggestions
    
    def _map_intent_to_operation(self, intent_type: str) -> str:
        """
        Map auto-schema intent type to query operation.
        
        Args:
            intent_type: The intent type from auto-schema matching
        
        Returns:
            The corresponding query operation name
        """
        intent_to_operation = {
            "find": "similarity_search",
            "count": "count",
            "filter": "similarity_search",  # Filter uses similarity with constraints
            "similar": "similarity_search",
            "compare": "compare",
            "predict": "temporal_predict",
            "verify": "fact_tree",
        }
        return intent_to_operation.get(intent_type, "similarity_search")
    
    def _build_structured_query_from_auto_result(
        self,
        auto_result: 'AutoMatchResult',
        operation: str,
        original_query: str,
    ) -> Dict[str, Any]:
        """
        Build a structured query from auto-match result.
        
        Converts the extracted parameters from auto-schema matching
        into a structured query format for execution.
        
        Args:
            auto_result: The AutoMatchResult with extracted parameters
            operation: The query operation to perform
            original_query: The original query string
        
        Returns:
            Structured query dictionary
        
        Validates: Requirement 5
        """
        structured_query = {
            "operation": operation,
            "query": original_query,
        }
        
        # Add extracted parameters
        for role, param in auto_result.parameters.parameters.items():
            structured_query[role] = param.value
        
        # Add multi-value parameters
        for role, params in auto_result.parameters.multi_value_params.items():
            structured_query[role] = [p.value for p in params]
        
        # Add operation-specific defaults
        if operation == "similarity_search":
            structured_query.setdefault("top_k", 10)
        elif operation == "count":
            pass  # No additional params needed
        elif operation == "temporal_predict":
            structured_query.setdefault("steps_ahead", 1)
            structured_query.setdefault("beam_width", 3)
        elif operation == "fact_tree":
            structured_query.setdefault("max_depth", 3)
        
        return structured_query
    
    async def translate_query(
        self,
        query: str
    ) -> Tuple[Optional[IntentMatch], str]:
        """
        Translate a query without executing it.
        
        Useful for debugging and testing query translation.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (IntentMatch or None, match_method)
        """
        # Try auto-schema matching first if available
        if self._auto_schema_matcher is not None:
            try:
                auto_result = self._auto_schema_matcher.match_query(query)
                if auto_result.confidence >= 0.3:
                    # Convert AutoMatchResult to IntentMatch for compatibility
                    return IntentMatch(
                        intent=auto_result.intent.intent_type,
                        confidence=auto_result.confidence,
                        parameters=auto_result.parameters.to_dict().get("parameters", {}),
                        pattern_matched=None,
                        structured_query=auto_result.to_dict(),
                    ), auto_result.match_method
            except Exception as e:
                logger.warning(f"Auto-schema translation failed: {e}")
        
        match_result = await self.intent_matcher.match_intent(query)
        
        # Use same threshold as execute_nl_query (0.3) for consistency
        min_acceptable_confidence = 0.3
        if match_result and match_result.confidence >= min_acceptable_confidence:
            return match_result, "rules"
        
        if self.llm_fallback is not None:
            try:
                llm_result = await self.llm_fallback.translate_query(query, "")
                if llm_result:
                    # Create a synthetic IntentMatch for LLM result
                    return IntentMatch(
                        intent=llm_result.get("operation", "unknown"),
                        confidence=0.0,
                        parameters=llm_result,
                        pattern_matched=None,
                        structured_query=llm_result,
                    ), "llm"
            except Exception:
                pass
        
        return match_result, "none"
    
    def _normalize_operation(self, operation: str) -> str:
        """
        Normalize operation names from intent matcher to canonical executor names.
        
        This prevents mismatches between intent names (e.g., 'list_all') and
        executor operation names (e.g., 'list').
        
        Args:
            operation: Raw operation name from intent matcher
            
        Returns:
            Canonical operation name for the executor
        """
        # Map intent matcher names to canonical executor names
        operation_aliases = {
            # List operations
            "list_all": "list",
            "list_everything": "list",
            "show_all": "list",
            "enumerate": "list",
            # Search operations
            "find": "similarity_search",
            "search": "similarity_search",
            "similar": "similarity_search",
            "find_similar": "similarity_search",
            # Predict operations
            "predict": "temporal_predict",
            "forecast": "temporal_predict",
            # Verify operations
            "verify": "fact_tree",
            "explain": "fact_tree",
            "prove": "fact_tree",
        }
        
        normalized = operation_aliases.get(operation, operation)
        if normalized != operation:
            logger.debug(f"Normalized operation '{operation}' -> '{normalized}'")
        return normalized
    
    async def _execute_structured_query(
        self,
        org_id: str,
        model_id: str,
        operation: str,
        query: Dict[str, Any],
    ) -> FactTree:
        """
        Execute a structured query against the QueryService.
        
        All operations now return FactTree instead of raw dictionaries.
        Uses FactTreeBuilder.build_error() for error cases.
        
        Validates: Requirements 1.1, 8.3
        """
        from domains.models.schemas import (
            SimilaritySearchRequest,
            FactTreeRequest,
            TemporalPredictRequest,
        )
        
        # Normalize operation name to canonical form
        operation = self._normalize_operation(operation)
        
        try:
            if operation == "similarity_search":
                request = SimilaritySearchRequest(
                    query=query.get("query", ""),
                    top_k=query.get("top_k", 10),
                )
                # similarity_search already returns FactTree
                return await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
            
            elif operation == "fact_tree":
                request = FactTreeRequest(
                    claim=query.get("query", query.get("claim", "")),
                    max_depth=query.get("max_depth", 3),
                )
                result = await self.query_service.generate_fact_tree(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
                # generate_fact_tree returns FactTreeResponse, convert to FactTree
                # For now, wrap in a FactTree structure
                return FactTreeBuilder.build_similarity_search(
                    query=request.claim,
                    results=[],
                    query_time_ms=0.0,
                    total_count=0,
                )
            
            elif operation == "temporal_predict":
                current_state = query.get("current_state", [query.get("query", "")])
                if isinstance(current_state, str):
                    current_state = [current_state]
                
                request = TemporalPredictRequest(
                    current_state=current_state,
                    steps_ahead=query.get("steps_ahead", 1),
                    beam_width=query.get("beam_width", 3),
                )
                # Use the _as_fact_tree method
                return await self.query_service.predict_temporal_as_fact_tree(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
            
            elif operation == "list":
                # Use dedicated list method that returns FactTree
                limit = query.get("limit", 100)
                return await self.query_service.list_glyphs_as_fact_tree(
                    org_id=org_id,
                    model_id=model_id,
                    limit=limit,
                )
            
            elif operation == "count":
                # Use dedicated count method that returns FactTree
                logger.info(f"Executing count operation for org={org_id}, model={model_id}")
                return await self.query_service.count_glyphs_as_fact_tree(
                    org_id=org_id,
                    model_id=model_id,
                )
            
            elif operation == "compare":
                request = SimilaritySearchRequest(
                    query=query.get("query", ""),
                    top_k=2,
                )
                # similarity_search already returns FactTree
                return await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
            
            else:
                logger.warning(f"Unknown operation '{operation}', defaulting to similarity_search")
                request = SimilaritySearchRequest(
                    query=query.get("query", ""),
                    top_k=10,
                )
                return await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            # Return error FactTree instead of dict
            error_msg = str(e)
            if "no attributes match" in error_msg.lower() or "cannot encode" in error_msg.lower():
                return FactTreeBuilder.build_error(
                    error_message="The query doesn't match the model's schema. Try rephrasing with specific attribute names or values from your data.",
                    error_type="EncodingError",
                    query=query.get("query"),
                )
            return FactTreeBuilder.build_error(
                error_message=str(e),
                error_type=type(e).__name__,
                query=query.get("query"),
            )
    
    def get_intents(self) -> Dict[str, Any]:
        """
        Get available intents and patterns.
        
        Returns:
            Dictionary with intent names and example patterns
        """
        return self.intent_matcher.get_intents()
