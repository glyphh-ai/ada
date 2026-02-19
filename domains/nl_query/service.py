"""
Natural Language Query Service.

Provides query translation and execution via stored procedures and similarity search.

Updated to support AutoSchemaMatcher for automatic schema-based NL query matching.
Updated to return SDK FactTree in all responses for unified response format.
Updated with response state pattern: DONE/ASK/BLOCKED/AUTH_REQUIRED/ERROR.

Design Principle: "When your LLM can't afford to be wrong, sidecar it with Glyphh"
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from glyphh.fact_tree.builder import FactTree

from domains.query.service import QueryService
from domains.query.fact_tree_builder import FactTreeBuilder

if TYPE_CHECKING:
    from domains.nl_query.schema_index import SchemaIndex
    from glyphh.nl.auto_schema_matcher import AutoSchemaMatcher, AutoMatchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IntentMatch (lightweight result for translate_query)
# ---------------------------------------------------------------------------

@dataclass
class IntentMatch:
    """Result of matching a query against intent patterns."""
    intent: str
    confidence: float
    parameters: Dict[str, str]
    pattern_matched: Optional[str]
    structured_query: Dict[str, Any]
    match_method: str = "default"


# ---------------------------------------------------------------------------
# Response State Model
# ---------------------------------------------------------------------------

class ResponseState(str, Enum):
    """Response state indicating the outcome of an operation."""
    DONE = "DONE"
    ASK = "ASK"
    BLOCKED = "BLOCKED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    ERROR = "ERROR"


@dataclass
class AskPayload:
    """Payload for ASK state - missing slots or disambiguation needed."""
    question: str
    missing_slots: List[str] = field(default_factory=list)
    disambiguation_options: List[Dict[str, Any]] = field(default_factory=list)
    provided_slots: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "missing_slots": self.missing_slots,
            "disambiguation_options": self.disambiguation_options,
            "provided_slots": self.provided_slots,
        }


@dataclass
class BlockedPayload:
    """Payload for BLOCKED state - policy/permission prevented execution."""
    reason: str
    policy_refs: List[str] = field(default_factory=list)
    constraint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "policy_refs": self.policy_refs,
            "constraint": self.constraint,
        }


@dataclass
class AuthRequiredPayload:
    """Payload for AUTH_REQUIRED state - authentication needed."""
    provider: str
    app: str
    auth_hint: str = "Connect the required account to proceed."
    tool_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "app": self.app,
            "auth_hint": self.auth_hint,
            "tool_id": self.tool_id,
        }


@dataclass
class ErrorPayload:
    """Payload for ERROR state - unexpected failure."""
    error_code: str
    message: str
    debug_info: Optional[Dict[str, Any]] = None
    recoverable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "error_code": self.error_code,
            "message": self.message,
            "recoverable": self.recoverable,
        }
        if self.debug_info:
            result["debug_info"] = self.debug_info
        return result


# ---------------------------------------------------------------------------
# NLQueryResult with Response State Pattern
# ---------------------------------------------------------------------------

@dataclass
class NLQueryResult:
    """
    Result of executing a natural language query.

    Uses the response state pattern:
    - DONE: Completed successfully; has fact_tree + trace_id
    - ASK: Missing slots / needs disambiguation; returns question + slots[]
    - BLOCKED: Policy/permission prevented execution; returns reason + policy_refs
    - AUTH_REQUIRED: Authentication needed; returns provider + app + auth_hint
    - ERROR: Unexpected failure; returns error_code + debug_info (safe)
    """
    state: ResponseState
    fact_tree: Optional[FactTree] = None
    query_type: str = ""
    match_method: str = ""  # "auto", "similarity", or "none"
    confidence: float = 0.0
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    translated_query: Optional[Dict[str, Any]] = None
    query_time_ms: float = 0.0
    matched_route: Optional[str] = None

    # State-specific payloads
    ask: Optional[AskPayload] = None
    blocked: Optional[BlockedPayload] = None
    auth_required: Optional[AuthRequiredPayload] = None
    error: Optional[ErrorPayload] = None

    @property
    def disambiguation_needed(self) -> bool:
        """Backward compat: True when state is ASK with disambiguation options."""
        return (
            self.state == ResponseState.ASK
            and self.ask is not None
            and len(self.ask.disambiguation_options) > 0
        )

    @property
    def disambiguation_suggestions(self) -> List[str]:
        """Backward compat: Disambiguation suggestions as strings."""
        if self.ask and self.ask.disambiguation_options:
            return [
                opt.get("suggestion", opt.get("intent", str(opt)))
                for opt in self.ask.disambiguation_options
            ]
        return []

    @property
    def result(self) -> Any:
        """Backward compat: Returns fact_tree JSON for DONE, error dict otherwise."""
        if self.state == ResponseState.DONE and self.fact_tree is not None:
            return self.fact_tree.to_json()
        if self.state == ResponseState.ERROR and self.error is not None:
            return {"error": self.error.to_dict()}
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API response."""
        result = {
            "state": self.state.value,
            "confidence": self.confidence,
            "trace_id": self.trace_id,
            "query_type": self.query_type,
            "match_method": self.match_method,
            "query_time_ms": self.query_time_ms,
        }

        if self.matched_route:
            result["matched_route"] = self.matched_route

        if self.translated_query:
            result["translated_query"] = self.translated_query

        if self.state == ResponseState.DONE and self.fact_tree is not None:
            result["fact_tree"] = self.fact_tree.to_json()
        elif self.state == ResponseState.ASK and self.ask:
            result["ask"] = self.ask.to_dict()
        elif self.state == ResponseState.BLOCKED and self.blocked:
            result["blocked"] = self.blocked.to_dict()
        elif self.state == ResponseState.AUTH_REQUIRED and self.auth_required:
            result["auth_required"] = self.auth_required.to_dict()
        elif self.state == ResponseState.ERROR and self.error:
            result["error"] = self.error.to_dict()

        return result


class NLQueryService:
    """
    Natural Language Query Service.
    
    Routes NL queries to the appropriate operation:
    - AutoSchemaMatcher for schema-based matching (when available)
    - Direct similarity search as the default path
    
    The legacy IntentMatcher has been removed — each model's own encoder
    handles NL→query translation via encode_query_fn.
    
    Validates: Requirements 3, 4, 5, 12.2, 12.5
    """
    
    def __init__(
        self,
        query_service: QueryService,
        confidence_threshold: float = 0.85,
        schema_index: Optional['SchemaIndex'] = None,
        auto_schema_matcher: Optional['AutoSchemaMatcher'] = None,
    ):
        """
        Initialize the NL Query Service.

        Args:
            query_service: QueryService for executing structured queries
            confidence_threshold: Minimum confidence threshold
            schema_index: Optional SchemaIndex for auto-schema matching
            auto_schema_matcher: Optional AutoSchemaMatcher for auto-schema matching
        """
        self.query_service = query_service
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
        2. Default to similarity search (model's encode_query_fn handles NL)
        3. If encoding fails: ERROR
        """
        start_time = time.time()

        logger.info(f"NL query received: '{query}' for org={org_id}, model={model_id}")

        # Step 0: Try auto-schema matching if available
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
                logger.warning(f"Auto-schema matching failed: {e}, falling back to similarity search")

        # Step 1: Direct similarity search — the model's encode_query_fn
        # handles NL→embedding translation
        try:
            fact_tree = await self._execute_structured_query(
                org_id,
                model_id,
                "similarity_search",
                {"query": query, "top_k": 10},
            )

            elapsed_ms = (time.time() - start_time) * 1000

            return NLQueryResult(
                state=ResponseState.DONE,
                fact_tree=fact_tree,
                query_type="similarity_search",
                match_method="direct",
                confidence=1.0,
                translated_query={"operation": "similarity_search", "query": query} if debug else None,
                query_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")

        # Step 2: No match -> ERROR
        logger.info(f"No match for query: '{query}'")

        elapsed_ms = (time.time() - start_time) * 1000

        return NLQueryResult(
            state=ResponseState.ERROR,
            query_type="unknown",
            match_method="none",
            confidence=0.0,
            query_time_ms=elapsed_ms,
            error=ErrorPayload(
                error_code="NO_MATCH",
                message="Could not process query",
                debug_info={"query": query},
                recoverable=True,
            ),
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

        Returns NLQueryResult with appropriate state:
        - DONE: Match found and executed
        - ASK: Disambiguation needed
        - None: Confidence too low, fall through to rules
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

        # Use AutoSchemaMatcher to match the query
        auto_result = self._auto_schema_matcher.match_query(query)

        # Cache the match result if schema index is available
        if self._schema_index is not None and cache_key is not None:
            self._schema_index.cache_match_result(cache_key, auto_result.match_result)

        # Check if disambiguation is needed -> ASK
        if auto_result.disambiguation_needed:
            logger.info(
                f"Disambiguation needed for query: '{query}', "
                f"options: {[opt.intent_type for opt in auto_result.disambiguation_options]}"
            )

            options = [
                {
                    "intent": opt.intent_type,
                    "confidence": opt.confidence,
                    "suggestion": f"Did you mean '{opt.intent_type}'? (confidence: {opt.confidence:.0%})",
                }
                for opt in auto_result.disambiguation_options
            ]

            elapsed_ms = (time.time() - start_time) * 1000

            return NLQueryResult(
                state=ResponseState.ASK,
                query_type=auto_result.intent.intent_type,
                match_method=auto_result.match_method,
                confidence=auto_result.confidence,
                translated_query=auto_result.to_dict() if debug else None,
                query_time_ms=elapsed_ms,
                ask=AskPayload(
                    question="Your query is ambiguous. Please clarify:",
                    disambiguation_options=options,
                ),
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

        # Execute the query -> DONE
        fact_tree = await self._execute_structured_query(
            org_id,
            model_id,
            operation,
            structured_query,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return NLQueryResult(
            state=ResponseState.DONE,
            fact_tree=fact_tree,
            query_type=auto_result.intent.intent_type,
            match_method=auto_result.match_method,
            confidence=auto_result.confidence,
            translated_query=auto_result.to_dict() if debug else None,
            query_time_ms=elapsed_ms,
        )
    
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
                    return IntentMatch(
                        intent=auto_result.intent.intent_type,
                        confidence=auto_result.confidence,
                        parameters=auto_result.parameters.to_dict().get("parameters", {}),
                        pattern_matched=None,
                        structured_query=auto_result.to_dict(),
                    ), auto_result.match_method
            except Exception as e:
                logger.warning(f"Auto-schema translation failed: {e}")
        
        # Default: similarity search
        return IntentMatch(
            intent="similarity_search",
            confidence=1.0,
            parameters={},
            pattern_matched=None,
            structured_query={"operation": "similarity_search", "query": query},
        ), "direct"
    
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

        All operations return FactTree. On failure, raises so the caller
        can decide whether to return ERROR state or handle differently.
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
                # fact_tree intent is effectively a similarity search —
                # route through similarity_search which properly encodes
                # and returns scored results.
                request = SimilaritySearchRequest(
                    query=query.get("query", query.get("claim", "")),
                    top_k=query.get("top_k", 10),
                )
                return await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
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
        return {"intents": ["similarity_search"], "patterns": {}}
