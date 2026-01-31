"""
Natural Language Query Service.

Provides hybrid rules-first + LLM-fallback query translation
and execution.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from domains.nl_query.default_intents import (
    INTENT_SIMILARITY_SEARCH,
    INTENT_FACT_TREE,
    INTENT_TEMPORAL_PREDICTION,
    INTENT_GLYPH_LOOKUP,
    INTENT_EDGE_QUERY,
)
from domains.nl_query.intent_matcher import IntentMatcher, MatchResult
from domains.query.service import QueryService, Permissions
from domains.models.schemas import (
    SimilaritySearchRequest,
    FactTreeRequest,
    TemporalPredictRequest,
)
from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class NLQueryResult:
    """Result of natural language query."""
    result: Any
    query_type: str
    match_method: str  # "rules" or "llm"
    confidence: float
    translated_query: Optional[Dict[str, Any]] = None
    query_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "query_type": self.query_type,
            "match_method": self.match_method,
            "confidence": self.confidence,
            "translated_query": self.translated_query,
            "query_time_ms": self.query_time_ms,
        }


class NLQueryService:
    """
    Natural Language Query Service.
    
    Translates natural language queries to structured queries
    using rules-first approach with optional LLM fallback.
    
    Responsibilities:
    - Translate NL queries using IntentMatcher
    - Fall back to LLM when rules fail (if enabled)
    - Execute translated queries
    - Return results with match method indicator
    """
    
    def __init__(
        self,
        query_service: QueryService,
        intent_matcher: IntentMatcher,
        llm_fallback: Optional[Any] = None,
    ):
        """
        Initialize NLQueryService.
        
        Args:
            query_service: QueryService for executing queries
            intent_matcher: IntentMatcher for rules-based matching
            llm_fallback: Optional LLMFallback for when rules fail
        """
        self._query_service = query_service
        self._intent_matcher = intent_matcher
        self._llm_fallback = llm_fallback
    
    async def translate_query(
        self,
        query: str,
    ) -> tuple[Optional[MatchResult], str]:
        """
        Translate natural language query to structured query.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (MatchResult, match_method)
        """
        # Try rules-based matching first
        result = self._intent_matcher.match_intent(query)
        
        if result:
            logger.debug(
                f"Rules matched: intent={result.intent}, "
                f"confidence={result.confidence:.2f}"
            )
            return result, "rules"
        
        # Try LLM fallback if available
        if self._llm_fallback:
            try:
                result = await self._llm_fallback.translate(query)
                if result:
                    logger.debug(
                        f"LLM matched: intent={result.intent}, "
                        f"confidence={result.confidence:.2f}"
                    )
                    return result, "llm"
            except Exception as e:
                logger.warning(f"LLM fallback failed: {e}")
        
        return None, "none"
    
    async def execute_nl_query(
        self,
        namespace: str,
        query: str,
        permissions: Optional[Permissions] = None,
        debug: bool = False,
    ) -> NLQueryResult:
        """
        Execute a natural language query.
        
        Args:
            namespace: Model namespace
            query: Natural language query
            permissions: User permissions
            debug: Include translation details in response
            
        Returns:
            NLQueryResult with execution results
        """
        start_time = time.time()
        
        # Translate query
        match_result, match_method = await self.translate_query(query)
        
        if not match_result:
            # Return error result
            return NLQueryResult(
                result=None,
                query_type="unknown",
                match_method="none",
                confidence=0.0,
                translated_query={"original": query} if debug else None,
                query_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Execute based on intent
        result = await self._execute_intent(
            namespace,
            match_result,
            permissions,
        )
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return NLQueryResult(
            result=result,
            query_type=match_result.intent,
            match_method=match_method,
            confidence=match_result.confidence,
            translated_query=match_result.to_dict() if debug else None,
            query_time_ms=query_time_ms,
        )
    
    async def _execute_intent(
        self,
        namespace: str,
        match_result: MatchResult,
        permissions: Optional[Permissions],
    ) -> Any:
        """Execute the matched intent."""
        intent = match_result.intent
        params = match_result.parameters
        
        if intent == INTENT_SIMILARITY_SEARCH:
            request = SimilaritySearchRequest(
                query=params.get("query", ""),
                top_k=10,
            )
            return await self._query_service.similarity_search(
                namespace, request, permissions
            )
        
        elif intent == INTENT_FACT_TREE:
            request = FactTreeRequest(
                claim=params.get("claim", params.get("query", "")),
                max_depth=3,
            )
            return await self._query_service.generate_fact_tree(
                namespace, request, permissions
            )
        
        elif intent == INTENT_TEMPORAL_PREDICTION:
            state = params.get("state", params.get("query", ""))
            request = TemporalPredictRequest(
                current_state=[state],
                steps_ahead=3,
            )
            return await self._query_service.predict_temporal(
                namespace, request, permissions
            )
        
        elif intent == INTENT_GLYPH_LOOKUP:
            # Return glyph ID for lookup
            return {"glyph_id": params.get("id", params.get("query", ""))}
        
        elif intent == INTENT_EDGE_QUERY:
            # Return edge query parameters
            return {
                "source": params.get("source", params.get("query", "")),
                "target": params.get("target"),
            }
        
        else:
            logger.warning(f"Unknown intent: {intent}")
            return None
    
    def get_intents(self) -> Dict[str, Any]:
        """Get available intents and patterns."""
        return {
            "intents": list(self._intent_matcher.get_patterns().keys()),
            "patterns": self._intent_matcher.get_patterns(),
        }
