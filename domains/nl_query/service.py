"""
Natural Language Query Service.

Provides hybrid rules-first + LLM-fallback query translation and execution.
The core principle: "When your LLM can't afford to be wrong, sidecar it with Glyphh."
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from domains.nl_query.intent_matcher import IntentMatcher, IntentMatch
from domains.query.service import QueryService

logger = logging.getLogger(__name__)


@dataclass
class NLQueryResult:
    """Result of executing a natural language query."""
    result: Any
    query_type: str
    match_method: str  # "rules", "llm", or "none"
    confidence: float
    translated_query: Optional[Dict[str, Any]]
    query_time_ms: float


class NLQueryService:
    """
    Natural Language Query Service with hybrid rules + LLM approach.
    
    Uses rules-first matching via IntentMatcher (HDC similarity).
    Falls back to LLM only when rules can't confidently match.
    """
    
    def __init__(
        self,
        query_service: QueryService,
        intent_matcher: IntentMatcher,
        llm_fallback: Optional[Any] = None,
        confidence_threshold: float = 0.85,
    ):
        """
        Initialize the NL Query Service.
        
        Args:
            query_service: QueryService for executing structured queries
            intent_matcher: IntentMatcher for rules-based matching
            llm_fallback: Optional LLMFallback for low-confidence queries
            confidence_threshold: Minimum confidence for rules-based match
        """
        self.query_service = query_service
        self.intent_matcher = intent_matcher
        self.llm_fallback = llm_fallback
        self.confidence_threshold = confidence_threshold
    
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
        1. Try rules-based matching (IntentMatcher)
        2. If confidence >= threshold: execute directly
        3. If confidence < threshold AND LLM enabled: use LLM fallback
        4. If confidence < threshold AND no LLM: return "none" match_method
        """
        start_time = time.time()
        
        logger.info(f"NL query received: '{query}' for org={org_id}, model={model_id}")
        
        # Step 1: Try rules-based matching
        match_result = await self.intent_matcher.match_intent(query)
        
        if match_result and match_result.confidence >= self.confidence_threshold:
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
                result=result,
                query_type=match_result.intent,
                match_method="rules",
                confidence=match_result.confidence,
                translated_query=match_result.structured_query if debug else None,
                query_time_ms=elapsed_ms,
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
                        result=result,
                        query_type=llm_result.get("operation", "unknown"),
                        match_method="llm",
                        confidence=0.0,
                        translated_query=llm_result if debug else None,
                        query_time_ms=elapsed_ms,
                    )
            except Exception as e:
                logger.warning(f"LLM fallback failed: {e}")
        
        # Step 3: No match
        logger.info(f"No intent match for query: '{query}'")
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return NLQueryResult(
            result=None,
            query_type="unknown",
            match_method="none",
            confidence=match_result.confidence if match_result else 0.0,
            translated_query=None,
            query_time_ms=elapsed_ms,
        )
    
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
        match_result = await self.intent_matcher.match_intent(query)
        
        if match_result and match_result.confidence >= self.confidence_threshold:
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
    
    async def _execute_structured_query(
        self,
        org_id: str,
        model_id: str,
        operation: str,
        query: Dict[str, Any],
    ) -> Any:
        """Execute a structured query against the QueryService."""
        from domains.models.schemas import (
            SimilaritySearchRequest,
            FactTreeRequest,
            TemporalPredictRequest,
        )
        
        try:
            if operation == "similarity_search":
                request = SimilaritySearchRequest(
                    query=query.get("query", ""),
                    top_k=query.get("top_k", 10),
                )
                result = await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
                return result.model_dump() if hasattr(result, 'model_dump') else result
            
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
                return result.model_dump() if hasattr(result, 'model_dump') else result
            
            elif operation == "temporal_predict":
                current_state = query.get("current_state", [query.get("query", "")])
                if isinstance(current_state, str):
                    current_state = [current_state]
                
                request = TemporalPredictRequest(
                    current_state=current_state,
                    steps_ahead=query.get("steps_ahead", 1),
                    beam_width=query.get("beam_width", 3),
                )
                result = await self.query_service.predict_temporal(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
                return result.model_dump() if hasattr(result, 'model_dump') else result
            
            elif operation == "list":
                request = SimilaritySearchRequest(
                    query="",
                    top_k=query.get("limit", 100),
                )
                result = await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
                return result.model_dump() if hasattr(result, 'model_dump') else result
            
            elif operation == "count":
                request = SimilaritySearchRequest(query="", top_k=1000)
                result = await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
                return {"count": result.total_count}
            
            elif operation == "compare":
                request = SimilaritySearchRequest(
                    query=query.get("query", ""),
                    top_k=2,
                )
                result = await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
                return result.model_dump() if hasattr(result, 'model_dump') else result
            
            else:
                logger.warning(f"Unknown operation '{operation}', defaulting to similarity_search")
                request = SimilaritySearchRequest(
                    query=query.get("query", ""),
                    top_k=10,
                )
                result = await self.query_service.similarity_search(
                    org_id=org_id,
                    model_id=model_id,
                    request=request,
                )
                return result.model_dump() if hasattr(result, 'model_dump') else result
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_intents(self) -> Dict[str, Any]:
        """
        Get available intents and patterns.
        
        Returns:
            Dictionary with intent names and example patterns
        """
        return self.intent_matcher.get_intents()
