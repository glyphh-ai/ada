"""
Natural Language Query API Routes.

Endpoints for NL query translation and execution.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/{namespace}", tags=["nl_query"])
settings = get_settings()


# Request/Response Models
class NLQueryRequest(BaseModel):
    """Natural language query request."""
    query: str = Field(..., description="Natural language query", min_length=1)
    debug: bool = Field(default=False, description="Include translation details")


class NLQueryResponse(BaseModel):
    """Natural language query response."""
    result: Any
    query_type: str
    match_method: str
    confidence: float
    translated_query: Optional[Dict[str, Any]] = None
    query_time_ms: float


class IntentsResponse(BaseModel):
    """Available intents response."""
    intents: List[str]
    patterns: Dict[str, List[str]]


class TranslationDebugRequest(BaseModel):
    """Debug translation request."""
    query: str = Field(..., description="Query to translate")


class TranslationDebugResponse(BaseModel):
    """Debug translation response."""
    intent: Optional[str]
    confidence: float
    parameters: Dict[str, str]
    pattern_matched: Optional[str]
    match_method: str


# Service factory
def get_nl_query_service():
    """Get NL query service instance."""
    from main import model_manager
    from domains.query.service import QueryService
    from domains.nl_query.intent_matcher import IntentMatcher
    from domains.nl_query.service import NLQueryService
    from infrastructure.database import async_session_maker
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    if not settings.enable_nl_query:
        raise HTTPException(
            status_code=501,
            detail="Natural language query is not enabled. Set ENABLE_NL_QUERY=true"
        )
    
    # Create services
    query_service = QueryService(model_manager, async_session_maker)
    intent_matcher = IntentMatcher(confidence_threshold=0.85)
    
    # Create LLM fallback if available
    llm_fallback = None
    try:
        from domains.nl_query.llm_fallback import LLMFallback
        llm_fallback = LLMFallback(model_name=settings.nl_model)
        if not llm_fallback.is_available():
            llm_fallback = None
            logger.info("LLM fallback disabled (transformers not available)")
    except ImportError:
        logger.info("LLM fallback disabled (import error)")
    
    return NLQueryService(
        query_service=query_service,
        intent_matcher=intent_matcher,
        llm_fallback=llm_fallback,
        confidence_threshold=0.85,
    )


# Endpoints
@router.post("/query", response_model=NLQueryResponse)
async def execute_nl_query(
    namespace: str,
    request: NLQueryRequest,
) -> NLQueryResponse:
    """
    Execute a natural language query.
    
    Translates the query using rules-first approach with optional
    LLM fallback, then executes the translated query.
    
    Returns 422 if translation fails and LLM is disabled.
    """
    service = get_nl_query_service()
    
    logger.info(f"NL query: namespace={namespace}, query='{request.query}'")
    
    result = await service.execute_nl_query(
        namespace=namespace,
        query=request.query,
        debug=request.debug,
    )
    
    if result.match_method == "none":
        # Translation failed
        suggestions = [
            "Try rephrasing your query",
            "Use keywords like 'find', 'similar', 'verify', 'predict'",
            "Example: 'find similar to machine learning'",
            "Example: 'verify that X is related to Y'",
            "Example: 'predict what comes after X'",
        ]
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Could not understand query",
                "suggestions": suggestions,
                "original_query": request.query,
                "confidence": result.confidence,
            }
        )
    
    return NLQueryResponse(
        result=result.result,
        query_type=result.query_type,
        match_method=result.match_method,
        confidence=result.confidence,
        translated_query=result.translated_query,
        query_time_ms=result.query_time_ms,
    )


@router.get("/intents", response_model=IntentsResponse)
async def get_intents(namespace: str) -> IntentsResponse:
    """
    Get available intents and patterns.
    
    Returns the list of supported query types and their
    pattern templates.
    """
    service = get_nl_query_service()
    intents = service.get_intents()
    
    return IntentsResponse(
        intents=intents["intents"],
        patterns=intents["patterns"],
    )


@router.post("/query/debug", response_model=TranslationDebugResponse)
async def debug_translation(
    namespace: str,
    request: TranslationDebugRequest,
) -> TranslationDebugResponse:
    """
    Debug query translation without execution.
    
    Shows how the query would be translated without
    actually executing it.
    """
    service = get_nl_query_service()
    
    logger.info(f"Debug translation: namespace={namespace}, query='{request.query}'")
    
    match_result, match_method = await service.translate_query(request.query)
    
    if match_result:
        return TranslationDebugResponse(
            intent=match_result.intent,
            confidence=match_result.confidence,
            parameters=match_result.parameters,
            pattern_matched=match_result.pattern_matched,
            match_method=match_method,
        )
    else:
        return TranslationDebugResponse(
            intent=None,
            confidence=0.0,
            parameters={},
            pattern_matched=None,
            match_method="none",
        )
