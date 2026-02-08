"""
Natural Language Query API Routes.

Endpoints for NL query translation and execution.
All routes scoped by /{org_id}/{model_id}/...
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}", tags=["nl_query"])
settings = get_settings()


# Request/Response Models
class NLQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query", min_length=1)
    debug: bool = Field(default=False, description="Include translation details")


class NLQueryResponse(BaseModel):
    result: Any
    query_type: str
    match_method: str
    confidence: float
    translated_query: Optional[Dict[str, Any]] = None
    query_time_ms: float


class IntentsResponse(BaseModel):
    intents: List[str]
    patterns: Dict[str, List[str]]


class TranslationDebugRequest(BaseModel):
    query: str = Field(..., description="Query to translate")


class TranslationDebugResponse(BaseModel):
    intent: Optional[str]
    confidence: float
    parameters: Dict[str, str]
    pattern_matched: Optional[str]
    match_method: str


def get_nl_query_service():
    from main import model_manager
    from domains.query.service import QueryService
    from domains.nl_query.intent_matcher import IntentMatcher
    from domains.nl_query.service import NLQueryService
    from infrastructure.database import async_session_maker
    from shared.encoder_config_factory import EncoderConfigFactory
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    if not settings.enable_nl_query:
        raise HTTPException(
            status_code=501,
            detail="Natural language query is not enabled. Set ENABLE_NL_QUERY=true"
        )
    
    model_nl_config = None
    try:
        model = model_manager.get_current_model() if hasattr(model_manager, 'get_current_model') else None
        if model is not None:
            model_nl_config = EncoderConfigFactory.extract_nl_encoder_config(model)
            if model_nl_config:
                logger.info(f"Using NL encoder config from model with {len(model_nl_config.get('patterns', []))} patterns")
    except Exception as e:
        logger.warning(f"Failed to extract NL config from model: {e}")
    
    query_service = QueryService(model_manager, async_session_maker)
    intent_matcher = IntentMatcher(
        confidence_threshold=0.85,
        model_nl_config=model_nl_config
    )
    
    llm_fallback = None
    try:
        from domains.nl_query.llm_fallback import LLMFallback
        llm_fallback = LLMFallback(model_name=settings.nl_model)
        if not llm_fallback.is_available():
            llm_fallback = None
    except ImportError:
        pass
    
    return NLQueryService(
        query_service=query_service,
        intent_matcher=intent_matcher,
        llm_fallback=llm_fallback,
        confidence_threshold=0.85,
    )


@router.post("/query", response_model=NLQueryResponse)
async def execute_nl_query(
    org_id: str,
    model_id: str,
    request: NLQueryRequest,
) -> NLQueryResponse:
    """Execute a natural language query."""
    service = get_nl_query_service()
    
    logger.info(f"NL query: org={org_id}, model={model_id}, query='{request.query}'")
    
    result = await service.execute_nl_query(
        org_id=org_id,
        model_id=model_id,
        query=request.query,
        debug=request.debug,
    )
    
    if result.match_method == "none":
        suggestions = [
            "Try rephrasing your query",
            "Use keywords like 'find', 'similar', 'verify', 'predict'",
            "Example: 'find similar to machine learning'",
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
async def get_intents(org_id: str, model_id: str) -> IntentsResponse:
    """Get available intents and patterns."""
    service = get_nl_query_service()
    intents = service.get_intents()
    return IntentsResponse(intents=intents["intents"], patterns=intents["patterns"])


@router.post("/query/debug", response_model=TranslationDebugResponse)
async def debug_translation(
    org_id: str,
    model_id: str,
    request: TranslationDebugRequest,
) -> TranslationDebugResponse:
    """Debug query translation without execution."""
    service = get_nl_query_service()
    
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
            intent=None, confidence=0.0, parameters={},
            pattern_matched=None, match_method="none",
        )
