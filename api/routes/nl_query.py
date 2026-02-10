"""
Natural Language Query API Routes.

Endpoints for NL query translation and execution.
All routes scoped by /{org_id}/{model_id}/...

Updated to support AutoSchemaMatcher for automatic schema-based NL query matching.
Validates: Requirements 3, 4, 5, 12.2, 12.5
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


class DisambiguationOption(BaseModel):
    """A disambiguation option for ambiguous queries."""
    intent: str
    description: str
    confidence: float


class NLQueryResponse(BaseModel):
    result: Any
    query_type: str
    match_method: str
    confidence: float
    translated_query: Optional[Dict[str, Any]] = None
    query_time_ms: float
    disambiguation_needed: bool = False
    disambiguation_suggestions: List[str] = Field(default_factory=list)


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
    disambiguation_needed: bool = False
    disambiguation_suggestions: List[str] = Field(default_factory=list)


def get_nl_query_service():
    """
    Get the NL Query Service with AutoSchemaMatcher support.
    
    Initializes the service with:
    - IntentMatcher for rules-based matching
    - AutoSchemaMatcher for auto-schema matching (if model has schema index)
    - LLM fallback (if configured)
    
    Validates: Requirements 3, 4, 5
    """
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
    auto_schema_matcher = None
    schema_index = None
    
    try:
        model = model_manager.get_current_model() if hasattr(model_manager, 'get_current_model') else None
        if model is not None:
            model_nl_config = EncoderConfigFactory.extract_nl_encoder_config(model)
            if model_nl_config:
                logger.info(f"Using NL encoder config from model with {len(model_nl_config.get('patterns', []))} patterns")
            
            # Try to get or create AutoSchemaMatcher for the model
            # Validates: Requirements 3, 4, 5
            auto_schema_matcher, schema_index = _get_auto_schema_matcher(model)
            
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
        schema_index=schema_index,
        auto_schema_matcher=auto_schema_matcher,
    )


def _get_auto_schema_matcher(model):
    """
    Get or create AutoSchemaMatcher for a model.
    
    Creates an AutoSchemaMatcher using the model's encoder and config,
    and optionally a SchemaIndex for caching.
    
    Args:
        model: The deployed model object
    
    Returns:
        Tuple of (AutoSchemaMatcher or None, SchemaIndex or None)
    
    Validates: Requirements 3, 4, 5
    """
    try:
        # Import SDK components
        from glyphh.nl.auto_schema_matcher import AutoSchemaMatcher, AutoMatchConfig
        from domains.nl_query.schema_index import SchemaIndex
        
        # Check if model has required attributes
        if not hasattr(model, 'encoder') or not hasattr(model, 'config'):
            logger.warning("Model missing encoder or config, cannot create AutoSchemaMatcher")
            return None, None
        
        # Get or create schema index
        model_id = getattr(model, 'model_id', 'unknown')
        schema_index = SchemaIndex(model_id=model_id)
        
        # Build schema index from model
        try:
            schema_index.build_from_model(model)
            logger.info(f"Schema index built for model '{model_id}' with {schema_index.get_metrics().vector_count} vectors")
        except Exception as e:
            logger.warning(f"Failed to build schema index: {e}")
            schema_index = None
        
        # Create AutoSchemaMatcher
        auto_config = AutoMatchConfig(
            enable_compound_matching=True,
            enable_synonyms=True,
            fallback_to_manual=True,
        )
        
        # Get manual patterns from model if available
        manual_patterns = None
        if hasattr(model, 'nl_encoder_config'):
            manual_patterns = model.nl_encoder_config
        
        auto_schema_matcher = AutoSchemaMatcher(
            encoder=model.encoder,
            config=model.config,
            auto_config=auto_config,
            manual_patterns=manual_patterns,
        )
        
        logger.info(f"AutoSchemaMatcher created for model '{model_id}'")
        return auto_schema_matcher, schema_index
        
    except ImportError as e:
        logger.warning(f"AutoSchemaMatcher not available: {e}")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to create AutoSchemaMatcher: {e}")
        return None, None


@router.post("/query", response_model=NLQueryResponse)
async def execute_nl_query(
    org_id: str,
    model_id: str,
    request: NLQueryRequest,
) -> NLQueryResponse:
    """
    Execute a natural language query.
    
    Supports auto-schema matching for automatic intent inference and
    parameter extraction. Returns disambiguation suggestions when
    the query is ambiguous.
    
    Validates: Requirements 3, 4, 5, 12.2, 12.5
    """
    service = get_nl_query_service()
    
    logger.info(f"NL query: org={org_id}, model={model_id}, query='{request.query}'")
    
    result = await service.execute_nl_query(
        org_id=org_id,
        model_id=model_id,
        query=request.query,
        debug=request.debug,
    )
    
    # Handle disambiguation response
    # Validates: Requirements 12.2, 12.5
    if result.disambiguation_needed:
        return NLQueryResponse(
            result=None,
            query_type=result.query_type,
            match_method=result.match_method,
            confidence=result.confidence,
            translated_query=result.translated_query,
            query_time_ms=result.query_time_ms,
            disambiguation_needed=True,
            disambiguation_suggestions=result.disambiguation_suggestions,
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
        disambiguation_needed=result.disambiguation_needed,
        disambiguation_suggestions=result.disambiguation_suggestions,
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
    """
    Debug query translation without execution.
    
    Returns detailed information about how the query was matched,
    including disambiguation options if the query is ambiguous.
    
    Validates: Requirements 3, 4, 5, 12.2, 12.5
    """
    service = get_nl_query_service()
    
    match_result, match_method = await service.translate_query(request.query)
    
    # Check for disambiguation in the structured query
    disambiguation_needed = False
    disambiguation_suggestions = []
    
    if match_result and match_result.structured_query:
        disambiguation_needed = match_result.structured_query.get("disambiguation_needed", False)
        if disambiguation_needed:
            # Build suggestions from disambiguation options
            options = match_result.structured_query.get("disambiguation_options", [])
            for opt in options:
                intent_type = opt.get("intent_type", "unknown")
                confidence = opt.get("confidence", 0.0)
                disambiguation_suggestions.append(
                    f"Did you mean '{intent_type}'? (confidence: {confidence:.0%})"
                )
    
    if match_result:
        return TranslationDebugResponse(
            intent=match_result.intent,
            confidence=match_result.confidence,
            parameters=match_result.parameters,
            pattern_matched=match_result.pattern_matched,
            match_method=match_method,
            disambiguation_needed=disambiguation_needed,
            disambiguation_suggestions=disambiguation_suggestions,
        )
    else:
        return TranslationDebugResponse(
            intent=None, confidence=0.0, parameters={},
            pattern_matched=None, match_method="none",
            disambiguation_needed=False,
            disambiguation_suggestions=[],
        )
