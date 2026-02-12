"""
Chat Streaming API Routes.

SSE streaming endpoint for chat queries with progress events.
Publicly accessible for any client (Studio, custom UIs, integrations).

Updated to support AutoSchemaMatcher for automatic schema-based NL query matching.
Validates: Requirements 3, 4, 5, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 12.2, 12.5
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/chat", tags=["chat"])
settings = get_settings()


class ChatStreamRequest(BaseModel):
    """Request for streaming chat."""
    query: str = Field(..., description="Natural language query", min_length=1)
    session_id: Optional[str] = Field(None, description="Optional session ID for context")


class ChatResponse(BaseModel):
    """Response from synchronous chat endpoint.
    
    Returns both raw result (for programmatic use) and formatted response (for display).
    This aligns with MCP response format for consistency.
    """
    response: str  # Formatted response for display
    result: Optional[Any] = None  # Raw result data, same as MCP
    source: str  # "glyphh", "llm", or "auto"
    confidence: float
    query_type: Optional[str] = None
    match_method: Optional[str] = None
    session_id: Optional[str] = None
    disambiguation_needed: bool = False
    disambiguation_suggestions: List[str] = Field(default_factory=list)
    query_time_ms: Optional[float] = None


def get_chat_services():
    """
    Get services needed for chat.
    
    Initializes the NL Query Service with AutoSchemaMatcher support
    for automatic schema-based matching.
    
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
    
    # Get NL config from model if available
    model_nl_config = None
    auto_schema_matcher = None
    schema_index = None
    
    try:
        model = model_manager.get_current_model() if hasattr(model_manager, 'get_current_model') else None
        if model is not None:
            model_nl_config = EncoderConfigFactory.extract_nl_encoder_config(model)
            
            # Try to get or create AutoSchemaMatcher for the model
            # Validates: Requirements 3, 4, 5
            auto_schema_matcher, schema_index = _get_auto_schema_matcher_for_chat(model)
            
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
    
    nl_service = NLQueryService(
        query_service=query_service,
        intent_matcher=intent_matcher,
        llm_fallback=llm_fallback,
        confidence_threshold=0.85,
        schema_index=schema_index,
        auto_schema_matcher=auto_schema_matcher,
    )
    
    return nl_service, query_service


def _get_auto_schema_matcher_for_chat(model):
    """
    Get or create AutoSchemaMatcher for chat.
    
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
            logger.info(f"Schema index built for chat model '{model_id}' with {schema_index.get_metrics().vector_count} vectors")
        except Exception as e:
            logger.warning(f"Failed to build schema index for chat: {e}")
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
        
        logger.info(f"AutoSchemaMatcher created for chat model '{model_id}'")
        return auto_schema_matcher, schema_index
        
    except ImportError as e:
        logger.warning(f"AutoSchemaMatcher not available for chat: {e}")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to create AutoSchemaMatcher for chat: {e}")
        return None, None


@router.post("/stream")
async def chat_stream(
    org_id: str,
    model_id: str,
    request: ChatStreamRequest,
) -> StreamingResponse:
    """
    SSE streaming endpoint for chat queries.
    
    Streams progress events during query processing:
    - thinking: Query is being analyzed/matched
    - searching: Similarity search in progress
    - generating: Response being formatted
    - disambiguation: Query is ambiguous, suggestions provided
    - complete: Final response with results
    
    For fast queries (<500ms), may skip intermediate events.
    
    Validates: Requirements 3, 4, 5, 11.1, 11.2, 11.3, 11.5, 12.2, 12.5
    """
    async def stream_response():
        start_time = time.time()
        
        try:
            nl_service, query_service = get_chat_services()
            
            # Phase 1: Thinking
            yield _sse_event("thinking", {
                "message": "Analyzing query...",
                "progress": 10,
            })
            
            # Match intent
            match_result, match_method = await nl_service.translate_query(request.query)
            
            elapsed = time.time() - start_time
            
            # For fast queries, we might skip some events
            if elapsed < 0.5 and match_result:
                # Fast path - skip to searching
                yield _sse_event("thinking", {
                    "message": f"Matched intent: {match_result.intent}",
                    "progress": 30,
                    "confidence": match_result.confidence,
                    "match_method": match_method,
                })
            else:
                yield _sse_event("thinking", {
                    "message": f"Matched intent: {match_result.intent if match_result else 'none'}",
                    "progress": 30,
                    "confidence": match_result.confidence if match_result else 0.0,
                    "match_method": match_method,
                })
            
            if not match_result or match_method == "none":
                # No match - return error
                yield _sse_event("complete", {
                    "message": "Could not understand query",
                    "response": "I couldn't understand your query. Try rephrasing or use keywords like 'find', 'similar', 'list'.",
                    "source": "error",
                    "confidence": 0.0,
                    "progress": 100,
                })
                return
            
            # Check for disambiguation
            # Validates: Requirements 12.2, 12.5
            disambiguation_needed = False
            disambiguation_suggestions = []
            
            if match_result.structured_query:
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
            
            if disambiguation_needed:
                # Return disambiguation response
                yield _sse_event("disambiguation", {
                    "message": "Your query is ambiguous. Please clarify:",
                    "suggestions": disambiguation_suggestions,
                    "confidence": match_result.confidence,
                    "progress": 100,
                })
                return
            
            # Phase 2: Searching
            yield _sse_event("searching", {
                "message": "Searching knowledge base...",
                "progress": 50,
            })
            
            # Execute the query
            result = await nl_service.execute_nl_query(
                org_id=org_id,
                model_id=model_id,
                query=request.query,
                debug=False,
            )
            
            # Check if result indicates disambiguation needed
            if result.disambiguation_needed:
                yield _sse_event("disambiguation", {
                    "message": "Your query is ambiguous. Please clarify:",
                    "suggestions": result.disambiguation_suggestions,
                    "confidence": result.confidence,
                    "progress": 100,
                })
                return
            
            # Check if result contains an error (e.g., encoding failure)
            if isinstance(result.result, dict) and "error" in result.result:
                error_result = result.result
                yield _sse_event("complete", {
                    "message": error_result.get("message", "Query could not be processed"),
                    "result": error_result,
                    "source": "error",
                    "confidence": 0.0,
                    "query_type": result.query_type,
                    "match_method": result.match_method,
                    "progress": 100,
                    "query_time_ms": result.query_time_ms,
                })
                return
            
            yield _sse_event("searching", {
                "message": "Search complete",
                "progress": 70,
            })
            
            # Phase 3: Generating response
            yield _sse_event("generating", {
                "message": "Formatting response...",
                "progress": 85,
            })
            
            # Determine source based on match method
            if result.match_method in ("auto", "hybrid"):
                source = "auto"
            elif result.match_method == "llm":
                source = "llm"
            else:
                source = "glyphh"
            
            # Phase 4: Complete - return raw result like MCP does
            # The Studio formats the result on the client side
            yield _sse_event("complete", {
                "message": "Query complete",
                "result": result.result,  # Raw result, same as MCP response
                "source": source,
                "confidence": result.confidence,
                "query_type": result.query_type,
                "match_method": result.match_method,
                "progress": 100,
                "query_time_ms": result.query_time_ms,
            })
            
        except HTTPException as e:
            yield _sse_event("error", {
                "message": str(e.detail),
                "progress": 0,
            })
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield _sse_event("error", {
                "message": f"An error occurred: {str(e)}",
                "progress": 0,
            })
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("", response_model=ChatResponse)
async def chat_sync(
    org_id: str,
    model_id: str,
    request: ChatStreamRequest,
) -> ChatResponse:
    """
    Synchronous chat endpoint for clients that don't need streaming.
    
    Supports auto-schema matching and returns disambiguation suggestions
    when the query is ambiguous.
    
    Validates: Requirements 3, 4, 5, 11.6, 12.2, 12.5
    """
    nl_service, _ = get_chat_services()
    
    result = await nl_service.execute_nl_query(
        org_id=org_id,
        model_id=model_id,
        query=request.query,
        debug=False,
    )
    
    # Handle disambiguation
    # Validates: Requirements 12.2, 12.5
    if result.disambiguation_needed:
        return ChatResponse(
            response="Your query is ambiguous. Please clarify your intent.",
            source="disambiguation",
            confidence=result.confidence,
            query_type=result.query_type,
            match_method=result.match_method,
            session_id=request.session_id,
            disambiguation_needed=True,
            disambiguation_suggestions=result.disambiguation_suggestions,
        )
    
    if result.match_method == "none":
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Could not understand query",
                "confidence": result.confidence,
            }
        )
    
    response_text = _format_result(result.result, result.query_type)
    
    # Determine source based on match method
    if result.match_method in ("auto", "hybrid"):
        source = "auto"
    elif result.match_method == "llm":
        source = "llm"
    else:
        source = "glyphh"
    
    return ChatResponse(
        response=response_text,
        result=result.result,  # Raw result, same as MCP
        source=source,
        confidence=result.confidence,
        query_type=result.query_type,
        match_method=result.match_method,
        session_id=request.session_id,
        disambiguation_needed=False,
        disambiguation_suggestions=[],
        query_time_ms=result.query_time_ms,
    )


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format an SSE event with UUID serialization support."""
    data["type"] = event_type
    data["timestamp"] = datetime.utcnow().isoformat()
    return f"event: {event_type}\ndata: {json.dumps(data, default=_json_serializer)}\n\n"


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _format_result(result: Any, query_type: str) -> str:
    """Format query result for display (used by sync endpoint)."""
    if result is None:
        return "No results found."
    
    if isinstance(result, str):
        return result
    
    if isinstance(result, dict):
        # Handle count results FIRST (before checking for "results" key)
        if "count" in result and "results" not in result:
            return f"Count: {result['count']}"
        
        # Handle list/similarity search results
        if "results" in result and isinstance(result["results"], list):
            results = result["results"]
            if not results:
                return "No matching results found."
            
            # Check if this is a list (no scores) or similarity search
            is_list = len(results) > 0 and "similarity_score" not in results[0] and "score" not in results[0]
            
            if is_list:
                # Format as clean list
                total = result.get("total_count", len(results))
                lines = [f"Found {total} record{'s' if total != 1 else ''}:\n"]
                
                for i, r in enumerate(results[:10], 1):
                    # Try to get meaningful display text
                    glyph = r.get("glyph", r)
                    concept = glyph.get("concept_text", "")
                    metadata = glyph.get("metadata", {})
                    
                    if concept:
                        lines.append(f"{i}. {concept}")
                    elif metadata:
                        # Format key metadata fields
                        highlights = _format_metadata_highlights(metadata)
                        lines.append(f"{i}. {highlights}")
                    else:
                        lines.append(f"{i}. (Record {i})")
                
                if len(results) > 10:
                    lines.append(f"\n... and {len(results) - 10} more")
                
                return "\n".join(lines)
            
            # Similarity search format
            lines = []
            for i, r in enumerate(results[:5], 1):
                concept = r.get("concept_text") or r.get("concept") or ""
                score = r.get("similarity_score") or r.get("score") or 0
                lines.append(f"{i}. {concept} ({score*100:.0f}% match)")
            
            return "\n".join(lines)
        
        # Handle verification results
        if "verified" in result:
            verified = result["verified"]
            confidence = result.get("confidence", 0)
            return f"{'✓ Verified' if verified else '✗ Not verified'} (confidence: {confidence*100:.0f}%)"
        
        # Handle prediction results
        if "predictions" in result:
            predictions = result["predictions"]
            if not predictions:
                return "No predictions available."
            
            lines = []
            for p in predictions[:5]:
                concept = p.get("concept") or p.get("text") or ""
                prob = p.get("probability") or p.get("score") or 0
                lines.append(f"• {concept} ({prob*100:.0f}%)")
            
            return "\n".join(lines)
    
    # Fallback to JSON
    return json.dumps(result, indent=2, default=_json_serializer)


def _format_metadata_highlights(metadata: Dict[str, Any]) -> str:
    """Format metadata into readable highlights."""
    priority_keys = ["make", "model", "year", "vin", "name", "title", "type", "service_type", "service_date"]
    highlights = []
    
    # Add priority keys first
    for key in priority_keys:
        if key in metadata and metadata[key] is not None:
            value = metadata[key]
            if isinstance(value, (str, int, float)):
                highlights.append(str(value))
        if len(highlights) >= 4:
            break
    
    # If not enough, add other keys
    if len(highlights) < 3:
        for key, value in metadata.items():
            if key in priority_keys:
                continue
            if value is None or isinstance(value, (dict, list)):
                continue
            highlights.append(f"{key}: {value}")
            if len(highlights) >= 4:
                break
    
    return " | ".join(highlights) if highlights else "(no details)"
