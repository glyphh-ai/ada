"""
Chat Streaming API Routes.

SSE streaming endpoint for chat queries with progress events.
Publicly accessible for any client (Studio, custom UIs, integrations).

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

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
    """Response from synchronous chat endpoint."""
    response: str
    source: str  # "glyphh" or "llm"
    confidence: float
    query_type: Optional[str] = None
    match_method: Optional[str] = None
    session_id: Optional[str] = None


def get_chat_services():
    """Get services needed for chat."""
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
    try:
        model = model_manager.get_current_model() if hasattr(model_manager, 'get_current_model') else None
        if model is not None:
            model_nl_config = EncoderConfigFactory.extract_nl_encoder_config(model)
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
    )
    
    return nl_service, query_service


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
    - complete: Final response with results
    
    For fast queries (<500ms), may skip intermediate events.
    
    Requirements: 11.1, 11.2, 11.3, 11.5
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
                })
            else:
                yield _sse_event("thinking", {
                    "message": f"Matched intent: {match_result.intent if match_result else 'none'}",
                    "progress": 30,
                    "confidence": match_result.confidence if match_result else 0.0,
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
            
            yield _sse_event("searching", {
                "message": "Search complete",
                "progress": 70,
            })
            
            # Phase 3: Generating response
            yield _sse_event("generating", {
                "message": "Formatting response...",
                "progress": 85,
            })
            
            # Format the response
            response_text = _format_result(result.result, result.query_type)
            
            # Phase 4: Complete
            yield _sse_event("complete", {
                "message": "Query complete",
                "response": response_text,
                "source": "glyphh" if result.match_method != "llm" else "llm",
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
    
    Requirement: 11.6
    """
    nl_service, _ = get_chat_services()
    
    result = await nl_service.execute_nl_query(
        org_id=org_id,
        model_id=model_id,
        query=request.query,
        debug=False,
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
    
    return ChatResponse(
        response=response_text,
        source="glyphh" if result.match_method != "llm" else "llm",
        confidence=result.confidence,
        query_type=result.query_type,
        match_method=result.match_method,
        session_id=request.session_id,
    )


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format an SSE event."""
    data["type"] = event_type
    data["timestamp"] = datetime.utcnow().isoformat()
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _format_result(result: Any, query_type: str) -> str:
    """Format query result for display."""
    if result is None:
        return "No results found."
    
    if isinstance(result, str):
        return result
    
    if isinstance(result, dict):
        # Handle similarity search results
        if "results" in result and isinstance(result["results"], list):
            results = result["results"]
            if not results:
                return "No matching results found."
            
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
        
        # Handle count results
        if "count" in result:
            return f"Count: {result['count']}"
    
    # Fallback to JSON
    return json.dumps(result, indent=2)
