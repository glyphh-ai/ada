"""
Org-Scoped API Routes for Glyphh Runtime (Cloud Mode).

Endpoints scoped by org_id and model_id for multi-tenant cloud deployments.
URL pattern: /{org_id}/{model_id}/...

No namespace concept — org_id and model_id are passed directly to services.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from domains.auth.service import AuthService, User
from domains.mcp.server import MCPServer
from domains.models.storage import GlyphStorage
from domains.query.service import QueryService
from infrastructure.config import get_settings
from infrastructure.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}", tags=["org-scoped"])
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


# Dependency injection
async def get_auth_service() -> AuthService:
    return AuthService()


async def validate_org_access(
    org_id: str,
    model_id: str,
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """Validate that the user has access to the org and model."""
    if settings.deployment_mode == "local":
        return User(
            user_id="local",
            org_permissions={"*": {"read", "write", "admin"}},
            org_id=org_id,
        )
    
    raise HTTPException(
        status_code=501,
        detail="Org-scoped authentication not yet implemented"
    )


async def get_mcp_server(
    org_id: str,
    model_id: str,
) -> MCPServer:
    """Get MCP server for org/model."""
    from main import model_manager
    from infrastructure.database import async_session_maker
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    query_service = QueryService(model_manager, async_session_maker)
    auth_service = AuthService()
    
    return MCPServer(query_service, auth_service)


# MCP Endpoint
@router.post("/mcp")
async def mcp_endpoint(
    org_id: str,
    model_id: str,
    request: Dict[str, Any],
    mcp_server: MCPServer = Depends(get_mcp_server),
    user: User = Depends(validate_org_access),
) -> Dict[str, Any]:
    """MCP endpoint for org-scoped model access."""
    tool_name = request.get("tool")
    arguments = request.get("arguments", {})
    
    if not tool_name:
        raise HTTPException(status_code=400, detail="Missing 'tool' field")
    
    # Pass org_id and model_id directly — no namespace construction
    arguments["org_id"] = org_id
    arguments["model_id"] = model_id
    
    response = await mcp_server.handle_tool_call(
        tool_name=tool_name,
        arguments=arguments,
        auth_token="",
    )
    
    return response.to_dict()


@router.get("/mcp/tools")
async def list_mcp_tools(
    org_id: str,
    model_id: str,
    mcp_server: MCPServer = Depends(get_mcp_server),
) -> Dict[str, Any]:
    return {"tools": mcp_server.get_tools_list()}


# Listener Endpoints
@router.websocket("/listener")
async def listener_websocket(
    websocket: WebSocket,
    org_id: str,
    model_id: str,
):
    """WebSocket listener for real-time glyph ingestion."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "create_glyph":
                await websocket.send_json({
                    "type": "glyph_created",
                    "status": "not_implemented",
                    "message": "Listener service not yet implemented"
                })
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for org={org_id}, model={model_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


@router.post("/listener")
async def listener_batch(
    org_id: str,
    model_id: str,
    request: Dict[str, Any],
    user: User = Depends(validate_org_access),
) -> Dict[str, Any]:
    """HTTP endpoint for batch glyph ingestion."""
    concepts = request.get("concepts", [])
    metadata = request.get("metadata", {})
    
    if not concepts:
        raise HTTPException(status_code=400, detail="No concepts provided")
    
    return {
        "status": "not_implemented",
        "message": "Batch listener not yet implemented",
        "org_id": org_id,
        "model_id": model_id,
        "concepts_received": len(concepts)
    }


# Batch Glyph Creation
class BatchGlyphRequest(BaseModel):
    concepts: List[str] = Field(..., description="List of concept texts to encode")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Shared metadata for all glyphs")


@router.post("/glyphs/batch")
async def create_glyphs_batch_org_scoped(
    org_id: str,
    model_id: str,
    request: BatchGlyphRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Create multiple glyphs in a batch for an org-scoped model.
    
    Requires the model to be loaded in the runtime — no fallback encoders.
    """
    from main import model_manager
    from domains.models.storage import GlyphStorage
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    # Model must be loaded — no fallback
    model = await model_manager.get_model(org_id, model_id)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: org={org_id}, model={model_id}. Deploy the model to the runtime first."
        )
    
    encoder = model.encoder
    storage = GlyphStorage(db)
    
    results = []
    errors = []
    
    for i, concept in enumerate(request.concepts):
        try:
            embedding = encoder.encode_text(concept)
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
            result = await storage.create_glyph(
                org_id=org_id,
                model_id=model_id,
                concept_text=concept,
                embedding=embedding_list,
                metadata=request.metadata,
            )
            results.append({
                "index": i,
                "glyph_id": str(result.glyph_id),
                "status": "created"
            })
        except Exception as e:
            logger.error(f"Failed to create glyph {i}: {e}")
            errors.append({
                "index": i,
                "concept": concept[:50] + "..." if len(concept) > 50 else concept,
                "error": str(e)
            })
    
    logger.info(f"Batch create for org={org_id}, model={model_id}: {len(results)} created, {len(errors)} failed")
    
    return {
        "created": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None,
    }


# NL Query Service Factory
def get_nl_query_service_for_org():
    """Get NL query service instance for org-scoped queries."""
    from main import model_manager
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
            logger.info("LLM fallback disabled (transformers not available)")
    except ImportError:
        logger.info("LLM fallback disabled (import error)")
    
    return NLQueryService(
        query_service=query_service,
        intent_matcher=intent_matcher,
        llm_fallback=llm_fallback,
        confidence_threshold=0.85,
    )


# NL Query Endpoints
@router.post("/query", response_model=NLQueryResponse)
async def execute_nl_query(
    org_id: str,
    model_id: str,
    request: NLQueryRequest,
    user: User = Depends(validate_org_access),
) -> NLQueryResponse:
    """Execute a natural language query against the model."""
    service = get_nl_query_service_for_org()
    
    logger.info(f"NL query: org={org_id}, model={model_id}, query='{request.query}'")
    
    start_time = time.time()
    
    result = await service.execute_nl_query(
        org_id=org_id,
        model_id=model_id,
        query=request.query,
        debug=request.debug,
    )
    
    query_time_ms = (time.time() - start_time) * 1000
    
    if result.match_method == "none":
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
        translated_query=result.translated_query if request.debug else None,
        query_time_ms=query_time_ms,
    )


@router.get("/intents", response_model=IntentsResponse)
async def get_intents(
    org_id: str,
    model_id: str,
    user: User = Depends(validate_org_access),
) -> IntentsResponse:
    """Get available intents and patterns for the model."""
    service = get_nl_query_service_for_org()
    intents = service.get_intents()
    
    return IntentsResponse(
        intents=intents["intents"],
        patterns=intents["patterns"],
    )
