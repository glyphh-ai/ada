"""
Org-Scoped API Routes for Glyphh Runtime (Cloud Mode).

Endpoints scoped by org_id and model_id for multi-tenant cloud deployments.
URL pattern: /{org_id}/{model_id}/...

No namespace concept — org_id and model_id are passed directly to services.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from domains.auth.service import AuthService, User
from domains.mcp.server import MCPServer
from domains.query.service import QueryService
from infrastructure.config import get_settings
from shared.auth import AuthenticatedUser, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}", tags=["org-scoped"])
settings = get_settings()


# Request/Response Models
class NLQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query", min_length=1)
    debug: bool = Field(default=False, description="Include translation details")


class NLQueryResponse(BaseModel):
    state: str
    result: Optional[Any] = None
    fact_tree: Optional[Dict[str, Any]] = None
    query_type: str = ""
    match_method: str = ""
    confidence: float = 0.0
    trace_id: Optional[str] = None
    translated_query: Optional[Dict[str, Any]] = None
    query_time_ms: float = 0.0
    matched_route: Optional[str] = None
    ask: Optional[Dict[str, Any]] = None
    blocked: Optional[Dict[str, Any]] = None
    auth_required: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class NormalizeRequest(BaseModel):
    query: str = Field(..., description="Query text to normalize", min_length=1)


class NormalizeResponse(BaseModel):
    normalized_query: str
    changed: bool


class IntentsResponse(BaseModel):
    intents: List[str]
    patterns: Dict[str, List[str]]


# Dependency injection
async def get_auth_service() -> AuthService:
    return AuthService()


async def validate_org_access(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """
    Validate that the authenticated user has access to the org and model.
    
    Uses JWT authentication from shared/auth.py which handles:
    - Local mode bypass (returns mock user)
    - JWT token validation and claim extraction
    - Token expiry and signature verification
    
    Additionally validates that the JWT org_id matches the URL org_id.
    """
    # In local mode, the auth module returns a mock user - allow access
    if current_user.org_id == "local-dev-org":
        return current_user
    
    # Validate org_id matches the authenticated user's org
    if current_user.org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail="Organization mismatch - you don't have access to this organization"
        )
    
    return current_user


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
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """
    MCP endpoint for org-scoped model access.
    
    Accepts JWT authentication from Studio for direct queries.
    Validates org_id in URL matches org_id in JWT token.
    """
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
    current_user: AuthenticatedUser = Depends(validate_org_access),
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
    
    # Serialize via to_dict() which handles all states
    response_data = result.to_dict()
    
    # Backward compat: include result field for DONE state
    if result.fact_tree is not None:
        response_data["result"] = result.fact_tree.to_json()
    
    return NLQueryResponse(**response_data)


@router.get("/intents", response_model=IntentsResponse)
async def get_intents(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> IntentsResponse:
    """Get available intents and patterns for the model."""
    service = get_nl_query_service_for_org()
    intents = service.get_intents()
    
    return IntentsResponse(
        intents=intents["intents"],
        patterns=intents["patterns"],
    )

# Normalize Endpoint
def _get_llm_fallback():
    """Get LLM fallback instance for normalization. Returns None if unavailable."""
    try:
        from domains.nl_query.llm_fallback import LLMFallback
        llm_fallback = LLMFallback(model_name=settings.nl_model)
        if not llm_fallback.is_available():
            return None
        return llm_fallback
    except ImportError:
        return None


@router.post("/normalize")
async def normalize_query(
    org_id: str,
    model_id: str,
    request: NormalizeRequest,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> NormalizeResponse:
    """Normalize query text: fix spelling and grammar via LLM."""
    llm_fallback = _get_llm_fallback()

    if llm_fallback is None:
        return NormalizeResponse(normalized_query=request.query, changed=False)

    normalized_text, changed = await llm_fallback.normalize_text(request.query)
    return NormalizeResponse(normalized_query=normalized_text, changed=changed)

