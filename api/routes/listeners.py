"""
Listener API Routes for Glyphh Runtime.

WebSocket and HTTP endpoints for real-time glyph ingestion.
All routes scoped by /{org_id}/{model_id}/listener/...
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket
from pydantic import BaseModel, Field

from domains.listeners.service import ListenerService
from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/listener", tags=["listeners"])
settings = get_settings()

_listener_service: Optional[ListenerService] = None


def get_listener_service() -> ListenerService:
    global _listener_service
    if _listener_service is None:
        from main import model_manager
        from infrastructure.database import async_session_maker
        
        async def get_encoder(org_id: str, model_id: str):
            if model_manager is None:
                raise ValueError("Model manager not initialized")
            model = await model_manager.get_model(org_id, model_id)
            if model is None:
                raise ValueError(f"Model not found: org={org_id}, model={model_id}")
            return model.encoder
        
        _listener_service = ListenerService(
            session_maker=async_session_maker,
            encoder_getter=get_encoder,
        )
    return _listener_service


# Request/Response Models
class BatchCreateRequest(BaseModel):
    concepts: List[str] = Field(..., description="List of concept texts to encode")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Shared metadata for all glyphs")


class BatchCreateResponse(BaseModel):
    total: int
    created: int
    failed: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]


class ConnectionStatsResponse(BaseModel):
    active_connections: int
    connections: List[Dict[str, Any]]


@router.websocket("")
async def websocket_listener(
    websocket: WebSocket,
    org_id: str,
    model_id: str,
):
    """WebSocket endpoint for streaming glyph creation."""
    service = get_listener_service()
    await service.handle_websocket(websocket, org_id, model_id)


@router.post("", response_model=BatchCreateResponse)
async def batch_create(
    org_id: str,
    model_id: str,
    request: BatchCreateRequest,
) -> BatchCreateResponse:
    """HTTP endpoint for batch glyph creation."""
    service = get_listener_service()
    result = await service.handle_batch_create(
        org_id=org_id,
        model_id=model_id,
        concepts=request.concepts,
        metadata=request.metadata,
    )
    
    return BatchCreateResponse(
        total=result.total,
        created=result.created,
        failed=result.failed,
        results=result.results,
        errors=result.errors,
    )


@router.get("/stats", response_model=ConnectionStatsResponse)
async def get_connection_stats(org_id: str, model_id: str) -> ConnectionStatsResponse:
    """Get WebSocket connection statistics."""
    service = get_listener_service()
    stats = service.get_connection_stats()
    
    model_connections = [
        c for c in stats["connections"]
        if c.get("org_id") == org_id and c.get("model_id") == model_id
    ]
    
    return ConnectionStatsResponse(
        active_connections=len(model_connections),
        connections=model_connections,
    )
