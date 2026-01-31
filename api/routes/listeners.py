"""
Listener API Routes for Glyphh Runtime.

WebSocket and HTTP endpoints for real-time glyph ingestion.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket
from pydantic import BaseModel, Field

from domains.listeners.service import ListenerService
from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/{namespace}/listener", tags=["listeners"])
settings = get_settings()

# Global listener service instance
_listener_service: Optional[ListenerService] = None


def get_listener_service() -> ListenerService:
    """Get the listener service instance."""
    global _listener_service
    if _listener_service is None:
        from main import model_manager
        from infrastructure.database import async_session_maker
        
        async def get_encoder(namespace: str):
            if model_manager is None:
                raise ValueError("Model manager not initialized")
            model = await model_manager.get_model(namespace)
            if model is None:
                raise ValueError(f"Model not found: {namespace}")
            return model.encoder
        
        _listener_service = ListenerService(
            session_maker=async_session_maker,
            encoder_getter=get_encoder,
        )
    return _listener_service


# Request/Response Models
class BatchCreateRequest(BaseModel):
    """Batch glyph creation request."""
    concepts: List[str] = Field(..., description="List of concept texts to encode")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Shared metadata for all glyphs")


class BatchCreateResponse(BaseModel):
    """Batch glyph creation response."""
    total: int
    created: int
    failed: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]


class ConnectionStatsResponse(BaseModel):
    """Connection statistics response."""
    active_connections: int
    connections: List[Dict[str, Any]]


# WebSocket Endpoint
@router.websocket("")
async def websocket_listener(
    websocket: WebSocket,
    namespace: str,
):
    """
    WebSocket endpoint for streaming glyph creation.
    
    Protocol:
    - Send: {"type": "create_glyph", "concept": "...", "metadata": {...}}
    - Receive: {"type": "glyph_created", "glyph_id": "...", "status": "success"}
    - Send: {"type": "batch_create", "concepts": [...], "metadata": {...}}
    - Receive: {"type": "batch_created", "total": N, "created": M, ...}
    - Send: {"type": "ping"}
    - Receive: {"type": "pong"}
    """
    service = get_listener_service()
    await service.handle_websocket(websocket, namespace)


# HTTP Batch Endpoint
@router.post("", response_model=BatchCreateResponse)
async def batch_create(
    namespace: str,
    request: BatchCreateRequest,
) -> BatchCreateResponse:
    """
    HTTP endpoint for batch glyph creation.
    
    Encodes all concepts and stores them. Returns results for each concept.
    """
    service = get_listener_service()
    result = await service.handle_batch_create(
        namespace=namespace,
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


# Stats Endpoint
@router.get("/stats", response_model=ConnectionStatsResponse)
async def get_connection_stats(namespace: str) -> ConnectionStatsResponse:
    """
    Get WebSocket connection statistics.
    
    Returns information about active connections for the namespace.
    """
    service = get_listener_service()
    stats = service.get_connection_stats()
    
    # Filter to namespace
    namespace_connections = [
        c for c in stats["connections"]
        if c["namespace"] == namespace
    ]
    
    return ConnectionStatsResponse(
        active_connections=len(namespace_connections),
        connections=namespace_connections,
    )
