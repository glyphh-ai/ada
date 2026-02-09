"""
Listener API Routes for Glyphh Runtime.

WebSocket and HTTP endpoints for real-time glyph ingestion.
All routes scoped by /{org_id}/{model_id}/listener/...

Updated to support async data loading with job tracking (Requirements 7.2).
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, WebSocket
from pydantic import BaseModel, Field

from domains.jobs.manager import get_job_manager
from domains.listeners.async_service import AsyncListenerService
from domains.listeners.service import ListenerService
from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/listener", tags=["listeners"])
settings = get_settings()

_listener_service: Optional[ListenerService] = None
_async_listener_service: Optional[AsyncListenerService] = None


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


def get_async_listener_service() -> AsyncListenerService:
    """Get or create AsyncListenerService for async data loading."""
    global _async_listener_service
    if _async_listener_service is None:
        from main import model_manager
        from infrastructure.database import async_session_maker
        
        async def get_encoder(org_id: str, model_id: str):
            if model_manager is None:
                raise ValueError("Model manager not initialized")
            model = await model_manager.get_model(org_id, model_id)
            if model is None:
                raise ValueError(f"Model not found: org={org_id}, model={model_id}")
            return model.encoder
        
        _async_listener_service = AsyncListenerService(
            session_maker=async_session_maker,
            encoder_getter=get_encoder,
            job_manager=get_job_manager(),
        )
    return _async_listener_service


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


class AsyncDataLoadRequest(BaseModel):
    """Request for async data loading."""
    records: List[Dict[str, Any]] = Field(..., description="Records to load (each with concept/text field)")
    batch_size: Optional[int] = Field(50, description="Records per batch")


class AsyncDataLoadResponse(BaseModel):
    """Response from async data load endpoint."""
    job_id: UUID
    status: str
    total_records: int


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


@router.post("", response_model=AsyncDataLoadResponse)
async def async_data_load(
    org_id: str,
    model_id: str,
    request: AsyncDataLoadRequest,
) -> AsyncDataLoadResponse:
    """
    Start async data load. Returns immediately with job_id.
    
    Processing happens in background. Use GET /jobs/{job_id}/events
    to stream progress events.
    
    Requirement: 7.2
    """
    service = get_async_listener_service()
    
    job_id = await service.start_load(
        org_id=org_id,
        model_id=model_id,
        records=request.records,
        batch_size=request.batch_size or 50,
    )
    
    return AsyncDataLoadResponse(
        job_id=job_id,
        status="queued",
        total_records=len(request.records),
    )


@router.post("/batch", response_model=BatchCreateResponse)
async def batch_create(
    org_id: str,
    model_id: str,
    request: BatchCreateRequest,
) -> BatchCreateResponse:
    """
    Synchronous batch glyph creation (legacy endpoint).
    
    For large datasets, use POST /listener with records array
    for async processing with progress tracking.
    """
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
