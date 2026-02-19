"""
Listener API Routes for Glyphh Runtime.

HTTP endpoint for async glyph ingestion.
Route: POST /{org_id}/{model_id}/listener
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from domains.jobs.manager import get_job_manager
from domains.listeners.async_service import AsyncListenerService
from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/listener", tags=["listeners"])
settings = get_settings()

_async_listener_service: Optional[AsyncListenerService] = None


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
class AsyncDataLoadRequest(BaseModel):
    """Request for async data loading."""
    records: List[Dict[str, Any]] = Field(..., description="Records to load (each with concept/text field)")
    batch_size: Optional[int] = Field(50, description="Records per batch")


class AsyncDataLoadResponse(BaseModel):
    """Response from async data load endpoint."""
    job_id: UUID
    status: str
    total_records: int


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
