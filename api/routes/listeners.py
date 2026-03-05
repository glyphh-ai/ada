"""
Listener API Routes for Glyphh Runtime.

HTTP endpoint for async glyph ingestion.
Route: POST /{org_id}/{model_id}/listener
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from domains.jobs.manager import get_job_manager
from domains.listeners.async_service import AsyncListenerService
from infrastructure.config import get_settings
from shared.auth import AuthenticatedUser, get_current_user, require_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}/listener", tags=["listeners"])
settings = get_settings()


async def validate_listener_access(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """Validate that the user has access to load data into this org/model.

    Uses get_current_user (supports local mode bypass) instead of require_token
    so the web dashboard dropzone can load data without a token in local mode.
    """
    from fastapi import HTTPException

    if current_user.org_id == "local-dev-org":
        return current_user

    if current_user.org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail="Organization mismatch — you don't have access to this organization",
        )

    return current_user

_async_listener_service: Optional[AsyncListenerService] = None


def get_async_listener_service() -> AsyncListenerService:
    """Get or create AsyncListenerService for async data loading."""
    global _async_listener_service
    if _async_listener_service is None:
        from glyphh.server import model_manager
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
    current_user: AuthenticatedUser = Depends(validate_listener_access),
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
        plan_slug=current_user.plan,
    )

    return AsyncDataLoadResponse(
        job_id=job_id,
        status="queued",
        total_records=len(request.records),
    )


@router.get("/jobs/{job_id}")
async def get_job_status(
    org_id: str,
    model_id: str,
    job_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get job progress. Polled by the dashboard dropzone."""
    from fastapi import HTTPException

    manager = get_job_manager()
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": str(job.id),
        "status": "completed" if job.status.value == "complete" else job.status.value,
        "processed": job.processed_count,
        "encoded": job.encoded_count,
        "failed": job.failed_count,
        "total": job.total_records,
        "progress": job.progress,
        "error": job.error_message,
    }
