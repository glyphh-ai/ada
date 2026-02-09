"""Job API Routes for Runtime.

SSE endpoints for job progress events and job status queries.

Requirements: 8.1, 8.2, 8.3, 8.4, 12.1, 12.2
"""

import asyncio
import json
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from domains.jobs.manager import DataLoadStatus, JobManager, get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/{org_id}/{model_id}/jobs", tags=["jobs"])


class JobStatusResponse(BaseModel):
    """Response for job status endpoint."""
    id: str
    org_id: str
    model_id: str
    status: str
    total_records: int
    processed_count: int
    encoded_count: int
    failed_count: int
    skipped_count: int
    progress: int
    error_message: Optional[str]
    created_at: str
    updated_at: str


def get_manager() -> JobManager:
    """Dependency to get JobManager."""
    return get_job_manager()


@router.get("/{job_id}/events")
async def job_events_stream(
    org_id: str,
    model_id: str,
    job_id: UUID,
    job_manager: JobManager = Depends(get_manager),
) -> StreamingResponse:
    """
    SSE endpoint for data load progress events.
    
    Streams events by polling in-memory job state.
    Sends keepalive every 30 seconds.
    Closes on complete or error.
    
    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify job belongs to this org/model
    if job.org_id != org_id or job.model_id != model_id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        """Generate SSE events from job manager."""
        queue = job_manager.subscribe(job_id)
        
        try:
            # Send current state first (Requirement 8.2)
            current_job = job_manager.get_job(job_id)
            if current_job:
                initial_event = {
                    "type": current_job.status.value,
                    "job_id": str(job_id),
                    "progress": current_job.progress,
                    "processed": current_job.processed_count,
                    "encoded": current_job.encoded_count,
                    "failed": current_job.failed_count,
                    "skipped": current_job.skipped_count,
                    "total": current_job.total_records,
                    "message": None,
                    "timestamp": current_job.updated_at.isoformat(),
                }
                yield f"data: {json.dumps(initial_event)}\n\n"
                
                # If already complete/error, close stream (Requirement 8.4)
                if current_job.status in (DataLoadStatus.COMPLETE, DataLoadStatus.ERROR):
                    return
            
            # Stream new events
            while True:
                try:
                    # Wait for event with timeout for keepalive
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                    
                    # Close on terminal events (Requirement 8.4)
                    if event.get("type") in ("complete", "error"):
                        break
                        
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    
                    # Check if job still exists and is terminal
                    current_job = job_manager.get_job(job_id)
                    if not current_job:
                        break
                    if current_job.status in (DataLoadStatus.COMPLETE, DataLoadStatus.ERROR):
                        break
                        
        except Exception as e:
            logger.error(f"SSE stream error for job {job_id}: {e}")
            error_event = {
                "type": "error",
                "message": f"Stream error: {str(e)}",
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            job_manager.unsubscribe(job_id, queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    org_id: str,
    model_id: str,
    job_id: UUID,
    job_manager: JobManager = Depends(get_manager),
) -> JobStatusResponse:
    """
    Get job status.
    
    Requirement: 12.1
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.org_id != org_id or job.model_id != model_id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        id=str(job.id),
        org_id=job.org_id,
        model_id=job.model_id,
        status=job.status.value,
        total_records=job.total_records,
        processed_count=job.processed_count,
        encoded_count=job.encoded_count,
        failed_count=job.failed_count,
        skipped_count=job.skipped_count,
        progress=job.progress,
        error_message=job.error_message,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
    )


@router.get("", response_model=list[JobStatusResponse])
async def list_jobs(
    org_id: str,
    model_id: str,
    limit: int = 10,
    job_manager: JobManager = Depends(get_manager),
) -> list[JobStatusResponse]:
    """
    List recent jobs for a model.
    
    Requirement: 12.2
    """
    jobs = job_manager.list_jobs(org_id, model_id, limit)
    
    return [
        JobStatusResponse(
            id=str(job.id),
            org_id=job.org_id,
            model_id=job.model_id,
            status=job.status.value,
            total_records=job.total_records,
            processed_count=job.processed_count,
            encoded_count=job.encoded_count,
            failed_count=job.failed_count,
            skipped_count=job.skipped_count,
            progress=job.progress,
            error_message=job.error_message,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
        )
        for job in jobs
    ]
