"""In-Memory Job Manager for Runtime data load operations.

Unlike Platform's Redis-based queue, Runtime uses simple in-memory
tracking since data loads are processed immediately by the same
instance that received the request.

Requirements: 7.1, 7.5, 12.1, 12.2, 12.3
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class DataLoadStatus(str, Enum):
    """Status of a data load job."""
    QUEUED = "queued"
    VALIDATING = "validating"
    ENCODING = "encoding"
    INDEXING = "indexing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class DataLoadJob:
    """A data load job with progress tracking."""
    id: UUID
    org_id: str
    model_id: str
    status: DataLoadStatus
    total_records: int
    processed_count: int = 0
    encoded_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    progress: int = 0  # 0-100
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    failed_records: list[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": str(self.id),
            "org_id": self.org_id,
            "model_id": self.model_id,
            "status": self.status.value,
            "total_records": self.total_records,
            "processed_count": self.processed_count,
            "encoded_count": self.encoded_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
        }


class JobManager:
    """
    In-memory job manager for Runtime data load operations.
    
    Provides:
    - Job creation and tracking
    - Progress updates with subscriber notifications
    - Background cleanup of old jobs
    
    Requirements: 7.1, 7.5, 12.1, 12.2, 12.3
    """
    
    JOB_TTL = timedelta(hours=1)  # Requirement 7.5, 12.3
    
    def __init__(self):
        self._jobs: Dict[UUID, DataLoadJob] = {}
        self._subscribers: Dict[UUID, list[asyncio.Queue]] = defaultdict(list)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start background cleanup task."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("JobManager started")
    
    async def stop(self) -> None:
        """Stop cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("JobManager stopped")
    
    def create_job(
        self,
        org_id: str,
        model_id: str,
        total_records: int,
    ) -> DataLoadJob:
        """Create a new data load job.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            total_records: Total number of records to process
            
        Returns:
            Created DataLoadJob with status QUEUED
        """
        job = DataLoadJob(
            id=uuid4(),
            org_id=org_id,
            model_id=model_id,
            status=DataLoadStatus.QUEUED,
            total_records=total_records,
        )
        self._jobs[job.id] = job
        logger.info(f"Created job {job.id} for {org_id}/{model_id} with {total_records} records")
        return job
    
    def get_job(self, job_id: UUID) -> Optional[DataLoadJob]:
        """Get job by ID.
        
        Requirement: 12.1
        """
        return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        org_id: str,
        model_id: str,
        limit: int = 10,
    ) -> list[DataLoadJob]:
        """List recent jobs for a model.
        
        Requirement: 12.2
        """
        jobs = [
            j for j in self._jobs.values()
            if j.org_id == org_id and j.model_id == model_id
        ]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]
    
    async def update_progress(
        self,
        job_id: UUID,
        status: DataLoadStatus,
        processed: int = 0,
        encoded: int = 0,
        failed: int = 0,
        skipped: int = 0,
        message: Optional[str] = None,
    ) -> None:
        """Update job progress and notify subscribers.
        
        Args:
            job_id: Job UUID
            status: New status
            processed: Records processed so far
            encoded: Successfully encoded count
            failed: Failed count
            skipped: Skipped count
            message: Optional status message
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found for progress update")
            return
        
        job.status = status
        job.processed_count = processed
        job.encoded_count = encoded
        job.failed_count = failed
        job.skipped_count = skipped
        job.progress = int((processed / job.total_records) * 100) if job.total_records > 0 else 0
        job.updated_at = datetime.utcnow()
        
        if message and status == DataLoadStatus.ERROR:
            job.error_message = message
        
        # Build event for subscribers
        event = {
            "type": status.value,
            "job_id": str(job_id),
            "progress": job.progress,
            "processed": processed,
            "encoded": encoded,
            "failed": failed,
            "skipped": skipped,
            "total": job.total_records,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Notify all subscribers
        for queue in self._subscribers[job_id]:
            try:
                await queue.put(event)
            except Exception as e:
                logger.warning(f"Failed to notify subscriber for job {job_id}: {e}")
    
    def subscribe(self, job_id: UUID) -> asyncio.Queue:
        """Subscribe to job events.
        
        Returns an asyncio.Queue that will receive events.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id].append(queue)
        logger.debug(f"Subscriber added for job {job_id}")
        return queue
    
    def unsubscribe(self, job_id: UUID, queue: asyncio.Queue) -> None:
        """Unsubscribe from job events."""
        if job_id in self._subscribers:
            try:
                self._subscribers[job_id].remove(queue)
                logger.debug(f"Subscriber removed for job {job_id}")
            except ValueError:
                pass
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up old jobs (Requirement 7.5)."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                cutoff = datetime.utcnow() - self.JOB_TTL
                expired = [
                    jid for jid, job in self._jobs.items()
                    if job.updated_at < cutoff
                ]
                for jid in expired:
                    del self._jobs[jid]
                    self._subscribers.pop(jid, None)
                    logger.debug(f"Cleaned up expired job {jid}")
                
                if expired:
                    logger.info(f"Cleaned up {len(expired)} expired jobs")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")


# Singleton instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get or create JobManager singleton."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


async def start_job_manager() -> JobManager:
    """Start the job manager."""
    manager = get_job_manager()
    await manager.start()
    return manager


async def stop_job_manager() -> None:
    """Stop the job manager."""
    global _job_manager
    if _job_manager:
        await _job_manager.stop()
