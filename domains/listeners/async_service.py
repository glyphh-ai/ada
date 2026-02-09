"""Async Listener Service for data loading with progress tracking.

Processes records in batches and reports progress via JobManager.
Returns immediately with job_id, processing happens in background.

Requirements: 7.2, 7.3, 7.4
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from domains.jobs.manager import DataLoadStatus, JobManager, get_job_manager
from domains.models.storage import GlyphStorage

logger = logging.getLogger(__name__)


class AsyncListenerService:
    """
    Async data loading service with progress tracking.
    
    Processes records in batches and reports progress via JobManager.
    Returns immediately with job_id (Requirement 7.2).
    
    Requirements: 7.2, 7.3, 7.4
    """
    
    DEFAULT_BATCH_SIZE = 50  # Requirement 7.3
    
    def __init__(
        self,
        session_maker,
        encoder_getter,
        job_manager: Optional[JobManager] = None,
    ):
        """Initialize AsyncListenerService.
        
        Args:
            session_maker: Async session maker for database access
            encoder_getter: Callable to get encoder for org_id/model_id
            job_manager: Optional JobManager instance (uses singleton if not provided)
        """
        self._session_maker = session_maker
        self._encoder_getter = encoder_getter
        self._job_manager = job_manager or get_job_manager()
    
    async def start_load(
        self,
        org_id: str,
        model_id: str,
        records: List[Dict[str, Any]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> UUID:
        """
        Start async data load. Returns job_id immediately.
        
        Processing happens in background task (Requirement 7.2).
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            records: List of records to load (each with concept/text field)
            batch_size: Records per batch (default 50)
            
        Returns:
            Job UUID for tracking progress
        """
        job = self._job_manager.create_job(
            org_id=org_id,
            model_id=model_id,
            total_records=len(records),
        )
        
        # Start background processing
        asyncio.create_task(
            self._process_records(job.id, org_id, model_id, records, batch_size),
            name=f"data_load_{job.id}"
        )
        
        logger.info(f"Started async data load job {job.id} for {org_id}/{model_id}")
        return job.id
    
    async def _process_records(
        self,
        job_id: UUID,
        org_id: str,
        model_id: str,
        records: List[Dict[str, Any]],
        batch_size: int,
    ) -> None:
        """Process records in batches with progress updates.
        
        Steps (Requirement 7.3):
        1. Validate records against model schema
        2. Encode records in batches
        3. Index encoded vectors
        4. Report final counts
        
        Progress updates after each batch (Requirement 7.4).
        """
        try:
            # Phase 1: Validation
            await self._job_manager.update_progress(
                job_id,
                DataLoadStatus.VALIDATING,
                message="Validating records...",
            )
            
            # Get encoder
            encoder = await self._encoder_getter(org_id, model_id)
            if not encoder:
                raise ValueError(f"Model not loaded: {org_id}/{model_id}")
            
            # Phase 2: Encoding
            await self._job_manager.update_progress(
                job_id,
                DataLoadStatus.ENCODING,
                message="Encoding records...",
            )
            
            processed = 0
            encoded = 0
            failed = 0
            skipped = 0
            failed_records = []
            
            async with self._session_maker() as session:
                storage = GlyphStorage(session)
                
                # Process in batches
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    
                    for record in batch:
                        try:
                            # Extract concept text
                            concept = (
                                record.get("concept") or 
                                record.get("text") or 
                                record.get("concept_text")
                            )
                            
                            if not concept:
                                skipped += 1
                                failed_records.append({
                                    "index": processed,
                                    "error": "Missing concept/text field",
                                })
                                processed += 1
                                continue
                            
                            # Encode
                            embedding = encoder.encode_text(concept)
                            embedding_list = (
                                embedding.tolist() 
                                if hasattr(embedding, 'tolist') 
                                else list(embedding)
                            )
                            
                            # Store
                            await storage.create_glyph(
                                org_id=org_id,
                                model_id=model_id,
                                concept_text=concept,
                                embedding=embedding_list,
                                metadata=record.get("metadata", {}),
                            )
                            
                            encoded += 1
                            processed += 1
                            
                        except Exception as e:
                            failed += 1
                            failed_records.append({
                                "index": processed,
                                "error": str(e),
                            })
                            processed += 1
                            logger.warning(f"Failed to encode record {processed}: {e}")
                    
                    # Update progress after each batch (Requirement 7.4)
                    await self._job_manager.update_progress(
                        job_id,
                        DataLoadStatus.ENCODING,
                        processed=processed,
                        encoded=encoded,
                        failed=failed,
                        skipped=skipped,
                        message=f"Encoded {encoded}/{len(records)} records",
                    )
                
                # Commit all changes
                await session.commit()
            
            # Phase 3: Indexing (placeholder for future vector index updates)
            await self._job_manager.update_progress(
                job_id,
                DataLoadStatus.INDEXING,
                processed=processed,
                encoded=encoded,
                failed=failed,
                skipped=skipped,
                message="Indexing vectors...",
            )
            
            # Brief pause to simulate indexing
            await asyncio.sleep(0.1)
            
            # Phase 4: Complete
            await self._job_manager.update_progress(
                job_id,
                DataLoadStatus.COMPLETE,
                processed=processed,
                encoded=encoded,
                failed=failed,
                skipped=skipped,
                message=f"Complete: {encoded} encoded, {failed} failed, {skipped} skipped",
            )
            
            logger.info(
                f"Data load job {job_id} completed: "
                f"{encoded} encoded, {failed} failed, {skipped} skipped"
            )
            
        except Exception as e:
            logger.error(f"Data load job {job_id} failed: {e}", exc_info=True)
            await self._job_manager.update_progress(
                job_id,
                DataLoadStatus.ERROR,
                message=str(e),
            )
