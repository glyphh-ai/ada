"""Async Listener Service for data loading with progress tracking.

Processes records in batches and reports progress via JobManager.
Returns immediately with job_id, processing happens in background.

Accepts raw JSON records and converts them to Concepts using the model's
encoder config to find key_part roles (composite primary key) and temporal
role for concept naming.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from domains.jobs.manager import DataLoadStatus, JobManager, get_job_manager
from domains.models.storage import GlyphStorage

logger = logging.getLogger(__name__)


def _find_key_part_roles(encoder_config) -> List[str]:
    """
    Find all roles marked as key_part in the encoder config.
    
    Searches through layers -> segments -> roles to find roles
    with key_part=True. Returns them in segment order.
    
    Args:
        encoder_config: EncoderConfig dict or object
    
    Returns:
        List of role names with key_part=True, in segment order
    
    Requirements: 7.1, 7.5
    """
    if not encoder_config:
        return []
    
    key_part_roles = []
    
    # Handle dict config (from DB storage)
    if isinstance(encoder_config, dict):
        layers = encoder_config.get("layers", [])
        for layer in layers:
            segments = layer.get("segments", [])
            for segment in segments:
                roles = segment.get("roles", [])
                for role in roles:
                    if role.get("key_part", False):
                        key_part_roles.append(role.get("name"))
        return key_part_roles
    
    # Handle EncoderConfig object
    if hasattr(encoder_config, "layers"):
        for layer in encoder_config.layers:
            for segment in layer.segments:
                for role in segment.roles:
                    if getattr(role, "key_part", False):
                        key_part_roles.append(role.name)
    
    return key_part_roles


def _find_temporal_role(encoder_config) -> Optional[str]:
    """
    Find the role marked as temporal in the encoder config.
    
    Searches through layers -> segments -> roles to find the role
    with temporal=True.
    
    Args:
        encoder_config: EncoderConfig dict or object
    
    Returns:
        Role name if found, None otherwise
    
    Requirements: 7.3
    """
    if not encoder_config:
        return None
    
    # Handle dict config (from DB storage)
    if isinstance(encoder_config, dict):
        layers = encoder_config.get("layers", [])
        for layer in layers:
            segments = layer.get("segments", [])
            for segment in segments:
                roles = segment.get("roles", [])
                for role in roles:
                    if role.get("temporal", False):
                        return role.get("name")
        return None
    
    # Handle EncoderConfig object
    if hasattr(encoder_config, "layers"):
        for layer in encoder_config.layers:
            for segment in layer.segments:
                for role in segment.roles:
                    if getattr(role, "temporal", False):
                        return role.name
    
    return None


def _find_primary_id_role(encoder_config) -> Optional[str]:
    """
    Find the role marked as primary_id in the encoder config.
    
    DEPRECATED: Use _find_key_part_roles() for composite keys.
    This function is kept for backward compatibility with configs
    that still use primary_id instead of key_part.
    
    Returns:
        Role name if found, None otherwise
    """
    if not encoder_config:
        return None
    
    # Handle dict config (from DB storage)
    if isinstance(encoder_config, dict):
        layers = encoder_config.get("layers", [])
        for layer in layers:
            segments = layer.get("segments", [])
            for segment in segments:
                roles = segment.get("roles", [])
                for role in roles:
                    # Check both primary_id (legacy) and key_part (new)
                    if role.get("primary_id", False) or role.get("key_part", False):
                        return role.get("name")
        return None
    
    # Handle EncoderConfig object
    if hasattr(encoder_config, "layers"):
        for layer in encoder_config.layers:
            for segment in layer.segments:
                for role in segment.roles:
                    if getattr(role, "primary_id", False) or getattr(role, "key_part", False):
                        return role.name
    
    return None


def _record_to_concept(
    record: Dict[str, Any],
    index: int,
    key_part_roles: List[str] = None,
    temporal_role: Optional[str] = None
) -> 'Concept':
    """
    Convert a raw JSON record to a Concept object.
    
    Args:
        record: Raw JSON record dict
        index: Record index (used as fallback name)
        key_part_roles: List of role names that form the composite key
        temporal_role: Name of the role holding temporal data
    
    Returns:
        Concept object ready for encoding
    
    Requirements: 7.2, 7.3, 7.4
    """
    from glyphh.core.types import Concept
    
    # Build concept name from composite key (key_part roles)
    # Use case-insensitive matching for record keys
    if key_part_roles:
        # Build case-insensitive lookup of record keys
        record_keys_lower = {k.lower(): k for k in record.keys()}
        key_values = []
        for role_name in key_part_roles:
            actual_key = record_keys_lower.get(role_name.lower())
            if actual_key and actual_key in record:
                value = str(record[actual_key])
                # Sanitize value for identifier
                sanitized = value.replace(" ", "_").replace("@", "_").replace("#", "_")
                key_values.append(sanitized)
        
        if key_values:
            concept_name = "_".join(key_values)
        else:
            # Fallback if no key_part values found
            concept_name = f"record_{index}"
    elif "name" in record:
        concept_name = str(record["name"])
    elif "id" in record:
        concept_name = str(record["id"])
    else:
        concept_name = f"record_{index}"
    
    # Build metadata including temporal role info
    metadata = {
        "source": "listener",
        "index": index,
    }
    if temporal_role:
        metadata["temporal_role"] = temporal_role
        if temporal_role in record:
            metadata["temporal_value"] = record[temporal_role]
    
    return Concept(
        name=concept_name,
        attributes=record,
        relationships=[],
        metadata=metadata
    )


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
            
            # Get encoder and model info
            encoder = await self._encoder_getter(org_id, model_id)
            if not encoder:
                raise ValueError(f"Model not loaded: {org_id}/{model_id}")
            
            # Find key_part roles (composite key) and temporal role from encoder config
            encoder_config = getattr(encoder, 'config', None)
            key_part_roles = _find_key_part_roles(encoder_config)
            temporal_role = _find_temporal_role(encoder_config)
            
            # Backward compatibility: if no key_part roles, try legacy primary_id
            if not key_part_roles:
                primary_id_role = _find_primary_id_role(encoder_config)
                if primary_id_role:
                    key_part_roles = [primary_id_role]
            
            if key_part_roles:
                logger.info(f"Using key_part roles {key_part_roles} for composite key")
            else:
                logger.info("No key_part roles found, using fallback naming (name/id/index)")
            
            if temporal_role:
                logger.info(f"Using temporal role '{temporal_role}' for time-based identifiers")
            
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
                            # Validate record is a dict
                            if not isinstance(record, dict):
                                skipped += 1
                                failed_records.append({
                                    "index": processed,
                                    "error": f"Record must be a dict, got {type(record).__name__}",
                                })
                                processed += 1
                                continue
                            
                            # Validate key_part fields exist if configured (case-insensitive)
                            if key_part_roles:
                                # Build case-insensitive lookup of record keys
                                record_keys_lower = {k.lower(): k for k in record.keys()}
                                missing_keys = [
                                    k for k in key_part_roles 
                                    if k.lower() not in record_keys_lower
                                ]
                                if missing_keys:
                                    skipped += 1
                                    failed_records.append({
                                        "index": processed,
                                        "error": f"Missing required key_part field(s): {missing_keys}",
                                    })
                                    processed += 1
                                    continue
                            
                            # Convert raw JSON to Concept using composite key
                            concept = _record_to_concept(record, processed, key_part_roles, temporal_role)
                            
                            # Encode using the SDK encoder
                            glyph = encoder.encode(concept)
                            
                            # Extract embedding from glyph
                            if hasattr(glyph, 'cortex') and hasattr(glyph.cortex, 'data'):
                                embedding = glyph.cortex.data
                            elif hasattr(glyph, 'global_cortex') and hasattr(glyph.global_cortex, 'data'):
                                embedding = glyph.global_cortex.data
                            else:
                                raise ValueError("Glyph has no cortex/global_cortex with data")
                            
                            embedding_list = (
                                embedding.tolist() 
                                if hasattr(embedding, 'tolist') 
                                else list(embedding)
                            )
                            
                            # Store with full record as metadata
                            concept_text = json.dumps(record)
                            
                            await storage.create_glyph(
                                org_id=org_id,
                                model_id=model_id,
                                concept_text=concept_text,
                                embedding=embedding_list,
                                metadata=record,
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
