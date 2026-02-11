"""Async Listener Service for data loading with progress tracking.

Processes records in batches and reports progress via JobManager.
Returns immediately with job_id, processing happens in background.

Accepts JSON records in two formats:

1. Flat format (auto-mapped to config structure):
{
    "role_name": "value",
    "another_role": "value"
}

2. Hierarchical format matching the encoder config:
{
    "layer_name": {
        "segment_name": {
            "role_name": "value"
        }
    }
}

Flat records are automatically mapped to the hierarchical structure
by matching keys to role names in the encoder config.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from domains.jobs.manager import DataLoadStatus, JobManager, get_job_manager
from domains.models.storage import GlyphStorage

logger = logging.getLogger(__name__)


def _get_config_structure(encoder_config) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract the layer/segment/role structure from encoder config.
    
    Returns:
        Dict mapping layer_name -> segment_name -> [role_names]
    """
    structure = {}
    
    if not encoder_config:
        return structure
    
    # Handle dict config (from DB storage)
    if isinstance(encoder_config, dict):
        layers = encoder_config.get("layers", [])
        for layer in layers:
            layer_name = layer.get("name", "")
            if not layer_name:
                continue
            structure[layer_name] = {}
            for segment in layer.get("segments", []):
                segment_name = segment.get("name", "")
                if not segment_name:
                    continue
                structure[layer_name][segment_name] = [
                    role.get("name") for role in segment.get("roles", [])
                    if role.get("name")
                ]
        return structure
    
    # Handle EncoderConfig object
    if hasattr(encoder_config, "layers"):
        for layer in encoder_config.layers:
            structure[layer.name] = {}
            for segment in layer.segments:
                structure[layer.name][segment.name] = [
                    role.name for role in segment.roles
                ]
    
    return structure


def _find_key_part_roles(encoder_config) -> List[Tuple[str, str, str]]:
    """
    Find all roles marked as key_part in the encoder config.
    
    Returns:
        List of (layer_name, segment_name, role_name) tuples with key_part=True
    
    Requirements: 7.1, 7.5
    """
    if not encoder_config:
        return []
    
    key_part_roles = []
    
    # Handle dict config (from DB storage)
    if isinstance(encoder_config, dict):
        layers = encoder_config.get("layers", [])
        for layer in layers:
            layer_name = layer.get("name", "")
            for segment in layer.get("segments", []):
                segment_name = segment.get("name", "")
                for role in segment.get("roles", []):
                    if role.get("key_part", False):
                        key_part_roles.append((layer_name, segment_name, role.get("name")))
        return key_part_roles
    
    # Handle EncoderConfig object
    if hasattr(encoder_config, "layers"):
        for layer in encoder_config.layers:
            for segment in layer.segments:
                for role in segment.roles:
                    if getattr(role, "key_part", False):
                        key_part_roles.append((layer.name, segment.name, role.name))
    
    return key_part_roles


def _find_temporal_role(encoder_config) -> Optional[Tuple[str, str, str]]:
    """
    Find the temporal source role from the encoder config.
    
    Uses the temporal_source field which can be:
    - "auto" or empty: No role-based temporal, auto-generate timestamps
    - "layer.segment.role": Path to the role providing temporal values
    
    Returns:
        (layer_name, segment_name, role_name) tuple if found, None otherwise
    
    Requirements: 7.3
    """
    if not encoder_config:
        return None
    
    # Handle dict config (from DB storage)
    if isinstance(encoder_config, dict):
        temporal_source = encoder_config.get("temporal_source", "auto")
        if temporal_source and temporal_source != "auto":
            # Parse "layer.segment.role" path
            parts = temporal_source.split(".")
            if len(parts) == 3:
                return (parts[0], parts[1], parts[2])
        return None
    
    # Handle EncoderConfig object
    if hasattr(encoder_config, "temporal_source"):
        temporal_source = encoder_config.temporal_source
        if temporal_source and temporal_source != "auto":
            parts = temporal_source.split(".")
            if len(parts) == 3:
                return (parts[0], parts[1], parts[2])
    
    return None


def _auto_map_flat_to_hierarchical(
    record: Dict[str, Any],
    config_structure: Dict[str, Dict[str, List[str]]]
) -> Optional[Dict[str, Any]]:
    """
    Auto-map a flat record to hierarchical format based on config structure.
    
    Matches flat keys to role names in the config. If a role name matches
    a key in the flat record, it maps the value to the hierarchical path.
    
    Args:
        record: Flat record dict (e.g., {"vin": "123", "make": "Toyota"})
        config_structure: Expected hierarchy from encoder config
        
    Returns:
        Hierarchical record if mapping succeeds, None if not enough matches
    """
    if not config_structure:
        return None
    
    record_keys = set(record.keys())
    hierarchical = {}
    matched_roles = 0
    total_roles = 0
    
    for layer_name, segments in config_structure.items():
        hierarchical[layer_name] = {}
        for segment_name, roles in segments.items():
            hierarchical[layer_name][segment_name] = {}
            for role_name in roles:
                total_roles += 1
                # Check if flat record has this role name as a key
                if role_name in record_keys:
                    hierarchical[layer_name][segment_name][role_name] = record[role_name]
                    matched_roles += 1
                else:
                    # Role not found in flat data - set to None
                    hierarchical[layer_name][segment_name][role_name] = None
    
    # Only return mapped record if we matched at least one role
    # This allows partial data to be encoded
    if matched_roles > 0:
        return hierarchical
    
    return None


def _validate_record_structure(
    record: Dict[str, Any],
    config_structure: Dict[str, Dict[str, List[str]]],
    index: int
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate that a record matches the expected hierarchical structure.
    Auto-maps flat records to hierarchical format when possible.
    
    Expected format:
    {
        "layer_name": {
            "segment_name": {
                "role_name": "value"
            }
        }
    }
    
    Returns:
        (is_valid, error_message, mapped_record) tuple
        - mapped_record is the hierarchical version (same as input if already hierarchical,
          or auto-mapped if flat)
    """
    if not isinstance(record, dict):
        return False, f"Record {index}: Must be a JSON object, got {type(record).__name__}", None
    
    if not config_structure:
        # No config structure defined, accept flat records as-is
        return True, None, record
    
    # Check if record has hierarchical structure (layer keys)
    expected_layers = set(config_structure.keys())
    record_keys = set(record.keys())
    
    # If record has any expected layer names, validate hierarchical structure
    if record_keys & expected_layers:
        # Hierarchical format - validate structure
        missing_layers = expected_layers - record_keys
        if missing_layers:
            layer_format = ', '.join(f'"{l}": {{...}}' for l in expected_layers)
            return False, (
                f"Record {index}: Missing layer(s): {list(missing_layers)}. "
                f"Expected format: {{{layer_format}}}"
            ), None
        
        for layer_name, segments in config_structure.items():
            layer_data = record.get(layer_name)
            if not isinstance(layer_data, dict):
                return False, (
                    f"Record {index}: Layer '{layer_name}' must be an object, "
                    f"got {type(layer_data).__name__}"
                ), None
            
            expected_segments = set(segments.keys())
            layer_keys = set(layer_data.keys())
            missing_segments = expected_segments - layer_keys
            
            if missing_segments:
                return False, (
                    f"Record {index}: Layer '{layer_name}' missing segment(s): {list(missing_segments)}. "
                    f"Expected: {list(expected_segments)}"
                ), None
            
            for segment_name, roles in segments.items():
                segment_data = layer_data.get(segment_name)
                if not isinstance(segment_data, dict):
                    return False, (
                        f"Record {index}: Segment '{layer_name}.{segment_name}' must be an object, "
                        f"got {type(segment_data).__name__}"
                    ), None
                
                expected_roles = set(roles)
                segment_keys = set(segment_data.keys())
                missing_roles = expected_roles - segment_keys
                
                if missing_roles:
                    return False, (
                        f"Record {index}: Segment '{layer_name}.{segment_name}' missing role(s): {list(missing_roles)}. "
                        f"Your data has: {list(segment_keys)}"
                    ), None
        
        return True, None, record
    else:
        # Flat format - try to auto-map to hierarchical structure
        mapped_record = _auto_map_flat_to_hierarchical(record, config_structure)
        
        if mapped_record:
            # Successfully auto-mapped flat record
            return True, None, mapped_record
        
        # Could not auto-map - provide helpful error
        # Collect all role names from config for hint
        all_role_names = []
        for segments in config_structure.values():
            for roles in segments.values():
                all_role_names.extend(roles)
        
        example_structure = {}
        for layer_name, segments in config_structure.items():
            example_structure[layer_name] = {}
            for segment_name, roles in segments.items():
                example_structure[layer_name][segment_name] = {
                    role: "<value>" for role in roles[:3]  # Show first 3 roles
                }
                if len(roles) > 3:
                    example_structure[layer_name][segment_name]["..."] = "..."
        
        return False, (
            f"Record {index}: Could not map flat data to config structure.\n"
            f"Your data has keys: {list(record_keys)[:5]}{'...' if len(record_keys) > 5 else ''}\n"
            f"Expected role names: {all_role_names[:8]}{'...' if len(all_role_names) > 8 else ''}\n"
            f"Either use flat keys matching role names, or hierarchical format:\n"
            f"{json.dumps(example_structure, indent=2)}"
        ), None


def _flatten_hierarchical_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a hierarchical record to a flat dict for Concept attributes.
    
    Input: {"layer": {"segment": {"role": "value"}}}
    Output: {"role": "value", ...} (all roles flattened)
    
    Also preserves the full path as metadata.
    """
    flat = {}
    for layer_name, layer_data in record.items():
        if isinstance(layer_data, dict):
            for segment_name, segment_data in layer_data.items():
                if isinstance(segment_data, dict):
                    for role_name, value in segment_data.items():
                        # Use full path as key to avoid collisions
                        flat[f"{layer_name}.{segment_name}.{role_name}"] = value
                        # Also store just the role name for backward compat
                        # (only if no collision)
                        if role_name not in flat:
                            flat[role_name] = value
    return flat


def _get_value_from_path(
    record: Dict[str, Any],
    layer: str,
    segment: str,
    role: str
) -> Optional[Any]:
    """Get a value from a hierarchical record using layer.segment.role path."""
    try:
        return record[layer][segment][role]
    except (KeyError, TypeError):
        return None


def _extract_hierarchical_vectors(glyph) -> List[Dict[str, Any]]:
    """
    Extract hierarchical vectors from an encoded glyph.
    
    Extracts layer cortices, segment cortices, and role bindings
    for storage in the glyph_vectors table.
    
    Args:
        glyph: Encoded SDK Glyph object
        
    Returns:
        List of dicts with 'level', 'path', 'embedding' keys
    """
    vectors = []
    
    # Check if glyph has layers attribute
    if not hasattr(glyph, 'layers') or not glyph.layers:
        return vectors
    
    for layer_name, layer in glyph.layers.items():
        # Extract layer cortex
        if hasattr(layer, 'cortex') and layer.cortex is not None:
            layer_embedding = _vector_to_list(layer.cortex)
            if layer_embedding:
                vectors.append({
                    'level': 'layer',
                    'path': layer_name,
                    'embedding': layer_embedding,
                })
        
        # Extract segment cortices
        if hasattr(layer, 'segments') and layer.segments:
            for segment_name, segment in layer.segments.items():
                segment_path = f"{layer_name}.{segment_name}"
                
                # Segment cortex
                if hasattr(segment, 'cortex') and segment.cortex is not None:
                    segment_embedding = _vector_to_list(segment.cortex)
                    if segment_embedding:
                        vectors.append({
                            'level': 'segment',
                            'path': segment_path,
                            'embedding': segment_embedding,
                        })
                
                # Role bindings
                if hasattr(segment, 'roles') and segment.roles:
                    for role_name, role_vector in segment.roles.items():
                        role_path = f"{layer_name}.{segment_name}.{role_name}"
                        role_embedding = _vector_to_list(role_vector)
                        if role_embedding:
                            vectors.append({
                                'level': 'role',
                                'path': role_path,
                                'embedding': role_embedding,
                            })
    
    return vectors


def _vector_to_list(vector) -> Optional[List[float]]:
    """
    Convert an SDK Vector to a list of floats.
    
    Args:
        vector: SDK Vector object or array-like
        
    Returns:
        List of floats or None if conversion fails
    """
    if vector is None:
        return None
    
    # Handle SDK Vector with data attribute
    if hasattr(vector, 'data'):
        data = vector.data
        if hasattr(data, 'tolist'):
            return data.tolist()
        return list(data)
    
    # Handle numpy array
    if hasattr(vector, 'tolist'):
        return vector.tolist()
    
    # Handle list
    if isinstance(vector, list):
        return vector
    
    return None


def _record_to_concept(
    record: Dict[str, Any],
    index: int,
    key_part_roles: List[Tuple[str, str, str]] = None,
    temporal_role: Optional[Tuple[str, str, str]] = None
) -> 'Concept':
    """
    Convert a hierarchical JSON record to a Concept object.
    
    Args:
        record: Hierarchical JSON record dict
        index: Record index (used as fallback name)
        key_part_roles: List of (layer, segment, role) tuples for composite key
        temporal_role: (layer, segment, role) tuple for temporal data
    
    Returns:
        Concept object ready for encoding
    
    Requirements: 7.2, 7.3, 7.4
    """
    from glyphh.core.types import Concept
    
    # Build concept name from composite key (key_part roles)
    if key_part_roles:
        key_values = []
        for layer, segment, role in key_part_roles:
            value = _get_value_from_path(record, layer, segment, role)
            if value is not None:
                sanitized = str(value).replace(" ", "_").replace("@", "_").replace("#", "_")
                key_values.append(sanitized)
        
        if key_values:
            concept_name = "_".join(key_values)
        else:
            concept_name = f"record_{index}"
    else:
        concept_name = f"record_{index}"
    
    # Flatten record for Concept attributes
    flat_attributes = _flatten_hierarchical_record(record)
    
    # Build metadata
    metadata = {
        "source": "listener",
        "index": index,
        "original_record": record,  # Keep original hierarchical structure
    }
    
    if temporal_role:
        layer, segment, role = temporal_role
        metadata["temporal_role"] = f"{layer}.{segment}.{role}"
        temporal_value = _get_value_from_path(record, layer, segment, role)
        if temporal_value is not None:
            metadata["temporal_value"] = temporal_value
    
    return Concept(
        name=concept_name,
        attributes=flat_attributes,
        relationships=[],
        metadata=metadata
    )


class AsyncListenerService:
    """
    Async data loading service with progress tracking.
    
    Processes records in batches and reports progress via JobManager.
    Returns immediately with job_id (Requirement 7.2).
    
    Accepts flat or hierarchical records:
    - Flat: {"role_name": "value"} - auto-mapped to config structure
    - Hierarchical: {"layer": {"segment": {"role": "value"}}}
    
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
            records: List of records in hierarchical format matching encoder config
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
        1. Validate records against encoder config structure (STRICT)
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
                message="Validating records against encoder config...",
            )
            
            # Get encoder and model info
            encoder = await self._encoder_getter(org_id, model_id)
            if not encoder:
                raise ValueError(f"Model not loaded: {org_id}/{model_id}")
            
            # Get encoder config structure for validation
            encoder_config = getattr(encoder, 'config', None)
            config_structure = _get_config_structure(encoder_config)
            key_part_roles = _find_key_part_roles(encoder_config)
            temporal_role = _find_temporal_role(encoder_config)
            
            if config_structure:
                layer_names = list(config_structure.keys())
                logger.info(f"Validating records against config structure: layers={layer_names}")
            else:
                logger.info("No config structure found, accepting flat records")
            
            if key_part_roles:
                key_paths = [f"{l}.{s}.{r}" for l, s, r in key_part_roles]
                logger.info(f"Using key_part roles {key_paths} for composite key")
            else:
                logger.info("No key_part roles found, using index-based naming")
            
            if temporal_role:
                l, s, r = temporal_role
                logger.info(f"Using temporal role '{l}.{s}.{r}' for time-based identifiers")
            
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
                            # Validate and auto-map flat records to hierarchical format
                            is_valid, error_msg, mapped_record = _validate_record_structure(
                                record, config_structure, processed
                            )
                            
                            if not is_valid:
                                skipped += 1
                                failed_records.append({
                                    "index": processed,
                                    "error": error_msg,
                                })
                                processed += 1
                                # Log first validation error for debugging
                                if skipped == 1:
                                    logger.warning(f"Record validation failed: {error_msg}")
                                continue
                            
                            # Use mapped record (hierarchical format)
                            record_to_encode = mapped_record if mapped_record else record
                            
                            # Convert hierarchical JSON to Concept
                            concept = _record_to_concept(
                                record_to_encode, processed, key_part_roles, temporal_role
                            )
                            
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
                            
                            # Store with original record as metadata (user's format)
                            concept_text = json.dumps(record)
                            
                            glyph_response = await storage.create_glyph(
                                org_id=org_id,
                                model_id=model_id,
                                concept_text=concept_text,
                                embedding=embedding_list,
                                metadata=record,  # Store original flat format for display
                            )
                            
                            # Extract and store hierarchical vectors
                            hierarchical_vectors = _extract_hierarchical_vectors(glyph)
                            if hierarchical_vectors:
                                await storage.create_glyph_vectors_batch(
                                    glyph_id=glyph_response.glyph_id,
                                    org_id=org_id,
                                    model_id=model_id,
                                    vectors=hierarchical_vectors,
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
                            # Rollback the session to clear the failed transaction state
                            await session.rollback()
                    
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
            if skipped > 0 and encoded == 0:
                # All records failed validation - provide helpful message
                first_error = failed_records[0]["error"] if failed_records else "Unknown error"
                await self._job_manager.update_progress(
                    job_id,
                    DataLoadStatus.ERROR,
                    processed=processed,
                    encoded=encoded,
                    failed=failed,
                    skipped=skipped,
                    message=f"All {skipped} records failed validation. {first_error}",
                )
            else:
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
