"""
Model Manager for Glyphh Runtime.

Handles loading, unloading, and managing multiple .glyphh models with
org_id/model_id isolation. Uses SDK's GlyphhModel for model loading only
(packaging), and creates separate Encoder and SimilarityCalculator instances
for runtime operations.

All registry keys are (org_id, model_id) tuples. No namespace concept.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Edge, Glyph, ModelConfig
from domains.models.schemas import (
    ClearDataResponse,
    ModelConfigResponse,
    ModelConfigUpdate,
    ModelInfoResponse,
    ReEncodeResponse,
    ReEncodeStatusResponse,
)
from infrastructure.config import get_settings
from shared.exceptions import (
    ModelIncompatibleException,
    ModelLoadException,
    ModelNotFoundException,
    QuotaExceededException,
)
from shared.sdk_adapter import get_sdk_adapter, SDKNotAvailableError
from shared.encoder_config_factory import EncoderConfigFactory, ConfigurationError
from shared.config_validator import get_config_validator

logger = logging.getLogger(__name__)
settings = get_settings()


class LoadedModel:
    """
    In-memory representation of a loaded model.
    
    Stores org_id and model_id as separate attributes.
    """
    
    def __init__(
        self,
        org_id: str,
        model_id: str,
        model_path: str,
        sdk_model: Any,
        encoder: Any,
        similarity_calculator: Optional[Any],
        loaded_at: datetime,
        meta_name: str,
        short_description: str,
        long_description: str,
        encode_query_fn: Optional[Any] = None,
    ):
        self.org_id = org_id
        self.model_id = model_id
        self.model_path = model_path
        self.sdk_model = sdk_model
        self.encoder = encoder
        self.similarity_calculator = similarity_calculator
        self.loaded_at = loaded_at
        self.lock = asyncio.Lock()
        self.meta_name = meta_name
        self.short_description = short_description
        self.long_description = long_description
        self.encode_query_fn = encode_query_fn


class ReEncodeJob:
    """Tracks background re-encode job status."""
    
    def __init__(self, job_id: str, org_id: str, model_id: str):
        self.job_id = job_id
        self.org_id = org_id
        self.model_id = model_id
        self.status = "pending"
        self.progress = 0.0
        self.glyphs_total = 0
        self.glyphs_processed = 0
        self.edges_regenerated = 0
        self.started_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None


class ModelManager:
    """
    Manages multiple .glyphh models with (org_id, model_id) isolation.
    
    Registry keys are Tuple[str, str] of (org_id, model_id).
    """
    
    def __init__(self, db_session_factory):
        self._models: Dict[Tuple[str, str], LoadedModel] = {}
        self._db_session_factory = db_session_factory
        self._re_encode_jobs: Dict[str, ReEncodeJob] = {}
        self._sdk_version: Optional[str] = None
        
    async def _get_sdk_version(self) -> str:
        if self._sdk_version is None:
            try:
                import glyphh
                self._sdk_version = glyphh.__version__
            except ImportError:
                self._sdk_version = "unknown"
        return self._sdk_version
    
    async def load_model(
        self,
        model_path: str,
        org_id: str,
        model_id: str,
    ) -> LoadedModel:
        """
        Load a .glyphh model and assign it to (org_id, model_id).
        
        Args:
            model_path: Path to .glyphh file
            org_id: Organization ID
            model_id: Model ID
        """
        adapter = get_sdk_adapter()
        if not adapter.is_available:
            raise ModelLoadException("SDK not available")
        
        try:
            from glyphh import GlyphhModel
        except ImportError as e:
            raise ModelLoadException(f"SDK not available: {e}")
        
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadException(f"Model file not found: {model_path}")
        
        if not path.suffix == ".glyphh":
            raise ModelLoadException(f"Invalid file extension: {path.suffix}")
        
        try:
            sdk_model = GlyphhModel.from_file(str(path))
        except Exception as e:
            raise ModelLoadException(f"Failed to parse model file: {e}")
        
        validation_result = await self.validate_compatibility(sdk_model)
        if not validation_result["compatible"]:
            raise ModelIncompatibleException(
                model_version=sdk_model.version,
                sdk_version=await self._get_sdk_version()
            )
        
        key = (org_id, model_id)
        is_redeploy = key in self._models
        
        # Check local mode model limit (only for new deploys, not re-deploys)
        if not is_redeploy and settings.deployment_mode == "local":
            if len(self._models) >= settings.local_mode_max_models:
                raise ModelLoadException(
                    f"Local mode limit: maximum {settings.local_mode_max_models} model(s). "
                    f"Upgrade to a production license for unlimited models."
                )
        
        # Extract and validate encoder config
        try:
            encoder_config = EncoderConfigFactory.create_from_model(sdk_model)
            validator = get_config_validator()
            validator.validate_encoder_config_or_raise(encoder_config)
            encoder_config = validator.apply_defaults(encoder_config)
        except ConfigurationError as e:
            raise ModelLoadException(f"Invalid model configuration: {e}")
        
        try:
            encoder = adapter.create_encoder(encoder_config)
        except Exception as e:
            raise ModelLoadException(f"Failed to create encoder: {e}")
        
        similarity_calculator = adapter.create_similarity_calculator()
        if similarity_calculator is None:
            logger.warning(
                f"SimilarityCalculator not available for org={org_id}, model={model_id}, "
                f"will use fallback similarity"
            )
        
        meta_name = getattr(sdk_model, 'meta_name', None) or \
                    getattr(sdk_model, 'name', None) or \
                    path.stem
        short_description = getattr(sdk_model, 'short_description', '') or ''
        long_description = getattr(sdk_model, 'long_description', '') or ''
        
        loaded_model = LoadedModel(
            org_id=org_id,
            model_id=model_id,
            model_path=str(path),
            sdk_model=sdk_model,
            encoder=encoder,
            similarity_calculator=similarity_calculator,
            loaded_at=datetime.utcnow(),
            meta_name=meta_name,
            short_description=short_description,
            long_description=long_description,
        )
        
        # Serialize encoder config for DB storage
        encoder_config_dict = None
        if hasattr(encoder_config, 'to_dict'):
            encoder_config_dict = encoder_config.to_dict()
        
        # DB upsert first — if this fails, the old model stays intact in memory
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == org_id,
                    ModelConfig.model_id == model_id,
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.model_path = str(path)
                existing.model_version = sdk_model.version
                existing.sdk_version = await self._get_sdk_version()
                existing.meta_name = meta_name
                existing.short_description = short_description
                existing.long_description = long_description
                existing.encoder_config = encoder_config_dict
                existing.updated_at = datetime.utcnow()
            else:
                config = ModelConfig(
                    org_id=org_id,
                    model_id=model_id,
                    model_path=str(path),
                    model_version=sdk_model.version,
                    sdk_version=await self._get_sdk_version(),
                    meta_name=meta_name,
                    short_description=short_description,
                    long_description=long_description,
                    encoder_config=encoder_config_dict,
                )
                session.add(config)
            await session.commit()
        
        # Everything succeeded — now swap the in-memory model atomically
        old_model = self._models.get(key)
        if old_model is not None:
            logger.info(f"Re-deploying: replacing model org={org_id}, model={model_id}")
            if hasattr(old_model.encoder, 'clear_cache'):
                old_model.encoder.clear_cache()
        
        self._models[key] = loaded_model
        
        logger.info(
            f"Loaded model '{meta_name}' v{sdk_model.version} "
            f"into org={org_id}, model={model_id}"
        )
        
        return loaded_model

    

    async def unload_model(
        self,
        org_id: str,
        model_id: str,
        delete_data: bool = False,
    ) -> None:
        """Unload a model and optionally clean up its data."""
        key = (org_id, model_id)
        
        # Check if model is in memory
        loaded_model = self._models.get(key)
        
        # If not in memory, try to load from DB to verify it exists
        if loaded_model is None:
            loaded_model = await self._load_from_db(org_id, model_id)
            if loaded_model is not None:
                # Model exists in DB, add to cache so we can clean it up
                self._models[key] = loaded_model
        
        # If still not found, check if there's data in the database anyway
        if loaded_model is None:
            async with self._db_session_factory() as session:
                result = await session.execute(
                    select(ModelConfig).where(
                        ModelConfig.org_id == org_id,
                        ModelConfig.model_id == model_id,
                    )
                )
                db_config = result.scalar_one_or_none()
                
                if db_config is None:
                    # No config and no model - truly not found
                    raise ModelNotFoundException(org_id, model_id)
                
                # Config exists but couldn't load model - still allow cleanup
                logger.info(f"Model config exists but couldn't load encoder for org={org_id}, model={model_id}")
        
        if delete_data:
            await self.clear_model_data(org_id, model_id)
            
            async with self._db_session_factory() as session:
                await session.execute(
                    delete(ModelConfig).where(
                        ModelConfig.org_id == org_id,
                        ModelConfig.model_id == model_id,
                    )
                )
                await session.commit()
        
        # Remove from memory if present
        if key in self._models:
            loaded_model = self._models[key]
            del self._models[key]
            
            if hasattr(loaded_model, 'encoder') and hasattr(loaded_model.encoder, 'clear_cache'):
                loaded_model.encoder.clear_cache()
        
        logger.info(f"Unloaded model org={org_id}, model={model_id}")
    
    async def get_model(self, org_id: str, model_id: str) -> Optional[LoadedModel]:
        """
        Retrieve a loaded model by (org_id, model_id).
        
        Checks in-memory cache first. If not present, attempts to
        restore from the database (lazy-load on first request after restart).
        """
        key = (org_id, model_id)
        model = self._models.get(key)
        if model is not None:
            return model
        
        # Not in memory — try to restore from DB
        return await self._load_from_db(org_id, model_id)
    
    async def _load_from_db(self, org_id: str, model_id: str) -> Optional[LoadedModel]:
        """
        Restore a model from its DB config row.
        
        Reconstructs the encoder and similarity calculator from the
        stored encoder_config JSONB. Returns None if the row doesn't
        exist or the encoder_config is missing.
        """
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == org_id,
                    ModelConfig.model_id == model_id,
                )
            )
            db_config = result.scalar_one_or_none()
        
        if db_config is None:
            return None
        
        if not db_config.encoder_config:
            logger.warning(
                f"Model org={org_id}, model={model_id} exists in DB but has no "
                f"encoder_config — redeploy required"
            )
            return None
        
        try:
            adapter = get_sdk_adapter()
            if not adapter.is_available:
                logger.error("SDK not available, cannot restore model from DB")
                return None
            
            encoder_config = EncoderConfigFactory.create_from_dict(db_config.encoder_config)
            validator = get_config_validator()
            validator.validate_encoder_config_or_raise(encoder_config)
            encoder_config = validator.apply_defaults(encoder_config)
            
            encoder = adapter.create_encoder(encoder_config)
            similarity_calculator = adapter.create_similarity_calculator()
            
            class RestoredModel:
                """Minimal model proxy for DB-restored models."""
                def __init__(self, name, version, config):
                    self.name = name
                    self.version = version
                    self.encoder_config = config
            
            sdk_model_proxy = RestoredModel(
                db_config.meta_name or model_id,
                db_config.model_version or "unknown",
                encoder_config,
            )
            
            loaded_model = LoadedModel(
                org_id=org_id,
                model_id=model_id,
                model_path=db_config.model_path,
                sdk_model=sdk_model_proxy,
                encoder=encoder,
                similarity_calculator=similarity_calculator,
                loaded_at=datetime.utcnow(),
                meta_name=db_config.meta_name or model_id,
                short_description=db_config.short_description or "",
                long_description=db_config.long_description or "",
            )
            
            self._models[(org_id, model_id)] = loaded_model
            
            logger.info(
                f"Restored model '{db_config.meta_name}' v{db_config.model_version} "
                f"from DB for org={org_id}, model={model_id}"
            )
            
            return loaded_model
            
        except Exception as e:
            logger.error(
                f"Failed to restore model org={org_id}, model={model_id} from DB: {e}"
            )
            return None
    
    async def list_models(self) -> List[ModelInfoResponse]:
        """List all currently loaded models."""
        models = []
        for (org_id, model_id), loaded_model in self._models.items():
            models.append(ModelInfoResponse(
                org_id=org_id,
                model_id=model_id,
                name=loaded_model.sdk_model.name,
                version=loaded_model.sdk_model.version,
                deployed_at=loaded_model.loaded_at,
                status="Active",
            ))
        return models
    
    async def validate_compatibility(self, sdk_model: Any) -> Dict[str, Any]:
        """Validate model compatibility with current SDK version."""
        errors = []
        sdk_version = await self._get_sdk_version()
        
        if not hasattr(sdk_model, 'version'):
            errors.append("Model missing version attribute")
        
        if not hasattr(sdk_model, 'encoder_config'):
            errors.append("Model missing encoder_config")
        
        if hasattr(sdk_model, 'validate_completeness'):
            validation_errors = sdk_model.validate_completeness()
            errors.extend(validation_errors)
        
        return {
            "compatible": len(errors) == 0,
            "errors": errors,
            "model_version": getattr(sdk_model, 'version', 'unknown'),
            "sdk_version": sdk_version,
        }
    
    async def update_config(
        self,
        org_id: str,
        model_id: str,
        config_update: ModelConfigUpdate,
    ) -> ModelConfigResponse:
        """Update model configuration (weights, beam params) without re-encoding."""
        key = (org_id, model_id)
        if key not in self._models:
            raise ModelNotFoundException(org_id, model_id)
        
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == org_id,
                    ModelConfig.model_id == model_id,
                )
            )
            config = result.scalar_one_or_none()
            
            if config is None:
                raise ModelNotFoundException(org_id, model_id)
            
            if config_update.similarity_weights:
                current_weights = config.similarity_weights or {}
                weights_dict = config_update.similarity_weights.model_dump(exclude_none=True)
                current_weights.update(weights_dict)
                config.similarity_weights = current_weights
            
            if config_update.beam_width is not None:
                config.beam_width = config_update.beam_width
            
            if config_update.max_tree_depth is not None:
                config.max_tree_depth = config_update.max_tree_depth
            
            config.updated_at = datetime.utcnow()
            await session.commit()
            await session.refresh(config)
            
            logger.info(f"Updated config for org={org_id}, model={model_id}")
            
            return ModelConfigResponse(
                org_id=config.org_id,
                model_id=config.model_id,
                similarity_weights=config.similarity_weights,
                beam_width=config.beam_width,
                max_tree_depth=config.max_tree_depth,
                resource_quotas=config.resource_quotas,
                resource_usage=config.resource_usage,
                updated_at=config.updated_at,
            )
    
    async def re_encode_model(
        self,
        org_id: str,
        model_id: str,
        regenerate_edges: bool = True,
        background: bool = True,
    ) -> ReEncodeResponse:
        """Re-encode all glyphs for an org/model with the current encoder."""
        key = (org_id, model_id)
        if key not in self._models:
            raise ModelNotFoundException(org_id, model_id)
        
        loaded_model = self._models[key]
        
        if background:
            job_id = str(uuid4())
            job = ReEncodeJob(job_id, org_id, model_id)
            self._re_encode_jobs[job_id] = job
            
            asyncio.create_task(
                self._run_re_encode(job, loaded_model, regenerate_edges)
            )
            
            return ReEncodeResponse(status="started", job_id=job_id)
        else:
            return await self._run_re_encode_sync(loaded_model, regenerate_edges)

    async def _run_re_encode(
        self,
        job: ReEncodeJob,
        loaded_model: LoadedModel,
        regenerate_edges: bool,
    ) -> None:
        """Background task for re-encoding."""
        try:
            async with loaded_model.lock:
                job.status = "running"
                
                async with self._db_session_factory() as session:
                    result = await session.execute(
                        select(Glyph).where(
                            Glyph.org_id == loaded_model.org_id,
                            Glyph.model_id == loaded_model.model_id,
                        )
                    )
                    glyphs = result.scalars().all()
                    job.glyphs_total = len(glyphs)
                    
                    for i, glyph in enumerate(glyphs):
                        new_embedding = await self._encode_concept(
                            loaded_model.encoder,
                            glyph.concept_text,
                        )
                        
                        await session.execute(
                            update(Glyph)
                            .where(Glyph.id == glyph.id)
                            .values(
                                embedding=new_embedding,
                                updated_at=datetime.utcnow(),
                            )
                        )
                        
                        job.glyphs_processed = i + 1
                        job.progress = (i + 1) / job.glyphs_total
                    
                    await session.commit()
                
                if regenerate_edges:
                    job.edges_regenerated = await self._regenerate_edges(
                        loaded_model.org_id, loaded_model.model_id
                    )
                
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                
                logger.info(
                    f"Re-encode completed for org={loaded_model.org_id}, model={loaded_model.model_id}: "
                    f"{job.glyphs_processed} glyphs, {job.edges_regenerated} edges"
                )
                
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Re-encode failed for org={loaded_model.org_id}, model={loaded_model.model_id}: {e}")
    
    async def _run_re_encode_sync(
        self,
        loaded_model: LoadedModel,
        regenerate_edges: bool,
    ) -> ReEncodeResponse:
        """Synchronous re-encoding."""
        import time
        start_time = time.time()
        
        async with loaded_model.lock:
            glyphs_processed = 0
            
            async with self._db_session_factory() as session:
                result = await session.execute(
                    select(Glyph).where(
                        Glyph.org_id == loaded_model.org_id,
                        Glyph.model_id == loaded_model.model_id,
                    )
                )
                glyphs = result.scalars().all()
                
                for glyph in glyphs:
                    new_embedding = await self._encode_concept(
                        loaded_model.encoder,
                        glyph.concept_text,
                    )
                    
                    await session.execute(
                        update(Glyph)
                        .where(Glyph.id == glyph.id)
                        .values(
                            embedding=new_embedding,
                            updated_at=datetime.utcnow(),
                        )
                    )
                    glyphs_processed += 1
                
                await session.commit()
            
            edges_regenerated = 0
            if regenerate_edges:
                edges_regenerated = await self._regenerate_edges(
                    loaded_model.org_id, loaded_model.model_id
                )
            
            duration_ms = (time.time() - start_time) * 1000
            
            return ReEncodeResponse(
                status="completed",
                glyphs_processed=glyphs_processed,
                edges_regenerated=edges_regenerated,
                duration_ms=duration_ms,
            )
    
    async def _encode_concept(self, encoder: Any, concept_text: str) -> List[float]:
        """Encode a concept text using the SDK encoder."""
        try:
            from glyphh import Concept
            
            concept = Concept(
                name=concept_text,
                attributes={"text": concept_text},
            )
            glyph = encoder.encode(concept)
            return glyph.global_cortex.data.astype(float).tolist()
            
        except Exception as e:
            logger.error(f"Failed to encode concept: {e}")
            raise
    
    async def _encode_concept_with_hierarchy(
        self, 
        encoder: Any, 
        concept_text: str
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Encode a concept and extract hierarchical vectors.
        
        Returns:
            Tuple of (global_cortex, hierarchical_vectors)
            where hierarchical_vectors is a list of dicts with 'level', 'path', 'embedding'
        """
        try:
            from glyphh import Concept
            
            concept = Concept(
                name=concept_text,
                attributes={"text": concept_text},
            )
            glyph = encoder.encode(concept)
            
            global_cortex = glyph.global_cortex.data.astype(float).tolist()
            hierarchical_vectors: List[Dict[str, Any]] = []
            
            # Extract layer-level vectors
            for layer_name, layer in glyph.layers.items():
                hierarchical_vectors.append({
                    "level": "layer",
                    "path": layer_name,
                    "embedding": layer.cortex.data.astype(float).tolist(),
                })
                
                # Extract segment-level vectors
                for segment_name, segment in layer.segments.items():
                    hierarchical_vectors.append({
                        "level": "segment",
                        "path": f"{layer_name}.{segment_name}",
                        "embedding": segment.cortex.data.astype(float).tolist(),
                    })
                    
                    # Extract role-level vectors
                    for role_name, role_vector in segment.roles.items():
                        hierarchical_vectors.append({
                            "level": "role",
                            "path": f"{layer_name}.{segment_name}.{role_name}",
                            "embedding": role_vector.data.astype(float).tolist(),
                        })
            
            return global_cortex, hierarchical_vectors
            
        except Exception as e:
            logger.error(f"Failed to encode concept with hierarchy: {e}")
            raise
    
    async def _regenerate_edges(self, org_id: str, model_id: str) -> int:
        """Regenerate all edges for an org/model."""
        async with self._db_session_factory() as session:
            await session.execute(
                delete(Edge).where(
                    Edge.org_id == org_id,
                    Edge.model_id == model_id,
                )
            )
            await session.commit()
        
        # Edge generation handled by EdgeGeneratorService
        return 0
    
    async def get_re_encode_status(self, job_id: str) -> ReEncodeStatusResponse:
        """Get status of a re-encode job."""
        job = self._re_encode_jobs.get(job_id)
        if job is None:
            raise ValueError(f"Re-encode job not found: {job_id}")
        
        return ReEncodeStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            glyphs_total=job.glyphs_total,
            glyphs_processed=job.glyphs_processed,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
        )
    
    async def clear_model_data(self, org_id: str, model_id: str) -> ClearDataResponse:
        """Clear all glyphs and edges for an org/model, preserving config."""
        key = (org_id, model_id)
        
        # Check if model exists (in memory or DB)
        if key not in self._models:
            # Try to load from DB
            loaded = await self._load_from_db(org_id, model_id)
            if loaded is None:
                # Check if config exists even without loadable encoder
                async with self._db_session_factory() as session:
                    result = await session.execute(
                        select(ModelConfig).where(
                            ModelConfig.org_id == org_id,
                            ModelConfig.model_id == model_id,
                        )
                    )
                    if result.scalar_one_or_none() is None:
                        raise ModelNotFoundException(org_id, model_id)
        
        async with self._db_session_factory() as session:
            edge_result = await session.execute(
                delete(Edge).where(
                    Edge.org_id == org_id,
                    Edge.model_id == model_id,
                )
            )
            edges_deleted = edge_result.rowcount
            
            glyph_result = await session.execute(
                delete(Glyph).where(
                    Glyph.org_id == org_id,
                    Glyph.model_id == model_id,
                )
            )
            glyphs_deleted = glyph_result.rowcount
            
            await session.execute(
                update(ModelConfig)
                .where(
                    ModelConfig.org_id == org_id,
                    ModelConfig.model_id == model_id,
                )
                .values(
                    resource_usage={"memory_mb": 0, "storage_gb": 0, "glyph_count": 0},
                    updated_at=datetime.utcnow(),
                )
            )
            
            await session.commit()
        
        logger.info(
            f"Cleared data org={org_id}, model={model_id}: "
            f"{glyphs_deleted} glyphs, {edges_deleted} edges"
        )
        
        return ClearDataResponse(
            org_id=org_id,
            model_id=model_id,
            glyphs_deleted=glyphs_deleted,
            edges_deleted=edges_deleted,
        )
    
    async def get_config(self, org_id: str, model_id: str) -> ModelConfigResponse:
        """Get current model configuration."""
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == org_id,
                    ModelConfig.model_id == model_id,
                )
            )
            config = result.scalar_one_or_none()
            
            if config is None:
                raise ModelNotFoundException(org_id, model_id)
            
            return ModelConfigResponse(
                org_id=config.org_id,
                model_id=config.model_id,
                similarity_weights=config.similarity_weights,
                beam_width=config.beam_width,
                max_tree_depth=config.max_tree_depth,
                resource_quotas=config.resource_quotas,
                resource_usage=config.resource_usage,
                updated_at=config.updated_at,
            )
    
    async def check_quota(
        self,
        org_id: str,
        model_id: str,
        additional_glyphs: int = 0,
        additional_storage_mb: float = 0,
    ) -> bool:
        """Check if org/model has quota for additional resources."""
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == org_id,
                    ModelConfig.model_id == model_id,
                )
            )
            config = result.scalar_one_or_none()
            
            if config is None:
                raise ModelNotFoundException(org_id, model_id)
            
            quotas = config.resource_quotas or {}
            usage = config.resource_usage or {}
            
            max_glyphs = quotas.get("max_glyphs", 1000000)
            current_glyphs = usage.get("glyph_count", 0)
            if current_glyphs + additional_glyphs > max_glyphs:
                raise QuotaExceededException(
                    org_id=org_id,
                    model_id=model_id,
                    resource="max_glyphs",
                    limit=max_glyphs,
                    current=current_glyphs + additional_glyphs,
                )
            
            max_storage = quotas.get("storage_gb", 10) * 1024
            current_storage = usage.get("storage_mb", 0)
            if current_storage + additional_storage_mb > max_storage:
                raise QuotaExceededException(
                    org_id=org_id,
                    model_id=model_id,
                    resource="storage_gb",
                    limit=quotas.get("storage_gb", 10),
                    current=(current_storage + additional_storage_mb) / 1024,
                )
            
            return True
    
    def is_model_locked(self, org_id: str, model_id: str) -> bool:
        """Check if model is locked (e.g., during re-encode)."""
        key = (org_id, model_id)
        if key not in self._models:
            return False
        return self._models[key].lock.locked()

    async def get_active_config(
        self,
        org_id: str,
        model_id: str,
    ) -> Dict[str, Any]:
        """
        Get the currently active config for a loaded model.
        
        Returns the encoder config as a dictionary for comparison
        with Platform-stored config.
        """
        key = (org_id, model_id)
        loaded_model = self._models.get(key)
        
        if loaded_model is None:
            # Try to restore from DB
            loaded_model = await self._load_from_db(org_id, model_id)
        
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)
        
        # Extract config from loaded model
        config = {}
        if hasattr(loaded_model.sdk_model, 'encoder_config'):
            ec = loaded_model.sdk_model.encoder_config
            if hasattr(ec, 'to_dict'):
                config = ec.to_dict()
            elif hasattr(ec, 'dimension'):
                # Manual extraction for proxy models
                config = {
                    "dimension": getattr(ec, 'dimension', 10000),
                    "seed": getattr(ec, 'seed', 42),
                    "similarity_weight": getattr(ec, 'similarity_weight', 1.0),
                    "security_weight": getattr(ec, 'security_weight', 1.0),
                    "apply_weights_during_encoding": getattr(ec, 'apply_weights_during_encoding', False),
                    "layers": getattr(ec, 'layers', []),
                    "nl_encoder_config": getattr(ec, 'nl_encoder_config', None),
                }
        
        return config

    async def apply_config_update(
        self,
        org_id: str,
        model_id: str,
        new_config: Dict[str, Any],
        change_type: str,
    ) -> Dict[str, Any]:
        """
        Apply config update to a loaded model.
        
        Args:
            org_id: Organization ID
            model_id: Model ID
            new_config: New configuration dict
            change_type: Type of change ("nl_only", "encoder_only", "mixed")
        
        Returns:
            Dict with status, change_type, and optional job_id
        """
        key = (org_id, model_id)
        loaded_model = self._models.get(key)
        
        if loaded_model is None:
            # Try to restore from DB
            loaded_model = await self._load_from_db(org_id, model_id)
        
        if loaded_model is None:
            raise ModelNotFoundException(org_id, model_id)
        
        if change_type == "nl_only":
            # Hot-reload NL config only
            await self._apply_nl_config(loaded_model, new_config)
            return {
                "status": "applied",
                "change_type": "nl_only",
                "message": "NL config hot-reloaded successfully",
            }
        else:
            # Encoder changes require re-encoding
            await self._apply_encoder_config(loaded_model, new_config)
            
            # Start background re-encode job
            job_result = await self.re_encode_model(
                org_id, model_id,
                regenerate_edges=True,
                background=True,
            )
            
            return {
                "status": "re_encoding",
                "change_type": change_type,
                "job_id": job_result.job_id,
                "message": "Encoder updated, re-encoding glyphs in background",
            }

    async def _apply_nl_config(
        self,
        loaded_model: LoadedModel,
        new_config: Dict[str, Any],
    ) -> None:
        """Hot-reload NL encoder config (IntentMatcher patterns)."""
        nl_config = new_config.get("nl_encoder_config")
        
        if nl_config is None:
            logger.info(
                f"No NL config to apply for {loaded_model.org_id}/{loaded_model.model_id}"
            )
            return
        
        # Update the model's NL config
        if hasattr(loaded_model.sdk_model, 'encoder_config'):
            ec = loaded_model.sdk_model.encoder_config
            if hasattr(ec, 'nl_encoder_config'):
                # Direct attribute update
                ec.nl_encoder_config = nl_config
            elif hasattr(ec, '__dict__'):
                # Fallback for proxy models
                ec.__dict__['nl_encoder_config'] = nl_config
        
        # Update DB config
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == loaded_model.org_id,
                    ModelConfig.model_id == loaded_model.model_id,
                )
            )
            db_config = result.scalar_one_or_none()
            
            if db_config and db_config.encoder_config:
                encoder_config_dict = dict(db_config.encoder_config)
                encoder_config_dict['nl_encoder_config'] = nl_config
                
                await session.execute(
                    update(ModelConfig)
                    .where(
                        ModelConfig.org_id == loaded_model.org_id,
                        ModelConfig.model_id == loaded_model.model_id,
                    )
                    .values(
                        encoder_config=encoder_config_dict,
                        updated_at=datetime.utcnow(),
                    )
                )
                await session.commit()
        
        logger.info(
            f"NL config hot-reloaded for {loaded_model.org_id}/{loaded_model.model_id}"
        )

    async def _apply_encoder_config(
        self,
        loaded_model: LoadedModel,
        new_config: Dict[str, Any],
    ) -> None:
        """Apply encoder config changes (requires re-encoding)."""
        adapter = get_sdk_adapter()
        
        # Create new encoder config from the updated config
        try:
            encoder_config = EncoderConfigFactory.create_from_dict(new_config)
            validator = get_config_validator()
            validator.validate_encoder_config_or_raise(encoder_config)
            encoder_config = validator.apply_defaults(encoder_config)
        except ConfigurationError as e:
            raise ModelLoadException(f"Invalid encoder config: {e}")
        
        # Create new encoder with updated config
        try:
            new_encoder = adapter.create_encoder(encoder_config)
        except Exception as e:
            raise ModelLoadException(f"Failed to create encoder: {e}")
        
        # Swap encoder atomically
        old_encoder = loaded_model.encoder
        loaded_model.encoder = new_encoder
        
        # Update SDK model reference
        loaded_model.sdk_model.encoder_config = encoder_config
        
        # Clean up old encoder
        if hasattr(old_encoder, 'clear_cache'):
            old_encoder.clear_cache()
        
        # Update DB config
        encoder_config_dict = encoder_config.to_dict() if hasattr(encoder_config, 'to_dict') else new_config
        
        async with self._db_session_factory() as session:
            await session.execute(
                update(ModelConfig)
                .where(
                    ModelConfig.org_id == loaded_model.org_id,
                    ModelConfig.model_id == loaded_model.model_id,
                )
                .values(
                    encoder_config=encoder_config_dict,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()
        
        logger.info(
            f"Encoder config updated for {loaded_model.org_id}/{loaded_model.model_id}"
        )
