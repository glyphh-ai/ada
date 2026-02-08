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
        
        # Re-deploy: unload existing first
        if key in self._models:
            logger.info(f"Re-deploying: unloading existing model org={org_id}, model={model_id}")
            await self.unload_model(org_id, model_id, delete_data=False)
        
        # Check local mode model limit (after re-deploy unload)
        if settings.deployment_mode == "local":
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
        
        # Store model config in database
        async with self._db_session_factory() as session:
            config = ModelConfig(
                org_id=org_id,
                model_id=model_id,
                model_path=str(path),
                model_version=sdk_model.version,
                sdk_version=await self._get_sdk_version(),
                meta_name=meta_name,
                short_description=short_description,
                long_description=long_description,
            )
            session.add(config)
            await session.commit()
        
        self._models[key] = loaded_model
        
        logger.info(
            f"Loaded model '{meta_name}' v{sdk_model.version} "
            f"into org={org_id}, model={model_id}"
        )
        
        return loaded_model

    async def load_model_from_bytes(
        self,
        content: bytes,
        org_id: str,
        model_id: str,
    ) -> LoadedModel:
        """
        Load a .glyphh model from raw bytes.
        
        Supports gzipped .glyphh files and plain JSON config from platform.
        """
        import gzip
        import json as json_module
        
        # Try gzipped format first
        try:
            decompressed = gzip.decompress(content)
            import tempfile
            import os
            
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, f"{org_id}_{model_id}.glyphh")
            
            try:
                with open(tmp_path, "wb") as f:
                    f.write(content)
                return await self.load_model(tmp_path, org_id, model_id)
            finally:
                try:
                    os.unlink(tmp_path)
                    os.rmdir(tmp_dir)
                except OSError:
                    pass
        except gzip.BadGzipFile:
            pass
        
        # Plain JSON config from platform
        try:
            config_data = json_module.loads(content)
        except json_module.JSONDecodeError as e:
            raise ModelLoadException(f"Invalid model data: not gzipped .glyphh and not valid JSON: {e}")
        
        return await self._load_from_platform_config(config_data, org_id, model_id)
    
    async def _load_from_platform_config(
        self,
        config_data: dict,
        org_id: str,
        model_id: str,
    ) -> LoadedModel:
        """Load a model from platform JSON config (no glyphs, just encoder config)."""
        key = (org_id, model_id)
        
        # Re-deploy: unload existing first
        if key in self._models:
            logger.info(f"Re-deploying: unloading existing model org={org_id}, model={model_id}")
            await self.unload_model(org_id, model_id, delete_data=False)
        
        # Check local mode model limit
        if settings.deployment_mode == "local":
            if len(self._models) >= settings.local_mode_max_models:
                raise ModelLoadException(
                    f"Local mode limit: maximum {settings.local_mode_max_models} model(s). "
                    f"Upgrade to a production license for unlimited models."
                )
        
        model_name = config_data.get("name", model_id)
        model_config = config_data.get("config", {})
        model_version = str(config_data.get("version", "1"))
        
        adapter = get_sdk_adapter()
        if not adapter.is_available:
            raise ModelLoadException("SDK not available")
        
        try:
            encoder_config = EncoderConfigFactory.create_from_dict(model_config)
            validator = get_config_validator()
            validator.validate_encoder_config_or_raise(encoder_config)
            encoder_config = validator.apply_defaults(encoder_config)
        except ConfigurationError as e:
            raise ModelLoadException(f"Invalid model configuration: {e}")
        except Exception as e:
            raise ModelLoadException(f"Failed to create encoder config from platform config: {e}")
        
        try:
            encoder = adapter.create_encoder(encoder_config)
        except Exception as e:
            raise ModelLoadException(f"Failed to create encoder: {e}")
        
        similarity_calculator = adapter.create_similarity_calculator()
        if similarity_calculator is None:
            logger.warning(f"SimilarityCalculator not available for org={org_id}, model={model_id}")
        
        class PlatformModel:
            """Minimal model object for platform-originated deployments."""
            def __init__(self, name, version, config):
                self.name = name
                self.version = version
                self.encoder_config = config
        
        sdk_model_proxy = PlatformModel(model_name, model_version, encoder_config)
        
        loaded_model = LoadedModel(
            org_id=org_id,
            model_id=model_id,
            model_path="platform-deploy",
            sdk_model=sdk_model_proxy,
            encoder=encoder,
            similarity_calculator=similarity_calculator,
            loaded_at=datetime.utcnow(),
            meta_name=model_name,
            short_description=config_data.get("description", ""),
            long_description="",
        )
        
        # Store model config in database
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == org_id,
                    ModelConfig.model_id == model_id,
                )
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                existing.model_version = model_version
                existing.sdk_version = await self._get_sdk_version()
                existing.meta_name = model_name
                existing.short_description = config_data.get("description", "")
                existing.updated_at = datetime.utcnow()
            else:
                config = ModelConfig(
                    org_id=org_id,
                    model_id=model_id,
                    model_path="platform-deploy",
                    model_version=model_version,
                    sdk_version=await self._get_sdk_version(),
                    meta_name=model_name,
                    short_description=config_data.get("description", ""),
                    long_description="",
                )
                session.add(config)
            await session.commit()
        
        self._models[key] = loaded_model
        
        logger.info(
            f"Loaded platform model '{model_name}' v{model_version} "
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
        
        if key not in self._models:
            raise ModelNotFoundException(org_id, model_id)
        
        loaded_model = self._models[key]
        
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
        
        del self._models[key]
        
        if hasattr(loaded_model.encoder, 'clear_cache'):
            loaded_model.encoder.clear_cache()
        
        logger.info(f"Unloaded model org={org_id}, model={model_id}")
    
    async def get_model(self, org_id: str, model_id: str) -> Optional[LoadedModel]:
        """Retrieve a loaded model by (org_id, model_id)."""
        return self._models.get((org_id, model_id))
    
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
        if key not in self._models:
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
