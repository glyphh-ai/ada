"""
Model Manager for Glyphh Runtime.

Handles loading, unloading, and managing multiple .glyphh models with namespace isolation.
Uses SDK's GlyphhModel for model loading only (packaging), and creates separate Encoder
and SimilarityCalculator instances for runtime operations.

Updated to use the new SDK API with explicit EncoderConfig structure.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    NamespaceNotFoundException,
    NamespaceQuotaExceededException,
)
from shared.sdk_adapter import get_sdk_adapter, SDKNotAvailableError
from shared.encoder_config_factory import EncoderConfigFactory, ConfigurationError
from shared.config_validator import get_config_validator

logger = logging.getLogger(__name__)
settings = get_settings()


class LoadedModel:
    """
    In-memory representation of a loaded model.
    
    Updated to follow correct SDK usage pattern:
    - sdk_model: GlyphhModel for metadata only (packaging)
    - encoder: Encoder instance for encoding operations
    - similarity_calculator: SimilarityCalculator for similarity operations
    - Metadata fields for marketplace display
    """
    
    def __init__(
        self,
        namespace: str,
        model_path: str,
        sdk_model: Any,  # GlyphhModel from SDK - for metadata only
        encoder: Any,  # Encoder from SDK - for encoding operations
        similarity_calculator: Optional[Any],  # SimilarityCalculator from SDK
        loaded_at: datetime,
        # Metadata fields for marketplace display
        meta_name: str,
        short_description: str,
        long_description: str,
    ):
        self.namespace = namespace
        self.model_path = model_path
        self.sdk_model = sdk_model  # For metadata access only
        self.encoder = encoder  # For encoding operations
        self.similarity_calculator = similarity_calculator  # For similarity operations
        self.loaded_at = loaded_at
        self.lock = asyncio.Lock()  # For re-encode operations
        # Metadata for marketplace
        self.meta_name = meta_name
        self.short_description = short_description
        self.long_description = long_description


class ReEncodeJob:
    """Tracks background re-encode job status."""
    
    def __init__(self, job_id: str, namespace: str):
        self.job_id = job_id
        self.namespace = namespace
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0.0
        self.glyphs_total = 0
        self.glyphs_processed = 0
        self.edges_regenerated = 0
        self.started_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None


class ModelManager:
    """
    Manages multiple .glyphh models with namespace isolation.
    
    Responsibilities:
    - Load/unload models from .glyphh files
    - Maintain in-memory registry of loaded models
    - Validate model compatibility with SDK version
    - Provide access to SDK components (encoder, etc.)
    - Handle model configuration updates
    - Manage re-encode and clear data operations
    """
    
    def __init__(self, db_session_factory):
        """
        Initialize ModelManager.
        
        Args:
            db_session_factory: Async session factory for database access
        """
        self._models: Dict[str, LoadedModel] = {}
        self._db_session_factory = db_session_factory
        self._re_encode_jobs: Dict[str, ReEncodeJob] = {}
        self._sdk_version: Optional[str] = None
        
    async def _get_sdk_version(self) -> str:
        """Get the SDK version (cached)."""
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
        namespace: Optional[str] = None,
    ) -> LoadedModel:
        """
        Load a .glyphh model and assign it a namespace.
        
        Uses GlyphhModel.from_file() for loading only, then extracts
        encoder_config to create separate Encoder and SimilarityCalculator
        instances for runtime operations.
        
        Args:
            model_path: Path to .glyphh file
            namespace: Optional namespace (generated if not provided)
            
        Returns:
            LoadedModel instance
            
        Raises:
            ModelLoadException: If model fails to load
            ModelIncompatibleException: If model is incompatible with SDK
        """
        # Check local mode model limit
        if settings.deployment_mode == "local":
            if len(self._models) >= settings.local_mode_max_models:
                raise ModelLoadException(
                    f"Local mode limit: maximum {settings.local_mode_max_models} model(s). "
                    f"Upgrade to a production license for unlimited models."
                )
        
        # Get SDK adapter
        adapter = get_sdk_adapter()
        if not adapter.is_available:
            raise ModelLoadException("SDK not available")
        
        # Import GlyphhModel for loading
        try:
            from glyphh import GlyphhModel
        except ImportError as e:
            raise ModelLoadException(f"SDK not available: {e}")
        
        # Validate file exists
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadException(f"Model file not found: {model_path}")
        
        if not path.suffix == ".glyphh":
            raise ModelLoadException(f"Invalid file extension: {path.suffix}")
        
        # Load model from file (GlyphhModel for packaging/loading only)
        try:
            sdk_model = GlyphhModel.from_file(str(path))
        except Exception as e:
            raise ModelLoadException(f"Failed to parse model file: {e}")
        
        # Validate compatibility
        validation_result = await self.validate_compatibility(sdk_model)
        if not validation_result["compatible"]:
            raise ModelIncompatibleException(
                model_version=sdk_model.version,
                sdk_version=await self._get_sdk_version()
            )
        
        # Generate namespace if not provided
        if namespace is None:
            namespace = f"{sdk_model.name}_{uuid4().hex[:8]}"
        
        # If namespace already exists, unload the old model first (re-deploy)
        if namespace in self._models:
            logger.info(f"Re-deploying: unloading existing model from namespace '{namespace}'")
            await self.unload_model(namespace, delete_data=False)
        
        # Extract encoder config from model and validate
        try:
            encoder_config = EncoderConfigFactory.create_from_model(sdk_model)
            
            # Validate and apply defaults
            validator = get_config_validator()
            validator.validate_encoder_config_or_raise(encoder_config)
            encoder_config = validator.apply_defaults(encoder_config)
            
        except ConfigurationError as e:
            raise ModelLoadException(f"Invalid model configuration: {e}")
        
        # Create Encoder instance via SDK adapter (not using GlyphhModel directly)
        try:
            encoder = adapter.create_encoder(encoder_config)
        except Exception as e:
            raise ModelLoadException(f"Failed to create encoder: {e}")
        
        # Create SimilarityCalculator instance via SDK adapter
        similarity_calculator = adapter.create_similarity_calculator()
        if similarity_calculator is None:
            logger.warning(
                f"SimilarityCalculator not available for namespace '{namespace}', "
                f"will use fallback similarity"
            )
        
        # Extract model metadata with defaults for missing fields
        meta_name = getattr(sdk_model, 'meta_name', None) or \
                    getattr(sdk_model, 'name', None) or \
                    path.stem  # Filename without extension as fallback
        short_description = getattr(sdk_model, 'short_description', '') or ''
        long_description = getattr(sdk_model, 'long_description', '') or ''
        
        # Create loaded model with all components
        loaded_model = LoadedModel(
            namespace=namespace,
            model_path=str(path),
            sdk_model=sdk_model,  # For metadata access only
            encoder=encoder,  # For encoding operations
            similarity_calculator=similarity_calculator,  # For similarity operations
            loaded_at=datetime.utcnow(),
            meta_name=meta_name,
            short_description=short_description,
            long_description=long_description,
        )
        
        # Store model config in database (including metadata)
        async with self._db_session_factory() as session:
            config = ModelConfig(
                namespace=namespace,
                model_path=str(path),
                model_version=sdk_model.version,
                sdk_version=await self._get_sdk_version(),
                meta_name=meta_name,
                short_description=short_description,
                long_description=long_description,
            )
            session.add(config)
            await session.commit()
        
        # Add to registry
        self._models[namespace] = loaded_model
        
        logger.info(
            f"Loaded model '{meta_name}' v{sdk_model.version} "
            f"into namespace '{namespace}'"
        )
        
        return loaded_model
    
    async def load_model_from_bytes(
        self,
        content: bytes,
        namespace: str,
    ) -> LoadedModel:
        """
        Load a .glyphh model from raw bytes and assign it a namespace.
        
        Supports two formats:
        1. Gzipped .glyphh files (from SDK's GlyphhModel.to_file)
        2. Plain JSON config from the platform's export_glyphh
        
        For plain JSON, creates an encoder directly from the config
        without requiring a full GlyphhModel with glyphs.
        
        Args:
            content: Raw bytes of the .glyphh file or JSON config
            namespace: Namespace to assign (e.g. '{org_id}/{model_id}')
            
        Returns:
            LoadedModel instance
            
        Raises:
            ModelLoadException: If model fails to load
        """
        import gzip
        import json as json_module
        
        # Try gzipped format first (SDK .glyphh files)
        try:
            decompressed = gzip.decompress(content)
            # It's a gzipped .glyphh — write to temp file and use standard load
            import tempfile
            import os
            
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, f"{namespace.replace('/', '_')}.glyphh")
            
            try:
                with open(tmp_path, "wb") as f:
                    f.write(content)
                return await self.load_model(tmp_path, namespace)
            finally:
                try:
                    os.unlink(tmp_path)
                    os.rmdir(tmp_dir)
                except OSError:
                    pass
        except gzip.BadGzipFile:
            pass  # Not gzipped — try plain JSON config
        
        # Plain JSON config from platform
        try:
            config_data = json_module.loads(content)
        except json_module.JSONDecodeError as e:
            raise ModelLoadException(f"Invalid model data: not gzipped .glyphh and not valid JSON: {e}")
        
        return await self._load_from_platform_config(config_data, namespace)
    
    async def _load_from_platform_config(
        self,
        config_data: dict,
        namespace: str,
    ) -> LoadedModel:
        """
        Load a model from platform JSON config (no glyphs, just encoder config).
        
        This is used when the platform deploys a model that hasn't been
        packaged as a full .glyphh file yet. Creates an encoder from the
        config so the runtime can encode data sent later.
        
        Args:
            config_data: Platform export JSON with name, config, etc.
            namespace: Namespace to assign
            
        Returns:
            LoadedModel instance
        """
        # Check local mode model limit
        if settings.deployment_mode == "local":
            if len(self._models) >= settings.local_mode_max_models:
                raise ModelLoadException(
                    f"Local mode limit: maximum {settings.local_mode_max_models} model(s). "
                    f"Upgrade to a production license for unlimited models."
                )
        
        # Check if namespace already exists
        if namespace in self._models:
            # Unload existing model first for re-deploy
            await self.unload_model(namespace, delete_data=False)
        
        model_name = config_data.get("name", namespace)
        model_config = config_data.get("config", {})
        model_version = str(config_data.get("version", "1"))
        
        # Get SDK adapter
        adapter = get_sdk_adapter()
        if not adapter.is_available:
            raise ModelLoadException("SDK not available")
        
        # Build encoder config from platform config
        try:
            encoder_config = EncoderConfigFactory.create_from_dict(model_config)
            
            validator = get_config_validator()
            validator.validate_encoder_config_or_raise(encoder_config)
            encoder_config = validator.apply_defaults(encoder_config)
        except ConfigurationError as e:
            raise ModelLoadException(f"Invalid model configuration: {e}")
        except Exception as e:
            # If config doesn't map to EncoderConfig, create with defaults
            logger.warning(f"Could not create encoder config from platform config, using defaults: {e}")
            try:
                encoder_config = EncoderConfigFactory.create_default()
                validator = get_config_validator()
                encoder_config = validator.apply_defaults(encoder_config)
            except Exception as e2:
                raise ModelLoadException(f"Failed to create default encoder config: {e2}")
        
        # Create Encoder instance
        try:
            encoder = adapter.create_encoder(encoder_config)
        except Exception as e:
            raise ModelLoadException(f"Failed to create encoder: {e}")
        
        # Create SimilarityCalculator
        similarity_calculator = adapter.create_similarity_calculator()
        if similarity_calculator is None:
            logger.warning(
                f"SimilarityCalculator not available for namespace '{namespace}'"
            )
        
        # Create a minimal sdk_model-like object for metadata
        class PlatformModel:
            """Minimal model object for platform-originated deployments."""
            def __init__(self, name, version, config):
                self.name = name
                self.version = version
                self.encoder_config = config
        
        sdk_model_proxy = PlatformModel(model_name, model_version, encoder_config)
        
        loaded_model = LoadedModel(
            namespace=namespace,
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
            # Check if config already exists
            result = await session.execute(
                select(ModelConfig).where(ModelConfig.namespace == namespace)
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
                    namespace=namespace,
                    model_path="platform-deploy",
                    model_version=model_version,
                    sdk_version=await self._get_sdk_version(),
                    meta_name=model_name,
                    short_description=config_data.get("description", ""),
                    long_description="",
                )
                session.add(config)
            await session.commit()
        
        # Add to registry
        self._models[namespace] = loaded_model
        
        logger.info(
            f"Loaded platform model '{model_name}' v{model_version} "
            f"into namespace '{namespace}'"
        )
        
        return loaded_model
    
    async def unload_model(
        self,
        namespace: str,
        delete_data: bool = False,
    ) -> None:
        """
        Unload a model and optionally clean up its data.
        
        Args:
            namespace: Namespace of model to unload
            delete_data: If True, delete all glyphs and edges
            
        Raises:
            ModelNotFoundException: If model not found
        """
        if namespace not in self._models:
            raise ModelNotFoundException(namespace)
        
        loaded_model = self._models[namespace]
        
        # Delete data if requested
        if delete_data:
            await self.clear_namespace_data(namespace)
            
            # Also delete model config
            async with self._db_session_factory() as session:
                await session.execute(
                    delete(ModelConfig).where(ModelConfig.namespace == namespace)
                )
                await session.commit()
        
        # Remove from registry
        del self._models[namespace]
        
        # Clear encoder cache to release memory
        if hasattr(loaded_model.encoder, 'clear_cache'):
            loaded_model.encoder.clear_cache()
        
        logger.info(f"Unloaded model from namespace '{namespace}'")
    
    async def get_model(self, namespace: str) -> Optional[LoadedModel]:
        """
        Retrieve a loaded model by namespace.
        
        Args:
            namespace: Namespace to look up
            
        Returns:
            LoadedModel if found, None otherwise
        """
        return self._models.get(namespace)
    
    async def list_models(self) -> List[ModelInfoResponse]:
        """
        List all currently loaded models.
        
        Returns:
            List of ModelInfoResponse
        """
        models = []
        for namespace, loaded_model in self._models.items():
            models.append(ModelInfoResponse(
                model_id=namespace,
                name=loaded_model.sdk_model.name,
                version=loaded_model.sdk_model.version,
                deployed_at=loaded_model.loaded_at,
                status="Active",
            ))
        return models
    
    async def validate_compatibility(self, sdk_model: Any) -> Dict[str, Any]:
        """
        Validate model compatibility with current SDK version.
        
        Args:
            sdk_model: GlyphhModel instance
            
        Returns:
            Dict with 'compatible' bool and 'errors' list
        """
        errors = []
        sdk_version = await self._get_sdk_version()
        
        # Check model has required attributes
        if not hasattr(sdk_model, 'version'):
            errors.append("Model missing version attribute")
        
        if not hasattr(sdk_model, 'encoder_config'):
            errors.append("Model missing encoder_config")
        
        # Validate model completeness
        if hasattr(sdk_model, 'validate_completeness'):
            validation_errors = sdk_model.validate_completeness()
            errors.extend(validation_errors)
        
        # Version compatibility check (basic - could be more sophisticated)
        # For now, we accept all versions
        
        return {
            "compatible": len(errors) == 0,
            "errors": errors,
            "model_version": getattr(sdk_model, 'version', 'unknown'),
            "sdk_version": sdk_version,
        }

    
    async def update_config(
        self,
        namespace: str,
        config_update: ModelConfigUpdate,
    ) -> ModelConfigResponse:
        """
        Update model configuration (weights, beam params) without re-encoding.
        
        Args:
            namespace: Namespace of model to update
            config_update: Configuration updates to apply
            
        Returns:
            Updated ModelConfigResponse
            
        Raises:
            ModelNotFoundException: If model not found
        """
        if namespace not in self._models:
            raise ModelNotFoundException(namespace)
        
        async with self._db_session_factory() as session:
            # Get current config
            result = await session.execute(
                select(ModelConfig).where(ModelConfig.namespace == namespace)
            )
            config = result.scalar_one_or_none()
            
            if config is None:
                raise NamespaceNotFoundException(namespace)
            
            # Update similarity weights if provided
            if config_update.similarity_weights:
                current_weights = config.similarity_weights or {}
                weights_dict = config_update.similarity_weights.model_dump(exclude_none=True)
                current_weights.update(weights_dict)
                config.similarity_weights = current_weights
            
            # Update beam width if provided
            if config_update.beam_width is not None:
                config.beam_width = config_update.beam_width
            
            # Update max tree depth if provided
            if config_update.max_tree_depth is not None:
                config.max_tree_depth = config_update.max_tree_depth
            
            config.updated_at = datetime.utcnow()
            await session.commit()
            await session.refresh(config)
            
            logger.info(f"Updated config for namespace '{namespace}'")
            
            return ModelConfigResponse(
                namespace=config.namespace,
                similarity_weights=config.similarity_weights,
                beam_width=config.beam_width,
                max_tree_depth=config.max_tree_depth,
                resource_quotas=config.resource_quotas,
                resource_usage=config.resource_usage,
                updated_at=config.updated_at,
            )
    
    async def re_encode_namespace(
        self,
        namespace: str,
        regenerate_edges: bool = True,
        background: bool = True,
    ) -> ReEncodeResponse:
        """
        Re-encode all glyphs in a namespace with the current encoder.
        
        Args:
            namespace: Namespace to re-encode
            regenerate_edges: Whether to regenerate edges after re-encoding
            background: Whether to run as background job
            
        Returns:
            ReEncodeResponse with job status
            
        Raises:
            ModelNotFoundException: If model not found
        """
        if namespace not in self._models:
            raise ModelNotFoundException(namespace)
        
        loaded_model = self._models[namespace]
        
        if background:
            # Create background job
            job_id = str(uuid4())
            job = ReEncodeJob(job_id, namespace)
            self._re_encode_jobs[job_id] = job
            
            # Start background task
            asyncio.create_task(
                self._run_re_encode(job, loaded_model, regenerate_edges)
            )
            
            return ReEncodeResponse(
                status="started",
                job_id=job_id,
            )
        else:
            # Run synchronously
            result = await self._run_re_encode_sync(
                loaded_model, regenerate_edges
            )
            return result
    
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
                    # Count total glyphs
                    result = await session.execute(
                        select(Glyph).where(Glyph.namespace == loaded_model.namespace)
                    )
                    glyphs = result.scalars().all()
                    job.glyphs_total = len(glyphs)
                    
                    # Re-encode each glyph
                    for i, glyph in enumerate(glyphs):
                        # Re-encode using SDK encoder
                        new_embedding = await self._encode_concept(
                            loaded_model.encoder,
                            glyph.concept_text,
                        )
                        
                        # Update glyph embedding
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
                
                # Regenerate edges if requested
                if regenerate_edges:
                    job.edges_regenerated = await self._regenerate_edges(
                        loaded_model.namespace
                    )
                
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                
                logger.info(
                    f"Re-encode completed for namespace '{loaded_model.namespace}': "
                    f"{job.glyphs_processed} glyphs, {job.edges_regenerated} edges"
                )
                
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Re-encode failed for namespace '{loaded_model.namespace}': {e}")
    
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
                # Get all glyphs
                result = await session.execute(
                    select(Glyph).where(Glyph.namespace == loaded_model.namespace)
                )
                glyphs = result.scalars().all()
                
                # Re-encode each glyph
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
            
            # Regenerate edges if requested
            edges_regenerated = 0
            if regenerate_edges:
                edges_regenerated = await self._regenerate_edges(
                    loaded_model.namespace
                )
            
            duration_ms = (time.time() - start_time) * 1000
            
            return ReEncodeResponse(
                status="completed",
                glyphs_processed=glyphs_processed,
                edges_regenerated=edges_regenerated,
                duration_ms=duration_ms,
            )
    
    async def _encode_concept(self, encoder: Any, concept_text: str) -> List[float]:
        """
        Encode a concept text using the SDK encoder.
        
        Returns the global cortex embedding as a list of floats.
        """
        try:
            from glyphh import Concept
            
            # Create concept from text
            concept = Concept(
                name=concept_text,
                attributes={"text": concept_text},
            )
            
            # Encode to glyph
            glyph = encoder.encode(concept)
            
            # Return global cortex as float list
            return glyph.global_cortex.data.astype(float).tolist()
            
        except Exception as e:
            logger.error(f"Failed to encode concept: {e}")
            raise
    
    async def _regenerate_edges(self, namespace: str) -> int:
        """
        Regenerate all edges for a namespace.
        
        Returns the number of edges regenerated.
        """
        # Delete existing edges
        async with self._db_session_factory() as session:
            await session.execute(
                delete(Edge).where(Edge.namespace == namespace)
            )
            await session.commit()
        
        # Edge generation will be handled by EdgeGeneratorService (Task 8)
        # For now, return 0 as placeholder
        return 0
    
    async def get_re_encode_status(self, job_id: str) -> ReEncodeStatusResponse:
        """
        Get status of a re-encode job.
        
        Args:
            job_id: Job ID to look up
            
        Returns:
            ReEncodeStatusResponse
            
        Raises:
            ValueError: If job not found
        """
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
    
    async def clear_namespace_data(self, namespace: str) -> ClearDataResponse:
        """
        Clear all glyphs and edges in a namespace, preserving model config.
        
        Args:
            namespace: Namespace to clear
            
        Returns:
            ClearDataResponse with deletion counts
            
        Raises:
            ModelNotFoundException: If model not found
        """
        if namespace not in self._models:
            raise ModelNotFoundException(namespace)
        
        async with self._db_session_factory() as session:
            # Delete edges first (foreign key constraint)
            edge_result = await session.execute(
                delete(Edge).where(Edge.namespace == namespace)
            )
            edges_deleted = edge_result.rowcount
            
            # Delete glyphs
            glyph_result = await session.execute(
                delete(Glyph).where(Glyph.namespace == namespace)
            )
            glyphs_deleted = glyph_result.rowcount
            
            # Update resource usage in config
            await session.execute(
                update(ModelConfig)
                .where(ModelConfig.namespace == namespace)
                .values(
                    resource_usage={"memory_mb": 0, "storage_gb": 0, "glyph_count": 0},
                    updated_at=datetime.utcnow(),
                )
            )
            
            await session.commit()
        
        logger.info(
            f"Cleared namespace '{namespace}': "
            f"{glyphs_deleted} glyphs, {edges_deleted} edges"
        )
        
        return ClearDataResponse(
            namespace=namespace,
            glyphs_deleted=glyphs_deleted,
            edges_deleted=edges_deleted,
        )
    
    async def get_config(self, namespace: str) -> ModelConfigResponse:
        """
        Get current model configuration.
        
        Args:
            namespace: Namespace to get config for
            
        Returns:
            ModelConfigResponse
            
        Raises:
            NamespaceNotFoundException: If namespace not found
        """
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(ModelConfig.namespace == namespace)
            )
            config = result.scalar_one_or_none()
            
            if config is None:
                raise NamespaceNotFoundException(namespace)
            
            return ModelConfigResponse(
                namespace=config.namespace,
                similarity_weights=config.similarity_weights,
                beam_width=config.beam_width,
                max_tree_depth=config.max_tree_depth,
                resource_quotas=config.resource_quotas,
                resource_usage=config.resource_usage,
                updated_at=config.updated_at,
            )
    
    async def check_quota(
        self,
        namespace: str,
        additional_glyphs: int = 0,
        additional_storage_mb: float = 0,
    ) -> bool:
        """
        Check if namespace has quota for additional resources.
        
        Args:
            namespace: Namespace to check
            additional_glyphs: Number of glyphs to add
            additional_storage_mb: Storage to add in MB
            
        Returns:
            True if within quota, False otherwise
            
        Raises:
            NamespaceQuotaExceededException: If quota would be exceeded
        """
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(ModelConfig.namespace == namespace)
            )
            config = result.scalar_one_or_none()
            
            if config is None:
                raise NamespaceNotFoundException(namespace)
            
            quotas = config.resource_quotas or {}
            usage = config.resource_usage or {}
            
            # Check glyph count
            max_glyphs = quotas.get("max_glyphs", 1000000)
            current_glyphs = usage.get("glyph_count", 0)
            if current_glyphs + additional_glyphs > max_glyphs:
                raise NamespaceQuotaExceededException(
                    namespace=namespace,
                    resource="max_glyphs",
                    limit=max_glyphs,
                    current=current_glyphs + additional_glyphs,
                )
            
            # Check storage
            max_storage = quotas.get("storage_gb", 10) * 1024  # Convert to MB
            current_storage = usage.get("storage_mb", 0)
            if current_storage + additional_storage_mb > max_storage:
                raise NamespaceQuotaExceededException(
                    namespace=namespace,
                    resource="storage_gb",
                    limit=quotas.get("storage_gb", 10),
                    current=(current_storage + additional_storage_mb) / 1024,
                )
            
            return True
    
    def is_namespace_locked(self, namespace: str) -> bool:
        """Check if namespace is locked (e.g., during re-encode)."""
        if namespace not in self._models:
            return False
        return self._models[namespace].lock.locked()
