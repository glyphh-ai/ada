"""
Model Manager for Glyphh Runtime.

Handles loading, unloading, and managing multiple .glyphh models with namespace isolation.
Uses SDK's GlyphhModel for model loading and maintains an in-memory registry.
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

logger = logging.getLogger(__name__)
settings = get_settings()


class LoadedModel:
    """In-memory representation of a loaded model."""
    
    def __init__(
        self,
        namespace: str,
        model_path: str,
        sdk_model: Any,  # GlyphhModel from SDK
        encoder: Any,  # Encoder from SDK
        loaded_at: datetime,
    ):
        self.namespace = namespace
        self.model_path = model_path
        self.sdk_model = sdk_model
        self.encoder = encoder
        self.loaded_at = loaded_at
        self.lock = asyncio.Lock()  # For re-encode operations


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
        
        Args:
            model_path: Path to .glyphh file
            namespace: Optional namespace (generated if not provided)
            
        Returns:
            LoadedModel instance
            
        Raises:
            ModelLoadException: If model fails to load
            ModelIncompatibleException: If model is incompatible with SDK
        """
        # Import SDK components
        try:
            from glyphh import GlyphhModel, Encoder
        except ImportError as e:
            raise ModelLoadException(f"SDK not available: {e}")
        
        # Validate file exists
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadException(f"Model file not found: {model_path}")
        
        if not path.suffix == ".glyphh":
            raise ModelLoadException(f"Invalid file extension: {path.suffix}")
        
        # Load model from file
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
        
        # Check if namespace already exists
        if namespace in self._models:
            raise ModelLoadException(f"Namespace already in use: {namespace}")
        
        # Create encoder from model config
        try:
            encoder = Encoder(sdk_model.encoder_config)
        except Exception as e:
            raise ModelLoadException(f"Failed to create encoder: {e}")
        
        # Create loaded model
        loaded_model = LoadedModel(
            namespace=namespace,
            model_path=str(path),
            sdk_model=sdk_model,
            encoder=encoder,
            loaded_at=datetime.utcnow(),
        )
        
        # Store model config in database
        async with self._db_session_factory() as session:
            config = ModelConfig(
                namespace=namespace,
                model_path=str(path),
                model_version=sdk_model.version,
                sdk_version=await self._get_sdk_version(),
            )
            session.add(config)
            await session.commit()
        
        # Add to registry
        self._models[namespace] = loaded_model
        
        logger.info(
            f"Loaded model '{sdk_model.name}' v{sdk_model.version} "
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
        
        if not hasattr(sdk_model, 'glyphs'):
            errors.append("Model missing glyphs")
        
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
