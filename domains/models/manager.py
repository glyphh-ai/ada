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
        assess_query_fn: Optional[Any] = None,
        entry_to_record_fn: Optional[Any] = None,
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
        self.assess_query_fn = assess_query_fn
        self.entry_to_record_fn = entry_to_record_fn


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
        
        _enc_fn, _assess_fn, _etr_fn = self._load_model_fns(str(path))
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
            encode_query_fn=_enc_fn,
            assess_query_fn=_assess_fn,
            entry_to_record_fn=_etr_fn,
        )
        
        # Serialize encoder config for DB storage
        encoder_config_dict = None
        if hasattr(encoder_config, 'to_dict'):
            encoder_config_dict = encoder_config.to_dict()

        # Read source files for DB restore (ZIP unpack leaves .py files in parent dir)
        source_files_dict: Optional[Dict[str, str]] = None
        model_dir = path.parent if path.is_file() else path
        if model_dir.is_dir():
            source_files_dict = self._read_source_files(model_dir) or None

        # Embed similarity config from config.yaml into encoder_config for DB persistence
        # (allows default_filter to work even when config.yaml isn't on disk)
        if model_dir.is_dir() and encoder_config_dict is not None:
            try:
                _cfg_path = model_dir / "config.yaml"
                if _cfg_path.exists():
                    import yaml
                    with open(_cfg_path) as _f:
                        _raw = yaml.safe_load(_f) or {}
                    _sim_cfg = _raw.get("similarity")
                    if _sim_cfg:
                        encoder_config_dict["_similarity_config"] = _sim_cfg
            except Exception:
                pass

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
                existing.source_files = source_files_dict
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
                    source_files=source_files_dict,
                )
                session.add(config)
            await session.commit()

            # Sync SDK-bundled stored procedures to the database
            if sdk_model.has_stored_procedures():
                from domains.procedures.service import StoredProcedureService
                from domains.procedures.schemas import StoredProcedureCreate
                procedure_service = StoredProcedureService(session)
                sdk_proc_names = set()
                for proc in sdk_model.stored_procedures:
                    await procedure_service.upsert(
                        org_id=org_id,
                        model_id=model_id,
                        data=StoredProcedureCreate(
                            name=proc.name,
                            gql_query=proc.gql_query,
                            lexicons=proc.lexicons,
                            description=proc.description,
                        ),
                    )
                    sdk_proc_names.add(proc.name)
                # Remove stale procedures no longer in the SDK model
                existing_procs = await procedure_service.list(org_id, model_id)
                for existing_proc in existing_procs:
                    if existing_proc.name not in sdk_proc_names:
                        await procedure_service.delete(org_id, model_id, existing_proc.name)
                logger.info(
                    f"Synced {len(sdk_model.stored_procedures)} stored procedures "
                    f"for org={org_id}, model={model_id}"
                )
        
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

    async def load_model_from_directory(
        self,
        model_dir: Path,
        org_id: str,
        model_id: str,
    ) -> LoadedModel:
        """
        Load a model from an unpacked directory (ZIP-based .glyphh format).

        Uses the directory-based loader (encoder.py / config.yaml / manifest.yaml)
        rather than GlyphhModel.from_file (gzip JSON).

        Args:
            model_dir: Path to unpacked model directory
            org_id: Organization ID
            model_id: Model ID
        """
        from domains.models.loader import load_model as load_model_def

        adapter = get_sdk_adapter()
        if not adapter.is_available:
            raise ModelLoadException("SDK not available")

        if not model_dir.is_dir():
            raise ModelLoadException(f"Model directory not found: {model_dir}")

        loaded = load_model_def(model_dir)

        if loaded.encoder_config is None:
            raise ModelLoadException(
                f"No encoder config found in model directory. "
                f"Ensure encoder.py with ENCODER_CONFIG or config.yaml exists."
            )

        key = (org_id, model_id)
        is_redeploy = key in self._models

        # Check local mode model limit (only for new deploys)
        if not is_redeploy and settings.deployment_mode == "local":
            if len(self._models) >= settings.local_mode_max_models:
                raise ModelLoadException(
                    f"Local mode limit: maximum {settings.local_mode_max_models} model(s). "
                    f"Upgrade to a production license for unlimited models."
                )

        # Validate and create encoder
        try:
            # encoder_config from loader is already an EncoderConfig object
            # (imported from encoder.py's ENCODER_CONFIG)
            encoder_config = loaded.encoder_config
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

        meta_name = loaded.manifest.name or model_id
        short_description = loaded.manifest.description or ""
        manifest_version = loaded.manifest.version or "0.1.0"

        # Build a proxy SDK model object
        class DirectoryModel:
            def __init__(self, name, version, config):
                self.name = name
                self.version = version
                self.encoder_config = config
                self.meta_name = name

            def has_stored_procedures(self):
                return False

        sdk_model_proxy = DirectoryModel(meta_name, manifest_version, encoder_config)

        loaded_model = LoadedModel(
            org_id=org_id,
            model_id=model_id,
            model_path=str(model_dir),
            sdk_model=sdk_model_proxy,
            encoder=encoder,
            similarity_calculator=similarity_calculator,
            loaded_at=datetime.utcnow(),
            meta_name=meta_name,
            short_description=short_description,
            long_description="",
            encode_query_fn=loaded.encode_query_fn,
            assess_query_fn=loaded.assess_query_fn,
            entry_to_record_fn=loaded.entry_to_record_fn,
        )

        # Serialize encoder config for DB storage
        encoder_config_dict = None
        if hasattr(encoder_config, 'to_dict'):
            encoder_config_dict = encoder_config.to_dict()

        # Embed similarity config from config.yaml into encoder_config for DB persistence
        # (allows default_filter to work even when config.yaml isn't on disk)
        try:
            _cfg_path = model_dir / "config.yaml"
            if _cfg_path.exists() and encoder_config_dict is not None:
                import yaml
                with open(_cfg_path) as _f:
                    _raw = yaml.safe_load(_f) or {}
                _sim_cfg = _raw.get("similarity")
                if _sim_cfg:
                    encoder_config_dict["_similarity_config"] = _sim_cfg
        except Exception:
            pass

        # Read source files for DB restore
        source_files_dict = self._read_source_files(model_dir)

        # DB upsert
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == org_id,
                    ModelConfig.model_id == model_id,
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.model_path = str(model_dir)
                existing.model_version = manifest_version
                existing.sdk_version = await self._get_sdk_version()
                existing.meta_name = meta_name
                existing.short_description = short_description
                existing.long_description = ""
                existing.encoder_config = encoder_config_dict
                existing.source_files = source_files_dict or None
                existing.updated_at = datetime.utcnow()
            else:
                config = ModelConfig(
                    org_id=org_id,
                    model_id=model_id,
                    model_path=str(model_dir),
                    model_version=manifest_version,
                    sdk_version=await self._get_sdk_version(),
                    meta_name=meta_name,
                    short_description=short_description,
                    long_description="",
                    encoder_config=encoder_config_dict,
                    source_files=source_files_dict or None,
                )
                session.add(config)
            await session.commit()

        # Swap in-memory model
        old_model = self._models.get(key)
        if old_model is not None:
            logger.info(f"Re-deploying: replacing model org={org_id}, model={model_id}")
            if hasattr(old_model.encoder, 'clear_cache'):
                old_model.encoder.clear_cache()

        self._models[key] = loaded_model

        logger.info(
            f"Loaded model '{meta_name}' v{manifest_version} "
            f"from directory into org={org_id}, model={model_id}"
        )

        # Auto-load exemplars from data/ if present.
        # Stage raw JSONL text to the DB (survives dyno restarts), then
        # kick off a background task to encode.  No large in-memory list.
        data_dir = model_dir / "data"
        if data_dir.exists():
            jsonl_text = self._read_exemplar_jsonl_raw(data_dir)
            if jsonl_text:
                line_count = sum(1 for ln in jsonl_text.split("\n") if ln.strip())
                logger.info(
                    f"Staging {line_count} exemplars for {model_id} to DB"
                )
                async with self._db_session_factory() as session:
                    result = await session.execute(
                        select(ModelConfig).where(
                            ModelConfig.org_id == org_id,
                            ModelConfig.model_id == model_id,
                        )
                    )
                    cfg = result.scalar_one_or_none()
                    if cfg:
                        cfg.staged_exemplars = jsonl_text
                        await session.commit()

                asyncio.create_task(
                    self._process_staged_exemplars(org_id, model_id),
                    name=f"exemplar_load_{org_id}_{model_id}",
                )

        return loaded_model

    @staticmethod
    def _read_exemplar_jsonl_raw(data_dir: Path) -> str:
        """Read exemplar JSONL files from a data directory as raw text.

        Returns concatenated JSONL string (one JSON object per line).
        Only reads files matching exemplars*.jsonl.
        """
        parts: list[str] = []
        for jsonl_file in sorted(data_dir.glob("exemplars*.jsonl")):
            text = jsonl_file.read_text()
            if text:
                parts.append(text.rstrip("\n"))
        return "\n".join(parts) if parts else ""

    async def _process_staged_exemplars(
        self,
        org_id: str,
        model_id: str,
    ) -> None:
        """Encode staged exemplars from the DB and store as glyphs.

        Reads raw JSONL from model_configs.staged_exemplars, encodes in
        batches of BATCH_SIZE with per-batch commits, and NULLs the
        staged column when complete.  Supports resume: if the process
        is interrupted, the staged data remains and will be picked up
        on the next startup or deploy.
        """
        import json as json_mod
        from domains.models.storage import GlyphStorage
        from glyphh.core.types import Concept
        from glyphh.core.ops import bundle
        from domains.listeners.async_service import _extract_hierarchical_vectors

        BATCH_SIZE = 500

        try:
            # Read staged JSONL from DB
            async with self._db_session_factory() as session:
                result = await session.execute(
                    select(ModelConfig).where(
                        ModelConfig.org_id == org_id,
                        ModelConfig.model_id == model_id,
                    )
                )
                cfg = result.scalar_one_or_none()
                if not cfg or not cfg.staged_exemplars:
                    logger.info(f"No staged exemplars for {org_id}/{model_id}")
                    return
                raw_jsonl = cfg.staged_exemplars

            # Parse lines (streaming — one line at a time, not a big list)
            lines = [ln for ln in raw_jsonl.split("\n") if ln.strip()]
            total = len(lines)
            logger.info(f"Processing {total} staged exemplars for {model_id}")

            # Get the loaded model's encoder and entry_to_record_fn
            key = (org_id, model_id)
            loaded_model = self._models.get(key)
            if not loaded_model:
                logger.error(f"Model {model_id} not in memory, cannot encode exemplars")
                return

            encoder = loaded_model.encoder
            entry_to_record_fn = loaded_model.entry_to_record_fn

            # If not available (e.g. model restored from DB after restart),
            # try to reconstruct from stored source_files
            if entry_to_record_fn is None:
                entry_to_record_fn = await self._restore_entry_to_record(org_id, model_id)
                if entry_to_record_fn:
                    loaded_model.entry_to_record_fn = entry_to_record_fn

            # Check existing glyph count for resume
            async with self._db_session_factory() as session:
                storage = GlyphStorage(session)
                existing_count = await storage.count_glyphs(org_id, model_id)

            if existing_count >= total:
                logger.info(
                    f"Exemplars already loaded for {org_id}/{model_id} "
                    f"({existing_count} glyphs), clearing staged data"
                )
                await self._clear_staged_exemplars(org_id, model_id)
                return

            # Check license limits
            from glyphh.licensing import get_current_license
            license_info = get_current_license()
            max_glyphs = license_info.max_glyphs_per_model
            if max_glyphs >= 0 and total > max_glyphs:
                logger.warning(
                    f"Model {model_id} has {total:,} entries but "
                    f"{license_info.tier} tier allows {max_glyphs:,}. "
                    f"Encoding first {max_glyphs:,}."
                )
                lines = lines[:max_glyphs]
                total = len(lines)

            # Resume: skip already-encoded entries
            start_index = existing_count if 0 < existing_count < total else 0
            if existing_count > 0 and start_index == 0:
                async with self._db_session_factory() as session:
                    storage = GlyphStorage(session)
                    logger.info(f"Re-deploying {model_id}: clearing {existing_count} old glyphs")
                    await storage.delete_model_data(org_id, model_id)
            elif start_index > 0:
                logger.info(f"Resuming exemplar load for {model_id}: {start_index}/{total}")

            created = start_index

            for batch_start in range(start_index, total, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total)

                async with self._db_session_factory() as session:
                    storage = GlyphStorage(session)

                    for i in range(batch_start, batch_end):
                        try:
                            entry = json_mod.loads(lines[i])

                            if entry_to_record_fn:
                                record = entry_to_record_fn(entry)
                                concept_text = record["concept_text"]
                                metadata = record["metadata"]
                                attrs = record["attributes"]
                            else:
                                concept_text = entry.get("question", entry.get("text", ""))
                                metadata = {k: v for k, v in entry.items() if k != "question"}
                                attrs = {"text": concept_text}

                            concept = Concept(
                                name=f"entry_{created}",
                                attributes=attrs,
                                metadata=metadata,
                            )
                            glyph = encoder.encode(concept)

                            non_temporal = [
                                layer.cortex.data
                                for name, layer in glyph.layers.items()
                                if name != "_temporal"
                                and hasattr(layer, "cortex")
                                and layer.cortex is not None
                            ]
                            if non_temporal:
                                embedding = bundle(non_temporal).astype(float).tolist()
                            else:
                                embedding = glyph.global_cortex.data.astype(float).tolist()

                            glyph_response = await storage.create_glyph(
                                org_id=org_id,
                                model_id=model_id,
                                concept_text=concept_text,
                                embedding=embedding,
                                metadata={**metadata, "record_type": "pattern"},
                            )

                            hierarchical = _extract_hierarchical_vectors(glyph)
                            if hierarchical:
                                await storage.create_glyph_vectors_batch(
                                    glyph_id=glyph_response.glyph_id,
                                    org_id=org_id,
                                    model_id=model_id,
                                    vectors=hierarchical,
                                )

                            created += 1

                        except Exception as e:
                            logger.warning(f"Failed to encode entry {i} in {model_id}: {e}")
                            continue

                    await session.commit()

                logger.info(f"Encoded {created}/{total} exemplars for {model_id}")

            # Done — clear staged data
            await self._clear_staged_exemplars(org_id, model_id)

            logger.info(
                f"Background exemplar load complete for {org_id}/{model_id}: "
                f"{created} glyphs from {total} entries"
            )

        except Exception as e:
            logger.error(
                f"Background exemplar load failed for {model_id}: {e}",
                exc_info=True,
            )

    async def _clear_staged_exemplars(self, org_id: str, model_id: str) -> None:
        """NULL out the staged_exemplars column after encoding is complete."""
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == org_id,
                    ModelConfig.model_id == model_id,
                )
            )
            cfg = result.scalar_one_or_none()
            if cfg:
                cfg.staged_exemplars = None
                await session.commit()

    async def _restore_entry_to_record(self, org_id: str, model_id: str) -> Optional[Any]:
        """Try to restore entry_to_record from the model's stored source_files.

        Called when the model is in memory but entry_to_record_fn was not
        loaded (e.g. old deploy before this field was added).  Reads
        source_files from DB and evals encoder.py to get the function.
        """
        try:
            async with self._db_session_factory() as session:
                result = await session.execute(
                    select(ModelConfig).where(
                        ModelConfig.org_id == org_id,
                        ModelConfig.model_id == model_id,
                    )
                )
                cfg = result.scalar_one_or_none()
                if cfg and cfg.source_files:
                    _, _, entry_to_record_fn = self._load_model_fns_from_source(
                        cfg.source_files
                    )
                    return entry_to_record_fn
        except Exception as e:
            logger.debug(f"Could not restore entry_to_record for {model_id}: {e}")
        return None

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
        stored encoder_config JSONB. Also attempts to load the custom
        encode_query_fn from the model directory on disk (if it still
        exists), so that query encoding works the same as after a fresh
        startup via register_model_encoders.
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
            
            # Try to load model fns: stored source first, then disk fallback
            encode_query_fn = None
            assess_query_fn = None
            entry_to_record_fn = None
            if db_config.source_files:
                encode_query_fn, assess_query_fn, entry_to_record_fn = (
                    self._load_model_fns_from_source(db_config.source_files)
                )
            if encode_query_fn is None and db_config.model_path:
                encode_query_fn, assess_query_fn, entry_to_record_fn = (
                    self._load_model_fns(db_config.model_path)
                )
            
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
                encode_query_fn=encode_query_fn,
                assess_query_fn=assess_query_fn,
                entry_to_record_fn=entry_to_record_fn,
            )

            self._models[(org_id, model_id)] = loaded_model

            logger.info(
                f"Restored model '{db_config.meta_name}' v{db_config.model_version} "
                f"from DB for org={org_id}, model={model_id}"
                f"{' (with custom encode_query_fn)' if encode_query_fn else ''}"
            )

            # Check for staged exemplars that need processing (e.g. after dyno restart)
            if db_config.staged_exemplars:
                logger.info(
                    f"Found staged exemplars for {model_id}, resuming encoding"
                )
                asyncio.create_task(
                    self._process_staged_exemplars(org_id, model_id),
                    name=f"exemplar_load_{org_id}_{model_id}",
                )

            return loaded_model
            
        except Exception as e:
            logger.error(
                f"Failed to restore model org={org_id}, model={model_id} from DB: {e}"
            )
            return None
    
    @staticmethod
    def _load_model_fns(model_path: str) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Load encode_query_fn, assess_query_fn, entry_to_record_fn from encoder.py.

        Used by load_model and _load_from_db to restore custom query functions when
        lazy-loading a model. Returns (None, None, None) silently if the file doesn't
        exist or has no matching functions.
        """
        try:
            from domains.models.loader import load_encoder_config
            model_dir = Path(model_path)
            # model_path may be a .glyphh file — resolve to its parent directory
            if model_dir.is_file():
                model_dir = model_dir.parent
            if not model_dir.is_dir():
                return None, None, None
            _, _, encode_query_fn, entry_to_record_fn, assess_query_fn = load_encoder_config(model_dir)
            return encode_query_fn, assess_query_fn, entry_to_record_fn
        except Exception as e:
            logger.debug(f"Could not load model fns from {model_path}: {e}")
            return None, None, None

    @staticmethod
    def _load_encode_query_fn(model_path: str) -> Optional[Any]:
        """Backward-compat wrapper — returns only encode_query_fn."""
        encode_query_fn, _, _ = ModelManager._load_model_fns(model_path)
        return encode_query_fn

    @staticmethod
    def _read_source_files(model_dir: Path) -> Dict[str, str]:
        """Read all .py source files + config.yaml from a model directory for DB storage.

        Returns a dict mapping filename to source text, e.g.
        {"encoder.py": "...", "intent.py": "...", "config.yaml": "..."}.
        """
        sources: Dict[str, str] = {}
        if not model_dir.is_dir():
            return sources
        for py_file in sorted(model_dir.glob("*.py")):
            if py_file.name.startswith("."):
                continue
            try:
                sources[py_file.name] = py_file.read_text()
            except Exception:
                pass
        # Also store config.yaml and gql.json for runtime settings
        for extra in ("config.yaml", "gql.json"):
            extra_file = model_dir / extra
            if extra_file.exists():
                try:
                    sources[extra] = extra_file.read_text()
                except Exception:
                    pass
        return sources

    @staticmethod
    def _load_model_fns_from_source(
        source_files: Dict[str, str],
    ) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Load model functions from stored source code.

        Creates temporary module objects so local imports work
        (e.g., encoder.py can ``from intent import extract_keywords``).
        Returns (encode_query_fn, assess_query_fn, entry_to_record_fn)
        or (None, None, None) on failure.
        """
        import sys
        import types

        if not source_files or "encoder.py" not in source_files:
            return None, None, None

        created_modules: list[str] = []
        try:
            # Load all non-encoder .py files first as importable modules.
            # Skip build.py — it's a build-time script that imports from
            # encoder.py (circular at this stage) and is never needed at runtime.
            for filename, source in sorted(source_files.items()):
                if filename in ("encoder.py", "build.py") or not filename.endswith(".py"):
                    continue
                mod_name = filename[:-3]  # "intent.py" → "intent"
                mod = types.ModuleType(mod_name)
                mod.__file__ = f"<db:{filename}>"
                try:
                    exec(compile(source, f"<db:{filename}>", "exec"), mod.__dict__)
                except Exception as mod_err:
                    logger.debug(f"Skipping {filename} during source restore: {mod_err}")
                    continue
                sys.modules[mod_name] = mod
                created_modules.append(mod_name)

            # Now load encoder.py — it can import from the modules above
            encoder_mod = types.ModuleType("model_encoder_db")
            encoder_mod.__file__ = "<db:encoder.py>"
            exec(
                compile(source_files["encoder.py"], "<db:encoder.py>", "exec"),
                encoder_mod.__dict__,
            )

            encode_query_fn = getattr(encoder_mod, "encode_query", None)
            assess_query_fn = getattr(encoder_mod, "assess_query", None)
            entry_to_record_fn = getattr(encoder_mod, "entry_to_record", None)
            return encode_query_fn, assess_query_fn, entry_to_record_fn

        except Exception as e:
            logger.warning(f"Failed to load model fns from stored source: {e}")
            return None, None, None
        finally:
            # Clean up temporary modules from sys.modules
            for mod_name in created_modules:
                sys.modules.pop(mod_name, None)

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
        """Hot-reload NL encoder config."""
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
