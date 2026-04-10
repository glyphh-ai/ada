"""
Model Manager for Glyphh Runtime.

Manages loading and querying .glyphh models with (org_id, model_id) isolation.
Creates separate Encoder and SimilarityCalculator instances for runtime operations.

All registry keys are (org_id, model_id) tuples.
"""

import asyncio
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, text

from domains.models.db_models import ModelConfig
from domains.models.schemas import ModelConfigResponse
from infrastructure.config import get_settings
from shared.exceptions import (
    ModelLoadException,
    ModelNotFoundException,
)
from shared.sdk_adapter import get_sdk_adapter
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
        mcp_tools: Optional[List] = None,
        handle_mcp_tool_fn: Optional[Any] = None,
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
        self.mcp_tools: List = mcp_tools or []
        self.handle_mcp_tool_fn = handle_mcp_tool_fn
        self.deploy_dir: Optional[Path] = None  # Materialized temp dir (cleaned on unload)


class ModelManager:
    """
    Manages multiple .glyphh models with (org_id, model_id) isolation.
    
    Registry keys are Tuple[str, str] of (org_id, model_id).
    """
    
    def __init__(self, db_session_factory):
        self._models: Dict[Tuple[str, str], LoadedModel] = {}
        self._db_session_factory = db_session_factory
        self._sdk_version: Optional[str] = None
        self._encoding_in_progress: set = set()  # (org_id, model_id) keys with active encoding tasks
        
    async def _get_sdk_version(self) -> str:
        if self._sdk_version is None:
            try:
                import glyphh
                self._sdk_version = glyphh.__version__
            except ImportError:
                self._sdk_version = "unknown"
        return self._sdk_version
    
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

        try:
            loaded = load_model_def(model_dir)
        except ModelLoadException:
            raise
        except Exception as e:
            raise ModelLoadException(
                f"Failed to load model: {e}"
            ) from e

        if loaded.encoder_config is None:
            raise ModelLoadException(
                f"No encoder config found in model directory. "
                f"Ensure encoder.py with ENCODER_CONFIG or config.yaml exists."
            )

        key = (org_id, model_id)
        is_redeploy = key in self._models

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
            mcp_tools=loaded.mcp_tools,
            handle_mcp_tool_fn=loaded.handle_mcp_tool_fn,
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

                self._encoding_in_progress.add((org_id, model_id))
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
        # Advisory lock ID derived from model_id — prevents duplicate encoding
        # across Heroku worker processes (WEB_CONCURRENCY > 1).
        # The lock session must stay open for the entire encoding duration.
        lock_id = hash(f"exemplar_load_{org_id}_{model_id}") & 0x7FFFFFFF

        try:
            # Open a dedicated session that holds the advisory lock for the
            # entire encoding run. Other workers will skip immediately.
            # Advisory locks are PostgreSQL-only — skip on SQLite.
            async with self._db_session_factory() as lock_session:
                _engine_url = str(lock_session.bind.url) if lock_session.bind else ""
                if "postgresql" in _engine_url:
                    lock_result = await lock_session.execute(
                        text(f"SELECT pg_try_advisory_lock({lock_id})")
                    )
                    if not lock_result.scalar():
                        logger.info(f"Another worker is already encoding {model_id}, skipping")
                        return

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

                lines = [ln for ln in raw_jsonl.split("\n") if ln.strip()]
                total = len(lines)
                del raw_jsonl
                logger.info(f"Processing {total} staged exemplars for {model_id}")

                # Get the loaded model's encoder and entry_to_record_fn
                key = (org_id, model_id)
                loaded_model = self._models.get(key)
                if not loaded_model:
                    logger.error(f"Model {model_id} not in memory, cannot encode exemplars")
                    return

                encoder = loaded_model.encoder
                entry_to_record_fn = loaded_model.entry_to_record_fn

                if entry_to_record_fn is None:
                    entry_to_record_fn = await self._restore_entry_to_record(org_id, model_id)
                    if entry_to_record_fn:
                        loaded_model.entry_to_record_fn = entry_to_record_fn

                # Read model config to check store_hierarchical_vectors
                store_hierarchical = False
                try:
                    model_path = Path(loaded_model.model_path) if loaded_model.model_path else None
                    if model_path:
                        config_dir = model_path.parent if model_path.is_file() else model_path
                        _cfg_path = config_dir / "config.yaml"
                        if _cfg_path.exists():
                            import yaml as _yaml
                            _raw = _yaml.safe_load(_cfg_path.read_text()) or {}
                            _sim = _raw.get("similarity") or {}
                            store_hierarchical = bool(_sim.get("store_hierarchical_vectors", False))
                    # Fallback: check DB-persisted encoder_config._similarity_config
                    if not store_hierarchical:
                        async with self._db_session_factory() as _cfg_sess:
                            _cfg_row = (await _cfg_sess.execute(
                                select(ModelConfig).where(
                                    ModelConfig.org_id == org_id,
                                    ModelConfig.model_id == model_id,
                                )
                            )).scalar_one_or_none()
                            if _cfg_row and _cfg_row.encoder_config:
                                _scfg = _cfg_row.encoder_config.get("_similarity_config") or {}
                                store_hierarchical = bool(_scfg.get("store_hierarchical_vectors", False))
                except Exception:
                    pass
                if store_hierarchical:
                    logger.info(f"Hierarchical vectors enabled for {model_id}")

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

                                if store_hierarchical:
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
                    await asyncio.sleep(0)

                # Done — clear staged data
                await self._clear_staged_exemplars(org_id, model_id)

                logger.info(
                    f"Background exemplar load complete for {org_id}/{model_id}: "
                    f"{created} glyphs from {total} entries"
                )

                # Lock released when lock_session closes

        except Exception as e:
            logger.error(
                f"Background exemplar load failed for {model_id}: {e}",
                exc_info=True,
            )
        finally:
            self._encoding_in_progress.discard((org_id, model_id))

    async def resume_staged_encoding(self) -> None:
        """Resume any incomplete exemplar encoding from a previous run.

        Called once from lifespan startup — scans model_configs for rows with
        non-null staged_exemplars and kicks off one background task per model.
        """
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig.org_id, ModelConfig.model_id).where(
                    ModelConfig.staged_exemplars.isnot(None),
                )
            )
            pending = result.all()

        if not pending:
            return

        for org_id, model_id in pending:
            key = (org_id, model_id)
            if key in self._encoding_in_progress:
                continue

            # Ensure model is loaded in memory first
            loaded = self._models.get(key)
            if not loaded:
                loaded = await self._load_from_db(org_id, model_id)
            if not loaded:
                logger.warning(f"Cannot resume encoding for {model_id}: model not loadable")
                continue

            logger.info(f"Resuming staged encoding for {org_id}/{model_id}")
            self._encoding_in_progress.add(key)
            asyncio.create_task(
                self._process_staged_exemplars(org_id, model_id),
                name=f"exemplar_load_{org_id}_{model_id}",
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
                    _, _, entry_to_record_fn, _, _, _tmp = self._load_model_fns_from_source(
                        cfg.source_files, model_id=model_id
                    )
                    # Clean up any materialized dir — we only needed the function
                    if _tmp:
                        shutil.rmtree(_tmp, ignore_errors=True)
                    return entry_to_record_fn
        except Exception as e:
            logger.debug(f"Could not restore entry_to_record for {model_id}: {e}")
        return None

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
            mcp_tools: list = []
            handle_mcp_tool_fn = None
            deploy_dir = None
            if db_config.source_files:
                encode_query_fn, assess_query_fn, entry_to_record_fn, mcp_tools, handle_mcp_tool_fn, deploy_dir = (
                    self._load_model_fns_from_source(db_config.source_files, model_id=model_id)
                )
            if encode_query_fn is None and db_config.model_path:
                _eq, _aq, _etr, _mcp, _hmcp = self._load_model_fns(db_config.model_path)
                encode_query_fn = _eq
                assess_query_fn = _aq or assess_query_fn
                entry_to_record_fn = _etr or entry_to_record_fn
                mcp_tools = _mcp or mcp_tools
                handle_mcp_tool_fn = _hmcp or handle_mcp_tool_fn
            
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
                mcp_tools=mcp_tools,
                handle_mcp_tool_fn=handle_mcp_tool_fn,
            )
            loaded_model.deploy_dir = deploy_dir

            self._models[(org_id, model_id)] = loaded_model

            logger.info(
                f"Restored model '{db_config.meta_name}' v{db_config.model_version} "
                f"from DB for org={org_id}, model={model_id}"
                f"{' (with custom encode_query_fn)' if encode_query_fn else ''}"
            )

            # Staged exemplars are resumed from lifespan startup via
            # resume_staged_encoding(), not from the query path.

            return loaded_model
            
        except Exception as e:
            logger.error(
                f"Failed to restore model org={org_id}, model={model_id} from DB: {e}"
            )
            return None
    
    @staticmethod
    def _load_model_fns(model_path: str) -> tuple[Optional[Any], Optional[Any], Optional[Any], list, Optional[Any]]:
        """Load model functions from encoder.py.

        Returns (encode_query_fn, assess_query_fn, entry_to_record_fn, mcp_tools, handle_mcp_tool_fn).
        Used by load_model and _load_from_db to restore custom query functions when
        lazy-loading a model. Returns (None, None, None, [], None) silently if the file
        doesn't exist or has no matching functions.
        """
        try:
            from domains.models.loader import load_encoder_config
            model_dir = Path(model_path)
            # model_path may be a .glyphh file — resolve to its parent directory
            if model_dir.is_file():
                model_dir = model_dir.parent
            if not model_dir.is_dir():
                return None, None, None, [], None
            _, _, encode_query_fn, entry_to_record_fn, assess_query_fn, mcp_tools, handle_mcp_tool_fn = load_encoder_config(model_dir)
            return encode_query_fn, assess_query_fn, entry_to_record_fn, mcp_tools, handle_mcp_tool_fn
        except Exception as e:
            logger.debug(f"Could not load model fns from {model_path}: {e}")
            return None, None, None, [], None

    # Directories excluded from source capture (mirrors packaging.py blocklist)
    _SOURCE_EXCLUDE_DIRS = {
        "__pycache__", ".git", ".pytest_cache", ".venv", "venv",
        "tests", "results", ".mypy_cache", ".ruff_cache", "node_modules",
    }

    # Root-level files excluded from source capture (build-time only)
    _SOURCE_EXCLUDE_ROOT_FILES = {
        "build.py", "discover.py", "tests.py", "test.py",
        "gap_analysis.py", "run_bfcl.py",
    }

    # File basenames excluded at any depth (build/test artifacts, not runtime)
    _SOURCE_EXCLUDE_FILENAMES = {
        "tests.jsonl", "test_queries.jsonl",
    }

    # File extensions to capture (text files needed at runtime)
    _SOURCE_INCLUDE_EXTENSIONS = {
        ".py", ".yaml", ".yml", ".json", ".jsonl",
    }

    @staticmethod
    def _read_source_files(model_dir: Path) -> Dict[str, str]:
        """Recursively read all source files from a model directory for DB storage.

        Walks the entire model directory tree and captures .py, .yaml, .json,
        .jsonl files. Keys are relative paths preserving subdirectory structure:
          {"encoder.py": "...", "apps/slack_v2/intent.py": "...", "data/exemplars.jsonl": "..."}

        Excludes build artifacts and build-time scripts (same blocklist as packaging.py).
        """
        sources: Dict[str, str] = {}
        if not model_dir.is_dir():
            return sources

        for child in sorted(model_dir.rglob("*")):
            if not child.is_file():
                continue

            rel = child.relative_to(model_dir)
            parts = rel.parts

            # Skip files in excluded directories
            if any(p in ModelManager._SOURCE_EXCLUDE_DIRS or p.startswith(".") for p in parts[:-1]):
                continue

            # Skip hidden files
            if rel.name.startswith("."):
                continue

            # Skip excluded root-level files
            if len(parts) == 1 and rel.name in ModelManager._SOURCE_EXCLUDE_ROOT_FILES:
                continue

            # Skip .glyphh files (packages)
            if rel.suffix == ".glyphh":
                continue

            # Skip build/test files at any depth
            if rel.name in ModelManager._SOURCE_EXCLUDE_FILENAMES:
                continue

            # Only capture known text file extensions
            if rel.suffix not in ModelManager._SOURCE_INCLUDE_EXTENSIONS:
                continue

            try:
                sources[str(rel)] = child.read_text()
            except Exception:
                pass

        return sources

    @staticmethod
    def _has_subdirectory_files(source_files: Dict[str, str]) -> bool:
        """Check if source_files contains subdirectory keys (paths with '/')."""
        return any("/" in key for key in source_files)

    @staticmethod
    def _materialize_source_files(source_files: Dict[str, str], model_id: str) -> Path:
        """Write source_files to a temp directory preserving structure.

        Returns the temp directory path. Caller is responsible for cleanup
        (tracked via LoadedModel.deploy_dir).
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"glyphh_model_{model_id}_"))
        for rel_path, content in source_files.items():
            dest = tmp_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        return tmp_dir

    @staticmethod
    def _load_model_fns_from_source(
        source_files: Dict[str, str],
        model_id: str = "",
        model_path: str = "",
    ) -> tuple[Optional[Any], Optional[Any], Optional[Any], list, Optional[Any], Optional[Path]]:
        """Load model functions from stored source code.

        For complex models with subdirectory files (apps/, data/, classes/),
        materializes to a temp directory and loads via importlib so that
        Path(__file__).parent resolves correctly for data file access.

        For simple models (flat root-only files), uses the existing in-memory
        exec() approach with namespaced modules.

        Returns (encode_query_fn, assess_query_fn, entry_to_record_fn, mcp_tools, handle_mcp_tool_fn, deploy_dir)
        where deploy_dir is a Path to the materialized temp directory (or None
        for simple models). Caller must track deploy_dir for cleanup on unload.
        """
        import re as _re
        import sys
        import types

        if not source_files or "encoder.py" not in source_files:
            return None, None, None, [], None, None

        # Complex models with subdirectory files: materialize and load from filesystem
        if ModelManager._has_subdirectory_files(source_files):
            return ModelManager._load_model_fns_materialized(source_files, model_id)

        # --- Simple models: in-memory exec() with namespaced modules ---

        # Build the namespace prefix and the set of local module names
        safe_id = _re.sub(r"[^a-zA-Z0-9_]", "_", model_id) if model_id else "default"
        prefix = f"_glm_{safe_id}"
        local_names: set[str] = set()
        for filename in source_files:
            if filename.endswith(".py") and filename not in ("build.py",):
                local_names.add(filename[:-3])  # "intent.py" → "intent"

        def _rewrite_imports(source: str) -> str:
            """Rewrite bare local imports to use the namespaced prefix."""
            lines = source.split("\n")
            result = []
            for line in lines:
                # "from <local> import ..."
                m = _re.match(r"^(\s*)from\s+(" + "|".join(_re.escape(n) for n in local_names) + r")\s+import\s+(.+)", line)
                if m:
                    indent, mod, rest = m.group(1), m.group(2), m.group(3)
                    result.append(f"{indent}from {prefix}.{mod} import {rest}")
                    continue
                # "import <local>"
                m2 = _re.match(r"^(\s*)import\s+(" + "|".join(_re.escape(n) for n in local_names) + r")\s*$", line)
                if m2:
                    indent, mod = m2.group(1), m2.group(2)
                    result.append(f"{indent}import {prefix}.{mod} as {mod}")
                    continue
                result.append(line)
            return "\n".join(result)

        # Create the namespace package
        pkg = types.ModuleType(prefix)
        pkg.__path__ = []
        pkg.__package__ = prefix
        sys.modules[prefix] = pkg

        # Resolve real filesystem path for __file__
        _real_model_dir = None
        if model_path:
            _p = Path(model_path)
            if _p.is_file():
                _p = _p.parent
            if _p.is_dir():
                _real_model_dir = _p

        def _load_module(filename: str, source: str) -> bool:
            """Load a single source file as a namespaced module."""
            mod_name = filename[:-3]
            fqn = f"{prefix}.{mod_name}"
            mod = types.ModuleType(fqn)
            if _real_model_dir and (_real_model_dir / filename).exists():
                mod.__file__ = str(_real_model_dir / filename)
            else:
                mod.__file__ = f"<db:{model_id}/{filename}>"
            mod.__package__ = prefix
            try:
                rewritten = _rewrite_imports(source)
                exec(compile(rewritten, mod.__file__, "exec"), mod.__dict__)
            except Exception as mod_err:
                logger.debug(f"Deferring {filename} during source restore: {mod_err}")
                return False
            sys.modules[fqn] = mod
            setattr(pkg, mod_name, mod)
            return True

        try:
            skip = {"build.py", "tests.py", "test.py"}
            py_files = {
                fn: src for fn, src in source_files.items()
                if fn.endswith(".py") and fn not in skip
            }

            deferred: dict[str, str] = {}

            # Pass 1: load non-encoder files
            for filename, source in sorted(py_files.items()):
                if filename == "encoder.py":
                    continue
                if not _load_module(filename, source):
                    deferred[filename] = source

            # Load encoder.py
            enc_fqn = f"{prefix}.encoder"
            encoder_mod = types.ModuleType(enc_fqn)
            encoder_mod.__file__ = f"<db:{model_id}/encoder.py>"
            encoder_mod.__package__ = prefix
            rewritten = _rewrite_imports(py_files["encoder.py"])
            exec(compile(rewritten, encoder_mod.__file__, "exec"), encoder_mod.__dict__)
            sys.modules[enc_fqn] = encoder_mod
            setattr(pkg, "encoder", encoder_mod)

            # Pass 2: retry deferred files
            for filename, source in deferred.items():
                if not _load_module(filename, source):
                    logger.debug(f"Skipping {filename} after retry: still fails")

            encode_query_fn = getattr(encoder_mod, "encode_query", None)
            assess_query_fn = getattr(encoder_mod, "assess_query", None)
            entry_to_record_fn = getattr(encoder_mod, "entry_to_record", None)
            mcp_tools = getattr(encoder_mod, "MCP_TOOLS", []) or []
            handle_mcp_tool_fn = getattr(encoder_mod, "handle_mcp_tool", None)
            return encode_query_fn, assess_query_fn, entry_to_record_fn, mcp_tools, handle_mcp_tool_fn, None

        except Exception as e:
            logger.warning(f"Failed to load model fns from stored source: {e}")
            return None, None, None, [], None, None

    @staticmethod
    def _load_model_fns_materialized(
        source_files: Dict[str, str],
        model_id: str,
    ) -> tuple[Optional[Any], Optional[Any], Optional[Any], list, Optional[Any], Optional[Path]]:
        """Load model functions by materializing source files to a temp directory.

        Used for complex models with subdirectories (apps/, data/, classes/).
        Writes all source_files to disk, then imports encoder.py via importlib
        so Path(__file__).parent resolves to the real temp directory.

        Returns (encode_query_fn, assess_query_fn, entry_to_record_fn, mcp_tools, handle_mcp_tool_fn, deploy_dir).
        """
        try:
            tmp_dir = ModelManager._materialize_source_files(source_files, model_id)
            logger.info(f"Materialized {len(source_files)} source files for {model_id} -> {tmp_dir}")

            from domains.models.loader import load_encoder_config
            _, _, encode_query_fn, entry_to_record_fn, assess_query_fn, mcp_tools, handle_mcp_tool_fn = load_encoder_config(tmp_dir)

            if encode_query_fn is None:
                logger.warning(f"Materialized restore for {model_id}: no encode_query_fn found")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return None, None, None, [], None, None

            return encode_query_fn, assess_query_fn, entry_to_record_fn, mcp_tools, handle_mcp_tool_fn, tmp_dir

        except Exception as e:
            logger.warning(f"Failed materialized restore for {model_id}: {e}")
            return None, None, None, [], None, None

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
