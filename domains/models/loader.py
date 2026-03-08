"""
Model loader — discovers and loads models from the filesystem.

Handles the standard model directory structure:
  models/<model_id>/
  ├── manifest.yaml      # identity metadata
  ├── config.yaml        # encoder config (fallback)
  ├── encoder.py         # custom encoder (optional, takes priority)
  ├── <model_id>.glyphh  # compiled artifact
  ├── build.py           # build script
  └── data/              # training data

If encoder.py exists, imports ENCODER_CONFIG from it.
If not, builds encoder config from config.yaml.
No special cases — every model is the same shape.
"""

from __future__ import annotations

import importlib.util
import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelManifest:
    """Model identity and catalog metadata."""
    model_id: str
    name: str
    description: str = ""
    version: str = ""
    author: str = "Glyphh AI"
    license: str = ""
    icon: str = ""
    category: str = ""
    public: bool = True
    load_on_startup: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class LoadedModel:
    """A fully resolved model ready for the runtime."""
    model_id: str
    manifest: ModelManifest
    model_dir: Path
    glyphh_path: Optional[Path] = None
    encoder_config: Any = None
    has_custom_encoder: bool = False
    has_build_script: bool = False
    encode_query_fn: Any = None  # callable(query: str) -> Concept
    entry_to_record_fn: Any = None  # callable(entry: dict) -> dict
    assess_query_fn: Any = None  # callable(query: str) -> dict (completeness check)
    source: str = "core"  # "core" or "custom"
    exemplar_count: int = 0


def load_manifest(model_dir: Path) -> ModelManifest:
    """Load manifest.yaml from a model directory."""
    manifest_path = model_dir / "manifest.yaml"
    model_id = model_dir.name

    if not manifest_path.exists():
        # Fall back to config.yaml for legacy models
        config_path = model_dir / "config.yaml"
        if config_path.exists():
            try:
                config = yaml.safe_load(config_path.read_text()) or {}
                return ModelManifest(
                    model_id=model_id,
                    name=config.get("name", model_id),
                    description=config.get("description", ""),
                    version=config.get("version", ""),
                    author=config.get("author", "Glyphh AI"),
                    category=config.get("category", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to read config for {model_id}: {e}")
        return ModelManifest(model_id=model_id, name=model_id)

    try:
        data = yaml.safe_load(manifest_path.read_text()) or {}
        return ModelManifest(
            model_id=model_id,
            name=data.get("name", model_id),
            description=data.get("description", ""),
            version=data.get("version", ""),
            author=data.get("author", "Glyphh AI"),
            license=data.get("license", ""),
            icon=data.get("icon", ""),
            category=data.get("category", ""),
            public=data.get("public", True),
            load_on_startup=data.get("load_on_startup", False),
            tags=data.get("tags", []),
        )
    except Exception as e:
        logger.warning(f"Failed to read manifest for {model_id}: {e}")
        return ModelManifest(model_id=model_id, name=model_id)


def load_encoder_config(model_dir: Path) -> tuple[Any, bool, Any, Any, Any]:
    """Load encoder config from a model directory.

    Returns (encoder_config, is_custom, encode_query_fn, entry_to_record_fn, assess_query_fn).
    If encoder.py exists, imports ENCODER_CONFIG and optional functions.
    Otherwise returns (None, False, None, None, None).

    Each model's sibling .py modules (intent.py, scorer.py, etc.) are loaded
    into isolated sys.modules entries to prevent cross-model contamination when
    multiple models have identically-named files.
    """
    encoder_path = model_dir / "encoder.py"
    if not encoder_path.exists():
        return None, False, None, None, None

    import sys

    try:
        # Identify bare module names that exist as sibling .py files.
        # These must be isolated per-model to prevent cross-contamination
        # (e.g., toolrouter/intent.py vs pipedream/intent.py).
        local_names = {
            p.stem for p in model_dir.glob("*.py")
            if not p.name.startswith(".")
        }

        # Save any existing sys.modules entries for these names so we can
        # restore them after loading (don't clobber other models' modules).
        saved_modules = {}
        for name in local_names:
            if name in sys.modules:
                saved_modules[name] = sys.modules.pop(name)

        model_dir_str = str(model_dir)
        added_to_path = False
        if model_dir_str not in sys.path:
            sys.path.insert(0, model_dir_str)
            added_to_path = True

        try:
            spec = importlib.util.spec_from_file_location(
                f"model_encoder_{model_dir.name}", encoder_path
            )
            if spec is None or spec.loader is None:
                return None, False, None, None, None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            encoder_config = getattr(module, "ENCODER_CONFIG", None)
            if encoder_config is None:
                logger.warning(f"encoder.py in {model_dir.name} has no ENCODER_CONFIG")
                return None, False, None, None, None

            encode_query_fn = getattr(module, "encode_query", None)
            entry_to_record_fn = getattr(module, "entry_to_record", None)
            assess_query_fn = getattr(module, "assess_query", None)

            logger.info(f"Loaded custom encoder for {model_dir.name}")
            return encoder_config, True, encode_query_fn, entry_to_record_fn, assess_query_fn

        finally:
            if added_to_path:
                sys.path.remove(model_dir_str)
            # Clean up bare module names that were loaded as side effects
            # (e.g., `import intent` cached in sys.modules). The functions
            # we extracted still hold references to their module's globals,
            # so they keep working even after the sys.modules entry is removed.
            for name in local_names:
                sys.modules.pop(name, None)
            # Restore any previously-existing modules we saved
            sys.modules.update(saved_modules)

    except Exception as e:
        logger.warning(f"Failed to load encoder.py for {model_dir.name}: {e}")
        return None, False, None, None, None


def count_exemplars(model_dir: Path) -> int:
    """Count training exemplars in data/ directory."""
    data_dir = model_dir / "data"
    if not data_dir.exists():
        return 0
    count = 0
    for jsonl_file in data_dir.glob("*.jsonl"):
        try:
            count += sum(1 for line in jsonl_file.open() if line.strip())
        except Exception:
            pass
    return count


def load_model(model_dir: Path, source: str = "core") -> LoadedModel:
    """Load a model from its directory."""
    manifest = load_manifest(model_dir)
    encoder_config, has_custom, encode_query_fn, entry_to_record_fn, assess_query_fn = load_encoder_config(model_dir)
    glyphh_files = list(model_dir.glob("*.glyphh"))

    return LoadedModel(
        model_id=manifest.model_id,
        manifest=manifest,
        model_dir=model_dir,
        glyphh_path=glyphh_files[0] if glyphh_files else None,
        encoder_config=encoder_config,
        has_custom_encoder=has_custom,
        has_build_script=(model_dir / "build.py").exists(),
        encode_query_fn=encode_query_fn,
        entry_to_record_fn=entry_to_record_fn,
        assess_query_fn=assess_query_fn,
        source=source,
        exemplar_count=count_exemplars(model_dir),
    )


def discover_models(
    core_dir: Path,
    custom_dir: Optional[Path] = None,
) -> list[LoadedModel]:
    """Discover all models in core and custom directories."""
    models = []

    for base_dir, source in [(core_dir, "core"), (custom_dir, "custom")]:
        if base_dir is None or not base_dir.exists():
            continue
        for model_dir in sorted(base_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            try:
                models.append(load_model(model_dir, source))
            except Exception as e:
                logger.warning(f"Failed to load model {model_dir.name}: {e}")

    return models
