"""
Model Catalog API — lists models available on this runtime instance.

The runtime ships with curated models in models/. Enterprise users
can also place custom models in custom_models/. This endpoint exposes
what's available for deployment.

GET /catalog/models — list all available models (core + custom)
GET /catalog/models/{model_id} — get details for a specific model
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/catalog", tags=["catalog"])

# Model directories relative to runtime root
_RUNTIME_ROOT = Path(__file__).parent.parent.parent
_CORE_MODELS_DIR = _RUNTIME_ROOT / "models"
_CUSTOM_MODELS_DIR = _RUNTIME_ROOT / "custom_models"


class CatalogModel(BaseModel):
    """A model available in the catalog."""
    model_id: str
    name: str
    description: str = ""
    version: str = ""
    category: str = ""
    source: str = "core"  # "core" or "custom"
    has_config: bool = False
    has_model_file: bool = False
    exemplar_count: int = 0


class CatalogResponse(BaseModel):
    """Response from the catalog endpoint."""
    models: List[CatalogModel]
    total: int
    runtime_version: str = ""


def _scan_model_dir(base_dir: Path, source: str) -> List[CatalogModel]:
    """Scan a directory for model subdirectories with .glyphh files."""
    models = []
    if not base_dir.exists():
        return models

    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue

        model_id = model_dir.name
        config_path = model_dir / "config.yaml"
        glyphh_files = list(model_dir.glob("*.glyphh"))

        # Read config if available
        config = {}
        if config_path.exists():
            try:
                config = yaml.safe_load(config_path.read_text()) or {}
            except Exception as e:
                logger.warning(f"Failed to read config for {model_id}: {e}")

        # Count exemplars from data/ directory
        data_dir = model_dir / "data"
        exemplar_count = 0
        if data_dir.exists():
            for jsonl_file in data_dir.glob("*.jsonl"):
                try:
                    exemplar_count += sum(1 for _ in jsonl_file.open())
                except Exception:
                    pass

        models.append(CatalogModel(
            model_id=model_id,
            name=config.get("name", model_id),
            description=config.get("description", ""),
            version=config.get("version", ""),
            category=config.get("category", ""),
            source=source,
            has_config=config_path.exists(),
            has_model_file=len(glyphh_files) > 0,
            exemplar_count=exemplar_count,
        ))

    return models


@router.get("/models", response_model=CatalogResponse)
async def list_catalog_models() -> CatalogResponse:
    """List all models available on this runtime instance.

    Scans models/ (core, shipped with runtime) and custom_models/
    (enterprise, user-created) directories.
    """
    from main import app

    core_models = _scan_model_dir(_CORE_MODELS_DIR, "core")
    custom_models = _scan_model_dir(_CUSTOM_MODELS_DIR, "custom")
    all_models = core_models + custom_models

    return CatalogResponse(
        models=all_models,
        total=len(all_models),
        runtime_version=app.version,
    )


@router.get("/models/{model_id}", response_model=CatalogModel)
async def get_catalog_model(model_id: str) -> CatalogModel:
    """Get details for a specific model in the catalog."""
    # Check core first, then custom
    for base_dir, source in [(_CORE_MODELS_DIR, "core"), (_CUSTOM_MODELS_DIR, "custom")]:
        model_dir = base_dir / model_id
        if model_dir.exists() and model_dir.is_dir():
            models = _scan_model_dir(base_dir, source)
            for m in models:
                if m.model_id == model_id:
                    return m

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in catalog")
