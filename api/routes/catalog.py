"""
Model Catalog API — lists models available on this runtime instance.

The runtime ships with curated models in models/. Enterprise users
can also place custom models in custom_models/. This endpoint exposes
what's available for deployment.

GET /catalog/models — list all public models (hub-visible)
GET /catalog/models?include_private=true — list all models
GET /catalog/models/{model_id} — get details for a specific model
"""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from domains.models import discover_models, LoadedModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/catalog", tags=["catalog"])

_RUNTIME_ROOT = Path(__file__).parent.parent.parent
_CORE_MODELS_DIR = _RUNTIME_ROOT / "models"
_CUSTOM_MODELS_DIR = _RUNTIME_ROOT / "custom_models"


class CatalogModel(BaseModel):
    """A model available in the catalog."""
    model_id: str
    name: str
    description: str = ""
    version: str = ""
    author: str = "Glyphh AI"
    icon: str = ""
    category: str = ""
    public: bool = True
    tags: List[str] = []
    source: str = "core"
    has_custom_encoder: bool = False
    has_build_script: bool = False
    has_model_file: bool = False
    exemplar_count: int = 0


class CatalogResponse(BaseModel):
    """Response from the catalog endpoint."""
    models: List[CatalogModel]
    total: int
    runtime_version: str = ""


def _to_catalog_model(m: LoadedModel) -> CatalogModel:
    return CatalogModel(
        model_id=m.model_id,
        name=m.manifest.name,
        description=m.manifest.description,
        version=m.manifest.version,
        author=m.manifest.author,
        icon=m.manifest.icon,
        category=m.manifest.category,
        public=m.manifest.public,
        tags=m.manifest.tags,
        source=m.source,
        has_custom_encoder=m.has_custom_encoder,
        has_build_script=m.has_build_script,
        has_model_file=m.glyphh_path is not None,
        exemplar_count=m.exemplar_count,
    )


@router.get("/models", response_model=CatalogResponse)
async def list_catalog_models(
    include_private: bool = Query(False, description="Include private/internal models"),
) -> CatalogResponse:
    """List models available on this runtime.

    By default only public models are returned (hub-visible).
    Pass include_private=true to see all models including internal ones.
    """
    from main import app

    all_models = discover_models(_CORE_MODELS_DIR, _CUSTOM_MODELS_DIR)

    if not include_private:
        all_models = [m for m in all_models if m.manifest.public]

    catalog = [_to_catalog_model(m) for m in all_models]

    return CatalogResponse(
        models=catalog,
        total=len(catalog),
        runtime_version=app.version,
    )


@router.get("/models/{model_id}", response_model=CatalogModel)
async def get_catalog_model(model_id: str) -> CatalogModel:
    """Get details for a specific model."""
    all_models = discover_models(_CORE_MODELS_DIR, _CUSTOM_MODELS_DIR)

    for m in all_models:
        if m.model_id == model_id:
            return _to_catalog_model(m)

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in catalog")
