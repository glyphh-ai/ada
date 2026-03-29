"""
Org-Scoped API Routes for Glyphh Runtime (Cloud Mode).

Endpoints scoped by org_id and model_id for multi-tenant cloud deployments.
URL pattern: /{org_id}/{model_id}/...

No namespace concept — org_id and model_id are passed directly to services.
"""

import logging
import shutil
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from domains.query.service import QueryService
from infrastructure.config import get_settings
from shared.auth import AuthenticatedUser, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}", tags=["org-scoped"])
org_router = APIRouter(prefix="/{org_id}", tags=["org-scoped"])
settings = get_settings()


# Dependency injection
async def validate_org_access(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """Validate that the authenticated user belongs to the requested org."""
    if current_user.org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail="Organization mismatch - you don't have access to this organization"
        )

    return current_user





@org_router.get("/models", response_model=dict)
async def list_models(
    org_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """List all deployed models for an org."""
    from glyphh.server import model_manager
    from infrastructure.database import async_session_maker
    from domains.models.db_models import ModelConfig
    from domains.models.storage import GlyphStorage
    from sqlalchemy import select

    if current_user.org_id != org_id:
        raise HTTPException(status_code=403, detail="Organization mismatch")

    # Query all model configs from DB for this org
    async with async_session_maker() as session:
        result = await session.execute(
            select(ModelConfig).where(ModelConfig.org_id == org_id)
        )
        configs = result.scalars().all()

        models = []
        storage = GlyphStorage(session)
        for cfg in configs:
            count = await storage.count_glyphs(org_id, cfg.model_id)
            models.append({
                "model_id": cfg.model_id,
                "name": cfg.meta_name or cfg.model_id,
                "version": cfg.model_version or "—",
                "glyphs": count,
                "status": "active",
            })

    return {"models": models}


class UndeployRequest(BaseModel):
    delete_data: bool = False


async def get_model_manager():
    """Get the ModelManager instance from glyphh.server."""
    from glyphh.server import model_manager

    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return model_manager




# ── File Query Endpoint ──

@router.post("/query/file")
async def query_with_file(
    org_id: str,
    model_id: str,
    file: UploadFile = File(...),
    top_k: int = Form(default=10),
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """
    Query a model using a file upload (multipart/form-data).

    The file is saved to a temporary path and passed to the model's
    encode_query() function as the query string. The model decides
    what to do with it — e.g., Iris extracts visual features from
    images, another model could parse PDFs or audio.

    Returns similarity search results against the deployed model's data.
    """
    from domains.models.schemas import SimilaritySearchRequest
    from infrastructure.database import async_session_maker

    model_manager = await get_model_manager()

    # Preserve original extension so encode_query can detect file type
    original_name = file.filename or "upload"
    _, ext = os.path.splitext(original_name)
    suffix = ext if ext else ""

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        content = await file.read()
        os.write(tmp_fd, content)
        os.close(tmp_fd)

        query_service = QueryService(model_manager, async_session_maker)
        request = SimilaritySearchRequest(
            query=tmp_path,
            top_k=min(max(top_k, 1), 100),
        )
        fact_tree = await query_service.similarity_search(
            org_id=org_id,
            model_id=model_id,
            request=request,
        )
        return fact_tree.to_json() if hasattr(fact_tree, 'to_json') else fact_tree
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Model Lifecycle Endpoints

@router.get("/ready")
async def readiness_check(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """Check if a model is deployed and ready to serve queries."""
    from glyphh.server import model_manager

    if model_manager is None:
        return {"ready": False, "status": "model_manager_not_initialized", "model_id": model_id}

    loaded_model = await model_manager.get_model(org_id, model_id)
    if loaded_model is None:
        return {"ready": False, "status": "not_deployed", "model_id": model_id}

    is_locked = model_manager.is_model_locked(org_id, model_id)
    if is_locked:
        return {"ready": False, "status": "locked", "model_id": model_id}

    return {
        "ready": True,
        "status": "ready",
        "model_id": model_id,
        "meta_name": loaded_model.meta_name,
    }


@router.post("/model/deploy")
async def deploy_model(
    org_id: str,
    model_id: str,
    file: Optional[UploadFile] = File(None),
    model_path: Optional[str] = None,
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """
    Deploy a .glyphh model.

    Accepts either:
    - A multipart file upload of a .glyphh file (field name: file)
    - A form field model_path pointing to a filesystem path

    Supports two .glyphh formats:
    - ZIP archive (from CLI packaging): unpacked and loaded via directory path
    - Gzip JSON (from SDK GlyphhModel.to_file): loaded via GlyphhModel.from_file
    """
    from shared.exceptions import ModelLoadException, ModelIncompatibleException

    resolved_path: Optional[str] = None
    tmp_path: Optional[str] = None
    tmp_dir: Optional[str] = None

    try:
        if file is not None:
            # Multipart file upload — save to temp file
            suffix = ".glyphh"
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            try:
                content = await file.read()
                os.write(tmp_fd, content)
            finally:
                os.close(tmp_fd)
            resolved_path = tmp_path
        elif model_path is not None:
            resolved_path = model_path
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a .glyphh file upload or a model_path.",
            )

        # Detect file format by magic bytes
        with open(resolved_path, "rb") as f:
            magic = f.read(2)

        if magic == b"PK":
            # ZIP format (CLI packaging) — unpack and load from directory
            from glyphh.cli.packaging import unpack_model

            tmp_dir = tempfile.mkdtemp(prefix="glyphh_deploy_")
            model_dir = unpack_model(Path(resolved_path), dest=Path(tmp_dir) / model_id)

            loaded_model = await model_manager.load_model_from_directory(
                model_dir=model_dir,
                org_id=org_id,
                model_id=model_id,
            )
        elif magic == b"\x1f\x8b":
            # Gzip JSON format (SDK GlyphhModel.to_file)
            loaded_model = await model_manager.load_model(
                model_path=resolved_path,
                org_id=org_id,
                model_id=model_id,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unrecognized .glyphh file format (magic bytes: {magic!r}). "
                       f"Expected ZIP (CLI package) or gzip (SDK model).",
            )

        return {
            "status": "deployed",
            "model_id": model_id,
            "version": getattr(loaded_model.sdk_model, "version", "unknown"),
            "meta_name": loaded_model.meta_name,
        }

    except ModelLoadException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelIncompatibleException as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp file if we created one
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)




@router.post("/model/undeploy")
async def undeploy_model(
    org_id: str,
    model_id: str,
    request: UndeployRequest = UndeployRequest(),
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """Unload a model from memory, optionally deleting its data."""
    from shared.exceptions import ModelNotFoundException

    try:
        await model_manager.unload_model(
            org_id=org_id,
            model_id=model_id,
            delete_data=request.delete_data,
        )
        return {"status": "undeployed", "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")


@router.post("/model/re-encode")
async def re_encode_model(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """Re-encode all glyphs for a deployed model."""
    from shared.exceptions import ModelNotFoundException

    # Check if model is loaded
    loaded_model = await model_manager.get_model(org_id, model_id)
    if loaded_model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")

    # Check if re-encode is already in progress
    if model_manager.is_model_locked(org_id, model_id):
        raise HTTPException(status_code=409, detail="Re-encode already in progress for this model")

    try:
        result = await model_manager.re_encode_model(
            org_id=org_id,
            model_id=model_id,
        )
        return {"job_id": result.job_id, "status": result.status}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")


@router.delete("/model")
async def delete_model(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """Full delete: unload model + purge all data (config, glyphs, vectors, edges, procedures)."""
    from shared.exceptions import ModelNotFoundException
    from infrastructure.database import async_session_maker
    from sqlalchemy import delete as sql_delete
    from domains.procedures.models import StoredProcedureModel

    # Check if model exists
    loaded_model = await model_manager.get_model(org_id, model_id)
    if loaded_model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")

    # Check if re-encode is in progress
    if model_manager.is_model_locked(org_id, model_id):
        raise HTTPException(status_code=409, detail="Cannot delete model while re-encode is in progress")

    try:
        # Unload model and delete config + glyph data
        await model_manager.unload_model(
            org_id=org_id,
            model_id=model_id,
            delete_data=True,
        )

        # Purge stored procedures
        async with async_session_maker() as session:
            await session.execute(
                sql_delete(StoredProcedureModel).where(
                    StoredProcedureModel.org_id == org_id,
                    StoredProcedureModel.model_id == model_id,
                )
            )
            await session.commit()

        return {"status": "deleted", "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")


# ── Data Management Endpoints ──

@router.get("/data")
async def list_data(
    org_id: str,
    model_id: str,
    limit: int = 20,
    offset: int = 0,
    stage: str = "auto",
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """List glyphs stored for this model with pagination.

    stage param controls filtering:
      auto     — all glyphs (patterns + data)
      patterns — only exemplar patterns (record_type == 'pattern')
      data     — only user-loaded data (record_type != 'pattern')
    """
    from infrastructure.database import async_session_maker
    from domains.models.storage import GlyphStorage
    from domains.models.db_models import Glyph
    from sqlalchemy import select, func

    async with async_session_maker() as session:
        # Fetch raw SQLAlchemy objects (not Pydantic) to access embedding + glyph_metadata
        result = await session.execute(
            select(Glyph).where(
                Glyph.org_id == org_id, Glyph.model_id == model_id
            ).order_by(Glyph.created_at.desc())
        )
        all_glyphs = result.scalars().all()

        # Filter by stage
        if stage == "patterns":
            glyphs = [g for g in all_glyphs if _is_pattern_glyph(g)]
        elif stage == "data":
            glyphs = [g for g in all_glyphs if not _is_pattern_glyph(g)]
        else:
            glyphs = all_glyphs

        total = len(glyphs)
        page = glyphs[offset:offset + limit]

        glyph_list = [
            {
                "id": str(g.id),
                "concept_text": (g.concept_text or str(g.id))[:200],
                "node_type": (g.glyph_metadata or {}).get("node_type", ""),
                "record_type": (g.glyph_metadata or {}).get("record_type", "data"),
                "has_embedding": g.embedding is not None,
                "vector_dim": len(g.embedding) if g.embedding else 0,
                "created_at": g.created_at.isoformat() + "Z" if g.created_at else None,
            }
            for g in page
        ]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "glyphs": glyph_list,
    }


@router.get("/data/count")
async def count_data(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """Count glyphs and edges for this model."""
    from infrastructure.database import async_session_maker
    from domains.models.storage import GlyphStorage

    async with async_session_maker() as session:
        storage = GlyphStorage(session)
        glyph_count = await storage.count_glyphs(org_id, model_id)
        vector_count = await storage.count_glyph_vectors(org_id, model_id)

    return {
        "glyphs": glyph_count,
        "vectors": vector_count,
        "model_id": model_id,
    }


@router.delete("/data")
async def clear_data(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """Clear all glyphs and edges for this model without unloading it."""
    from infrastructure.database import async_session_maker
    from domains.models.storage import GlyphStorage

    async with async_session_maker() as session:
        storage = GlyphStorage(session)
        glyphs_deleted, edges_deleted = await storage.delete_model_data(org_id, model_id)
        await session.commit()

    return {
        "status": "cleared",
        "model_id": model_id,
        "glyphs_deleted": glyphs_deleted,
        "edges_deleted": edges_deleted,
    }



def _is_pattern_glyph(glyph) -> bool:
    """Check if glyph is a deployed exemplar pattern (not user-loaded data)."""
    meta = glyph.glyph_metadata or {}
    return meta.get("record_type") == "pattern"

