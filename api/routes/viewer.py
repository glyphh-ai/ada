from __future__ import annotations

from fastapi import Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from ..core.db import get_db
from ..services.auth_runtime import enforce_model_access, require_scopes
from ..services.viewer_builder import build_viewer_payload
from .router import api_router


router = api_router(tags=["viewer"])


@router.get("/api/viewer/data/{model_id}")
def viewer_data(
    model_id: str,
    request: Request,
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["viewer:read"])
    enforce_model_access(claims, model_id)
    try:
        return build_viewer_payload(db, model_id, limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
