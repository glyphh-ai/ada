from __future__ import annotations

from fastapi import Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..core.db import get_db
from ..services.viewer_builder import build_viewer_payload
from .router import api_router


router = api_router(tags=["viewer"])


@router.get("/api/viewer/data/{model_id}")
def viewer_data(
    model_id: str,
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    try:
        return build_viewer_payload(db, model_id, limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
