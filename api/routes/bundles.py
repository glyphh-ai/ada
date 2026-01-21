from __future__ import annotations

from fastapi import Depends, Query, Request
from sqlalchemy.orm import Session

from ..core.db import get_db
from ..core.schemas import SampleImportResponse, SampleUploadBundle
from ..services.auth_runtime import require_scopes
from ..services.bundle_import import import_bundle
from .router import api_router


router = api_router(tags=["bundles"])


@router.post("/bundles/import", response_model=SampleImportResponse)
def import_bundle_route(
    payload: SampleUploadBundle,
    clear_existing: bool = Query(True),
    request: Request,
    db: Session = Depends(get_db),
) -> SampleImportResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["bundles:write"])
    return import_bundle(db, payload, clear_existing=clear_existing)
