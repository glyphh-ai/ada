from __future__ import annotations

import datetime as dt

from fastapi import Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from ..core import models
from ..core.config import get_settings
from ..core.db import get_db, SessionLocal
from ..core.schemas import ConceptInput, IngestRequest, ModelIngestRequest, ModelRefreshResponse, QueryResult
from ..services.auth_runtime import decode_jwt, parse_license_expiry
from ..services.ingest import ingest_concepts, refresh_model_data
from ..services.usage_runtime import record_usage
from .router import api_router


router = api_router(tags=["ingest"])
settings = get_settings()


@router.post("/ingest", response_model=QueryResult)
def ingest(
    payload: IngestRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> QueryResult:
    model = db.get(models.Model, payload.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    record_usage(request.app, "glyph_crud", count=len(payload.concepts))
    return ingest_concepts(model, payload.concepts, payload.clear_existing, db)


@router.post("/models/{model_id}/ingest", response_model=QueryResult)
def ingest_for_model(
    model_id: str,
    payload: ModelIngestRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> QueryResult:
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    record_usage(request.app, "glyph_crud", count=len(payload.concepts))
    result = ingest_concepts(model, payload.concepts, payload.clear_existing, db)
    if payload.force_refresh:
        refresh_model_data(model, db)
    return result


@router.post("/models/{model_id}/refresh", response_model=ModelRefreshResponse)
def refresh_model_endpoint(
    model_id: str,
    db: Session = Depends(get_db),
) -> ModelRefreshResponse:
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return refresh_model_data(model, db)


@router.websocket("/models/{model_id}/ingest-stream")
async def ingest_stream(model_id: str, websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        payload = decode_jwt(token, settings.jwt_secret, settings.jwt_algorithm)
        now_ts = int(dt.datetime.utcnow().timestamp())
        exp = payload.get("exp")
        if isinstance(exp, int) and exp < now_ts:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        license_expiry = parse_license_expiry(payload.get("license_expires_at"))
        if license_expiry and license_expiry < dt.datetime.utcnow():
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="License expired")
        if payload.get("token_type") != "runtime":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid runtime token")
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    db = SessionLocal()
    try:
        model = db.get(models.Model, model_id)
        if not model:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        buffer: list[ConceptInput] = []
        while True:
            try:
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            record_usage(websocket.app, "websocket_events", count=1)
            action = payload.get("type")
            if action == "concept":
                batch = payload.get("concept")
                if isinstance(batch, dict):
                    buffer.append(ConceptInput(**batch))
            elif action == "flush":
                if buffer:
                    record_usage(websocket.app, "glyph_crud", count=len(buffer))
                    result = ingest_concepts(
                        model, buffer, bool(payload.get("clear_existing")), db
                    )
                    buffer.clear()
                    await websocket.send_json(
                        {"status": "ok", "matches": len(result.matches)}
                    )
                    if payload.get("force_refresh"):
                        refresh = refresh_model_data(model, db)
                        await websocket.send_json({"refresh": refresh.dict()})
            elif action == "refresh":
                refresh = refresh_model_data(model, db)
                await websocket.send_json({"refresh": refresh.dict()})
            elif action == "close":
                await websocket.close()
                break
    finally:
        db.close()
