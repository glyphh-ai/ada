from __future__ import annotations

import json

from fastapi import Body, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..core import models
from ..core.db import SessionLocal, get_db
from ..core.schemas import ListenerControl, ListenerLogs, ListenerOfflineOverride, ListenerOverridesResponse
from ..services.auth_runtime import enforce_model_access, require_scopes
from ..services.listener_runtime import ListenerManager
from ..services.listener_overrides import load_overrides, update_listener_override
from .router import api_router


router = api_router(tags=["listeners"])


@router.get("/listeners/{listener_id}/logs", response_model=ListenerLogs)
def listener_logs(listener_id: str, request: Request) -> ListenerLogs:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["listeners:read"])
    manager: ListenerManager = request.app.state.listener_manager
    return ListenerLogs(listener_id=listener_id, entries=manager.get_logs(listener_id))


@router.get("/listeners/offline/overrides", response_model=ListenerOverridesResponse)
def get_listener_overrides(request: Request) -> ListenerOverridesResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["listeners:read"])
    payload = load_overrides()
    return ListenerOverridesResponse(
        allowlist=payload.get("allowlist") or [],
        disabled=payload.get("disabled") or [],
        rate_limits=payload.get("rate_limits") or {},
    )


@router.patch("/listeners/{listener_id}/offline", response_model=ListenerOverridesResponse)
def update_listener_offline_override(
    listener_id: str,
    payload: ListenerOfflineOverride,
    request: Request,
) -> ListenerOverridesResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["listeners:write"])
    updated = update_listener_override(
        listener_id=listener_id,
        enabled=payload.enabled,
        allowlisted=payload.allowlisted,
        throttle=payload.throttle,
    )
    return ListenerOverridesResponse(
        allowlist=updated.get("allowlist") or [],
        disabled=updated.get("disabled") or [],
        rate_limits=updated.get("rate_limits") or {},
    )


@router.get("/listeners/{listener_id}/logs/stream")
async def listener_logs_stream(
    listener_id: str,
    request: Request,
    level: str | None = Query(None),
) -> StreamingResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["listeners:read"])
    manager: ListenerManager = request.app.state.listener_manager
    normalized_level = level.lower() if level else None
    queue, unsubscribe = manager.subscribe_logs(listener_id)

    async def event_generator():
        try:
            for entry in manager.get_logs(listener_id):
                entry_level = (entry.get("level") or "info").lower()
                if normalized_level and entry_level != normalized_level:
                    continue
                payload = {**entry, "level": entry_level}
                yield f"data: {json.dumps(payload)}\n\n"
            while True:
                entry = await queue.get()
                entry_level = (entry.get("level") or "info").lower()
                if normalized_level and entry_level != normalized_level:
                    continue
                if await request.is_disconnected():
                    break
                payload = {**entry, "level": entry_level}
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            unsubscribe()

    headers = {"Cache-Control": "no-store"}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@router.post("/listeners/{listener_id}/control")
async def control_listener(
    listener_id: str,
    payload: ListenerControl,
    request: Request,
):
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["listeners:write"])
    db = SessionLocal()
    try:
        listener = db.get(models.WebSocketListener, listener_id)
        if not listener:
            raise HTTPException(status_code=404, detail="Listener not found")
        listener.enabled = 1 if payload.action == "start" else 0
        db.commit()
    finally:
        db.close()
    manager: ListenerManager = request.app.state.listener_manager
    ok = await manager.control_listener(listener_id, payload.action)
    if not ok:
        raise HTTPException(status_code=404, detail="Listener not found or invalid action")
    return {"status": "ok", "action": payload.action}


@router.post("/listeners/{listener_id}/import-data")
async def import_listener_data(
    listener_id: str,
    request: Request,
    payload: list[dict] = Body(...),
):
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["listeners:write"])
    manager: ListenerManager = request.app.state.listener_manager
    if not manager:
        raise HTTPException(status_code=500, detail="Listener manager unavailable")
    if not manager.is_ingest_allowed(listener_id):
        raise HTTPException(status_code=403, detail="Listener is disabled or not allowlisted")
    try:
        if not isinstance(payload, list):
            raise HTTPException(status_code=400, detail="Payload must be an array of records")
        config = manager.refresh_listener_config(listener_id, include_disabled=True)
        if not config:
            raise HTTPException(status_code=404, detail="Listener not found")
        processed = manager.import_records(listener_id, payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Import failed")
    return {"status": "ok", "processed": processed}


@router.post("/listeners/{listener_id}/ingest")
async def ingest_listener_data(
    listener_id: str,
    request: Request,
    payload: dict | list[dict] = Body(...),
):
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["listeners:write"])
    manager: ListenerManager = request.app.state.listener_manager
    if not manager:
        raise HTTPException(status_code=500, detail="Listener manager unavailable")
    if not manager.is_ingest_allowed(listener_id):
        raise HTTPException(status_code=403, detail="Listener is disabled or not allowlisted")
    try:
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("records") if isinstance(payload.get("records"), list) else [payload]
        else:
            raise HTTPException(status_code=400, detail="Payload must be an object or array of records")
        config = manager.refresh_listener_config(listener_id, include_disabled=True)
        if not config:
            raise HTTPException(status_code=404, detail="Listener not found")
        processed = manager.import_records(listener_id, records)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Ingest failed")
    return {"status": "ok", "processed": processed}


@router.post("/org/{org_id}/model/{model_id}/listeners/{endpoint_name}")
async def ingest_scoped_listener_data(
    org_id: str,
    model_id: str,
    endpoint_name: str,
    request: Request,
    payload: dict | list[dict] = Body(...),
    db: Session = Depends(get_db),
):
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["listeners:write"])
    enforce_model_access(claims, model_id)
    claim_org = claims.get("org_id") or claims.get("org")
    if claim_org and str(claim_org) != org_id:
        raise HTTPException(status_code=403, detail="Org access denied")

    normalized = endpoint_name.strip().strip("/")
    if not normalized:
        raise HTTPException(status_code=400, detail="Endpoint name is required")

    listener = (
        db.query(models.WebSocketListener)
        .join(models.ModelWebSocketListener, models.ModelWebSocketListener.listener_id == models.WebSocketListener.id)
        .filter(models.ModelWebSocketListener.model_id == model_id)
        .all()
    )
    matched = None
    for item in listener:
        candidate = (item.url or "").strip().strip("/")
        if candidate == normalized:
            matched = item
            break
    if not matched:
        raise HTTPException(status_code=404, detail="Listener endpoint not found")

    manager: ListenerManager = request.app.state.listener_manager
    if not manager:
        raise HTTPException(status_code=500, detail="Listener manager unavailable")
    if not manager.is_ingest_allowed(matched.id):
        raise HTTPException(status_code=403, detail="Listener is disabled or not allowlisted")

    try:
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("records") if isinstance(payload.get("records"), list) else [payload]
        else:
            raise HTTPException(status_code=400, detail="Payload must be an object or array of records")
        config = manager.refresh_listener_config(matched.id, include_disabled=True)
        if not config:
            raise HTTPException(status_code=404, detail="Listener not found")
        processed = manager.import_records(matched.id, records)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Ingest failed")
    return {"status": "ok", "processed": processed}
