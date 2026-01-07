from __future__ import annotations

import json

from fastapi import Body, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from ..core import models
from ..core.db import SessionLocal
from ..core.schemas import ListenerControl, ListenerLogs
from ..services.listener_runtime import ListenerManager
from .router import api_router


router = api_router(tags=["listeners"])


@router.get("/listeners/{listener_id}/logs", response_model=ListenerLogs)
def listener_logs(listener_id: str, request: Request) -> ListenerLogs:
    manager: ListenerManager = request.app.state.listener_manager
    return ListenerLogs(listener_id=listener_id, entries=manager.get_logs(listener_id))


@router.get("/listeners/{listener_id}/logs/stream")
async def listener_logs_stream(
    listener_id: str,
    request: Request,
    level: str | None = Query(None),
) -> StreamingResponse:
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
    manager: ListenerManager = request.app.state.listener_manager
    if not manager:
        raise HTTPException(status_code=500, detail="Listener manager unavailable")
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
    manager: ListenerManager = request.app.state.listener_manager
    if not manager:
        raise HTTPException(status_code=500, detail="Listener manager unavailable")
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
