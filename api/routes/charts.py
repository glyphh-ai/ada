from __future__ import annotations

import datetime as dt
import json
import queue
import threading
import uuid
from typing import Any

from fastapi import Depends, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from ..core import models
from ..core.db import SessionLocal, get_db
from ..core.durations import DURATION_SECONDS, align_timestamp
from ..core.schemas import (
    ChartData,
    ChartPreviewRequest,
    PredictionRequest,
    SnapshotCaptureRequest,
    SnapshotCaptureResponse,
    SnapshotCaptureResult,
    TrendChartPayload,
    TrendChartRead,
)
from ..services.auth_runtime import enforce_model_access, require_scopes
from ..services.charts import build_chart_data
from ..services.predictions import compute_prediction_response
from ..services.trends_helpers import get_chart_or_404, get_trend_or_404, list_trend_charts, parse_iso_timestamp
from .router import api_router
import numpy as np


router = api_router(tags=["charts"])


def _chart_data_stream_generator(
    trend_id: str,
    chart_id: str,
    refresh_interval: float,
) -> Any:
    def worker():
        while not stop_event.is_set():
            with SessionLocal() as worker_db:
                trend = get_trend_or_404(worker_db, trend_id)
                chart = get_chart_or_404(worker_db, trend_id, chart_id)
                model = worker_db.get(models.Model, trend.model_id)
                if not model:
                    break
                chart_data = build_chart_data(worker_db, trend, model, chart)
                q.put(jsonable_encoder(chart_data))
            threading.Event().wait(refresh_interval)
        q.put(None)

    q: queue.Queue[dict[str, Any] | None] = queue.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        payload = q.get()
        if payload is None:
            break
        yield (f"data: {json.dumps(payload)}\n\n").encode("utf-8")
    stop_event.set()
    thread.join()


@router.get("/trends/{trend_id}/charts", response_model=list[TrendChartRead])
def list_trend_charts_endpoint(
    trend_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> list[TrendChartRead]:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["charts:read"])
    trend = get_trend_or_404(db, trend_id)
    enforce_model_access(claims, trend.model_id)
    return list_trend_charts(db, trend.id)


@router.get("/trends/{trend_id}/charts/{chart_id}/data")
def get_trend_chart_data(
    trend_id: str,
    chart_id: str,
    request: Request,
    db: Session = Depends(get_db),
    stream: bool = Query(False),
    start: str | None = Query(None),
    end: str | None = Query(None),
):
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["charts:read"])
    trend = get_trend_or_404(db, trend_id)
    enforce_model_access(claims, trend.model_id)
    chart = get_chart_or_404(db, trend_id, chart_id)
    model = db.get(models.Model, trend.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    refresh_interval = max(0.2, DURATION_SECONDS.get(trend.duration, 60))
    if stream:
        generator = _chart_data_stream_generator(
            trend_id,
            chart_id,
            refresh_interval,
        )
        return StreamingResponse(
            generator,
            media_type="application/json",
        )
    start_ts = parse_iso_timestamp(start)
    end_ts = parse_iso_timestamp(end)
    if start_ts and end_ts and start_ts > end_ts:
        raise HTTPException(status_code=400, detail="start must be before end")
    if not end_ts:
        end_ts = align_timestamp(dt.datetime.utcnow(), trend.duration)
    chart_data = build_chart_data(
        db,
        trend,
        model,
        chart,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    return JSONResponse(jsonable_encoder(chart_data))


@router.post("/trends/{trend_id}/chart/data", response_model=ChartData)
def preview_trend_chart_data(
    trend_id: str,
    payload: ChartPreviewRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ChartData:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["charts:write"])
    trend = get_trend_or_404(db, trend_id)
    enforce_model_access(claims, trend.model_id)
    if payload.model_id != trend.model_id:
        raise HTTPException(status_code=400, detail="Model mismatch")
    model = db.get(models.Model, payload.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    chart_payload = payload.chart_config
    chart = models.TrendChart(
        id=str(uuid.uuid4()),
        trend_id=trend.id,
        name=chart_payload.name,
        description=chart_payload.description,
        chart_type=chart_payload.chart_type,
        x_axis=chart_payload.x_axis,
        y_axes=[axis.dict(by_alias=True) for axis in chart_payload.y_axes],
        width=chart_payload.width,
        height=chart_payload.height,
        trail_length=chart_payload.trail_length,
        prediction_steps=chart_payload.prediction_steps,
        predictions_enabled=chart_payload.predictions_enabled,
        timeline_marker_color=chart_payload.timeline_marker_color,
    )
    return build_chart_data(
        db,
        trend,
        model,
        chart,
        duration_override=payload.duration,
        alpha_override=payload.alpha,
    )


@router.post(
    "/trends/{trend_id}/charts/{chart_id}/snapshots",
    response_model=SnapshotCaptureResponse,
)
def capture_trend_snapshot(
    trend_id: str,
    chart_id: str,
    payload: SnapshotCaptureRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> SnapshotCaptureResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["charts:write"])
    trend = get_trend_or_404(db, trend_id)
    enforce_model_access(claims, trend.model_id)
    chart = get_chart_or_404(db, trend_id, chart_id)
    model = db.get(models.Model, trend.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    axis_roles = {
        (axis.get("role") if isinstance(axis, dict) else axis.role)
        for axis in chart.y_axes
    }
    if payload.role not in axis_roles:
        raise HTTPException(status_code=400, detail=f"Role {payload.role} is not on this chart")
    duration = payload.duration or trend.duration
    alpha_value: float | None
    if payload.alpha is not None:
        alpha_value = payload.alpha
    elif trend.alpha is not None:
        alpha_value = float(trend.alpha)
    else:
        alpha_value = None
    layer = payload.layer
    segment_index = payload.segment_index
    if layer is None or segment_index is None:
        latest_actual = (
            db.query(models.GlyphTrend)
            .filter(
                models.GlyphTrend.trend_definition_id == trend.id,
                models.GlyphTrend.role == payload.role,
                models.GlyphTrend.source.notin_(["prediction", "snapshot"]),
            )
            .order_by(models.GlyphTrend.timestamp.desc(), models.GlyphTrend.id.desc())
            .first()
        )
        if not latest_actual:
            raise HTTPException(status_code=404, detail="No trend history for role")
        layer = latest_actual.layer if layer is None else layer
        segment_index = latest_actual.segment_index if segment_index is None else segment_index
    history_length = max(chart.trail_length, 1)
    prediction_payload = PredictionRequest(
        model_id=model.id,
        role=payload.role,
        layer=layer,
        segment_index=segment_index,
        duration=duration,
        alpha=alpha_value,
        steps=chart.prediction_steps,
        history_length=history_length,
    )
    response = compute_prediction_response(
        db,
        prediction_payload,
        max_timestamp=align_timestamp(dt.datetime.utcnow(), duration),
        persist_predictions=False,
    )
    future_steps = [step for step in response.steps if step.direction == "future"]
    if not future_steps:
        raise HTTPException(status_code=404, detail="No future prediction available")
    target_step = future_steps[-1]
    vector = np.array(response.predicted_vector, dtype=np.int8).tobytes()
    default_label = f"snapshot:{target_step.predicted_timestamp.isoformat()}"
    attr_label = payload.label or default_label
    attr: dict[str, Any] = {"value": target_step.predicted_value, "label": attr_label}
    if payload.color:
        attr["color"] = payload.color
    existing_snapshots = (
        db.query(models.GlyphTrend)
        .filter(
            models.GlyphTrend.model_id == model.id,
            models.GlyphTrend.role == payload.role,
            models.GlyphTrend.layer == layer,
            models.GlyphTrend.segment_index == segment_index,
            models.GlyphTrend.source == "snapshot",
            models.GlyphTrend.timestamp == target_step.predicted_timestamp,
        )
        .all()
    )
    if existing_snapshots:
        existing_ids = [snap.id for snap in existing_snapshots]
        db.query(models.GlyphTrend).filter(models.GlyphTrend.id.in_(existing_ids)).delete(
            synchronize_session=False
        )
    snapshot_row = models.GlyphTrend(
        model_id=model.id,
        role=payload.role,
        layer=layer,
        segment_index=segment_index,
        vector=vector,
        timestamp=target_step.predicted_timestamp,
        source="snapshot",
        attribute_value=attr,
        aligned_to_actual=False,
        trend_definition_id=trend.id,
    )
    db.add(snapshot_row)
    db.commit()
    snap_value = (
        target_step.predicted_value
        if not isinstance(attr, dict)
        else attr.get("value")
    )
    result = SnapshotCaptureResult(
        role=payload.role,
        layer=layer,
        segment_index=segment_index,
        timestamp=target_step.predicted_timestamp,
        value=snap_value if isinstance(snap_value, (int, float)) else None,
        label=payload.label,
        color=payload.color,
        vector=response.predicted_vector,
    )
    return SnapshotCaptureResponse(snapshots=[result])


@router.post("/trends/{trend_id}/charts", response_model=TrendChartRead)
def create_trend_chart(
    trend_id: str,
    payload: TrendChartPayload,
    request: Request,
    db: Session = Depends(get_db),
) -> TrendChartRead:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["charts:write"])
    trend = get_trend_or_404(db, trend_id)
    enforce_model_access(claims, trend.model_id)
    if len(payload.y_axes) > 4:
        raise HTTPException(status_code=400, detail="Up to 4 y-axis definitions supported")
    chart = models.TrendChart(
        id=str(uuid.uuid4()),
        trend_id=trend.id,
        name=payload.name,
        description=payload.description,
        chart_type=payload.chart_type,
        x_axis=payload.x_axis,
        y_axes=[axis.dict(by_alias=True) for axis in payload.y_axes],
        width=payload.width,
        height=payload.height,
        trail_length=payload.trail_length,
        prediction_steps=payload.prediction_steps,
        predictions_enabled=payload.predictions_enabled,
        timeline_marker_color=payload.timeline_marker_color,
    )
    db.add(chart)
    db.commit()
    db.refresh(chart)
    return TrendChartRead(
        id=chart.id,
        created_at=chart.created_at,
        updated_at=chart.updated_at,
        name=chart.name,
        description=chart.description,
        chart_type=chart.chart_type,
        x_axis=chart.x_axis,
        y_axes=chart.y_axes,
        width=chart.width,
        height=chart.height,
        trail_length=chart.trail_length,
        prediction_steps=chart.prediction_steps,
        predictions_enabled=chart.predictions_enabled,
        timeline_marker_color=chart.timeline_marker_color,
    )


@router.put("/trends/{trend_id}/charts/{chart_id}", response_model=TrendChartRead)
def update_trend_chart(
    trend_id: str,
    chart_id: str,
    payload: TrendChartPayload,
    request: Request,
    db: Session = Depends(get_db),
) -> TrendChartRead:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["charts:write"])
    trend = get_trend_or_404(db, trend_id)
    enforce_model_access(claims, trend.model_id)
    chart = db.get(models.TrendChart, chart_id)
    if not chart or chart.trend_id != trend.id:
        raise HTTPException(status_code=404, detail="Chart not found")
    if len(payload.y_axes) > 4:
        raise HTTPException(status_code=400, detail="Up to 4 y-axis definitions supported")
    chart.name = payload.name
    chart.description = payload.description
    chart.chart_type = payload.chart_type
    chart.x_axis = payload.x_axis
    chart.y_axes = [axis.dict(by_alias=True) for axis in payload.y_axes]
    chart.width = payload.width
    chart.height = payload.height
    chart.trail_length = payload.trail_length
    chart.prediction_steps = payload.prediction_steps
    chart.predictions_enabled = payload.predictions_enabled
    chart.timeline_marker_color = payload.timeline_marker_color
    db.commit()
    db.refresh(chart)
    return TrendChartRead(
        id=chart.id,
        created_at=chart.created_at,
        updated_at=chart.updated_at,
        name=chart.name,
        description=chart.description,
        chart_type=chart.chart_type,
        x_axis=chart.x_axis,
        y_axes=chart.y_axes,
        width=chart.width,
        height=chart.height,
        trail_length=chart.trail_length,
        prediction_steps=chart.prediction_steps,
        predictions_enabled=chart.predictions_enabled,
        timeline_marker_color=chart.timeline_marker_color,
    )


@router.delete("/trends/{trend_id}/charts/{chart_id}")
def delete_trend_chart(
    trend_id: str,
    chart_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["charts:write"])
    trend = get_trend_or_404(db, trend_id)
    enforce_model_access(claims, trend.model_id)
    chart = db.get(models.TrendChart, chart_id)
    if not chart or chart.trend_id != trend.id:
        raise HTTPException(status_code=404, detail="Chart not found")
    db.delete(chart)
    db.commit()
    return {"status": "deleted"}
