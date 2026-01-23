from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, List

import numpy as np
from fastapi import Depends, HTTPException, Query, Request, Response, status
from sqlalchemy.orm import Session

from ..core import models
from ..core.db import get_db
from ..core.durations import align_timestamp
from ..core.schemas import (
    ChartData,
    HistoricalTrendChartRequest,
    PredictionRequest,
    PredictionResponse,
    TrendDefinitionPayload,
    TrendDefinitionRead,
    TrendEntry,
    TrendListResponse,
    TrendResponse,
)
from ..services.auth_runtime import enforce_model_access, require_scopes
from ..services.charts import build_historical_chart_data
from ..services.predictions import compute_prediction_response
from ..services.trends_helpers import get_trend_or_404, list_trend_charts
from .router import api_router


router = api_router(tags=["trends"])


@router.get("/analysis/trends", response_model=TrendResponse)
def analysis_trends(
    model_id: str,
    request: Request,
    roles: List[str] | None = Query(None),
    limit: int = Query(200, ge=10, le=2000),
    trend_id: str | None = Query(None),
    db: Session = Depends(get_db),
) -> TrendResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["trends:read"])
    enforce_model_access(claims, model_id)
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    rows = (
        db.query(models.GlyphTrend)
        .filter(models.GlyphTrend.model_id == model_id)
        .order_by(
            models.GlyphTrend.timestamp.desc(),
            models.GlyphTrend.id.desc(),
        )
    )
    trend_definition = None
    trend_charts = None
    if trend_id:
        trend = db.get(models.TrendDefinition, trend_id)
        if not trend or trend.model_id != model_id or not trend.enabled:
            raise HTTPException(status_code=404, detail="Trend not found")
        rows = rows.filter(models.GlyphTrend.trend_definition_id == trend_id)
        trend_charts = list_trend_charts(db, trend.id)
        trend_definition = TrendDefinitionRead(
            id=trend.id,
            created_at=trend.created_at,
            updated_at=trend.updated_at,
            name=trend.name,
            description=trend.description,
            model_id=trend.model_id,
            roles=trend.roles,
            duration=trend.duration,
            alpha=float(trend.alpha),
            enabled=trend.enabled,
            charts=trend_charts or [],
        )
    elif roles:
        rows = rows.filter(models.GlyphTrend.role.in_(roles))
    rows = rows.limit(limit).all()
    rows = list(reversed(rows))
    prev_segments: dict[tuple[str, int, int], np.ndarray] = {}
    entries: list[TrendEntry] = []
    for row in rows:
        key = (row.role, row.layer, row.segment_index)
        current_vec = np.frombuffer(row.vector, dtype=np.int8).astype(np.float32)
        similarity = None
        delta_norm = None
        prev = prev_segments.get(key)
        if prev is not None:
            denom = np.linalg.norm(prev) * np.linalg.norm(current_vec)
            if denom > 0:
                similarity = float(np.dot(prev, current_vec) / denom)
            delta_norm = float(np.linalg.norm(current_vec - prev))
        prev_segments[key] = current_vec
        entries.append(
            TrendEntry(
                timestamp=row.timestamp,
                role=row.role,
                layer=row.layer,
                segment_index=row.segment_index,
                similarity=similarity,
                delta_norm=delta_norm,
                source=row.source,
                attribute_value=row.attribute_value,
            )
        )
    return TrendResponse(
        model_id=model_id,
        entries=entries,
        charts=trend_charts,
        trend_definition=trend_definition,
    )


@router.post("/analysis/predictions", response_model=PredictionResponse)
def analysis_predictions(
    payload: PredictionRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["trends:read"])
    enforce_model_access(claims, payload.model_id)
    model = db.get(models.Model, payload.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return compute_prediction_response(db, payload)


@router.post("/analysis/trend-data", response_model=ChartData)
def analysis_trend_data(
    payload: HistoricalTrendChartRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ChartData:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["trends:read"])
    enforce_model_access(claims, payload.model_id)
    model = db.get(models.Model, payload.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return build_historical_chart_data(db, payload)


@router.post("/trends", response_model=TrendDefinitionRead)
def create_trend(
    payload: TrendDefinitionPayload,
    request: Request,
    db: Session = Depends(get_db),
) -> TrendDefinitionRead:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["trends:write"])
    enforce_model_access(claims, payload.model_id)
    model = db.get(models.Model, payload.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    trend = models.TrendDefinition(
        id=str(uuid.uuid4()),
        name=payload.name,
        description=payload.description,
        model_id=payload.model_id,
        roles=payload.roles,
        duration=payload.duration,
        alpha=payload.alpha,
        enabled=payload.enabled,
    )
    db.add(trend)
    db.commit()
    db.refresh(trend)
    return TrendDefinitionRead(
        id=trend.id,
        created_at=trend.created_at,
        updated_at=trend.updated_at,
        **payload.dict(),
        charts=[],
    )


@router.delete("/trends/{trend_id}")
def delete_trend(
    trend_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Response:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["trends:write"])
    trend = get_trend_or_404(db, trend_id)
    enforce_model_access(claims, trend.model_id)
    db.query(models.TrendChart).filter(models.TrendChart.trend_id == trend_id).delete(
        synchronize_session=False
    )
    db.query(models.TrendPredictionFrontier).filter(
        models.TrendPredictionFrontier.trend_id == trend_id
    ).delete(synchronize_session=False)
    db.query(models.GlyphTrend).filter(
        models.GlyphTrend.trend_definition_id == trend_id
    ).delete(synchronize_session=False)
    db.delete(trend)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/trends/{trend_id}", response_model=TrendDefinitionRead)
def update_trend(
    trend_id: str,
    payload: TrendDefinitionPayload,
    request: Request,
    db: Session = Depends(get_db),
) -> TrendDefinitionRead:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["trends:write"])
    trend = db.get(models.TrendDefinition, trend_id)
    if not trend:
        raise HTTPException(status_code=404, detail="Trend not found")
    enforce_model_access(claims, trend.model_id)
    if payload.model_id != trend.model_id:
        next_model = db.get(models.Model, payload.model_id)
        if not next_model:
            raise HTTPException(status_code=404, detail="Model not found")
        enforce_model_access(claims, payload.model_id)
        trend.model_id = payload.model_id
        db.query(models.GlyphTrend).filter(
            models.GlyphTrend.trend_definition_id == trend_id
        ).delete(synchronize_session=False)
        db.query(models.TrendPredictionFrontier).filter(
            models.TrendPredictionFrontier.trend_id == trend_id
        ).delete(synchronize_session=False)
    trend.name = payload.name
    trend.description = payload.description
    trend.roles = payload.roles
    trend.duration = payload.duration
    trend.alpha = payload.alpha
    trend.enabled = payload.enabled
    db.commit()
    db.refresh(trend)
    return TrendDefinitionRead(
        id=trend.id,
        created_at=trend.created_at,
        updated_at=trend.updated_at,
        name=trend.name,
        description=trend.description,
        model_id=trend.model_id,
        roles=trend.roles,
        duration=trend.duration,
        alpha=float(trend.alpha),
        enabled=trend.enabled,
        charts=list_trend_charts(db, trend.id),
    )


@router.get("/trends", response_model=TrendListResponse)
def list_trends(
    request: Request,
    db: Session = Depends(get_db),
) -> TrendListResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["trends:read"])
    model_filter = claims.get("model_id")
    query = db.query(models.TrendDefinition)
    if model_filter:
        query = query.filter(models.TrendDefinition.model_id == model_filter)
    items = query.order_by(models.TrendDefinition.updated_at.desc()).all()
    return TrendListResponse(
        items=[
            TrendDefinitionRead(
                id=item.id,
                created_at=item.created_at,
                updated_at=item.updated_at,
                name=item.name,
                description=item.description,
                model_id=item.model_id,
                roles=item.roles,
                duration=item.duration,
                alpha=float(item.alpha),
                enabled=item.enabled,
                charts=list_trend_charts(db, item.id),
            )
            for item in items
        ]
    )


@router.get("/trends/{trend_id}", response_model=TrendDefinitionRead)
def get_trend(
    trend_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> TrendDefinitionRead:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["trends:read"])
    trend = db.get(models.TrendDefinition, trend_id)
    if not trend:
        raise HTTPException(status_code=404, detail="Trend not found")
    enforce_model_access(claims, trend.model_id)
    return TrendDefinitionRead(
        id=trend.id,
        created_at=trend.created_at,
        updated_at=trend.updated_at,
        name=trend.name,
        description=trend.description,
        model_id=trend.model_id,
        roles=trend.roles,
        duration=trend.duration,
        alpha=float(trend.alpha),
        enabled=trend.enabled,
        charts=list_trend_charts(db, trend.id),
    )


@router.post("/trends/{trend_id}/rebuild")
def rebuild_trend_history(
    trend_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["trends:write"])
    trend = get_trend_or_404(db, trend_id)
    enforce_model_access(claims, trend.model_id)
    roles = trend.roles or []
    if not roles:
        return {"status": "no_roles", "inserted": 0}
    deleted = (
        db.query(models.GlyphTrend)
        .filter(models.GlyphTrend.trend_definition_id == trend.id)
        .delete(synchronize_session=False)
    )
    db.flush()
    query = (
        db.query(models.GlyphTrend)
        .filter(
            models.GlyphTrend.model_id == trend.model_id,
            models.GlyphTrend.role.in_(roles),
            models.GlyphTrend.trend_definition_id.is_(None),
        )
        .yield_per(500)
    )
    rows = list(query)
    inserted = 0
    batch = 0
    for row in rows:
        row_copy = models.GlyphTrend(
            model_id=row.model_id,
            role=row.role,
            layer=row.layer,
            segment_index=row.segment_index,
            vector=row.vector,
            timestamp=row.timestamp,
            source=row.source,
            attribute_value=row.attribute_value,
            trend_definition_id=trend.id,
        )
        align_timestamp(row.timestamp if row.timestamp else dt.datetime.utcnow(), trend.duration)
        db.add(row_copy)
        inserted += 1
        batch += 1
        if batch >= 200:
            db.flush()
            batch = 0
    db.commit()
    return {"status": "ok", "inserted": inserted, "removed": deleted}
