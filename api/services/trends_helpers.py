from __future__ import annotations

import datetime as dt

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..core import models
from ..core.schemas import TrendChartRead


def get_trend_or_404(db: Session, trend_id: str) -> models.TrendDefinition:
    trend = db.get(models.TrendDefinition, trend_id)
    if not trend:
        raise HTTPException(status_code=404, detail="Trend not found")
    return trend


def get_chart_or_404(db: Session, trend_id: str, chart_id: str) -> models.TrendChart:
    chart = db.get(models.TrendChart, chart_id)
    if not chart or chart.trend_id != trend_id:
        raise HTTPException(status_code=404, detail="Chart not found")
    return chart


def list_trend_charts(db: Session, trend_id: str) -> list[TrendChartRead]:
    rows = (
        db.query(models.TrendChart)
        .filter(models.TrendChart.trend_id == trend_id)
        .order_by(models.TrendChart.created_at.asc())
        .all()
    )
    return [
        TrendChartRead(
            id=row.id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            name=row.name,
            description=row.description,
            chart_type=row.chart_type,
            x_axis=row.x_axis,
            y_axes=row.y_axes,
            width=row.width,
            height=row.height,
            trail_length=row.trail_length,
            prediction_steps=row.prediction_steps,
            predictions_enabled=row.predictions_enabled,
            timeline_marker_color=row.timeline_marker_color,
        )
        for row in rows
    ]


def parse_iso_timestamp(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp")
