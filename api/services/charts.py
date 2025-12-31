from __future__ import annotations

import datetime as dt
import logging
from typing import Any
from datetime import datetime, timezone

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..core import models
from ..core.durations import DURATION_SECONDS, align_timestamp
from ..core.schemas import (
    ChartAggregatedPrediction,
    ChartData,
    ChartDataPoint,
    ChartDataRow,
    ChartSlot,
    HistoricalTrendChartRequest,
    PredictionRequest,
)
from .predictions import aggregate_role_predictions, compute_prediction_response

logger = logging.getLogger("glyphai.charts")
logger.setLevel(logging.DEBUG)


def _parse_numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_chart_data(
    db: Session,
    trend: models.TrendDefinition,
    model: models.Model,
    chart: models.TrendChart,
    *,
    duration_override: str | None = None,
    alpha_override: float | None = None,
    start_ts: dt.datetime | None = None,
    end_ts: dt.datetime | None = None,
) -> ChartData:
    window_size = max(10, min(10000, chart.trail_length))
    prediction_steps = max(0, chart.prediction_steps)
    horizon_index = max(0, window_size - prediction_steps - 1)
    duration = duration_override or trend.duration
    if alpha_override is not None:
        alpha_value = alpha_override
    elif trend.alpha is not None:
        alpha_value = float(trend.alpha)
    else:
        alpha_value = None
    duration_seconds = DURATION_SECONDS.get(duration, 60)
    duration_delta = dt.timedelta(seconds=duration_seconds)
    prediction_offset = duration_delta * prediction_steps
    now_ts = align_timestamp(dt.datetime.utcnow(), duration)
    if end_ts:
        slot_end = align_timestamp(end_ts, duration)
    elif start_ts:
        slot_end = align_timestamp(start_ts + duration_delta * (window_size - 1), duration)
    else:
        slot_end = align_timestamp(now_ts + prediction_offset, duration)
    slot_start = slot_end - duration_delta * (window_size - 1)
    slot_timestamps = [slot_start + duration_delta * idx for idx in range(window_size)]
    range_start = align_timestamp(start_ts, duration) if start_ts else slot_start
    range_end = align_timestamp(end_ts, duration) if end_ts else slot_end
    slots: list[ChartSlot] = []
    for idx, slot_ts in enumerate(slot_timestamps):
        if idx < horizon_index:
            slot_type = "history"
        elif idx == horizon_index:
            slot_type = "horizon"
        else:
            slot_type = "future"
        slots.append(
            ChartSlot(
                index=idx,
                timestamp=slot_ts,
                type=slot_type,
                color=chart.timeline_marker_color if slot_type == "horizon" else None,
                metadata={"slot_label": slot_ts.isoformat()},
            )
        )
    slot_index_map = {slot_ts: idx for idx, slot_ts in enumerate(slot_timestamps)}
    rows: list[ChartDataRow] = []
    confidences: list[float] = []
    prediction_payloads: list[PredictionRequest] = []
    aggregated_prediction: ChartAggregatedPrediction | None = None
    range_start_naive = range_start.replace(tzinfo=None)
    range_end_naive = (range_end + duration_delta).replace(tzinfo=None)
    for axis in chart.y_axes:
        axis_conf = axis if isinstance(axis, dict) else axis.dict(by_alias=True)
        role = axis_conf.get("role", "")
        if not role:
            continue
        actual_query = (
            db.query(models.GlyphTrend)
            .filter(
                models.GlyphTrend.trend_definition_id == trend.id,
                models.GlyphTrend.source.notin_(["prediction", "snapshot"]),
                models.GlyphTrend.role == role,
            )
        )
        actual_query = actual_query.filter(
            models.GlyphTrend.timestamp >= range_start_naive,
            models.GlyphTrend.timestamp <= range_end_naive,
        )
        actual_entries = (
            actual_query.order_by(
                models.GlyphTrend.timestamp.desc(),
                models.GlyphTrend.id.desc(),
            )
            .limit(window_size * 2)
            .all()
        )
        aligned_actuals: dict[dt.datetime, models.GlyphTrend] = {}
        aligned_vectors: dict[dt.datetime, np.ndarray] = {}
        layer_idx: int | None = None
        segment_idx: int | None = None
        segment_candidates: dict[str, tuple[int, int]] = {}
        latest_trend_slot_idx: int | None = None
        latest_trend_slot_value: float | None = None
        for entry in actual_entries:
            if layer_idx is None:
                layer_idx = entry.layer
                segment_idx = entry.segment_index
            aligned = align_timestamp(entry.timestamp, duration)
            logger.debug(
                "chart slot aligned entry trend=%s role=%s ts=%s slot=%s value=%s",
                trend.id,
                role,
                entry.timestamp,
                aligned.isoformat(),
                entry.attribute_value,
            )
            aligned_actuals.setdefault(aligned, entry)
            try:
                aligned_vectors[aligned] = np.frombuffer(entry.vector, dtype=np.int8).astype(np.float32)
            except Exception as exc:
                logger.debug(
                    "failed to buffer vector trend=%s role=%s ts=%s error=%s",
                    trend.id,
                    entry.role,
                    entry.timestamp,
                    exc,
                )
            if latest_trend_slot_idx is None:
                slot_idx = slot_index_map.get(aligned)
                if slot_idx is not None:
                    parsed_value = _parse_numeric_value(entry.attribute_value)
                    if parsed_value is not None:
                        latest_trend_slot_idx = slot_idx
                        latest_trend_slot_value = parsed_value
            if entry.source != "prediction":
                seg_key = f"{entry.layer}-{entry.segment_index}"
                if seg_key not in segment_candidates:
                    segment_candidates[seg_key] = (entry.layer, entry.segment_index)
        actual_points: list[ChartDataPoint] = []
        prev_vector: np.ndarray | None = None
        for idx, slot_ts in enumerate(slot_timestamps):
            entry = aligned_actuals.get(slot_ts)
            value = _parse_numeric_value(entry.attribute_value) if entry else None
            similarity: float | None = None
            delta_norm: float | None = None
            vector = aligned_vectors.get(slot_ts)
            if vector is not None and prev_vector is not None:
                denom = np.linalg.norm(prev_vector) * np.linalg.norm(vector)
                if denom > 0:
                    similarity = float(np.dot(prev_vector, vector) / denom)
                delta_norm = float(np.linalg.norm(vector - prev_vector))
            if vector is not None:
                prev_vector = vector
            actual_points.append(
                ChartDataPoint(
                    slot_index=idx,
                    value=value,
                    similarity=similarity,
                    delta_norm=delta_norm,
                    source=entry.source if entry else None,
                )
            )
        # Hide actual values beyond the horizon so the trend line stops at the horizon marker.
        for idx in range(horizon_index + 1, len(actual_points)):
            actual_points[idx] = ChartDataPoint(
                slot_index=idx,
                value=None,
                similarity=None,
                delta_norm=None,
            )
        rows.append(
            ChartDataRow(
                label=role,
                color=axis_conf.get("color"),
                opacity=1.0,
                type=chart.chart_type,
                points=actual_points,
                metadata={"role": role, "type": "actual"},
            )
        )
        if (
            latest_trend_slot_idx is not None
            and latest_trend_slot_value is not None
            and latest_trend_slot_idx < horizon_index
        ):
            connector_points = [
                ChartDataPoint(slot_index=idx) for idx in range(window_size)
            ]
            connector_points[latest_trend_slot_idx] = ChartDataPoint(
                slot_index=latest_trend_slot_idx,
                value=latest_trend_slot_value,
            )
            connector_points[horizon_index] = ChartDataPoint(
                slot_index=horizon_index,
                value=latest_trend_slot_value,
            )
            rows.append(
                ChartDataRow(
                    label=f"{role} horizon connector",
                    color=axis_conf.get("color"),
                    opacity=0.8,
                    type=chart.chart_type,
                    points=connector_points,
                    metadata={
                        "role": role,
                        "type": "actual-horizon-connector",
                        "hide_from_legend": True,
                    },
                )
            )
        if (
            chart.predictions_enabled
            and chart.prediction_steps > 0
            and layer_idx is not None
            and segment_idx is not None
        ):
            history_length = max(chart.trail_length, 1)
            similarity_threshold = _parse_numeric_value(
                axis_conf.get("similarityThreshold")
            )
            delta_norm_threshold = _parse_numeric_value(
                axis_conf.get("deltaNormThreshold")
            )
            axis_segments_list = [
                {"layer": layer_value, "segment": segment_value}
                for layer_value, segment_value in segment_candidates.values()
            ]
            if not axis_segments_list and layer_idx is not None and segment_idx is not None:
                axis_segments_list.append({"layer": layer_idx, "segment": segment_idx})
            prediction_points: list[ChartDataPoint] = [
                ChartDataPoint(slot_index=idx) for idx in range(window_size)
            ]
            history_points: list[ChartDataPoint] = [
                ChartDataPoint(slot_index=idx) for idx in range(window_size)
            ]
            axis_color = (
                axis_conf.get("prediction_color")
                or axis_conf.get("predictionColor")
                or axis_conf.get("color")
            )
            axis_opacity = (
                axis_conf.get("prediction_opacity")
                or axis_conf.get("predictionOpacity")
            )
            last_history_slot_idx: int | None = None
            last_history_value: float | None = None
            aggregated_for_axis: ChartAggregatedPrediction | None = None
            axis_response: PredictionResponse | None = None
            prediction_payloads: list[PredictionRequest] = []
            for coords in axis_segments_list:
                prediction_payloads.append(
                    PredictionRequest(
                        model_id=model.id,
                        role=role,
                        layer=coords["layer"],
                        segment_index=coords["segment"],
                        duration=duration,
                        alpha=alpha_value,
                        steps=chart.prediction_steps,
                        history_length=history_length,
                        similarity_threshold=similarity_threshold,
                        delta_norm_threshold=delta_norm_threshold,
                    )
                )
            if prediction_payloads:
                try:
                    multi_result = aggregate_role_predictions(
                        db,
                        prediction_payloads,
                        max_timestamp=slot_end,
                        persist_predictions=True,
                    )
                    axis_response = next(
                        (
                            res
                            for res in multi_result.responses
                            if res.layer == layer_idx and res.segment_index == segment_idx
                        ),
                        multi_result.responses[0] if multi_result.responses else None,
                    )
                    if multi_result.responses:
                        pred_vec = np.array(
                            multi_result.aggregated_predicted_vector, dtype=np.float32
                        )
                        curr_vec = np.array(
                            multi_result.aggregated_current_vector, dtype=np.float32
                        )
                        delta_norm = (
                            float(np.linalg.norm(pred_vec - curr_vec))
                            if pred_vec.shape == curr_vec.shape
                            else None
                        )
                        aggregated_for_axis = ChartAggregatedPrediction(
                            similarity=multi_result.similarity_to_current,
                            delta_norm=delta_norm,
                            roles=len(prediction_payloads),
                        )
                        aggregated_prediction = aggregated_for_axis
                except HTTPException as exc:
                    logger.debug(
                        "skipping trend prediction for trend=%s role=%s: %s",
                        trend.id,
                        role,
                        exc.detail,
                    )
                except Exception:
                    logger.exception(
                        "failed to compute trend prediction for trend=%s role=%s",
                        trend.id,
                        role,
                    )
            if axis_response:
                future_steps = [
                    step for step in axis_response.steps if step.direction == "future"
                ]
                history_steps = [
                    step for step in axis_response.steps if step.direction == "history"
                ]
                furthest_history_future: tuple[int, Any] | None = None
                for step in future_steps:
                    slot_idx = slot_index_map.get(step.predicted_timestamp)
                    if slot_idx is None:
                        continue
                    prediction_points[slot_idx] = ChartDataPoint(
                        slot_index=slot_idx,
                        value=step.predicted_value,
                        similarity=step.similarity_to_previous,
                        delta_norm=step.delta_norm,
                        source="prediction",
                    )
                    if step.similarity_to_previous is not None:
                        confidences.append(step.similarity_to_previous)
                for step in history_steps:
                    slot_idx = slot_index_map.get(step.predicted_timestamp)
                    if slot_idx is None:
                        continue
                    if slot_idx > horizon_index:
                        if (
                            furthest_history_future is None
                            or slot_idx > furthest_history_future[0]
                        ):
                            furthest_history_future = (slot_idx, step)
                        continue
                    history_points[slot_idx] = ChartDataPoint(
                        slot_index=slot_idx,
                        value=step.predicted_value,
                        similarity=step.similarity_to_previous,
                        delta_norm=step.delta_norm,
                        source="prediction",
                    )
                    if step.similarity_to_previous is not None:
                        confidences.append(step.similarity_to_previous)
                    last_history_slot_idx = slot_idx
                    last_history_value = step.predicted_value
                if furthest_history_future:
                    slot_idx, step = furthest_history_future
                    history_points[slot_idx] = ChartDataPoint(
                        slot_index=slot_idx,
                        value=step.predicted_value,
                        similarity=step.similarity_to_previous,
                        delta_norm=step.delta_norm,
                        source="prediction",
                    )
                    if step.similarity_to_previous is not None:
                        confidences.append(step.similarity_to_previous)
                    last_history_slot_idx = slot_idx
                    last_history_value = step.predicted_value
            # Anchor the predicted dashed line at the most recent actual slot before the horizon.
            anchor_slot_index = latest_trend_slot_idx
            anchor_value = latest_trend_slot_value
            if anchor_slot_index is None or anchor_value is None:
                for idx in range(min(horizon_index, len(actual_points) - 1), -1, -1):
                    val = actual_points[idx].value
                    if val is not None:
                        anchor_slot_index = idx
                        anchor_value = val
                        break
            if anchor_slot_index is not None and anchor_value is not None:
                current_point = prediction_points[anchor_slot_index]
                if current_point.value is None:
                    prediction_points[anchor_slot_index] = ChartDataPoint(
                        slot_index=anchor_slot_index,
                        value=anchor_value,
                        source="prediction",
                    )
            if any(point.value is not None for point in history_points):
                rows.append(
                    ChartDataRow(
                        label=f"{role} prediction history",
                        color=axis_color or axis_conf.get("color"),
                        opacity=axis_opacity or 0.45,
                        type=chart.chart_type,
                        points=history_points,
                        metadata={
                            "role": role,
                            "layer": layer_idx,
                            "segment_index": segment_idx,
                            "type": "prediction-history",
                        },
                    )
                )
            if (
                anchor_slot_index is not None
                and anchor_value is not None
                and last_history_slot_idx is not None
                and last_history_value is not None
                and anchor_slot_index != last_history_slot_idx
            ):
                history_connector_points = [
                    ChartDataPoint(slot_index=idx) for idx in range(window_size)
                ]
                history_connector_points[anchor_slot_index] = ChartDataPoint(
                    slot_index=anchor_slot_index,
                    value=anchor_value,
                )
                history_connector_points[last_history_slot_idx] = ChartDataPoint(
                    slot_index=last_history_slot_idx,
                    value=last_history_value,
                )
                rows.append(
                    ChartDataRow(
                        label=f"{role} prediction history connector",
                        color=axis_color or axis_conf.get("color"),
                        opacity=axis_opacity or 0.4,
                        type=chart.chart_type,
                        points=history_connector_points,
                        metadata={
                            "role": role,
                            "type": "prediction-history-connector",
                            "hide_from_legend": True,
                        },
                    )
                )
            if any(point.value is not None for point in prediction_points):
                rows.append(
                    ChartDataRow(
                        label=f"{role} prediction",
                        color=axis_color or axis_conf.get("color"),
                        opacity=axis_opacity or 0.65,
                        type=chart.chart_type,
                        points=prediction_points,
                        metadata={
                            "role": role,
                            "layer": layer_idx,
                            "segment_index": segment_idx,
                            "type": "prediction",
                        },
                    )
                )
        snapshot_query = (
            db.query(models.GlyphTrend)
            .filter(
                models.GlyphTrend.trend_definition_id == trend.id,
                models.GlyphTrend.source == "snapshot",
                models.GlyphTrend.role == role,
            )
        )
        if layer_idx is not None:
            snapshot_query = snapshot_query.filter(models.GlyphTrend.layer == layer_idx)
        if segment_idx is not None:
            snapshot_query = snapshot_query.filter(
                models.GlyphTrend.segment_index == segment_idx
            )
        snapshot_entries = (
            snapshot_query.order_by(
                models.GlyphTrend.timestamp.desc(),
                models.GlyphTrend.id.desc(),
            )
            .limit(window_size * 2)
            .all()
        )
        snapshot_by_slot: dict[int, models.GlyphTrend] = {}
        for entry in reversed(snapshot_entries):
            aligned = align_timestamp(entry.timestamp, duration)
            if aligned < slot_start or aligned > slot_end:
                continue
            slot_idx = slot_index_map.get(aligned)
            if slot_idx is not None and slot_idx not in snapshot_by_slot:
                snapshot_by_slot[slot_idx] = entry
        snapshot_points: list[ChartDataPoint] = [
            ChartDataPoint(slot_index=idx) for idx in range(window_size)
        ]
        snapshot_color_override: str | None = None
        snapshot_label_override: str | None = None
        for slot_idx, entry in snapshot_by_slot.items():
            raw_value = entry.attribute_value
            snapshot_value = (
                _parse_numeric_value(raw_value.get("value"))
                if isinstance(raw_value, dict)
                else _parse_numeric_value(raw_value)
            )
            if snapshot_value is None:
                continue
            if isinstance(raw_value, dict):
                snap_color = raw_value.get("color")
                if isinstance(raw_value.get("label"), str):
                    snapshot_label_override = raw_value.get("label")
                if isinstance(snap_color, str):
                    snapshot_color_override = snap_color
            snapshot_points[slot_idx] = ChartDataPoint(
                slot_index=slot_idx,
                value=snapshot_value,
                source="snapshot",
            )
        if any(point.value is not None for point in snapshot_points):
            rows.append(
                ChartDataRow(
                    label=snapshot_label_override or f"{role} snapshot",
                    color=(
                        snapshot_color_override
                        or axis_conf.get("snapshot_color")
                        or axis_conf.get("snapshotColor")
                        or axis_conf.get("color")
                    ),
                    opacity=axis_conf.get("snapshot_opacity")
                    or axis_conf.get("snapshotOpacity")
                    or 0.8,
                    type=chart.chart_type,
                    points=snapshot_points,
                    metadata={
                        "role": role,
                        "layer": layer_idx,
                        "segment_index": segment_idx,
                        "type": "snapshot",
                    },
                )
            )
    average_confidence = (
        sum(confidences) / len(confidences) if confidences else None
    )
    return ChartData(
        slots=slots,
        rows=rows,
        horizon_slot_index=horizon_index,
        average_confidence=average_confidence,
        duration=duration,
        aggregated_prediction=aggregated_prediction,
    )


def build_historical_chart_data(
    db: Session,
    payload: HistoricalTrendChartRequest,
) -> ChartData:
    window_size = max(10, min(200, payload.trail_length))
    duration = payload.duration
    duration_seconds = DURATION_SECONDS.get(duration, 60)
    duration_delta = dt.timedelta(seconds=duration_seconds)
    start_ts = payload.start
    end_ts = payload.end
    now_ts = align_timestamp(dt.datetime.utcnow(), duration)
    if end_ts:
        slot_end = align_timestamp(end_ts, duration)
    elif start_ts:
        slot_end = align_timestamp(
            start_ts + duration_delta * (window_size - 1), duration
        )
    else:
        slot_end = align_timestamp(now_ts, duration)
    slot_start = slot_end - duration_delta * (window_size - 1)
    slot_timestamps = [
        slot_start + duration_delta * idx for idx in range(window_size)
    ]
    horizon_index = max(0, window_size - 1)
    slots: list[ChartSlot] = []
    for idx, slot_ts in enumerate(slot_timestamps):
        slot_type = "horizon" if idx == horizon_index else "history"
        slots.append(
            ChartSlot(
                index=idx,
                timestamp=slot_ts,
                type=slot_type,
                metadata={"slot_label": slot_ts.isoformat()},
            )
        )
    range_start = start_ts or slot_start
    range_end = end_ts or slot_end
    range_start_naive = range_start.replace(tzinfo=None)
    range_end_naive = (range_end + duration_delta).replace(tzinfo=None)
    rows: list[ChartDataRow] = []
    for axis in payload.axes:
        axis_role = axis.role.strip()
        if not axis_role:
            continue
        axis_query = (
            db.query(models.GlyphTrend)
            .filter(
                models.GlyphTrend.model_id == payload.model_id,
                models.GlyphTrend.role == axis_role,
                models.GlyphTrend.source != "prediction",
            )
            .filter(
                models.GlyphTrend.timestamp >= range_start_naive,
                models.GlyphTrend.timestamp <= range_end_naive,
            )
            .order_by(
                models.GlyphTrend.timestamp.desc(),
                models.GlyphTrend.id.desc(),
            )
        )
        if axis.layer is not None:
            axis_query = axis_query.filter(models.GlyphTrend.layer == axis.layer)
        if axis.segment_index is not None:
            axis_query = axis_query.filter(
                models.GlyphTrend.segment_index == axis.segment_index
            )
        axis_entries = axis_query.limit(window_size * 2).all()
        aligned_actuals: dict[dt.datetime, models.GlyphTrend] = {}
        aligned_vectors: dict[dt.datetime, np.ndarray] = {}
        for entry in axis_entries:
            aligned = align_timestamp(entry.timestamp, duration)
            existing_entry = aligned_actuals.get(aligned)
            if existing_entry is None:
                aligned_actuals[aligned] = entry
            else:
                logger.debug(
                    "dropping glyph trend entry slot=%s trend=%s role=%s entry=%s value=%s existing=%s value=%s",
                    aligned.isoformat(),
                    trend.id,
                    role,
                    entry.id,
                    entry.attribute_value,
                    existing_entry.id,
                    existing_entry.attribute_value,
                )
            try:
                aligned_vectors[aligned] = np.frombuffer(
                    entry.vector, dtype=np.int8
                ).astype(np.float32)
            except Exception:
                pass
        actual_points: list[ChartDataPoint] = []
        prev_vector: np.ndarray | None = None
        for idx, slot_ts in enumerate(slot_timestamps):
            entry = aligned_actuals.get(slot_ts)
            value = _parse_numeric_value(entry.attribute_value) if entry else None
            similarity: float | None = None
            delta_norm: float | None = None
            vector = aligned_vectors.get(slot_ts)
            if vector is not None and prev_vector is not None:
                denom = np.linalg.norm(prev_vector) * np.linalg.norm(vector)
                if denom > 0:
                    similarity = float(np.dot(prev_vector, vector) / denom)
                delta_norm = float(np.linalg.norm(vector - prev_vector))
            if vector is not None:
                prev_vector = vector
            actual_points.append(
                ChartDataPoint(
                    slot_index=idx,
                    value=value,
                    similarity=similarity,
                    delta_norm=delta_norm,
                    source=entry.source if entry else None,
                )
            )
        rows.append(
            ChartDataRow(
                label=axis_role,
                color=axis.color,
                opacity=1.0,
                type="line",
                points=actual_points,
                metadata={
                    "role": axis_role,
                    "layer": axis.layer,
                    "segment_index": axis.segment_index,
                    "type": "historical",
                },
            )
        )
    return ChartData(
        slots=slots,
        rows=rows,
        horizon_slot_index=horizon_index,
        average_confidence=None,
        duration=duration,
        aggregated_prediction=None,
    )
