from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from typing import Any, NamedTuple, Sequence

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session

from glyphh.encoder import Encoder
from glyphh.vector import bind

from ..core import models
from ..core.durations import DURATION_SECONDS, align_timestamp
from ..core.schemas import PredictionRequest, PredictionResponse, PredictionStep

logger = logging.getLogger("glyphai.predictions")


class GlyphTrendSnapshot(NamedTuple):
    timestamp: dt.datetime
    vector: bytes
    attribute_value: Any
    source: str


def parse_numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value_token(raw_value: Any) -> str:
    if raw_value is None:
        return "null"
    if isinstance(raw_value, (str, bool, int, float)):
        return str(raw_value)
    return json.dumps(raw_value, sort_keys=True)


def _role_value_vector(encoder: Encoder, role: str, raw_value: Any) -> np.ndarray:
    value_token = _value_token(raw_value)
    return bind(encoder.role_vec(role), encoder.value_vec(value_token))


def _encode_prediction_cortex(
    encoder: Encoder,
    name: str,
    attributes: dict[str, Any],
) -> np.ndarray:
    glyph = encoder.encode(name, attributes)
    if glyph.global_cortex is None:
        return np.zeros(encoder.dim, dtype=np.int8)
    return glyph.global_cortex


def _weighted_average(
    values: Sequence[np.ndarray],
    min_weight: float = 0.5,
    modifiers: Sequence[float] | None = None,
) -> tuple[np.ndarray, float] | None:
    if not values:
        return None
    weights = np.linspace(min_weight, 1.0, len(values))
    if modifiers is not None:
        if len(modifiers) != len(values):
            raise ValueError("Modifiers must align with values")
        modifier_arr = np.array(modifiers, dtype=np.float32)
        weights = weights * modifier_arr
    stacked = np.stack(values)
    avg = np.average(stacked, axis=0, weights=weights)
    return avg, float(weights.sum())


def _weighted_scalar(
    values: Sequence[float],
    min_weight: float = 0.5,
    modifiers: Sequence[float] | None = None,
) -> tuple[float, float] | None:
    if not values:
        return None
    weights = np.linspace(min_weight, 1.0, len(values))
    if modifiers is not None:
        if len(modifiers) != len(values):
            raise ValueError("Modifiers must align with values")
        modifier_arr = np.array(modifiers, dtype=np.float32)
        weights = weights * modifier_arr
    weighted_sum = sum(val * w for val, w in zip(values, weights))
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight, total_weight


def _vector_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return None
    return float(np.dot(a, b) / denom)


def _history_reliability(
    similarity: float | None,
    delta_norm: float | None,
    *,
    vector_dim: int | None = None,
) -> float:
    sim_score = similarity if similarity is not None else 0.0
    sim_factor = max(0.1, 1.0 + sim_score)
    delta = delta_norm if delta_norm is not None else 0.0
    # Make delta penalty dimension-aware so large vectors don't dominate.
    denom = float(np.sqrt(vector_dim)) if vector_dim and vector_dim > 0 else 1.0
    delta_factor = 1.0 / (1.0 + (delta / denom))
    return sim_factor * delta_factor


def persist_prediction_glyph(
    db: Session,
    payload: PredictionRequest,
    encoder: Encoder,
    predicted_cortex: np.ndarray,
    predicted_timestamp: dt.datetime,
    similarity: float | None,
    delta_norm: float | None,
    step: int,
    direction: str,
    attribute_value: float | None = None,
) -> None:
    glyph_name = f"prediction:{payload.role}:{uuid.uuid4()}"
    prediction_name = payload.role
    semantic = {
        "prediction": {
            "role": payload.role,
            "layer": payload.layer,
            "segment_index": payload.segment_index,
            "predicted_timestamp": predicted_timestamp.isoformat(),
            "similarity": similarity,
            "delta_norm": delta_norm,
            "step": step,
            "direction": direction,
            "concept_name": prediction_name,
            "attributes": {payload.role: attribute_value},
        }
    }
    bound_vec = _role_value_vector(encoder, payload.role, attribute_value)
    glyph = models.Glyph(
        name=glyph_name,
        model_id=payload.model_id,
        node_type="prediction",
        semantic=semantic,
        cortex=predicted_cortex.astype(np.int8, copy=False).tobytes(),
        prediction_flag=True,
    )
    db.add(glyph)
    db.add(
        models.GlyphTrend(
            model_id=payload.model_id,
            role=payload.role,
            layer=payload.layer,
            segment_index=payload.segment_index,
            vector=bound_vec.astype(np.int8, copy=False).tobytes(),
            timestamp=predicted_timestamp,
            source="prediction",
            attribute_value=attribute_value,
            aligned_to_actual=False,
        )
    )
    db.commit()


def compute_prediction_response(
    db: Session,
    payload: PredictionRequest,
    max_timestamp: dt.datetime | None = None,
    force_persist: bool = False,
    persist_predictions: bool = True,
) -> PredictionResponse:
    model = db.get(models.Model, payload.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    encoder = Encoder(
        config=model.roles_config,
        dim=model.vector_dim,
        seed=model.encoder_seed,
    )
    duration_secs = DURATION_SECONDS.get(payload.duration, 60)
    history_limit = max(payload.history_length, 1)
    rows_query = (
        db.query(models.GlyphTrend)
        .filter(
            models.GlyphTrend.model_id == payload.model_id,
            models.GlyphTrend.role == payload.role,
            models.GlyphTrend.layer == payload.layer,
            models.GlyphTrend.segment_index == payload.segment_index,
            models.GlyphTrend.source != "snapshot",
        )
    )
    if max_timestamp is not None:
        rows_query = rows_query.filter(models.GlyphTrend.timestamp <= max_timestamp)
    rows_query = rows_query.order_by(
        models.GlyphTrend.timestamp.desc(),
        models.GlyphTrend.id.desc(),
    )
    rows_result = rows_query.all()
    logger.debug(
        "analysis_predictions processing %d rows (history limit %d)",
        len(rows_result),
        history_limit,
    )
    if len(rows_result) < 2:
        raise HTTPException(status_code=404, detail="Not enough data to predict")
    rows = list(reversed(rows_result))
    snapshots = [
        GlyphTrendSnapshot(
            timestamp=row.timestamp,
            vector=bytes(row.vector),
            attribute_value=row.attribute_value,
            source=row.source,
        )
        for row in rows
    ]
    history_steps: list[PredictionStep] = []
    history_step_count = 0
    history_value_deltas: list[float] = []
    history_value_modifiers: list[float] = []
    for idx in range(1, len(snapshots)):
        prev_row = snapshots[idx - 1]
        curr_row = snapshots[idx]
        prev_vec_hist = np.frombuffer(prev_row.vector, dtype=np.int8).astype(np.float32)
        curr_vec_hist = np.frombuffer(curr_row.vector, dtype=np.int8).astype(np.float32)
        delta_hist = curr_vec_hist - prev_vec_hist
        denom_hist = np.linalg.norm(prev_vec_hist) * np.linalg.norm(curr_vec_hist)
        similarity_hist = None
        if denom_hist > 0:
            similarity_hist = float(np.dot(prev_vec_hist, curr_vec_hist) / denom_hist)
        delta_norm_hist = float(np.linalg.norm(delta_hist))
        prev_value_hist = parse_numeric_value(prev_row.attribute_value)
        curr_value_hist = parse_numeric_value(curr_row.attribute_value)
        if curr_row.source != "prediction":
            reliability = _history_reliability(
                similarity_hist,
                delta_norm_hist,
                vector_dim=int(prev_vec_hist.size),
            )
            if prev_value_hist is not None and curr_value_hist is not None:
                history_value_deltas.append(curr_value_hist - prev_value_hist)
                history_value_modifiers.append(reliability)
            continue
        history_step_count += 1
        history_steps.append(
            PredictionStep(
                step=-history_step_count,
                direction="history",
                predicted_timestamp=align_timestamp(curr_row.timestamp, payload.duration),
                predicted_value=parse_numeric_value(curr_row.attribute_value),
                similarity_to_previous=similarity_hist,
                delta_norm=float(np.linalg.norm(delta_hist)),
            )
        )
    latest_prediction_step = history_steps[-1] if history_steps else None
    latest_trend_snapshot = None
    latest_prediction_snapshot = None
    latest_trend_index: int | None = None
    latest_prediction_index: int | None = None
    for idx in range(len(snapshots) - 1, -1, -1):
        snapshot = snapshots[idx]
        if latest_prediction_snapshot is None and snapshot.source == "prediction":
            latest_prediction_snapshot = snapshot
            latest_prediction_index = idx
        if latest_trend_snapshot is None and snapshot.source != "prediction":
            latest_trend_snapshot = snapshot
            latest_trend_index = idx
        if latest_prediction_snapshot and latest_trend_snapshot:
            break
    base_snapshot = latest_trend_snapshot or latest_prediction_snapshot or snapshots[-1]
    base_index = (
        latest_trend_index
        if latest_trend_index is not None
        else latest_prediction_index if latest_prediction_index is not None else len(snapshots) - 1
    )
    previous_snapshot = (
        snapshots[base_index - 1] if base_index is not None and base_index > 0 else base_snapshot
    )
    curr_vec = np.frombuffer(base_snapshot.vector, dtype=np.int8).astype(np.float32)
    prev_vec = np.frombuffer(previous_snapshot.vector, dtype=np.int8).astype(np.float32)
    delta = curr_vec - prev_vec
    prev_value = parse_numeric_value(previous_snapshot.attribute_value)
    curr_value = parse_numeric_value(base_snapshot.attribute_value)
    latest_value_delta = None
    if prev_value is not None and curr_value is not None:
        latest_value_delta = curr_value - prev_value
    trend_snapshots = [snap for snap in snapshots if snap.source != "prediction"]
    trend_current_snapshot = trend_snapshots[-1] if trend_snapshots else None
    trend_previous_snapshot = (
        trend_snapshots[-2] if len(trend_snapshots) > 1 else trend_current_snapshot
    )
    trend_curr_vec = (
        np.frombuffer(trend_current_snapshot.vector, dtype=np.int8).astype(np.float32)
        if trend_current_snapshot
        else None
    )
    trend_prev_vec = (
        np.frombuffer(trend_previous_snapshot.vector, dtype=np.int8).astype(np.float32)
        if trend_previous_snapshot
        else None
    )
    history_value_delta = None
    history_value_weight = 0.0
    value_pair = _weighted_scalar(
        history_value_deltas, modifiers=history_value_modifiers
    )
    if value_pair:
        history_value_delta, history_value_weight = value_pair
    trend_value_delta = None
    if trend_current_snapshot and trend_previous_snapshot:
        trend_curr_value = parse_numeric_value(trend_current_snapshot.attribute_value)
        trend_prev_value = parse_numeric_value(trend_previous_snapshot.attribute_value)
        if trend_curr_value is not None and trend_prev_value is not None:
            trend_value_delta = trend_curr_value - trend_prev_value
    effective_value_delta = trend_value_delta if trend_value_delta is not None else latest_value_delta
    if effective_value_delta is not None:
        if history_value_delta is not None:
            value_delta = (
                history_value_delta * history_value_weight + effective_value_delta
            ) / (history_value_weight + 1)
        else:
            value_delta = effective_value_delta
    else:
        value_delta = history_value_delta
    if trend_current_snapshot and trend_previous_snapshot:
        time_delta = max(
            (trend_current_snapshot.timestamp - trend_previous_snapshot.timestamp).total_seconds(),
            1e-6,
        )
    else:
        base_prev_time = previous_snapshot.timestamp
        base_curr_time = base_snapshot.timestamp
        time_delta = max((base_curr_time - base_prev_time).total_seconds(), 1e-6)
    raw_factor = duration_secs / time_delta
    factor = (
        payload.alpha
        if payload.alpha and payload.alpha > 0
        else float(np.clip(raw_factor, 0.5, 1.5))
    )
    trend_delta_vec = None
    if trend_curr_vec is not None and trend_prev_vec is not None:
        trend_delta_vec = trend_curr_vec - trend_prev_vec
    similarity_drift = False
    delta_norm_drift = False
    if (
        payload.similarity_threshold is not None
        and latest_prediction_snapshot
        and trend_curr_vec is not None
    ):
        prediction_snapshot_vec = np.frombuffer(
            latest_prediction_snapshot.vector, dtype=np.int8
        ).astype(np.float32)
        prediction_similarity = _vector_similarity(prediction_snapshot_vec, trend_curr_vec)
        if (
            prediction_similarity is not None
            and prediction_similarity < payload.similarity_threshold
        ):
            similarity_drift = True
    actual_delta_norm = (
        float(np.linalg.norm(trend_delta_vec)) if trend_delta_vec is not None else None
    )
    if (
        payload.delta_norm_threshold is not None
        and latest_prediction_step
        and actual_delta_norm is not None
    ):
        delta_norm_drift = (
            abs(actual_delta_norm - latest_prediction_step.delta_norm)
            > payload.delta_norm_threshold
        )
    num_steps = max(payload.steps, 0)
    predictions: list[PredictionStep] = []
    base_ts = base_snapshot.timestamp
    prediction_is_stale = (
        latest_prediction_snapshot is None
        or (
            latest_trend_snapshot
            and latest_prediction_snapshot
            and latest_trend_snapshot.timestamp > latest_prediction_snapshot.timestamp
        )
        or similarity_drift
        or delta_norm_drift
    )
    should_persist_prediction = force_persist or (
        persist_predictions and prediction_is_stale
    )
    current_vec = curr_vec.copy()
    current_value = curr_value
    # If time has advanced past the last base snapshot, roll the prediction forward so the horizon
    # always stays ahead of "now" even before the next trend arrives.
    if num_steps > 0 and duration_secs > 0:
        aligned_now = align_timestamp(dt.datetime.utcnow(), payload.duration)
        aligned_base = align_timestamp(base_ts, payload.duration)
        elapsed = (aligned_now - aligned_base).total_seconds()
        drift_steps = int(elapsed // duration_secs)
        if drift_steps > 0:
            if current_value is not None and value_delta is not None:
                current_value = current_value + value_delta * factor * drift_steps
                current_vec = _role_value_vector(encoder, payload.role, current_value)
            base_ts = aligned_base + dt.timedelta(seconds=duration_secs * drift_steps)
    for step in range(1, num_steps + 1):
        prev_vec_future = current_vec
        predicted_vec = prev_vec_future
        similarity_step = None
        predicted_ts = align_timestamp(
            base_ts + dt.timedelta(seconds=duration_secs * step), payload.duration
        )
        predicted_value = None
        if current_value is not None and value_delta is not None:
            predicted_value = current_value + value_delta * factor
            current_value = predicted_value
        if predicted_value is not None:
            predicted_vec = _role_value_vector(encoder, payload.role, predicted_value)
        denom = np.linalg.norm(prev_vec_future) * np.linalg.norm(predicted_vec)
        if denom > 0:
            similarity_step = float(np.dot(prev_vec_future, predicted_vec) / denom)
        step_delta_norm = float(np.linalg.norm(predicted_vec - prev_vec_future))
        prediction_cortex = _encode_prediction_cortex(
            encoder,
            payload.role,
            {payload.role: predicted_value},
        )
        if should_persist_prediction:
            persist_prediction_glyph(
                db,
                payload,
                encoder,
                prediction_cortex,
                predicted_ts,
                similarity_step,
                step_delta_norm,
                step=step,
                direction="future",
                attribute_value=predicted_value,
            )
        predictions.append(
            PredictionStep(
                step=step,
                direction="future",
                predicted_timestamp=predicted_ts,
                predicted_value=predicted_value,
                similarity_to_previous=similarity_step,
                delta_norm=step_delta_norm,
            )
        )
        current_vec = predicted_vec
    final_vector = current_vec if num_steps > 0 else curr_vec
    similarity_to_previous = (
        predictions[-1].similarity_to_previous
        if predictions
        else history_steps[-1].similarity_to_previous
        if history_steps
        else None
    )
    delta_norm = (
        predictions[-1].delta_norm if predictions else float(np.linalg.norm(delta))
    )
    response_current_vec = trend_curr_vec if trend_curr_vec is not None else curr_vec
    response_current_vector = [
        int(np.clip(round(float(v)), -128, 127)) for v in response_current_vec
    ]
    final_predictions = predictions[-1:] if predictions else []
    return PredictionResponse(
        model_id=payload.model_id,
        role=payload.role,
        layer=payload.layer,
        segment_index=payload.segment_index,
        predicted_vector=[int(np.clip(round(float(v)), -128, 127)) for v in final_vector],
        current_vector=response_current_vector,
        steps=[*history_steps, *final_predictions],
        similarity_to_previous=similarity_to_previous,
        delta_norm=delta_norm,
        factor=factor,
    )


class MultiRolePredictionResult(NamedTuple):
    responses: list[PredictionResponse]
    aggregated_predicted_vector: list[int]
    aggregated_current_vector: list[int]
    similarity_to_current: float | None


def aggregate_role_predictions(
    db: Session,
    payloads: Sequence[PredictionRequest],
    max_timestamp: dt.datetime | None = None,
    force_persist: bool = False,
    persist_predictions: bool = False,
) -> MultiRolePredictionResult:
    """Combine forecasts from multiple roles into a single glyph-level trend."""
    if not payloads:
        raise ValueError("At least one prediction payload is required")
    model_id = payloads[0].model_id
    if any(payload.model_id != model_id for payload in payloads):
        raise ValueError("All payloads must target the same model")
    responses: list[PredictionResponse] = []
    aggregated_pred_vec: np.ndarray | None = None
    aggregated_current_vec: np.ndarray | None = None
    for payload in payloads:
        response = compute_prediction_response(
            db,
            payload,
            max_timestamp=max_timestamp,
            force_persist=force_persist,
            persist_predictions=persist_predictions,
        )
        responses.append(response)
        pred_array = np.array(response.predicted_vector, dtype=np.float32)
        current_array = (
            np.array(response.current_vector, dtype=np.float32)
            if response.current_vector is not None
            else pred_array.copy()
        )
        if aggregated_pred_vec is None:
            aggregated_pred_vec = pred_array.copy()
            aggregated_current_vec = current_array.copy()
        else:
            if pred_array.shape != aggregated_pred_vec.shape:
                raise ValueError(
                    "Predicted vectors must share the same dimension to combine"
                )
            if current_array.shape != aggregated_current_vec.shape:
                raise ValueError(
                    "Current vectors must share the same dimension to combine"
                )
            aggregated_pred_vec += pred_array
            aggregated_current_vec += current_array
    if aggregated_pred_vec is None or aggregated_current_vec is None:
        raise ValueError("Failed to aggregate prediction vectors")
    similarity = _vector_similarity(aggregated_pred_vec, aggregated_current_vec)
    aggregated_predicted_vector = [
        int(np.clip(round(float(v)), -128, 127)) for v in aggregated_pred_vec
    ]
    aggregated_current_vector = [
        int(np.clip(round(float(v)), -128, 127)) for v in aggregated_current_vec
    ]
    return MultiRolePredictionResult(
        responses=responses,
        aggregated_predicted_vector=aggregated_predicted_vector,
        aggregated_current_vector=aggregated_current_vector,
        similarity_to_current=similarity,
    )
