from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from collections import defaultdict
from typing import List

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from glyphh.encoder import Encoder
from glyphh.vector import bind
import numpy as np
from ..core import models
from ..core.durations import align_timestamp
from ..core.schemas import (
    ConceptInput,
    GlyphSummary,
    ModelRefreshResponse,
    QueryResult,
)

logger = logging.getLogger(__name__)


def _concept_identity(name: str, observed_at: str | None) -> str:
    if not observed_at:
        return name
    return f"{name}@{observed_at}"


def _primary_id_role(roles_config: dict | None) -> str | None:
    if not isinstance(roles_config, dict):
        return None
    for layer in roles_config.get("layers", []):
        if not isinstance(layer, dict):
            continue
        for segment in layer.get("segments", []):
            if not isinstance(segment, dict):
                continue
            for role_entry in segment.get("roles", []):
                if not isinstance(role_entry, dict):
                    continue
                if role_entry.get("primary_id") is True:
                    role_name = role_entry.get("role")
                    if isinstance(role_name, str) and role_name:
                        return role_name
    return None


def _required_roles(roles_config: dict | None) -> list[str]:
    required: list[str] = []
    if not isinstance(roles_config, dict):
        return required
    for layer in roles_config.get("layers", []):
        if not isinstance(layer, dict):
            continue
        for segment in layer.get("segments", []):
            if not isinstance(segment, dict):
                continue
            for role_entry in segment.get("roles", []):
                if not isinstance(role_entry, dict):
                    continue
                if role_entry.get("required") is True:
                    role_name = role_entry.get("role")
                    if isinstance(role_name, str) and role_name:
                        required.append(role_name)
    return list(dict.fromkeys(required))


def _finalize_policy(roles_config: dict | None) -> tuple[list[str], int | None]:
    required_roles = _required_roles(roles_config)
    timebox_seconds: int | None = None
    if isinstance(roles_config, dict):
        policy = roles_config.get("finalize_policy") or {}
        if isinstance(policy, dict):
            policy_roles = policy.get("required_roles")
            if isinstance(policy_roles, list) and policy_roles:
                required_roles = [str(role) for role in policy_roles if str(role)]
            if policy.get("timebox_seconds") is not None:
                try:
                    timebox_seconds = int(policy["timebox_seconds"])
                except (TypeError, ValueError):
                    logger.warning("Invalid timebox_seconds in finalize_policy: %s", policy["timebox_seconds"])
    return required_roles, timebox_seconds


def _coerce_observed_at(value: str | dt.datetime | None) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError:
            logger.warning("Invalid observed_at '%s', defaulting to now()", value)
    return dt.datetime.utcnow()


def stage_concepts(
    model: models.Model,
    concepts: List[ConceptInput],
    listener_id: str,
    db: Session,
) -> None:
    primary_role = _primary_id_role(model.roles_config or {})
    for concept in concepts:
        attributes = dict(concept.attributes or {})
        observed_at = _coerce_observed_at(attributes.get("observed_at"))
        primary_value = attributes.get(primary_role) if primary_role else None
        primary_id = str(primary_value) if primary_value not in (None, "") else concept.name
        record = (
            db.query(models.ModelStagedConcept)
            .filter(
                models.ModelStagedConcept.model_id == model.id,
                models.ModelStagedConcept.primary_id == primary_id,
                models.ModelStagedConcept.observed_at == observed_at,
            )
            .one_or_none()
        )
        payload = dict(record.payload) if record and record.payload else {}
        listeners_payload = dict(payload.get("listeners", {}))
        listeners_payload[listener_id] = attributes
        payload["listeners"] = listeners_payload
        if record:
            record.payload = payload
        else:
            db.add(
                models.ModelStagedConcept(
                    id=str(uuid.uuid4()),
                    model_id=model.id,
                    primary_id=primary_id,
                    observed_at=observed_at,
                    payload=payload,
                )
            )
    db.flush()


def finalize_staged_concepts(
    model: models.Model,
    concepts: List[ConceptInput],
    db: Session,
) -> List[ConceptInput]:
    required_roles, timebox_seconds = _finalize_policy(model.roles_config or {})
    primary_role = _primary_id_role(model.roles_config or {})
    now = dt.datetime.utcnow()
    final_concepts: List[ConceptInput] = []
    for concept in concepts:
        attributes = dict(concept.attributes or {})
        observed_at = _coerce_observed_at(attributes.get("observed_at"))
        primary_value = attributes.get(primary_role) if primary_role else None
        primary_id = str(primary_value) if primary_value not in (None, "") else concept.name
        record = (
            db.query(models.ModelStagedConcept)
            .filter(
                models.ModelStagedConcept.model_id == model.id,
                models.ModelStagedConcept.primary_id == primary_id,
                models.ModelStagedConcept.observed_at == observed_at,
            )
            .one_or_none()
        )
        if not record or not isinstance(record.payload, dict):
            continue
        listeners_payload = record.payload.get("listeners")
        if not isinstance(listeners_payload, dict):
            continue
        merged: dict[str, object] = {}
        for listener_key in sorted(listeners_payload.keys()):
            payload = listeners_payload.get(listener_key)
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                if key not in merged:
                    merged[key] = value
        if "observed_at" not in merged:
            merged["observed_at"] = record.observed_at.isoformat()
        required_ready = not required_roles or all(
            role in merged and merged[role] not in (None, "") for role in required_roles
        )
        timebox_ready = (
            timebox_seconds is not None
            and (now - (record.updated_at or record.created_at)).total_seconds() >= timebox_seconds
        )
        if required_roles or timebox_seconds is not None:
            ready = required_ready or timebox_ready
        else:
            ready = True
        if not ready:
            continue
        final_concepts.append(ConceptInput(name=str(primary_id), attributes=merged))
        db.delete(record)
    db.flush()
    return final_concepts


def _clear_model_glyphs(model: models.Model, db: Session) -> None:
    glyph_names = [
        name
        for (name,) in db.query(models.Glyph.name)
        .filter(models.Glyph.model_id == model.id)
        .all()
    ]
    if not glyph_names:
        return
    chunk_size = 1000
    for i in range(0, len(glyph_names), chunk_size):
        chunk = glyph_names[i : i + chunk_size]
        db.query(models.Edge).filter(
            models.Edge.model_id == model.id,
            models.Edge.source.in_(chunk),
        ).delete(
            synchronize_session=False
        )
        db.query(models.Embedding).filter(
            models.Embedding.model_id == model.id,
            models.Embedding.glyph_name.in_(chunk),
        ).delete(
            synchronize_session=False
        )
        db.query(models.Segment).filter(
            models.Segment.model_id == model.id,
            models.Segment.glyph_name.in_(chunk),
        ).delete(
            synchronize_session=False
        )
        db.query(models.Glyph).filter(
            models.Glyph.model_id == model.id,
            models.Glyph.name.in_(chunk),
        ).delete(
            synchronize_session=False
        )
    db.query(models.GlyphTrend).filter(models.GlyphTrend.model_id == model.id).delete(
        synchronize_session=False
    )
    db.flush()


def _clear_future_predictions(
    db: Session,
    model_id: str,
    role: str,
    layer: int,
    segment_index: int,
    from_timestamp: dt.datetime,
) -> None:
    slot_predictions = (
        db.query(models.GlyphTrend)
        .filter(
            models.GlyphTrend.model_id == model_id,
            models.GlyphTrend.role == role,
            models.GlyphTrend.layer == layer,
            models.GlyphTrend.segment_index == segment_index,
            models.GlyphTrend.source == "prediction",
            models.GlyphTrend.timestamp == from_timestamp,
        )
        .order_by(models.GlyphTrend.id.desc())
        .all()
    )
    duplicate_ids = []
    if slot_predictions:
        slot_predictions[0].aligned_to_actual = True
        duplicate_ids = [pred.id for pred in slot_predictions[1:]]
    if duplicate_ids:
        (
            db.query(models.GlyphTrend)
            .filter(models.GlyphTrend.id.in_(duplicate_ids))
            .delete(synchronize_session=False)
        )
    (
        db.query(models.GlyphTrend)
        .filter(
            models.GlyphTrend.model_id == model_id,
            models.GlyphTrend.role == role,
            models.GlyphTrend.layer == layer,
            models.GlyphTrend.segment_index == segment_index,
            models.GlyphTrend.source == "prediction",
            models.GlyphTrend.timestamp > from_timestamp,
        )
        .delete(synchronize_session=False)
    )
    (
        db.query(models.GlyphTrend)
        .filter(
            models.GlyphTrend.model_id == model_id,
            models.GlyphTrend.role == role,
            models.GlyphTrend.layer == layer,
            models.GlyphTrend.segment_index == segment_index,
            models.GlyphTrend.source == "prediction",
            models.GlyphTrend.aligned_to_actual.is_(False),
            models.GlyphTrend.timestamp < from_timestamp,
        )
        .delete(synchronize_session=False)
    )


def ingest_concepts(model: models.Model, concepts: List[ConceptInput], clear_existing: bool, db: Session) -> QueryResult:
    encoder = Encoder(
        config=model.roles_config,
        dim=model.vector_dim,
        seed=model.encoder_seed,
    )
    trend_defs = (
        db.query(models.TrendDefinition)
        .filter(
            models.TrendDefinition.model_id == model.id,
            models.TrendDefinition.enabled.is_(True),
        )
        .all()
    )
    role_to_trends: dict[str, list[models.TrendDefinition]] = defaultdict(list)
    for trend in trend_defs:
        for role_name in trend.roles:
            role_to_trends[role_name].append(trend)
    if clear_existing:
        _clear_model_glyphs(model, db)

    primary_role = _primary_id_role(model.roles_config or {})
    key_versions: dict[str, list[str | None]] = defaultdict(list)
    for concept in concepts:
        observed_at_raw = (concept.attributes or {}).get("observed_at")
        identity_key = concept.name
        if primary_role:
            primary_value = (concept.attributes or {}).get(primary_role)
            if primary_value not in (None, ""):
                identity_key = str(primary_value)
        key_versions[identity_key].append(observed_at_raw)
    for name, observed_list in key_versions.items():
        if len(observed_list) > 1 and any(not value for value in observed_list):
            raise HTTPException(
                status_code=400,
                detail=f"Concept '{name}' must include observed_at for all versions.",
            )

    seen_names: set[str] = set()
    for concept in concepts:
        observed_at_raw = (concept.attributes or {}).get("observed_at")
        identity_name = _concept_identity(concept.name, observed_at_raw)
        if identity_name in seen_names:
            continue
        seen_names.add(identity_name)

        attributes = dict(concept.attributes or {})
        if "observed_at" not in attributes:
            attributes["observed_at"] = dt.datetime.utcnow().isoformat()

        glyph = encoder.encode(identity_name, attributes, node_type=concept.node_type)
        if concept.taxonomy:
            glyph.semantic = dict(glyph.semantic)
            glyph.semantic["taxonomy"] = concept.taxonomy

        existing = db.get(models.Glyph, identity_name)
        if existing:
            db.query(models.Edge).filter(
                models.Edge.model_id == model.id,
                models.Edge.source == identity_name,
            ).delete()
            db.query(models.Embedding).filter(
                models.Embedding.model_id == model.id,
                models.Embedding.glyph_name == identity_name,
            ).delete()
            db.query(models.Segment).filter(
                models.Segment.model_id == model.id,
                models.Segment.glyph_name == identity_name,
            ).delete()
            db.delete(existing)
            db.flush()

        g_row = models.Glyph(
            name=identity_name,
            model_id=model.id,
            node_type=concept.node_type,
            semantic=glyph.semantic,
            cortex=glyph.global_cortex.tobytes(),
        )
        db.add(g_row)
        db.flush()

        embedding_vec = glyph.global_cortex.astype(float).tolist()
        db.add(
            models.Embedding(
                glyph_name=identity_name,
                model_id=model.id,
                embedding=embedding_vec,
                node_type=concept.node_type,
                meta={},
            )
        )

        for li, layer in glyph.layers.items():
            for si, seg_vec in enumerate(layer.segments):
                db.add(
                    models.Segment(
                        glyph_name=identity_name,
                        model_id=model.id,
                        layer=li,
                        seg_index=si,
                        vec=seg_vec.tobytes(),
                    )
                )
        if concept.edges:
            for edge in concept.edges:
                db.add(
                    models.Edge(
                        model_id=model.id,
                        source=identity_name,
                        target=edge.target,
                        type=edge.type,
                        weight=edge.weight,
                        layer=edge.layer,
                    )
                )
        for role_name, raw_value in attributes.items():
            slot = encoder.role_to_slot.get(role_name)
            if slot is None:
                continue
            layer_idx, seg_idx = slot
            if raw_value is None:
                value_token = "null"
            elif isinstance(raw_value, (str, bool, int, float)):
                value_token = str(raw_value)
            else:
                value_token = json.dumps(raw_value, sort_keys=True)
            bound_vec = bind(encoder.role_vec(role_name), encoder.value_vec(value_token))
            base_trend = models.GlyphTrend(
                model_id=model.id,
                role=role_name,
                layer=layer_idx,
                segment_index=seg_idx,
                vector=bound_vec.tobytes(),
                source="trend",
                attribute_value=raw_value,
            )
            db.add(base_trend)
            for trend in role_to_trends.get(role_name, []):
                aligned_ts = align_timestamp(dt.datetime.utcnow(), trend.duration)
                _clear_future_predictions(
                    db, model.id, role_name, layer_idx, seg_idx, aligned_ts
                )
                (
                    db.query(models.GlyphTrend)
                    .filter(
                        models.GlyphTrend.model_id == model.id,
                        models.GlyphTrend.role == role_name,
                        models.GlyphTrend.layer == layer_idx,
                        models.GlyphTrend.segment_index == seg_idx,
                        models.GlyphTrend.trend_definition_id == trend.id,
                        models.GlyphTrend.timestamp == aligned_ts,
                        # Preserve prediction snapshots at this timestamp; they are used to
                        # compare "what was predicted" vs the newly-ingested actual.
                        models.GlyphTrend.source == "trend",
                    )
                    .delete(synchronize_session=False)
                )
                trend_row = models.GlyphTrend(
                    model_id=model.id,
                    role=role_name,
                    layer=layer_idx,
                    segment_index=seg_idx,
                    vector=bound_vec.tobytes(),
                    source="trend",
                    trend_definition_id=trend.id,
                    attribute_value=raw_value,
                    timestamp=aligned_ts,
                )
                db.add(trend_row)
    db.flush()

    glyph_rows = (
        db.query(models.Glyph)
        .filter(models.Glyph.model_id == model.id)
        .order_by(models.Glyph.created_at.desc())
        .limit(10)
        .all()
    )
    summaries = [
        GlyphSummary(name=g.name, node_type=g.node_type, semantic=g.semantic, model_id=model.id)
        for g in glyph_rows
    ]
    db.commit()
    return QueryResult(matches=summaries)


def refresh_model_data(model: models.Model, db: Session) -> ModelRefreshResponse:
    glyph_rows = (
        db.query(models.Glyph)
        .filter(models.Glyph.model_id == model.id)
        .all()
    )
    existing_embeddings = {
        row.glyph_name
        for row in db.query(models.Embedding.glyph_name)
        .filter(models.Embedding.model_id == model.id)
        .all()
    }
    for glyph in glyph_rows:
        if glyph.name in existing_embeddings:
            continue
        if not glyph.cortex:
            continue
        vec = np.frombuffer(glyph.cortex, dtype=np.int8).astype(float).tolist()
        db.add(
            models.Embedding(
                glyph_name=glyph.name,
                model_id=model.id,
                embedding=vec,
                node_type=glyph.node_type,
                meta={},
            )
        )
    db.flush()

    glyph_count = (
        db.query(func.count(models.Glyph.name))
        .filter(models.Glyph.model_id == model.id)
        .scalar()
        or 0
    )
    model.updated_at = dt.datetime.utcnow()
    db.commit()
    return ModelRefreshResponse(model_id=model.id, glyph_count=glyph_count, status="refreshed")
