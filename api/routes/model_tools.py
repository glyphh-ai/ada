from __future__ import annotations

import datetime as dt
import uuid

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..core import models
from ..core.db import get_db
from ..core.schemas import (
    ConceptInput,
    ModelTestsPayload,
    ModelTestsResponse,
    ModelTestsRunResponse,
    SimilarityIngestRequest,
    SimilarityReportRequest,
    SimilarityReportResponse,
    LineageWindowResponse,
)
from ..services.auth_runtime import enforce_model_access, require_scopes
from ..services.similarity import build_similarity_report
from ..services.temporal_sidecar import ensure_temporal_sidecar_model
from ..services.ingest import ingest_concepts
from ..services.nl_helpers import build_nl_configs, to_nl_glyphs, to_simple_glyphs
from ..services.lineage import get_lineage_window
from .router import api_router
from glyphh.encoder import Encoder
from glyphh.metrics import compute_model_metrics
from glyphh.nl.tests import run_tests as run_model_tests_package


router = api_router(tags=["model-tools"])


@router.get("/models/{model_id}/metrics")
def model_metrics(
    model_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["model:read"])
    enforce_model_access(claims, model_id)
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    glyphs = db.query(models.Glyph).filter(models.Glyph.model_id == model.id).all()
    return compute_model_metrics(model.roles_config or {}, to_simple_glyphs(glyphs))


@router.get("/models/{model_id}/tests", response_model=ModelTestsResponse)
def get_model_tests(
    model_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ModelTestsResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["model:read"])
    enforce_model_access(claims, model_id)
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    tests_row = db.query(models.ModelTests).filter(models.ModelTests.model_id == model.id).first()
    if not tests_row:
        return ModelTestsResponse(
            model_id=model.id,
            tests=[],
            version=0,
            updated_at=model.updated_at or dt.datetime.utcnow(),
        )
    tests_payload = tests_row.tests
    tests_list = tests_payload.get("tests", []) if isinstance(tests_payload, dict) else tests_payload or []
    return ModelTestsResponse(
        model_id=model.id,
        tests=tests_list,
        version=tests_row.version,
        updated_at=tests_row.updated_at,
    )


@router.post("/models/{model_id}/tests", response_model=ModelTestsResponse)
def upsert_model_tests(
    model_id: str,
    payload: ModelTestsPayload,
    request: Request,
    db: Session = Depends(get_db),
) -> ModelTestsResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["model:write"])
    enforce_model_access(claims, model_id)
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    tests_row = db.query(models.ModelTests).filter(models.ModelTests.model_id == model.id).first()
    if tests_row:
        tests_row.tests = {"tests": [t.dict() for t in payload.tests]}
        tests_row.version = (tests_row.version or 1) + 1
    else:
        tests_row = models.ModelTests(
            id=str(uuid.uuid4()),
            model_id=model.id,
            tests={"tests": [t.dict() for t in payload.tests]},
            version=1,
        )
        db.add(tests_row)
    db.commit()
    db.refresh(tests_row)
    return ModelTestsResponse(
        model_id=model.id,
        tests=payload.tests,
        version=tests_row.version,
        updated_at=tests_row.updated_at,
    )


@router.post("/models/{model_id}/tests/run", response_model=ModelTestsRunResponse)
def run_model_tests(
    model_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ModelTestsRunResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["model:write"])
    enforce_model_access(claims, model_id)
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    tests_row = db.query(models.ModelTests).filter(models.ModelTests.model_id == model.id).first()
    if not tests_row:
        raise HTTPException(status_code=404, detail="Tests not found")
    glyphs = db.query(models.Glyph).filter(models.Glyph.model_id == model.id).all()
    nl_configs = build_nl_configs(model, db)
    tests_payload = tests_row.tests
    tests_list = tests_payload.get("tests", []) if isinstance(tests_payload, dict) else tests_payload or []
    results = run_model_tests_package(
        tests=tests_list,
        roles_config=model.roles_config or {},
        nl_configs=nl_configs,
        glyphs=to_nl_glyphs(glyphs, db, model_id=model.id),
    )
    return ModelTestsRunResponse(results=results)


@router.post("/models/{model_id}/similarity/report", response_model=SimilarityReportResponse)
def model_similarity_report(
    model_id: str,
    payload: SimilarityReportRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> SimilarityReportResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["model:read"])
    enforce_model_access(claims, model_id)
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    glyph_rows = db.query(models.Glyph).filter(models.Glyph.model_id == model.id).all()
    encoder = Encoder(
        config=model.roles_config or {},
        dim=model.vector_dim,
        seed=model.encoder_seed,
    )
    expected_space_id = None
    if hasattr(encoder, "space_id"):
        space_id_attr = encoder.space_id
        expected_space_id = space_id_attr() if callable(space_id_attr) else space_id_attr
        if not isinstance(expected_space_id, str) or not expected_space_id:
            expected_space_id = None
    report = build_similarity_report(
        ((g.name, g.semantic or {}, g.cortex) for g in glyph_rows),
        groups=[g.model_dump() for g in payload.groups],
        pairs=[(p.left, p.right) for p in (payload.pairs or [])],
        expected_space_id=expected_space_id,
        enforce_single_space=payload.enforce_single_space,
    )
    return SimilarityReportResponse(**report)


@router.post("/models/{model_id}/similarity/ingest")
def ingest_similarity_sidecar(
    model_id: str,
    payload: SimilarityIngestRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["model:write"])
    enforce_model_access(claims, model_id)
    primary_model = db.get(models.Model, model_id)
    if not primary_model:
        raise HTTPException(status_code=404, detail="Model not found")
    sidecar_model = ensure_temporal_sidecar_model(db)
    concepts = []
    for entry in payload.entries:
        ts = entry.timestamp
        if isinstance(ts, dt.datetime):
            ts_value = ts.isoformat()
        else:
            ts_value = str(ts)
        name = f"{model_id}:{entry.test_name}:{ts_value}"
        attributes = {
            "model_id": model_id,
            "test_name": entry.test_name,
            "timestamp": ts_value,
            "cortex_similarity": entry.cortex_similarity,
        }
        if entry.role:
            attributes["role"] = entry.role
        if entry.segment:
            attributes["segment"] = entry.segment
        if entry.layer_similarity is not None:
            attributes["layer_similarity"] = entry.layer_similarity
        if entry.segment_similarity is not None:
            attributes["segment_similarity"] = entry.segment_similarity
        if entry.intent_name:
            attributes["intent_name"] = entry.intent_name
        if entry.notes:
            attributes["notes"] = entry.notes
        concepts.append(ConceptInput(name=name, attributes=attributes, node_type="similarity"))
    return ingest_concepts(sidecar_model, concepts, payload.clear_existing, db)


@router.get("/models/{model_id}/lineage", response_model=LineageWindowResponse)
def get_model_lineage(
    model_id: str,
    name: str,
    request: Request,
    observed_at: str | None = None,
    db: Session = Depends(get_db),
) -> LineageWindowResponse:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    require_scopes(claims, ["model:read"])
    enforce_model_access(claims, model_id)
    model = db.get(models.Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    payload = get_lineage_window(db, model_id=model.id, name=name, observed_at=observed_at)
    return LineageWindowResponse(**payload)
