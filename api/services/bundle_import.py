from __future__ import annotations

import json
import uuid
import logging
from typing import Any

from sqlalchemy.orm import Session

from ..core import models
from ..core.schemas import ConceptInput, SampleImportResponse, SampleImportResult, SampleUploadBundle
from ..core.vector_space import build_vector_space_metadata
from ..services.ingest import ingest_concepts
from ..services.roles_config_helpers import validate_roles_config

logger = logging.getLogger(__name__)


def _bundle_entry_content(content: Any, *, expect_object: bool = True) -> Any:
    if isinstance(content, bytes):
        text = content.decode("utf-8")
    elif isinstance(content, str):
        text = content
    else:
        text = None
    if text is not None:
        parsed = json.loads(text)
    else:
        parsed = content
    if expect_object and not isinstance(parsed, dict):
        raise ValueError("Bundle entry content must be a JSON object")
    return parsed


def _create_model_from_roles(
    db: Session, roles_config: dict, model_meta: dict | None
) -> models.Model:
    model_id = str(uuid.uuid4())
    space_meta = build_vector_space_metadata(roles_config=roles_config or {})
    model = models.Model(
        id=model_id,
        name="Imported Model",
        description="Created from bundle import",
        version=1,
        roles_config=roles_config or {},
        vector_dim=space_meta["vector_dim"],
        encoder_seed=space_meta["encoder_seed"],
        space_id=space_meta["space_id"],
        space_version=space_meta["space_version"],
        data_mode="live",
        status="draft",
    )
    if model_meta:
        if isinstance(model_meta.get("name"), str):
            model.name = model_meta.get("name") or model.name
        if "description" in model_meta:
            model.description = model_meta.get("description")
        if isinstance(model_meta.get("status"), str):
            model.status = model_meta.get("status") or model.status
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def _create_encoder_from_json(db: Session, data: dict) -> models.Encoder:
    kind = (data.get("type") or "nl").lower()
    enc = models.Encoder(
        id=str(uuid.uuid4()),
        name=data.get("name") or "Imported Encoder",
        domain=data.get("domain") or "aligned",
        description=data.get("description"),
        type=kind,
        status=data.get("status") or "draft",
        version=data.get("version") or 1,
        is_public=bool(data.get("is_public")),
        config=data.get("config") or {},
    )
    db.add(enc)
    db.commit()
    db.refresh(enc)
    return enc


def _create_runtime_config(db: Session, model_id: str, data: dict) -> models.ModelRuntimeConfig:
    cfg = models.ModelRuntimeConfig(
        id=str(uuid.uuid4()),
        model_id=model_id,
        config=data or {},
        version=1,
    )
    db.add(cfg)
    db.commit()
    db.refresh(cfg)
    return cfg


def _create_tests(db: Session, model_id: str, data: dict) -> models.ModelTests:
    tests = models.ModelTests(
        id=str(uuid.uuid4()),
        model_id=model_id,
        tests=data.get("tests") or [],
        version=1,
    )
    db.add(tests)
    db.commit()
    db.refresh(tests)
    return tests


def _extract_concepts(entry_content: Any) -> list[ConceptInput]:
    if isinstance(entry_content, dict):
        raw = entry_content.get("concepts") or []
    elif isinstance(entry_content, list):
        raw = entry_content
    else:
        raw = []
    concepts: list[ConceptInput] = []
    for item in raw:
        try:
            concepts.append(ConceptInput.model_validate(item))
        except Exception:
            logger.warning("Skipping invalid concept entry during bundle import")
    return concepts


def import_bundle(
    db: Session,
    payload: SampleUploadBundle,
    clear_existing: bool = True,
) -> SampleImportResponse:
    results: list[SampleImportResult] = []
    model: models.Model | None = None
    if not payload.roles_configs:
        return SampleImportResponse(
            results=[
                SampleImportResult(
                    category="roles_configs",
                    file="roles_configs",
                    status="error",
                    detail="No roles config provided",
                )
            ]
        )
    for entry in payload.roles_configs:
        try:
            data = _bundle_entry_content(entry.content)
            validate_roles_config(data)
            model = _create_model_from_roles(db, data, payload.model or {})
            results.append(
                SampleImportResult(
                    category="roles_configs",
                    file=entry.file,
                    status="success",
                )
            )
        except Exception as exc:
            results.append(
                SampleImportResult(
                    category="roles_configs",
                    file=entry.file,
                    status="error",
                    detail=str(exc),
                )
            )

    if model is None:
        return SampleImportResponse(results=results)

    for entry in payload.encoders:
        try:
            data = _bundle_entry_content(entry.content)
            _create_encoder_from_json(db, data)
            results.append(
                SampleImportResult(category="encoders", file=entry.file, status="success")
            )
        except Exception as exc:
            results.append(
                SampleImportResult(
                    category="encoders",
                    file=entry.file,
                    status="error",
                    detail=str(exc),
                )
            )

    for entry in payload.runtime:
        try:
            data = _bundle_entry_content(entry.content)
            _create_runtime_config(db, model.id, data)
            results.append(
                SampleImportResult(category="runtime", file=entry.file, status="success")
            )
        except Exception as exc:
            results.append(
                SampleImportResult(
                    category="runtime",
                    file=entry.file,
                    status="error",
                    detail=str(exc),
                )
            )

    for entry in payload.tests:
        try:
            data = _bundle_entry_content(entry.content)
            _create_tests(db, model.id, data)
            results.append(
                SampleImportResult(category="tests", file=entry.file, status="success")
            )
        except Exception as exc:
            results.append(
                SampleImportResult(
                    category="tests",
                    file=entry.file,
                    status="error",
                    detail=str(exc),
                )
            )

    for entry in payload.concepts:
        try:
            data = _bundle_entry_content(entry.content, expect_object=False)
            concepts = _extract_concepts(data)
            if concepts:
                ingest_concepts(model, concepts, clear_existing, db)
            results.append(
                SampleImportResult(category="concepts", file=entry.file, status="success")
            )
        except Exception as exc:
            results.append(
                SampleImportResult(
                    category="concepts",
                    file=entry.file,
                    status="error",
                    detail=str(exc),
                )
            )

    return SampleImportResponse(results=results)
