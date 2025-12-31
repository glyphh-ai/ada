from __future__ import annotations

import numpy as np
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core import models
from ..core.db import get_db
from ..core.schemas import QueryRequest, QueryResult, GlyphSummary
from ..services.usage_runtime import record_usage
from .router import api_router


router = api_router(tags=["query"])


@router.post("/query", response_model=QueryResult)
def query(
    payload: QueryRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> QueryResult:
    model = db.get(models.Model, payload.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    record_usage(request.app, "glyph_crud", count=1)

    if payload.glyph_name:
        emb_row = db.get(models.Embedding, payload.glyph_name)
        if not emb_row:
            raise HTTPException(status_code=404, detail="Glyph embedding not found")
        query_vec = np.array(emb_row.embedding, dtype=float).tolist()

        stmt = (
            select(models.Glyph)
            .join(models.Embedding, models.Glyph.name == models.Embedding.glyph_name)
            .where(models.Glyph.model_id == payload.model_id)
            .order_by(models.Embedding.embedding.l2_distance(query_vec))
            .limit(payload.top_k)
        )
        glyphs = db.execute(stmt).scalars().all()
    else:
        glyphs = (
            db.query(models.Glyph)
            .filter(models.Glyph.model_id == payload.model_id)
            .order_by(models.Glyph.created_at.desc())
            .limit(payload.top_k)
            .all()
        )

    if not glyphs:
        raise HTTPException(status_code=404, detail="No glyphs for model")

    summaries = [
        GlyphSummary(
            name=g.name, node_type=g.node_type, semantic=g.semantic, model_id=model.id
        )
        for g in glyphs
    ]
    return QueryResult(matches=summaries)
