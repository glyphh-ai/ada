from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GLYPHH_SDK_ROOT = ROOT / "glyphh-sdk"
if GLYPHH_SDK_ROOT.exists() and str(GLYPHH_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(GLYPHH_SDK_ROOT))

from sqlalchemy import select

from api.core import models
from api.core.db import SessionLocal
from api.core.schemas import QueryRequest, GlyphSummary, QueryResult
from glyphh import Encoder, Glyph, GlyphMemory
from glyphh import reasoning as glyphh_reasoning
from glyphh import vector as glyphh_vector
from glyphh.nl.configs import resolve_nl_configs, run_nl_query
from glyphh.nl.inference import SimpleGlyph as NLInferenceGlyph
from api.services.intent_inference import infer_intent_with_model


class _NullConfigSource:
    def fetch_nl_config(self, identifier: str) -> Dict[str, Any] | None:
        return None

    def can_access(self, identifier: str) -> bool:
        return False


def _build_memory(
    model: models.Model, db: SessionLocal
) -> tuple[GlyphMemory, Encoder, dict[str, Glyph]]:
    encoder = Encoder(
        config=model.roles_config or {},
        dim=model.vector_dim,
        seed=model.encoder_seed,
    )
    memory = GlyphMemory(num_layers=encoder.num_layers)
    glyph_rows = (
        db.query(models.Glyph)
        .filter(models.Glyph.model_id == model.id)
        .all()
    )
    glyph_map: dict[str, Glyph] = {}
    if not glyph_rows:
        return memory, encoder, glyph_map
    names = [g.name for g in glyph_rows]
    segments = (
        db.query(models.Segment)
        .filter(models.Segment.glyph_name.in_(names))
        .all()
    )
    seg_map: dict[tuple[str, int, int], bytes] = {}
    for seg in segments:
        seg_map[(seg.glyph_name, seg.layer, seg.seg_index)] = seg.vec
    for row in glyph_rows:
        glyph = Glyph.empty(
            row.name,
            num_layers=encoder.num_layers,
            num_segments=encoder.num_segments,
            dim=model.vector_dim,
        )
        glyph.node_type = row.node_type
        glyph.semantic = row.semantic or {}
        glyph.global_cortex = np.frombuffer(row.cortex, dtype=np.int8)
        for layer_idx in range(encoder.num_layers):
            glyph.layers[layer_idx].cortex = glyph.global_cortex.copy()
            for seg_idx in range(encoder.num_segments):
                key = (row.name, layer_idx, seg_idx)
                if key in seg_map:
                    glyph.layers[layer_idx].segments[seg_idx] = np.frombuffer(
                        seg_map[key], dtype=np.int8
                    )
        memory.add_glyph(glyph)
        glyph_map[row.name] = glyph
    return memory, encoder, glyph_map


def _nl_glyphs(rows: Iterable[models.Glyph]) -> List[NLInferenceGlyph]:
    return [
        NLInferenceGlyph(
            name=row.name,
            node_type=row.node_type,
            semantic=row.semantic or {},
            cortex=row.cortex,
        )
        for row in rows
    ]


class MCPServer:
    def _health(self) -> Dict[str, Any]:
        return {"status": "ok"}

    def _similar_to(self, payload: Dict[str, Any], db: SessionLocal) -> Dict[str, Any]:
        req = QueryRequest(
            model_id=payload["model_id"],
            glyph_name=payload["glyph_name"],
            top_k=payload.get("top_k", 5),
        )
        model = db.get(models.Model, req.model_id)
        if not model:
            return {"error": "model_not_found"}

        emb_row = db.get(models.Embedding, req.glyph_name)
        if not emb_row:
            return {"error": "glyph_embedding_not_found"}

        query_vec = np.array(emb_row.embedding, dtype=float).tolist()
        stmt = (
            select(models.Glyph)
            .join(models.Embedding, models.Glyph.name == models.Embedding.glyph_name)
            .where(models.Glyph.model_id == req.model_id)
            .order_by(models.Embedding.embedding.l2_distance(query_vec))
            .limit(req.top_k)
        )
        glyphs = db.execute(stmt).scalars().all()
        if not glyphs:
            return {"error": "no_glyphs_for_model"}

        summaries = [
            GlyphSummary(
                name=row.name,
                node_type=row.node_type,
                semantic=row.semantic,
                model_id=model.id,
            )
            for row in glyphs
        ]
        return QueryResult(matches=summaries).model_dump()

    def handle_tool(self, tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            if tool == "nl_query":
                model = db.get(models.Model, payload["model_id"])
                if not model:
                    return {"error": "model_not_found"}
                base_cfg = model.nl_config.config if model.nl_config else None
                nl_configs = resolve_nl_configs(base_cfg, _NullConfigSource())
                glyph_rows = (
                    db.query(models.Glyph)
                    .filter(models.Glyph.model_id == model.id)
                    .all()
                )
                result = run_nl_query(
                    payload["text"],
                    nl_configs,
                    model.roles_config or {},
                    _nl_glyphs(glyph_rows),
                )
                if result:
                    return result
                inferred = infer_intent_with_model(payload["text"], list(nl_configs))
                if inferred:
                    intent_name, score = inferred
                    return {
                        "query": payload["text"],
                        "matched_glyph": None,
                        "intent": intent_name,
                        "intent_score": score,
                        "intent_source": "model_fallback",
                    }
                return {"query": payload["text"], "matched_glyph": None}
            if tool == "find_by_properties":
                model = db.get(models.Model, payload["model_id"])
                if not model:
                    return {"error": "model_not_found"}
                constraints = [
                    (c["role"], c["value"])
                    for c in payload.get("constraints", [])
                    if isinstance(c, dict) and c.get("role") and c.get("value")
                ]
                memory, encoder, _ = _build_memory(model, db)
                results = glyphh_reasoning.find_by_properties(
                    memory, encoder, constraints, top_k=payload.get("top_k", 5)
                )
                return {
                    "matches": [
                        {"name": g.name, "score": score} for g, score in results
                    ]
                }
            if tool == "similar_to":
                return self._similar_to(payload, db)
            if tool == "explain_link":
                model = db.get(models.Model, payload["model_id"])
                if not model:
                    return {"error": "model_not_found"}
                memory, _encoder, glyph_map = _build_memory(model, db)
                src = glyph_map.get(payload["source"])
                tgt = glyph_map.get(payload["target"])
                if not src or not tgt:
                    return {"error": "glyph_not_found"}
                if src.global_cortex is None or tgt.global_cortex is None:
                    return {"error": "missing_cortex_vectors"}
                similarity = glyphh_vector.similarity(src.global_cortex, tgt.global_cortex)
                shared = {
                    k: v
                    for k, v in (src.semantic or {}).items()
                    if (tgt.semantic or {}).get(k) == v
                }
                return {"similarity": similarity, "shared_semantic": shared}
            if tool == "trend_role":
                rows = (
                    db.query(models.GlyphTrend)
                    .filter(
                        models.GlyphTrend.model_id == payload["model_id"],
                        models.GlyphTrend.role == payload["role"],
                    )
                    .order_by(models.GlyphTrend.timestamp.desc())
                    .limit(payload.get("limit", 50))
                    .all()
                )
                return {
                    "role": payload["role"],
                    "entries": [
                        {
                            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                            "value": row.attribute_value,
                            "source": row.source,
                            "layer": row.layer,
                            "segment_index": row.segment_index,
                        }
                        for row in rows
                    ],
                }
            if tool == "predict_next":
                rows = (
                    db.query(models.GlyphTrend)
                    .filter(
                        models.GlyphTrend.model_id == payload["model_id"],
                        models.GlyphTrend.role == payload["role"],
                    )
                    .order_by(models.GlyphTrend.timestamp.desc())
                    .all()
                )
                prediction = next((r for r in rows if r.source == "prediction"), None)
                fallback = rows[0] if rows else None
                row = prediction or fallback
                if not row:
                    return {"status": "no_data"}
                return {
                    "role": payload["role"],
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "value": row.attribute_value,
                    "source": row.source,
                }
            if tool == "what_if_modify":
                model = db.get(models.Model, payload["model_id"])
                if not model:
                    return {"error": "model_not_found"}
                memory, encoder, glyph_map = _build_memory(model, db)
                target = glyph_map.get(payload["glyph_name"])
                if not target:
                    return {"error": "glyph_not_found"}
                patch = payload.get("patch") or {}
                updated = dict(target.semantic or {})
                updated.update(patch)
                new_glyph = encoder.encode(
                    name=target.name,
                    attrs=updated,
                    node_type=target.node_type,
                )
                similarity = glyphh_vector.similarity(
                    new_glyph.global_cortex, target.global_cortex
                )
                return {
                    "glyph": target.name,
                    "semantic": updated,
                    "similarity": similarity,
                    "note": "not_persisted",
                }
            if tool == "health":
                return self._health()
            return {"error": "unknown_tool", "tool": tool}
        finally:
            db.close()


def main() -> None:
    server = MCPServer()
    while True:
        raw = input()
        if not raw:
            continue
        msg = json.loads(raw)
        tool = msg.get("tool")
        payload = msg.get("payload", {})
        resp = server.handle_tool(tool, payload)
        print(json.dumps(resp))


if __name__ == "__main__":
    main()
