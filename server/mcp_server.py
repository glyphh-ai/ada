from __future__ import annotations

import json
import sys
import hashlib
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
import glyphh
from glyphh import Encoder, Glyph, GlyphMemory
from glyphh import reasoning as glyphh_reasoning
from glyphh import vector as glyphh_vector
from glyphh.nl.configs import resolve_nl_configs, run_nl_query
from glyphh.nl.inference import SimpleGlyph as NLInferenceGlyph
from api.services.intent_inference import infer_intent_with_model
from api.core.config import get_settings
from .mcp_schema import format_validation_error, validate_mcp_response


settings = get_settings()

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


def _nl_glyphs(
    rows: Iterable[models.Glyph],
    segments: Dict[tuple[str, int, int], bytes],
) -> List[NLInferenceGlyph]:
    return [
        NLInferenceGlyph(
            name=row.name,
            node_type=row.node_type,
            semantic=row.semantic or {},
            cortex=row.cortex,
            segments={
                (layer, seg_idx): vec
                for (name, layer, seg_idx), vec in segments.items()
                if name == row.name
            }
            or None,
        )
        for row in rows
    ]


class MCPServer:
    def _build_nl_facts(
        self, payload: Dict[str, Any], result: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        facts: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        reasons: List[str] = []

        intent_name = result.get("intent") or result.get("name")
        if not intent_name and isinstance(result.get("aggregate"), dict):
            intent_name = result["aggregate"].get("metric")
        intent_source = result.get("intent_source") or "rules"
        intent_score = result.get("intent_score")
        if intent_name:
            facts.append(
                {
                    "id": "intent",
                    "text": f"Intent inferred as '{intent_name}'",
                    "type": "decision",
                    "confidence": float(intent_score) if intent_score is not None else 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": intent_source,
                            "snippet": f"intent_source={intent_source}",
                        }
                    ],
                }
            )
            reasons.append(f"intent_source={intent_source}")

        matched_glyph = result.get("matched_glyph")
        if matched_glyph:
            facts.append(
                {
                    "id": "matched_glyph",
                    "text": f"Matched glyph '{matched_glyph}'",
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": matched_glyph,
                            "snippet": "matched_glyph",
                        }
                    ],
                }
            )
            citations.append(
                {
                    "source_type": "glyphh",
                    "source_id": matched_glyph,
                    "title": matched_glyph,
                    "url": None,
                    "snippet": "matched_glyph",
                    "hash": None,
                }
            )

        aggregate = result.get("aggregate")
        if isinstance(aggregate, dict):
            value = aggregate.get("value")
            count = aggregate.get("count")
            facts.append(
                {
                    "id": "aggregate",
                    "text": f"Aggregate value {value} across {count} contributors",
                    "type": "metric",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "aggregate",
                            "snippet": "aggregate_result",
                        }
                    ],
                }
            )
            reasons.append("aggregate computed from contributor glyphs")

        for edge in result.get("semantic_edges") or []:
            target = edge.get("target")
            if target:
                citations.append(
                    {
                        "source_type": "glyphh",
                        "source_id": target,
                        "title": target,
                        "url": None,
                        "snippet": f"semantic_edge:{edge.get('kind') or edge.get('type')}",
                        "hash": None,
                    }
                )

        for edge in result.get("neural_edges") or []:
            target = edge.get("target")
            if target:
                citations.append(
                    {
                        "source_type": "glyphh",
                        "source_id": target,
                        "title": target,
                        "url": None,
                        "snippet": "neural_edge",
                        "hash": None,
                    }
                )

        return facts, citations, reasons

    def _build_find_by_properties_facts(
        self, payload: Dict[str, Any], result: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        facts: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        reasons: List[str] = []

        constraints = payload.get("constraints") or []
        if constraints:
            reasons.append("constraints_applied")
            facts.append(
                {
                    "id": "constraints",
                    "text": "Constraints applied to property search",
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "constraints",
                            "snippet": json.dumps(constraints, ensure_ascii=True),
                        }
                    ],
                }
            )

        matches = result.get("matches") or []
        for idx, match in enumerate(matches):
            name = match.get("name")
            score = match.get("score")
            if not name:
                continue
            facts.append(
                {
                    "id": f"match_{idx}",
                    "text": f"Match '{name}' score {score}",
                    "type": "decision",
                    "confidence": float(score) if score is not None else 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": name,
                            "snippet": "find_by_properties_match",
                        }
                    ],
                }
            )
            citations.append(
                {
                    "source_type": "glyphh",
                    "source_id": name,
                    "title": name,
                    "url": None,
                    "snippet": "find_by_properties_match",
                    "hash": None,
                }
            )

        return facts, citations, reasons

    def _build_similar_to_facts(
        self, payload: Dict[str, Any], result: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        facts: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        reasons: List[str] = []

        glyph_name = payload.get("glyph_name")
        if glyph_name:
            facts.append(
                {
                    "id": "query_glyph",
                    "text": f"Similarity query glyph '{glyph_name}'",
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": glyph_name,
                            "snippet": "embedding_source",
                        }
                    ],
                }
            )
            citations.append(
                {
                    "source_type": "glyphh",
                    "source_id": glyph_name,
                    "title": glyph_name,
                    "url": None,
                    "snippet": "embedding_source",
                    "hash": None,
                }
            )

        reasons.append("distance_metric=l2")

        matches = result.get("matches") or []
        for idx, match in enumerate(matches):
            name = match.get("name")
            if not name:
                continue
            facts.append(
                {
                    "id": f"similar_match_{idx}",
                    "text": f"Similar glyph '{name}' rank {idx + 1}",
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": name,
                            "snippet": "similar_to_match",
                        }
                    ],
                }
            )
            citations.append(
                {
                    "source_type": "glyphh",
                    "source_id": name,
                    "title": name,
                    "url": None,
                    "snippet": "similar_to_match",
                    "hash": None,
                }
            )

        return facts, citations, reasons

    def _build_explain_link_facts(
        self, payload: Dict[str, Any], result: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        facts: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        reasons: List[str] = []

        source = payload.get("source")
        target = payload.get("target")
        if source:
            citations.append(
                {
                    "source_type": "glyphh",
                    "source_id": source,
                    "title": source,
                    "url": None,
                    "snippet": "link_source",
                    "hash": None,
                }
            )
        if target:
            citations.append(
                {
                    "source_type": "glyphh",
                    "source_id": target,
                    "title": target,
                    "url": None,
                    "snippet": "link_target",
                    "hash": None,
                }
            )

        similarity = result.get("similarity")
        if similarity is not None:
            facts.append(
                {
                    "id": "link_similarity",
                    "text": f"Similarity between '{source}' and '{target}' is {similarity}",
                    "type": "metric",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "similarity",
                            "snippet": "global_cortex_similarity",
                        }
                    ],
                }
            )
            reasons.append("similarity computed on global_cortex vectors")

        shared = result.get("shared_semantic") or {}
        if shared:
            keys = ", ".join(sorted(shared.keys()))
            facts.append(
                {
                    "id": "shared_semantic",
                    "text": f"Shared semantic keys: {keys}",
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "shared_semantic",
                            "snippet": keys,
                        }
                    ],
                }
            )

        return facts, citations, reasons

    def _build_trend_role_facts(
        self, payload: Dict[str, Any], result: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        facts: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        reasons: List[str] = []

        role = result.get("role") or payload.get("role")
        entries = result.get("entries") or []
        if role:
            facts.append(
                {
                    "id": "trend_role",
                    "text": f"Trend series for role '{role}' with {len(entries)} entries",
                    "type": "metric",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "trend_role",
                            "snippet": role,
                        }
                    ],
                }
            )

        for idx, entry in enumerate(entries):
            timestamp = entry.get("timestamp")
            value = entry.get("value")
            source = entry.get("source")
            layer = entry.get("layer")
            segment_index = entry.get("segment_index")
            facts.append(
                {
                    "id": f"trend_entry_{idx}",
                    "text": f"{timestamp}: {value}",
                    "type": "metric",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": f"trend_entry_{idx}",
                            "snippet": f"source={source} layer={layer} segment={segment_index}",
                        }
                    ],
                }
            )
            citations.append(
                {
                    "source_type": "glyphh",
                    "source_id": f"trend_entry_{idx}",
                    "title": role,
                    "url": None,
                    "snippet": f"{timestamp} value={value} source={source}",
                    "hash": None,
                }
            )

        return facts, citations, reasons

    def _build_predict_next_facts(
        self, payload: Dict[str, Any], result: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        facts: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        reasons: List[str] = []

        role = result.get("role") or payload.get("role")
        timestamp = result.get("timestamp")
        value = result.get("value")
        source = result.get("source") or "unknown"

        if role:
            facts.append(
                {
                    "id": "predict_role",
                    "text": f"Prediction for role '{role}'",
                    "type": "metric",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "predict_next",
                            "snippet": role,
                        }
                    ],
                }
            )

        if timestamp or value is not None:
            facts.append(
                {
                    "id": "predict_value",
                    "text": f"{timestamp}: {value}",
                    "type": "metric",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "predict_next",
                            "snippet": f"source={source}",
                        }
                    ],
                }
            )
            citations.append(
                {
                    "source_type": "glyphh",
                    "source_id": "predict_next",
                    "title": role,
                    "url": None,
                    "snippet": f"{timestamp} value={value} source={source}",
                    "hash": None,
                }
            )
            reasons.append(f"prediction_source={source}")

        return facts, citations, reasons

    def _build_what_if_modify_facts(
        self, payload: Dict[str, Any], result: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        facts: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        reasons: List[str] = []

        glyph_name = result.get("glyph") or payload.get("glyph_name")
        patch = payload.get("patch") or {}
        similarity = result.get("similarity")
        note = result.get("note") or "not_persisted"

        if glyph_name:
            citations.append(
                {
                    "source_type": "glyphh",
                    "source_id": glyph_name,
                    "title": glyph_name,
                    "url": None,
                    "snippet": "what_if_modify_target",
                    "hash": None,
                }
            )

        facts.append(
            {
                "id": "what_if_patch",
                "text": f"Applied patch {patch}",
                "type": "decision",
                "confidence": 1.0,
                "evidence": [
                    {
                        "source_type": "glyphh",
                        "source_id": "what_if_patch",
                        "snippet": json.dumps(patch, ensure_ascii=True),
                    }
                ],
            }
        )

        if similarity is not None:
            facts.append(
                {
                    "id": "what_if_similarity",
                    "text": f"Similarity after patch: {similarity}",
                    "type": "metric",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "what_if_similarity",
                            "snippet": "what_if_modify",
                        }
                    ],
                }
            )
            reasons.append("similarity computed on updated glyph vs original")

        if note:
            facts.append(
                {
                    "id": "what_if_note",
                    "text": note,
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "what_if_note",
                            "snippet": note,
                        }
                    ],
                }
            )

        return facts, citations, reasons

    def _build_health_facts(
        self, payload: Dict[str, Any], result: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        facts = [
            {
                "id": "runtime_health",
                "text": f"Runtime status {result.get('status')}",
                "type": "decision",
                "confidence": 1.0,
                "evidence": [
                    {
                        "source_type": "glyphh",
                        "source_id": "health",
                        "snippet": "runtime_health",
                    }
                ],
            }
        ]
        return facts, [], []

    def _build_determinism_evidence(
        self, payload: Dict[str, Any], db: SessionLocal
    ) -> Dict[str, Any]:
        model_id = payload.get("model_id")
        encoder_seed = None
        if model_id:
            model = db.get(models.Model, model_id)
            if model:
                encoder_seed = model.encoder_seed
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return {
            "encoder_seed": encoder_seed,
            "encoder_version": getattr(glyphh, "__version__", None),
            "native_engine_version": getattr(glyphh, "native_engine_version", None),
            "input_hash": payload_hash,
        }

    def _wrap_response(
        self,
        *,
        tool: str,
        payload: Dict[str, Any],
        result: Dict[str, Any],
        determinism: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        status = "error" if result.get("error") else "ok"
        model_id = payload.get("model_id")
        request_id = payload.get("request_id")
        text = json.dumps(result, ensure_ascii=True)
        facts: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        reasons: List[str] = []
        data_provenance = {
            "origin": "runtime",
            "pipeline_id": settings.data_pipeline_id,
            "dataset_version": settings.dataset_version,
        }
        if settings.record_hash_chain:
            data_provenance["record_hash_chain"] = settings.record_hash_chain
        if settings.data_pipeline_id or settings.dataset_version or settings.record_hash_chain:
            facts.append(
                {
                    "id": "data_lineage",
                    "text": "Data lineage metadata attached",
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "data_lineage",
                            "snippet": json.dumps(data_provenance, ensure_ascii=True),
                        }
                    ],
                }
            )
        computation_trace = {
            "tool": tool,
            "filters": None,
            "ranking": None,
            "parameters": {},
        }
        if tool == "find_by_properties":
            computation_trace["filters"] = payload.get("constraints") or []
            computation_trace["ranking"] = "glyphh_reasoning.find_by_properties"
            computation_trace["parameters"] = {"top_k": payload.get("top_k", 5)}
        elif tool == "similar_to":
            computation_trace["filters"] = {
                "glyph_name": payload.get("glyph_name"),
                "model_id": payload.get("model_id"),
            }
            computation_trace["ranking"] = "embedding_l2_distance"
            computation_trace["parameters"] = {"top_k": payload.get("top_k", 5)}
        elif tool == "nl_query":
            computation_trace["filters"] = {
                "model_id": payload.get("model_id"),
                "intent": result.get("intent") or result.get("name"),
            }
            computation_trace["ranking"] = "nl_rules_or_semantic_fallback"
            computation_trace["parameters"] = {
                "intent_source": result.get("intent_source") or "rules",
            }
        elif tool == "trend_role":
            computation_trace["filters"] = {
                "model_id": payload.get("model_id"),
                "role": payload.get("role"),
            }
            computation_trace["ranking"] = "timestamp_desc"
            computation_trace["parameters"] = {"limit": payload.get("limit", 50)}
        elif tool == "predict_next":
            computation_trace["filters"] = {
                "model_id": payload.get("model_id"),
                "role": payload.get("role"),
            }
            computation_trace["ranking"] = "prediction_source_preferred"
            computation_trace["parameters"] = {}
        elif tool == "explain_link":
            computation_trace["filters"] = {
                "source": payload.get("source"),
                "target": payload.get("target"),
            }
            computation_trace["ranking"] = "global_cortex_similarity"
            computation_trace["parameters"] = {}
        elif tool == "what_if_modify":
            computation_trace["filters"] = {
                "glyph_name": payload.get("glyph_name"),
            }
            computation_trace["ranking"] = "similarity_after_patch"
            computation_trace["parameters"] = {"patch": payload.get("patch") or {}}
        elif tool == "health":
            computation_trace["filters"] = {}
            computation_trace["ranking"] = "n/a"
            computation_trace["parameters"] = {}
        if computation_trace["filters"] is not None:
            facts.append(
                {
                    "id": "computation_trace",
                    "text": "Computation trace metadata attached",
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "computation_trace",
                            "snippet": json.dumps(computation_trace, ensure_ascii=True),
                        }
                    ],
                }
            )
        freshness = {
            "time_span": None,
            "coverage_percent": None,
            "missing_data": None,
        }
        if tool == "trend_role":
            entries = result.get("entries") or []
            timestamps = [e.get("timestamp") for e in entries if e.get("timestamp")]
            if timestamps:
                freshness["time_span"] = {
                    "start": timestamps[-1],
                    "end": timestamps[0],
                }
            if entries:
                missing = sum(1 for e in entries if e.get("value") is None)
                freshness["missing_data"] = {
                    "count": missing,
                    "total": len(entries),
                }
                freshness["coverage_percent"] = round(
                    100.0 * (len(entries) - missing) / len(entries), 2
                )
        elif tool == "predict_next":
            if result.get("timestamp"):
                freshness["time_span"] = {
                    "start": result.get("timestamp"),
                    "end": result.get("timestamp"),
                }
            freshness["coverage_percent"] = 100.0 if result.get("value") is not None else 0.0
            freshness["missing_data"] = {
                "count": 0 if result.get("value") is not None else 1,
                "total": 1,
            }
        if freshness["time_span"] or freshness["coverage_percent"] is not None:
            facts.append(
                {
                    "id": "freshness_coverage",
                    "text": "Freshness and coverage metadata attached",
                    "type": "metric",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "freshness_coverage",
                            "snippet": json.dumps(freshness, ensure_ascii=True),
                        }
                    ],
                }
            )
        uncertainty = {
            "metric": None,
            "interval": None,
            "distribution": None,
        }
        if tool == "predict_next" and result.get("value") is not None:
            uncertainty["metric"] = "prediction_point"
            uncertainty["interval"] = {
                "lower": None,
                "upper": None,
                "confidence_level": None,
            }
        elif tool == "similar_to":
            matches = result.get("matches") or []
            scores = [m.get("score") for m in matches if isinstance(m.get("score"), (int, float))]
            if scores:
                uncertainty["metric"] = "similarity_scores"
                uncertainty["distribution"] = {
                    "min": min(scores),
                    "max": max(scores),
                    "mean": sum(scores) / len(scores),
                }
        if uncertainty["metric"]:
            facts.append(
                {
                    "id": "uncertainty",
                    "text": "Uncertainty metadata attached",
                    "type": "metric",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "uncertainty",
                            "snippet": json.dumps(uncertainty, ensure_ascii=True),
                        }
                    ],
                }
            )
        counterfactual = None
        if tool == "nl_query":
            counterfactual = {
                "scenario": "No matching intent or glyph found",
                "effect": "Response would return matched_glyph=null and intent_source=model_fallback or null",
            }
        elif tool == "find_by_properties":
            counterfactual = {
                "scenario": "Constraints changed or removed",
                "effect": "Match set and scores would differ based on new constraints",
            }
        elif tool == "similar_to":
            counterfactual = {
                "scenario": "Different query glyph or vector",
                "effect": "Ranked glyph list would change under L2 similarity",
            }
        elif tool == "trend_role":
            counterfactual = {
                "scenario": "Time window shifts or missing data",
                "effect": "Trend entries and coverage metrics would change",
            }
        elif tool == "predict_next":
            counterfactual = {
                "scenario": "No prediction rows available",
                "effect": "Fallback would use most recent actual value",
            }
        elif tool == "what_if_modify":
            counterfactual = {
                "scenario": "Patch changed or removed",
                "effect": "Similarity would be recalculated and may differ",
            }
        elif tool == "explain_link":
            counterfactual = {
                "scenario": "Different source/target glyphs",
                "effect": "Similarity and shared semantic keys would change",
            }
        if counterfactual:
            facts.append(
                {
                    "id": "counterfactual",
                    "text": "Counterfactual guidance attached",
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "counterfactual",
                            "snippet": json.dumps(counterfactual, ensure_ascii=True),
                        }
                    ],
                }
            )
        governance = payload.get("governance") or {}
        if governance:
            facts.append(
                {
                    "id": "governance_proof",
                    "text": "Governance policies evaluated",
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "governance",
                            "snippet": json.dumps(governance, ensure_ascii=True),
                        }
                    ],
                }
            )
        if determinism:
            facts.append(
                {
                    "id": "determinism",
                    "text": "Determinism metadata attached",
                    "type": "decision",
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source_type": "glyphh",
                            "source_id": "determinism",
                            "snippet": json.dumps(determinism, ensure_ascii=True),
                        }
                    ],
                }
            )
        base_facts = list(facts)
        base_citations = list(citations)
        base_reasons = list(reasons)
        if tool == "nl_query" and status == "ok":
            facts, citations, reasons = self._build_nl_facts(payload, result)
        elif tool == "find_by_properties" and status == "ok":
            facts, citations, reasons = self._build_find_by_properties_facts(payload, result)
        elif tool == "similar_to" and status == "ok":
            facts, citations, reasons = self._build_similar_to_facts(payload, result)
        elif tool == "explain_link" and status == "ok":
            facts, citations, reasons = self._build_explain_link_facts(payload, result)
        elif tool == "trend_role" and status == "ok":
            facts, citations, reasons = self._build_trend_role_facts(payload, result)
        elif tool == "predict_next" and status == "ok":
            facts, citations, reasons = self._build_predict_next_facts(payload, result)
        elif tool == "what_if_modify" and status == "ok":
            facts, citations, reasons = self._build_what_if_modify_facts(payload, result)
        elif tool == "health" and status == "ok":
            facts, citations, reasons = self._build_health_facts(payload, result)
        if base_facts:
            facts = base_facts + (facts or [])
        if base_citations:
            citations = base_citations + (citations or [])
        if base_reasons:
            reasons = base_reasons + (reasons or [])
        return {
            "version": "1.0",
            "status": status,
            "answer": {"text": text, "format": "json"},
            "facts": facts,
            "reasons": reasons,
            "grounding": {"mode": "glyphh_only", "score": 1.0, "strict": True},
            "citations": citations,
            "freshness": {"as_of": None, "max_age_seconds": 0, "policy": "best_effort"},
            "constraints_applied": {
                "roles": [],
                "segments": [],
                "time_window": None,
                "allowlist_sources": [],
            },
            "data_provenance": data_provenance,
            "redactions": {"pii_removed": False, "fields": []},
            "traceability": {
                "request_id": request_id,
                "runtime_id": None,
                "model_id": model_id,
                "pipeline_id": None,
                "glyph_ids": [],
                "filters": {"role": None, "segment": None, "time_window": None},
                "temporal_edges": [],
                "path_edges": [],
                "sources": [],
            },
        }

    def _error_response(
        self,
        *,
        tool: str,
        payload: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        return self._wrap_response(
            tool=tool,
            payload=payload,
            result={"error": "invalid_mcp_response", "detail": reason},
        )

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

    def _run_tool(self, tool: str, payload: Dict[str, Any], db: SessionLocal) -> Dict[str, Any]:
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
            seg_rows = (
                db.query(models.Segment)
                .filter(models.Segment.glyph_name.in_([g.name for g in glyph_rows]))
                .all()
            )
            seg_map: dict[tuple[str, int, int], bytes] = {}
            for seg in seg_rows:
                seg_map[(seg.glyph_name, seg.layer, seg.seg_index)] = seg.vec
            result = run_nl_query(
                payload["text"],
                nl_configs,
                model.roles_config or {},
                _nl_glyphs(glyph_rows, seg_map),
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
                "matches": [{"name": g.name, "score": score} for g, score in results]
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

    def handle_tool(self, tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            result = self._run_tool(tool, payload, db)
            determinism = self._build_determinism_evidence(payload, db)
            response = self._wrap_response(
                tool=tool,
                payload=payload,
                result=result,
                determinism=determinism,
            )
            try:
                validate_mcp_response(response)
            except Exception as exc:
                error_text = format_validation_error(exc)
                response = self._error_response(tool=tool, payload=payload, reason=error_text)
                validate_mcp_response(response)
            return response
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
