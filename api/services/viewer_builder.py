from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from ..core import models
from glyphh.config import (
    DEFAULT_NUM_SEGMENTS,
    NEURAL_EDGES_MIN_SIM,
    NEURAL_EDGES_TOP_N,
    SEMANTIC_EDGES_MIN_WEIGHT,
    SEMANTIC_EDGES_TOP_N,
)
from glyphh.vector import bipolar_to_bits, similarity as hv_similarity


def build_viewer_payload(
    db: Session,
    model_id: str,
    limit: int,
) -> dict[str, Any]:
    model = db.get(models.Model, model_id)
    if not model:
        raise ValueError("Model not found")

    roles_config = model.roles_config or {}
    layers_cfg = roles_config.get("layers", [])
    num_layers = len(layers_cfg)
    segments_per_layer = {
        layer.get("index", idx): len(layer.get("segments", [])) or DEFAULT_NUM_SEGMENTS
        for idx, layer in enumerate(layers_cfg)
    }

    glyph_rows = (
        db.query(models.Glyph)
        .filter(models.Glyph.model_id == model_id)
        .order_by(models.Glyph.created_at.desc())
        .limit(limit)
        .all()
    )
    names = [g.name for g in glyph_rows]

    seg_rows = (
        db.query(models.Segment)
        .filter(models.Segment.glyph_name.in_(names))
        .all()
    )

    seg_map: dict[str, dict[int, dict[int, bytes]]] = {}
    for seg in seg_rows:
        seg_map.setdefault(seg.glyph_name, {}).setdefault(seg.layer, {})[seg.seg_index] = seg.vec

    glyphs_json: list[dict[str, Any]] = []

    def build_layers(cortex_vec: np.ndarray):
        layers = []
        for li in range(num_layers):
            num_segs = segments_per_layer.get(li, DEFAULT_NUM_SEGMENTS)
            seg_list = []
            for _ in range(num_segs):
                seg_list.append([0] * cortex_vec.size)
            layers.append(
                {
                    "index": li,
                    "cortex": bipolar_to_bits(cortex_vec),
                    "segments": seg_list,
                }
            )
        return layers

    for g in glyph_rows:
        cortex_vec = np.frombuffer(g.cortex, dtype=np.int8)
        layers_json = []

        for li in range(num_layers):
            num_segs = segments_per_layer.get(li, DEFAULT_NUM_SEGMENTS)
            seg_list = []
            for si in range(num_segs):
                vec_bytes = seg_map.get(g.name, {}).get(li, {}).get(si)
                if vec_bytes is None:
                    seg_bits = [0] * (
                        cortex_vec.size if cortex_vec.size else DEFAULT_NUM_SEGMENTS
                    )
                else:
                    seg_vec = np.frombuffer(vec_bytes, dtype=np.int8)
                    seg_bits = bipolar_to_bits(seg_vec)
                seg_list.append(seg_bits)

            layers_json.append(
                {
                    "index": li,
                    "cortex": bipolar_to_bits(cortex_vec),
                    "segments": seg_list,
                }
            )

        glyphs_json.append(
            {
                "name": g.name,
                "node_type": g.node_type,
                "semantic": g.semantic,
                "layers": layers_json,
            }
        )

    taxonomy_terms = set()
    for g in glyph_rows:
        sem = g.semantic or {}
        path = sem.get("taxonomy")
        if path and isinstance(path, list):
            taxonomy_terms.update(path)
    taxonomy_terms -= set(names)

    dim = 0
    if glyph_rows:
        dim = len(glyph_rows[0].cortex)
    if dim <= 0:
        dim = 1024
    zero_cortex = np.zeros(dim, dtype=np.int8)

    for term in sorted(taxonomy_terms):
        glyphs_json.append(
            {
                "name": term,
                "node_type": "taxonomy",
                "semantic": {},
                "layers": build_layers(zero_cortex),
            }
        )
        names.append(term)

    semantic_edges: list[dict[str, Any]] = []
    hierarchy_edges: list[dict[str, Any]] = []
    neural_edges: list[dict[str, Any]] = []

    sem_pair_weight: dict[tuple[str, str], float] = {}
    for i, gi in enumerate(glyph_rows):
        sem_i = gi.semantic or {}
        for gj in glyph_rows[i + 1 :]:
            sem_j = gj.semantic or {}
            weight = 0.0
            for k, vi in sem_i.items():
                if isinstance(vi, (list, dict)):
                    continue
                if k in sem_j and sem_j[k] == vi and not isinstance(sem_j[k], (list, dict)):
                    weight += 1.0
            if weight > 0.0:
                a, b = (gi.name, gj.name) if gi.name < gj.name else (gj.name, gi.name)
                sem_pair_weight[(a, b)] = max(sem_pair_weight.get((a, b), 0.0), weight)

    cfg_sem = roles_config.get("semantic_edges", {}) if isinstance(roles_config, dict) else {}
    sem_top_n = cfg_sem.get("top_n", SEMANTIC_EDGES_TOP_N)
    sem_min_w = cfg_sem.get("min_weight", SEMANTIC_EDGES_MIN_WEIGHT)

    sem_neighbors: dict[str, list[tuple[str, float]]] = {g.name: [] for g in glyph_rows}
    for (a, b), w in sem_pair_weight.items():
        if w < sem_min_w:
            continue
        sem_neighbors[a].append((b, w))
        sem_neighbors[b].append((a, w))

    used_sem_pairs: set[tuple[str, str]] = set()
    for src, neigh_list in sem_neighbors.items():
        if not neigh_list:
            continue
        neigh_list.sort(key=lambda x: x[1], reverse=True)
        if sem_top_n and sem_top_n > 0:
            neigh_list = neigh_list[:sem_top_n]
        for tgt, w in neigh_list:
            a, b = (src, tgt) if src < tgt else (tgt, src)
            key = (a, b)
            if key in used_sem_pairs:
                continue
            used_sem_pairs.add(key)
            semantic_edges.append(
                {
                    "source": a,
                    "target": b,
                    "weight": w,
                    "layer": 0,
                }
            )

    for g in glyph_rows:
        sem = g.semantic or {}
        if "linked_customer" in sem:
            hierarchy_edges.append(
                {
                    "source": g.name,
                    "target": sem["linked_customer"],
                    "kind": "linked_customer",
                    "weight": 1.0,
                    "layer": 0,
                }
            )
        if "customer" in sem:
            hierarchy_edges.append(
                {
                    "source": g.name,
                    "target": sem["customer"],
                    "kind": "customer",
                    "weight": 1.0,
                    "layer": 0,
                }
            )
        if "prospect" in sem:
            hierarchy_edges.append(
                {
                    "source": g.name,
                    "target": sem["prospect"],
                    "kind": "prospect",
                    "weight": 1.0,
                    "layer": 0,
                }
            )

        tax_path = sem.get("taxonomy") if sem else None
        if not tax_path and sem and isinstance(sem.get("taxonomy"), list):
            tax_path = sem.get("taxonomy")
        if tax_path and isinstance(tax_path, list) and len(tax_path) >= 1:
            for i in range(len(tax_path) - 1):
                hierarchy_edges.append(
                    {
                        "source": tax_path[i],
                        "target": tax_path[i + 1],
                        "kind": "taxonomy",
                        "weight": 1.0,
                        "layer": 0,
                    }
                )
            for term in tax_path:
                hierarchy_edges.append(
                    {
                        "source": g.name,
                        "target": term,
                        "kind": "taxonomy_leaf",
                        "weight": 1.0,
                        "layer": 0,
                    }
                )

    cfg_neural = roles_config.get("neural_edges", {}) if isinstance(roles_config, dict) else {}
    neural_top_n = cfg_neural.get("top_n", NEURAL_EDGES_TOP_N)
    neural_min_sim = cfg_neural.get("min_similarity", NEURAL_EDGES_MIN_SIM)

    cortex_vecs = [np.frombuffer(g.cortex, dtype=np.int8) for g in glyph_rows]
    n = len(glyph_rows)
    candidates: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = hv_similarity(cortex_vecs[i], cortex_vecs[j])
            if sim >= neural_min_sim:
                candidates.append((i, j, sim))

    neighbors: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for i, j, sim in candidates:
        neighbors[i].append((j, sim))
        neighbors[j].append((i, sim))

    used_pairs: set[tuple[int, int]] = set()
    for i in range(n):
        neigh_list = neighbors[i]
        if not neigh_list:
            continue
        neigh_list.sort(key=lambda x: x[1], reverse=True)
        if neural_top_n and neural_top_n > 0:
            neigh_list = neigh_list[:neural_top_n]
        for j, sim in neigh_list:
            a, b = (i, j) if i < j else (j, i)
            key = (a, b)
            if key in used_pairs:
                continue
            used_pairs.add(key)
            neural_edges.append(
                {
                    "source": glyph_rows[a].name,
                    "target": glyph_rows[b].name,
                    "weight": sim,
                    "similarity": sim,
                    "layer": 0,
                }
            )

    return {
        "glyphs": glyphs_json,
        "edges": {
            "semantic": semantic_edges,
            "neural": neural_edges,
            "hierarchy": hierarchy_edges,
        },
        "roles_config": roles_config,
    }
