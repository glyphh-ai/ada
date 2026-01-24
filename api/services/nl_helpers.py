from __future__ import annotations

from sqlalchemy.orm import Session

from ..core import models
from glyphh.metrics import SimpleGlyph as MetricsGlyph
from glyphh.nl.configs import resolve_nl_configs
from glyphh.nl.inference import SimpleGlyph as NLInferenceGlyph


def build_nl_configs(model: models.Model, db: Session):
    nl_cfg_obj = getattr(model, "nl_config", None)
    base_cfg = getattr(nl_cfg_obj, "config", nl_cfg_obj) if nl_cfg_obj is not None else {}

    class Source:
        def fetch_nl_config(self_inner, identifier: str):
            enc = db.get(models.Encoder, identifier)
            if not enc:
                return None
            return enc.config if isinstance(enc.config, dict) else None

        def can_access(self_inner, identifier: str) -> bool:
            return db.get(models.Encoder, identifier) is not None

    return resolve_nl_configs(base_cfg if isinstance(base_cfg, dict) else {}, Source())


def to_simple_glyphs(rows: list[models.Glyph]) -> list[MetricsGlyph]:
    return [MetricsGlyph(g.name, g.node_type, g.semantic or {}, g.cortex) for g in rows]


def to_nl_glyphs(
    rows: list[models.Glyph],
    db: Session,
    model_id: str | None = None,
) -> list[NLInferenceGlyph]:
    names = [g.name for g in rows]
    seg_query = db.query(models.Segment).filter(models.Segment.glyph_name.in_(names))
    if model_id:
        seg_query = seg_query.filter(models.Segment.model_id == model_id)
    seg_rows = seg_query.all()
    seg_map: dict[str, dict[tuple[int, int], bytes]] = {}
    for seg in seg_rows:
        entry = seg_map.setdefault(seg.glyph_name, {})
        entry[(seg.layer, seg.seg_index)] = seg.vec
    return [
        NLInferenceGlyph(
            g.name,
            g.node_type,
            g.semantic or {},
            g.cortex,
            segments=seg_map.get(g.name),
        )
        for g in rows
    ]
