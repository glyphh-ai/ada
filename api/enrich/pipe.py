"""
pipe.py — join the enrich FRONT half to the existing ingest BACK half.
======================================================================
raw text ─[enricher.universal]→ {layer:{role:value}} ─[flatten]→ {role:value}
         → ConceptInput(name, attributes) → ingest_concepts → both structures.

`enrich_to_concept` is pure (unit-tested, no runtime deps). `enrich_and_ingest`
wires it to the existing `ingest_concepts` (imported lazily — needs glyphh + DB).
"""
from __future__ import annotations

from typing import Optional

try:
    from .enricher import Enricher, HeuristicEnricher
except ImportError:                       # allow standalone import (tests)
    from enricher import Enricher, HeuristicEnricher


def enrich_to_concept(text: str, enricher: Optional[Enricher] = None) -> Optional[dict]:
    """text -> {name, attributes, node_type} (ConceptInput-shaped dict), or None.

    attributes are FLAT {role: value} — the shape encoder.encode consumes
    (roles_config keys the layer/segment structure; attributes key by role name)."""
    enricher = enricher or HeuristicEnricher()
    mapped = enricher.universal(text)          # {layer: {role: value}}
    if not mapped:
        return None
    attributes: dict[str, str] = {}
    for _layer, roles in mapped.items():
        for role, value in roles.items():
            attributes[role] = value           # flatten: layer nesting dropped
    name = attributes.get("name") or (mapped.get("relational", {}) or {}).get("subject")
    if not name:
        return None
    return {"name": str(name), "attributes": attributes, "node_type": "concept"}


def enrich_and_ingest(model, text: str, db, enricher: Optional[Enricher] = None,
                      clear_existing: bool = False):
    """Full raw path: enrich -> ConceptInput -> ingest_concepts (needs runtime)."""
    from ..core.schemas import ConceptInput          # runtime deps, lazy
    from ..services.ingest import ingest_concepts
    payload = enrich_to_concept(text, enricher)
    if payload is None:
        return None
    concept = ConceptInput(**payload)
    return ingest_concepts(model, [concept], clear_existing, db)
