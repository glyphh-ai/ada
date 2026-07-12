"""
enricher.py — raw text -> universal-schema slots (the ingest FRONT half).
=========================================================================
Ported from the retired adaL repo (de-branded: no "Ada"). This is the
schema-on-write distiller that turns free text into `{role: value}` attributes,
which the existing ingest back-half (`ingest_concepts`) encodes into both
structures (semantic + cortex + segments).

`Enricher` is the interface. `HeuristicEnricher` is a cheap, offline, narrow
regex default; a model-backed enricher (local or API) implements the same
`universal(text)` contract and slots in without touching the ingest path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class StructuredFact:
    text: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    topic: str = "general"

    def is_structured(self) -> bool:
        return bool(self.subject or self.predicate or self.object)


class Enricher(Protocol):
    def universal(self, text: str) -> dict[str, dict[str, str]]:
        """text -> {layer: {role: value}} universal-schema mapping (empty if none)."""
        ...


def _strip(s: str) -> str:
    return s.strip().strip("?.,!").lower()


def _try_facts(text: str) -> Optional[StructuredFact]:
    patterns = [
        (r"^(\w+)(?:'s|s)\s+favorite\s+(\w+)\s+is\s+(.+?)\.?$",
         lambda m: StructuredFact(text, _strip(m.group(1)), f"favorite_{m.group(2).lower()}",
                                  _strip(m.group(3)), "person.preference")),
        (r"^(\w+)\s+is\s+(\d+)\s+years?\s+old\.?$",
         lambda m: StructuredFact(text, _strip(m.group(1)), "age", m.group(2), "person.age")),
        (r"^(\w+)\s+lives?\s+in\s+(.+?)\.?$",
         lambda m: StructuredFact(text, _strip(m.group(1)), "residence", _strip(m.group(2)), "person.location")),
        (r"^(\w+)\s+works?\s+as\s+(?:a|an)?\s*(.+?)\.?$",
         lambda m: StructuredFact(text, _strip(m.group(1)), "job", _strip(m.group(2)), "person.job")),
        (r"^The\s+capital\s+of\s+(.+?)\s+is\s+(.+?)\.?$",
         lambda m: StructuredFact(text, _strip(m.group(1)), "capital", _strip(m.group(2)), "geography.capital")),
        (r"^(\w+)\s+stands\s+for\s+(.+?)\.?$",
         lambda m: StructuredFact(text, _strip(m.group(1)), "abbreviation_of", _strip(m.group(2)), "knowledge.definition")),
    ]
    for pat, build in patterns:
        m = re.match(pat, text, re.I)
        if m:
            return build(m)
    return None


def _try_queries(text: str) -> Optional[StructuredFact]:
    patterns = [
        (r"^(?:what\s+is\s+(\w+)(?:'s|s)\s+age|how\s+old\s+is\s+(\w+))\??$", "age", "person.age"),
        (r"^where\s+does\s+(\w+)\s+live\??$", "residence", "person.location"),
        (r"^(?:what\s+job\s+does\s+(\w+)\s+have|what\s+does\s+(\w+)\s+do)\??$", "job", "person.job"),
        (r"^what\s+color\s+does\s+(\w+)\s+like\??$", "favorite_color", "person.preference"),
        (r"^what\s+does\s+(\w+)\s+stand\s+for\??$", "abbreviation_of", "knowledge.definition"),
    ]
    for pat, predicate, topic in patterns:
        m = re.match(pat, text, re.I)
        if m:
            subj = next((g for g in m.groups() if g), "")
            return StructuredFact(text=text, subject=_strip(subj), predicate=predicate, topic=topic)
    return None


class HeuristicEnricher:
    """Regex-based fact + query parser. Cheap, offline, narrow coverage."""

    # predicate -> (layer, role) to project a parsed triple into universal slots
    _PRED_SLOT = {
        "age": ("temporal", "age"),
        "residence": ("spatial", "location"),
        "favorite_color": ("perceptual", "color"),
        "capital": ("spatial", "location"),
    }

    def enrich(self, text: str) -> StructuredFact:
        clean = text.strip()
        return _try_facts(clean) or _try_queries(clean) or StructuredFact(text=clean)

    def universal(self, text: str) -> dict[str, dict[str, str]]:
        sf = self.enrich(text)
        if not sf.is_structured() or not sf.object:
            return {}
        mapped: dict[str, dict[str, str]] = {}
        if sf.subject:
            mapped["entity"] = {"name": sf.subject}
        layer_role = self._PRED_SLOT.get(sf.predicate)
        if layer_role:
            layer, role = layer_role
            mapped.setdefault(layer, {})[role] = sf.object
            if sf.subject:
                mapped["relational"] = {"subject": sf.subject, "predicate": sf.predicate}
        else:
            mapped["relational"] = {"subject": sf.subject, "predicate": sf.predicate, "object": sf.object}
        return mapped
