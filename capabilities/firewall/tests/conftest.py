"""Shared fixtures for firewall model tests."""

import json
from pathlib import Path

import pytest

from encoder import ENCODER_CONFIG, encode_prompt, entry_to_record, score_prompt
from glyphh.core.types import Concept
from glyphh.encoder import Encoder


@pytest.fixture(scope="session")
def encoder():
    """Session-scoped encoder instance."""
    return Encoder(ENCODER_CONFIG)


@pytest.fixture(scope="session")
def exemplar_entries():
    """Load all exemplar entries from JSONL."""
    data_path = Path(__file__).parent.parent / "data" / "exemplars.jsonl"
    entries = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


@pytest.fixture(scope="session")
def attack_entries(exemplar_entries):
    """Only attack exemplars (non-benign)."""
    return [e for e in exemplar_entries if e.get("label") != "benign"]


@pytest.fixture(scope="session")
def benign_entries(exemplar_entries):
    """Only benign exemplars."""
    return [e for e in exemplar_entries if e.get("label") == "benign"]


@pytest.fixture(scope="session")
def exemplar_glyphs(encoder, exemplar_entries):
    """Encode all exemplars into glyphs with metadata."""
    glyphs = []
    for entry in exemplar_entries:
        record = entry_to_record(entry)
        concept = Concept(
            name=record["concept_text"],
            attributes=record["attributes"],
        )
        glyphs.append((encoder.encode(concept), record["metadata"]))
    return glyphs


@pytest.fixture(scope="session")
def attack_glyphs(exemplar_glyphs):
    """Only attack exemplar glyphs."""
    return [(g, m) for g, m in exemplar_glyphs if m.get("label") != "benign"]
