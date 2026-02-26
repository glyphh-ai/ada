"""Shared fixtures for glyphh.intent tests."""

import json
from pathlib import Path

import pytest
from glyphh.intent import IntentExtractor


@pytest.fixture(scope="session")
def extractor():
    """Shared IntentExtractor instance (base vocabulary, no packs)."""
    return IntentExtractor()


@pytest.fixture(scope="session")
def benchmark_queries():
    """Load all benchmark queries from benchmark/queries.json."""
    path = Path(__file__).parent / "benchmark" / "queries.json"
    with open(path) as f:
        data = json.load(f)
    return data["queries"]
