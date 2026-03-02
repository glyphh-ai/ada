"""Tests for IntentCache — HDC cache for LLM classifications."""

import pytest
import numpy as np

from glyphh.cognitive.intent_cache import IntentCache


DIM = 1000  # Smaller dimension for fast tests


class TestIntentCacheBasics:
    """Core cache operations."""

    def test_empty_lookup_returns_none(self):
        cache = IntentCache(dimension=DIM)
        assert cache.lookup("show the report") is None

    def test_store_and_lookup(self):
        cache = IntentCache(dimension=DIM, threshold=0.5)
        classification = {
            "functions": ["display"],
            "arguments": {"display": {"item_name": "report.txt"}},
            "confidence": 0.9,
        }
        cache.store("show the report", "root.workspace", classification)
        assert cache.size == 1

        result = cache.lookup("show the report", "root.workspace")
        assert result is not None
        assert result["functions"] == ["display"]
        assert result["confidence"] == 0.9

    def test_similar_query_hits_cache(self):
        cache = IntentCache(dimension=DIM, threshold=0.5)
        classification = {
            "functions": ["display"],
            "arguments": {},
            "confidence": 0.85,
        }
        cache.store("show the report", "root.workspace", classification)

        # Similar query (shared words "show" and "report")
        result = cache.lookup("show report", "root.workspace")
        assert result is not None
        assert result["functions"] == ["display"]

    def test_dissimilar_query_misses_cache(self):
        cache = IntentCache(dimension=DIM, threshold=0.85)
        classification = {
            "functions": ["display"],
            "arguments": {},
            "confidence": 0.9,
        }
        cache.store("show the report", "root.workspace", classification)

        # Completely different query
        result = cache.lookup("delete all temporary files", "other.context")
        assert result is None

    def test_cache_returns_similarity(self):
        cache = IntentCache(dimension=DIM, threshold=0.3)
        cache.store("navigate to archive", "", {"functions": ["cd"], "arguments": {}, "confidence": 0.9})

        result = cache.lookup("navigate to archive", "")
        assert result is not None
        assert "cache_similarity" in result


class TestIntentCacheHebbian:
    """Hebbian reinforcement tests."""

    def test_reinforce_correct_strengthens(self):
        cache = IntentCache(dimension=DIM, threshold=0.3)
        classification = {"functions": ["display"], "arguments": {}, "confidence": 0.9}
        cache.store("show report", "", classification)

        # Trigger a lookup to set _last_hit_index
        cache.lookup("show report", "")

        initial_strength = cache._entries[0].strength
        cache.reinforce(correct=True)
        assert cache._entries[0].strength > initial_strength

    def test_reinforce_incorrect_weakens(self):
        cache = IntentCache(dimension=DIM, threshold=0.3)
        classification = {"functions": ["display"], "arguments": {}, "confidence": 0.9}
        cache.store("show report", "", classification)

        cache.lookup("show report", "")
        initial_strength = cache._entries[0].strength
        cache.reinforce(correct=False)
        assert cache._entries[0].strength < initial_strength

    def test_weak_entry_evicted(self):
        cache = IntentCache(dimension=DIM, threshold=0.3)
        classification = {"functions": ["display"], "arguments": {}, "confidence": 0.9}
        cache.store("show report", "", classification)

        # Weaken repeatedly until evicted
        for _ in range(10):
            cache.lookup("show report", "")
            cache.reinforce(correct=False)

        assert cache.size == 0

    def test_reinforce_without_lookup_is_noop(self):
        cache = IntentCache(dimension=DIM)
        cache.reinforce(correct=True)  # Should not raise


class TestIntentCacheCapacity:
    """Capacity management tests."""

    def test_evicts_weakest_at_capacity(self):
        cache = IntentCache(dimension=DIM, capacity=3, threshold=0.3)

        # Use very distinct queries to avoid duplicate detection
        distinct_queries = [
            "navigate to archive",
            "delete the spreadsheet",
            "search for budget",
        ]
        for i, q in enumerate(distinct_queries):
            cache.store(q, "", {
                "functions": [f"func_{i}"], "arguments": {}, "confidence": 0.9,
            })
        assert cache.size == 3

        # Strengthen entry 1
        cache.lookup(distinct_queries[1], "")
        cache.reinforce(correct=True)

        # Add a 4th — should evict the weakest (0 or 2, both at 1.0)
        cache.store("create a completely new item", "", {
            "functions": ["func_new"], "arguments": {}, "confidence": 0.9,
        })
        assert cache.size == 3

    def test_clear_empties_cache(self):
        cache = IntentCache(dimension=DIM)
        cache.store("query", "", {"functions": ["f"], "arguments": {}, "confidence": 0.9})
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0


class TestIntentCacheEncoding:
    """HDC encoding quality tests."""

    def test_encode_deterministic(self):
        cache = IntentCache(dimension=DIM)
        v1 = cache.encode("show the report", "root")
        v2 = cache.encode("show the report", "root")
        assert np.array_equal(v1, v2)

    def test_encode_different_queries(self):
        cache = IntentCache(dimension=DIM)
        v1 = cache.encode("show the report", "root")
        v2 = cache.encode("delete all files", "archive")
        # Should be different vectors
        assert not np.array_equal(v1, v2)

    def test_encode_empty_query(self):
        cache = IntentCache(dimension=DIM)
        v = cache.encode("", "")
        assert v.shape == (DIM,)

    def test_duplicate_store_updates_existing(self):
        cache = IntentCache(dimension=DIM, threshold=0.3)
        cache.store("show report", "", {"functions": ["old"], "arguments": {}, "confidence": 0.5})
        cache.store("show report", "", {"functions": ["new"], "arguments": {}, "confidence": 0.9})
        # Should update existing, not add duplicate
        assert cache.size == 1
        result = cache.lookup("show report", "")
        assert result["functions"] == ["new"]
