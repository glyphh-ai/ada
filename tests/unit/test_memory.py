"""
Tests for glyphh.memory — thought storage and semantic recall.

Tests verify encoding, storage, recall by similarity, persistence,
Hebbian reinforcement, and temporal decay.
"""

import json
import time

import numpy as np
import pytest

from glyphh.memory import Thought, ThoughtEncoder, ThoughtStore
from glyphh.core.ops import cosine_similarity


# ── ThoughtEncoder ─────────────────────────────────────────────────────────


class TestThoughtEncoder:
    def test_encode_produces_bipolar(self):
        enc = ThoughtEncoder(dimension=10_000)
        vec = enc.encode("hello world")
        assert vec.shape == (10_000,)
        assert set(np.unique(vec)).issubset({-1, 1})

    def test_encode_deterministic(self):
        enc = ThoughtEncoder(dimension=10_000)
        a = enc.encode("cosine similarity")
        b = enc.encode("cosine similarity")
        assert np.array_equal(a, b)

    def test_similar_texts_high_similarity(self):
        enc = ThoughtEncoder(dimension=10_000)
        a = enc.encode("HDC uses bipolar vectors")
        b = enc.encode("bipolar vectors in HDC")
        sim = enc.similarity(a, b)
        assert sim > 0.4, f"Expected > 0.4, got {sim:.3f}"

    def test_unrelated_texts_low_similarity(self):
        enc = ThoughtEncoder(dimension=10_000)
        a = enc.encode("HDC uses bipolar vectors")
        b = enc.encode("the weather is sunny today")
        sim = enc.similarity(a, b)
        assert sim < 0.2, f"Expected < 0.2, got {sim:.3f}"

    def test_create_thought(self):
        enc = ThoughtEncoder(dimension=10_000)
        thought = enc.create_thought("bind multiplies vectors", metadata={"source": "test"})
        assert thought.content == "bind multiplies vectors"
        assert thought.vector.shape == (10_000,)
        assert thought.strength == 1.0
        assert thought.metadata == {"source": "test"}
        assert len(thought.id) == 12


# ── Thought dataclass ──────────────────────────────────────────────────────


class TestThought:
    def test_reinforce(self):
        enc = ThoughtEncoder(dimension=100)
        t = enc.create_thought("test")
        t.strength = 0.5
        t.reinforce(0.2)
        assert t.strength == pytest.approx(0.7)
        assert t.recalled_at is not None

    def test_reinforce_cap(self):
        enc = ThoughtEncoder(dimension=100)
        t = enc.create_thought("test")
        t.strength = 0.95
        t.reinforce(0.2)
        assert t.strength == 1.0

    def test_decay(self):
        enc = ThoughtEncoder(dimension=100)
        t = enc.create_thought("test")
        t.decay(0.3)
        assert t.strength == pytest.approx(0.7)

    def test_decay_floor(self):
        enc = ThoughtEncoder(dimension=100)
        t = enc.create_thought("test")
        t.strength = 0.01
        t.decay(0.1)
        assert t.strength == 0.0


# ── ThoughtStore ───────────────────────────────────────────────────────────


class TestThoughtStore:
    @pytest.fixture
    def store(self, tmp_path):
        enc = ThoughtEncoder(dimension=10_000)
        return ThoughtStore(enc, storage_dir=tmp_path)

    def test_remember(self, store):
        t = store.remember("bind is element-wise multiplication")
        assert store.count == 1
        assert t.content == "bind is element-wise multiplication"

    def test_recall_by_similarity(self, store):
        store.remember("bind is element-wise multiplication of bipolar vectors")
        store.remember("bundle is majority vote aggregation")
        store.remember("cosine similarity measures vector alignment")
        store.remember("the weather is nice today")

        results = store.recall("what is bind?", top_k=2)
        assert len(results) >= 1
        # The bind thought should be the top result
        top_thought, top_score = results[0]
        assert "bind" in top_thought.content.lower()
        assert top_score > 0.05

    def test_recall_empty_store(self, store):
        results = store.recall("anything")
        assert results == []

    def test_recall_reinforces(self, store):
        t = store.remember("important concept")
        original_strength = t.strength
        store.recall("important concept")
        assert t.strength >= original_strength

    def test_forget(self, store):
        t = store.remember("to be forgotten")
        assert store.count == 1
        assert store.forget(t.id)
        assert store.count == 0

    def test_forget_missing(self, store):
        assert not store.forget("nonexistent")

    def test_decay_all(self, store):
        store.remember("a")
        store.remember("b")
        for t in store.thoughts:
            t.strength = 0.005  # Below decay rate
        forgotten = store.decay_all(rate=0.01)
        assert forgotten == 2
        assert store.count == 0

    def test_persistence(self, tmp_path):
        enc = ThoughtEncoder(dimension=10_000)

        # Store and save
        store1 = ThoughtStore(enc, storage_dir=tmp_path)
        store1.remember("HDC is awesome")
        store1.remember("Ada thinks in vectors")
        store1.save()

        # Load in a new instance
        store2 = ThoughtStore(enc, storage_dir=tmp_path)
        store2.load()

        assert store2.count == 2
        contents = [t.content for t in store2.thoughts]
        assert "HDC is awesome" in contents
        assert "Ada thinks in vectors" in contents

    def test_persistence_vectors_match(self, tmp_path):
        enc = ThoughtEncoder(dimension=10_000)

        store1 = ThoughtStore(enc, storage_dir=tmp_path)
        t = store1.remember("test persistence vectors")
        original_vec = t.vector.copy()
        store1.save()

        store2 = ThoughtStore(enc, storage_dir=tmp_path)
        store2.load()
        loaded_vec = store2.thoughts[0].vector
        assert np.array_equal(original_vec, loaded_vec)

    def test_save_empty_store(self, tmp_path):
        enc = ThoughtEncoder(dimension=10_000)
        store = ThoughtStore(enc, storage_dir=tmp_path)
        store.remember("temp")
        store.forget(store.thoughts[0].id)
        store.save()
        assert (tmp_path / "thoughts.jsonl").exists()

    def test_recall_ordering(self, store):
        """Most relevant thought should come first."""
        store.remember("Python is a programming language")
        store.remember("bind multiplies two bipolar vectors element-wise")
        store.remember("Java is also a programming language")

        results = store.recall("bipolar vector multiplication", top_k=3)
        assert len(results) >= 1
        assert "bind" in results[0][0].content.lower()

    def test_min_score_filter(self, store):
        store.remember("completely unrelated topic about cooking pasta")
        results = store.recall("HDC bipolar vectors", top_k=5, min_score=0.5)
        # With a high min_score, unrelated thoughts should be filtered
        assert len(results) == 0
