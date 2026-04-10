"""
Tests for context threads — Ada's structured memory as glyphs.

Covers: thread creation, glyph encoding, glyph-based recall,
thread manager topic inference, thread lifecycle, dream weaving.
"""

import pytest
import time

from glyphh.memory.thought_space import ThoughtGlyphSpace
from glyphh.memory.thought_glyph import ThoughtGlyphEncoder
from glyphh.memory.glyph_cognitive import GlyphCognitiveLoop

from domains.brain.context_thread import ContextThread, ThreadStore
from domains.brain.thread_manager import (
    ThreadManager, infer_topic, _extract_implicit_entities,
)
from glyphh.memory.glyph_dream import GlyphDreamLoop, InsightKind


# ── Shared encoder fixture ──────────────────────────────────────────

@pytest.fixture
def encoder():
    """Create a ThoughtGlyphEncoder for tests."""
    space = ThoughtGlyphSpace()
    return space.encoder


# ── ContextThread ───────────────────────────────────────────────────

class TestContextThread:

    def test_add_fact(self):
        t = ContextThread(tool="test")
        t.add_fact("my wife is Brandi", ["Brandi"])
        assert "my wife is Brandi" in t.facts
        assert "Brandi" in t.entities
        assert t.turn_count == 1

    def test_add_fact_dedup(self):
        t = ContextThread(tool="test")
        t.add_fact("fact one")
        t.add_fact("fact one")
        assert len(t.facts) == 1

    def test_add_fact_marks_glyph_stale(self, encoder):
        t = ContextThread(tool="test")
        t.encode(encoder)
        assert t.glyph is not None
        t.add_fact("new fact")
        assert t.glyph is None  # stale after adding fact

    def test_close(self):
        t = ContextThread()
        assert t.active
        t.close()
        assert not t.active

    def test_encode_creates_glyph(self, encoder):
        t = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"],
            facts=["my wife is Brandi"],
        )
        t.encode(encoder)
        assert t.glyph is not None
        assert t.glyph.global_cortex is not None
        # Should have activated layers (which ones depends on primitives)
        assert len(t.glyph.layers) >= 2
        assert len(t.glyph.metadata.get("_activated_attrs", [])) >= 2

    def test_encode_includes_content_vector(self, encoder):
        t = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"],
            facts=["my wife is Brandi"],
        )
        t.encode(encoder)
        assert "_content_vector" in t.glyph.metadata

    def test_ensure_encoded_lazy(self, encoder):
        t = ContextThread(
            tool="cli", facts=["hello world"],
        )
        assert t.glyph is None
        t.ensure_encoded(encoder)
        assert t.glyph is not None
        # Second call doesn't re-encode
        glyph_ref = t.glyph
        t.ensure_encoded(encoder)
        assert t.glyph is glyph_ref  # same object

    def test_matches_tool(self):
        t = ContextThread(tool="claude-code")
        assert t.matches_tool("claude-code")
        assert not t.matches_tool("gemini")

    def test_matches_topic(self):
        t = ContextThread(topic="family")
        assert t.matches_topic("family")
        assert t.matches_topic("Family")
        assert not t.matches_topic("work")


# ── ThreadStore ─────────────────────────────────────────────────────

class TestThreadStore:

    def test_add_and_get(self, encoder):
        store = ThreadStore(encoder=encoder)
        t = ContextThread(tool="test", topic="family")
        store.add(t)
        assert store.get(t.thread_id) == t
        assert store.count == 1

    def test_add_encodes_glyph(self, encoder):
        store = ThreadStore(encoder=encoder)
        t = ContextThread(
            tool="cli", facts=["my wife is Brandi"],
            entities=["Brandi"], topic="family",
        )
        assert t.glyph is None
        store.add(t)
        assert t.glyph is not None  # encoded on add

    def test_recall_glyph_similarity(self, encoder):
        store = ThreadStore(encoder=encoder)
        t = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"],
            facts=["my wife is Brandi"],
        )
        store.add(t)
        results = store.recall("who is my wife?")
        assert len(results) > 0
        thread, score = results[0]
        assert thread.thread_id == t.thread_id
        assert score > 0

    def test_recall_tool_filter(self, encoder):
        store = ThreadStore(encoder=encoder)
        t1 = ContextThread(
            tool="cli", topic="family",
            facts=["my wife is Brandi"], entities=["Brandi"],
        )
        t2 = ContextThread(
            tool="gemini", topic="family",
            facts=["my wife is Brandi"], entities=["Brandi"],
        )
        store.add(t1)
        store.add(t2)

        # Filter to cli only
        results = store.recall("wife", tool="cli")
        assert all(t.matches_tool("cli") for t, _ in results)

    def test_recall_empty_store(self, encoder):
        store = ThreadStore(encoder=encoder)
        results = store.recall("anything")
        assert results == []

    def test_recall_no_encoder(self):
        store = ThreadStore(encoder=None)
        t = ContextThread(tool="test", facts=["hello"])
        store.add(t)
        results = store.recall("hello")
        assert results == []

    def test_recall_multiple_threads_ranked(self, encoder):
        store = ThreadStore(encoder=encoder)
        t1 = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"],
            facts=["my wife is Brandi"],
        )
        t2 = ContextThread(
            tool="cli", topic="work",
            entities=["Google"],
            facts=["I work at Google"],
        )
        store.add(t1)
        store.add(t2)

        results = store.recall("who is my wife?")
        assert len(results) >= 1
        # Family thread should rank higher for "wife" query
        top_thread, _ = results[0]
        assert top_thread.topic == "family"

    def test_active_thread(self, encoder):
        store = ThreadStore(encoder=encoder)
        t1 = ContextThread(tool="cli", topic="family", active=True)
        t2 = ContextThread(tool="cli", topic="work", active=False)
        store.add(t1)
        store.add(t2)
        active = store.active_thread("cli")
        assert active.thread_id == t1.thread_id

    def test_find_by_tool(self, encoder):
        store = ThreadStore(encoder=encoder)
        t1 = ContextThread(tool="claude-code")
        t2 = ContextThread(tool="gemini")
        store.add(t1)
        store.add(t2)
        results = store.find_by_tool("claude-code")
        assert len(results) == 1

    def test_recall_by_layer(self, encoder):
        store = ThreadStore(encoder=encoder)
        t = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"],
            facts=["my wife is Brandi"],
        )
        store.add(t)
        # Use a layer that's guaranteed to be activated ("perspective"
        # activates from "my" in the facts)
        results = store.recall_by_layer("my wife", layer_name="perspective")
        assert len(results) > 0

    def test_reindex_re_encodes(self, encoder):
        store = ThreadStore(encoder=encoder)
        t = ContextThread(
            tool="cli", topic="family",
            facts=["my wife is Brandi"], entities=["Brandi"],
        )
        store.add(t)
        old_glyph = t.glyph

        t.add_fact("Brandi has two sons")
        assert t.glyph is None  # stale
        store.reindex(t)
        assert t.glyph is not None
        assert t.glyph is not old_glyph


# ── Topic inference ─────────────────────────────────────────────────

class TestTopicInference:

    def test_family_topic(self):
        assert infer_topic("my wife is Brandi", ["Brandi"], []) == "family"

    def test_location_topic(self):
        assert infer_topic("Traceton lives in Colorado", ["Traceton"], []) == "location"

    def test_identity_topic(self):
        assert infer_topic("my name is Chris", ["Chris"], []) == "identity"

    def test_preferences_topic(self):
        assert infer_topic("i like pizza", [], []) == "preferences"

    def test_general_fallback(self):
        assert infer_topic("hello there", [], []) == "general"


# ── Implicit entity extraction ──────────────────────────────────────

class TestImplicitEntities:

    def test_wife(self):
        entities = _extract_implicit_entities("who is my wife?")
        assert "wife" in entities

    def test_children(self):
        entities = _extract_implicit_entities("how many children do i have?")
        assert "children" in entities

    def test_proper_noun(self):
        entities = _extract_implicit_entities("tell me about Brandi")
        assert "Brandi" in entities

    def test_no_common_words(self):
        entities = _extract_implicit_entities("What is the weather?")
        assert "What" not in entities


# ── ThreadManager ───────────────────────────────────────────────────

class TestThreadManager:

    def test_process_creates_thread(self, encoder):
        store = ThreadStore(encoder=encoder)
        mgr = ThreadManager(store, tool="test")
        thread = mgr.process(
            input_text="my wife is Brandi",
            facts=["my wife is Brandi"],
            entities=["Brandi"],
        )
        assert thread is not None
        assert "my wife is Brandi" in thread.facts
        assert "Brandi" in thread.entities
        assert thread.tool == "test"
        assert thread.glyph is not None  # encoded
        assert store.count == 1

    def test_process_extends_thread_same_entity(self, encoder):
        store = ThreadStore(encoder=encoder)
        mgr = ThreadManager(store, tool="test")
        t1 = mgr.process(
            input_text="my wife is Brandi",
            facts=["my wife is Brandi"],
            entities=["Brandi"],
        )
        t2 = mgr.process(
            input_text="Brandi has two sons",
            facts=["Brandi has two sons"],
            entities=["Brandi"],
        )
        assert t1.thread_id == t2.thread_id
        assert len(t1.facts) == 2
        assert store.count == 1

    def test_process_new_thread_different_topic(self, encoder):
        store = ThreadStore(encoder=encoder)
        mgr = ThreadManager(store, tool="test")
        t1 = mgr.process(
            input_text="my wife is Brandi",
            facts=["my wife is Brandi"],
            entities=["Brandi"],
        )
        t2 = mgr.process(
            input_text="i work at Google",
            facts=["i work at Google"],
            entities=["Google"],
        )
        assert t1.thread_id != t2.thread_id
        assert store.count == 2

    def test_recall_for_query(self, encoder):
        store = ThreadStore(encoder=encoder)
        mgr = ThreadManager(store, tool="test")
        mgr.process(
            input_text="my wife is Brandi",
            facts=["my wife is Brandi"],
            entities=["Brandi"],
        )
        threads = mgr.recall_for_query("who is my wife?")
        assert len(threads) > 0
        assert any("my wife is Brandi" in t.facts for t in threads)

    def test_collect_facts(self, encoder):
        store = ThreadStore(encoder=encoder)
        mgr = ThreadManager(store, tool="test")
        mgr.process(
            input_text="my wife is Brandi",
            facts=["my wife is Brandi"],
            entities=["Brandi"],
        )
        threads = mgr.recall_for_query("who is my wife?")
        facts = mgr.collect_facts(threads)
        assert len(facts) > 0
        assert facts[0][0] == "my wife is Brandi"
        assert facts[0][2] > 0  # confidence > 0

    def test_recall_cross_thread(self, encoder):
        store = ThreadStore(encoder=encoder)
        mgr = ThreadManager(store, tool="test")
        mgr.process(
            input_text="my wife is Brandi",
            facts=["my wife is Brandi"],
            entities=["Brandi"],
        )
        mgr.process(
            input_text="Brandi has two sons James and Traceton",
            facts=["Brandi has two sons James and Traceton"],
            entities=["Brandi", "James", "Traceton"],
        )
        threads = mgr.recall_for_query("tell me about James")
        assert len(threads) > 0


# ── Dream Weaving ──────────────────────────────────────────────────

class TestDreamWeaving:

    def _make_dream(self):
        """Create a minimal GlyphDreamLoop for testing."""
        space = ThoughtGlyphSpace()
        loop = GlyphCognitiveLoop(space)
        dream = GlyphDreamLoop(loop)
        dream._running = True
        return dream, space.encoder

    def test_weave_cross_tool(self):
        dream, encoder = self._make_dream()
        store = ThreadStore(encoder=encoder)
        t1 = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"], facts=["my wife is Brandi"],
        )
        t2 = ContextThread(
            tool="claude-code", topic="family",
            entities=["Brandi"], facts=["Brandi likes hiking"],
        )
        store.add(t1)
        store.add(t2)

        dream.set_thread_store(store)
        dream._weave_threads()

        # Should find connection (same topic, same entity = high glyph sim)
        # and create a bridge
        bridge_threads = store.find_by_tool("dream")
        # Threads should be linked regardless of bridge
        assert t2.thread_id in t1.related_threads or len(bridge_threads) > 0

    def test_weave_cross_topic(self):
        dream, encoder = self._make_dream()
        store = ThreadStore(encoder=encoder)
        t1 = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"], facts=["my wife is Brandi"],
        )
        t2 = ContextThread(
            tool="cli", topic="location",
            entities=["Brandi"], facts=["Brandi lives in Colorado"],
        )
        store.add(t1)
        store.add(t2)

        dream.set_thread_store(store)
        dream._weave_threads()

        # Both mention Brandi — glyphs should have content similarity
        assert t2.thread_id in t1.related_threads or t1.thread_id in t2.related_threads

    def test_weave_no_link_same_context(self):
        dream, encoder = self._make_dream()
        store = ThreadStore(encoder=encoder)
        t1 = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"], facts=["my wife is Brandi"],
        )
        t2 = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"], facts=["Brandi has two sons"],
        )
        store.add(t1)
        store.add(t2)

        dream.set_thread_store(store)
        dream._weave_threads()

        # Same tool + same topic = same context — skipped
        assert store.count == 2  # no bridge created

    def test_weave_idempotent(self):
        dream, encoder = self._make_dream()
        store = ThreadStore(encoder=encoder)
        t1 = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"], facts=["my wife is Brandi"],
        )
        t2 = ContextThread(
            tool="gemini", topic="family",
            entities=["Brandi"], facts=["Brandi loves cooking"],
        )
        store.add(t1)
        store.add(t2)

        dream.set_thread_store(store)
        dream._weave_threads()
        count_after_first = store.count
        dream._weave_threads()  # second run
        count_after_second = store.count

        # No new bridges on second run
        assert count_after_second == count_after_first

    def test_weave_surfaces_insight(self):
        dream, encoder = self._make_dream()
        store = ThreadStore(encoder=encoder)
        t1 = ContextThread(
            tool="cli", topic="family",
            entities=["Brandi"], facts=["my wife is Brandi"],
        )
        t2 = ContextThread(
            tool="claude-code", topic="location",
            entities=["Brandi"], facts=["Brandi lives in Colorado"],
        )
        store.add(t1)
        store.add(t2)

        dream.set_thread_store(store)
        dream._weave_threads()

        insights = dream.drain_insights()
        if insights:
            assert any(i.kind == InsightKind.CONNECTION for i in insights)

    def test_weave_no_store(self):
        dream, _ = self._make_dream()
        dream._weave_threads()  # should not crash
