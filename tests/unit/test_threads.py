"""
Tests for context threads — Ada's structured memory.

Covers: thread creation, entity/topic indexing, structured recall,
thread manager topic inference, thread lifecycle.
"""

import pytest
import time

from domains.brain.context_thread import ContextThread, ThreadStore
from domains.brain.thread_manager import (
    ThreadManager, infer_topic, _extract_implicit_entities,
)


# ── ContextThread ──────��─────────────────────────────────────────

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

    def test_matches_entity(self):
        t = ContextThread(entities=["Brandi", "Chris"])
        assert t.matches_entity("brandi")
        assert t.matches_entity("Brandi")
        assert not t.matches_entity("James")

    def test_matches_topic(self):
        t = ContextThread(topic="family")
        assert t.matches_topic("family")
        assert t.matches_topic("Family")
        assert not t.matches_topic("work")

    def test_matches_tool(self):
        t = ContextThread(tool="claude-code")
        assert t.matches_tool("claude-code")
        assert not t.matches_tool("gemini")

    def test_close(self):
        t = ContextThread()
        assert t.active
        t.close()
        assert not t.active

    def test_relevance_entity_match(self):
        t = ContextThread(entities=["wife", "Brandi"])
        score = t.relevance_to(["wife"])
        assert score > 0

    def test_relevance_no_match(self):
        t = ContextThread(entities=["Brandi"])
        score = t.relevance_to(["work", "project"])
        # No entity overlap — only recency boost (0.1 for <1h old)
        assert score <= 0.1


# ── ThreadStore ──────────────────────────────────────────────────

class TestThreadStore:

    def test_add_and_get(self):
        store = ThreadStore()
        t = ContextThread(tool="test", topic="family")
        store.add(t)
        assert store.get(t.thread_id) == t
        assert store.count == 1

    def test_find_by_entity(self):
        store = ThreadStore()
        t1 = ContextThread(tool="test", entities=["Brandi"])
        t2 = ContextThread(tool="test", entities=["James"])
        store.add(t1)
        store.add(t2)
        results = store.find_by_entity("Brandi")
        assert len(results) == 1
        assert results[0].thread_id == t1.thread_id

    def test_find_by_topic(self):
        store = ThreadStore()
        t1 = ContextThread(tool="test", topic="family")
        t2 = ContextThread(tool="test", topic="work")
        store.add(t1)
        store.add(t2)
        results = store.find_by_topic("family")
        assert len(results) == 1

    def test_find_by_tool(self):
        store = ThreadStore()
        t1 = ContextThread(tool="claude-code")
        t2 = ContextThread(tool="gemini")
        store.add(t1)
        store.add(t2)
        results = store.find_by_tool("claude-code")
        assert len(results) == 1

    def test_recall_by_entity(self):
        store = ThreadStore()
        t = ContextThread(tool="test", entities=["wife", "Brandi"])
        t.add_fact("my wife is Brandi")
        store.add(t)
        results = store.recall(entities=["wife"])
        assert len(results) == 1
        assert "my wife is Brandi" in results[0].facts

    def test_recall_empty(self):
        store = ThreadStore()
        results = store.recall(entities=["nothing"])
        assert results == []

    def test_recall_no_filters_returns_recent(self):
        store = ThreadStore()
        t1 = ContextThread(tool="test")
        t2 = ContextThread(tool="test")
        store.add(t1)
        store.add(t2)
        results = store.recall()
        assert len(results) == 2

    def test_active_thread(self):
        store = ThreadStore()
        t1 = ContextThread(tool="cli", topic="family", active=True)
        t2 = ContextThread(tool="cli", topic="work", active=False)
        store.add(t1)
        store.add(t2)
        active = store.active_thread("cli")
        assert active.thread_id == t1.thread_id


# ── Topic inference ──────────────────────────────────────────────

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


# ── Implicit entity extraction ──────────��────────────────────────

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


# ── ThreadManager ────────────────��───────────────────────────────

class TestThreadManager:

    def test_process_creates_thread(self):
        store = ThreadStore()
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
        assert store.count == 1

    def test_process_extends_thread_same_entity(self):
        store = ThreadStore()
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
        # Same thread — entity overlap
        assert t1.thread_id == t2.thread_id
        assert len(t1.facts) == 2
        assert store.count == 1

    def test_process_new_thread_different_topic(self):
        store = ThreadStore()
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
        # Different entities, different topic → new thread
        assert t1.thread_id != t2.thread_id
        assert store.count == 2

    def test_recall_for_query(self):
        store = ThreadStore()
        mgr = ThreadManager(store, tool="test")
        mgr.process(
            input_text="my wife is Brandi",
            facts=["my wife is Brandi"],
            entities=["Brandi"],
        )
        threads = mgr.recall_for_query("who is my wife?")
        assert len(threads) > 0
        # Should find the family thread via "wife" implicit entity
        assert any("my wife is Brandi" in t.facts for t in threads)

    def test_collect_facts(self):
        store = ThreadStore()
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

    def test_recall_cross_thread(self):
        store = ThreadStore()
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
        # Ask about a child
        threads = mgr.recall_for_query("tell me about James")
        assert len(threads) > 0
