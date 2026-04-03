"""
Tests for Ada's DreamLoop — background reasoning with curiosity + guardrails.
"""

import time

import pytest

from glyphh.memory.atom import AtomForge
from glyphh.memory.binding import FactStore
from glyphh.memory.cognitive import CognitiveLoop
from glyphh.memory.dream import DreamLoop, InsightKind, Insight
from glyphh.memory.thought import ThoughtEncoder
from glyphh.memory.store import ThoughtStore


@pytest.fixture
def system():
    forge = AtomForge(dimension=2048)
    facts = FactStore(forge)
    loop = CognitiveLoop(forge, facts)
    return forge, facts, loop


@pytest.fixture
def org_chart(system):
    """Teach an org chart for reasoning tests."""
    forge, facts, loop = system
    facts.teach("alice", "manages", "payments_team")
    facts.teach("bob", "manages", "infrastructure_team")
    facts.teach("payments_team", "owns", "billing_system")
    facts.teach("infrastructure_team", "owns", "database")
    facts.teach("billing_system", "is", "down")
    return forge, facts, loop


class TestDreamLifecycle:

    def test_start_stop(self, org_chart):
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        assert not dream.is_running

        dream.start()
        assert dream.is_running

        dream.stop()
        assert not dream.is_running

    def test_double_start_is_safe(self, org_chart):
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        dream.start()
        dream.start()  # should not crash or spawn a second thread
        assert dream.is_running
        dream.stop()

    def test_pause_resume(self, org_chart):
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        dream.start()
        assert dream.is_running

        dream.pause()
        assert not dream.is_running

        dream.resume()
        assert dream.is_running

        dream.stop()

    def test_stats(self, org_chart):
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        stats = dream.stats
        assert "cycles" in stats
        assert "total_chains" in stats
        assert "total_insights" in stats
        assert stats["cycles"] == 0


class TestDreamThinking:

    def test_discovers_connections(self, org_chart):
        """After a few cycles, dream should discover multi-hop chains."""
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1, cycle_budget=20)
        dream.start()
        time.sleep(3.0)  # let it think longer — needs multiple cycles to explore
        dream.stop()

        insights = dream.drain_insights()
        # Should have found at least one insight (connection, convergence, etc.)
        assert len(insights) >= 1

    def test_background_patterns_capped(self, org_chart):
        """Background-discovered patterns should not exceed self_reinforce_cap."""
        forge, facts, loop = org_chart
        dream = DreamLoop(
            loop, forge, facts,
            cycle_interval=0.1,
            cycle_budget=10,
            self_reinforce_cap=0.5,
        )
        dream.start()
        time.sleep(1.5)
        dream.stop()

        # Check that no pathway exceeds the cap
        for name, pathway in loop.library._pathways.items():
            assert pathway.strength <= 0.55  # small tolerance for race

    def test_cycles_increment(self, org_chart):
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        dream.start()
        time.sleep(0.8)
        dream.stop()
        assert dream.stats["cycles"] >= 2

    def test_total_chains_tracked(self, org_chart):
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1, cycle_budget=5)
        dream.start()
        time.sleep(1.0)
        dream.stop()
        assert dream.stats["total_chains"] > 0


class TestInsightQueue:

    def test_drain_empties_queue(self, org_chart):
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        dream.start()
        time.sleep(1.0)
        dream.stop()

        insights = dream.drain_insights()
        remaining = dream.drain_insights()
        assert len(remaining) == 0

    def test_peek_count(self, org_chart):
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        dream.start()
        time.sleep(1.0)
        dream.stop()

        count = dream.peek_insights()
        insights = dream.drain_insights()
        assert count == len(insights)

    def test_insight_has_fields(self, org_chart):
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        dream.start()
        time.sleep(1.0)
        dream.stop()

        insights = dream.drain_insights()
        if insights:
            insight = insights[0]
            assert isinstance(insight.kind, InsightKind)
            assert isinstance(insight.summary, str)
            assert isinstance(insight.atoms, list)
            assert insight.timestamp > 0


class TestCuriosity:

    def test_curiosity_prefers_fresh_atoms(self, system):
        """Atoms not yet explored should have higher curiosity."""
        forge, facts, loop = system
        facts.teach("alpha", "is", "new")
        facts.teach("beta", "is", "new")

        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        # Manually update curiosity
        atoms = forge.all_atoms()
        dream._update_curiosity(atoms)

        # All non-role atoms should have curiosity > 0
        for name, state in dream._curiosity.items():
            assert state.score > 0


class TestGaps:

    def test_find_gaps(self, org_chart):
        """Should find knowledge gaps (missing transitive links)."""
        forge, facts, loop = org_chart
        dream = DreamLoop(loop, forge, facts)
        gaps = dream.find_gaps()
        # alice → payments_team → billing_system, but alice has no direct link to billing_system
        assert len(gaps) >= 1
        gap_summaries = " ".join(g.summary for g in gaps)
        # Should ask about connections that exist transitively but not directly
        assert len(gap_summaries) > 0


class TestGuardrails:

    def test_self_reinforce_cap_respected(self, org_chart):
        """DreamLoop should never self-promote a pattern above the cap."""
        forge, facts, loop = org_chart
        dream = DreamLoop(
            loop, forge, facts,
            cycle_interval=0.1,
            self_reinforce_cap=0.4,
        )

        # Pre-create a chain so there's a pattern to reinforce
        chains = loop.reason("alice", "owns")
        assert len(chains) >= 1

        dream.start()
        time.sleep(1.5)
        dream.stop()

        # All patterns should be capped
        for name, pathway in loop.library._pathways.items():
            assert pathway.strength <= 0.45  # tolerance

    def test_user_confirm_overrides_cap(self, org_chart):
        """User confirmation should still strengthen above the dream cap."""
        forge, facts, loop = org_chart
        dream = DreamLoop(
            loop, forge, facts,
            cycle_interval=0.1,
            self_reinforce_cap=0.4,
        )

        chains = loop.reason("alice", "owns")
        loop.confirm(chains[0])
        name = chains[0].pathway_name

        # User-confirmed pattern should be above cap
        assert loop.library._pathways[name].strength > 0.5


class TestThoughtAbsorption:

    def test_absorbs_co_occurrences(self, system):
        """DreamLoop should create atoms and co-occurrence facts from thoughts."""
        forge, facts, loop = system
        thoughts = ThoughtStore(ThoughtEncoder(), storage_dir="/tmp/test_dream_thoughts")

        thoughts.remember("Chris is a developer")
        thoughts.remember("Chris likes coffee")

        dream = DreamLoop(
            loop, forge, facts,
            thoughts=thoughts,
            cycle_interval=0.1,
        )
        dream.start()
        time.sleep(1.0)
        dream.stop()

        # Atoms should exist for content words
        assert forge.has("chris")
        assert forge.has("developer")
        assert forge.has("coffee")
        # Co-occurrence facts should exist
        assert facts.count >= 2
        assert dream.stats["thoughts_absorbed"] == 2

    def test_repetition_strengthens(self, system):
        """Same co-occurrence in multiple thoughts should reinforce."""
        forge, facts, loop = system
        thoughts = ThoughtStore(ThoughtEncoder(), storage_dir="/tmp/test_dream_thoughts2")

        # "chris" and "name" co-occur twice
        thoughts.remember("my name is chris")
        thoughts.remember("chris is my name")

        dream = DreamLoop(
            loop, forge, facts,
            thoughts=thoughts,
            cycle_interval=0.1,
        )
        dream.start()
        time.sleep(1.0)
        dream.stop()

        assert forge.has("chris")
        assert forge.has("name")
        assert dream.stats["thoughts_absorbed"] == 2

    def test_does_not_reprocess_thoughts(self, system):
        """Already-absorbed thoughts should not be processed again."""
        forge, facts, loop = system
        thoughts = ThoughtStore(ThoughtEncoder(), storage_dir="/tmp/test_dream_thoughts3")

        thoughts.remember("Alice manages payments")

        dream = DreamLoop(
            loop, forge, facts,
            thoughts=thoughts,
            cycle_interval=0.1,
        )
        dream.start()
        time.sleep(0.5)
        dream.stop()

        fact_count_after_first = facts.count

        # Run again — same thought should not create duplicate facts
        dream.start()
        time.sleep(0.5)
        dream.stop()

        assert facts.count == fact_count_after_first
        assert dream.stats["thoughts_absorbed"] == 1

    def test_works_without_thought_store(self, system):
        """DreamLoop should work fine if no ThoughtStore is provided."""
        forge, facts, loop = system
        facts.teach("alpha", "is", "new")

        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        dream.start()
        time.sleep(0.5)
        dream.stop()

        assert dream.stats["cycles"] >= 1
        assert dream.stats["thoughts_absorbed"] == 0

    def test_single_word_gets_direction(self, system):
        """Even single-word thoughts get tagged with direction."""
        forge, facts, loop = system
        thoughts = ThoughtStore(ThoughtEncoder(), storage_dir="/tmp/test_dream_thoughts4")

        thoughts.remember("hello", metadata={"speaker": "incoming"})

        dream = DreamLoop(
            loop, forge, facts,
            thoughts=thoughts,
            cycle_interval=0.1,
        )
        dream.start()
        time.sleep(0.5)
        dream.stop()

        assert dream.stats["thoughts_absorbed"] == 1
        # "hello" + "incoming" = 2 words, creates a co-occurrence
        assert forge.has("hello")


class TestEmptyMemory:

    def test_dream_with_no_facts(self, system):
        """DreamLoop should handle empty memory gracefully."""
        forge, facts, loop = system
        dream = DreamLoop(loop, forge, facts, cycle_interval=0.1)
        dream.start()
        time.sleep(0.5)
        dream.stop()

        insights = dream.drain_insights()
        assert len(insights) == 0
        assert dream.stats["cycles"] >= 1
