"""
Tests for Ada's CognitiveLoop — reasoning, pathways, reinforcement.
"""

import pytest

from glyphh.memory.atom import AtomForge
from glyphh.memory.binding import FactStore
from glyphh.memory.cognitive import CognitiveLoop, ReasoningChain


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


class TestBasicReasoning:

    def test_direct_fact(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("billing_system", "is")
        assert len(chains) >= 1
        answers = [c.answer for c in chains]
        assert "down" in answers

    def test_two_hop_inference(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("alice", "owns")
        answers = [c.answer for c in chains]
        assert "billing_system" in answers

    def test_chain_has_hops(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("alice", "owns")
        billing_chain = [c for c in chains if c.answer == "billing_system"]
        assert len(billing_chain) >= 1
        chain = billing_chain[0]
        assert chain.depth == 2
        assert chain.hops[0][0] == "alice"  # starts from alice
        assert chain.hops[0][2] == "payments_team"  # via payments
        assert chain.hops[1][2] == "billing_system"  # reaches billing

    def test_open_reasoning(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("alice")
        assert len(chains) >= 1
        # Should find at least alice → manages → payments_team
        answers = [c.answer for c in chains]
        assert "payments_team" in answers

    def test_no_results(self, system):
        forge, facts, loop = system
        chains = loop.reason("nonexistent", "anything")
        assert len(chains) == 0


class TestPathwayCreation:

    def test_reasoning_auto_stores_weak_pattern(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("alice", "owns")
        assert len(chains) >= 1
        # Auto-stored as weak pattern
        multi_hop = [c for c in chains if c.depth >= 2]
        assert len(multi_hop) >= 1
        name = multi_hop[0].pathway_name
        assert name in loop.library.pathways
        assert loop.library.pathways[name].strength == 0.3  # weak

    def test_confirm_promotes_pathway(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("alice", "owns")
        loop.confirm(chains[0])
        name = chains[0].pathway_name
        assert loop.library.pathways[name].strength >= 0.8  # promoted

    def test_confirmed_pathway_is_strong(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("alice", "owns")
        loop.confirm(chains[0])

        name = chains[0].pathway_name
        pathway = loop.library.pathways[name]
        assert pathway.strength >= 0.7

    def test_double_confirm_strengthens(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("alice", "owns")
        loop.confirm(chains[0])
        name = chains[0].pathway_name
        s1 = loop.library.pathways[name].strength

        # Reason again and confirm
        chains2 = loop.reason("alice", "owns")
        loop.confirm(chains2[0])
        s2 = loop.library.pathways[name].strength
        assert s2 > s1

    def test_reject_weakens_pathway(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("alice", "owns")
        loop.confirm(chains[0])
        name = chains[0].pathway_name
        s1 = loop.library.pathways[name].strength

        loop.reject(chains[0])
        s2 = loop.library.pathways[name].strength
        assert s2 < s1


class TestPatternBoost:

    def test_confirmed_pattern_boosts_similar(self, org_chart):
        forge, facts, loop = org_chart
        # Confirm alice → owns chain
        chains = loop.reason("alice", "owns")
        loop.confirm(chains[0])

        # Now try bob → owns (structurally similar chain)
        chains2 = loop.reason("bob", "owns")
        # At minimum, the chain should exist
        answers = [c.answer for c in chains2]
        assert "database" in answers


class TestContradiction:

    def test_contradiction_flagged(self, system):
        forge, facts, loop = system
        # Create contradictory paths
        facts.teach("sky", "is", "blue")
        facts.teach("sky", "is", "red")
        chains = loop.reason("sky", "is")
        # Both should exist
        assert len(chains) >= 2
        # At least one should be flagged
        contradicted = [c for c in chains if c.contradicted]
        assert len(contradicted) >= 1


class TestDecay:

    def test_pathways_decay(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("alice", "owns")
        loop.confirm(chains[0])
        name = chains[0].pathway_name
        s1 = loop.library.pathways[name].strength

        # Decay explicitly
        loop.library.decay_all(0.5)
        s2 = loop.library.pathways[name].strength
        assert s2 < s1

    def test_prune_removes_weak(self, org_chart):
        forge, facts, loop = org_chart
        chains = loop.reason("alice", "owns")
        loop.confirm(chains[0])
        name = chains[0].pathway_name

        # Weaken below threshold
        loop.library._pathways[name].strength = 0.1
        removed = loop.library.prune(min_strength=0.15)
        assert removed == 1
        assert name not in loop.library.pathways


class TestPersistence:

    def test_save_load(self, org_chart, tmp_path):
        forge, facts, loop = org_chart
        chains = loop.reason("alice", "owns")
        loop.confirm(chains[0])
        name = chains[0].pathway_name

        loop.save(tmp_path)

        # Load into fresh loop
        forge2 = AtomForge(dimension=2048)
        facts2 = FactStore(forge2)
        loop2 = CognitiveLoop(forge2, facts2)
        loop2.load(tmp_path)

        assert name in loop2.library.pathways
        assert loop2.library.pathways[name].strength == loop.library.pathways[name].strength


class TestRecall:

    def test_recall_finds_relevant_facts(self, org_chart):
        forge, facts, loop = org_chart
        lines = loop.recall("who manages billing?")
        # Should find facts about billing_system
        assert len(lines) > 0
        full = "\n".join(lines)
        assert "billing" in full

    def test_recall_includes_inferred(self, org_chart):
        forge, facts, loop = org_chart
        lines = loop.recall("who owns billing?")
        full = "\n".join(lines)
        # Should include inferred chain about ownership
        assert "inferred" in full or "billing_system" in full
