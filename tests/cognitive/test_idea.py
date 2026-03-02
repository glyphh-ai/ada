"""Tests for IdeaEncoder and IdeaSpace."""

import numpy as np
import pytest

from glyphh.cognitive.idea import IdeaEncoder, IdeaSpace, Idea


DIM = 1000  # Small for fast tests


class TestIdeaEncoder:
    """IdeaEncoder: compose situations into HDC vectors."""

    @pytest.fixture
    def encoder(self):
        return IdeaEncoder(dimension=DIM, seed=42)

    def test_encode_returns_ndarray(self, encoder):
        vec = encoder.encode(action="search", target="report")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (DIM,)

    def test_encode_bipolar(self, encoder):
        vec = encoder.encode(action="search", target="report")
        unique = set(vec.tolist())
        assert unique <= {-1, 1}

    def test_same_input_same_output(self, encoder):
        v1 = encoder.encode(action="search", target="report")
        v2 = encoder.encode(action="search", target="report")
        assert np.array_equal(v1, v2)

    def test_different_actions_different_vectors(self, encoder):
        v1 = encoder.encode(action="search")
        v2 = encoder.encode(action="create")
        # Different actions should produce different vectors
        assert not np.array_equal(v1, v2)

    def test_similar_ideas_higher_cosine(self, encoder):
        from glyphh.core.ops import cosine_similarity
        # Same action+target with different keywords should be more similar
        # than completely different ideas
        v1 = encoder.encode(action="search", target="report", keywords=["budget"])
        v2 = encoder.encode(action="search", target="report", keywords=["finance"])
        v3 = encoder.encode(action="delete", target="user", keywords=["admin"])

        sim_close = cosine_similarity(v1, v2)
        sim_far = cosine_similarity(v1, v3)
        assert sim_close > sim_far

    def test_empty_encode_returns_fallback(self, encoder):
        vec = encoder.encode()
        assert vec.shape == (DIM,)
        unique = set(vec.tolist())
        assert unique <= {-1, 1}

    def test_keywords_limit(self, encoder):
        # Should not error with many keywords (capped at 12)
        many_kw = [f"word{i}" for i in range(50)]
        vec = encoder.encode(action="search", keywords=many_kw)
        assert vec.shape == (DIM,)

    def test_context_encoded(self, encoder):
        v1 = encoder.encode(action="search", context=["prev_action"])
        v2 = encoder.encode(action="search", context=["different_action"])
        assert not np.array_equal(v1, v2)

    def test_encode_from_query(self, encoder):
        vec = encoder.encode_from_query(
            query="search for budget report",
            intent={"action": "search", "target": "report", "keywords": "budget"},
            state="root.workspace",
            recent_actions=["navigate"],
        )
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (DIM,)


class TestIdea:
    """Idea dataclass: Hebbian reinforcement."""

    def test_initial_strength(self):
        idea = Idea(
            vector=np.ones(10, dtype=np.int8),
            outcome=[{"lookup": {"pattern": "test"}}],
        )
        assert idea.strength == 1.0
        assert idea.fire_count == 0

    def test_strengthen(self):
        idea = Idea(vector=np.ones(10, dtype=np.int8), outcome=[])
        idea.strengthen(amount=0.15)
        assert idea.strength > 1.0
        assert idea.fire_count == 1

    def test_strengthen_diminishing_returns(self):
        idea = Idea(vector=np.ones(10, dtype=np.int8), outcome=[])
        idea.strengthen(0.15)
        first_delta = idea.strength - 1.0
        old = idea.strength
        idea.strengthen(0.15)
        second_delta = idea.strength - old
        # Diminishing returns: second delta smaller
        assert second_delta < first_delta

    def test_strengthen_capped(self):
        idea = Idea(vector=np.ones(10, dtype=np.int8), outcome=[])
        for _ in range(100):
            idea.strengthen(0.5)
        assert idea.strength <= 3.0

    def test_weaken(self):
        idea = Idea(vector=np.ones(10, dtype=np.int8), outcome=[], strength=2.0)
        idea.weaken(factor=0.85)
        assert idea.strength == pytest.approx(1.7)

    def test_weaken_floored(self):
        idea = Idea(vector=np.ones(10, dtype=np.int8), outcome=[], strength=0.5)
        for _ in range(100):
            idea.weaken(0.5)
        assert idea.strength >= 0.3


class TestIdeaSpace:
    """IdeaSpace: episodic memory with recall and reinforcement."""

    @pytest.fixture
    def space(self):
        return IdeaSpace(dimension=DIM, seed=42)

    def test_empty_recall(self, space):
        query = space.encoder.encode(action="search")
        assert space.recall(query) == []

    def test_store_and_recall(self, space):
        vec = space.encoder.encode(action="search", target="report")
        outcome = [{"lookup": {"pattern": "budget"}}]
        space.store(vec, outcome, label="test_idea")

        results = space.recall(vec, top_k=1)
        assert len(results) == 1
        idea, score = results[0]
        assert idea.label == "test_idea"
        assert idea.outcome == outcome
        assert score > 0.0

    def test_recall_similarity_ordering(self, space):
        # Store two ideas: one similar, one different
        similar = space.encoder.encode(action="search", target="report")
        different = space.encoder.encode(action="delete", target="user")

        space.store(similar, [{"lookup": {}}], label="similar")
        space.store(different, [{"delete": {}}], label="different")

        query = space.encoder.encode(action="search", target="report", keywords=["budget"])
        results = space.recall(query, top_k=2)

        # Similar idea should rank higher
        assert results[0][0].label == "similar"

    def test_recall_respects_min_similarity(self, space):
        vec = space.encoder.encode(action="search")
        space.store(vec, [{"lookup": {}}])

        # Query with completely different idea
        different = space.encoder.encode(action="delete", target="user")
        results = space.recall(different, min_similarity=0.9)
        assert len(results) == 0

    def test_temporal_decay(self, space):
        vec = space.encoder.encode(action="search", target="report")
        space.store(vec, [{"lookup": {}}])

        # Recall immediately — high score
        res1 = space.recall(vec, top_k=1)
        score_fresh = res1[0][1]

        # Advance many ticks
        for _ in range(20):
            space.tick()

        # Recall again — lower score due to decay
        res2 = space.recall(vec, top_k=1)
        score_aged = res2[0][1]

        assert score_aged < score_fresh

    def test_reinforce_strengthens(self, space):
        vec = space.encoder.encode(action="search")
        idea = space.store(vec, [{"lookup": {}}])
        original = idea.strength

        space.recall(vec)
        space.reinforce_last(was_correct=True)

        assert idea.strength > original

    def test_reinforce_weakens(self, space):
        vec = space.encoder.encode(action="search")
        idea = space.store(vec, [{"lookup": {}}])
        original = idea.strength

        space.recall(vec)
        space.reinforce_last(was_correct=False)

        assert idea.strength < original

    def test_tick_advances_turn(self, space):
        assert space.turn == 0
        space.tick()
        assert space.turn == 1

    def test_size(self, space):
        assert space.size == 0
        vec = space.encoder.encode(action="search")
        space.store(vec, [])
        assert space.size == 1

    def test_clear(self, space):
        vec = space.encoder.encode(action="search")
        space.store(vec, [])
        space.tick()
        space.clear()
        assert space.size == 0
        assert space.turn == 0

    def test_reset_keeps_ideas(self, space):
        vec = space.encoder.encode(action="search")
        space.store(vec, [])
        space.reset()
        assert space.size == 1  # Ideas preserved
        assert space.turn == 0  # Turn reset

    def test_seed_from_episodes(self, space):
        episodes = [
            {
                "query": "search for budget",
                "intent": {"action": "search", "target": "report", "keywords": "budget"},
                "outcome": [{"lookup": {"pattern": "budget"}}],
                "label": "ep1",
            },
            {
                "query": "create new item",
                "intent": {"action": "create", "target": "item"},
                "outcome": [{"make": {"item_name": "test"}}],
                "label": "ep2",
            },
            {
                "query": "empty outcome",
                "intent": {"action": "list"},
                "outcome": [],  # Should be skipped
            },
        ]
        count = space.seed_from_episodes(episodes, state="root")
        assert count == 2
        assert space.size == 2
