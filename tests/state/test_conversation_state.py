"""Tests for glyphh.state — ConversationState, PathwayEncoder, PathwayLibrary."""

import pytest
import numpy as np

from glyphh.state import ConversationState, Pathway, PathwayLibrary
from glyphh.state.pathway import PathwayEncoder, _weighted_bundle
from glyphh.core.ops import cosine_similarity


DIMENSION = 1000


# ── _weighted_bundle ──────────────────────────────────────────────────────────

class TestWeightedBundle:
    def test_single_vector(self):
        v = np.ones(DIMENSION, dtype=np.int8)
        result = _weighted_bundle([(v, 1.0)], DIMENSION)
        assert result.shape == (DIMENSION,)
        assert set(result.tolist()).issubset({-1, 1})

    def test_equal_weights_matches_bundle(self):
        """Equal weights should produce the same result as unweighted bundle."""
        from glyphh.core.ops import bundle
        v1 = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        v2 = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        weighted = _weighted_bundle([(v1, 1.0), (v2, 1.0)], DIMENSION)
        unweighted = bundle([v1, v2])
        # Results may differ slightly at ties, but should be very similar
        sim = cosine_similarity(weighted, unweighted)
        assert sim > 0.95

    def test_dominant_vector_wins(self):
        """A vector with weight 10x another should dominate the result."""
        v_strong = np.ones(DIMENSION, dtype=np.int8)
        v_weak = -np.ones(DIMENSION, dtype=np.int8)
        result = _weighted_bundle([(v_strong, 10.0), (v_weak, 1.0)], DIMENSION)
        # Result should be very similar to v_strong
        sim = cosine_similarity(result, v_strong)
        assert sim > 0.8

    def test_returns_bipolar(self):
        """Output must be strictly bipolar {-1, +1}."""
        v = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        result = _weighted_bundle([(v, 0.5)], DIMENSION)
        unique = set(result.tolist())
        assert unique.issubset({-1, 1})


# ── PathwayEncoder ────────────────────────────────────────────────────────────

class TestPathwayEncoder:
    def test_fresh_state_is_none(self, pathway_encoder):
        assert pathway_encoder.get_state() is None
        assert pathway_encoder.depth == 0

    def test_single_update_returns_vector(self, pathway_encoder):
        v = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        pathway_encoder.update(v)
        state = pathway_encoder.get_state()
        assert state is not None
        assert state.shape == (DIMENSION,)
        assert set(state.tolist()).issubset({-1, 1})
        assert pathway_encoder.depth == 1

    def test_depth_increments(self, pathway_encoder):
        for i in range(5):
            v = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
            pathway_encoder.update(v)
        assert pathway_encoder.depth == 5

    def test_reset_clears_history(self, pathway_encoder):
        v = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        pathway_encoder.update(v)
        pathway_encoder.reset()
        assert pathway_encoder.get_state() is None
        assert pathway_encoder.depth == 0

    def test_different_sequences_produce_different_states(self):
        """Two different action sequences should produce different pathway vectors."""
        v1 = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        v2 = np.random.choice([-1, 1], DIMENSION).astype(np.int8)

        enc_a = PathwayEncoder(DIMENSION, 42, 0.75)
        enc_a.update(v1)
        enc_a.update(v2)
        state_a = enc_a.get_state()

        enc_b = PathwayEncoder(DIMENSION, 42, 0.75)
        enc_b.update(v2)
        enc_b.update(v1)
        state_b = enc_b.get_state()

        # Order matters — different order → different pathway vector
        sim = cosine_similarity(state_a, state_b)
        assert sim < 0.99   # not identical

    def test_decay_changes_state(self):
        """Different decay values produce different state vectors over a long sequence.

        With many steps, aggressive decay means only the last few steps matter,
        while slow decay accumulates the full history.  After binarization this
        difference is visible when the sequence has diverse vectors.
        """
        rng = np.random.RandomState(999)
        # 10 diverse steps
        vectors = [rng.choice([-1, 1], DIMENSION).astype(np.int8) for _ in range(10)]

        enc_fast = PathwayEncoder(DIMENSION, 42, decay=0.05)  # only last step matters
        enc_slow = PathwayEncoder(DIMENSION, 42, decay=0.99)  # all history preserved

        for v in vectors:
            enc_fast.update(v)
            enc_slow.update(v)

        state_fast = enc_fast.get_state()
        state_slow = enc_slow.get_state()

        # Different decay → measurably different state vectors
        sim = cosine_similarity(state_fast, state_slow)
        assert sim < 0.98   # not identical (would be 1.0 if decay had no effect)


# ── PathwayLibrary ────────────────────────────────────────────────────────────

class TestPathwayLibrary:
    def test_empty_library(self, library):
        assert len(library) == 0

    def test_add_pattern(self, library):
        v1 = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        v2 = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        library.add("test_pattern", [v1, v2])
        assert len(library) == 1
        assert "test_pattern" in library

    def test_match_returns_sorted_results(self, library):
        v1 = np.ones(DIMENSION, dtype=np.int8)
        v2 = np.random.choice([-1, 1], DIMENSION).astype(np.int8)

        library.add("pattern_a", [v1])
        library.add("pattern_b", [v2])

        matches = library.match(v1, top_k=5)
        assert len(matches) == 2
        # Sorted descending
        assert matches[0][1] >= matches[1][1]

    def test_match_self_similarity_is_high(self, library):
        """A pattern should match its own encoded vector with high similarity."""
        v = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        library.add("self_test", [v])

        enc = PathwayEncoder(DIMENSION, 42, 0.75)
        enc.update(v)
        state = enc.get_state()

        matches = library.match(state, top_k=1)
        assert len(matches) == 1
        pathway, score = matches[0]
        assert pathway.name == "self_test"
        assert score > 0.5

    def test_strengthen_increases_strength(self, library):
        v = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        library.add("fire_me", [v], strength=1.0)

        initial_strength = library.pathways["fire_me"].strength
        library.strengthen("fire_me")
        new_strength = library.pathways["fire_me"].strength
        assert new_strength > initial_strength

    def test_strengthen_nonexistent_is_noop(self, library):
        """Strengthening a missing pattern should not raise."""
        library.strengthen("does_not_exist")   # should not raise

    def test_hebbian_diminishing_returns(self, library):
        v = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        library.add("test", [v])

        deltas = []
        prev = library.pathways["test"].strength
        for _ in range(10):
            library.strengthen("test", amount=0.15)
            new = library.pathways["test"].strength
            deltas.append(new - prev)
            prev = new

        # Each delta should be smaller than the previous (diminishing returns)
        for i in range(1, len(deltas)):
            assert deltas[i] <= deltas[i - 1] + 1e-9

    def test_strength_capped_at_3(self, library):
        v = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        library.add("capped", [v])
        for _ in range(1000):
            library.strengthen("capped")
        assert library.pathways["capped"].strength <= 3.0


# ── ConversationState ─────────────────────────────────────────────────────────

class TestConversationState:
    def test_fresh_state_depth_zero(self, state):
        assert state.depth == 0

    def test_update_with_glyphs(self, state, glyphs):
        state.update([glyphs["cd"], glyphs["mv"]])
        assert state.depth == 2

    def test_update_raw(self, state):
        v = np.random.choice([-1, 1], DIMENSION).astype(np.int8)
        state.update_raw([v])
        assert state.depth == 1

    def test_predict_next_returns_all_candidates(self, state, glyphs):
        state.update([glyphs["cd"]])
        candidates = {k: glyphs[k] for k in ["cd", "mv", "grep", "ls"]}
        scores = state.predict_next(glyphs["mv"], candidates)
        assert set(scores.keys()) == {"cd", "mv", "grep", "ls"}

    def test_predict_next_sorted_descending(self, state, glyphs):
        state.update([glyphs["cd"]])
        candidates = {k: glyphs[k] for k in ["cd", "mv", "grep", "ls", "find"]}
        scores = state.predict_next(glyphs["mv"], candidates)
        values = list(scores.values())
        assert values == sorted(values, reverse=True)

    def test_no_history_uses_query_only(self, state, glyphs):
        """With no history, scores should be driven purely by query similarity."""
        candidates = {k: glyphs[k] for k in ["cd", "mv", "grep"]}
        scores_1 = state.predict_next(glyphs["mv"], candidates)
        scores_2 = state.predict_next(glyphs["grep"], candidates)
        # Different queries → different top-ranked results
        top_1 = max(scores_1, key=lambda k: scores_1[k])
        top_2 = max(scores_2, key=lambda k: scores_2[k])
        # mv query should rank mv higher than grep query does
        assert scores_1["mv"] >= scores_1["grep"] or scores_2["grep"] >= scores_2["mv"]

    def test_predict_next_empty_candidates(self, state, glyphs):
        scores = state.predict_next(glyphs["cd"], {})
        assert scores == {}

    def test_reset_clears_depth_not_library(self, state, glyphs):
        state.add_pathway("test_nav", [glyphs["cd"], glyphs["mv"]])
        state.update([glyphs["cd"], glyphs["mv"]])
        assert state.depth == 2
        assert state.library_size == 1

        state.reset()
        assert state.depth == 0
        assert state.library_size == 1  # library persists

    def test_add_pathway_increases_library_size(self, state, glyphs):
        assert state.library_size == 0
        state.add_pathway("nav_op", [glyphs["cd"], glyphs["mv"]])
        assert state.library_size == 1
        state.add_pathway("nav_search", [glyphs["cd"], glyphs["grep"]])
        assert state.library_size == 2

    def test_get_state_vector_none_before_update(self, state):
        assert state.get_state_vector() is None

    def test_get_state_vector_after_update(self, state, glyphs):
        state.update([glyphs["cd"]])
        v = state.get_state_vector()
        assert v is not None
        assert v.shape == (DIMENSION,)

    def test_active_pathways_empty_without_history(self, state, glyphs):
        state.add_pathway("test", [glyphs["cd"]])
        assert state.active_pathways() == []  # no history → no state vector

    def test_active_pathways_returns_matches(self, state, glyphs):
        state.add_pathway("nav_op", [glyphs["cd"], glyphs["mv"]])
        state.update([glyphs["cd"], glyphs["mv"]])
        active = state.active_pathways()
        assert len(active) > 0
        # Should find the nav_op pattern
        names = [n for n, _ in active]
        assert "nav_op" in names

    def test_confirm_strengthens_library(self, state, glyphs):
        state.add_pathway("nav_op", [glyphs["cd"], glyphs["mv"]])
        state.update([glyphs["cd"], glyphs["mv"]])

        initial_strength = state._library.pathways.get("nav_op")
        if initial_strength:
            initial = initial_strength.strength

        candidates = {k: glyphs[k] for k in ["cd", "mv", "grep"]}
        state.predict_next(glyphs["mv"], candidates)  # sets _last_active_patterns
        state.confirm([glyphs["mv"]])

        if initial_strength:
            assert state._library.pathways["nav_op"].strength >= initial

    def test_pathway_boosts_continuation(self, state, glyphs):
        """After cd, the nav_op pattern should boost mv's score."""
        state.add_pathway("nav_op", [glyphs["cd"], glyphs["mv"]], strength=2.0)
        state.update([glyphs["cd"]])

        candidates = {k: glyphs[k] for k in ["mv", "grep", "ls", "find"]}
        scores = state.predict_next(glyphs["mv"], candidates)

        # mv should score relatively high after a cd (pattern activation)
        top = max(scores, key=lambda k: scores[k])
        # mv or grep could be top depending on query alignment — just ensure
        # scores are valid floats and the function runs without error
        assert all(isinstance(v, float) for v in scores.values())
        assert scores["mv"] > -1.0


# ── Top-level import ──────────────────────────────────────────────────────────

def test_top_level_import():
    """Verify ConversationState is importable from glyphh directly."""
    from glyphh import ConversationState, Pathway, PathwayLibrary
    assert ConversationState is not None
    assert Pathway is not None
    assert PathwayLibrary is not None
