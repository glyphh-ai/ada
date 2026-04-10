"""
Tests for ThoughtGlyphSpace — Ada's long-term memory.

Covers: absorb, dedup, recall ranking, user identity vs Ada identity,
content vector similarity, and the core recall pipeline.
"""

import pytest

from glyphh.memory.thought_space import ThoughtGlyphSpace, StoredThought, RecallResult


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def space():
    """Fresh ThoughtGlyphSpace for each test."""
    return ThoughtGlyphSpace()


@pytest.fixture
def space_with_user_identity(space):
    """Space seeded with user identity facts (multiple phrasings)."""
    space.absorb("The user's name is Chris.", speaker="ada")
    space.absorb("You are Chris.", speaker="ada")
    space.absorb("I am talking to Chris.", speaker="ada")
    return space


@pytest.fixture
def space_with_ada_seeds(space):
    """Space seeded with Ada's third-person identity (matches production seeds)."""
    space.absorb("My name is Ada.", speaker="ada")
    space.absorb("Ada is a cognitive brain for LLMs.", speaker="ada")
    space.absorb("Ada is not Claude. Ada is not ChatGPT.", speaker="ada")
    space.absorb("Ada thinks using hyperdimensional computing vectors.", speaker="ada")
    space.absorb("Ada controls what Haiku sees. Ada is the information boundary.", speaker="ada")
    space.absorb("Ada does not hallucinate. If she does not know something, she says so.", speaker="ada")
    space.absorb("Ada is concise and direct.", speaker="ada")
    space.absorb("Ada responds in one or two sentences.", speaker="ada")
    return space


@pytest.fixture
def space_with_both(space_with_ada_seeds):
    """Space with both Ada seeds and user identity."""
    space_with_ada_seeds.absorb("The user's name is Chris.", speaker="ada")
    space_with_ada_seeds.absorb("You are Chris.", speaker="ada")
    space_with_ada_seeds.absorb("I am talking to Chris.", speaker="ada")
    return space_with_ada_seeds


# ── Absorb tests ────────────────────────────────────────────────────────

class TestAbsorb:

    def test_absorb_returns_stored_thought(self, space):
        result = space.absorb("hello world")
        assert isinstance(result, StoredThought)
        assert result.content == "hello world"
        assert result.speaker == "incoming"

    def test_absorb_with_speaker(self, space):
        result = space.absorb("I am Ada.", speaker="ada")
        assert result.speaker == "ada"

    def test_absorb_dedup_exact(self, space):
        first = space.absorb("hello world")
        second = space.absorb("hello world")
        assert first is not None
        assert second is None

    def test_absorb_dedup_case_insensitive(self, space):
        first = space.absorb("Hello World")
        second = space.absorb("hello world")
        assert first is not None
        assert second is None

    def test_absorb_dedup_strips_whitespace(self, space):
        first = space.absorb("  hello world  ")
        second = space.absorb("hello world")
        assert first is not None
        assert second is None

    def test_absorb_empty_string_returns_none(self, space):
        assert space.absorb("") is None
        assert space.absorb("   ") is None

    def test_absorb_increments_count(self, space):
        assert space.count == 0
        space.absorb("first thought")
        assert space.count == 1
        space.absorb("second thought")
        assert space.count == 2

    def test_absorb_generates_unique_ids(self, space):
        t1 = space.absorb("first")
        t2 = space.absorb("second")
        assert t1.thought_id != t2.thought_id

    def test_absorb_creates_glyph(self, space):
        result = space.absorb("my name is chris")
        assert result.glyph is not None
        assert hasattr(result.glyph, 'layers')

    def test_absorb_default_strength(self, space):
        result = space.absorb("test thought")
        assert result.strength == 1.0


# ── Recall basic tests ──────────────────────────────────────────────────

class TestRecallBasic:

    def test_recall_empty_space(self, space):
        results = space.recall("anything")
        assert results == []

    def test_recall_returns_recall_results(self, space):
        space.absorb("the sky is blue")
        results = space.recall("what color is the sky")
        assert len(results) > 0
        assert isinstance(results[0], RecallResult)

    def test_recall_has_similarity_score(self, space):
        space.absorb("the cat sat on the mat")
        results = space.recall("where did the cat sit")
        assert len(results) > 0
        assert 0.0 <= results[0].global_similarity <= 1.0

    def test_recall_exact_match_high_similarity(self, space):
        space.absorb("my favorite color is blue")
        results = space.recall("my favorite color is blue")
        assert len(results) > 0
        assert results[0].global_similarity > 0.5

    def test_recall_respects_top_k(self, space):
        for i in range(10):
            space.absorb(f"thought number {i} about various topics")
        results = space.recall("thought", top_k=3)
        assert len(results) <= 3

    def test_recall_sorted_by_similarity(self, space):
        space.absorb("the weather is sunny today")
        space.absorb("cats are wonderful pets")
        space.absorb("the forecast says rain tomorrow")
        results = space.recall("what is the weather like")
        sims = [r.global_similarity for r in results]
        assert sims == sorted(sims, reverse=True)


# ── User identity recall (the critical bug) ─────────────────────────────

class TestUserIdentityRecall:

    def test_who_am_i_finds_user_identity(self, space_with_user_identity):
        """'who am i?' should match user identity facts."""
        results = space_with_user_identity.recall("who am i?", top_k=3)
        assert len(results) > 0
        top_content = results[0].thought.content.lower()
        assert "chris" in top_content

    def test_what_is_my_name_finds_user_identity(self, space_with_user_identity):
        """'what is my name?' should match user identity facts."""
        results = space_with_user_identity.recall("what is my name?", top_k=3)
        assert len(results) > 0
        top_content = results[0].thought.content.lower()
        assert "chris" in top_content

    def test_do_you_know_me_finds_user_identity(self, space_with_user_identity):
        """'do you know me?' should match user identity facts."""
        results = space_with_user_identity.recall("do you know me?", top_k=3)
        assert len(results) > 0
        # At least one result in top 3 should mention Chris
        contents = [r.thought.content.lower() for r in results[:3]]
        assert any("chris" in c for c in contents)

    def test_who_am_i_prefers_user_over_ada_identity(self, space_with_both):
        """When both Ada seeds and user identity exist, 'who am i?'
        should prefer the user identity over Ada's self-description."""
        results = space_with_both.recall("who am i?", top_k=3)
        assert len(results) > 0
        top_content = results[0].thought.content.lower()
        # Should match user identity, not "Ada is the information boundary"
        assert "chris" in top_content, (
            f"Expected user identity in top result, got: '{results[0].thought.content}'"
        )

    def test_what_is_my_name_prefers_user_over_ada(self, space_with_both):
        """'what is my name?' should rank user identity above Ada seeds."""
        results = space_with_both.recall("what is my name?", top_k=3)
        assert len(results) > 0
        top_content = results[0].thought.content.lower()
        assert "chris" in top_content, (
            f"Expected user name in top result, got: '{results[0].thought.content}'"
        )

    def test_ada_identity_questions_find_ada(self, space_with_both):
        """'what is your name?' should find Ada in top results."""
        results = space_with_both.recall("what is your name?", top_k=5)
        assert len(results) > 0
        # Ada's identity should appear somewhere in top results
        contents = [r.thought.content.lower() for r in results[:5]]
        assert any("ada" in c for c in contents), (
            f"Expected Ada in top 5 for 'what is your name?', got: {contents}"
        )

    def test_who_is_ada_finds_ada(self, space_with_both):
        """'who is Ada?' should find Ada's identity facts."""
        results = space_with_both.recall("who is Ada?", top_k=3)
        assert len(results) > 0
        top_content = results[0].thought.content.lower()
        assert "ada" in top_content


# ── Ada seed isolation tests ────────────────────────────────────────────

class TestAdaSeedIsolation:

    def test_information_boundary_not_top_for_who_am_i(self, space_with_both):
        """'I am the information boundary' should NOT be the top result
        for 'who am i?' — that's Ada's self-description, not user identity."""
        results = space_with_both.recall("who am i?", top_k=1)
        assert len(results) > 0
        assert "information boundary" not in results[0].thought.content.lower()

    def test_third_person_seeds_reduce_i_am_pollution(self, space_with_ada_seeds):
        """Third-person seeds ('Ada is...') should not dominate
        queries containing 'I am' or 'am I'."""
        # Add a user statement
        space_with_ada_seeds.absorb("I am a software engineer.", speaker="incoming")
        results = space_with_ada_seeds.recall("what am I?", top_k=3)
        # The user's statement should rank above Ada's self-description
        contents = [r.thought.content for r in results[:3]]
        assert any("software engineer" in c for c in contents), (
            f"User's self-description not in top 3: {contents}"
        )


# ── Reinforce / Decay tests ────────────────────────────────────────────

class TestReinforceDecay:

    def test_reinforce_increases_strength(self, space):
        t = space.absorb("important fact")
        original = t.strength
        space.reinforce(t.thought_id, amount=0.2)
        assert t.strength > original

    def test_strength_caps_at_3(self, space):
        t = space.absorb("very important")
        for _ in range(100):
            space.reinforce(t.thought_id, amount=1.0)
        assert t.strength <= 3.0

    def test_decay_reduces_strength(self, space):
        t = space.absorb("fading memory")
        original = t.strength
        space.decay_all(factor=0.5)
        assert t.strength < original

    def test_decay_prunes_weak_thoughts(self, space):
        t = space.absorb("weak thought")
        t.strength = 0.005  # Below 0.01 threshold
        pruned = space.decay_all(factor=0.5)
        assert pruned == 1
        assert space.count == 0


# ── Clear tests ─────────────────────────────────────────────────────────

class TestClear:

    def test_clear_empties_space(self, space):
        space.absorb("one")
        space.absorb("two")
        space.clear()
        assert space.count == 0
        assert space.recall("one") == []

    def test_clear_allows_re_absorb(self, space):
        space.absorb("hello")
        space.clear()
        result = space.absorb("hello")
        assert result is not None  # Not deduped after clear
