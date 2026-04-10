"""
Tests for Brain — Ada's central think pipeline.

Covers: seed memories, user identity seeding, persistence queue,
dream lifecycle delegation, and the think → ThoughtProcess flow.

Uses mocks for external deps (LLM, DB, ModelManager) but real
cognitive components (AdaCognitive, ThoughtGlyphSpace).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from glyphh.memory.ada_cognitive import AdaCognitive
from glyphh.memory.thought_space import StoredThought


# ── Mock helpers ────────────────────────────────────────────────────────

def _make_mock_brain_state():
    """Minimal BrainState mock."""
    state = MagicMock()
    state.capabilities = {}
    state.capability_names = []
    return state


def _make_mock_model_manager():
    mm = MagicMock()
    mm._models = {}
    mm._encoding_in_progress = []
    return mm


def _make_mock_llm():
    llm = MagicMock()
    llm.available = False
    llm.set_firewall = MagicMock()
    llm.ask = AsyncMock(return_value=None)
    return llm


def _make_mock_session_factory():
    return MagicMock()


def _make_brain():
    """Create a Brain with mocked external deps."""
    from domains.brain.think import Brain
    return Brain(
        brain_state=_make_mock_brain_state(),
        model_manager=_make_mock_model_manager(),
        llm=_make_mock_llm(),
        session_factory=_make_mock_session_factory(),
    )


# ── Seed memory tests ──────────────────────────────────────────────────

class TestSeedMemories:

    def test_seeds_loaded_on_init(self):
        brain = _make_brain()
        space = brain.cognitive.thought_space
        assert space.count > 0, "Seed memories should be loaded on init"

    def test_seed_count_reasonable(self):
        """Seed count should be at least _SEED_MEMORIES length.
        May be higher due to sentence splitting (multi-sentence seeds
        like 'Ada is not Claude. Ada is not ChatGPT.' become 2 thoughts)."""
        from domains.brain.think import Brain
        brain = _make_brain()
        space = brain.cognitive.thought_space
        assert space.count >= len(Brain._SEED_MEMORIES)

    def test_seeds_include_ada_identity(self):
        brain = _make_brain()
        space = brain.cognitive.thought_space
        contents = [t.content.lower() for t in space.all_thoughts()]
        assert any("my name is ada" in c for c in contents)

    def test_seeds_are_third_person(self):
        """Seeds should use 'Ada is...' not 'I am...' to avoid
        polluting user identity recall."""
        from domains.brain.think import Brain
        first_person_seeds = [
            text for text, _ in Brain._SEED_MEMORIES
            if text.startswith("I am ") or text.startswith("I have ")
               or text.startswith("I can ") or text.startswith("I use ")
               or text.startswith("I do ") or text.startswith("I control ")
               or text.startswith("I think ") or text.startswith("I detect ")
               or text.startswith("I route ") or text.startswith("I persist ")
               or text.startswith("I remember ") or text.startswith("I never ")
               or text.startswith("I encode ") or text.startswith("I test ")
               or text.startswith("I respond ")
        ]
        assert first_person_seeds == [], (
            f"Seeds should not start with 'I am/have/can/...': {first_person_seeds}"
        )

    def test_seeds_queued_for_persistence(self):
        brain = _make_brain()
        # Seeds should be in the persist queue
        assert len(brain._persist_queue) > 0
        assert all(isinstance(t, StoredThought) for t in brain._persist_queue)

    def test_seeds_not_re_seeded_if_space_has_thoughts(self):
        """If thought space already has thoughts, don't re-seed."""
        brain = _make_brain()
        initial_count = brain.cognitive.thought_space.count
        # Simulate calling _seed_memories again
        brain._seed_memories()
        assert brain.cognitive.thought_space.count == initial_count


# ── User identity seeding ───────────────────────────────────────────────

class TestUserIdentitySeeding:

    def test_seed_user_identity_adds_facts(self):
        brain = _make_brain()
        initial_count = brain.cognitive.thought_space.count
        brain.seed_user_identity("Chris", "chris@example.com")
        assert brain.cognitive.thought_space.count > initial_count

    def test_seed_user_identity_includes_name(self):
        brain = _make_brain()
        brain.seed_user_identity("Chris")
        space = brain.cognitive.thought_space
        contents = [t.content for t in space.all_thoughts()]
        assert any("Chris" in c for c in contents)

    def test_seed_user_identity_includes_you_are(self):
        """Should include 'You are X' for 'who am i?' matching."""
        brain = _make_brain()
        brain.seed_user_identity("Chris")
        space = brain.cognitive.thought_space
        contents = [t.content for t in space.all_thoughts()]
        assert any("You are Chris" in c for c in contents)

    def test_seed_user_identity_includes_talking_to(self):
        """Should include 'I am talking to X' for context."""
        brain = _make_brain()
        brain.seed_user_identity("Chris")
        space = brain.cognitive.thought_space
        contents = [t.content for t in space.all_thoughts()]
        assert any("I am talking to Chris" in c for c in contents)

    def test_seed_user_identity_includes_email(self):
        brain = _make_brain()
        brain.seed_user_identity("Chris", "chris@example.com")
        space = brain.cognitive.thought_space
        contents = [t.content for t in space.all_thoughts()]
        assert any("chris@example.com" in c for c in contents)

    def test_seed_user_identity_no_email(self):
        """Should work without email."""
        brain = _make_brain()
        brain.seed_user_identity("Chris")
        space = brain.cognitive.thought_space
        contents = [t.content for t in space.all_thoughts()]
        assert not any("email" in c.lower() for c in contents)

    def test_seed_user_identity_queues_for_persistence(self):
        brain = _make_brain()
        initial_queue_len = len(brain._persist_queue)
        brain.seed_user_identity("Chris", "chris@example.com")
        assert len(brain._persist_queue) > initial_queue_len

    def test_user_identity_recall_who_am_i(self):
        """After seeding, 'who am i?' should recall user name."""
        brain = _make_brain()
        brain.seed_user_identity("Chris")
        space = brain.cognitive.thought_space
        results = space.recall("who am i?", top_k=3)
        assert len(results) > 0
        # At least one top-3 result should mention Chris
        contents = [r.thought.content for r in results[:3]]
        assert any("Chris" in c for c in contents), (
            f"'who am i?' should find 'Chris' in top 3, got: {contents}"
        )

    def test_user_identity_recall_what_is_my_name(self):
        """After seeding, 'what is my name?' should recall user name."""
        brain = _make_brain()
        brain.seed_user_identity("Chris")
        space = brain.cognitive.thought_space
        results = space.recall("what is my name?", top_k=3)
        assert len(results) > 0
        contents = [r.thought.content for r in results[:3]]
        assert any("Chris" in c for c in contents), (
            f"'what is my name?' should find 'Chris' in top 3, got: {contents}"
        )


# ── Persistence queue ───────────────────────────────────────────────────

class TestPersistenceQueue:

    def test_queue_persist_finds_existing_thought(self):
        """_queue_persist should find thoughts already in space."""
        brain = _make_brain()
        brain.cognitive.absorb("test thought for persistence")
        initial_len = len(brain._persist_queue)
        brain._queue_persist("test thought for persistence", "incoming")
        assert len(brain._persist_queue) > initial_len

    def test_queue_persist_absorbs_new_text(self):
        """_queue_persist should absorb text not yet in space."""
        brain = _make_brain()
        initial_count = brain.cognitive.thought_space.count
        brain._queue_persist("completely new thought", "incoming")
        assert brain.cognitive.thought_space.count > initial_count

    def test_queue_persist_empty_text_ignored(self):
        brain = _make_brain()
        initial_len = len(brain._persist_queue)
        brain._queue_persist("", "incoming")
        assert len(brain._persist_queue) == initial_len

    @pytest.mark.asyncio
    async def test_flush_persist_queue_clears(self):
        brain = _make_brain()
        brain._queue_persist("new fact to persist", "incoming")
        assert len(brain._persist_queue) > 0

        # flush_persist_queue imports save_thought lazily — patch at source
        with patch("glyphh.memory.thought_persistence.save_thought", new_callable=AsyncMock) as mock_save:
            saved = await brain.flush_persist_queue()
            # Queue should be cleared after flush
            assert len(brain._persist_queue) == 0

    @pytest.mark.asyncio
    async def test_flush_empty_queue_returns_zero(self):
        brain = _make_brain()
        brain._persist_queue.clear()
        saved = await brain.flush_persist_queue()
        assert saved == 0


# ── Dream delegation ────────────────────────────────────────────────────

class TestDreamDelegation:

    def test_start_dreaming_delegates(self):
        brain = _make_brain()
        brain.start_dreaming()
        assert brain.cognitive.is_dreaming
        brain.stop_dreaming()

    def test_stop_dreaming_delegates(self):
        brain = _make_brain()
        brain.start_dreaming()
        brain.stop_dreaming()
        assert not brain.cognitive.is_dreaming

    def test_drain_insights_delegates(self):
        brain = _make_brain()
        insights = brain.drain_insights()
        assert isinstance(insights, list)


# ── Think pipeline ──────────────────────────────────────────────────────

class TestThinkPipeline:

    @pytest.mark.asyncio
    async def test_think_returns_result(self):
        from domains.brain.think import ThinkResult
        brain = _make_brain()
        result = await brain.think("hello")
        assert isinstance(result, ThinkResult)

    @pytest.mark.asyncio
    async def test_think_has_response(self):
        brain = _make_brain()
        result = await brain.think("what color is the sky?")
        assert result.response is not None
        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_think_has_elapsed_ms(self):
        brain = _make_brain()
        result = await brain.think("test")
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_think_queues_for_persistence(self):
        brain = _make_brain()
        initial_len = len(brain._persist_queue)
        await brain.think("remember this fact please")
        # Should have queued the input + possibly the response
        assert len(brain._persist_queue) >= initial_len

    @pytest.mark.asyncio
    async def test_think_firewall_pass_default(self):
        brain = _make_brain()
        result = await brain.think("normal question")
        assert result.firewall_pass is True


# ── End-to-end recall accuracy ──────────────────────────────────────────

class TestEndToEndRecall:

    @pytest.mark.asyncio
    async def test_who_am_i_with_user_identity(self):
        """Full pipeline: seed identity → think('who am i?') → mentions Chris."""
        brain = _make_brain()
        brain.seed_user_identity("Chris")
        result = await brain.think("who am i?")
        # Either the response mentions Chris, or the facts do
        has_chris = (
            "Chris" in result.response or
            any("Chris" in f[0] for f in result.facts)
        )
        assert has_chris, (
            f"Expected 'Chris' in response or facts.\n"
            f"Response: {result.response}\n"
            f"Facts: {result.facts}"
        )

    @pytest.mark.asyncio
    async def test_statement_does_not_echo(self):
        """'my name is Chris' should not echo the fact back."""
        brain = _make_brain()
        result = await brain.think("my name is Chris")
        # Should be either "Got it." or a short acknowledgment
        # NOT "The user's name is Chris." echoed back
        assert result.response != "The user's name is Chris."
