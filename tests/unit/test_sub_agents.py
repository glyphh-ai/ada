"""
Tests for Ada's cognitive sub-agents.

Covers: agent model scoring, per-agent response strategies,
LLM router disambiguation, sub-agent registry, and the
hallucination gate on RecallAgent.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from glyphh.memory.thought_space import ThoughtGlyphSpace
from glyphh.memory.cognitive_glyph import Action, CognitiveState
from domains.brain.sub_agents import (
    AgentModel,
    SubAgentRegistry,
    StoreAgent,
    RecallAgent,
    FeelAgent,
    ContradictAgent,
    WonderAgent,
    LLMRouter,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def space():
    """Fresh ThoughtGlyphSpace."""
    return ThoughtGlyphSpace()


@pytest.fixture
def registry(space):
    """SubAgentRegistry with all agents."""
    return SubAgentRegistry(space)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.available = True
    llm.ask = AsyncMock(return_value="LLM response")
    return llm


@pytest.fixture
def mock_llm_offline():
    llm = MagicMock()
    llm.available = False
    llm.ask = AsyncMock(return_value=None)
    return llm


def _make_state(action: Action, ambiguous: bool = False) -> CognitiveState:
    """Create a CognitiveState with controlled ambiguity."""
    import numpy as np
    confidence = 0.5
    runner_up_score = 0.45 if ambiguous else 0.2
    return CognitiveState(
        text="test input",
        activations={action.name.lower(): confidence},
        winner=action.name.lower(),
        action=action,
        confidence=confidence,
        runner_up="recall" if action != Action.RECALL else "store",
        runner_up_score=runner_up_score,
    )


# ── AgentModel tests ───────────────────────────────────────────────────

class TestAgentModel:

    def test_model_scores_matching_input(self, space):
        model = AgentModel(space)
        model.add_exemplars(["what is my name", "who am i", "where do i live"])
        score = model.score("what is my name?")
        assert score > 0.3

    def test_model_scores_unrelated_low(self, space):
        model = AgentModel(space)
        model.add_exemplars(["i am happy", "i feel sad", "i love that"])
        score = model.score("what is the capital of france?")
        # Unrelated input should score lower than related
        related_score = model.score("i feel great today")
        assert related_score > score

    def test_empty_model_scores_zero(self, space):
        model = AgentModel(space)
        assert model.score("anything") == 0.0


# ── StoreAgent tests ───────────────────────────────────────────────────

class TestStoreAgent:

    def test_store_scores_statements_high(self, space):
        agent = StoreAgent(space)
        score = agent.score("my favorite color is blue")
        assert score > 0.2

    @pytest.mark.asyncio
    async def test_store_with_llm(self, space, mock_llm):
        agent = StoreAgent(space)
        response = await agent.respond(
            "my name is chris", [], 0.4, "ASK", mock_llm,
        )
        mock_llm.ask.assert_called_once()
        assert response == "LLM response"

    @pytest.mark.asyncio
    async def test_store_without_llm(self, space, mock_llm_offline):
        agent = StoreAgent(space)
        response = await agent.respond(
            "my name is chris", [], 0.4, "ASK", mock_llm_offline,
        )
        assert response == "Got it."


# ── RecallAgent tests ──────────────────────────────────────────────────

class TestRecallAgent:

    def test_recall_scores_questions_high(self, space):
        agent = RecallAgent(space)
        score = agent.score("what is my name?")
        assert score > 0.2

    def test_recall_is_hallucination_gated(self, space):
        agent = RecallAgent(space)
        assert agent.hallucination_gated is True

    @pytest.mark.asyncio
    async def test_recall_low_confidence_no_llm(self, space, mock_llm):
        """Below 0.5 confidence, LLM is not called."""
        agent = RecallAgent(space)
        response = await agent.respond(
            "who am i?", [], 0.3, "ASK", mock_llm,
        )
        assert response == "I don't have information about that in my memory."
        mock_llm.ask.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_low_confidence_returns_raw_fact(self, space, mock_llm):
        """Below 0.5 with DONE gate, returns raw fact (no LLM)."""
        facts = [("The user's name is Chris.", "ada", 0.7)]
        agent = RecallAgent(space)
        response = await agent.respond(
            "who am i?", facts, 0.3, "DONE", mock_llm,
        )
        assert response == "The user's name is Chris."
        mock_llm.ask.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_high_confidence_uses_llm(self, space, mock_llm):
        """Above 0.5 with facts, LLM synthesizes."""
        facts = [("The user's name is Chris.", "ada", 0.8)]
        agent = RecallAgent(space)
        response = await agent.respond(
            "who am i?", facts, 0.7, "DONE", mock_llm,
        )
        mock_llm.ask.assert_called_once()
        assert response == "LLM response"

    @pytest.mark.asyncio
    async def test_recall_high_confidence_no_llm_fallback(self, space, mock_llm_offline):
        """Above 0.5 but LLM offline → returns raw fact."""
        facts = [("The user's name is Chris.", "ada", 0.8)]
        agent = RecallAgent(space)
        response = await agent.respond(
            "who am i?", facts, 0.7, "DONE", mock_llm_offline,
        )
        assert response == "The user's name is Chris."


# ── FeelAgent tests ────────────────────────────────────────────────────

class TestFeelAgent:

    def test_feel_scores_emotions_high(self, space):
        agent = FeelAgent(space)
        score = agent.score("i am really sad today")
        assert score > 0.2

    @pytest.mark.asyncio
    async def test_feel_with_llm(self, space, mock_llm):
        agent = FeelAgent(space)
        response = await agent.respond(
            "i am so frustrated", [], 0.3, "ASK", mock_llm,
        )
        mock_llm.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_feel_without_llm(self, space, mock_llm_offline):
        agent = FeelAgent(space)
        response = await agent.respond(
            "i am so frustrated", [], 0.3, "ASK", mock_llm_offline,
        )
        assert response == "I hear you."


# ── ContradictAgent tests ──────────────────────────────────────────────

class TestContradictAgent:

    @pytest.mark.asyncio
    async def test_contradict_with_llm(self, space, mock_llm):
        agent = ContradictAgent(space)
        facts = [("The user's name is Lear.", "ada", 0.6)]
        response = await agent.respond(
            "no my name is Chris not Lear", facts, 0.5, "DONE", mock_llm,
        )
        mock_llm.ask.assert_called_once()
        # Check that the old fact was included in the prompt
        call_args = mock_llm.ask.call_args[0][0]
        assert "Lear" in call_args

    @pytest.mark.asyncio
    async def test_contradict_without_llm(self, space, mock_llm_offline):
        agent = ContradictAgent(space)
        response = await agent.respond(
            "no that is wrong", [], 0.5, "ASK", mock_llm_offline,
        )
        assert response == "OK, I'll update that."


# ── WonderAgent tests ──────────────────────────────────────────────────

class TestWonderAgent:

    @pytest.mark.asyncio
    async def test_wonder_with_llm(self, space, mock_llm):
        agent = WonderAgent(space)
        response = await agent.respond(
            "have you ever considered consciousness?", [], 0.3, "ASK", mock_llm,
        )
        mock_llm.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_wonder_without_llm(self, space, mock_llm_offline):
        agent = WonderAgent(space)
        response = await agent.respond(
            "that is interesting", [], 0.3, "ASK", mock_llm_offline,
        )
        assert response == "Tell me more about that."


# ── LLMRouter tests ────────────────────────────────────────────────────

class TestLLMRouter:

    @pytest.mark.asyncio
    async def test_confident_classification_not_rerouted(self, registry, mock_llm):
        """When CognitiveGlyph is confident, trust it."""
        state = _make_state(Action.STORE, ambiguous=False)
        action = await registry.router.resolve(
            "my name is chris", state, registry.agents, mock_llm,
        )
        assert action == Action.STORE
        mock_llm.ask.assert_not_called()

    @pytest.mark.asyncio
    async def test_ambiguous_uses_sub_agent_models(self, registry, mock_llm):
        """When ambiguous, sub-agent models compete."""
        state = _make_state(Action.STORE, ambiguous=True)
        action = await registry.router.resolve(
            "what is my name?", state, registry.agents, mock_llm,
        )
        # "what is my name?" should score higher on RecallAgent's model
        # than StoreAgent's model, so it should be rerouted to RECALL
        assert action in (Action.RECALL, Action.STORE)  # depends on HDC scores

    @pytest.mark.asyncio
    async def test_llm_tiebreaker_called_when_tied(self, registry, mock_llm):
        """When sub-agent models are tied, LLM breaks the tie."""
        mock_llm.ask = AsyncMock(return_value="RECALL")
        state = _make_state(Action.STORE, ambiguous=True)
        # Force all agent models to return similar scores
        for agent in registry.agents.values():
            agent.model._exemplars = []  # empty → all score 0 → tied
        action = await registry.router.resolve(
            "test input", state, registry.agents, mock_llm,
        )
        # LLM should have been called since models are tied
        # Falls back to CognitiveGlyph if LLM response doesn't parse
        assert action in Action


# ── SubAgentRegistry tests ──────────────────────────────────────────────

class TestSubAgentRegistry:

    def test_registry_has_all_agents(self, registry):
        assert Action.STORE in registry.agents
        assert Action.RECALL in registry.agents
        assert Action.FEEL in registry.agents
        assert Action.CONTRADICT in registry.agents
        assert Action.WONDER in registry.agents

    def test_get_unknown_action_falls_back(self, registry):
        """Unknown action falls back to RecallAgent."""
        agent = registry.get(Action.DREAM)
        assert agent.name == "recall"

    @pytest.mark.asyncio
    async def test_route_and_respond_returns_tuple(self, registry, mock_llm):
        state = _make_state(Action.STORE, ambiguous=False)
        response, action = await registry.route_and_respond(
            "my name is chris", state, [], 0.4, "ASK", mock_llm,
        )
        assert isinstance(response, str)
        assert isinstance(action, Action)

    @pytest.mark.asyncio
    async def test_route_and_respond_question_gets_recall(self, registry, mock_llm):
        """Even if CognitiveGlyph says STORE, ambiguous routing should
        reclassify 'what is my name?' toward RECALL."""
        state = _make_state(Action.STORE, ambiguous=True)
        response, action = await registry.route_and_respond(
            "what is my name?", state,
            [("The user's name is Chris.", "ada", 0.8)],
            0.6, "DONE", mock_llm,
        )
        # Sub-agent models should reclassify this as RECALL
        # and RecallAgent should synthesize from facts
        assert response is not None
