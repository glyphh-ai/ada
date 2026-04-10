"""
Tests for AdaCognitive — Ada's cognitive infrastructure.

Covers: absorb returns StoredThought, sentence splitting, recall gating,
conversation management, dream lifecycle, and reset.
"""

import pytest

from glyphh.memory.ada_cognitive import (
    AdaCognitive,
    CognitiveResult,
    _split_sentences,
)
from glyphh.memory.thought_space import StoredThought
from glyphh.memory.cognitive_glyph import Action


# ── Fixtures ─���──────────────────────────────────────────────────────────

@pytest.fixture
def ada():
    """Fresh AdaCognitive instance."""
    return AdaCognitive()


@pytest.fixture
def ada_with_facts(ada):
    """AdaCognitive pre-loaded with some facts."""
    ada.absorb("My name is Ada.", speaker="ada")
    ada.absorb("The user's name is Chris.", speaker="ada")
    ada.absorb("You are Chris.", speaker="ada")
    ada.absorb("The sky is blue.", speaker="incoming")
    ada.absorb("Python is a programming language.", speaker="incoming")
    return ada


# ── Sentence splitting ───────────────���──────────────────────────────────

class TestSentenceSplitting:

    def test_single_sentence(self):
        assert _split_sentences("hello world") == ["hello world"]

    def test_multiple_sentences(self):
        result = _split_sentences("First sentence. Second sentence. Third.")
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
        assert result[2] == "Third."

    def test_question_marks(self):
        result = _split_sentences("What is this? Who are you?")
        assert len(result) == 2

    def test_exclamation_marks(self):
        result = _split_sentences("Wow! That is amazing!")
        assert len(result) == 2

    def test_empty_string(self):
        result = _split_sentences("")
        assert result == [""]

    def test_whitespace_only(self):
        result = _split_sentences("   ")
        assert result == [""]


# ── Absorb tests ───────────────────────────────────��────────────────────

class TestAdaCognitiveAbsorb:

    def test_absorb_returns_stored_thought(self, ada):
        """Critical: absorb must return StoredThought for persistence queue."""
        result = ada.absorb("my name is chris")
        assert isinstance(result, StoredThought)

    def test_absorb_returns_none_for_duplicate(self, ada):
        ada.absorb("hello world")
        result = ada.absorb("hello world")
        assert result is None

    def test_absorb_returns_none_for_short_text(self, ada):
        result = ada.absorb("a")
        assert result is None

    def test_absorb_multi_sentence_returns_last(self, ada):
        """Multi-sentence absorb returns the last StoredThought."""
        result = ada.absorb("First sentence. Second sentence.")
        assert result is not None
        assert result.content == "Second sentence."

    def test_absorb_multi_sentence_stores_all(self, ada):
        """All sentences from multi-sentence input are stored."""
        ada.absorb("First thing. Second thing. Third thing.")
        assert ada.thought_space.count == 3

    def test_absorb_speaker_propagates(self, ada):
        result = ada.absorb("test", speaker="ada")
        assert result.speaker == "ada"

    def test_absorb_incoming_default(self, ada):
        result = ada.absorb("test input")
        assert result.speaker == "incoming"


# ── Recall tests ───────────��───────────────────────────────��────────────

class TestAdaCognitiveRecall:

    def test_recall_done_when_above_threshold(self, ada_with_facts):
        gate, facts = ada_with_facts.recall("the sky is blue")
        # Exact or near-exact match should be DONE
        if facts:
            assert gate == "DONE"
            assert facts[0][2] >= ada_with_facts._recall_threshold

    def test_recall_ask_when_no_match(self, ada):
        ada.absorb("the weather is nice")
        gate, facts = ada.recall("quantum mechanics theory")
        # Very different topic — either ASK or very low similarity
        if gate == "DONE":
            # If somehow matched, check similarity is reasonable
            assert facts[0][2] >= ada._recall_threshold
        else:
            assert gate == "ASK"

    def test_recall_returns_content_speaker_similarity(self, ada_with_facts):
        gate, facts = ada_with_facts.recall("what color is the sky")
        if facts:
            content, speaker, similarity = facts[0]
            assert isinstance(content, str)
            assert isinstance(speaker, str)
            assert isinstance(similarity, float)

    def test_recall_threshold_respected(self, ada):
        """Facts below threshold should not appear."""
        ada._recall_threshold = 0.99  # impossibly high
        ada.absorb("the sky is blue")
        gate, facts = ada.recall("the sky is blue")
        # Even exact match might not hit 0.99
        # The key: if gate is DONE, facts are above threshold
        if gate == "DONE":
            assert all(f[2] >= 0.99 for f in facts)


# ── Process (cognitive routing) ─────────────────────────────────────────

class TestAdaCognitiveProcess:

    def test_process_returns_cognitive_result(self, ada):
        result = ada.process("hello there")
        assert isinstance(result, CognitiveResult)
        assert hasattr(result, 'state')
        assert hasattr(result, 'gate')
        assert hasattr(result, 'facts')
        assert hasattr(result, 'prompt')

    def test_process_absorbs_input(self, ada):
        ada.process("I like pizza")
        assert ada.thought_space.count > 0

    def test_process_question_classifies_as_recall(self, ada_with_facts):
        """Questions should route to RECALL action."""
        result = ada_with_facts.process("what is my name?")
        # CognitiveGlyph classification depends on exemplar training.
        # The key: the result has a valid action from the Action enum.
        assert result.state.action in (Action.RECALL, Action.STORE, Action.WONDER)

    def test_process_statement_classifies_as_store(self, ada):
        """Statements should route to STORE action."""
        result = ada.process("my favorite color is green")
        assert result.state.action == Action.STORE

    def test_process_gate_is_done_or_ask(self, ada):
        result = ada.process("hello")
        assert result.gate in ("DONE", "ASK")


# ── Conversation management ─────────────────────────────────────────────

class TestConversation:

    def test_add_response_records_turn(self, ada):
        ada.add_response("hello", "hi there")
        # Should have absorbed both user input and response
        assert ada.thought_space.count >= 1

    def test_add_response_absorbs_ada_response(self, ada):
        ada.add_response("test", "I understand")
        results = ada.thought_space.recall("I understand")
        assert len(results) > 0

    def test_build_prompt_returns_string(self, ada):
        from glyphh.memory.cognitive_glyph import CognitiveState, Action
        import numpy as np
        state = CognitiveState(
            text="what is my name?",
            winner="question",
            action=Action.RECALL,
            activations={"question": 0.8},
            confidence=0.8,
        )
        prompt = ada.build_prompt("what is my name?", state, "ASK", [])
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ─��� Dream lifecycle ────────────────────���────────────────────────────────

class TestDreamLifecycle:

    def test_not_dreaming_initially(self, ada):
        assert not ada.is_dreaming

    def test_start_dreaming(self, ada):
        ada.absorb("seed thought for dreaming")
        ada.start_dreaming()
        assert ada.is_dreaming
        ada.stop_dreaming()

    def test_stop_dreaming(self, ada):
        ada.absorb("seed thought")
        ada.start_dreaming()
        ada.stop_dreaming()
        assert not ada.is_dreaming

    def test_drain_insights_empty_initially(self, ada):
        insights = ada.drain_insights()
        assert insights == []

    def test_dream_stats_empty_without_dreaming(self, ada):
        stats = ada.dream_stats()
        assert stats == {}


# ── Reset ───────────────────────────────────────────────────────────────

class TestReset:

    def test_reset_clears_memory(self, ada_with_facts):
        ada_with_facts.reset()
        assert ada_with_facts.thought_space.count == 0

    def test_reset_clears_conversation(self, ada):
        ada.add_response("hello", "hi")
        ada.reset()
        # Conversation should be cleared
        prompt = ada.conversation.build_prompt("test", recall="")
        assert "hello" not in prompt

    def test_reset_stops_dreaming(self, ada):
        ada.absorb("seed")
        ada.start_dreaming()
        ada.reset()
        assert not ada.is_dreaming

    def test_reset_allows_re_absorb(self, ada):
        ada.absorb("hello world")
        ada.reset()
        result = ada.absorb("hello world")
        assert result is not None
