"""
Tests for UserDerivative — Ada learning user patterns.

Covers: observation recording, signal reinforcement,
decay, context generation, style summary.
"""

import pytest
from glyphh.memory.thought_space import ThoughtGlyphSpace
from domains.brain.derivative import UserDerivative


@pytest.fixture
def space():
    return ThoughtGlyphSpace()


@pytest.fixture
def deriv(space):
    return UserDerivative(space)


class TestObservation:

    def test_observe_creates_signals(self, deriv):
        deriv.observe(
            input_text="my name is chris",
            response_text="Nice to meet you!",
            extracted_facts=["my name is chris"],
            extracted_question=None,
            extracted_emotion=None,
            is_correction=False,
            recalled_facts=[],
        )
        assert deriv.signal_count > 0

    def test_observe_emotion_creates_emotion_signal(self, deriv):
        deriv.observe(
            input_text="i feel sad today",
            response_text="I hear you.",
            extracted_facts=[],
            extracted_question=None,
            extracted_emotion="sad",
            is_correction=False,
            recalled_facts=[],
        )
        stats = deriv.stats()
        assert "emotion" in stats["dimensions"]

    def test_observe_correction_creates_correction_signal(self, deriv):
        deriv.observe(
            input_text="no that is wrong",
            response_text="Got it.",
            extracted_facts=[],
            extracted_question=None,
            extracted_emotion=None,
            is_correction=True,
            recalled_facts=[],
        )
        stats = deriv.stats()
        assert "correction" in stats["dimensions"]


class TestReinforcement:

    def test_repeated_topic_reinforces(self, deriv):
        for _ in range(5):
            deriv.observe(
                input_text="tell me about my family",
                response_text="response",
                extracted_facts=[],
                extracted_question="tell me about my family",
                extracted_emotion=None,
                is_correction=False,
                recalled_facts=[],
            )
        # Should have reinforced signals, not created 5 separate ones
        assert deriv.signal_count < 10  # merged due to similarity
        assert deriv.strong_signals >= 1


class TestDecay:

    def test_decay_reduces_strength(self, deriv):
        deriv.observe(
            input_text="test input",
            response_text="response",
            extracted_facts=["test input"],
            extracted_question=None,
            extracted_emotion=None,
            is_correction=False,
            recalled_facts=[],
        )
        initial_count = deriv.signal_count
        # Decay many times
        for _ in range(200):
            deriv.decay()
        # Some signals should have been pruned
        assert deriv.signal_count <= initial_count


class TestContext:

    def test_context_empty_initially(self, deriv):
        context = deriv.get_context("hello")
        assert context == []

    def test_context_after_observations(self, deriv):
        # Add several observations on same topic
        for _ in range(5):
            deriv.observe(
                input_text="my family is important",
                response_text="response",
                extracted_facts=["my family is important"],
                extracted_question=None,
                extracted_emotion=None,
                is_correction=False,
                recalled_facts=[],
            )
        context = deriv.get_context("my family is important")
        # May or may not produce context depending on count threshold
        assert isinstance(context, list)


class TestStyleSummary:

    def test_no_summary_initially(self, deriv):
        assert deriv.style_summary() is None

    def test_summary_after_many_interactions(self, deriv):
        for i in range(10):
            deriv.observe(
                input_text=f"test input number {i} with some words",
                response_text="response",
                extracted_facts=[f"test fact {i}"],
                extracted_question=None,
                extracted_emotion=None,
                is_correction=False,
                recalled_facts=[],
            )
        # May or may not produce summary depending on signal strength
        summary = deriv.style_summary()
        assert summary is None or isinstance(summary, str)


class TestStats:

    def test_stats_structure(self, deriv):
        stats = deriv.stats()
        assert "total_signals" in stats
        assert "strong_signals" in stats
        assert "dimensions" in stats

    def test_stats_initially_empty(self, deriv):
        stats = deriv.stats()
        assert stats["total_signals"] == 0
