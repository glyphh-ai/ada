"""Integration tests: CognitiveLoop + MockLLMEngine.

Tests that the SchemaIntentClassifier is invoked as the primary
intent classifier when an LLM engine is provided, and that results
are properly merged into the pipeline.
"""

import pytest

from glyphh.cognitive.loop import CognitiveLoop, StepResult
from glyphh.cognitive.domain import DomainConfig

# Reuse the test domain from cognitive tests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "cognitive"))
from conftest import DOMAIN_DICT, FUNC_SCHEMAS, make_state

from .conftest import MockLLMEngine


DIM = 1000


@pytest.fixture
def config():
    return DomainConfig.from_dict(DOMAIN_DICT)


@pytest.fixture
def loop_without_llm(config):
    """CognitiveLoop with no LLM engine."""
    loop = CognitiveLoop(
        packs=[],
        domain_config=config,
        dimension=DIM,
        confidence_threshold=0.25,
    )
    state = make_state(
        primary="root.workspace",
        items=["report.txt", "budget.csv"],
        locations=["archive", "temp"],
    )
    loop.begin(functions=FUNC_SCHEMAS, initial_state=state)
    return loop


@pytest.fixture
def loop_with_llm(config):
    """CognitiveLoop with a MockLLMEngine that returns display classification."""
    engine = MockLLMEngine(responses={
        "structured_generate": {
            "functions": ["display"],
            "arguments": {"display": {"item_name": "report.txt"}},
            "confidence": 0.9,
        },
    })
    loop = CognitiveLoop(
        packs=[],
        domain_config=config,
        dimension=DIM,
        confidence_threshold=0.25,
        llm_engine=engine,
    )
    state = make_state(
        primary="root.workspace",
        items=["report.txt", "budget.csv"],
        locations=["archive", "temp"],
    )
    loop.begin(functions=FUNC_SCHEMAS, initial_state=state)
    return loop, engine


class TestLoopWithoutLLM:
    """CognitiveLoop with llm_engine=None behaves identically to baseline."""

    def test_step_returns_result(self, loop_without_llm):
        result = loop_without_llm.step("show the content of report.txt")
        assert isinstance(result, StepResult)

    def test_no_classification_signals(self, loop_without_llm):
        result = loop_without_llm.step("show the content of report.txt")
        assert "classification" not in result.signals
        assert "classification_source" not in result.signals

    def test_keyword_extraction_without_llm(self, loop_without_llm):
        result = loop_without_llm.step("show the content of report.txt")
        assert "intent" in result.signals
        # Without LLM, only keyword extraction runs (no IntentExtractor)
        assert result.signals["intent"].get("action") == ""
        assert "report.txt" in result.signals["intent"].get("keywords", "")

    def test_ask_on_unknown_query(self, loop_without_llm):
        result = loop_without_llm.step("xyzzy plugh")
        assert result.action == "ASK"

    def test_no_classifier_created(self, loop_without_llm):
        assert loop_without_llm._classifier is None


class TestLoopWithLLM:
    """CognitiveLoop with MockLLMEngine uses SchemaIntentClassifier."""

    def test_classifier_created(self, loop_with_llm):
        loop, engine = loop_with_llm
        assert loop._classifier is not None
        assert loop._llm is engine

    def test_step_returns_call(self, loop_with_llm):
        loop, engine = loop_with_llm
        result = loop.step("show the content of report.txt")
        assert isinstance(result, StepResult)
        assert result.action == "CALL"

    def test_classification_in_signals(self, loop_with_llm):
        loop, engine = loop_with_llm
        result = loop.step("show the content of report.txt")
        assert "classification" in result.signals
        assert result.signals["classification"]["functions"] == ["display"]

    def test_classification_source_is_llm(self, loop_with_llm):
        loop, engine = loop_with_llm
        result = loop.step("show the content of report.txt")
        assert result.signals["classification_source"] == "llm"

    def test_llm_resolves_functions_directly(self, loop_with_llm):
        loop, engine = loop_with_llm
        result = loop.step("show report.txt")
        assert result.signals.get("resolve_source") == "llm_direct"
        assert "display" in result.signals["resolved_functions"]

    def test_llm_arguments_merged_into_slots(self, loop_with_llm):
        loop, engine = loop_with_llm
        result = loop.step("show the content of report.txt")
        # LLM provided item_name="report.txt"
        assert result.calls[0].get("display", {}).get("item_name") == "report.txt"

    def test_confidence_blended_with_llm(self, loop_with_llm):
        loop, engine = loop_with_llm
        result = loop.step("show report.txt")
        # LLM confidence is 0.9, blended with HDC confidence
        # Should be significantly above threshold
        assert result.confidence > 0.5

    def test_cache_hit_on_repeat_query(self, loop_with_llm):
        loop, engine = loop_with_llm
        # First call → LLM
        result1 = loop.step("show report.txt")
        assert result1.signals["classification_source"] == "llm"
        llm_calls_after_first = len(engine.calls)

        # Second identical call → should hit HDC cache
        result2 = loop.step("show report.txt")
        assert result2.signals["classification_source"] == "hdc_cache"
        # No new LLM calls
        assert len(engine.calls) == llm_calls_after_first

    def test_llm_called_for_unknown_query(self, config):
        """Unknown query still goes through LLM classification."""
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": ["delete"],
                "arguments": {"delete": {"item_name": "report.txt"}},
                "confidence": 0.85,
            },
        })
        loop = CognitiveLoop(
            packs=[],
            domain_config=config,
            dimension=DIM,
            confidence_threshold=0.25,
            llm_engine=engine,
        )
        state = make_state(
            primary="root.workspace",
            items=["report.txt"],
            locations=["archive"],
        )
        loop.begin(functions=FUNC_SCHEMAS, initial_state=state)

        result = loop.step("xyzzy the report.txt")
        assert isinstance(result, StepResult)
        # LLM was called
        assert len(engine.calls) > 0

    def test_low_confidence_falls_through(self, config):
        """When LLM returns low confidence, falls to rule-based resolve."""
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": [],
                "arguments": {},
                "confidence": 0.1,
            },
        })
        loop = CognitiveLoop(
            packs=[],
            domain_config=config,
            dimension=DIM,
            confidence_threshold=0.25,
            llm_engine=engine,
        )
        state = make_state(
            primary="root.workspace",
            items=["report.txt"],
            locations=["archive"],
        )
        loop.begin(functions=FUNC_SCHEMAS, initial_state=state)

        result = loop.step("show the content of report.txt")
        # Low LLM confidence → falls to rule-based path
        assert result.signals.get("resolve_source") == "rule_based"

    def test_confirm_reinforces_cache(self, loop_with_llm):
        loop, engine = loop_with_llm
        loop.step("show report.txt")
        # Should not raise
        loop.confirm(was_correct=True)


class TestMockEngineRecordsCalls:
    """Verify MockLLMEngine records calls for assertions."""

    def test_generate_records(self):
        engine = MockLLMEngine(responses={"generate": "hello"})
        result = engine.generate(system="sys", user="usr")
        assert result == "hello"
        assert len(engine.calls) == 1
        assert engine.calls[0]["method"] == "generate"
        assert engine.calls[0]["system"] == "sys"

    def test_structured_generate_records(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {"action": "test"},
        })
        result = engine.structured_generate(
            system="sys", user="usr", tools=[],
        )
        assert result.data["action"] == "test"
        assert result.latency_ms == 1.0
        assert len(engine.calls) == 1
        assert engine.calls[0]["method"] == "structured_generate"

    def test_unload(self):
        engine = MockLLMEngine()
        assert engine.is_loaded
        engine.unload()
        assert not engine.is_loaded
