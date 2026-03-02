"""Tests for SchemaIntentClassifier — LLM-primary intent classification."""

import pytest

from glyphh.cognitive.schema_classifier import SchemaIntentClassifier
from glyphh.llm.structured import LLMResult

# Reuse test domain fixtures
from .conftest import FUNC_SCHEMAS, DOMAIN_DICT


# ── Mock LLM Engine (inline, avoids cross-test-package import) ──

class MockLLMEngine:
    """Same interface as LLMEngine, returns canned responses."""

    def __init__(self, responses=None):
        self._responses = responses or {}
        self.calls = []
        self._loaded = True

    @property
    def is_loaded(self):
        return self._loaded

    def generate(self, system, user, max_tokens=256, temperature=0.0, stop=None):
        self.calls.append({"method": "generate", "system": system, "user": user})
        return self._responses.get("generate", "")

    def structured_generate(self, system, user, tools, max_tokens=256, temperature=0.0):
        self.calls.append({
            "method": "structured_generate",
            "system": system,
            "user": user,
            "tools": tools,
        })
        data = self._responses.get("structured_generate", {})
        return LLMResult(data=data, raw_text=str(data), tokens_used=10, latency_ms=5.0)

    def unload(self):
        self._loaded = False


ACTION_TO_FUNC = DOMAIN_DICT["action_to_func"]


class TestSchemaClassifierConfigure:
    """Tests for configure()."""

    def test_configure_sets_system_prompt(self):
        engine = MockLLMEngine()
        classifier = SchemaIntentClassifier(engine, dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        assert classifier._configured
        assert "navigate" in classifier._system_prompt
        assert "display" in classifier._system_prompt
        assert "lookup" in classifier._system_prompt

    def test_configure_builds_tool_schema(self):
        engine = MockLLMEngine()
        classifier = SchemaIntentClassifier(engine, dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        assert len(classifier._classify_tools) == 1
        tool = classifier._classify_tools[0]
        assert tool["function"]["name"] == "classify_and_call"
        # Check enum constraint contains function names
        enum = tool["function"]["parameters"]["properties"]["functions"]["items"]["enum"]
        assert "navigate" in enum
        assert "display" in enum
        assert "lookup" in enum

    def test_configure_clears_cache(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": ["display"],
                "arguments": {},
                "confidence": 0.9,
            },
        })
        classifier = SchemaIntentClassifier(engine, dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        # Add something to cache
        classifier.classify("show report", {"primary": "root"}, [])
        assert classifier.cache_size > 0

        # Re-configure clears cache
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)
        assert classifier.cache_size == 0


class TestSchemaClassifierClassify:
    """Tests for classify()."""

    def test_unconfigured_returns_empty(self):
        engine = MockLLMEngine()
        classifier = SchemaIntentClassifier(engine, dimension=1000)

        result = classifier.classify("show report", {"primary": "root"}, [])
        assert result["functions"] == []
        assert result["confidence"] == 0.0
        assert result["source"] == "unconfigured"

    def test_classify_calls_llm(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": ["display"],
                "arguments": {"display": {"item_name": "report.txt"}},
                "confidence": 0.9,
            },
        })
        classifier = SchemaIntentClassifier(engine, dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        result = classifier.classify(
            "show the report",
            {"primary": "root.workspace", "collections": {"items_here": ["report.txt"]}},
            [],
        )

        assert result["functions"] == ["display"]
        assert result["arguments"]["display"]["item_name"] == "report.txt"
        assert result["confidence"] == 0.9
        assert result["source"] == "llm"
        assert len(engine.calls) == 1

    def test_classify_validates_function_names(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": ["display", "nonexistent_func"],
                "arguments": {},
                "confidence": 0.8,
            },
        })
        classifier = SchemaIntentClassifier(engine, dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        result = classifier.classify("show report", {"primary": "root"}, [])
        # "nonexistent_func" should be filtered out
        assert result["functions"] == ["display"]

    def test_classify_caches_result(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": ["display"],
                "arguments": {},
                "confidence": 0.9,
            },
        })
        classifier = SchemaIntentClassifier(engine, dimension=1000, cache_threshold=0.5)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        # First call hits LLM
        result1 = classifier.classify("show the report", {"primary": "root"}, [])
        assert result1["source"] == "llm"
        assert len(engine.calls) == 1

        # Second identical call hits cache
        result2 = classifier.classify("show the report", {"primary": "root"}, [])
        assert result2["source"] == "hdc_cache"
        assert len(engine.calls) == 1  # No new LLM call

    def test_classify_similar_query_hits_cache(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": ["display"],
                "arguments": {},
                "confidence": 0.85,
            },
        })
        classifier = SchemaIntentClassifier(engine, dimension=1000, cache_threshold=0.5)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        # First call
        classifier.classify("show the report", {"primary": "root"}, [])
        assert len(engine.calls) == 1

        # Similar query
        result = classifier.classify("show report", {"primary": "root"}, [])
        assert result["source"] == "hdc_cache"
        assert len(engine.calls) == 1

    def test_low_confidence_not_cached(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": [],
                "arguments": {},
                "confidence": 0.1,
            },
        })
        classifier = SchemaIntentClassifier(engine, dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        classifier.classify("xyzzy plugh", {"primary": "root"}, [])
        assert classifier.cache_size == 0

    def test_state_summary_in_prompt(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": ["display"],
                "arguments": {},
                "confidence": 0.9,
            },
        })
        classifier = SchemaIntentClassifier(engine, dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        classifier.classify(
            "show the report",
            {
                "primary": "root.workspace",
                "collections": {"items_here": ["report.txt", "budget.csv"]},
            },
            ["navigate"],
        )

        call = engine.calls[0]
        assert "root.workspace" in call["user"]
        assert "navigate" in call["user"]
        assert "show the report" in call["user"]


class TestSchemaClassifierConfirm:
    """Tests for confirm() — Hebbian reinforcement."""

    def test_confirm_correct_strengthens(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": ["display"],
                "arguments": {},
                "confidence": 0.9,
            },
        })
        classifier = SchemaIntentClassifier(engine, dimension=1000, cache_threshold=0.5)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        # Classify to populate cache
        classifier.classify("show the report", {"primary": "root"}, [])
        # Trigger cache hit
        classifier.classify("show the report", {"primary": "root"}, [])

        initial_strength = classifier._cache._entries[0].strength
        classifier.confirm(correct=True)
        assert classifier._cache._entries[0].strength > initial_strength

    def test_confirm_incorrect_weakens(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": ["display"],
                "arguments": {},
                "confidence": 0.9,
            },
        })
        classifier = SchemaIntentClassifier(engine, dimension=1000, cache_threshold=0.5)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        classifier.classify("show the report", {"primary": "root"}, [])
        classifier.classify("show the report", {"primary": "root"}, [])

        initial_strength = classifier._cache._entries[0].strength
        classifier.confirm(correct=False)
        assert classifier._cache._entries[0].strength < initial_strength


class TestSchemaClassifierErrorHandling:
    """Tests for graceful LLM failure handling."""

    def test_llm_exception_returns_empty(self):
        class FailingEngine:
            is_loaded = True
            def structured_generate(self, **kwargs):
                raise RuntimeError("Model not loaded")

        classifier = SchemaIntentClassifier(FailingEngine(), dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        result = classifier.classify("show report", {"primary": "root"}, [])
        assert result["functions"] == []
        assert result["confidence"] == 0.0
        assert "error" in result

    def test_invalid_function_in_response(self):
        engine = MockLLMEngine(responses={
            "structured_generate": {
                "functions": ["totally_fake"],
                "arguments": {},
                "confidence": 0.95,
            },
        })
        classifier = SchemaIntentClassifier(engine, dimension=1000)
        classifier.configure(FUNC_SCHEMAS, ACTION_TO_FUNC)

        result = classifier.classify("do something", {"primary": "root"}, [])
        # No valid functions → confidence forced to 0.0
        assert result["functions"] == []
        assert result["confidence"] == 0.0
