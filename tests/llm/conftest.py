"""Test fixtures for glyphh.llm tests.

Provides MockLLMEngine — same interface as LLMEngine but returns
canned responses and records all calls for assertions.
"""

from __future__ import annotations

from typing import Any

import pytest

from glyphh.llm.structured import LLMResult


class MockLLMEngine:
    """Mock LLM engine for testing integration points.

    Usage::

        engine = MockLLMEngine(responses={
            "structured_generate": {"action": "delete", "confidence": 0.9},
        })
        result = engine.structured_generate(system="...", user="...", tools=[...])
        assert result.data["action"] == "delete"
        assert len(engine.calls) == 1
    """

    def __init__(self, responses: dict[str, Any] | None = None):
        self._responses = responses or {}
        self.calls: list[dict[str, Any]] = []
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        self.calls.append({
            "method": "generate",
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
        })
        return self._responses.get("generate", "")

    def structured_generate(
        self,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResult:
        self.calls.append({
            "method": "structured_generate",
            "system": system,
            "user": user,
            "tools": tools,
            "max_tokens": max_tokens,
        })
        data = self._responses.get("structured_generate", {})
        return LLMResult(
            data=data,
            raw_text=str(data),
            tokens_used=10,
            latency_ms=1.0,
        )

    def unload(self) -> None:
        self._loaded = False


@pytest.fixture
def mock_engine():
    """Basic mock engine with empty responses."""
    return MockLLMEngine()


@pytest.fixture
def intent_engine():
    """Mock engine that returns intent classification."""
    return MockLLMEngine(responses={
        "structured_generate": {
            "action": "delete",
            "target": "file",
            "domain": "filesystem",
            "confidence": 0.9,
        },
    })


@pytest.fixture
def slot_engine():
    """Mock engine that returns slot extraction."""
    return MockLLMEngine(responses={
        "structured_generate": {
            "arguments": {"file_name": "report.txt"},
        },
    })


@pytest.fixture
def arbitration_engine():
    """Mock engine that accepts the pipeline result."""
    return MockLLMEngine(responses={
        "structured_generate": {
            "accept": True,
            "reason": "Resolution looks correct",
        },
    })
