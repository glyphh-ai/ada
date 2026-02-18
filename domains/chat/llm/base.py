"""Base LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMMessage:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Generate a completion from a list of messages."""
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable provider name."""
        ...
