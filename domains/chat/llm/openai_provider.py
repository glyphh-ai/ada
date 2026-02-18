"""OpenAI LLM provider implementation."""

from __future__ import annotations

import logging
from typing import Optional

from domains.chat.llm.base import LLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False


class OpenAIProvider(LLMProvider):
    """OpenAI-backed LLM provider (gpt-4o-mini by default)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. pip install openai")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason or "",
        )

    def provider_name(self) -> str:
        return f"openai/{self._model}"
