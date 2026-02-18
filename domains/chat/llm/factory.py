"""LLM provider factory — returns the configured provider instance.

The runtime owns LLM selection. In air-gapped mode, returns None
and the chat service falls back to pure Glyphh output.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from domains.chat.llm.base import LLMProvider

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider | None:
    """Create and cache the LLM provider based on runtime settings.

    Returns None if no API key is configured (air-gapped / offline mode).
    """
    from infrastructure.config import get_settings
    settings = get_settings()

    provider_name = getattr(settings, "llm_provider", "openai")

    if provider_name == "openai":
        api_key = getattr(settings, "openai_api_key", None)
        if not api_key:
            logger.info("No OpenAI API key configured — LLM features disabled (pure Glyphh mode)")
            return None
        model = getattr(settings, "openai_model", "gpt-4o-mini")
        from domains.chat.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key, model=model)

    logger.warning(f"Unknown LLM provider '{provider_name}' — LLM features disabled")
    return None
