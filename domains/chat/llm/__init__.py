"""LLM provider abstraction layer for the runtime.

Pluggable interface for language model providers.
The runtime decides whether to use a local LLM or an external one.
"""

from domains.chat.llm.base import LLMProvider, LLMMessage, LLMResponse
from domains.chat.llm.factory import get_llm_provider

__all__ = ["LLMProvider", "LLMMessage", "LLMResponse", "get_llm_provider"]
