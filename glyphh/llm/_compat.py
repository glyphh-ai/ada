"""Lazy import guard for llama-cpp-python.

The import happens at model load time, not at module import time.
This allows `from glyphh.llm import LLMEngine` to work even when
llama-cpp-python is not yet installed (e.g. during development).
"""

try:
    from llama_cpp import Llama  # noqa: F401
except ImportError as _e:
    _import_error = _e

    class Llama:  # type: ignore[no-redef]
        """Placeholder that raises on instantiation."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "llama-cpp-python is required for LLM inference. "
                "Install with: pip install llama-cpp-python>=0.3.0"
            ) from _import_error
