"""LLMEngine — local LLM inference via llama-cpp-python.

Core structural component of the Glyphh runtime. Lazy-loads the GGUF
model on first inference call. Thread-safe. Works with any GGUF model;
Qwen3-4B-Instruct Q4_K_M is the default.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .discovery import resolve_model_path
from .structured import LLMResult, parse_tool_call_response

logger = logging.getLogger(__name__)


class LLMEngine:
    """Local LLM inference engine.

    Usage::

        from glyphh.llm import LLMEngine

        engine = LLMEngine()
        engine = LLMEngine(model_path="/path/to/model.gguf")

        # Structured generation (function calling)
        result = engine.structured_generate(
            system="You extract arguments.",
            user="Query: list files in /tmp",
            tools=[...],
        )
        print(result.data)

        # Free-form generation
        text = engine.generate(
            system="You are helpful.",
            user="Explain HDC in one sentence.",
        )
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        model_filename: str = "Qwen3-4B-Q4_K_M.gguf",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        self._model_path_arg = model_path
        self._model_filename = model_filename
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        self._verbose = verbose

        self._model: Any = None  # Llama instance, lazy-loaded
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        return self._loaded

    def _ensure_loaded(self) -> None:
        """Load model on first use. Thread-safe. Lazy-imports llama_cpp."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            from ._compat import Llama

            resolved = resolve_model_path(self._model_path_arg, self._model_filename)
            logger.info("Loading LLM from %s ...", resolved)

            kwargs: dict[str, Any] = {
                "model_path": str(resolved),
                "n_ctx": self._n_ctx,
                "n_gpu_layers": self._n_gpu_layers,
                "verbose": self._verbose,
            }
            if self._n_threads is not None:
                kwargs["n_threads"] = self._n_threads

            self._model = Llama(**kwargs)
            self._loaded = True
            logger.info("LLM loaded: %s", resolved.name)

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Free-form text generation. Returns raw text."""
        self._ensure_loaded()
        assert self._model is not None

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

        content = response["choices"][0]["message"].get("content", "") or ""
        return content.strip()

    def structured_generate(
        self,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Structured generation via function calling format.

        Returns an LLMResult with parsed data from the model's tool call.
        Falls back to JSON extraction from raw text if the model doesn't
        emit a proper tool call.
        """
        self._ensure_loaded()
        assert self._model is not None

        start = time.monotonic()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        response = self._model.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=max_tokens,
            temperature=temperature,
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        choice = response["choices"][0]
        raw_text = choice["message"].get("content", "") or ""

        # Try tool calls first
        tool_calls = choice["message"].get("tool_calls") or []
        if tool_calls:
            if len(tool_calls) == 1:
                # Single tool call — return as data with name + arguments
                tc = tool_calls[0]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = parse_tool_call_response(tc["function"]["arguments"])
                data = {"name": tc["function"]["name"], "arguments": args}
            else:
                # Multiple tool calls — return as list in data
                calls = []
                for tc in tool_calls:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        args = parse_tool_call_response(tc["function"]["arguments"])
                    calls.append({"name": tc["function"]["name"], "arguments": args})
                data = {"tool_calls": calls}
        else:
            data = parse_tool_call_response(raw_text)

        tokens_used = response.get("usage", {}).get("completion_tokens", 0)

        return LLMResult(
            data=data,
            raw_text=raw_text,
            tokens_used=tokens_used,
            latency_ms=elapsed_ms,
        )

    def unload(self) -> None:
        """Explicitly unload the model to free memory."""
        with self._lock:
            self._model = None
            self._loaded = False
            logger.info("LLM unloaded")
