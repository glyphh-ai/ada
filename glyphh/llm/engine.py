"""
LLMEngine — unified local inference across MLX, llama-cpp, and API backends.

Auto-detects the best available backend and model path.  Lazy-loads on
first generate() or stream() call so import is free.

Backend priority:
  1. MLX        (Apple Silicon — fastest, native Metal)
  2. llama-cpp  (universal — CPU / CUDA / Metal via GGUF)
  3. API        (external OpenAI-compatible endpoint)

Model: Qwen3-1.5B (Q4 quantised).  ~1.1 GB on disk.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ── Default model identifiers (kept in sync with cli/commands/setup.py) ─────

MLX_MODEL_REPO = "mlx-community/Qwen3.5-2B-4bit"
MLX_MODEL_DIRNAME = "Qwen3.5-2B-4bit"

GGUF_MODEL_REPO = "Qwen/Qwen3-1.7B-GGUF"
GGUF_MODEL_FILENAME = "Qwen3-1.7B-Q4_K_M.gguf"


def _xdg_data_home() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg)
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".local" / "share"


def _model_dir() -> Path:
    return _xdg_data_home() / "glyphh" / "models"


# ── Generation defaults ─────────────────────────────────────────────────────

@dataclass
class GenerateConfig:
    """Controls for a single generation call."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop: list[str] = field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])


# ── Backend ABCs ────────────────────────────────────────────────────────────

class _Backend:
    """Abstract backend interface."""

    name: str = "base"

    def load(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str, config: GenerateConfig) -> str:
        raise NotImplementedError

    def stream(self, prompt: str, config: GenerateConfig) -> Iterator[str]:
        raise NotImplementedError

    def unload(self) -> None:
        pass


# ── MLX backend ─────────────────────────────────────────────────────────────

class _MLXBackend(_Backend):
    name = "mlx"

    def __init__(self, model_path: str | Path):
        self._model_path = str(model_path)
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        from mlx_lm import load  # type: ignore[import-untyped]

        logger.info("Loading MLX model from %s", self._model_path)
        self._model, self._tokenizer = load(self._model_path)
        logger.info("MLX model loaded")

    def _make_sampler(self, config: GenerateConfig):
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-untyped]
        return make_sampler(temp=config.temperature, top_p=config.top_p)

    def generate(self, prompt: str, config: GenerateConfig) -> str:
        from mlx_lm import generate as mlx_generate  # type: ignore[import-untyped]

        return mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=self._make_sampler(config),
        )

    def stream(self, prompt: str, config: GenerateConfig) -> Iterator[str]:
        from mlx_lm import stream_generate  # type: ignore[import-untyped]

        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=self._make_sampler(config),
        ):
            yield response.text

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None


# ── llama-cpp backend ───────────────────────────────────────────────────────

class _LlamaCppBackend(_Backend):
    name = "llama-cpp"

    def __init__(self, model_path: str | Path):
        self._model_path = str(model_path)
        self._llm = None

    def load(self) -> None:
        from llama_cpp import Llama  # type: ignore[import-untyped]

        logger.info("Loading GGUF model from %s", self._model_path)
        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
        logger.info("GGUF model loaded")

    def generate(self, prompt: str, config: GenerateConfig) -> str:
        result = self._llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repeat_penalty=config.repetition_penalty,
            stop=config.stop,
        )
        return result["choices"][0]["text"]

    def stream(self, prompt: str, config: GenerateConfig) -> Iterator[str]:
        for chunk in self._llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repeat_penalty=config.repetition_penalty,
            stop=config.stop,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    def unload(self) -> None:
        self._llm = None


# ── API backend ─────────────────────────────────────────────────────────────

class _APIBackend(_Backend):
    name = "api"

    def __init__(self, base_url: str, api_key: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def load(self) -> None:
        logger.info("API backend ready: %s (model: %s)", self._base_url, self._model)

    def generate(self, prompt: str, config: GenerateConfig) -> str:
        import httpx

        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def stream(self, prompt: str, config: GenerateConfig) -> Iterator[str]:
        import httpx
        import json

        with httpx.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "stream": True,
            },
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield token


# ── Auto-detection ──────────────────────────────────────────────────────────

def _detect_backend() -> _Backend:
    """Find the best available backend + model.

    Priority:
      1. GLYPHH_MODEL_PATH env override → infer backend from file type
      2. MLX on Apple Silicon
      3. GGUF via llama-cpp
      4. API via env vars
    """
    # Env override — explicit path
    env_path = os.environ.get("GLYPHH_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_dir() and (p / "config.json").exists():
            return _MLXBackend(p)
        if p.is_file() and p.suffix == ".gguf":
            return _LlamaCppBackend(p)
        raise FileNotFoundError(
            f"GLYPHH_MODEL_PATH={env_path} is not a valid MLX dir or GGUF file"
        )

    # MLX on Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        mlx_path = _model_dir() / MLX_MODEL_DIRNAME
        if mlx_path.is_dir() and (mlx_path / "config.json").exists():
            try:
                import mlx_lm  # noqa: F401
                return _MLXBackend(mlx_path)
            except ImportError:
                logger.debug("MLX model found but mlx-lm not installed")

    # GGUF via llama-cpp
    gguf_path = _model_dir() / GGUF_MODEL_FILENAME
    if gguf_path.is_file():
        try:
            import llama_cpp  # noqa: F401
            return _LlamaCppBackend(gguf_path)
        except ImportError:
            logger.debug("GGUF model found but llama-cpp-python not installed")

    # API fallback
    base_url = os.environ.get("GLYPHH_API_BASE") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("GLYPHH_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if base_url and api_key:
        model = os.environ.get("GLYPHH_API_MODEL", "qwen3-1.5b")
        return _APIBackend(base_url, api_key, model)

    raise RuntimeError(
        "No LLM backend available. Run `glyphh setup` to install one, "
        "or set GLYPHH_MODEL_PATH to a local model."
    )


# ── Public engine ───────────────────────────────────────────────────────────

class LLMEngine:
    """Unified LLM inference engine with lazy loading.

    Usage:
        engine = LLMEngine()
        text = engine.generate("Hello, Ada.")

        for token in engine.stream("Hello, Ada."):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        backend: _Backend | None = None,
        system_prompt: str | None = None,
    ):
        self._backend = backend
        self._loaded = False
        self._system_prompt = system_prompt

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._backend is None:
            self._backend = _detect_backend()
        self._backend.load()
        self._loaded = True
        logger.info("LLM engine ready (backend=%s)", self._backend.name)

    @property
    def backend_name(self) -> str:
        if self._backend is None:
            return "not loaded"
        return self._backend.name

    def _format_prompt(self, prompt: str, think: bool = False) -> str:
        """Format prompt using Qwen3 ChatML template.

        Args:
            prompt: User message.
            think: If True, allow the model's thinking/reasoning mode.
                   If False (default), suppress <think> blocks for direct answers.
        """
        parts = []
        if self._system_prompt:
            parts.append(f"<|im_start|>system\n{self._system_prompt}<|im_end|>")
        suffix = "" if think else " /no_think"
        parts.append(f"<|im_start|>user\n{prompt}{suffix}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> blocks from output."""
        import re
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        think: bool = False,
        raw: bool = False,
    ) -> str:
        """Generate a complete response.

        Args:
            prompt: User message (or raw prompt if raw=True).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Repetition penalty factor.
            think: If True, allow model's reasoning mode. Default suppresses it.
            raw: If True, skip ChatML formatting.

        Returns:
            Generated text (think blocks stripped unless think=True).
        """
        self._ensure_loaded()
        formatted = prompt if raw else self._format_prompt(prompt, think=think)
        config = GenerateConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        result = self._backend.generate(formatted, config)
        return result if think else self._strip_think(result)

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        think: bool = False,
        raw: bool = False,
    ) -> Iterator[str]:
        """Stream tokens as they are generated.

        Args:
            think: If True, yield think blocks too. Default filters them out.

        Yields:
            Individual tokens as strings.
        """
        self._ensure_loaded()
        formatted = prompt if raw else self._format_prompt(prompt, think=think)
        config = GenerateConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        if think:
            yield from self._backend.stream(formatted, config)
        else:
            # Filter out <think>...</think> blocks from stream.
            # The model may emit <think>...</think> at the start;
            # we suppress everything inside and yield the rest.
            in_think = False
            buf = ""
            for token in self._backend.stream(formatted, config):
                buf += token
                while buf:
                    if in_think:
                        end = buf.find("</think>")
                        if end == -1:
                            buf = ""
                            break
                        buf = buf[end + len("</think>"):]
                        in_think = False
                    else:
                        start = buf.find("<think>")
                        if start == -1:
                            # Check if buffer ends with a partial "<think" prefix
                            hold = 0
                            for k in range(min(len(buf), 6), 0, -1):
                                if "<think>".startswith(buf[-k:]):
                                    hold = k
                                    break
                            if hold:
                                yield buf[:-hold]
                                buf = buf[-hold:]
                            else:
                                yield buf
                                buf = ""
                            break
                        else:
                            if start > 0:
                                yield buf[:start]
                            buf = buf[start + len("<think>"):]
                            in_think = True
            if buf and not in_think:
                yield buf

    def unload(self) -> None:
        """Release model from memory."""
        if self._backend:
            self._backend.unload()
        self._loaded = False
