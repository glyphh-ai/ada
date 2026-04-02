"""
Glyphh LLM — local language model inference.

Provides Ada's voice: token generation from Qwen3-1.5B via the best
available backend (MLX on Apple Silicon, llama-cpp-python elsewhere,
or an external OpenAI-compatible API).

The engine lazy-loads on first call and auto-detects the backend.

Usage:
    from glyphh.llm import LLMEngine

    engine = LLMEngine()          # auto-detect backend + model
    text = engine.generate("Why is the sky blue?")

    # Streaming
    for token in engine.stream("Why is the sky blue?"):
        print(token, end="", flush=True)
"""

from glyphh.llm.engine import LLMEngine

__all__ = ["LLMEngine"]
