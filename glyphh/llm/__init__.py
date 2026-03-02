"""
glyphh.llm — Local LLM inference for the Glyphh runtime.

The LLM is a core structural component — the "voice box" to the HDC "brain."
It handles natural language understanding at specific pipeline points while
the HDC brain handles pattern matching, state tracking, and decision-making.

Usage::

    from glyphh.llm import LLMEngine

    engine = LLMEngine()
    engine = LLMEngine(model_path="/path/to/model.gguf")

Integration with CognitiveLoop::

    from glyphh.cognitive import CognitiveLoop
    from glyphh.llm import LLMEngine

    engine = LLMEngine()
    loop = CognitiveLoop(
        packs=["my_pack"],
        domain_config=config,
        llm_engine=engine,
    )
"""

from .engine import LLMEngine
from .structured import LLMResult

__all__ = [
    "LLMEngine",
    "LLMResult",
]
