"""
Glyphh Assistant module.

The assistant is a Glyphh model that routes natural language queries
to actions (commands, knowledge responses, quick actions, query help).

Primary API:
- Assistant: thin client (online -> runtime, offline -> local .glyphh)
- AssistantConfig: configuration (runtime_url, threshold, etc.)
- AssistantResponse: state-based response (DONE, ASK, ERROR)

Build tooling:
- encoder_config: ASSISTANT_ENCODER_CONFIG (router-style EncoderConfig)
- concept_converter: JSONL entry -> Concept conversion
"""

from glyphh.assistant.core import (
    Assistant,
    AssistantConfig,
    AssistantResponse,
)

__all__ = [
    "Assistant",
    "AssistantConfig",
    "AssistantResponse",
]
