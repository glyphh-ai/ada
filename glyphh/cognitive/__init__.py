"""
glyphh.cognitive — HDC + LLM cognitive loop for domain-agnostic tool calling.

Core classes:
    CognitiveLoop            — the main loop (perceive → recall → deduce → slot → decide)
    SchemaIntentClassifier   — LLM-primary intent classification from function schemas
    IntentCache              — HDC cache that learns from LLM decisions
    DomainConfig             — domain-specific configuration loader
    SlotExtractor            — data-driven argument extraction
    IdeaSpace                — episodic memory with Hebbian reinforcement
    StepResult               — result of one cognitive loop turn

The SDK classes are pure engines with zero domain knowledge.
Domain logic lives in config files provided by the model.
"""

from .domain import DomainConfig, SlotDefinition, StateEffect, TriggerSuppression, StateFormat
from .idea import IdeaEncoder, IdeaSpace, Idea
from .intent_cache import IntentCache
from .loop import CognitiveLoop, StepResult
from .schema_classifier import SchemaIntentClassifier
from .slots import SlotExtractor

__all__ = [
    "CognitiveLoop",
    "DomainConfig",
    "IdeaEncoder",
    "IdeaSpace",
    "Idea",
    "IntentCache",
    "SchemaIntentClassifier",
    "SlotDefinition",
    "SlotExtractor",
    "StateEffect",
    "StateFormat",
    "StepResult",
    "TriggerSuppression",
]
