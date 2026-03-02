"""
glyphh.cognitive — HDC cognitive loop for domain-agnostic tool calling.

Core classes:
    CognitiveLoop            — the main loop (perceive → recall → deduce → slot → decide)
    SchemaIntentClassifier   — intent classification from function schemas via GlyphSpace
    GlyphSpace               — unified glyph storage + scoring + caching
    ScoringStrategy          — protocol for model-specific scoring
    DefaultScoringStrategy   — cosine similarity on global cortex
    DomainConfig             — domain-specific configuration loader
    SlotExtractor            — data-driven argument extraction
    IdeaSpace                — episodic memory with Hebbian reinforcement
    StepResult               — result of one cognitive loop turn

The SDK classes are pure engines with zero domain knowledge.
Domain logic lives in config files provided by the model.
"""

from .domain import DomainConfig, SlotDefinition, StateEffect, TriggerSuppression, StateFormat
from .glyph_space import GlyphSpace, ScoringStrategy, DefaultScoringStrategy
from .idea import IdeaEncoder, IdeaSpace, Idea
from .intent_cache import IntentCache  # deprecated, kept for backward compat
from .loop import CognitiveLoop, StepResult
from .model_scorer import ModelScorer, ScorerResult
from .schema_classifier import SchemaIntentClassifier
from .slots import SlotExtractor

__all__ = [
    "CognitiveLoop",
    "DefaultScoringStrategy",
    "DomainConfig",
    "GlyphSpace",
    "IdeaEncoder",
    "IdeaSpace",
    "Idea",
    "IntentCache",
    "ModelScorer",
    "SchemaIntentClassifier",
    "ScorerResult",
    "ScoringStrategy",
    "SlotDefinition",
    "SlotExtractor",
    "StateEffect",
    "StateFormat",
    "StepResult",
    "TriggerSuppression",
]
