"""
Glyphh SDK - Hyperdimensional Computing for RAG

The Glyphh SDK is a hybrid Python + Rust SDK for Hyperdimensional Computing (HDC)
that serves as the core math layer for a RAG sidecar product.

Core Features:
- Bipolar vector operations with strict validation
- Hierarchical encoding (global cortex → layers → segments → roles)
- Edge-based reasoning with 8 edge types (4 spatial + 4 temporal)
- Dual weighting system (similarity + security)
- Comprehensive fact trees with citations
- Temporal prediction using beam search
- Custom encoder extension framework

Example:
    >>> from glyphh import Encoder, EncoderConfig, Concept
    >>> 
    >>> config = EncoderConfig(dimension=10000, seed=42)
    >>> encoder = Encoder(config)
    >>> 
    >>> concept = Concept(
    ...     name="red car",
    ...     attributes={"type": "car", "color": "red"}
    ... )
    >>> glyph = encoder.encode(concept)
"""

from .core import (
    Vector,
    Concept,
    Edge,
    Glyph,
    Layer,
    Segment,
    compute_space_id,
    EncoderConfig,
    NLEncoderConfig,
    Role,
    LayerConfig,
    SegmentConfig,
    migrate_legacy_config,
)

from .exceptions import (
    GlyphhException,
    ConfigurationException,
    VectorSpaceException,
    DimensionMismatchException,
    SpaceIdMismatchException,
    BipolarConstraintException,
    EncodingException,
    ModelValidationException,
    ValidationException,
    OperationException,
    log_encoding_failure,
    log_validation_failure,
)

from .core.ops import (
    bind,
    bundle,
    cosine_similarity,
    hamming_similarity,
    generate_symbol,
)

from .encoder import Encoder
from .encoder.intent import IntentEncoder, IntentPattern, IntentMatch
from .encoder.default_intents import DEFAULT_INTENT_PATTERNS

from .edges import EdgeGenerator
from .similarity import SimilarityCalculator, SimilarityResult
from .fact_tree import FactTree, FactNode, Citation
from .temporal import (
    TemporalEncoder,
    BeamSearchPredictor,
    PredictionResult,
    Prediction,
    TrendStatistics
)

from .model import GlyphhModel

# Conversation state — HDC pathway encoding for sequential decision-making
from .state import Centroid, ConversationState, DeductiveLayer, InductiveLayer, Pathway, PathwayLibrary, Transition

# Intent extraction — rule-based NL parsing with HDC fallback
from .intent import (
    IntentExtractor,
    VERB_CONFIG as INTENT_VERB_CONFIG,
    NOUN_CONFIG as INTENT_NOUN_CONFIG,
    get_extractor,
    extract_intent,
    extract_action,
    extract_target,
    infer_domain,
)

# Cognitive loop — HDC domain-agnostic tool calling
from .cognitive import CognitiveLoop, DomainConfig, IdeaSpace, StepResult

# Linguistics — layered HDC language engine (character → morphology → POS → syntax → attention)
from .linguistics import LinguisticIntentParser

from .visualization import (
    visualize_similarity_graph,
    create_similarity_matrix,
    visualize_vector_heatmap,
    visualize_vector_comparison,
    visualize_fact_tree,
    print_fact_tree
)

# GQL - Glyph Query Language
from .gql import (
    GQLError,
    LexerError,
    ParseError,
    PlanningError,
    ExecutionError,
    SlotValidationError,
    HierarchyError,
    PredictionError,
)

__version__ = "0.12.0"

__all__ = [
    # Core types (runtime)
    "Vector",
    "Concept",
    "Edge",
    "Glyph",
    "Layer",
    "Segment",
    "compute_space_id",
    # Configuration
    "EncoderConfig",
    "NLEncoderConfig",
    "Role",
    "LayerConfig",
    "SegmentConfig",
    "migrate_legacy_config",
    # Encoder
    "Encoder",
    # Intent Encoder (HDC pattern matching)
    "IntentEncoder",
    "IntentPattern",
    "IntentMatch",
    "DEFAULT_INTENT_PATTERNS",
    # Edge Generator
    "EdgeGenerator",
    # Similarity
    "SimilarityCalculator",
    "SimilarityResult",
    # Fact Tree
    "FactTree",
    "FactNode",
    "Citation",
    # Temporal
    "TemporalEncoder",
    "BeamSearchPredictor",
    "PredictionResult",
    "Prediction",
    "TrendStatistics",
    # Model Packaging
    "GlyphhModel",
    # Conversation State (HDC pathway encoding + deductive/inductive reasoning)
    "Centroid",
    "ConversationState",
    "DeductiveLayer",
    "InductiveLayer",
    "Pathway",
    "PathwayLibrary",
    "Transition",
    # Visualization
    "visualize_similarity_graph",
    "create_similarity_matrix",
    "visualize_vector_heatmap",
    "visualize_vector_comparison",
    "visualize_fact_tree",
    "print_fact_tree",
    # Exceptions
    "GlyphhException",
    "ConfigurationException",
    "VectorSpaceException",
    "DimensionMismatchException",
    "SpaceIdMismatchException",
    "BipolarConstraintException",
    "EncodingException",
    "ModelValidationException",
    "ValidationException",
    "OperationException",
    "log_encoding_failure",
    "log_validation_failure",
    # HDC operations
    "bind",
    "bundle",
    "cosine_similarity",
    "hamming_similarity",
    "generate_symbol",
    # Intent Extractor (rule-based NL parsing + HDC fallback)
    "IntentExtractor",
    "INTENT_VERB_CONFIG",
    "INTENT_NOUN_CONFIG",
    "get_extractor",
    "extract_intent",
    "extract_action",
    "extract_target",
    "infer_domain",
    # Cognitive Loop (HDC domain-agnostic tool calling)
    "CognitiveLoop",
    "DomainConfig",
    "IdeaSpace",
    "StepResult",
    # Linguistics Engine (layered HDC NLP)
    "LinguisticIntentParser",
    # GQL Exceptions
    "GQLError",
    "LexerError",
    "ParseError",
    "PlanningError",
    "ExecutionError",
    "SlotValidationError",
    "HierarchyError",
    "PredictionError",
]
