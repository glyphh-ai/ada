"""
Auto Schema Matcher module for the Glyphh SDK.

This module provides the main SDK interface for auto-schema NL query matching,
combining all the NL query matching components (tokenizer, matcher, inferrer,
extractor) into a unified interface.

Classes:
    AutoMatchConfig: Configuration for auto-schema matching
    AutoMatchResult: Complete result from auto-schema matching
    AutoSchemaMatcher: Main interface combining all components

The auto-schema approach uses the existing HDC encoder to create bipolar vectors
for schema elements (role names, value labels), then matches incoming query tokens
against these vectors to understand what the user is asking about.

Validates: Requirements 1-6
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from glyphh.nl.query_tokenizer import TokenizerConfig, QueryTokenizer, Token
from glyphh.nl.schema_matcher import MatchConfig, MatchResult, SchemaMatcher, TokenMatch
from glyphh.nl.schema_vectorizer import SchemaVectorizer, SchemaVector
from glyphh.nl.intent_inferrer import IntentKeywords, InferredIntent, IntentInferrer
from glyphh.nl.parameter_extractor import ExtractionResult, ParameterExtractor

if TYPE_CHECKING:
    from glyphh.encoder.base import Encoder
    from glyphh.core.config import EncoderConfig, NLEncoderConfig


# Module logger for deprecation warnings
# Validates: Requirement 14.6 - THE SDK SHALL log deprecation warnings for manual patterns
logger = logging.getLogger(__name__)


# Valid match methods supported by the system
VALID_MATCH_METHODS = {"auto", "manual", "hybrid"}

# Default threshold for deprecation warnings
# If auto-matching confidence exceeds this, a deprecation warning is logged
DEFAULT_DEPRECATION_THRESHOLD = 0.7


@dataclass
class AutoMatchConfig:
    """
    Configuration for auto-schema matching.
    
    Controls the behavior of the AutoSchemaMatcher, including tokenization
    settings, matching thresholds, intent keywords, and feature flags for
    compound matching, synonyms, and fallback behavior.
    
    This configuration combines settings from multiple components:
    - TokenizerConfig: Controls query tokenization (n-grams, stop words, normalization)
    - MatchConfig: Controls matching thresholds and bonuses
    - IntentKeywords: Controls intent detection keywords
    - Feature flags: Enable/disable specific matching features
    
    Attributes:
        tokenizer_config: Configuration for query tokenization.
            Controls n-gram generation, stop word handling, and text normalization.
            Default: TokenizerConfig() with standard settings.
        match_config: Configuration for schema matching.
            Controls similarity thresholds and bonuses for matching.
            Default: MatchConfig() with standard thresholds (0.3).
        intent_keywords: Keywords for intent detection.
            Controls which keywords trigger which intent types.
            Default: IntentKeywords() with standard keywords.
        enable_compound_matching: Whether to enable compound (multi-word) matching.
            When True, the matcher will attempt to match multi-word phrases
            like "brake pads" as single units before individual tokens.
            Default: True
        enable_synonyms: Whether to enable synonym matching.
            When True, the matcher will match queries against both primary
            values and their defined synonyms.
            Default: True
        fallback_to_manual: Whether to fall back to manual patterns.
            When True, if auto-matching fails or has low confidence, the
            system will attempt to match against manually configured patterns.
            Default: True
        hybrid_intent_config: Configuration for hybrid mode matching.
            Dictionary mapping intent types to matching method ("manual" or "auto").
            When configured, allows some intents to use manual patterns while
            others use auto-matching. Intents not in the config use the default
            fallback behavior.
            Example: {"find": "manual", "count": "auto", "similar": "auto"}
            Default: None (disabled, use fallback_to_manual behavior)
    
    Example:
        >>> # Default configuration
        >>> config = AutoMatchConfig()
        >>> config.tokenizer_config.max_ngram_size
        3
        >>> config.match_config.role_threshold
        0.3
        >>> config.enable_compound_matching
        True
        >>> config.enable_synonyms
        True
        >>> config.fallback_to_manual
        True
        
        >>> # Custom configuration with stricter thresholds
        >>> custom_config = AutoMatchConfig(
        ...     match_config=MatchConfig(
        ...         role_threshold=0.5,
        ...         value_threshold=0.5
        ...     ),
        ...     enable_compound_matching=True,
        ...     enable_synonyms=False,
        ...     fallback_to_manual=False
        ... )
        >>> custom_config.match_config.role_threshold
        0.5
        >>> custom_config.enable_synonyms
        False
        
        >>> # Configuration with custom tokenizer
        >>> from glyphh.nl.query_tokenizer import TokenizerConfig
        >>> tokenizer_cfg = TokenizerConfig(
        ...     max_ngram_size=4,
        ...     stop_words={"the", "a", "an"}
        ... )
        >>> config = AutoMatchConfig(tokenizer_config=tokenizer_cfg)
        >>> config.tokenizer_config.max_ngram_size
        4
        
        >>> # Configuration with custom intent keywords
        >>> from glyphh.nl.intent_inferrer import IntentKeywords
        >>> keywords = IntentKeywords(
        ...     count={"count", "total", "sum"},
        ...     find={"find", "locate", "retrieve"}
        ... )
        >>> config = AutoMatchConfig(intent_keywords=keywords)
        >>> "total" in config.intent_keywords.count
        True
        
        >>> # Hybrid mode configuration
        >>> hybrid_config = AutoMatchConfig(
        ...     hybrid_intent_config={
        ...         "find": "manual",
        ...         "count": "auto",
        ...         "similar": "auto"
        ...     }
        ... )
        >>> hybrid_config.hybrid_intent_config["find"]
        'manual'
        >>> hybrid_config.hybrid_intent_config["count"]
        'auto'
    
    Notes:
        - All sub-configurations use their default values if not specified
        - Feature flags can be used to disable specific matching behaviors
        - The fallback_to_manual flag enables hybrid matching mode
        - Compound matching is recommended for schemas with multi-word values
        - Synonym matching requires synonyms to be defined in the schema
        - hybrid_intent_config enables fine-grained control over which intents
          use manual vs auto matching
    
    Validates: Requirements 1-6
    Validates: Requirement 6.1 - THE SDK SHALL support manual pattern configuration alongside auto-matching
    Validates: Requirement 10.1 - THE SDK SHALL generate vectors for multi-word values as single units
    Validates: Requirement 11.2 - THE SDK SHALL support defining synonyms for values
    Validates: Requirement 14.4 - THE SDK SHALL support hybrid mode where some intents use manual patterns and others use auto-matching
    """
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    match_config: MatchConfig = field(default_factory=MatchConfig)
    intent_keywords: IntentKeywords = field(default_factory=IntentKeywords)
    enable_compound_matching: bool = True
    enable_synonyms: bool = True
    fallback_to_manual: bool = True
    hybrid_intent_config: Optional[Dict[str, str]] = None
    
    def __post_init__(self) -> None:
        """Validate AutoMatchConfig fields after initialization.
        
        Validates that:
        - tokenizer_config is a TokenizerConfig instance
        - match_config is a MatchConfig instance
        - intent_keywords is an IntentKeywords instance
        - Boolean flags are boolean types
        - hybrid_intent_config is None or a valid dictionary
        
        Raises:
            TypeError: If tokenizer_config is not a TokenizerConfig instance
            TypeError: If match_config is not a MatchConfig instance
            TypeError: If intent_keywords is not an IntentKeywords instance
            TypeError: If any boolean flag is not a boolean
            TypeError: If hybrid_intent_config is not None or a dictionary
            ValueError: If hybrid_intent_config contains invalid values
        """
        # Validate tokenizer_config type
        if not isinstance(self.tokenizer_config, TokenizerConfig):
            raise TypeError(
                f"tokenizer_config must be a TokenizerConfig instance, "
                f"got {type(self.tokenizer_config).__name__}"
            )
        
        # Validate match_config type
        if not isinstance(self.match_config, MatchConfig):
            raise TypeError(
                f"match_config must be a MatchConfig instance, "
                f"got {type(self.match_config).__name__}"
            )
        
        # Validate intent_keywords type
        if not isinstance(self.intent_keywords, IntentKeywords):
            raise TypeError(
                f"intent_keywords must be an IntentKeywords instance, "
                f"got {type(self.intent_keywords).__name__}"
            )
        
        # Validate enable_compound_matching is boolean
        if not isinstance(self.enable_compound_matching, bool):
            raise TypeError(
                f"enable_compound_matching must be a boolean, "
                f"got {type(self.enable_compound_matching).__name__}"
            )
        
        # Validate enable_synonyms is boolean
        if not isinstance(self.enable_synonyms, bool):
            raise TypeError(
                f"enable_synonyms must be a boolean, "
                f"got {type(self.enable_synonyms).__name__}"
            )
        
        # Validate fallback_to_manual is boolean
        if not isinstance(self.fallback_to_manual, bool):
            raise TypeError(
                f"fallback_to_manual must be a boolean, "
                f"got {type(self.fallback_to_manual).__name__}"
            )
        
        # Validate hybrid_intent_config
        # Validates: Requirement 14.4 - THE SDK SHALL support hybrid mode
        if self.hybrid_intent_config is not None:
            if not isinstance(self.hybrid_intent_config, dict):
                raise TypeError(
                    f"hybrid_intent_config must be a dictionary or None, "
                    f"got {type(self.hybrid_intent_config).__name__}"
                )
            
            # Valid matching methods for hybrid mode
            valid_methods = {"manual", "auto"}
            
            # Validate each entry in hybrid_intent_config
            for intent_type, method in self.hybrid_intent_config.items():
                if not isinstance(intent_type, str):
                    raise TypeError(
                        f"hybrid_intent_config keys must be strings, "
                        f"got {type(intent_type).__name__} for key {intent_type!r}"
                    )
                if not isinstance(method, str):
                    raise TypeError(
                        f"hybrid_intent_config values must be strings, "
                        f"got {type(method).__name__} for intent '{intent_type}'"
                    )
                if method not in valid_methods:
                    raise ValueError(
                        f"hybrid_intent_config values must be one of {valid_methods}, "
                        f"got '{method}' for intent '{intent_type}'"
                    )
    
    def __repr__(self) -> str:
        """Return a string representation of the AutoMatchConfig."""
        return (
            f"AutoMatchConfig("
            f"tokenizer_config={self.tokenizer_config!r}, "
            f"match_config={self.match_config!r}, "
            f"intent_keywords={self.intent_keywords!r}, "
            f"enable_compound_matching={self.enable_compound_matching}, "
            f"enable_synonyms={self.enable_synonyms}, "
            f"fallback_to_manual={self.fallback_to_manual}, "
            f"hybrid_intent_config={self.hybrid_intent_config!r})"
        )


@dataclass
class AutoMatchResult:
    """
    Complete result from auto-schema matching.
    
    Contains all information about a matched query, including the original query,
    inferred intent, extracted parameters, match details, confidence score,
    matching method used, and disambiguation information.
    
    This is the primary output of the AutoSchemaMatcher.match_query() method,
    providing a comprehensive view of how a natural language query was interpreted.
    
    Attributes:
        query: The original query string that was matched.
            Preserved for reference and debugging.
        intent: The inferred intent from the query.
            Contains intent_type, confidence, matched_keywords, and supporting_matches.
        parameters: The extracted parameters from the query.
            Contains single-value params, multi-value params, and unmatched tokens.
        match_result: The detailed match result from schema matching.
            Contains token_matches, role_matches, value_matches, compound_matches.
        confidence: Overall confidence score for the match, in range [0.0, 1.0].
            This is a combined score considering intent confidence and match quality.
            - 1.0: Very high confidence (clear intent + strong matches)
            - 0.7-0.9: High confidence (good intent + good matches)
            - 0.4-0.6: Medium confidence (partial signals)
            - 0.1-0.3: Low confidence (weak signals)
            - 0.0: No confidence (no matches found)
        match_method: The method used for matching. One of:
            - "auto": Fully automatic schema-based matching
            - "manual": Manual pattern matching was used
            - "hybrid": Combination of auto and manual matching
        disambiguation_needed: Whether the query is ambiguous and needs clarification.
            True if multiple intents have similar confidence scores.
        disambiguation_options: List of alternative intents when disambiguation is needed.
            Contains InferredIntent objects for each possible interpretation.
            Empty list if disambiguation_needed is False.
    
    Example:
        >>> from glyphh.nl.intent_inferrer import InferredIntent
        >>> from glyphh.nl.parameter_extractor import ExtractionResult
        >>> from glyphh.nl.schema_matcher import MatchResult
        >>> 
        >>> # Create a successful match result
        >>> result = AutoMatchResult(
        ...     query="Find Toyota brake pads",
        ...     intent=InferredIntent(
        ...         intent_type="find",
        ...         confidence=0.85,
        ...         matched_keywords=["find"],
        ...         supporting_matches=[]
        ...     ),
        ...     parameters=ExtractionResult(
        ...         parameters={"make": make_param, "category": category_param},
        ...         multi_value_params={},
        ...         unmatched_tokens=[]
        ...     ),
        ...     match_result=MatchResult(
        ...         query="Find Toyota brake pads",
        ...         token_matches=[toyota_match, brake_pads_match],
        ...         role_matches=[],
        ...         value_matches=[toyota_match],
        ...         compound_matches=[brake_pads_match],
        ...         all_candidates=[]
        ...     ),
        ...     confidence=0.85,
        ...     match_method="auto",
        ...     disambiguation_needed=False,
        ...     disambiguation_options=[]
        ... )
        >>> result.query
        'Find Toyota brake pads'
        >>> result.intent.intent_type
        'find'
        >>> result.confidence
        0.85
        >>> result.match_method
        'auto'
        >>> result.disambiguation_needed
        False
        
        >>> # Result requiring disambiguation
        >>> ambiguous_result = AutoMatchResult(
        ...     query="Toyota",
        ...     intent=InferredIntent(
        ...         intent_type="find",
        ...         confidence=0.5,
        ...         matched_keywords=[],
        ...         supporting_matches=[]
        ...     ),
        ...     parameters=ExtractionResult(
        ...         parameters={},
        ...         multi_value_params={},
        ...         unmatched_tokens=[]
        ...     ),
        ...     match_result=MatchResult(query="Toyota"),
        ...     confidence=0.5,
        ...     match_method="auto",
        ...     disambiguation_needed=True,
        ...     disambiguation_options=[find_intent, count_intent]
        ... )
        >>> ambiguous_result.disambiguation_needed
        True
        >>> len(ambiguous_result.disambiguation_options)
        2
        
        >>> # Invalid match_method raises ValueError
        >>> AutoMatchResult(
        ...     query="test",
        ...     intent=some_intent,
        ...     parameters=some_params,
        ...     match_result=some_result,
        ...     confidence=0.5,
        ...     match_method="invalid",
        ...     disambiguation_needed=False,
        ...     disambiguation_options=[]
        ... )
        Traceback (most recent call last):
            ...
        ValueError: match_method must be one of {'auto', 'manual', 'hybrid'}, got 'invalid'
        
        >>> # Invalid confidence raises ValueError
        >>> AutoMatchResult(
        ...     query="test",
        ...     intent=some_intent,
        ...     parameters=some_params,
        ...     match_result=some_result,
        ...     confidence=1.5,
        ...     match_method="auto",
        ...     disambiguation_needed=False,
        ...     disambiguation_options=[]
        ... )
        Traceback (most recent call last):
            ...
        ValueError: confidence must be between 0.0 and 1.0, got 1.5
    
    Notes:
        - The query field preserves the original user input
        - The intent field contains the primary inferred intent
        - The parameters field contains all extracted query parameters
        - The match_result field provides detailed matching information
        - The confidence field is an overall score combining multiple factors
        - The match_method indicates which matching strategy was used
        - When disambiguation_needed is True, disambiguation_options contains alternatives
        - The disambiguation_options list should be empty when disambiguation_needed is False
    
    Validates: Requirements 1-6
    Validates: Requirement 4.6 - THE SDK SHALL return confidence scores for inferred intents
    Validates: Requirement 6.4 - THE SDK SHALL provide a priority system for pattern matching
    Validates: Requirement 12.1 - WHEN multiple intents have similar confidence, THE SDK SHALL return all candidates
    Validates: Requirement 12.2 - THE SDK SHALL provide a disambiguation response suggesting clarifications
    """
    query: str
    intent: InferredIntent
    parameters: ExtractionResult
    match_result: MatchResult
    confidence: float
    match_method: str
    disambiguation_needed: bool
    disambiguation_options: List[InferredIntent]
    
    def __post_init__(self) -> None:
        """Validate AutoMatchResult fields after initialization.
        
        Validates that:
        - query is a string
        - intent is an InferredIntent instance
        - parameters is an ExtractionResult instance
        - match_result is a MatchResult instance
        - confidence is in range [0.0, 1.0]
        - match_method is one of "auto", "manual", "hybrid"
        - disambiguation_needed is a boolean
        - disambiguation_options is a list of InferredIntent instances
        
        Raises:
            TypeError: If any field has an incorrect type
            ValueError: If confidence is out of range
            ValueError: If match_method is not a valid value
        """
        # Validate query is a string
        if not isinstance(self.query, str):
            raise TypeError(
                f"query must be a string, got {type(self.query).__name__}"
            )
        
        # Validate intent is an InferredIntent instance
        if not isinstance(self.intent, InferredIntent):
            raise TypeError(
                f"intent must be an InferredIntent instance, "
                f"got {type(self.intent).__name__}"
            )
        
        # Validate parameters is an ExtractionResult instance
        if not isinstance(self.parameters, ExtractionResult):
            raise TypeError(
                f"parameters must be an ExtractionResult instance, "
                f"got {type(self.parameters).__name__}"
            )
        
        # Validate match_result is a MatchResult instance
        if not isinstance(self.match_result, MatchResult):
            raise TypeError(
                f"match_result must be a MatchResult instance, "
                f"got {type(self.match_result).__name__}"
            )
        
        # Validate confidence is a number
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(
                f"confidence must be a number, "
                f"got {type(self.confidence).__name__}"
            )
        
        # Validate confidence is in valid range [0.0, 1.0]
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, "
                f"got {self.confidence}"
            )
        
        # Validate match_method is one of the valid values
        if self.match_method not in VALID_MATCH_METHODS:
            raise ValueError(
                f"match_method must be one of {VALID_MATCH_METHODS}, "
                f"got '{self.match_method}'"
            )
        
        # Validate disambiguation_needed is a boolean
        if not isinstance(self.disambiguation_needed, bool):
            raise TypeError(
                f"disambiguation_needed must be a boolean, "
                f"got {type(self.disambiguation_needed).__name__}"
            )
        
        # Validate disambiguation_options is a list
        if not isinstance(self.disambiguation_options, list):
            raise TypeError(
                f"disambiguation_options must be a list, "
                f"got {type(self.disambiguation_options).__name__}"
            )
        
        # Validate all items in disambiguation_options are InferredIntent instances
        for i, option in enumerate(self.disambiguation_options):
            if not isinstance(option, InferredIntent):
                raise TypeError(
                    f"disambiguation_options[{i}] must be an InferredIntent instance, "
                    f"got {type(option).__name__}"
                )
    
    def __repr__(self) -> str:
        """Return a string representation of the AutoMatchResult."""
        return (
            f"AutoMatchResult("
            f"query='{self.query}', "
            f"intent={self.intent.intent_type}, "
            f"confidence={self.confidence:.4f}, "
            f"match_method='{self.match_method}', "
            f"disambiguation_needed={self.disambiguation_needed}, "
            f"disambiguation_options={len(self.disambiguation_options)})"
        )
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if this result has high confidence.
        
        Args:
            threshold: The confidence threshold for "high confidence".
                Default is 0.7.
        
        Returns:
            True if confidence >= threshold, False otherwise.
        
        Example:
            >>> result = AutoMatchResult(...)
            >>> result.confidence = 0.85
            >>> result.is_high_confidence()
            True
            >>> result.is_high_confidence(threshold=0.9)
            False
        """
        return self.confidence >= threshold
    
    def needs_clarification(self) -> bool:
        """Check if this result needs user clarification.
        
        Returns:
            True if disambiguation_needed is True, False otherwise.
        
        Example:
            >>> result = AutoMatchResult(...)
            >>> result.disambiguation_needed = True
            >>> result.needs_clarification()
            True
        """
        return self.disambiguation_needed
    
    def is_auto_matched(self) -> bool:
        """Check if this result was matched using auto-schema matching.
        
        Returns:
            True if match_method is "auto", False otherwise.
        
        Example:
            >>> result = AutoMatchResult(...)
            >>> result.match_method = "auto"
            >>> result.is_auto_matched()
            True
        """
        return self.match_method == "auto"
    
    def is_manual_matched(self) -> bool:
        """Check if this result was matched using manual patterns.
        
        Returns:
            True if match_method is "manual", False otherwise.
        
        Example:
            >>> result = AutoMatchResult(...)
            >>> result.match_method = "manual"
            >>> result.is_manual_matched()
            True
        """
        return self.match_method == "manual"
    
    def is_hybrid_matched(self) -> bool:
        """Check if this result was matched using hybrid matching.
        
        Returns:
            True if match_method is "hybrid", False otherwise.
        
        Example:
            >>> result = AutoMatchResult(...)
            >>> result.match_method = "hybrid"
            >>> result.is_hybrid_matched()
            True
        """
        return self.match_method == "hybrid"
    
    def get_primary_intent(self) -> str:
        """Get the primary intent type.
        
        Returns:
            The intent_type from the intent field.
        
        Example:
            >>> result = AutoMatchResult(...)
            >>> result.intent.intent_type = "find"
            >>> result.get_primary_intent()
            'find'
        """
        return self.intent.intent_type
    
    def get_extracted_roles(self) -> List[str]:
        """Get all roles that have extracted parameters.
        
        Returns:
            A list of role names from the parameters field.
        
        Example:
            >>> result = AutoMatchResult(...)
            >>> result.get_extracted_roles()
            ['make', 'category']
        """
        return self.parameters.get_all_roles()
    
    def to_dict(self) -> dict:
        """Convert the result to a dictionary for serialization.
        
        Returns:
            A dictionary representation of the AutoMatchResult suitable
            for JSON serialization or API responses.
        
        Example:
            >>> result = AutoMatchResult(...)
            >>> d = result.to_dict()
            >>> d["query"]
            'Find Toyota brake pads'
            >>> d["confidence"]
            0.85
        """
        return {
            "query": self.query,
            "intent": {
                "intent_type": self.intent.intent_type,
                "confidence": self.intent.confidence,
                "matched_keywords": self.intent.matched_keywords,
            },
            "parameters": self.parameters.to_dict(),
            "confidence": self.confidence,
            "match_method": self.match_method,
            "disambiguation_needed": self.disambiguation_needed,
            "disambiguation_options": [
                {
                    "intent_type": opt.intent_type,
                    "confidence": opt.confidence,
                    "matched_keywords": opt.matched_keywords,
                }
                for opt in self.disambiguation_options
            ],
        }


class AutoSchemaMatcher:
    """
    Main interface for auto-schema NL matching.
    
    The AutoSchemaMatcher combines all NL query matching components into a unified
    interface for processing natural language queries against a model schema. It
    orchestrates the full matching pipeline:
    
    1. Schema Vectorization: Generate vectors for all roles and values in the schema
    2. Query Tokenization: Break incoming queries into tokens and n-grams
    3. Token Matching: Compare token vectors against schema vectors
    4. Intent Inference: Determine query intent from keywords and matches
    5. Parameter Extraction: Extract matched values as structured parameters
    
    The matcher supports both automatic schema-based matching and manual pattern
    fallback for edge cases. When manual_patterns is provided and fallback_to_manual
    is enabled in the config, manual patterns take priority over auto-matching.
    
    Attributes:
        encoder: The Encoder instance used for vector generation.
            Must be the same encoder used by the model to ensure vectors
            are in the same vector space.
        config: The EncoderConfig containing the model schema definition.
            Defines layers, segments, and roles for vectorization.
        auto_config: Configuration for auto-schema matching behavior.
            Controls tokenization, matching thresholds, intent keywords,
            and feature flags.
        manual_patterns: Optional NLEncoderConfig for manual pattern fallback.
            When provided and fallback_to_manual is True, manual patterns
            are checked before auto-matching.
        _vectorizer: SchemaVectorizer for generating schema vectors.
        _tokenizer: QueryTokenizer for tokenizing queries.
        _matcher: SchemaMatcher for matching tokens against schema.
        _inferrer: IntentInferrer for inferring query intent.
        _extractor: ParameterExtractor for extracting parameters.
    
    Example:
        >>> from glyphh.encoder.base import Encoder
        >>> from glyphh.core.config import EncoderConfig, Layer, Segment, Role
        >>> 
        >>> # Create encoder config with schema
        >>> config = EncoderConfig(
        ...     dimension=10000,
        ...     seed=42,
        ...     layers=[
        ...         Layer(
        ...             name="vehicle",
        ...             segments=[
        ...                 Segment(
        ...                     name="identity",
        ...                     roles=[
        ...                         Role(name="make"),
        ...                         Role(name="model"),
        ...                     ]
        ...                 )
        ...             ]
        ...         )
        ...     ]
        ... )
        >>> 
        >>> # Create encoder and matcher
        >>> encoder = Encoder(config)
        >>> matcher = AutoSchemaMatcher(encoder, config)
        >>> 
        >>> # Access components
        >>> matcher.encoder is encoder
        True
        >>> matcher.config is config
        True
        >>> isinstance(matcher.auto_config, AutoMatchConfig)
        True
        >>> matcher.manual_patterns is None
        True
        >>> 
        >>> # With custom auto_config
        >>> custom_config = AutoMatchConfig(
        ...     match_config=MatchConfig(role_threshold=0.5, value_threshold=0.5),
        ...     enable_compound_matching=True,
        ...     enable_synonyms=False
        ... )
        >>> matcher = AutoSchemaMatcher(encoder, config, auto_config=custom_config)
        >>> matcher.auto_config.match_config.role_threshold
        0.5
        >>> matcher.auto_config.enable_synonyms
        False
    
    Notes:
        - The encoder and config must be compatible (same dimension, seed)
        - All sub-components are initialized during construction
        - Schema vectors are generated lazily when first needed
        - The matcher is stateless - each query is processed independently
        - Manual patterns take priority over auto-matching when configured
    
    Validates: Requirements 1-6
    Validates: Requirement 1.3 - WHEN generating schema vectors, THE SDK SHALL use the same encoder configuration as the model
    Validates: Requirement 6.1 - THE SDK SHALL support manual pattern configuration alongside auto-matching
    Validates: Requirement 6.2 - WHEN a manual pattern matches, THE SDK SHALL use it instead of auto-matching
    """
    
    def __init__(
        self,
        encoder: 'Encoder',
        config: 'EncoderConfig',
        auto_config: Optional[AutoMatchConfig] = None,
        manual_patterns: Optional['NLEncoderConfig'] = None
    ) -> None:
        """
        Initialize the AutoSchemaMatcher with encoder, config, and optional settings.
        
        Creates a new AutoSchemaMatcher instance and initializes all sub-components:
        - SchemaVectorizer: For generating schema vectors
        - QueryTokenizer: For tokenizing queries
        - SchemaMatcher: For matching tokens against schema
        - IntentInferrer: For inferring query intent
        - ParameterExtractor: For extracting parameters
        
        Args:
            encoder: The Encoder instance to use for vector generation.
                Must be the same encoder used by the model to ensure
                vectors are in the same vector space.
            config: The EncoderConfig containing the model schema definition.
                Defines the layers, segments, and roles to vectorize.
            auto_config: Optional AutoMatchConfig controlling matching behavior.
                If None, a default AutoMatchConfig is created with:
                - Default tokenizer config (max_ngram_size=3, standard stop words)
                - Default match config (thresholds=0.3, bonuses for compound/exact)
                - Default intent keywords (count, find, filter, similar)
                - enable_compound_matching=True
                - enable_synonyms=True
                - fallback_to_manual=True
            manual_patterns: Optional NLEncoderConfig for manual pattern fallback.
                When provided and auto_config.fallback_to_manual is True,
                manual patterns are checked before auto-matching.
        
        Raises:
            TypeError: If encoder is not an Encoder instance.
            TypeError: If config is not an EncoderConfig instance.
            TypeError: If auto_config is provided but not an AutoMatchConfig instance.
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> # Basic initialization with defaults
            >>> config = EncoderConfig(dimension=10000, seed=42)
            >>> encoder = Encoder(config)
            >>> matcher = AutoSchemaMatcher(encoder, config)
            >>> 
            >>> # Verify encoder and config are stored
            >>> matcher.encoder is encoder
            True
            >>> matcher.config is config
            True
            >>> 
            >>> # Verify default auto_config is created
            >>> isinstance(matcher.auto_config, AutoMatchConfig)
            True
            >>> matcher.auto_config.enable_compound_matching
            True
            >>> matcher.auto_config.enable_synonyms
            True
            >>> matcher.auto_config.fallback_to_manual
            True
            >>> 
            >>> # Verify manual_patterns is None by default
            >>> matcher.manual_patterns is None
            True
            >>> 
            >>> # Verify sub-components are initialized
            >>> isinstance(matcher._vectorizer, SchemaVectorizer)
            True
            >>> isinstance(matcher._tokenizer, QueryTokenizer)
            True
            >>> isinstance(matcher._matcher, SchemaMatcher)
            True
            >>> isinstance(matcher._inferrer, IntentInferrer)
            True
            >>> isinstance(matcher._extractor, ParameterExtractor)
            True
            >>> 
            >>> # With custom auto_config
            >>> custom_auto_config = AutoMatchConfig(
            ...     match_config=MatchConfig(role_threshold=0.4, value_threshold=0.4),
            ...     enable_compound_matching=False
            ... )
            >>> matcher = AutoSchemaMatcher(encoder, config, auto_config=custom_auto_config)
            >>> matcher.auto_config.match_config.role_threshold
            0.4
            >>> matcher.auto_config.enable_compound_matching
            False
            >>> 
            >>> # Invalid encoder type raises TypeError
            >>> AutoSchemaMatcher("not_encoder", config)
            Traceback (most recent call last):
                ...
            TypeError: encoder must be an Encoder instance, got str
            >>> 
            >>> # Invalid config type raises TypeError
            >>> AutoSchemaMatcher(encoder, "not_config")
            Traceback (most recent call last):
                ...
            TypeError: config must be an EncoderConfig instance, got str
            >>> 
            >>> # Invalid auto_config type raises TypeError
            >>> AutoSchemaMatcher(encoder, config, auto_config="not_config")
            Traceback (most recent call last):
                ...
            TypeError: auto_config must be an AutoMatchConfig instance or None, got str
        
        Validates: Requirements 1-6
        Validates: Requirement 1.3 - WHEN generating schema vectors, THE SDK SHALL use the same encoder configuration as the model
        Validates: Requirement 6.1 - THE SDK SHALL support manual pattern configuration alongside auto-matching
        """
        # Import types for validation (avoid circular imports at module level)
        from glyphh.encoder.base import Encoder
        from glyphh.core.config import EncoderConfig
        
        # Validate encoder is an Encoder instance
        if not isinstance(encoder, Encoder):
            raise TypeError(
                f"encoder must be an Encoder instance, "
                f"got {type(encoder).__name__}"
            )
        
        # Validate config is an EncoderConfig instance
        if not isinstance(config, EncoderConfig):
            raise TypeError(
                f"config must be an EncoderConfig instance, "
                f"got {type(config).__name__}"
            )
        
        # Validate auto_config if provided
        if auto_config is not None and not isinstance(auto_config, AutoMatchConfig):
            raise TypeError(
                f"auto_config must be an AutoMatchConfig instance or None, "
                f"got {type(auto_config).__name__}"
            )
        
        # Store encoder reference - ensures vectors are in the same space as the model
        # Validates: Requirement 1.3
        self.encoder = encoder
        
        # Store config reference - contains schema definition (layers, segments, roles)
        self.config = config
        
        # Store auto_config - use default if not provided
        # Default AutoMatchConfig has:
        #   - tokenizer_config: TokenizerConfig() with max_ngram_size=3, standard stop words
        #   - match_config: MatchConfig() with thresholds=0.3
        #   - intent_keywords: IntentKeywords() with standard keywords
        #   - enable_compound_matching: True
        #   - enable_synonyms: True
        #   - fallback_to_manual: True
        self.auto_config = auto_config if auto_config is not None else AutoMatchConfig()
        
        # Store manual_patterns for fallback matching
        # Validates: Requirement 6.1 - THE SDK SHALL support manual pattern configuration alongside auto-matching
        self.manual_patterns = manual_patterns
        
        # Initialize SchemaVectorizer
        # The vectorizer generates bipolar HDC vectors for all roles and values
        # in the model config. Uses the same encoder to ensure vectors are in
        # the same vector space.
        # Validates: Requirement 1 - Schema Vector Generation
        self._vectorizer = SchemaVectorizer(encoder, config)
        
        # Initialize QueryTokenizer with config from auto_config
        # The tokenizer breaks queries into tokens and n-grams for matching.
        # Uses tokenizer_config from auto_config for customization.
        # Validates: Requirement 2 - Query Tokenization
        self._tokenizer = QueryTokenizer(self.auto_config.tokenizer_config)
        
        # Initialize SchemaMatcher with schema vectors, encoder, and match config
        # The matcher compares token vectors against schema vectors to find matches.
        # Uses match_config from auto_config for threshold customization.
        # Note: Schema vectors are retrieved from vectorizer (may be empty initially,
        # will be populated when vectorize_schema() is called)
        # Validates: Requirement 3 - Token-to-Schema Matching
        self._matcher = SchemaMatcher(
            self._vectorizer.get_schema_vectors(),
            encoder,
            self.auto_config.match_config
        )
        
        # Initialize IntentInferrer with intent keywords from auto_config
        # The inferrer determines query intent from keywords and matches.
        # Uses intent_keywords from auto_config for customization.
        # Validates: Requirement 4 - Intent Inference from Matches
        self._inferrer = IntentInferrer(self.auto_config.intent_keywords)
        
        # Initialize ParameterExtractor with schema config
        # The extractor extracts matched values as structured parameters.
        # Uses the config for type inference based on schema definition.
        # Validates: Requirement 5 - Parameter Extraction
        self._extractor = ParameterExtractor(config)
    
    def __repr__(self) -> str:
        """Return a string representation of the AutoSchemaMatcher."""
        return (
            f"AutoSchemaMatcher("
            f"encoder_dim={self.encoder.dimension}, "
            f"config_layers={len(self.config.layers) if self.config.layers else 0}, "
            f"manual_patterns={'yes' if self.manual_patterns else 'no'}, "
            f"compound_matching={self.auto_config.enable_compound_matching}, "
            f"synonyms={self.auto_config.enable_synonyms})"
        )
    
    def get_schema_vectors(self) -> Dict[str, SchemaVector]:
        """
        Get all schema vectors for debugging.
        
        Returns the dictionary of schema vectors from the internal vectorizer.
        This is useful for debugging and understanding what schema elements
        are available for matching.
        
        Returns:
            Dictionary mapping schema element keys to SchemaVector instances.
            Keys are either role names (e.g., "make") or role=value pairs
            (e.g., "make=Toyota").
        
        Example:
            >>> matcher = AutoSchemaMatcher(encoder, config)
            >>> vectors = matcher.get_schema_vectors()
            >>> print(len(vectors))
            0  # Empty until vectorize_schema() is called on vectorizer
        """
        return self._vectorizer.get_schema_vectors()
    
    def add_synonym(self, value: str, synonym: str) -> None:
        """
        Add a synonym for a schema value.
        
        Delegates to the internal SchemaVectorizer to add a synonym mapping.
        When the synonym is matched during query processing, it will return
        the primary value.
        
        Args:
            value: The primary value string that the synonym maps to.
                This is the canonical value in the schema (e.g., "Toyota").
            synonym: The synonym string to add (e.g., "TYT", "Toy").
                This is an alternative way to refer to the primary value.
        
        Raises:
            ValueError: If value is empty or contains only whitespace.
            ValueError: If synonym is empty or contains only whitespace.
        
        Example:
            >>> matcher = AutoSchemaMatcher(encoder, config)
            >>> matcher.add_synonym("Toyota", "TYT")
            >>> # Now queries with "TYT" will match "Toyota"
        
        Validates: Requirements 11.2, 11.3
        """
        self._vectorizer.add_synonym(value, synonym)
    
    def add_alias(self, role: str, alias: str) -> None:
        """
        Add an alias for a role name.
        
        Delegates to the internal SchemaVectorizer to add an alias mapping.
        When the alias is matched during query processing, it will return
        the primary role.
        
        Args:
            role: The primary role name that the alias maps to.
                This is the canonical role in the schema (e.g., "make").
            alias: The alias string to add (e.g., "manufacturer", "brand").
                This is an alternative way to refer to the primary role.
        
        Raises:
            ValueError: If role is empty or contains only whitespace.
            ValueError: If alias is empty or contains only whitespace.
        
        Example:
            >>> matcher = AutoSchemaMatcher(encoder, config)
            >>> matcher.add_alias("make", "manufacturer")
            >>> # Now queries with "manufacturer" will match the "make" role
        
        Validates: Requirements 11.1, 11.3
        """
        self._vectorizer.add_alias(role, alias)
    
    def set_threshold(
        self,
        role_threshold: Optional[float] = None,
        value_threshold: Optional[float] = None
    ) -> None:
        """
        Update match thresholds.
        
        Updates the role and/or value thresholds in the matcher's config.
        Only provided thresholds are updated; None values are ignored.
        
        Args:
            role_threshold: New threshold for role matches (0.0 to 1.0).
                If None, the current role_threshold is preserved.
            value_threshold: New threshold for value matches (0.0 to 1.0).
                If None, the current value_threshold is preserved.
        
        Raises:
            ValueError: If role_threshold is not in range [0.0, 1.0].
            ValueError: If value_threshold is not in range [0.0, 1.0].
        
        Example:
            >>> matcher = AutoSchemaMatcher(encoder, config)
            >>> matcher._matcher.config.role_threshold
            0.3
            >>> matcher.set_threshold(role_threshold=0.5)
            >>> matcher._matcher.config.role_threshold
            0.5
            >>> matcher.set_threshold(value_threshold=0.4)
            >>> matcher._matcher.config.value_threshold
            0.4
        
        Validates: Requirements 9.1, 9.2
        """
        # Validate role_threshold if provided
        if role_threshold is not None:
            if not isinstance(role_threshold, (int, float)):
                raise TypeError(
                    f"role_threshold must be a number, "
                    f"got {type(role_threshold).__name__}"
                )
            if not 0.0 <= role_threshold <= 1.0:
                raise ValueError(
                    f"role_threshold must be between 0.0 and 1.0, "
                    f"got {role_threshold}"
                )
            self._matcher.config.role_threshold = role_threshold
        
        # Validate value_threshold if provided
        if value_threshold is not None:
            if not isinstance(value_threshold, (int, float)):
                raise TypeError(
                    f"value_threshold must be a number, "
                    f"got {type(value_threshold).__name__}"
                )
            if not 0.0 <= value_threshold <= 1.0:
                raise ValueError(
                    f"value_threshold must be between 0.0 and 1.0, "
                    f"got {value_threshold}"
                )
            self._matcher.config.value_threshold = value_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize for .glyphh file.
        
        Converts the AutoSchemaMatcher configuration to a dictionary
        suitable for serialization to a .glyphh file.
        
        Returns:
            A dictionary containing the serialized configuration.
        
        Example:
            >>> matcher = AutoSchemaMatcher(encoder, config)
            >>> d = matcher.to_dict()
            >>> d["auto_config"]["enable_compound_matching"]
            True
        
        Validates: Requirement 1.4
        """
        return {
            "auto_config": {
                "tokenizer_config": {
                    "max_ngram_size": self.auto_config.tokenizer_config.max_ngram_size,
                    "stop_words": list(self.auto_config.tokenizer_config.stop_words),
                    "normalize_case": self.auto_config.tokenizer_config.normalize_case,
                    "remove_punctuation": self.auto_config.tokenizer_config.remove_punctuation,
                },
                "match_config": {
                    "role_threshold": self.auto_config.match_config.role_threshold,
                    "value_threshold": self.auto_config.match_config.value_threshold,
                    "compound_bonus": self.auto_config.match_config.compound_bonus,
                    "exact_match_bonus": self.auto_config.match_config.exact_match_bonus,
                },
                "intent_keywords": {
                    "count": list(self.auto_config.intent_keywords.count),
                    "find": list(self.auto_config.intent_keywords.find),
                    "filter": list(self.auto_config.intent_keywords.filter),
                    "similar": list(self.auto_config.intent_keywords.similar),
                },
                "enable_compound_matching": self.auto_config.enable_compound_matching,
                "enable_synonyms": self.auto_config.enable_synonyms,
                "fallback_to_manual": self.auto_config.fallback_to_manual,
                "hybrid_intent_config": self.auto_config.hybrid_intent_config,
            },
            "has_manual_patterns": self.manual_patterns is not None,
        }
    
    def match_query(self, query: str) -> AutoMatchResult:
        """
        Match a query using auto-schema matching.
        
        Orchestrates the full matching pipeline to process a natural language
        query and return a comprehensive match result. The pipeline includes:
        
        1. Check manual patterns first (if configured and fallback_to_manual is True)
        2. Tokenize the query using the internal tokenizer
        3. Match tokens against schema using the internal matcher
        4. Apply compound match preference (if enable_compound_matching is True)
        5. Infer intent using the internal inferrer
        6. Check hybrid mode configuration and route accordingly
        7. Extract parameters using the internal extractor
        8. Compute overall confidence
        9. Detect disambiguation needs
        10. Return AutoMatchResult with all information
        
        Args:
            query: The natural language query string to match against the schema.
                Can be any non-empty string. Empty strings will result in a
                low-confidence result with no matches.
        
        Returns:
            AutoMatchResult containing:
            - query: The original query string
            - intent: The primary inferred intent (highest confidence)
            - parameters: Extracted parameters from the query
            - match_result: Detailed match information from schema matching
            - confidence: Overall confidence score [0.0, 1.0]
            - match_method: "auto", "manual", or "hybrid"
            - disambiguation_needed: True if multiple intents have similar confidence
            - disambiguation_options: List of alternative intents if disambiguation needed
        
        Raises:
            ValueError: If query is empty.
            TypeError: If query is not a string.
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig, Layer, Segment, Role
            >>> 
            >>> # Create encoder config with schema
            >>> config = EncoderConfig(
            ...     dimension=10000,
            ...     seed=42,
            ...     layers=[
            ...         Layer(
            ...             name="vehicle",
            ...             segments=[
            ...                 Segment(
            ...                     name="identity",
            ...                     roles=[Role(name="make"), Role(name="model")]
            ...                 )
            ...             ]
            ...         )
            ...     ]
            ... )
            >>> 
            >>> # Create encoder and matcher
            >>> encoder = Encoder(config)
            >>> matcher = AutoSchemaMatcher(encoder, config)
            >>> 
            >>> # Match a query
            >>> result = matcher.match_query("Find Toyota cars")
            >>> result.query
            'Find Toyota cars'
            >>> result.intent.intent_type
            'find'
            >>> result.match_method
            'auto'
            >>> 0.0 <= result.confidence <= 1.0
            True
            >>> 
            >>> # Query with no intent keywords
            >>> result = matcher.match_query("Toyota cars")
            >>> result.intent.intent_type
            'find'  # Default intent
            >>> result.confidence < 0.5  # Lower confidence without keywords
            True
        
        Notes:
            - Manual patterns take priority over auto-matching when configured
            - The overall confidence combines intent confidence and match quality
            - Disambiguation is triggered when multiple intents have similar confidence
            - Empty queries return a low-confidence result with default intent
            - The match_method indicates which matching strategy was used
            - Hybrid mode allows per-intent configuration of matching method
        
        Validates: Requirements 1-6
        Validates: Requirement 3.6 - THE SDK SHALL provide a match_query() method
        Validates: Requirement 4.1 - THE SDK SHALL infer intent based on query keywords
        Validates: Requirement 5.1 - THE SDK SHALL extract matched values as parameters
        Validates: Requirement 6.2 - WHEN a manual pattern matches, THE SDK SHALL use it
        Validates: Requirement 6.4 - THE SDK SHALL provide a priority system (manual > auto)
        Validates: Requirement 14.4 - THE SDK SHALL support hybrid mode where some intents use manual patterns and others use auto-matching
        """
        # Validate query is a string
        if not isinstance(query, str):
            raise TypeError(
                f"query must be a string, got {type(query).__name__}"
            )
        
        # Validate query is not empty
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Disambiguation threshold: if two intents have confidence within this
        # range of each other, disambiguation is needed
        DISAMBIGUATION_THRESHOLD = 0.15
        
        # Default intent type when no keywords are detected
        DEFAULT_INTENT_TYPE = "find"
        
        # Step 1: Check manual patterns first if configured (non-hybrid mode)
        # Validates: Requirement 6.2, 6.4 - Manual patterns take priority
        # Only do this if hybrid mode is NOT configured (hybrid mode handles routing differently)
        if (self.auto_config.hybrid_intent_config is None and 
            self.auto_config.fallback_to_manual and 
            self.manual_patterns is not None):
            # Try to match using manual patterns
            manual_result = self._match_manual_patterns(query)
            if manual_result is not None:
                # Manual pattern matched - return the result with match_method="manual"
                return manual_result
        
        # Step 2: Tokenize the query
        # Validates: Requirement 2 - Query Tokenization
        tokens = self._tokenizer.tokenize(query)
        
        # Step 3: Match tokens against schema
        # Validates: Requirement 3 - Token-to-Schema Matching
        match_result = self._matcher.match_query(query, self._tokenizer)
        
        # Step 4: Apply compound match preference if enabled
        # Validates: Requirements 10.2, 10.3 - Compound match preference
        if self.auto_config.enable_compound_matching and match_result.token_matches:
            filtered_matches = self._matcher.prefer_compound_matches(match_result.token_matches)
            
            # Update match_result with filtered matches
            # Rebuild the categorized lists based on filtered matches
            filtered_role_matches = [m for m in filtered_matches if m.schema_vector.element_type == "role"]
            filtered_value_matches = [m for m in filtered_matches if m.schema_vector.element_type == "value"]
            filtered_compound_matches = [m for m in filtered_matches if m.match_type == "compound"]
            
            # Create updated MatchResult with filtered matches
            match_result = MatchResult(
                query=match_result.query,
                token_matches=filtered_matches,
                role_matches=filtered_role_matches,
                value_matches=filtered_value_matches,
                compound_matches=filtered_compound_matches,
                all_candidates=match_result.all_candidates  # Keep all candidates for disambiguation
            )
        
        # Step 5: Infer intent from matches and query structure
        # Validates: Requirement 4 - Intent Inference
        inferred_intents = self._inferrer.infer_intent(query, match_result)
        
        # Step 6: Determine primary intent and disambiguation
        # Validates: Requirements 12.1, 12.4 - Disambiguation
        disambiguation_needed = False
        disambiguation_options: List[InferredIntent] = []
        
        if not inferred_intents:
            # No intent keywords found - create a default intent
            # Use "find" as the default intent type with low confidence
            primary_intent = InferredIntent(
                intent_type=DEFAULT_INTENT_TYPE,
                confidence=0.1,  # Low confidence for default intent
                matched_keywords=[],
                supporting_matches=match_result.token_matches
            )
        else:
            # Use the highest confidence intent as primary
            primary_intent = inferred_intents[0]
            
            # Check for disambiguation: multiple intents with similar confidence
            if len(inferred_intents) > 1:
                # Check if the top two intents have similar confidence
                confidence_diff = primary_intent.confidence - inferred_intents[1].confidence
                if confidence_diff < DISAMBIGUATION_THRESHOLD:
                    disambiguation_needed = True
                    # Include all intents with confidence within threshold of the top
                    for intent in inferred_intents:
                        if primary_intent.confidence - intent.confidence < DISAMBIGUATION_THRESHOLD:
                            disambiguation_options.append(intent)
        
        # Step 6.5: Check hybrid mode configuration
        # Validates: Requirement 14.4 - THE SDK SHALL support hybrid mode
        match_method = "auto"  # Default to auto
        
        if self.auto_config.hybrid_intent_config is not None:
            # Hybrid mode is configured - check if this intent should use manual patterns
            intent_type = primary_intent.intent_type
            configured_method = self.auto_config.hybrid_intent_config.get(intent_type)
            
            if configured_method == "manual" and self.manual_patterns is not None:
                # This intent is configured for manual matching
                manual_result = self._match_manual_patterns(query)
                if manual_result is not None:
                    # Manual pattern matched - update match_method to "hybrid"
                    # and return the result with the manual-matched intent
                    return AutoMatchResult(
                        query=manual_result.query,
                        intent=manual_result.intent,
                        parameters=manual_result.parameters,
                        match_result=manual_result.match_result,
                        confidence=manual_result.confidence,
                        match_method="hybrid",  # Hybrid mode was used
                        disambiguation_needed=manual_result.disambiguation_needed,
                        disambiguation_options=manual_result.disambiguation_options
                    )
                # Manual pattern didn't match - fall through to auto-matching
                # but still mark as hybrid since hybrid mode is configured
                match_method = "hybrid"
            elif configured_method == "auto":
                # This intent is explicitly configured for auto-matching
                match_method = "hybrid"
            elif configured_method is None:
                # Intent not in hybrid config - use default fallback behavior
                if self.auto_config.fallback_to_manual and self.manual_patterns is not None:
                    manual_result = self._match_manual_patterns(query)
                    if manual_result is not None:
                        return AutoMatchResult(
                            query=manual_result.query,
                            intent=manual_result.intent,
                            parameters=manual_result.parameters,
                            match_result=manual_result.match_result,
                            confidence=manual_result.confidence,
                            match_method="hybrid",
                            disambiguation_needed=manual_result.disambiguation_needed,
                            disambiguation_options=manual_result.disambiguation_options
                        )
                match_method = "hybrid"
        
        # Step 7: Extract parameters from matches
        # Validates: Requirement 5 - Parameter Extraction
        parameters = self._extractor.extract_parameters(match_result)
        
        # Step 8: Compute overall confidence
        # Overall confidence combines:
        # - Intent confidence (weight: 0.4)
        # - Match quality (weight: 0.4)
        # - Parameter extraction success (weight: 0.2)
        
        intent_confidence = primary_intent.confidence
        
        # Match quality: average similarity of top matches (if any)
        if match_result.token_matches:
            # Use top 3 matches for quality assessment
            top_matches = match_result.token_matches[:3]
            match_quality = sum(m.similarity for m in top_matches) / len(top_matches)
            # Normalize to [0, 1] range (similarity can be > 1 with bonuses)
            match_quality = min(1.0, max(0.0, match_quality))
        else:
            match_quality = 0.0
        
        # Parameter extraction success: ratio of extracted parameters to value matches
        if match_result.value_matches:
            param_success = len(parameters.parameters) / len(match_result.value_matches)
            param_success = min(1.0, param_success)  # Cap at 1.0
        else:
            param_success = 0.0 if not parameters.parameters else 1.0
        
        # Weighted combination
        overall_confidence = (
            0.4 * intent_confidence +
            0.4 * match_quality +
            0.2 * param_success
        )
        
        # Clamp to [0.0, 1.0]
        overall_confidence = max(0.0, min(1.0, overall_confidence))
        
        # Step 9: Create and return AutoMatchResult
        return AutoMatchResult(
            query=query,
            intent=primary_intent,
            parameters=parameters,
            match_result=match_result,
            confidence=overall_confidence,
            match_method=match_method,
            disambiguation_needed=disambiguation_needed,
            disambiguation_options=disambiguation_options
        )
    
    def _match_manual_patterns(self, query: str) -> Optional[AutoMatchResult]:
        """
        Match a query against manual patterns.
        
        Checks if the query matches any of the configured manual patterns.
        Manual patterns take priority over auto-matching when configured.
        
        This method implements the manual pattern fallback mechanism:
        1. Iterate through all patterns in manual_patterns
        2. For each pattern, check if any example phrase matches the query
        3. If a match is found, create an AutoMatchResult with match_method="manual"
        4. If no match is found, return None to fall back to auto-matching
        
        Args:
            query: The natural language query string to match.
        
        Returns:
            AutoMatchResult if a manual pattern matches, None otherwise.
            When a match is found:
            - match_method is set to "manual"
            - intent is set based on the matched pattern's intent_type
            - confidence is set to 1.0 for exact phrase matches, 0.8 for partial matches
            - parameters are extracted from the query_template if available
        
        Example:
            >>> # With manual patterns configured
            >>> nl_config = NLEncoderConfig(patterns=[
            ...     {
            ...         "intent_type": "find_customer",
            ...         "example_phrases": ["find customer", "lookup customer"],
            ...         "query_template": {"operation": "similarity_search"}
            ...     }
            ... ])
            >>> matcher = AutoSchemaMatcher(encoder, config, manual_patterns=nl_config)
            >>> result = matcher._match_manual_patterns("find customer John")
            >>> result.match_method
            'manual'
            >>> result.intent.intent_type
            'find_customer'
        
        Notes:
            - Matching is case-insensitive
            - Partial matches (query contains the phrase) are supported
            - The first matching pattern is used (patterns are checked in order)
            - Returns None if manual_patterns is None or empty
        
        Validates: Requirements 6.1, 6.2, 6.4
        Validates: Requirement 6.1 - THE SDK SHALL support manual pattern configuration alongside auto-matching
        Validates: Requirement 6.2 - WHEN a manual pattern matches, THE SDK SHALL use it instead of auto-matching
        Validates: Requirement 6.4 - THE SDK SHALL provide a priority system (manual > auto)
        """
        # Return None if no manual patterns configured
        if self.manual_patterns is None or not self.manual_patterns.patterns:
            return None
        
        # Normalize query for case-insensitive matching
        query_lower = query.lower().strip()
        
        # Iterate through all patterns to find a match
        for pattern in self.manual_patterns.patterns:
            intent_type = pattern.get("intent_type", "")
            example_phrases = pattern.get("example_phrases", [])
            query_template = pattern.get("query_template", {})
            
            # Check if any example phrase matches the query
            matched_phrase = None
            match_confidence = 0.0
            
            for phrase in example_phrases:
                phrase_lower = phrase.lower().strip()
                
                # Check for exact match (query equals the phrase)
                if query_lower == phrase_lower:
                    matched_phrase = phrase
                    match_confidence = 1.0
                    break
                
                # Check for partial match (query contains the phrase)
                if phrase_lower in query_lower:
                    # Partial match - use lower confidence
                    if matched_phrase is None or len(phrase) > len(matched_phrase):
                        # Prefer longer phrase matches
                        matched_phrase = phrase
                        # Confidence based on how much of the query the phrase covers
                        coverage = len(phrase_lower) / len(query_lower)
                        match_confidence = 0.6 + (0.3 * coverage)  # Range: 0.6 to 0.9
                
                # Check for phrase contains query (query is a subset)
                elif query_lower in phrase_lower:
                    if matched_phrase is None:
                        matched_phrase = phrase
                        # Lower confidence for subset matches
                        coverage = len(query_lower) / len(phrase_lower)
                        match_confidence = 0.5 + (0.3 * coverage)  # Range: 0.5 to 0.8
            
            # If a match was found, create and return the result
            if matched_phrase is not None:
                # Map the manual pattern's intent_type to a valid InferredIntent type
                # Manual patterns may have custom intent types like "find_customer"
                # We need to map these to valid types: count, find, filter, similar
                mapped_intent_type = self._map_manual_intent_type(intent_type)
                
                # Create InferredIntent for the matched pattern
                intent = InferredIntent(
                    intent_type=mapped_intent_type,
                    confidence=match_confidence,
                    matched_keywords=[matched_phrase],
                    supporting_matches=[]
                )
                
                # Create empty MatchResult (no schema matching was done)
                match_result = MatchResult(
                    query=query,
                    token_matches=[],
                    role_matches=[],
                    value_matches=[],
                    compound_matches=[],
                    all_candidates=[]
                )
                
                # Create empty ExtractionResult
                # Note: In a more complete implementation, we could extract
                # parameters from the query based on the query_template slots
                parameters = ExtractionResult(
                    parameters={},
                    multi_value_params={},
                    unmatched_tokens=[]
                )
                
                # Create the AutoMatchResult
                manual_match_result = AutoMatchResult(
                    query=query,
                    intent=intent,
                    parameters=parameters,
                    match_result=match_result,
                    confidence=match_confidence,
                    match_method="manual",
                    disambiguation_needed=False,
                    disambiguation_options=[]
                )
                
                # Check if this manual pattern could be auto-matched
                # and log a deprecation warning if so
                # Validates: Requirement 14.6 - THE SDK SHALL log deprecation warnings
                # for manual patterns that could be auto-matched
                self._check_deprecation_warning(
                    query=query,
                    manual_result=manual_match_result,
                    matched_phrase=matched_phrase,
                    pattern_intent_type=intent_type
                )
                
                return manual_match_result
        
        # No manual pattern matched - return None to fall back to auto-matching
        return None
    
    def _map_manual_intent_type(self, intent_type: str) -> str:
        """
        Map a manual pattern's intent_type to a valid InferredIntent type.
        
        Manual patterns may have custom intent types like "find_customer" or
        "count_orders". This method maps these to the valid intent types
        supported by InferredIntent: count, find, filter, similar.
        
        The mapping logic:
        1. If the intent_type is already valid, return it as-is
        2. If the intent_type contains a valid type as a prefix/suffix, use that
        3. Otherwise, default to "find" as the most common intent
        
        Args:
            intent_type: The intent type from the manual pattern.
        
        Returns:
            A valid intent type string (one of: count, find, filter, similar).
        
        Example:
            >>> matcher._map_manual_intent_type("find")
            'find'
            >>> matcher._map_manual_intent_type("find_customer")
            'find'
            >>> matcher._map_manual_intent_type("count_orders")
            'count'
            >>> matcher._map_manual_intent_type("custom_intent")
            'find'  # Default
        
        Validates: Requirements 6.1, 6.2
        """
        # Valid intent types
        valid_types = {"count", "find", "filter", "similar"}
        
        # Normalize to lowercase
        intent_lower = intent_type.lower()
        
        # If already a valid type, return it
        if intent_lower in valid_types:
            return intent_lower
        
        # Check if any valid type is contained in the intent_type
        # This handles cases like "find_customer", "count_orders", etc.
        for valid_type in valid_types:
            if valid_type in intent_lower:
                return valid_type
        
        # Default to "find" as the most common intent type
        return "find"
    
    def _check_deprecation_warning(
        self,
        query: str,
        manual_result: 'AutoMatchResult',
        matched_phrase: str,
        pattern_intent_type: str,
        deprecation_threshold: float = DEFAULT_DEPRECATION_THRESHOLD
    ) -> None:
        """
        Check if a manual pattern could be auto-matched and log a deprecation warning.
        
        When a manual pattern matches a query, this method runs auto-matching to
        check if the query could also be successfully auto-matched with high
        confidence. If so, it logs a deprecation warning suggesting the manual
        pattern may be unnecessary.
        
        This helps users migrate from manual patterns to auto-matching by
        identifying patterns that are redundant with the auto-matching capability.
        
        Args:
            query: The original query string that was matched.
            manual_result: The AutoMatchResult from manual pattern matching.
            matched_phrase: The manual pattern phrase that matched.
            pattern_intent_type: The intent type from the manual pattern.
            deprecation_threshold: The confidence threshold above which a
                deprecation warning is logged. Default is 0.7.
        
        Example:
            >>> # When a manual pattern matches "find Toyota cars"
            >>> # and auto-matching would also match with 0.85 confidence:
            >>> # A deprecation warning is logged suggesting the manual pattern
            >>> # "find {make} cars" could be replaced with auto-matching.
        
        Notes:
            - This method does not modify the manual_result
            - Warnings are logged using both logging.warning() and warnings.warn()
            - The warning includes the pattern details and auto-match confidence
            - No warning is logged if auto-matching confidence is below threshold
        
        Validates: Requirement 14.6
        Validates: Requirement 14.6 - THE SDK SHALL log deprecation warnings for manual patterns that could be auto-matched
        """
        try:
            # Run auto-matching on the query to check if it would succeed
            # We need to temporarily bypass manual pattern checking to get pure auto-match result
            
            # Step 1: Tokenize the query
            tokens = self._tokenizer.tokenize(query)
            
            # Step 2: Match tokens against schema
            match_result = self._matcher.match_query(query, self._tokenizer)
            
            # Step 3: Apply compound match preference if enabled
            if self.auto_config.enable_compound_matching and match_result.token_matches:
                filtered_matches = self._matcher.prefer_compound_matches(match_result.token_matches)
                
                # Update match_result with filtered matches
                filtered_role_matches = [m for m in filtered_matches if m.schema_vector.element_type == "role"]
                filtered_value_matches = [m for m in filtered_matches if m.schema_vector.element_type == "value"]
                filtered_compound_matches = [m for m in filtered_matches if m.match_type == "compound"]
                
                match_result = MatchResult(
                    query=match_result.query,
                    token_matches=filtered_matches,
                    role_matches=filtered_role_matches,
                    value_matches=filtered_value_matches,
                    compound_matches=filtered_compound_matches,
                    all_candidates=match_result.all_candidates
                )
            
            # Step 4: Infer intent from matches and query structure
            inferred_intents = self._inferrer.infer_intent(query, match_result)
            
            # Step 5: Determine primary intent
            if not inferred_intents:
                # No intent keywords found - use default with low confidence
                primary_intent = InferredIntent(
                    intent_type="find",
                    confidence=0.1,
                    matched_keywords=[],
                    supporting_matches=match_result.token_matches
                )
            else:
                primary_intent = inferred_intents[0]
            
            # Step 6: Extract parameters from matches
            parameters = self._extractor.extract_parameters(match_result)
            
            # Step 7: Compute overall auto-match confidence
            intent_confidence = primary_intent.confidence
            
            # Match quality: average similarity of top matches
            if match_result.token_matches:
                top_matches = match_result.token_matches[:3]
                match_quality = sum(m.similarity for m in top_matches) / len(top_matches)
                match_quality = min(1.0, max(0.0, match_quality))
            else:
                match_quality = 0.0
            
            # Parameter extraction success
            if match_result.value_matches:
                param_success = len(parameters.parameters) / len(match_result.value_matches)
                param_success = min(1.0, param_success)
            else:
                param_success = 0.0 if not parameters.parameters else 1.0
            
            # Weighted combination
            auto_confidence = (
                0.4 * intent_confidence +
                0.4 * match_quality +
                0.2 * param_success
            )
            auto_confidence = max(0.0, min(1.0, auto_confidence))
            
            # Step 8: Check if auto-matching confidence exceeds threshold
            if auto_confidence >= deprecation_threshold:
                # Log deprecation warning
                warning_message = (
                    f"Manual pattern '{matched_phrase}' (intent: {pattern_intent_type}) "
                    f"could be auto-matched with confidence {auto_confidence:.2f}. "
                    f"Consider removing this manual pattern and using auto-matching instead. "
                    f"Query: '{query}'"
                )
                
                # Log using the logging module
                logger.warning(
                    "DeprecationWarning: %s",
                    warning_message
                )
                
                # Also emit a DeprecationWarning using warnings module
                warnings.warn(
                    warning_message,
                    DeprecationWarning,
                    stacklevel=4  # Point to the caller of match_query
                )
        
        except Exception as e:
            # If auto-matching fails for any reason, don't block the manual match
            # Just log a debug message and continue
            logger.debug(
                "Failed to check deprecation warning for query '%s': %s",
                query,
                str(e)
            )
    
    def match_queries(self, queries: List[str]) -> List[AutoMatchResult]:
        """
        Match multiple queries using auto-schema matching in batch.
        
        Processes multiple queries efficiently by sharing schema vectors across
        all queries. This is more efficient than calling match_query() individually
        for each query because:
        
        1. Schema vectors are already cached and shared across all queries
        2. The tokenizer, matcher, inferrer, and extractor are reused
        3. No redundant initialization or vector regeneration
        
        The batch method processes each query through the same pipeline as
        match_query(), ensuring consistent results whether queries are processed
        individually or in batch.
        
        Args:
            queries: A list of natural language query strings to match against
                the schema. Can be empty (returns empty list). Each query must
                be a non-empty string.
        
        Returns:
            A list of AutoMatchResult objects, one for each input query.
            The results are in the same order as the input queries.
            If the input list is empty, returns an empty list.
        
        Raises:
            TypeError: If queries is not a list.
            TypeError: If any element in queries is not a string.
            ValueError: If any query in the list is empty.
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig, Layer, Segment, Role
            >>> 
            >>> # Create encoder config with schema
            >>> config = EncoderConfig(
            ...     dimension=10000,
            ...     seed=42,
            ...     layers=[
            ...         Layer(
            ...             name="vehicle",
            ...             segments=[
            ...                 Segment(
            ...                     name="identity",
            ...                     roles=[Role(name="make"), Role(name="model")]
            ...                 )
            ...             ]
            ...         )
            ...     ]
            ... )
            >>> 
            >>> # Create encoder and matcher
            >>> encoder = Encoder(config)
            >>> matcher = AutoSchemaMatcher(encoder, config)
            >>> 
            >>> # Match multiple queries in batch
            >>> queries = ["Find Toyota cars", "Count Honda vehicles", "Show BMW models"]
            >>> results = matcher.match_queries(queries)
            >>> len(results)
            3
            >>> results[0].query
            'Find Toyota cars'
            >>> results[1].query
            'Count Honda vehicles'
            >>> results[2].query
            'Show BMW models'
            >>> 
            >>> # Each result is an AutoMatchResult
            >>> all(isinstance(r, AutoMatchResult) for r in results)
            True
            >>> 
            >>> # Empty list returns empty list
            >>> matcher.match_queries([])
            []
            >>> 
            >>> # Invalid input raises TypeError
            >>> matcher.match_queries("not a list")
            Traceback (most recent call last):
                ...
            TypeError: queries must be a list, got str
            >>> 
            >>> # Non-string element raises TypeError
            >>> matcher.match_queries(["valid query", 123])
            Traceback (most recent call last):
                ...
            TypeError: queries[1] must be a string, got int
            >>> 
            >>> # Empty query raises ValueError
            >>> matcher.match_queries(["valid query", ""])
            Traceback (most recent call last):
                ...
            ValueError: queries[1] cannot be empty
        
        Notes:
            - Schema vectors are shared across all queries (already cached)
            - Each query is processed independently through the full pipeline
            - Results are guaranteed to be identical to individual match_query() calls
            - The method is more efficient for multiple queries due to shared state
            - Empty input list is handled gracefully (returns empty list)
            - All validation is performed upfront before processing any queries
        
        Validates: Requirement 13.2
        Validates: Requirement 13.2 - THE SDK SHALL support batch processing of multiple queries
        """
        # Validate queries is a list
        if not isinstance(queries, list):
            raise TypeError(
                f"queries must be a list, got {type(queries).__name__}"
            )
        
        # Handle empty list gracefully
        if not queries:
            return []
        
        # Validate all queries upfront before processing
        for i, query in enumerate(queries):
            # Validate each query is a string
            if not isinstance(query, str):
                raise TypeError(
                    f"queries[{i}] must be a string, got {type(query).__name__}"
                )
            
            # Validate each query is not empty
            if not query.strip():
                raise ValueError(f"queries[{i}] cannot be empty")
        
        # Process each query through match_query()
        # Schema vectors are already cached in self._vectorizer and shared
        # across all queries via self._matcher
        results: List[AutoMatchResult] = []
        for query in queries:
            result = self.match_query(query)
            results.append(result)
        
        return results

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        encoder: 'Encoder',
        config: 'EncoderConfig'
    ) -> 'AutoSchemaMatcher':
        """
        Deserialize from .glyphh file.
        
        Creates an AutoSchemaMatcher instance from a serialized dictionary.
        
        Args:
            data: Dictionary containing serialized configuration.
            encoder: The Encoder instance to use.
            config: The EncoderConfig to use.
        
        Returns:
            A new AutoSchemaMatcher instance.
        
        Example:
            >>> data = {"auto_config": {...}}
            >>> matcher = AutoSchemaMatcher.from_dict(data, encoder, config)
        
        Validates: Requirement 1.4
        """
        auto_config_data = data.get("auto_config", {})
        
        # Reconstruct tokenizer config
        tokenizer_data = auto_config_data.get("tokenizer_config", {})
        tokenizer_config = TokenizerConfig(
            max_ngram_size=tokenizer_data.get("max_ngram_size", 3),
            stop_words=set(tokenizer_data.get("stop_words", ["the", "a", "an", "is", "are"])),
            normalize_case=tokenizer_data.get("normalize_case", True),
            remove_punctuation=tokenizer_data.get("remove_punctuation", True),
        )
        
        # Reconstruct match config
        match_data = auto_config_data.get("match_config", {})
        match_config = MatchConfig(
            role_threshold=match_data.get("role_threshold", 0.3),
            value_threshold=match_data.get("value_threshold", 0.3),
            compound_bonus=match_data.get("compound_bonus", 0.1),
            exact_match_bonus=match_data.get("exact_match_bonus", 0.2),
        )
        
        # Reconstruct intent keywords
        keywords_data = auto_config_data.get("intent_keywords", {})
        intent_keywords = IntentKeywords(
            count=set(keywords_data.get("count", ["how many", "count", "number of"])),
            find=set(keywords_data.get("find", ["find", "show", "get", "list", "search"])),
            filter=set(keywords_data.get("filter", ["greater", "less", "between", "more", "fewer"])),
            similar=set(keywords_data.get("similar", ["similar", "like", "related"])),
        )
        
        # Reconstruct hybrid_intent_config (may be None)
        # Validates: Requirement 14.4 - THE SDK SHALL support hybrid mode
        hybrid_intent_config = auto_config_data.get("hybrid_intent_config", None)
        
        # Reconstruct auto config
        auto_config = AutoMatchConfig(
            tokenizer_config=tokenizer_config,
            match_config=match_config,
            intent_keywords=intent_keywords,
            enable_compound_matching=auto_config_data.get("enable_compound_matching", True),
            enable_synonyms=auto_config_data.get("enable_synonyms", True),
            fallback_to_manual=auto_config_data.get("fallback_to_manual", True),
            hybrid_intent_config=hybrid_intent_config,
        )
        
        return cls(encoder, config, auto_config=auto_config)
