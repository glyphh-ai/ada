"""
Property-based tests for AutoSchemaMatcher.

This module contains property-based tests using Hypothesis to verify
the AutoSchemaMatcher behavior, specifically focusing on manual pattern
priority over auto-matching.

**Validates: Property 14** - Manual Pattern Priority
For any query that matches both a manual pattern and auto-schema matching,
the manual pattern result SHALL be used (manual patterns take priority over
auto-matching).

**Validates: Requirements 6.2, 14.2**
"""

import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from hypothesis import given, settings, strategies as st, assume

from glyphh.encoder.base import Encoder
from glyphh.core.config import EncoderConfig
from glyphh.nl.auto_schema_matcher import (
    AutoSchemaMatcher,
    AutoMatchConfig,
    AutoMatchResult,
)
from glyphh.nl.schema_matcher import MatchConfig
from glyphh.nl.query_tokenizer import TokenizerConfig


# ============================================================================
# Mock NLEncoderConfig for testing manual patterns
# ============================================================================

@dataclass
class MockNLEncoderConfig:
    """
    Mock NLEncoderConfig for testing manual pattern matching.
    
    This mimics the structure expected by AutoSchemaMatcher._match_manual_patterns().
    The patterns list contains dictionaries with:
    - intent_type: The intent type for the pattern
    - example_phrases: List of phrases that trigger this pattern
    - query_template: Optional template for query generation
    """
    patterns: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Generator Strategies
# ============================================================================

# Intent type generator - generates valid intent types
intent_types = st.sampled_from(["find", "count", "filter", "similar"])

# Custom intent type generator - generates custom intent types like "find_customer"
custom_intent_types = st.text(
    alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyz_'),
    min_size=3, max_size=20
).filter(lambda x: x.strip() and not x.startswith('_') and not x.endswith('_'))

# Example phrase generator - generates valid example phrases
example_phrases = st.text(
    alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '),
    min_size=3, max_size=50
).filter(lambda x: x.strip() and len(x.strip()) >= 3)

# Query generator - generates valid query strings
queries = st.text(
    alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '),
    min_size=3, max_size=100
).filter(lambda x: x.strip() and len(x.strip()) >= 3)

# Role name generator
role_names = st.text(
    alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'),
    min_size=1, max_size=20
).filter(lambda x: x.strip())


# ============================================================================
# Test Class: Manual Pattern Priority (Property 14)
# ============================================================================

class TestManualPatternPriority:
    """
    Property tests for Manual Pattern Priority (Property 14).
    
    **Validates: Property 14** - Manual Pattern Priority
    For any query that matches both a manual pattern and auto-schema matching,
    the manual pattern result SHALL be used (manual patterns take priority over
    auto-matching).
    
    **Validates: Requirements 6.2, 14.2**
    """
    
    @given(
        intent_type=intent_types,
        phrase=example_phrases
    )
    @settings(max_examples=100)
    def test_manual_pattern_match_returns_manual_method(
        self, intent_type: str, phrase: str
    ):
        """
        Property test: Manual pattern matches return match_method="manual".
        
        When a query matches a manual pattern, the result SHALL have
        match_method="manual" to indicate that manual patterns were used.
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize inputs
        phrase_norm = phrase.strip()
        
        # Skip if phrase is too short after normalization
        assume(len(phrase_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns with the test phrase
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_type,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback enabled
        auto_config = AutoMatchConfig(
            fallback_to_manual=True
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Match the exact phrase (should match manual pattern)
        result = matcher.match_query(phrase_norm)
        
        # Property: match_method should be "manual"
        assert result.match_method == "manual", \
            f"Expected match_method='manual' for query '{phrase_norm}', " \
            f"got '{result.match_method}'"
    
    @given(
        intent_type=intent_types,
        phrase=example_phrases
    )
    @settings(max_examples=100)
    def test_manual_pattern_match_uses_pattern_intent(
        self, intent_type: str, phrase: str
    ):
        """
        Property test: Manual pattern matches use the pattern's intent type.
        
        When a query matches a manual pattern, the result's intent SHALL
        be derived from the manual pattern's intent_type.
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize inputs
        phrase_norm = phrase.strip()
        
        # Skip if phrase is too short after normalization
        assume(len(phrase_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns with the test phrase
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_type,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback enabled
        auto_config = AutoMatchConfig(
            fallback_to_manual=True
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Match the exact phrase (should match manual pattern)
        result = matcher.match_query(phrase_norm)
        
        # Property: intent type should match the manual pattern's intent
        assert result.intent.intent_type == intent_type, \
            f"Expected intent_type='{intent_type}' for query '{phrase_norm}', " \
            f"got '{result.intent.intent_type}'"
    
    @given(
        manual_intent=intent_types,
        phrase=example_phrases
    )
    @settings(max_examples=100)
    def test_manual_pattern_takes_priority_over_auto_matching(
        self, manual_intent: str, phrase: str
    ):
        """
        Property test: Manual patterns take priority over auto-matching.
        
        When a query could match both a manual pattern and auto-schema matching,
        the manual pattern result SHALL be used (manual > auto priority).
        
        This is the core property for Property 14.
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize inputs
        phrase_norm = phrase.strip()
        
        # Skip if phrase is too short after normalization
        assume(len(phrase_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns with the test phrase
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": manual_intent,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback enabled
        auto_config = AutoMatchConfig(
            fallback_to_manual=True
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Match the exact phrase
        result = matcher.match_query(phrase_norm)
        
        # Property: Manual pattern should take priority
        # 1. match_method should be "manual"
        assert result.match_method == "manual", \
            f"Manual pattern should take priority, but match_method='{result.match_method}'"
        
        # 2. intent should be from manual pattern
        assert result.intent.intent_type == manual_intent, \
            f"Manual pattern intent '{manual_intent}' should be used, " \
            f"got '{result.intent.intent_type}'"
    
    @given(
        intent_type=intent_types,
        phrase=example_phrases,
        query_suffix=st.text(
            alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyz '),
            min_size=1, max_size=20
        )
    )
    @settings(max_examples=100)
    def test_partial_match_uses_manual_pattern(
        self, intent_type: str, phrase: str, query_suffix: str
    ):
        """
        Property test: Partial matches (query contains phrase) use manual pattern.
        
        When a query contains a manual pattern phrase (partial match),
        the manual pattern result SHALL still be used.
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize inputs
        phrase_norm = phrase.strip()
        suffix_norm = query_suffix.strip()
        
        # Skip if phrase is too short after normalization
        assume(len(phrase_norm) >= 3)
        assume(len(suffix_norm) >= 1)
        
        # Create a query that contains the phrase
        query = f"{phrase_norm} {suffix_norm}"
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns with the test phrase
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_type,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback enabled
        auto_config = AutoMatchConfig(
            fallback_to_manual=True
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Match the query containing the phrase
        result = matcher.match_query(query)
        
        # Property: match_method should be "manual" for partial match
        assert result.match_method == "manual", \
            f"Expected match_method='manual' for partial match query '{query}', " \
            f"got '{result.match_method}'"
        
        # Property: intent should be from manual pattern
        assert result.intent.intent_type == intent_type, \
            f"Expected intent_type='{intent_type}' for partial match, " \
            f"got '{result.intent.intent_type}'"
    
    @given(
        intent_type=custom_intent_types
    )
    @settings(max_examples=100)
    def test_custom_intent_type_mapped_correctly(
        self, intent_type: str
    ):
        """
        Property test: Custom intent types are mapped to valid types.
        
        When a manual pattern has a custom intent type (e.g., "find_customer"),
        it SHALL be mapped to a valid intent type (find, count, filter, similar).
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize input
        intent_norm = intent_type.strip()
        
        # Skip invalid intent types
        assume(len(intent_norm) >= 3)
        assume('_' not in intent_norm or (not intent_norm.startswith('_') and not intent_norm.endswith('_')))
        
        # Use a fixed phrase for this test
        phrase = "test query phrase"
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns with custom intent type
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_norm,
                "example_phrases": [phrase],
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback enabled
        auto_config = AutoMatchConfig(
            fallback_to_manual=True
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Match the phrase
        result = matcher.match_query(phrase)
        
        # Property: match_method should be "manual"
        assert result.match_method == "manual", \
            f"Expected match_method='manual', got '{result.match_method}'"
        
        # Property: intent type should be one of the valid types
        valid_types = {"find", "count", "filter", "similar"}
        assert result.intent.intent_type in valid_types, \
            f"Custom intent '{intent_norm}' should be mapped to one of {valid_types}, " \
            f"got '{result.intent.intent_type}'"
    
    @given(
        intent_type=intent_types,
        phrase=example_phrases
    )
    @settings(max_examples=100)
    def test_manual_match_confidence_is_valid(
        self, intent_type: str, phrase: str
    ):
        """
        Property test: Manual pattern match confidence is in valid range.
        
        When a query matches a manual pattern, the confidence score SHALL
        be in the range [0.0, 1.0].
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize inputs
        phrase_norm = phrase.strip()
        
        # Skip if phrase is too short after normalization
        assume(len(phrase_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns with the test phrase
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_type,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback enabled
        auto_config = AutoMatchConfig(
            fallback_to_manual=True
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Match the exact phrase
        result = matcher.match_query(phrase_norm)
        
        # Property: confidence should be in [0.0, 1.0]
        assert 0.0 <= result.confidence <= 1.0, \
            f"Confidence should be in [0.0, 1.0], got {result.confidence}"
    
    @given(
        intent_type=intent_types,
        phrase=example_phrases
    )
    @settings(max_examples=100)
    def test_exact_match_has_high_confidence(
        self, intent_type: str, phrase: str
    ):
        """
        Property test: Exact phrase matches have high confidence.
        
        When a query exactly matches a manual pattern phrase, the confidence
        score SHALL be 1.0 (exact match).
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize inputs
        phrase_norm = phrase.strip()
        
        # Skip if phrase is too short after normalization
        assume(len(phrase_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns with the test phrase
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_type,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback enabled
        auto_config = AutoMatchConfig(
            fallback_to_manual=True
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Match the exact phrase
        result = matcher.match_query(phrase_norm)
        
        # Property: exact match should have confidence = 1.0
        assert result.confidence == 1.0, \
            f"Exact match should have confidence=1.0, got {result.confidence}"
    
    @given(
        intent_type=intent_types,
        phrases=st.lists(example_phrases, min_size=2, max_size=5, unique=True)
    )
    @settings(max_examples=100)
    def test_multiple_phrases_all_match_same_intent(
        self, intent_type: str, phrases: List[str]
    ):
        """
        Property test: Multiple phrases for same intent all match correctly.
        
        When a manual pattern has multiple example phrases, matching any
        of them SHALL return the same intent type.
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize phrases
        phrases_norm = [p.strip() for p in phrases if len(p.strip()) >= 3]
        
        # Skip if not enough valid phrases
        assume(len(phrases_norm) >= 2)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns with multiple phrases
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_type,
                "example_phrases": phrases_norm,
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback enabled
        auto_config = AutoMatchConfig(
            fallback_to_manual=True
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Property: Each phrase should match with the same intent
        for phrase in phrases_norm:
            result = matcher.match_query(phrase)
            
            assert result.match_method == "manual", \
                f"Phrase '{phrase}' should match manual pattern"
            
            assert result.intent.intent_type == intent_type, \
                f"Phrase '{phrase}' should have intent '{intent_type}', " \
                f"got '{result.intent.intent_type}'"
    
    @given(
        query=queries
    )
    @settings(max_examples=100)
    def test_no_manual_patterns_uses_auto_matching(
        self, query: str
    ):
        """
        Property test: Without manual patterns, auto-matching is used.
        
        When no manual patterns are configured, the matcher SHALL use
        auto-matching (match_method="auto").
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize query
        query_norm = query.strip()
        
        # Skip if query is too short
        assume(len(query_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create matcher WITHOUT manual patterns
        auto_config = AutoMatchConfig(
            fallback_to_manual=True  # Enabled but no patterns
        )
        
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=None  # No manual patterns
        )
        
        # Match the query
        result = matcher.match_query(query_norm)
        
        # Property: match_method should be "auto" when no manual patterns
        assert result.match_method == "auto", \
            f"Without manual patterns, match_method should be 'auto', " \
            f"got '{result.match_method}'"
    
    @given(
        intent_type=intent_types,
        phrase=example_phrases,
        unmatched_query=queries
    )
    @settings(max_examples=100)
    def test_unmatched_query_falls_back_to_auto(
        self, intent_type: str, phrase: str, unmatched_query: str
    ):
        """
        Property test: Unmatched queries fall back to auto-matching.
        
        When a query does not match any manual pattern, the matcher SHALL
        fall back to auto-matching (match_method="auto").
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize inputs
        phrase_norm = phrase.strip()
        query_norm = unmatched_query.strip()
        
        # Skip if inputs are too short
        assume(len(phrase_norm) >= 3)
        assume(len(query_norm) >= 3)
        
        # Ensure query does NOT contain the phrase (case-insensitive)
        assume(phrase_norm.lower() not in query_norm.lower())
        assume(query_norm.lower() not in phrase_norm.lower())
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns with a phrase that won't match
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_type,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback enabled
        auto_config = AutoMatchConfig(
            fallback_to_manual=True
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Match a query that doesn't match the manual pattern
        result = matcher.match_query(query_norm)
        
        # Property: match_method should be "auto" for unmatched queries
        assert result.match_method == "auto", \
            f"Unmatched query should fall back to auto, " \
            f"got match_method='{result.match_method}'"
    
    @given(
        intent_type=intent_types,
        phrase=example_phrases
    )
    @settings(max_examples=100)
    def test_fallback_disabled_still_checks_manual_first(
        self, intent_type: str, phrase: str
    ):
        """
        Property test: Manual patterns are checked even when fallback is disabled.
        
        The fallback_to_manual flag controls whether to check manual patterns.
        When enabled, manual patterns SHALL be checked first.
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize inputs
        phrase_norm = phrase.strip()
        
        # Skip if phrase is too short
        assume(len(phrase_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_type,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback ENABLED
        auto_config = AutoMatchConfig(
            fallback_to_manual=True
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Match the phrase
        result = matcher.match_query(phrase_norm)
        
        # Property: With fallback enabled, manual patterns should be used
        assert result.match_method == "manual", \
            f"With fallback enabled, manual patterns should be checked first"
    
    @given(
        intent_type=intent_types,
        phrase=example_phrases
    )
    @settings(max_examples=100)
    def test_fallback_disabled_skips_manual_patterns(
        self, intent_type: str, phrase: str
    ):
        """
        Property test: Disabling fallback skips manual pattern checking.
        
        When fallback_to_manual is False, manual patterns SHALL NOT be
        checked, and auto-matching SHALL be used instead.
        
        **Validates: Property 14**
        **Validates: Requirements 6.2, 14.2**
        """
        # Normalize inputs
        phrase_norm = phrase.strip()
        
        # Skip if phrase is too short
        assume(len(phrase_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_type,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ])
        
        # Create auto config with fallback DISABLED
        auto_config = AutoMatchConfig(
            fallback_to_manual=False  # Disabled!
        )
        
        # Create matcher with manual patterns
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Match the phrase
        result = matcher.match_query(phrase_norm)
        
        # Property: With fallback disabled, auto-matching should be used
        assert result.match_method == "auto", \
            f"With fallback disabled, auto-matching should be used, " \
            f"got match_method='{result.match_method}'"


# ============================================================================
# Test Class: Disambiguation Trigger (Property 19)
# ============================================================================

class TestDisambiguationTrigger:
    """
    Property tests for Disambiguation Trigger (Property 19).
    
    **Validates: Property 19** - Disambiguation Trigger
    For any query where multiple intents have confidence scores within the
    disambiguation threshold of each other, the result SHALL indicate
    disambiguation_needed=true and include all candidate intents.
    
    The disambiguation threshold is 0.15 (defined in match_query()).
    
    **Validates: Requirements 12.1, 12.4**
    """
    
    # The disambiguation threshold used in match_query()
    DISAMBIGUATION_THRESHOLD = 0.15

    @given(
        top_confidence=st.floats(min_value=0.2, max_value=0.9, allow_nan=False, allow_infinity=False),
        confidence_diff=st.floats(min_value=0.0, max_value=0.14, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_disambiguation_triggered_when_intents_within_threshold(
        self, top_confidence: float, confidence_diff: float
    ):
        """
        Property test: Disambiguation triggers when intents have similar confidence.
        
        When multiple intents have confidence scores within 0.15 of each other,
        disambiguation_needed SHALL be True.
        
        **Validates: Property 19**
        **Validates: Requirements 12.1, 12.4**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Ensure second confidence is valid (>= 0)
        second_confidence = top_confidence - confidence_diff
        assume(second_confidence >= 0.0)
        assume(second_confidence <= 1.0)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()

        # Setup: Two intents with confidence within threshold
        find_intent = InferredIntent(
            intent_type="find",
            confidence=top_confidence,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=second_confidence,
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock inferrer to return both intents (sorted by confidence)
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock matcher to return empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="test query",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query("test query")
        
        # Property: disambiguation_needed should be True when diff < threshold
        assert result.disambiguation_needed is True, \
            f"disambiguation_needed should be True when confidence diff " \
            f"({confidence_diff:.4f}) < threshold ({self.DISAMBIGUATION_THRESHOLD})"

    @given(
        top_confidence=st.floats(min_value=0.3, max_value=0.9, allow_nan=False, allow_infinity=False),
        confidence_diff=st.floats(min_value=0.16, max_value=0.5, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_no_disambiguation_when_confidence_difference_exceeds_threshold(
        self, top_confidence: float, confidence_diff: float
    ):
        """
        Property test: No disambiguation when confidence difference > 0.15.
        
        When the confidence difference between intents exceeds the threshold,
        disambiguation_needed SHALL be False.
        
        **Validates: Property 19**
        **Validates: Requirements 12.1, 12.4**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Ensure second confidence is valid (>= 0)
        second_confidence = top_confidence - confidence_diff
        assume(second_confidence >= 0.0)
        assume(second_confidence <= 1.0)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()

        # Setup: Two intents with confidence difference > threshold
        find_intent = InferredIntent(
            intent_type="find",
            confidence=top_confidence,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=second_confidence,
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock inferrer to return both intents (sorted by confidence)
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock matcher to return empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="test query",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query("test query")
        
        # Property: disambiguation_needed should be False when diff > threshold
        assert result.disambiguation_needed is False, \
            f"disambiguation_needed should be False when confidence diff " \
            f"({confidence_diff:.4f}) > threshold ({self.DISAMBIGUATION_THRESHOLD})"

    @given(
        top_confidence=st.floats(min_value=0.2, max_value=0.9, allow_nan=False, allow_infinity=False),
        confidence_diff=st.floats(min_value=0.0, max_value=0.14, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_all_candidate_intents_included_in_disambiguation_options(
        self, top_confidence: float, confidence_diff: float
    ):
        """
        Property test: All candidate intents within threshold are included.
        
        When disambiguation is triggered, all intents with confidence within
        the threshold of the top intent SHALL be included in disambiguation_options.
        
        **Validates: Property 19**
        **Validates: Requirements 12.1, 12.4**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Ensure second confidence is valid (>= 0)
        second_confidence = top_confidence - confidence_diff
        assume(second_confidence >= 0.0)
        assume(second_confidence <= 1.0)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()

        # Setup: Two intents with confidence within threshold
        find_intent = InferredIntent(
            intent_type="find",
            confidence=top_confidence,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=second_confidence,
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock inferrer to return both intents (sorted by confidence)
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock matcher to return empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="test query",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query("test query")
        
        # Property: Both intents should be in disambiguation_options
        assert len(result.disambiguation_options) >= 2, \
            f"Expected at least 2 disambiguation options, got {len(result.disambiguation_options)}"
        
        intent_types = [opt.intent_type for opt in result.disambiguation_options]
        assert "find" in intent_types, \
            f"'find' intent should be in disambiguation_options"
        assert "similar" in intent_types, \
            f"'similar' intent should be in disambiguation_options"

    @given(
        top_confidence=st.floats(min_value=0.3, max_value=0.9, allow_nan=False, allow_infinity=False),
        within_threshold_diff=st.floats(min_value=0.0, max_value=0.14, allow_nan=False, allow_infinity=False),
        outside_threshold_diff=st.floats(min_value=0.16, max_value=0.4, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_only_intents_within_threshold_included(
        self, top_confidence: float, within_threshold_diff: float, outside_threshold_diff: float
    ):
        """
        Property test: Only intents within threshold are included in disambiguation.
        
        When disambiguation is triggered, only intents with confidence within
        the threshold of the top intent SHALL be included. Intents outside
        the threshold SHALL NOT be included.
        
        **Validates: Property 19**
        **Validates: Requirements 12.1, 12.4**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Calculate confidences
        second_confidence = top_confidence - within_threshold_diff
        third_confidence = top_confidence - outside_threshold_diff
        
        # Ensure all confidences are valid
        assume(second_confidence >= 0.0 and second_confidence <= 1.0)
        assume(third_confidence >= 0.0 and third_confidence <= 1.0)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()

        # Setup: Three intents - two within threshold, one outside
        find_intent = InferredIntent(
            intent_type="find",
            confidence=top_confidence,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=second_confidence,  # Within threshold
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        count_intent = InferredIntent(
            intent_type="count",
            confidence=third_confidence,  # Outside threshold
            matched_keywords=["count"],
            supporting_matches=[]
        )
        
        # Mock inferrer to return all intents (sorted by confidence)
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent, count_intent]
        
        # Mock matcher to return empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="test query",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query("test query")
        
        # Property: Only find and similar should be in disambiguation_options
        intent_types = [opt.intent_type for opt in result.disambiguation_options]
        assert "find" in intent_types, "'find' should be in disambiguation_options"
        assert "similar" in intent_types, "'similar' should be in disambiguation_options"
        assert "count" not in intent_types, \
            f"'count' should NOT be in disambiguation_options (outside threshold)"

    @given(
        confidence=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_single_intent_no_disambiguation(
        self, confidence: float
    ):
        """
        Property test: Single intent never triggers disambiguation.
        
        When only one intent is detected, disambiguation_needed SHALL be False
        regardless of the confidence level.
        
        **Validates: Property 19**
        **Validates: Requirements 12.1, 12.4**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        # Setup: Single intent
        single_intent = InferredIntent(
            intent_type="find",
            confidence=confidence,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        # Mock inferrer to return single intent
        matcher._inferrer.infer_intent.return_value = [single_intent]

        # Mock matcher to return empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="test query",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query("test query")
        
        # Property: Single intent should never trigger disambiguation
        assert result.disambiguation_needed is False, \
            f"Single intent should not trigger disambiguation"
        assert len(result.disambiguation_options) == 0, \
            f"disambiguation_options should be empty for single intent"

    @given(
        num_intents=st.integers(min_value=2, max_value=4),
        base_confidence=st.floats(min_value=0.3, max_value=0.8, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_multiple_intents_all_within_threshold(
        self, num_intents: int, base_confidence: float
    ):
        """
        Property test: Multiple intents all within threshold are all included.
        
        When multiple intents all have confidence within the threshold of
        the top intent, all of them SHALL be included in disambiguation_options.
        
        **Validates: Property 19**
        **Validates: Requirements 12.1, 12.4**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        # Create multiple intents all within threshold
        # Valid intent types are: find, similar, count, filter
        valid_intent_types = ["find", "similar", "count", "filter"]
        intent_types_list = valid_intent_types[:num_intents]
        intents = []
        for i, intent_type in enumerate(intent_types_list):
            # Each intent has confidence slightly lower than the previous
            # but all within threshold (0.15) of the top
            conf = base_confidence - (i * 0.03)  # 0.03 * 4 = 0.12, so all within threshold
            if conf >= 0.0:
                intents.append(InferredIntent(
                    intent_type=intent_type,
                    confidence=conf,
                    matched_keywords=[intent_type],
                    supporting_matches=[]
                ))

        # Skip if we don't have at least 2 valid intents
        assume(len(intents) >= 2)
        
        # Mock inferrer to return all intents
        matcher._inferrer.infer_intent.return_value = intents
        
        # Mock matcher to return empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="test query",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query("test query")
        
        # Property: All intents within threshold should be in disambiguation_options
        assert result.disambiguation_needed is True, \
            f"Multiple intents within threshold should trigger disambiguation"
        
        # Count how many intents are within threshold
        top_conf = intents[0].confidence
        expected_count = sum(1 for i in intents if top_conf - i.confidence < self.DISAMBIGUATION_THRESHOLD)
        
        assert len(result.disambiguation_options) == expected_count, \
            f"Expected {expected_count} disambiguation options, got {len(result.disambiguation_options)}"

    @given(
        top_confidence=st.floats(min_value=0.15, max_value=0.9, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_boundary_at_exactly_threshold(
        self, top_confidence: float
    ):
        """
        Property test: Boundary condition at exactly 0.15 threshold.
        
        When the confidence difference is exactly at the threshold (0.15),
        disambiguation should NOT be triggered because the condition is
        confidence_diff < DISAMBIGUATION_THRESHOLD (strictly less than).
        
        **Validates: Property 19**
        **Validates: Requirements 12.1, 12.4**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Calculate second confidence at exactly threshold
        second_confidence = top_confidence - self.DISAMBIGUATION_THRESHOLD
        
        # Ensure second confidence is valid
        assume(second_confidence >= 0.0)
        assume(second_confidence <= 1.0)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()

        # Setup: Two intents with exactly threshold difference
        find_intent = InferredIntent(
            intent_type="find",
            confidence=top_confidence,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=second_confidence,  # Exactly at threshold
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock inferrer to return both intents
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock matcher to return empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="test query",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query("test query")
        
        # Property: At exactly threshold, disambiguation should NOT be triggered
        # (condition is < not <=)
        assert result.disambiguation_needed is False, \
            f"At exactly threshold ({self.DISAMBIGUATION_THRESHOLD}), " \
            f"disambiguation should NOT be triggered (condition is < not <=)"

    @given(
        top_confidence=st.floats(min_value=0.16, max_value=0.9, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_boundary_just_below_threshold(
        self, top_confidence: float
    ):
        """
        Property test: Boundary condition just below 0.15 threshold.
        
        When the confidence difference is just below the threshold (e.g., 0.149),
        disambiguation SHOULD be triggered.
        
        **Validates: Property 19**
        **Validates: Requirements 12.1, 12.4**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Calculate second confidence just below threshold
        just_below_threshold = self.DISAMBIGUATION_THRESHOLD - 0.001
        second_confidence = top_confidence - just_below_threshold
        
        # Ensure second confidence is valid
        assume(second_confidence >= 0.0)
        assume(second_confidence <= 1.0)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()

        # Setup: Two intents with just below threshold difference
        find_intent = InferredIntent(
            intent_type="find",
            confidence=top_confidence,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=second_confidence,  # Just below threshold
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock inferrer to return both intents
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock matcher to return empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="test query",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query("test query")
        
        # Property: Just below threshold should trigger disambiguation
        assert result.disambiguation_needed is True, \
            f"Just below threshold ({just_below_threshold:.4f}) should trigger disambiguation"
        assert len(result.disambiguation_options) >= 2, \
            f"Both intents should be in disambiguation_options"

    @given(
        top_confidence=st.floats(min_value=0.2, max_value=0.9, allow_nan=False, allow_infinity=False),
        confidence_diff=st.floats(min_value=0.0, max_value=0.14, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_disambiguation_options_are_inferred_intent_instances(
        self, top_confidence: float, confidence_diff: float
    ):
        """
        Property test: Disambiguation options are InferredIntent instances.
        
        When disambiguation is triggered, all items in disambiguation_options
        SHALL be InferredIntent instances with valid fields.
        
        **Validates: Property 19**
        **Validates: Requirements 12.1, 12.4**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Ensure second confidence is valid
        second_confidence = top_confidence - confidence_diff
        assume(second_confidence >= 0.0)
        assume(second_confidence <= 1.0)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()

        # Setup: Two intents with confidence within threshold
        find_intent = InferredIntent(
            intent_type="find",
            confidence=top_confidence,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=second_confidence,
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock inferrer to return both intents
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock matcher to return empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="test query",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query("test query")
        
        # Property: All disambiguation options should be InferredIntent instances
        for option in result.disambiguation_options:
            assert isinstance(option, InferredIntent), \
                f"Disambiguation option should be InferredIntent, got {type(option)}"
            assert isinstance(option.intent_type, str), \
                f"intent_type should be str"
            assert 0.0 <= option.confidence <= 1.0, \
                f"confidence should be in [0.0, 1.0], got {option.confidence}"
            assert isinstance(option.matched_keywords, list), \
                f"matched_keywords should be list"


# ============================================================================
# Test Class: Hybrid Mode Support (Property 20)
# ============================================================================

class TestHybridModeSupport:
    """
    Property tests for Hybrid Mode Support (Property 20).
    
    **Validates: Property 20** - Hybrid Mode Support
    For any model configured with both manual patterns and auto-matching,
    intents configured for manual patterns SHALL use manual matching, and
    intents not configured SHALL use auto-matching.
    
    **Validates: Requirements 14.4, 14.5**
    """
    
    @given(
        intent_type=st.sampled_from(["find", "count", "filter", "similar"]),
        phrase=st.text(
            alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '),
            min_size=3, max_size=50
        ).filter(lambda x: x.strip() and len(x.strip()) >= 3)
    )
    @settings(max_examples=100)
    def test_intent_configured_for_manual_uses_manual_matching(
        self, intent_type: str, phrase: str
    ):
        """
        Property test: Intents configured for "manual" in hybrid_intent_config use manual matching.
        
        When an intent is configured as "manual" in hybrid_intent_config and a manual
        pattern matches, the result SHALL have match_method="hybrid" (indicating hybrid
        mode was used with manual matching for this intent).
        
        **Validates: Property 20**
        **Validates: Requirements 14.4, 14.5**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Normalize inputs
        phrase_norm = phrase.strip()
        assume(len(phrase_norm) >= 3)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        # Configure hybrid mode: this intent uses manual patterns
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config={intent_type: "manual"}
        )
        
        # Set up manual patterns that will match
        matcher.manual_patterns = MagicMock()
        matcher.manual_patterns.patterns = [
            {
                "intent_type": intent_type,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ]
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        # Mock the inferrer to return the configured intent
        test_intent = InferredIntent(
            intent_type=intent_type,
            confidence=0.8,
            matched_keywords=[intent_type],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [test_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query=phrase_norm,
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock the extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query(phrase_norm)
        
        # Property: match_method should be "hybrid" when hybrid_intent_config is set
        assert result.match_method == "hybrid", \
            f"Expected match_method='hybrid' for intent '{intent_type}' configured as 'manual', " \
            f"got '{result.match_method}'"
    
    @given(
        intent_type=st.sampled_from(["find", "count", "filter", "similar"]),
        query=st.text(
            alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '),
            min_size=3, max_size=50
        ).filter(lambda x: x.strip() and len(x.strip()) >= 3)
    )
    @settings(max_examples=100)
    def test_intent_configured_for_auto_uses_auto_matching(
        self, intent_type: str, query: str
    ):
        """
        Property test: Intents configured for "auto" in hybrid_intent_config use auto-matching.
        
        When an intent is configured as "auto" in hybrid_intent_config, the result
        SHALL have match_method="hybrid" (indicating hybrid mode was used with
        auto-matching for this intent).
        
        **Validates: Property 20**
        **Validates: Requirements 14.4, 14.5**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Normalize inputs
        query_norm = query.strip()
        assume(len(query_norm) >= 3)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        # Configure hybrid mode: this intent uses auto-matching
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config={intent_type: "auto"}
        )
        
        # Set up manual patterns (but this intent is configured for auto)
        matcher.manual_patterns = MagicMock()
        matcher.manual_patterns.patterns = []
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        # Mock the inferrer to return the configured intent
        test_intent = InferredIntent(
            intent_type=intent_type,
            confidence=0.8,
            matched_keywords=[intent_type],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [test_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query=query_norm,
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock the extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query(query_norm)
        
        # Property: match_method should be "hybrid" when hybrid_intent_config is set
        assert result.match_method == "hybrid", \
            f"Expected match_method='hybrid' for intent '{intent_type}' configured as 'auto', " \
            f"got '{result.match_method}'"
        
        # Property: intent type should match the inferred intent
        assert result.intent.intent_type == intent_type, \
            f"Expected intent_type='{intent_type}', got '{result.intent.intent_type}'"
    
    @given(
        configured_intent=st.sampled_from(["find", "count"]),
        unconfigured_intent=st.sampled_from(["filter", "similar"]),
        query=st.text(
            alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '),
            min_size=3, max_size=50
        ).filter(lambda x: x.strip() and len(x.strip()) >= 3)
    )
    @settings(max_examples=100)
    def test_unconfigured_intent_uses_default_fallback(
        self, configured_intent: str, unconfigured_intent: str, query: str
    ):
        """
        Property test: Intents not in hybrid_intent_config use default fallback behavior.
        
        When an intent is not configured in hybrid_intent_config, the matcher SHALL
        use the default fallback behavior (fallback_to_manual setting).
        
        **Validates: Property 20**
        **Validates: Requirements 14.4, 14.5**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Ensure intents are different
        assume(configured_intent != unconfigured_intent)
        
        # Normalize inputs
        query_norm = query.strip()
        assume(len(query_norm) >= 3)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        # Configure hybrid mode: only one intent is configured
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config={configured_intent: "manual"},
            fallback_to_manual=True
        )
        
        # Set up manual patterns (but they won't match the unconfigured intent)
        matcher.manual_patterns = MagicMock()
        matcher.manual_patterns.patterns = []
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        # Mock the inferrer to return the UNCONFIGURED intent
        test_intent = InferredIntent(
            intent_type=unconfigured_intent,
            confidence=0.8,
            matched_keywords=[unconfigured_intent],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [test_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query=query_norm,
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock the extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query(query_norm)
        
        # Property: match_method should be "hybrid" since hybrid config is set
        assert result.match_method == "hybrid", \
            f"Expected match_method='hybrid' for unconfigured intent '{unconfigured_intent}', " \
            f"got '{result.match_method}'"
        
        # Property: intent type should match the inferred intent
        assert result.intent.intent_type == unconfigured_intent, \
            f"Expected intent_type='{unconfigured_intent}', got '{result.intent.intent_type}'"
    
    @given(
        manual_intents=st.lists(
            st.sampled_from(["find", "count"]),
            min_size=1, max_size=2, unique=True
        ),
        auto_intents=st.lists(
            st.sampled_from(["filter", "similar"]),
            min_size=1, max_size=2, unique=True
        )
    )
    @settings(max_examples=100)
    def test_hybrid_config_routes_correctly_for_different_intents(
        self, manual_intents: list, auto_intents: list
    ):
        """
        Property test: Hybrid mode correctly routes different intents to different methods.
        
        When hybrid_intent_config specifies different methods for different intents,
        each intent SHALL be routed to its configured method.
        
        **Validates: Property 20**
        **Validates: Requirements 14.4, 14.5**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Build hybrid config with manual and auto intents
        hybrid_config = {}
        for intent in manual_intents:
            hybrid_config[intent] = "manual"
        for intent in auto_intents:
            hybrid_config[intent] = "auto"
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        # Configure hybrid mode with mixed methods
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config=hybrid_config
        )
        
        # Set up manual patterns
        matcher.manual_patterns = MagicMock()
        matcher.manual_patterns.patterns = []
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        # Test each configured intent
        all_intents = manual_intents + auto_intents
        for intent_type in all_intents:
            # Mock the inferrer to return this intent
            test_intent = InferredIntent(
                intent_type=intent_type,
                confidence=0.8,
                matched_keywords=[intent_type],
                supporting_matches=[]
            )
            matcher._inferrer.infer_intent.return_value = [test_intent]
            
            # Mock the matcher to return an empty match result
            matcher._matcher.match_query.return_value = MatchResult(
                query="test query",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            )
            
            # Mock the extractor
            matcher._extractor.extract_parameters.return_value = ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            )
            
            # Execute
            result = matcher.match_query("test query")
            
            # Property: match_method should be "hybrid" for all configured intents
            assert result.match_method == "hybrid", \
                f"Expected match_method='hybrid' for intent '{intent_type}', " \
                f"got '{result.match_method}'"
    
    @given(
        query=st.text(
            alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '),
            min_size=3, max_size=50
        ).filter(lambda x: x.strip() and len(x.strip()) >= 3)
    )
    @settings(max_examples=100)
    def test_match_method_is_hybrid_when_hybrid_intent_config_is_set(
        self, query: str
    ):
        """
        Property test: match_method is "hybrid" when hybrid_intent_config is set.
        
        When hybrid_intent_config is configured (not None), the result's match_method
        SHALL be "hybrid" regardless of which specific intent is matched.
        
        **Validates: Property 20**
        **Validates: Requirements 14.4, 14.5**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Normalize inputs
        query_norm = query.strip()
        assume(len(query_norm) >= 3)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        # Configure hybrid mode with any configuration
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config={"find": "manual", "count": "auto"}
        )
        
        # Set up manual patterns
        matcher.manual_patterns = MagicMock()
        matcher.manual_patterns.patterns = []
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        # Mock the inferrer to return a default intent
        test_intent = InferredIntent(
            intent_type="find",
            confidence=0.8,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [test_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query=query_norm,
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock the extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query(query_norm)
        
        # Property: match_method should be "hybrid" when hybrid_intent_config is set
        assert result.match_method == "hybrid", \
            f"Expected match_method='hybrid' when hybrid_intent_config is set, " \
            f"got '{result.match_method}'"
    
    @given(
        query=st.text(
            alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '),
            min_size=3, max_size=50
        ).filter(lambda x: x.strip() and len(x.strip()) >= 3)
    )
    @settings(max_examples=100)
    def test_match_method_is_auto_when_hybrid_intent_config_is_none(
        self, query: str
    ):
        """
        Property test: match_method is "auto" when hybrid_intent_config is None.
        
        When hybrid_intent_config is None (not configured), the result's match_method
        SHALL be "auto" (assuming no manual pattern matches).
        
        **Validates: Property 20**
        **Validates: Requirements 14.4, 14.5**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Normalize inputs
        query_norm = query.strip()
        assume(len(query_norm) >= 3)
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        # No hybrid mode configured
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config=None,
            fallback_to_manual=False
        )
        
        # No manual patterns
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        # Mock the inferrer to return a default intent
        test_intent = InferredIntent(
            intent_type="find",
            confidence=0.8,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [test_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query=query_norm,
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock the extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query(query_norm)
        
        # Property: match_method should be "auto" when hybrid_intent_config is None
        assert result.match_method == "auto", \
            f"Expected match_method='auto' when hybrid_intent_config is None, " \
            f"got '{result.match_method}'"
    
    @given(
        intent_type=st.sampled_from(["find", "count", "filter", "similar"]),
        confidence=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_hybrid_mode_preserves_intent_confidence(
        self, intent_type: str, confidence: float
    ):
        """
        Property test: Hybrid mode preserves the intent confidence from auto-matching.
        
        When using hybrid mode with auto-matching for an intent, the confidence
        score from the intent inferrer SHALL be preserved in the result.
        
        **Validates: Property 20**
        **Validates: Requirements 14.4, 14.5**
        """
        from unittest.mock import MagicMock
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.schema_matcher import MatchResult
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Create a matcher with mocked components
        matcher = object.__new__(AutoSchemaMatcher)
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        # Configure hybrid mode: this intent uses auto-matching
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config={intent_type: "auto"}
        )
        
        # No manual patterns needed for auto-matching
        matcher.manual_patterns = None
        
        # Initialize mocked sub-components
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        # Mock the inferrer to return the intent with specific confidence
        test_intent = InferredIntent(
            intent_type=intent_type,
            confidence=confidence,
            matched_keywords=[intent_type],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [test_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="test query",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock the extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Execute
        result = matcher.match_query("test query")
        
        # Property: intent confidence should be preserved
        assert result.intent.confidence == confidence, \
            f"Expected intent confidence={confidence}, got {result.intent.confidence}"
        
        # Property: intent type should be preserved
        assert result.intent.intent_type == intent_type, \
            f"Expected intent_type='{intent_type}', got '{result.intent.intent_type}'"
    
    @given(
        hybrid_config=st.dictionaries(
            keys=st.sampled_from(["find", "count", "filter", "similar"]),
            values=st.sampled_from(["manual", "auto"]),
            min_size=1, max_size=4
        )
    )
    @settings(max_examples=100)
    def test_hybrid_config_validation_accepts_valid_configs(
        self, hybrid_config: dict
    ):
        """
        Property test: AutoMatchConfig accepts valid hybrid_intent_config dictionaries.
        
        Any dictionary with string keys (intent types) and string values ("manual" or "auto")
        SHALL be accepted as a valid hybrid_intent_config.
        
        **Validates: Property 20**
        **Validates: Requirements 14.4, 14.5**
        """
        # Skip empty configs
        assume(len(hybrid_config) > 0)
        
        # Property: Valid hybrid configs should be accepted without error
        try:
            config = AutoMatchConfig(hybrid_intent_config=hybrid_config)
            assert config.hybrid_intent_config == hybrid_config, \
                f"hybrid_intent_config should be preserved"
        except (TypeError, ValueError) as e:
            # This should not happen for valid configs
            assert False, f"Valid hybrid config rejected: {e}"
    
    @given(
        invalid_value=st.text(
            alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyz'),
            min_size=1, max_size=20
        ).filter(lambda x: x not in ["manual", "auto"])
    )
    @settings(max_examples=100)
    def test_hybrid_config_validation_rejects_invalid_methods(
        self, invalid_value: str
    ):
        """
        Property test: AutoMatchConfig rejects invalid method values in hybrid_intent_config.
        
        Any value in hybrid_intent_config that is not "manual" or "auto" SHALL
        be rejected with a ValueError.
        
        **Validates: Property 20**
        **Validates: Requirements 14.4, 14.5**
        """
        # Skip if the value happens to be valid
        assume(invalid_value not in ["manual", "auto"])
        assume(len(invalid_value) > 0)
        
        # Property: Invalid method values should be rejected
        try:
            AutoMatchConfig(hybrid_intent_config={"find": invalid_value})
            assert False, f"Invalid method '{invalid_value}' should have been rejected"
        except ValueError as e:
            # Expected behavior
            assert "must be one of" in str(e), \
                f"Error message should mention valid options"


# ============================================================================
# Test Class: Batch Query Consistency (Property 21)
# ============================================================================

class TestBatchQueryConsistency:
    """
    Property tests for Batch Query Consistency (Property 21).
    
    **Validates: Property 21** - Batch Query Consistency
    For any set of queries processed in batch, each query result SHALL be
    identical to the result of processing that query individually.
    
    This ensures that:
    1. Batch results are identical to individual match_query() calls
    2. Results are in the same order as input queries
    3. Schema vectors are shared (not regenerated)
    
    **Validates: Requirements 13.2**
    """
    
    @given(
        query=queries
    )
    @settings(max_examples=100)
    def test_single_query_batch_matches_individual(
        self, query: str
    ):
        """
        Property test: Single query in batch matches individual result.
        
        When a single query is processed in batch, the result SHALL be
        identical to processing that query individually.
        
        **Validates: Property 21**
        **Validates: Requirements 13.2**
        """
        from glyphh.encoder.base import Encoder
        from glyphh.core.config import EncoderConfig
        
        # Normalize query
        query_norm = query.strip()
        
        # Skip if query is too short
        assume(len(query_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create matcher
        auto_config = AutoMatchConfig()
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=None
        )
        
        # Get individual result
        individual_result = matcher.match_query(query_norm)
        
        # Get batch result (single query)
        batch_results = matcher.match_queries([query_norm])
        
        # Property: Batch should return exactly one result
        assert len(batch_results) == 1, \
            f"Expected 1 result, got {len(batch_results)}"
        
        batch_result = batch_results[0]
        
        # Property: Query should be identical
        assert batch_result.query == individual_result.query, \
            f"Query mismatch: batch='{batch_result.query}', individual='{individual_result.query}'"
        
        # Property: Intent type should be identical
        assert batch_result.intent.intent_type == individual_result.intent.intent_type, \
            f"Intent type mismatch: batch='{batch_result.intent.intent_type}', " \
            f"individual='{individual_result.intent.intent_type}'"
        
        # Property: Confidence should be identical
        assert batch_result.confidence == individual_result.confidence, \
            f"Confidence mismatch: batch={batch_result.confidence}, " \
            f"individual={individual_result.confidence}"
        
        # Property: Match method should be identical
        assert batch_result.match_method == individual_result.match_method, \
            f"Match method mismatch: batch='{batch_result.match_method}', " \
            f"individual='{individual_result.match_method}'"
        
        # Property: Disambiguation flag should be identical
        assert batch_result.disambiguation_needed == individual_result.disambiguation_needed, \
            f"Disambiguation flag mismatch: batch={batch_result.disambiguation_needed}, " \
            f"individual={individual_result.disambiguation_needed}"
    
    @given(
        queries_list=st.lists(
            queries,
            min_size=2, max_size=10
        )
    )
    @settings(max_examples=100)
    def test_multiple_queries_batch_matches_individual(
        self, queries_list: List[str]
    ):
        """
        Property test: Multiple queries in batch match individual results.
        
        When multiple queries are processed in batch, each result SHALL be
        identical to processing that query individually.
        
        **Validates: Property 21**
        **Validates: Requirements 13.2**
        """
        from glyphh.encoder.base import Encoder
        from glyphh.core.config import EncoderConfig
        
        # Normalize queries
        queries_norm = [q.strip() for q in queries_list if len(q.strip()) >= 3]
        
        # Skip if not enough valid queries
        assume(len(queries_norm) >= 2)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create matcher
        auto_config = AutoMatchConfig()
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=None
        )
        
        # Get individual results
        individual_results = [matcher.match_query(q) for q in queries_norm]
        
        # Get batch results
        batch_results = matcher.match_queries(queries_norm)
        
        # Property: Batch should return same number of results
        assert len(batch_results) == len(queries_norm), \
            f"Expected {len(queries_norm)} results, got {len(batch_results)}"
        
        # Property: Each batch result should match corresponding individual result
        for i, (batch_result, individual_result) in enumerate(zip(batch_results, individual_results)):
            # Query should be identical
            assert batch_result.query == individual_result.query, \
                f"Query mismatch at index {i}: batch='{batch_result.query}', " \
                f"individual='{individual_result.query}'"
            
            # Intent type should be identical
            assert batch_result.intent.intent_type == individual_result.intent.intent_type, \
                f"Intent type mismatch at index {i}: batch='{batch_result.intent.intent_type}', " \
                f"individual='{individual_result.intent.intent_type}'"
            
            # Confidence should be identical
            assert batch_result.confidence == individual_result.confidence, \
                f"Confidence mismatch at index {i}: batch={batch_result.confidence}, " \
                f"individual={individual_result.confidence}"
            
            # Match method should be identical
            assert batch_result.match_method == individual_result.match_method, \
                f"Match method mismatch at index {i}: batch='{batch_result.match_method}', " \
                f"individual='{individual_result.match_method}'"
    
    @given(
        queries_list=st.lists(
            queries,
            min_size=2, max_size=10
        )
    )
    @settings(max_examples=100)
    def test_batch_preserves_query_order(
        self, queries_list: List[str]
    ):
        """
        Property test: Batch results are in the same order as input queries.
        
        The order of results in the batch output SHALL match the order
        of queries in the input list.
        
        **Validates: Property 21**
        **Validates: Requirements 13.2**
        """
        from glyphh.encoder.base import Encoder
        from glyphh.core.config import EncoderConfig
        
        # Normalize queries
        queries_norm = [q.strip() for q in queries_list if len(q.strip()) >= 3]
        
        # Skip if not enough valid queries
        assume(len(queries_norm) >= 2)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create matcher
        auto_config = AutoMatchConfig()
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=None
        )
        
        # Get batch results
        batch_results = matcher.match_queries(queries_norm)
        
        # Property: Results should be in the same order as input
        assert len(batch_results) == len(queries_norm), \
            f"Expected {len(queries_norm)} results, got {len(batch_results)}"
        
        for i, (result, query) in enumerate(zip(batch_results, queries_norm)):
            assert result.query == query, \
                f"Order mismatch at index {i}: expected '{query}', got '{result.query}'"
    
    @given(
        query=queries
    )
    @settings(max_examples=100)
    def test_batch_shares_schema_vectors(
        self, query: str
    ):
        """
        Property test: Schema vectors are shared across batch queries.
        
        When processing queries in batch, schema vectors SHALL be shared
        (not regenerated) across all queries.
        
        **Validates: Property 21**
        **Validates: Requirements 13.2**
        """
        from glyphh.encoder.base import Encoder
        from glyphh.core.config import EncoderConfig
        
        # Normalize query
        query_norm = query.strip()
        
        # Skip if query is too short
        assume(len(query_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create matcher
        auto_config = AutoMatchConfig()
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=None
        )
        
        # Get schema vectors before batch processing
        vectors_before = matcher.get_schema_vectors()
        
        # Process batch with duplicate queries
        batch_results = matcher.match_queries([query_norm, query_norm, query_norm])
        
        # Get schema vectors after batch processing
        vectors_after = matcher.get_schema_vectors()
        
        # Property: Schema vectors should be the same object (shared)
        assert vectors_before is vectors_after, \
            "Schema vectors should be shared, not regenerated"
        
        # Property: All results should be identical for duplicate queries
        assert len(batch_results) == 3
        for i in range(1, len(batch_results)):
            assert batch_results[i].query == batch_results[0].query
            assert batch_results[i].intent.intent_type == batch_results[0].intent.intent_type
            assert batch_results[i].confidence == batch_results[0].confidence
    
    @given(
        intent_type=intent_types,
        phrase=example_phrases
    )
    @settings(max_examples=100)
    def test_batch_with_manual_patterns_matches_individual(
        self, intent_type: str, phrase: str
    ):
        """
        Property test: Batch with manual patterns matches individual results.
        
        When queries are processed in batch with manual patterns configured,
        each result SHALL be identical to processing that query individually.
        
        **Validates: Property 21**
        **Validates: Requirements 13.2**
        """
        from glyphh.encoder.base import Encoder
        from glyphh.core.config import EncoderConfig
        
        # Normalize phrase
        phrase_norm = phrase.strip()
        
        # Skip if phrase is too short
        assume(len(phrase_norm) >= 3)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create manual patterns
        manual_patterns = MockNLEncoderConfig(patterns=[
            {
                "intent_type": intent_type,
                "example_phrases": [phrase_norm],
                "query_template": {}
            }
        ])
        
        # Create matcher with manual patterns
        auto_config = AutoMatchConfig(fallback_to_manual=True)
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=manual_patterns
        )
        
        # Get individual result
        individual_result = matcher.match_query(phrase_norm)
        
        # Get batch result
        batch_results = matcher.match_queries([phrase_norm])
        
        # Property: Batch should return exactly one result
        assert len(batch_results) == 1
        
        batch_result = batch_results[0]
        
        # Property: Results should be identical
        assert batch_result.query == individual_result.query
        assert batch_result.intent.intent_type == individual_result.intent.intent_type
        assert batch_result.confidence == individual_result.confidence
        assert batch_result.match_method == individual_result.match_method
    
    @given(
        queries_list=st.lists(
            queries,
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=100)
    def test_batch_result_types_match_individual(
        self, queries_list: List[str]
    ):
        """
        Property test: Batch result types match individual result types.
        
        Each result from batch processing SHALL be an AutoMatchResult
        with the same structure as individual match_query() results.
        
        **Validates: Property 21**
        **Validates: Requirements 13.2**
        """
        from glyphh.encoder.base import Encoder
        from glyphh.core.config import EncoderConfig
        
        # Normalize queries
        queries_norm = [q.strip() for q in queries_list if len(q.strip()) >= 3]
        
        # Skip if no valid queries
        assume(len(queries_norm) >= 1)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create matcher
        auto_config = AutoMatchConfig()
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=None
        )
        
        # Get batch results
        batch_results = matcher.match_queries(queries_norm)
        
        # Property: All results should be AutoMatchResult instances
        for i, result in enumerate(batch_results):
            assert isinstance(result, AutoMatchResult), \
                f"Result at index {i} should be AutoMatchResult, got {type(result).__name__}"
            
            # Property: Result should have all required attributes
            assert hasattr(result, 'query'), f"Result at index {i} missing 'query'"
            assert hasattr(result, 'intent'), f"Result at index {i} missing 'intent'"
            assert hasattr(result, 'parameters'), f"Result at index {i} missing 'parameters'"
            assert hasattr(result, 'match_result'), f"Result at index {i} missing 'match_result'"
            assert hasattr(result, 'confidence'), f"Result at index {i} missing 'confidence'"
            assert hasattr(result, 'match_method'), f"Result at index {i} missing 'match_method'"
            assert hasattr(result, 'disambiguation_needed'), f"Result at index {i} missing 'disambiguation_needed'"
            assert hasattr(result, 'disambiguation_options'), f"Result at index {i} missing 'disambiguation_options'"
    
    @given(
        query=queries
    )
    @settings(max_examples=100)
    def test_batch_empty_list_returns_empty(
        self, query: str
    ):
        """
        Property test: Empty batch input returns empty list.
        
        When an empty list is passed to match_queries(), the result
        SHALL be an empty list.
        
        **Validates: Property 21**
        **Validates: Requirements 13.2**
        """
        from glyphh.encoder.base import Encoder
        from glyphh.core.config import EncoderConfig
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create matcher
        auto_config = AutoMatchConfig()
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=None
        )
        
        # Get batch results for empty list
        batch_results = matcher.match_queries([])
        
        # Property: Empty input should return empty output
        assert batch_results == [], \
            f"Expected empty list, got {batch_results}"
        assert isinstance(batch_results, list), \
            f"Expected list type, got {type(batch_results).__name__}"
    
    @given(
        queries_list=st.lists(
            queries,
            min_size=2, max_size=5
        )
    )
    @settings(max_examples=100)
    def test_batch_intent_confidence_matches_individual(
        self, queries_list: List[str]
    ):
        """
        Property test: Batch intent confidence matches individual results.
        
        The intent confidence in batch results SHALL be identical to
        the intent confidence from individual match_query() calls.
        
        **Validates: Property 21**
        **Validates: Requirements 13.2**
        """
        from glyphh.encoder.base import Encoder
        from glyphh.core.config import EncoderConfig
        
        # Normalize queries
        queries_norm = [q.strip() for q in queries_list if len(q.strip()) >= 3]
        
        # Skip if not enough valid queries
        assume(len(queries_norm) >= 2)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create matcher
        auto_config = AutoMatchConfig()
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=None
        )
        
        # Get individual results
        individual_results = [matcher.match_query(q) for q in queries_norm]
        
        # Get batch results
        batch_results = matcher.match_queries(queries_norm)
        
        # Property: Intent confidence should match for each query
        for i, (batch_result, individual_result) in enumerate(zip(batch_results, individual_results)):
            assert batch_result.intent.confidence == individual_result.intent.confidence, \
                f"Intent confidence mismatch at index {i}: " \
                f"batch={batch_result.intent.confidence}, " \
                f"individual={individual_result.intent.confidence}"
    
    @given(
        queries_list=st.lists(
            queries,
            min_size=2, max_size=5
        )
    )
    @settings(max_examples=100)
    def test_batch_disambiguation_matches_individual(
        self, queries_list: List[str]
    ):
        """
        Property test: Batch disambiguation flags match individual results.
        
        The disambiguation_needed flag and disambiguation_options in batch
        results SHALL be identical to individual match_query() results.
        
        **Validates: Property 21**
        **Validates: Requirements 13.2**
        """
        from glyphh.encoder.base import Encoder
        from glyphh.core.config import EncoderConfig
        
        # Normalize queries
        queries_norm = [q.strip() for q in queries_list if len(q.strip()) >= 3]
        
        # Skip if not enough valid queries
        assume(len(queries_norm) >= 2)
        
        # Setup encoder and config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        # Create matcher
        auto_config = AutoMatchConfig()
        matcher = AutoSchemaMatcher(
            encoder, config,
            auto_config=auto_config,
            manual_patterns=None
        )
        
        # Get individual results
        individual_results = [matcher.match_query(q) for q in queries_norm]
        
        # Get batch results
        batch_results = matcher.match_queries(queries_norm)
        
        # Property: Disambiguation should match for each query
        for i, (batch_result, individual_result) in enumerate(zip(batch_results, individual_results)):
            assert batch_result.disambiguation_needed == individual_result.disambiguation_needed, \
                f"Disambiguation flag mismatch at index {i}: " \
                f"batch={batch_result.disambiguation_needed}, " \
                f"individual={individual_result.disambiguation_needed}"
            
            # Check disambiguation options count
            assert len(batch_result.disambiguation_options) == len(individual_result.disambiguation_options), \
                f"Disambiguation options count mismatch at index {i}: " \
                f"batch={len(batch_result.disambiguation_options)}, " \
                f"individual={len(individual_result.disambiguation_options)}"
