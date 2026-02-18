"""
Unit tests for the AutoSchemaMatcher hybrid mode configuration.

Tests the hybrid mode functionality where some intents use manual patterns
and others use auto-matching.

Validates: Requirement 14.4
- THE SDK SHALL support hybrid mode where some intents use manual patterns
  and others use auto-matching
"""

import pytest
from unittest.mock import MagicMock

from glyphh.nl.auto_schema_matcher import (
    AutoSchemaMatcher,
    AutoMatchConfig,
    AutoMatchResult,
)
from glyphh.nl.query_tokenizer import TokenizerConfig
from glyphh.nl.schema_matcher import MatchConfig, MatchResult
from glyphh.nl.intent_inferrer import IntentKeywords, InferredIntent
from glyphh.nl.parameter_extractor import ExtractionResult


class TestHybridModeConfiguration:
    """Tests for hybrid_intent_config in AutoMatchConfig.
    
    Validates: Requirement 14.4
    """
    
    def test_hybrid_intent_config_default_is_none(self):
        """Test that hybrid_intent_config defaults to None.
        
        Validates: Requirement 14.4
        """
        config = AutoMatchConfig()
        assert config.hybrid_intent_config is None
    
    def test_hybrid_intent_config_accepts_valid_dict(self):
        """Test that hybrid_intent_config accepts a valid dictionary.
        
        Validates: Requirement 14.4
        """
        hybrid_config = {
            "find": "manual",
            "count": "auto",
            "similar": "auto"
        }
        config = AutoMatchConfig(hybrid_intent_config=hybrid_config)
        
        assert config.hybrid_intent_config is not None
        assert config.hybrid_intent_config["find"] == "manual"
        assert config.hybrid_intent_config["count"] == "auto"
        assert config.hybrid_intent_config["similar"] == "auto"
    
    def test_hybrid_intent_config_rejects_invalid_method(self):
        """Test that hybrid_intent_config rejects invalid matching methods.
        
        Validates: Requirement 14.4
        """
        with pytest.raises(ValueError) as exc_info:
            AutoMatchConfig(hybrid_intent_config={"find": "invalid"})
        
        assert "must be one of" in str(exc_info.value)
        assert "manual" in str(exc_info.value)
        assert "auto" in str(exc_info.value)
    
    def test_hybrid_intent_config_rejects_non_string_keys(self):
        """Test that hybrid_intent_config rejects non-string keys.
        
        Validates: Requirement 14.4
        """
        with pytest.raises(TypeError) as exc_info:
            AutoMatchConfig(hybrid_intent_config={123: "manual"})
        
        assert "keys must be strings" in str(exc_info.value)
    
    def test_hybrid_intent_config_rejects_non_string_values(self):
        """Test that hybrid_intent_config rejects non-string values.
        
        Validates: Requirement 14.4
        """
        with pytest.raises(TypeError) as exc_info:
            AutoMatchConfig(hybrid_intent_config={"find": 123})
        
        assert "values must be strings" in str(exc_info.value)
    
    def test_hybrid_intent_config_rejects_non_dict(self):
        """Test that hybrid_intent_config rejects non-dictionary values.
        
        Validates: Requirement 14.4
        """
        with pytest.raises(TypeError) as exc_info:
            AutoMatchConfig(hybrid_intent_config="not a dict")
        
        assert "must be a dictionary or None" in str(exc_info.value)
    
    def test_hybrid_intent_config_in_repr(self):
        """Test that hybrid_intent_config is included in repr.
        
        Validates: Requirement 14.4
        """
        config = AutoMatchConfig(hybrid_intent_config={"find": "manual"})
        repr_str = repr(config)
        
        assert "hybrid_intent_config" in repr_str
        assert "find" in repr_str
        assert "manual" in repr_str


class TestHybridModeMatching:
    """Tests for hybrid mode matching in AutoSchemaMatcher.match_query().
    
    Validates: Requirement 14.4
    """
    
    @pytest.fixture
    def matcher_with_mocked_components(self):
        """Create an AutoSchemaMatcher with mocked internal components."""
        # Create a matcher instance without calling __init__
        matcher = object.__new__(AutoSchemaMatcher)
        
        # Set up required attributes
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize sub-components with mocks
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.prefer_compound_matches.return_value = []
        
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        return matcher
    
    def test_hybrid_mode_uses_manual_for_configured_intent(self, matcher_with_mocked_components):
        """Test that hybrid mode uses manual patterns for intents configured as 'manual'.
        
        Validates: Requirement 14.4
        """
        matcher = matcher_with_mocked_components
        
        # Configure hybrid mode: "find" uses manual patterns
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config={"find": "manual", "count": "auto"}
        )
        
        # Set up manual patterns
        matcher.manual_patterns = MagicMock()
        matcher.manual_patterns.patterns = [
            {
                "intent_type": "find",
                "example_phrases": ["find customer"],
                "query_template": {}
            }
        ]
        
        # Mock the inferrer to return a "find" intent
        find_intent = InferredIntent(
            intent_type="find",
            confidence=0.8,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [find_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="find customer John",
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
        result = matcher.match_query("find customer John")
        
        # Verify: Should use hybrid mode and match_method should be "hybrid"
        assert result.match_method == "hybrid"
    
    def test_hybrid_mode_uses_auto_for_configured_intent(self, matcher_with_mocked_components):
        """Test that hybrid mode uses auto-matching for intents configured as 'auto'.
        
        Validates: Requirement 14.4
        """
        matcher = matcher_with_mocked_components
        
        # Configure hybrid mode: "count" uses auto-matching
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config={"find": "manual", "count": "auto"}
        )
        
        # Set up manual patterns (but count is configured for auto)
        matcher.manual_patterns = MagicMock()
        matcher.manual_patterns.patterns = [
            {
                "intent_type": "find",
                "example_phrases": ["find customer"],
                "query_template": {}
            }
        ]
        
        # Mock the inferrer to return a "count" intent
        count_intent = InferredIntent(
            intent_type="count",
            confidence=0.8,
            matched_keywords=["count"],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [count_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="count customers",
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
        result = matcher.match_query("count customers")
        
        # Verify: Should use hybrid mode with auto-matching
        assert result.match_method == "hybrid"
        assert result.intent.intent_type == "count"
    
    def test_hybrid_mode_fallback_for_unconfigured_intent(self, matcher_with_mocked_components):
        """Test that hybrid mode uses fallback behavior for intents not in config.
        
        Validates: Requirement 14.4
        """
        matcher = matcher_with_mocked_components
        
        # Configure hybrid mode: only "find" is configured
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config={"find": "manual"},
            fallback_to_manual=True
        )
        
        # Set up manual patterns
        matcher.manual_patterns = MagicMock()
        matcher.manual_patterns.patterns = []  # No patterns match
        
        # Mock the inferrer to return a "similar" intent (not in hybrid config)
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=0.8,
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [similar_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="similar to Toyota",
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
        result = matcher.match_query("similar to Toyota")
        
        # Verify: Should use hybrid mode (since hybrid config is set)
        assert result.match_method == "hybrid"
        assert result.intent.intent_type == "similar"
    
    def test_no_hybrid_mode_when_config_is_none(self, matcher_with_mocked_components):
        """Test that match_method is 'auto' when hybrid_intent_config is None.
        
        Validates: Requirement 14.4
        """
        matcher = matcher_with_mocked_components
        
        # No hybrid config (default)
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config=None,
            fallback_to_manual=False
        )
        matcher.manual_patterns = None
        
        # Mock the inferrer to return a "find" intent
        find_intent = InferredIntent(
            intent_type="find",
            confidence=0.8,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [find_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="find Toyota",
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
        result = matcher.match_query("find Toyota")
        
        # Verify: Should use auto mode (no hybrid config)
        assert result.match_method == "auto"
    
    def test_hybrid_mode_manual_pattern_match_returns_hybrid_method(self, matcher_with_mocked_components):
        """Test that when manual pattern matches in hybrid mode, match_method is 'hybrid'.
        
        Validates: Requirement 14.4
        """
        matcher = matcher_with_mocked_components
        
        # Configure hybrid mode: "find" uses manual patterns
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config={"find": "manual"}
        )
        
        # Set up manual patterns that will match
        matcher.manual_patterns = MagicMock()
        matcher.manual_patterns.patterns = [
            {
                "intent_type": "find",
                "example_phrases": ["find customer"],
                "query_template": {}
            }
        ]
        
        # Mock the inferrer to return a "find" intent
        find_intent = InferredIntent(
            intent_type="find",
            confidence=0.8,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        matcher._inferrer.infer_intent.return_value = [find_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="find customer",
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
        result = matcher.match_query("find customer")
        
        # Verify: Should use hybrid mode
        assert result.match_method == "hybrid"


class TestHybridModeSerialization:
    """Tests for hybrid mode serialization/deserialization.
    
    Validates: Requirement 14.4
    """
    
    @pytest.fixture
    def matcher_with_mocked_components(self):
        """Create an AutoSchemaMatcher with mocked internal components."""
        # Create a matcher instance without calling __init__
        matcher = object.__new__(AutoSchemaMatcher)
        
        # Set up required attributes
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        matcher.auto_config = AutoMatchConfig(
            hybrid_intent_config={"find": "manual", "count": "auto"}
        )
        matcher.manual_patterns = None
        
        # Initialize sub-components with mocks
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        return matcher
    
    def test_to_dict_includes_hybrid_intent_config(self, matcher_with_mocked_components):
        """Test that to_dict() includes hybrid_intent_config.
        
        Validates: Requirement 14.4
        """
        matcher = matcher_with_mocked_components
        
        result = matcher.to_dict()
        
        assert "auto_config" in result
        assert "hybrid_intent_config" in result["auto_config"]
        assert result["auto_config"]["hybrid_intent_config"] == {
            "find": "manual",
            "count": "auto"
        }
    
    def test_to_dict_includes_none_hybrid_intent_config(self):
        """Test that to_dict() includes None for hybrid_intent_config when not set.
        
        Validates: Requirement 14.4
        """
        # Create a matcher instance without calling __init__
        matcher = object.__new__(AutoSchemaMatcher)
        
        # Set up required attributes
        matcher.encoder = MagicMock()
        matcher.config = MagicMock()
        matcher.auto_config = AutoMatchConfig()  # Default, no hybrid config
        matcher.manual_patterns = None
        
        result = matcher.to_dict()
        
        assert "auto_config" in result
        assert "hybrid_intent_config" in result["auto_config"]
        assert result["auto_config"]["hybrid_intent_config"] is None


class TestAutoMatchResultHybridMethod:
    """Tests for AutoMatchResult with hybrid match_method.
    
    Validates: Requirement 14.4
    """
    
    def test_auto_match_result_accepts_hybrid_method(self):
        """Test that AutoMatchResult accepts 'hybrid' as match_method.
        
        Validates: Requirement 14.4
        """
        intent = InferredIntent(
            intent_type="find",
            confidence=0.8,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        result = AutoMatchResult(
            query="find customer",
            intent=intent,
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="find customer",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.8,
            match_method="hybrid",
            disambiguation_needed=False,
            disambiguation_options=[]
        )
        
        assert result.match_method == "hybrid"
        assert result.is_hybrid_matched() is True
        assert result.is_auto_matched() is False
        assert result.is_manual_matched() is False
    
    def test_auto_match_result_to_dict_includes_hybrid_method(self):
        """Test that to_dict() includes 'hybrid' match_method.
        
        Validates: Requirement 14.4
        """
        intent = InferredIntent(
            intent_type="find",
            confidence=0.8,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        result = AutoMatchResult(
            query="find customer",
            intent=intent,
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="find customer",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.8,
            match_method="hybrid",
            disambiguation_needed=False,
            disambiguation_options=[]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["match_method"] == "hybrid"
