"""
Unit tests for the AutoSchemaMatcher disambiguation detection.

Tests the disambiguation detection functionality in match_query() method.

Validates: Requirements 12.1, 12.4
- Requirement 12.1: WHEN multiple intents have similar confidence, THE SDK SHALL return all candidates
- Requirement 12.4: THE SDK SHALL set disambiguation_needed flag when clarification is needed
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from glyphh.nl.auto_schema_matcher import (
    AutoSchemaMatcher,
    AutoMatchConfig,
    AutoMatchResult,
)
from glyphh.nl.query_tokenizer import TokenizerConfig
from glyphh.nl.schema_matcher import MatchConfig, MatchResult
from glyphh.nl.intent_inferrer import IntentKeywords, InferredIntent
from glyphh.nl.parameter_extractor import ExtractionResult


class TestDisambiguationDetection:
    """Tests for disambiguation detection in AutoSchemaMatcher.match_query().
    
    Validates: Requirements 12.1, 12.4
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
    
    def test_no_disambiguation_when_single_intent(self, matcher_with_mocked_components):
        """Test that disambiguation_needed is False when only one intent is detected.
        
        Validates: Requirement 12.4
        """
        matcher = matcher_with_mocked_components
        
        # Setup: Single intent with high confidence
        single_intent = InferredIntent(
            intent_type="find",
            confidence=0.8,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        # Mock the inferrer to return a single intent
        matcher._inferrer.infer_intent.return_value = [single_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="Find Toyota",
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
        result = matcher.match_query("Find Toyota")
        
        # Verify
        assert result.disambiguation_needed is False
        assert len(result.disambiguation_options) == 0
        assert result.intent.intent_type == "find"
    
    def test_disambiguation_when_two_intents_similar_confidence(self, matcher_with_mocked_components):
        """Test that disambiguation_needed is True when two intents have similar confidence.
        
        Validates: Requirements 12.1, 12.4
        """
        matcher = matcher_with_mocked_components
        
        # Setup: Two intents with similar confidence (within 0.15 threshold)
        find_intent = InferredIntent(
            intent_type="find",
            confidence=0.6,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=0.55,  # Within 0.15 of 0.6
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock the inferrer to return both intents (sorted by confidence)
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="Find similar Toyota",
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
        result = matcher.match_query("Find similar Toyota")
        
        # Verify
        assert result.disambiguation_needed is True
        assert len(result.disambiguation_options) >= 2
        # Both intents should be in disambiguation options
        intent_types = [opt.intent_type for opt in result.disambiguation_options]
        assert "find" in intent_types
        assert "similar" in intent_types
    
    def test_no_disambiguation_when_confidence_difference_large(self, matcher_with_mocked_components):
        """Test that disambiguation_needed is False when confidence difference is large.
        
        Validates: Requirement 12.4
        """
        matcher = matcher_with_mocked_components
        
        # Setup: Two intents with large confidence difference (> 0.15 threshold)
        find_intent = InferredIntent(
            intent_type="find",
            confidence=0.8,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=0.5,  # More than 0.15 below 0.8
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock the inferrer to return both intents (sorted by confidence)
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="Find similar Toyota",
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
        result = matcher.match_query("Find similar Toyota")
        
        # Verify
        assert result.disambiguation_needed is False
        assert len(result.disambiguation_options) == 0
        assert result.intent.intent_type == "find"
    
    def test_disambiguation_includes_all_candidates_within_threshold(self, matcher_with_mocked_components):
        """Test that all intents within threshold are included in disambiguation_options.
        
        Validates: Requirement 12.1
        """
        matcher = matcher_with_mocked_components
        
        # Setup: Three intents, two within threshold of the top
        find_intent = InferredIntent(
            intent_type="find",
            confidence=0.6,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=0.55,  # Within 0.15 of 0.6
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        count_intent = InferredIntent(
            intent_type="count",
            confidence=0.3,  # More than 0.15 below 0.6
            matched_keywords=["count"],
            supporting_matches=[]
        )
        
        # Mock the inferrer to return all intents (sorted by confidence)
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent, count_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="Find similar count Toyota",
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
        result = matcher.match_query("Find similar count Toyota")
        
        # Verify
        assert result.disambiguation_needed is True
        # Only find and similar should be in disambiguation options (count is too far below)
        assert len(result.disambiguation_options) == 2
        intent_types = [opt.intent_type for opt in result.disambiguation_options]
        assert "find" in intent_types
        assert "similar" in intent_types
        assert "count" not in intent_types
    
    def test_disambiguation_threshold_boundary_at_exactly_threshold(self, matcher_with_mocked_components):
        """Test disambiguation at the exact threshold boundary (0.15).
        
        When the confidence difference is exactly at the threshold (0.15),
        disambiguation should NOT be triggered because the condition is
        confidence_diff < DISAMBIGUATION_THRESHOLD (strictly less than).
        
        Validates: Requirements 12.1, 12.4
        """
        matcher = matcher_with_mocked_components
        
        # Setup: Two intents with exactly 0.15 difference (at boundary)
        # Using 0.75 and 0.6 gives 0.15000000000000002 (slightly > 0.15)
        find_intent = InferredIntent(
            intent_type="find",
            confidence=0.75,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=0.6,  # 0.75 - 0.6 = 0.15000000000000002 (slightly > 0.15)
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock the inferrer to return both intents
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="Find similar Toyota",
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
        result = matcher.match_query("Find similar Toyota")
        
        # Verify: At exactly 0.15 difference (or slightly more), should NOT trigger disambiguation
        # (the condition is confidence_diff < DISAMBIGUATION_THRESHOLD, not <=)
        assert result.disambiguation_needed is False
        assert len(result.disambiguation_options) == 0
    
    def test_disambiguation_just_below_threshold(self, matcher_with_mocked_components):
        """Test disambiguation just below the threshold boundary.
        
        Validates: Requirements 12.1, 12.4
        """
        matcher = matcher_with_mocked_components
        
        # Setup: Two intents with just under 0.15 difference
        find_intent = InferredIntent(
            intent_type="find",
            confidence=0.6,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=0.46,  # 0.14 below 0.6 (just under threshold)
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock the inferrer to return both intents
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="Find similar Toyota",
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
        result = matcher.match_query("Find similar Toyota")
        
        # Verify: Just under 0.15 difference should trigger disambiguation
        assert result.disambiguation_needed is True
        assert len(result.disambiguation_options) == 2
    
    def test_no_disambiguation_when_no_intents(self, matcher_with_mocked_components):
        """Test that disambiguation_needed is False when no intents are detected.
        
        Validates: Requirement 12.4
        """
        matcher = matcher_with_mocked_components
        
        # Setup: No intents detected (empty list)
        matcher._inferrer.infer_intent.return_value = []
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="Toyota",
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
        result = matcher.match_query("Toyota")
        
        # Verify: No intents means no disambiguation needed
        # A default intent should be created
        assert result.disambiguation_needed is False
        assert len(result.disambiguation_options) == 0
        assert result.intent.intent_type == "find"  # Default intent
        assert result.intent.confidence == 0.1  # Low confidence for default
    
    def test_disambiguation_options_contain_correct_intent_objects(self, matcher_with_mocked_components):
        """Test that disambiguation_options contains proper InferredIntent objects.
        
        Validates: Requirement 12.1
        """
        matcher = matcher_with_mocked_components
        
        # Setup: Two intents with similar confidence
        find_intent = InferredIntent(
            intent_type="find",
            confidence=0.6,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=0.55,
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock the inferrer to return both intents
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="Find similar Toyota",
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
        result = matcher.match_query("Find similar Toyota")
        
        # Verify: All disambiguation options should be InferredIntent objects
        assert result.disambiguation_needed is True
        for option in result.disambiguation_options:
            assert isinstance(option, InferredIntent)
            assert option.intent_type in {"find", "similar"}
            assert 0.0 <= option.confidence <= 1.0
            assert isinstance(option.matched_keywords, list)
    
    def test_primary_intent_is_highest_confidence(self, matcher_with_mocked_components):
        """Test that the primary intent is the one with highest confidence.
        
        Validates: Requirement 12.1
        """
        matcher = matcher_with_mocked_components
        
        # Setup: Multiple intents with different confidences
        find_intent = InferredIntent(
            intent_type="find",
            confidence=0.6,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        similar_intent = InferredIntent(
            intent_type="similar",
            confidence=0.55,
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        # Mock the inferrer to return intents sorted by confidence
        matcher._inferrer.infer_intent.return_value = [find_intent, similar_intent]
        
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query="Find similar Toyota",
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
        result = matcher.match_query("Find similar Toyota")
        
        # Verify: Primary intent should be the highest confidence one
        assert result.intent.intent_type == "find"
        assert result.intent.confidence == 0.6


class TestAutoMatchResultDisambiguation:
    """Tests for AutoMatchResult disambiguation fields."""
    
    def test_auto_match_result_disambiguation_fields(self):
        """Test that AutoMatchResult has correct disambiguation fields."""
        # Create a result with disambiguation
        intent = InferredIntent(
            intent_type="find",
            confidence=0.6,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        disambiguation_options = [
            InferredIntent(
                intent_type="find",
                confidence=0.6,
                matched_keywords=["find"],
                supporting_matches=[]
            ),
            InferredIntent(
                intent_type="similar",
                confidence=0.55,
                matched_keywords=["similar"],
                supporting_matches=[]
            )
        ]
        
        result = AutoMatchResult(
            query="Find similar Toyota",
            intent=intent,
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="Find similar Toyota",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.6,
            match_method="auto",
            disambiguation_needed=True,
            disambiguation_options=disambiguation_options
        )
        
        # Verify
        assert result.disambiguation_needed is True
        assert len(result.disambiguation_options) == 2
        assert result.needs_clarification() is True
    
    def test_auto_match_result_no_disambiguation(self):
        """Test AutoMatchResult when no disambiguation is needed."""
        intent = InferredIntent(
            intent_type="find",
            confidence=0.8,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        result = AutoMatchResult(
            query="Find Toyota",
            intent=intent,
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="Find Toyota",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.8,
            match_method="auto",
            disambiguation_needed=False,
            disambiguation_options=[]
        )
        
        # Verify
        assert result.disambiguation_needed is False
        assert len(result.disambiguation_options) == 0
        assert result.needs_clarification() is False
    
    def test_auto_match_result_to_dict_includes_disambiguation(self):
        """Test that to_dict() includes disambiguation fields."""
        intent = InferredIntent(
            intent_type="find",
            confidence=0.6,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        disambiguation_options = [
            InferredIntent(
                intent_type="find",
                confidence=0.6,
                matched_keywords=["find"],
                supporting_matches=[]
            ),
            InferredIntent(
                intent_type="similar",
                confidence=0.55,
                matched_keywords=["similar"],
                supporting_matches=[]
            )
        ]
        
        result = AutoMatchResult(
            query="Find similar Toyota",
            intent=intent,
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="Find similar Toyota",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.6,
            match_method="auto",
            disambiguation_needed=True,
            disambiguation_options=disambiguation_options
        )
        
        # Convert to dict
        result_dict = result.to_dict()
        
        # Verify disambiguation fields are included
        assert "disambiguation_needed" in result_dict
        assert result_dict["disambiguation_needed"] is True
        assert "disambiguation_options" in result_dict
        assert len(result_dict["disambiguation_options"]) == 2
        
        # Verify disambiguation options structure
        for option in result_dict["disambiguation_options"]:
            assert "intent_type" in option
            assert "confidence" in option
            assert "matched_keywords" in option
