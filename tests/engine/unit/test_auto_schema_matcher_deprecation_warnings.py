"""
Unit tests for the AutoSchemaMatcher deprecation warnings for manual patterns.

Tests the deprecation warning functionality where the SDK logs warnings for
manual patterns that could be auto-matched with high confidence.

Validates: Requirement 14.6
- THE SDK SHALL log deprecation warnings for manual patterns that could be auto-matched
"""

import logging
import warnings
import pytest
from unittest.mock import MagicMock, patch

from glyphh.nl.auto_schema_matcher import (
    AutoSchemaMatcher,
    AutoMatchConfig,
    AutoMatchResult,
    DEFAULT_DEPRECATION_THRESHOLD,
)
from glyphh.nl.query_tokenizer import TokenizerConfig, Token
from glyphh.nl.schema_matcher import MatchConfig, MatchResult, TokenMatch
from glyphh.nl.schema_vectorizer import SchemaVector
from glyphh.nl.intent_inferrer import IntentKeywords, InferredIntent
from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter


class TestDeprecationWarningConfiguration:
    """Tests for deprecation warning configuration.
    
    Validates: Requirement 14.6
    """
    
    def test_default_deprecation_threshold_is_defined(self):
        """Test that DEFAULT_DEPRECATION_THRESHOLD is defined and reasonable.
        
        Validates: Requirement 14.6
        """
        assert DEFAULT_DEPRECATION_THRESHOLD is not None
        assert 0.0 < DEFAULT_DEPRECATION_THRESHOLD <= 1.0
        assert DEFAULT_DEPRECATION_THRESHOLD == 0.7  # Expected default
    
    def test_deprecation_threshold_is_configurable(self):
        """Test that deprecation threshold can be customized.
        
        Validates: Requirement 14.6
        """
        # Create a matcher instance without calling __init__
        matcher = object.__new__(AutoSchemaMatcher)
        
        # Set up required attributes
        matcher.encoder = MagicMock()
        matcher.config = MagicMock()
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize sub-components with mocks
        matcher._vectorizer = MagicMock()
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.match_query.return_value = MatchResult(
            query="test",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._inferrer.infer_intent.return_value = []
        matcher._extractor = MagicMock()
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Create a manual result
        manual_result = AutoMatchResult(
            query="test",
            intent=InferredIntent(
                intent_type="find",
                confidence=0.8,
                matched_keywords=["find"],
                supporting_matches=[]
            ),
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="test",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.8,
            match_method="manual",
            disambiguation_needed=False,
            disambiguation_options=[]
        )
        
        # Test with custom threshold - should not raise
        matcher._check_deprecation_warning(
            query="test",
            manual_result=manual_result,
            matched_phrase="test pattern",
            pattern_intent_type="find",
            deprecation_threshold=0.9  # Custom threshold
        )


class TestDeprecationWarningLogging:
    """Tests for deprecation warning logging behavior.
    
    Validates: Requirement 14.6
    """
    
    @pytest.fixture
    def matcher_with_high_confidence_auto_match(self):
        """Create a matcher that will produce high-confidence auto-match results."""
        # Create a matcher instance without calling __init__
        matcher = object.__new__(AutoSchemaMatcher)
        
        # Set up required attributes
        matcher.encoder = MagicMock()
        matcher.config = MagicMock()
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Create a mock schema vector for high-confidence matching
        mock_schema_vector = MagicMock(spec=SchemaVector)
        mock_schema_vector.element_type = "value"
        mock_schema_vector.key = "make=Toyota"
        mock_schema_vector.role_path = "make"
        mock_schema_vector.original_value = "Toyota"
        
        # Create a mock token match with high similarity
        mock_token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        mock_token_match = TokenMatch(
            token=mock_token,
            schema_vector=mock_schema_vector,
            similarity=0.9,  # High similarity
            match_type="exact"
        )
        
        # Initialize sub-components with mocks
        matcher._vectorizer = MagicMock()
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = [mock_token]
        
        matcher._matcher = MagicMock()
        matcher._matcher.match_query.return_value = MatchResult(
            query="find Toyota",
            token_matches=[mock_token_match],
            role_matches=[],
            value_matches=[mock_token_match],
            compound_matches=[],
            all_candidates=[]
        )
        matcher._matcher.prefer_compound_matches.return_value = [mock_token_match]
        
        # Mock inferrer to return high-confidence intent
        high_confidence_intent = InferredIntent(
            intent_type="find",
            confidence=0.9,  # High confidence
            matched_keywords=["find"],
            supporting_matches=[mock_token_match]
        )
        matcher._inferrer = MagicMock()
        matcher._inferrer.infer_intent.return_value = [high_confidence_intent]
        
        # Mock extractor to return extracted parameters
        extracted_param = ExtractedParameter(
            role="make",
            value="Toyota",
            original_text="Toyota",
            confidence=0.9,
            value_type="string"
        )
        matcher._extractor = MagicMock()
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={"make": extracted_param},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        return matcher
    
    @pytest.fixture
    def matcher_with_low_confidence_auto_match(self):
        """Create a matcher that will produce low-confidence auto-match results."""
        # Create a matcher instance without calling __init__
        matcher = object.__new__(AutoSchemaMatcher)
        
        # Set up required attributes
        matcher.encoder = MagicMock()
        matcher.config = MagicMock()
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize sub-components with mocks
        matcher._vectorizer = MagicMock()
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        
        matcher._matcher = MagicMock()
        matcher._matcher.match_query.return_value = MatchResult(
            query="custom query",
            token_matches=[],  # No matches
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        matcher._matcher.prefer_compound_matches.return_value = []
        
        # Mock inferrer to return low-confidence intent
        low_confidence_intent = InferredIntent(
            intent_type="find",
            confidence=0.2,  # Low confidence
            matched_keywords=[],
            supporting_matches=[]
        )
        matcher._inferrer = MagicMock()
        matcher._inferrer.infer_intent.return_value = [low_confidence_intent]
        
        # Mock extractor to return no parameters
        matcher._extractor = MagicMock()
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        return matcher
    
    def test_deprecation_warning_logged_when_auto_match_high_confidence(
        self, matcher_with_high_confidence_auto_match, caplog
    ):
        """Test that deprecation warning is logged when auto-match has high confidence.
        
        Validates: Requirement 14.6
        """
        matcher = matcher_with_high_confidence_auto_match
        
        # Create a manual result
        manual_result = AutoMatchResult(
            query="find Toyota",
            intent=InferredIntent(
                intent_type="find",
                confidence=0.8,
                matched_keywords=["find"],
                supporting_matches=[]
            ),
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="find Toyota",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.8,
            match_method="manual",
            disambiguation_needed=False,
            disambiguation_options=[]
        )
        
        # Capture warnings
        with caplog.at_level(logging.WARNING):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                matcher._check_deprecation_warning(
                    query="find Toyota",
                    manual_result=manual_result,
                    matched_phrase="find {make}",
                    pattern_intent_type="find"
                )
                
                # Check that a DeprecationWarning was issued
                deprecation_warnings = [
                    warning for warning in w 
                    if issubclass(warning.category, DeprecationWarning)
                ]
                assert len(deprecation_warnings) >= 1
                
                # Check warning message content
                warning_message = str(deprecation_warnings[0].message)
                assert "find {make}" in warning_message
                assert "auto-matched" in warning_message
                assert "find Toyota" in warning_message
        
        # Check that logging.warning was called
        assert "DeprecationWarning" in caplog.text
        assert "find {make}" in caplog.text
    
    def test_no_deprecation_warning_when_auto_match_low_confidence(
        self, matcher_with_low_confidence_auto_match, caplog
    ):
        """Test that no deprecation warning is logged when auto-match has low confidence.
        
        Validates: Requirement 14.6
        """
        matcher = matcher_with_low_confidence_auto_match
        
        # Create a manual result
        manual_result = AutoMatchResult(
            query="custom query",
            intent=InferredIntent(
                intent_type="find",
                confidence=0.8,
                matched_keywords=["find"],
                supporting_matches=[]
            ),
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="custom query",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.8,
            match_method="manual",
            disambiguation_needed=False,
            disambiguation_options=[]
        )
        
        # Capture warnings
        with caplog.at_level(logging.WARNING):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                matcher._check_deprecation_warning(
                    query="custom query",
                    manual_result=manual_result,
                    matched_phrase="custom pattern",
                    pattern_intent_type="find"
                )
                
                # Check that no DeprecationWarning was issued
                deprecation_warnings = [
                    warning for warning in w 
                    if issubclass(warning.category, DeprecationWarning)
                ]
                assert len(deprecation_warnings) == 0
        
        # Check that no warning was logged
        assert "DeprecationWarning" not in caplog.text
    
    def test_deprecation_warning_includes_pattern_details(
        self, matcher_with_high_confidence_auto_match
    ):
        """Test that deprecation warning includes pattern details.
        
        Validates: Requirement 14.6
        """
        matcher = matcher_with_high_confidence_auto_match
        
        # Create a manual result with valid intent_type
        manual_result = AutoMatchResult(
            query="find Toyota",
            intent=InferredIntent(
                intent_type="find",  # Use valid intent type
                confidence=0.8,
                matched_keywords=["find"],
                supporting_matches=[]
            ),
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="find Toyota",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.8,
            match_method="manual",
            disambiguation_needed=False,
            disambiguation_options=[]
        )
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            matcher._check_deprecation_warning(
                query="find Toyota",
                manual_result=manual_result,
                matched_phrase="find {make} cars",
                pattern_intent_type="find_customer"  # Pattern can have custom intent type
            )
            
            # Check warning message content
            deprecation_warnings = [
                warning for warning in w 
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            
            warning_message = str(deprecation_warnings[0].message)
            
            # Should include pattern phrase
            assert "find {make} cars" in warning_message
            
            # Should include intent type from pattern
            assert "find_customer" in warning_message
            
            # Should include confidence score
            assert "confidence" in warning_message.lower()
            
            # Should include query
            assert "find Toyota" in warning_message
    
    def test_deprecation_warning_includes_auto_match_confidence(
        self, matcher_with_high_confidence_auto_match
    ):
        """Test that deprecation warning includes auto-match confidence score.
        
        Validates: Requirement 14.6
        """
        matcher = matcher_with_high_confidence_auto_match
        
        # Create a manual result
        manual_result = AutoMatchResult(
            query="find Toyota",
            intent=InferredIntent(
                intent_type="find",
                confidence=0.8,
                matched_keywords=["find"],
                supporting_matches=[]
            ),
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="find Toyota",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.8,
            match_method="manual",
            disambiguation_needed=False,
            disambiguation_options=[]
        )
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            matcher._check_deprecation_warning(
                query="find Toyota",
                manual_result=manual_result,
                matched_phrase="find {make}",
                pattern_intent_type="find"
            )
            
            # Check warning message includes confidence
            deprecation_warnings = [
                warning for warning in w 
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            
            warning_message = str(deprecation_warnings[0].message)
            
            # Should include a confidence value (formatted as decimal)
            # The confidence should be >= 0.7 (the threshold)
            import re
            confidence_match = re.search(r'confidence\s+(\d+\.\d+)', warning_message)
            assert confidence_match is not None


class TestDeprecationWarningIntegration:
    """Integration tests for deprecation warnings in match_query flow.
    
    Validates: Requirement 14.6
    """
    
    @pytest.fixture
    def matcher_with_manual_patterns(self):
        """Create a matcher with manual patterns configured."""
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
            fallback_to_manual=True,
            hybrid_intent_config=None
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
        
        # Create mock schema vector for high-confidence matching
        mock_schema_vector = MagicMock(spec=SchemaVector)
        mock_schema_vector.element_type = "value"
        mock_schema_vector.key = "type=customer"
        mock_schema_vector.role_path = "type"
        mock_schema_vector.original_value = "customer"
        
        # Create mock token match with very high similarity
        mock_token = Token(
            text="customer",
            original="customer",
            position=1,
            is_stop_word=False,
            ngram_size=1
        )
        mock_token_match = TokenMatch(
            token=mock_token,
            schema_vector=mock_schema_vector,
            similarity=0.95,  # Very high similarity to ensure confidence > 0.7
            match_type="exact"
        )
        
        # Initialize sub-components with mocks
        matcher._vectorizer = MagicMock()
        matcher._vectorizer.get_schema_vectors.return_value = {}
        
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = [mock_token]
        
        matcher._matcher = MagicMock()
        matcher._matcher.config = MatchConfig()
        matcher._matcher.match_query.return_value = MatchResult(
            query="find customer",
            token_matches=[mock_token_match],
            role_matches=[],
            value_matches=[mock_token_match],
            compound_matches=[],
            all_candidates=[]
        )
        matcher._matcher.prefer_compound_matches.return_value = [mock_token_match]
        
        # Mock inferrer to return high-confidence intent
        high_confidence_intent = InferredIntent(
            intent_type="find",
            confidence=0.95,  # Very high confidence
            matched_keywords=["find"],
            supporting_matches=[mock_token_match]
        )
        matcher._inferrer = MagicMock()
        matcher._inferrer.infer_intent.return_value = [high_confidence_intent]
        
        # Mock extractor to return extracted parameters (important for confidence calc)
        extracted_param = ExtractedParameter(
            role="type",
            value="customer",
            original_text="customer",
            confidence=0.95,
            value_type="string"
        )
        matcher._extractor = MagicMock()
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={"type": extracted_param},  # Include parameter for higher confidence
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        return matcher
    
    def test_deprecation_warning_triggered_during_match_query(
        self, matcher_with_manual_patterns, caplog
    ):
        """Test that deprecation warning is triggered during match_query when manual pattern matches.
        
        Validates: Requirement 14.6
        """
        matcher = matcher_with_manual_patterns
        
        # Execute match_query with a query that matches manual pattern
        with caplog.at_level(logging.WARNING):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                result = matcher.match_query("find customer")
                
                # Verify the result is from manual matching
                assert result.match_method == "manual"
                
                # Check that a DeprecationWarning was issued
                deprecation_warnings = [
                    warning for warning in w 
                    if issubclass(warning.category, DeprecationWarning)
                ]
                assert len(deprecation_warnings) >= 1, (
                    f"Expected at least 1 DeprecationWarning, got {len(deprecation_warnings)}. "
                    f"Captured warnings: {[str(warning.message) for warning in w]}"
                )
        
        # Check that logging.warning was called
        assert "DeprecationWarning" in caplog.text
    
    def test_manual_match_still_returns_correct_result_with_deprecation_warning(
        self, matcher_with_manual_patterns
    ):
        """Test that manual match returns correct result even when deprecation warning is logged.
        
        Validates: Requirement 14.6
        """
        matcher = matcher_with_manual_patterns
        
        # Execute match_query
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            result = matcher.match_query("find customer")
            
            # Verify the result is correct
            assert result.query == "find customer"
            assert result.match_method == "manual"
            assert result.intent.intent_type == "find"
            assert result.confidence > 0


class TestDeprecationWarningErrorHandling:
    """Tests for error handling in deprecation warning checks.
    
    Validates: Requirement 14.6
    """
    
    def test_deprecation_check_handles_exceptions_gracefully(self, caplog):
        """Test that deprecation check handles exceptions without blocking manual match.
        
        Validates: Requirement 14.6
        """
        # Create a matcher instance without calling __init__
        matcher = object.__new__(AutoSchemaMatcher)
        
        # Set up required attributes
        matcher.encoder = MagicMock()
        matcher.config = MagicMock()
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize sub-components with mocks that raise exceptions
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.side_effect = Exception("Tokenization failed")
        
        matcher._matcher = MagicMock()
        matcher._inferrer = MagicMock()
        matcher._extractor = MagicMock()
        
        # Create a manual result
        manual_result = AutoMatchResult(
            query="test query",
            intent=InferredIntent(
                intent_type="find",
                confidence=0.8,
                matched_keywords=["find"],
                supporting_matches=[]
            ),
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query="test query",
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=0.8,
            match_method="manual",
            disambiguation_needed=False,
            disambiguation_options=[]
        )
        
        # Should not raise exception
        with caplog.at_level(logging.DEBUG):
            matcher._check_deprecation_warning(
                query="test query",
                manual_result=manual_result,
                matched_phrase="test pattern",
                pattern_intent_type="find"
            )
        
        # Should log debug message about failure
        assert "Failed to check deprecation warning" in caplog.text
    
    def test_deprecation_check_does_not_modify_manual_result(self):
        """Test that deprecation check does not modify the manual result.
        
        Validates: Requirement 14.6
        """
        # Create a matcher instance without calling __init__
        matcher = object.__new__(AutoSchemaMatcher)
        
        # Set up required attributes
        matcher.encoder = MagicMock()
        matcher.config = MagicMock()
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
        # Initialize sub-components with mocks
        matcher._tokenizer = MagicMock()
        matcher._tokenizer.tokenize.return_value = []
        matcher._matcher = MagicMock()
        matcher._matcher.match_query.return_value = MatchResult(
            query="test",
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        matcher._matcher.prefer_compound_matches.return_value = []
        matcher._inferrer = MagicMock()
        matcher._inferrer.infer_intent.return_value = []
        matcher._extractor = MagicMock()
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        # Create a manual result
        original_query = "test query"
        original_confidence = 0.8
        manual_result = AutoMatchResult(
            query=original_query,
            intent=InferredIntent(
                intent_type="find",
                confidence=original_confidence,
                matched_keywords=["find"],
                supporting_matches=[]
            ),
            parameters=ExtractionResult(
                parameters={},
                multi_value_params={},
                unmatched_tokens=[]
            ),
            match_result=MatchResult(
                query=original_query,
                token_matches=[],
                role_matches=[],
                value_matches=[],
                compound_matches=[],
                all_candidates=[]
            ),
            confidence=original_confidence,
            match_method="manual",
            disambiguation_needed=False,
            disambiguation_options=[]
        )
        
        # Call deprecation check
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            matcher._check_deprecation_warning(
                query=original_query,
                manual_result=manual_result,
                matched_phrase="test pattern",
                pattern_intent_type="find"
            )
        
        # Verify manual result is unchanged
        assert manual_result.query == original_query
        assert manual_result.confidence == original_confidence
        assert manual_result.match_method == "manual"
