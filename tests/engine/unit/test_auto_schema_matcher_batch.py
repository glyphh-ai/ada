"""
Unit tests for AutoSchemaMatcher.match_queries() batch method.

Tests the batch processing functionality that allows multiple queries
to be processed efficiently while sharing schema vectors.

Validates: Requirement 13.2 - THE SDK SHALL support batch processing of multiple queries
"""

import pytest
from unittest.mock import MagicMock

from glyphh.nl.auto_schema_matcher import (
    AutoSchemaMatcher,
    AutoMatchConfig,
    AutoMatchResult,
)
from glyphh.nl.schema_matcher import MatchConfig, MatchResult
from glyphh.nl.intent_inferrer import InferredIntent
from glyphh.nl.parameter_extractor import ExtractionResult


class TestMatchQueriesBatch:
    """Tests for the match_queries() batch method."""
    
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
    
    def _setup_mock_for_query(self, matcher, query):
        """Set up mocks to return valid results for a query."""
        # Mock the matcher to return an empty match result
        matcher._matcher.match_query.return_value = MatchResult(
            query=query,
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        # Mock the inferrer to return a find intent
        matcher._inferrer.infer_intent.return_value = [
            InferredIntent(
                intent_type="find",
                confidence=0.5,
                matched_keywords=["find"] if "find" in query.lower() else [],
                supporting_matches=[]
            )
        ]
        
        # Mock the extractor
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
    
    def test_match_queries_returns_list(self, matcher_with_mocked_components):
        """Test that match_queries returns a list of AutoMatchResult."""
        matcher = matcher_with_mocked_components
        queries = ["Find Toyota cars", "Count Honda vehicles"]
        
        # Set up mocks for each query
        for query in queries:
            self._setup_mock_for_query(matcher, query)
        
        results = matcher.match_queries(queries)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, AutoMatchResult) for r in results)
    
    def test_match_queries_empty_list_returns_empty(self, matcher_with_mocked_components):
        """Test that empty input list returns empty output list."""
        matcher = matcher_with_mocked_components
        results = matcher.match_queries([])
        
        assert results == []
        assert isinstance(results, list)
    
    def test_match_queries_preserves_order(self, matcher_with_mocked_components):
        """Test that results are in the same order as input queries."""
        matcher = matcher_with_mocked_components
        queries = ["Query A", "Query B", "Query C"]
        
        # Set up mocks for each query
        for query in queries:
            self._setup_mock_for_query(matcher, query)
        
        results = matcher.match_queries(queries)
        
        assert len(results) == 3
        assert results[0].query == "Query A"
        assert results[1].query == "Query B"
        assert results[2].query == "Query C"
    
    def test_match_queries_single_query(self, matcher_with_mocked_components):
        """Test batch processing with a single query."""
        matcher = matcher_with_mocked_components
        queries = ["Find Toyota cars"]
        
        self._setup_mock_for_query(matcher, queries[0])
        
        results = matcher.match_queries(queries)
        
        assert len(results) == 1
        assert results[0].query == "Find Toyota cars"
    
    def test_match_queries_rejects_non_list(self, matcher_with_mocked_components):
        """Test that non-list input raises TypeError."""
        matcher = matcher_with_mocked_components
        
        with pytest.raises(TypeError) as exc_info:
            matcher.match_queries("not a list")
        
        assert "queries must be a list" in str(exc_info.value)
        assert "got str" in str(exc_info.value)
    
    def test_match_queries_rejects_non_string_element(self, matcher_with_mocked_components):
        """Test that non-string elements raise TypeError."""
        matcher = matcher_with_mocked_components
        
        with pytest.raises(TypeError) as exc_info:
            matcher.match_queries(["valid query", 123])
        
        assert "queries[1] must be a string" in str(exc_info.value)
        assert "got int" in str(exc_info.value)
    
    def test_match_queries_rejects_empty_string(self, matcher_with_mocked_components):
        """Test that empty string queries raise ValueError."""
        matcher = matcher_with_mocked_components
        
        with pytest.raises(ValueError) as exc_info:
            matcher.match_queries(["valid query", ""])
        
        assert "queries[1] cannot be empty" in str(exc_info.value)
    
    def test_match_queries_rejects_whitespace_only(self, matcher_with_mocked_components):
        """Test that whitespace-only queries raise ValueError."""
        matcher = matcher_with_mocked_components
        
        with pytest.raises(ValueError) as exc_info:
            matcher.match_queries(["valid query", "   "])
        
        assert "queries[1] cannot be empty" in str(exc_info.value)
    
    def test_match_queries_validates_all_before_processing(self, matcher_with_mocked_components):
        """Test that all queries are validated before any processing."""
        matcher = matcher_with_mocked_components
        
        # If validation happens during processing, the first query would be processed
        # before the error on the second query is raised
        with pytest.raises(TypeError) as exc_info:
            matcher.match_queries(["valid query", 123, "another valid"])
        
        # Error should be for index 1 (the invalid one)
        assert "queries[1]" in str(exc_info.value)
    
    def test_match_queries_rejects_none_element(self, matcher_with_mocked_components):
        """Test that None elements raise TypeError."""
        matcher = matcher_with_mocked_components
        
        with pytest.raises(TypeError) as exc_info:
            matcher.match_queries(["valid query", None])
        
        assert "queries[1] must be a string" in str(exc_info.value)
        assert "got NoneType" in str(exc_info.value)
    
    def test_match_queries_rejects_tuple_input(self, matcher_with_mocked_components):
        """Test that tuple input raises TypeError."""
        matcher = matcher_with_mocked_components
        
        with pytest.raises(TypeError) as exc_info:
            matcher.match_queries(("query1", "query2"))
        
        assert "queries must be a list" in str(exc_info.value)
        assert "got tuple" in str(exc_info.value)
    
    def test_match_queries_rejects_dict_input(self, matcher_with_mocked_components):
        """Test that dict input raises TypeError."""
        matcher = matcher_with_mocked_components
        
        with pytest.raises(TypeError) as exc_info:
            matcher.match_queries({"query": "value"})
        
        assert "queries must be a list" in str(exc_info.value)
        assert "got dict" in str(exc_info.value)


class TestMatchQueriesConsistency:
    """Tests verifying batch results match individual query results."""
    
    @pytest.fixture
    def matcher_with_mocked_components(self):
        """Create an AutoSchemaMatcher with mocked internal components."""
        matcher = object.__new__(AutoSchemaMatcher)
        
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
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
    
    def _setup_mock_for_query(self, matcher, query):
        """Set up mocks to return valid results for a query."""
        matcher._matcher.match_query.return_value = MatchResult(
            query=query,
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        matcher._inferrer.infer_intent.return_value = [
            InferredIntent(
                intent_type="find",
                confidence=0.5,
                matched_keywords=[],
                supporting_matches=[]
            )
        ]
        
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
    
    def test_batch_results_match_individual_results(self, matcher_with_mocked_components):
        """Test that batch results are identical to individual match_query calls."""
        matcher = matcher_with_mocked_components
        queries = ["Find Toyota cars", "Count Honda vehicles", "Show BMW models"]
        
        # Set up mocks
        for query in queries:
            self._setup_mock_for_query(matcher, query)
        
        # Get batch results
        batch_results = matcher.match_queries(queries)
        
        # Get individual results
        individual_results = [matcher.match_query(q) for q in queries]
        
        # Compare results
        assert len(batch_results) == len(individual_results)
        
        for batch_result, individual_result in zip(batch_results, individual_results):
            assert batch_result.query == individual_result.query
            assert batch_result.intent.intent_type == individual_result.intent.intent_type
            assert batch_result.confidence == individual_result.confidence
            assert batch_result.match_method == individual_result.match_method
            assert batch_result.disambiguation_needed == individual_result.disambiguation_needed
    
    def test_batch_shares_schema_vectors(self, matcher_with_mocked_components):
        """Test that schema vectors are shared across batch queries."""
        matcher = matcher_with_mocked_components
        queries = ["Find Toyota", "Find Honda", "Find BMW"]
        
        # Set up mocks
        for query in queries:
            self._setup_mock_for_query(matcher, query)
        
        # Get the schema vectors before batch processing
        vectors_before = matcher.get_schema_vectors()
        
        # Process batch
        results = matcher.match_queries(queries)
        
        # Get the schema vectors after batch processing
        vectors_after = matcher.get_schema_vectors()
        
        # Vectors should be the same object (shared, not regenerated)
        assert vectors_before is vectors_after


class TestMatchQueriesEdgeCases:
    """Tests for edge cases in batch processing."""
    
    @pytest.fixture
    def matcher_with_mocked_components(self):
        """Create an AutoSchemaMatcher with mocked internal components."""
        matcher = object.__new__(AutoSchemaMatcher)
        
        matcher.encoder = MagicMock()
        matcher.encoder.dimension = 10000
        
        matcher.config = MagicMock()
        matcher.config.dimension = 10000
        matcher.config.seed = 42
        matcher.config.layers = []
        
        matcher.auto_config = AutoMatchConfig()
        matcher.manual_patterns = None
        
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
    
    def _setup_mock_for_query(self, matcher, query):
        """Set up mocks to return valid results for a query."""
        matcher._matcher.match_query.return_value = MatchResult(
            query=query,
            token_matches=[],
            role_matches=[],
            value_matches=[],
            compound_matches=[],
            all_candidates=[]
        )
        
        matcher._inferrer.infer_intent.return_value = [
            InferredIntent(
                intent_type="find",
                confidence=0.5,
                matched_keywords=[],
                supporting_matches=[]
            )
        ]
        
        matcher._extractor.extract_parameters.return_value = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
    
    def test_batch_with_duplicate_queries(self, matcher_with_mocked_components):
        """Test batch processing with duplicate queries."""
        matcher = matcher_with_mocked_components
        queries = ["Find Toyota", "Find Toyota", "Find Toyota"]
        
        for query in queries:
            self._setup_mock_for_query(matcher, query)
        
        results = matcher.match_queries(queries)
        
        assert len(results) == 3
        # All results should have the same query
        assert all(r.query == "Find Toyota" for r in results)
    
    def test_batch_with_varied_intents(self, matcher_with_mocked_components):
        """Test batch processing with queries of different intents."""
        matcher = matcher_with_mocked_components
        queries = [
            "Find Toyota cars",      # find intent
            "Count Honda vehicles",  # count intent
            "Show similar to BMW"    # similar intent
        ]
        
        for query in queries:
            self._setup_mock_for_query(matcher, query)
        
        results = matcher.match_queries(queries)
        
        assert len(results) == 3
        # Each query should have its own result
        assert results[0].query == "Find Toyota cars"
        assert results[1].query == "Count Honda vehicles"
        assert results[2].query == "Show similar to BMW"
    
    def test_batch_with_special_characters(self, matcher_with_mocked_components):
        """Test batch processing with queries containing special characters."""
        matcher = matcher_with_mocked_components
        queries = [
            "Find Toyota's cars",
            "Count Honda (2023)",
            "Show BMW - Series 3"
        ]
        
        for query in queries:
            self._setup_mock_for_query(matcher, query)
        
        results = matcher.match_queries(queries)
        
        assert len(results) == 3
        assert results[0].query == "Find Toyota's cars"
        assert results[1].query == "Count Honda (2023)"
        assert results[2].query == "Show BMW - Series 3"
    
    def test_batch_with_unicode_queries(self, matcher_with_mocked_components):
        """Test batch processing with unicode characters."""
        matcher = matcher_with_mocked_components
        queries = [
            "Find トヨタ cars",
            "Count 本田 vehicles",
            "Show BMW 車"
        ]
        
        for query in queries:
            self._setup_mock_for_query(matcher, query)
        
        results = matcher.match_queries(queries)
        
        assert len(results) == 3
        assert results[0].query == "Find トヨタ cars"
        assert results[1].query == "Count 本田 vehicles"
        assert results[2].query == "Show BMW 車"
    
    def test_batch_with_long_queries(self, matcher_with_mocked_components):
        """Test batch processing with long queries."""
        matcher = matcher_with_mocked_components
        long_query = "Find " + " ".join(["word"] * 100)
        queries = [long_query, "Short query"]
        
        for query in queries:
            self._setup_mock_for_query(matcher, query)
        
        results = matcher.match_queries(queries)
        
        assert len(results) == 2
        assert results[0].query == long_query
        assert results[1].query == "Short query"
    
    def test_batch_with_many_queries(self, matcher_with_mocked_components):
        """Test batch processing with many queries."""
        matcher = matcher_with_mocked_components
        queries = [f"Find item {i}" for i in range(100)]
        
        for query in queries:
            self._setup_mock_for_query(matcher, query)
        
        results = matcher.match_queries(queries)
        
        assert len(results) == 100
        for i, result in enumerate(results):
            assert result.query == f"Find item {i}"
    
    def test_batch_first_query_invalid_index_zero(self, matcher_with_mocked_components):
        """Test error message when first query is invalid."""
        matcher = matcher_with_mocked_components
        
        with pytest.raises(TypeError) as exc_info:
            matcher.match_queries([123, "valid query"])
        
        assert "queries[0]" in str(exc_info.value)
    
    def test_batch_last_query_invalid(self, matcher_with_mocked_components):
        """Test error message when last query is invalid."""
        matcher = matcher_with_mocked_components
        
        with pytest.raises(TypeError) as exc_info:
            matcher.match_queries(["valid query", "another valid", 456])
        
        assert "queries[2]" in str(exc_info.value)
