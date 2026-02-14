"""
Property-based tests for NLQueryResult.

This module contains property-based tests using Hypothesis to verify
that NLQueryResult preserves metadata fields correctly.

# Feature: unify-nl-query-facttree, Property 7: NLQueryResult Preserves Metadata Fields

**Validates: Requirements 7.2**
- WHEN NLQueryResult is returned, THE System SHALL preserve query_type, match_method, confidence, and query_time_ms fields.
"""

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite
from datetime import datetime
from uuid import uuid4

from glyphh.fact_tree.builder import FactTree

from domains.nl_query.service import NLQueryResult
from domains.query.fact_tree_builder import FactTreeBuilder


# =============================================================================
# Generator Strategies
# =============================================================================

@composite
def fact_tree_strategy(draw) -> FactTree:
    """Generate a valid FactTree for testing."""
    operation = draw(st.sampled_from(["similarity_search", "count", "list", "error"]))
    
    if operation == "similarity_search":
        return FactTreeBuilder.build_similarity_search(
            query=draw(st.text(min_size=1, max_size=50).filter(lambda s: s.strip())),
            results=[],
            query_time_ms=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
            total_count=draw(st.integers(min_value=0, max_value=100)),
        )
    elif operation == "count":
        return FactTreeBuilder.build_count(
            count=draw(st.integers(min_value=0, max_value=10000)),
            query_time_ms=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
        )
    elif operation == "list":
        return FactTreeBuilder.build_list(
            glyphs=[],
            query_time_ms=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
            total_count=draw(st.integers(min_value=0, max_value=100)),
            limit=100,
        )
    else:
        return FactTreeBuilder.build_error(
            error_message=draw(st.text(min_size=1, max_size=100)),
            error_type=draw(st.sampled_from(["ValueError", "RuntimeError", "NoMatchError"])),
            query=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        )


@composite
def nl_query_result_strategy(draw) -> NLQueryResult:
    """Generate a valid NLQueryResult for testing."""
    return NLQueryResult(
        fact_tree=draw(fact_tree_strategy()),
        query_type=draw(st.sampled_from([
            "similarity_search", "count", "list", "temporal_predict", "fact_tree", "unknown"
        ])),
        match_method=draw(st.sampled_from(["rules", "llm", "auto", "hybrid", "none"])),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        translated_query=draw(st.one_of(
            st.none(),
            st.dictionaries(
                keys=st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10),
                values=st.one_of(st.text(max_size=20), st.integers(-100, 100)),
                max_size=3
            )
        )),
        query_time_ms=draw(st.floats(min_value=0.0, max_value=10000.0, allow_nan=False)),
        disambiguation_needed=draw(st.booleans()),
        disambiguation_suggestions=draw(st.lists(st.text(min_size=1, max_size=50), max_size=3)),
    )


# =============================================================================
# Property Tests
# =============================================================================

class TestNLQueryResultPreservesMetadataFields:
    """
    Property tests for NLQueryResult metadata preservation.
    
    # Feature: unify-nl-query-facttree, Property 7: NLQueryResult Preserves Metadata Fields
    
    **Validates: Requirements 7.2**
    """
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_to_dict_preserves_query_type(self, nl_result: NLQueryResult):
        """
        Property test: to_dict() preserves query_type field.
        
        **Validates: Requirements 7.2**
        """
        result_dict = nl_result.to_dict()
        
        assert "query_type" in result_dict
        assert result_dict["query_type"] == nl_result.query_type
        assert isinstance(result_dict["query_type"], str)
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_to_dict_preserves_match_method(self, nl_result: NLQueryResult):
        """
        Property test: to_dict() preserves match_method field.
        
        **Validates: Requirements 7.2**
        """
        result_dict = nl_result.to_dict()
        
        assert "match_method" in result_dict
        assert result_dict["match_method"] == nl_result.match_method
        assert result_dict["match_method"] in {"rules", "llm", "auto", "hybrid", "none"}
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_to_dict_preserves_confidence(self, nl_result: NLQueryResult):
        """
        Property test: to_dict() preserves confidence field with valid range.
        
        **Validates: Requirements 7.2**
        """
        result_dict = nl_result.to_dict()
        
        assert "confidence" in result_dict
        assert result_dict["confidence"] == nl_result.confidence
        assert 0.0 <= result_dict["confidence"] <= 1.0
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_to_dict_preserves_query_time_ms(self, nl_result: NLQueryResult):
        """
        Property test: to_dict() preserves query_time_ms field with non-negative value.
        
        **Validates: Requirements 7.2**
        """
        result_dict = nl_result.to_dict()
        
        assert "query_time_ms" in result_dict
        assert result_dict["query_time_ms"] == nl_result.query_time_ms
        assert result_dict["query_time_ms"] >= 0.0
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_to_dict_contains_all_required_fields(self, nl_result: NLQueryResult):
        """
        Property test: to_dict() contains all required metadata fields.
        
        **Validates: Requirements 7.2**
        """
        result_dict = nl_result.to_dict()
        
        required_fields = {
            "result",
            "query_type",
            "match_method",
            "confidence",
            "query_time_ms",
            "translated_query",
            "disambiguation_needed",
            "disambiguation_suggestions",
        }
        
        for field in required_fields:
            assert field in result_dict, f"Missing required field: {field}"
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_to_dict_result_is_fact_tree_json(self, nl_result: NLQueryResult):
        """
        Property test: to_dict() result field contains FactTree JSON.
        
        **Validates: Requirements 7.1**
        """
        result_dict = nl_result.to_dict()
        
        assert "result" in result_dict
        assert isinstance(result_dict["result"], dict)
        
        # Verify FactTree JSON structure
        fact_tree_json = result_dict["result"]
        assert "description" in fact_tree_json
        assert "children" in fact_tree_json
        assert "citations" in fact_tree_json
        assert "data_context" in fact_tree_json
    
    @given(
        query_type=st.sampled_from([
            "similarity_search", "count", "list", "temporal_predict", "fact_tree", "unknown"
        ]),
        match_method=st.sampled_from(["rules", "llm", "auto", "hybrid", "none"]),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        query_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_metadata_fields_preserved_independently(
        self,
        query_type: str,
        match_method: str,
        confidence: float,
        query_time_ms: float,
    ):
        """
        Property test: Each metadata field is preserved independently.
        
        **Validates: Requirements 7.2**
        """
        # Create a simple FactTree
        fact_tree = FactTreeBuilder.build_count(count=42, query_time_ms=10.0)
        
        nl_result = NLQueryResult(
            fact_tree=fact_tree,
            query_type=query_type,
            match_method=match_method,
            confidence=confidence,
            translated_query=None,
            query_time_ms=query_time_ms,
        )
        
        result_dict = nl_result.to_dict()
        
        assert result_dict["query_type"] == query_type
        assert result_dict["match_method"] == match_method
        assert result_dict["confidence"] == confidence
        assert result_dict["query_time_ms"] == query_time_ms
    
    @given(nl_query_result_strategy())
    @settings(max_examples=50)
    def test_disambiguation_fields_preserved(self, nl_result: NLQueryResult):
        """
        Property test: Disambiguation fields are preserved in to_dict().
        
        **Validates: Requirements 7.3**
        """
        result_dict = nl_result.to_dict()
        
        assert "disambiguation_needed" in result_dict
        assert result_dict["disambiguation_needed"] == nl_result.disambiguation_needed
        
        assert "disambiguation_suggestions" in result_dict
        assert result_dict["disambiguation_suggestions"] == nl_result.disambiguation_suggestions
    
    @given(data=st.data())
    @settings(max_examples=30)
    def test_translated_query_preserved_when_present(self, data):
        """
        Property test: translated_query is preserved when present.
        
        **Validates: Requirements 7.2**
        """
        translated_query = data.draw(st.dictionaries(
            keys=st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10),
            values=st.one_of(st.text(max_size=20), st.integers(-100, 100)),
            min_size=1,
            max_size=5
        ))
        
        fact_tree = FactTreeBuilder.build_count(count=42, query_time_ms=10.0)
        
        nl_result = NLQueryResult(
            fact_tree=fact_tree,
            query_type="count",
            match_method="rules",
            confidence=0.9,
            translated_query=translated_query,
            query_time_ms=50.0,
        )
        
        result_dict = nl_result.to_dict()
        
        assert result_dict["translated_query"] == translated_query
