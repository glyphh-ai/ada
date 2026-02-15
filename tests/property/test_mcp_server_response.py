"""
Property-based tests for MCPServer response format.

This module contains property-based tests using Hypothesis to verify
that MCPServer responses maintain backward compatibility with legacy fields.

# Feature: unify-nl-query-facttree, Property 9: Backward Compatibility with Legacy Fields

**Validates: Requirements 11.1**
- WHEN the nl_query endpoint returns a response, THE System SHALL include both the FactTree structure and legacy fields (query_type, confidence, query_time_ms).
"""

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite
from datetime import datetime
from uuid import uuid4

from glyphh.fact_tree.builder import FactTree

from domains.nl_query.service import NLQueryResult, ResponseState
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
    """Generate a valid NLQueryResult with DONE state for testing."""
    return NLQueryResult(
        state=ResponseState.DONE,
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
    )


# =============================================================================
# Property Tests
# =============================================================================

class TestMCPServerBackwardCompatibility:
    """
    Property tests for MCPServer backward compatibility.
    
    # Feature: unify-nl-query-facttree, Property 9: Backward Compatibility with Legacy Fields
    
    **Validates: Requirements 11.1**
    """
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_response_contains_fact_tree_and_legacy_fields(self, nl_result: NLQueryResult):
        """
        Property test: Response contains both FactTree structure and legacy fields.
        
        **Validates: Requirements 11.1**
        """
        response = nl_result.to_dict()
        
        # Verify state is present
        assert "state" in response
        assert response["state"] == "DONE"
        
        # Verify FactTree structure is present
        assert "fact_tree" in response
        assert isinstance(response["fact_tree"], dict)
        assert "description" in response["fact_tree"]
        assert "children" in response["fact_tree"]
        
        # Verify legacy fields are present
        assert "query_type" in response
        assert "confidence" in response
        assert "query_time_ms" in response
        assert "match_method" in response
        assert "trace_id" in response
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_legacy_query_type_field_present(self, nl_result: NLQueryResult):
        """
        Property test: Legacy query_type field is always present.
        
        **Validates: Requirements 11.1**
        """
        response = nl_result.to_dict()
        
        assert "query_type" in response
        assert isinstance(response["query_type"], str)
        assert response["query_type"] == nl_result.query_type
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_legacy_confidence_field_present(self, nl_result: NLQueryResult):
        """
        Property test: Legacy confidence field is always present with valid range.
        
        **Validates: Requirements 11.1**
        """
        response = nl_result.to_dict()
        
        assert "confidence" in response
        assert isinstance(response["confidence"], float)
        assert 0.0 <= response["confidence"] <= 1.0
        assert response["confidence"] == nl_result.confidence
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_legacy_query_time_ms_field_present(self, nl_result: NLQueryResult):
        """
        Property test: Legacy query_time_ms field is always present with non-negative value.
        
        **Validates: Requirements 11.1**
        """
        response = nl_result.to_dict()
        
        assert "query_time_ms" in response
        assert isinstance(response["query_time_ms"], float)
        assert response["query_time_ms"] >= 0.0
        assert response["query_time_ms"] == nl_result.query_time_ms
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_legacy_match_method_field_present(self, nl_result: NLQueryResult):
        """
        Property test: Legacy match_method field is always present with valid value.
        
        **Validates: Requirements 11.1**
        """
        response = nl_result.to_dict()
        
        assert "match_method" in response
        assert response["match_method"] in {"rules", "llm", "auto", "hybrid", "none"}
        assert response["match_method"] == nl_result.match_method
    
    @given(nl_query_result_strategy())
    @settings(max_examples=100)
    def test_fact_tree_json_structure_valid(self, nl_result: NLQueryResult):
        """
        Property test: FactTree JSON structure is valid and complete.
        
        **Validates: Requirements 8.1, 11.3**
        """
        response = nl_result.to_dict()
        fact_tree_json = response["fact_tree"]
        
        # Verify required FactTree JSON fields
        assert "description" in fact_tree_json
        assert "value" in fact_tree_json or fact_tree_json.get("value") is None
        assert "children" in fact_tree_json
        assert isinstance(fact_tree_json["children"], list)
        assert "citations" in fact_tree_json
        assert isinstance(fact_tree_json["citations"], list)
        assert "data_context" in fact_tree_json
        assert isinstance(fact_tree_json["data_context"], dict)
    
    @given(
        query_type=st.sampled_from([
            "similarity_search", "count", "list", "temporal_predict", "unknown"
        ]),
        match_method=st.sampled_from(["rules", "llm", "auto", "hybrid", "none"]),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        query_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_all_legacy_fields_preserved_together(
        self,
        query_type: str,
        match_method: str,
        confidence: float,
        query_time_ms: float,
    ):
        """
        Property test: All legacy fields are preserved together in response.
        
        **Validates: Requirements 11.1**
        """
        fact_tree = FactTreeBuilder.build_count(count=42, query_time_ms=10.0)
        
        nl_result = NLQueryResult(
            state=ResponseState.DONE,
            fact_tree=fact_tree,
            query_type=query_type,
            match_method=match_method,
            confidence=confidence,
            query_time_ms=query_time_ms,
        )
        
        response = nl_result.to_dict()
        
        # All legacy fields must be present and correct
        assert response["query_type"] == query_type
        assert response["match_method"] == match_method
        assert response["confidence"] == confidence
        assert response["query_time_ms"] == query_time_ms
        
        # FactTree must also be present
        assert "fact_tree" in response
        assert isinstance(response["fact_tree"], dict)
    
    @given(nl_query_result_strategy())
    @settings(max_examples=50)
    def test_response_can_be_serialized_to_json(self, nl_result: NLQueryResult):
        """
        Property test: Response can be serialized to JSON without errors.
        
        **Validates: Requirements 8.1, 11.3**
        """
        import json
        
        response = nl_result.to_dict()
        
        # Should not raise any exceptions
        json_str = json.dumps(response)
        
        # Should be valid JSON that can be parsed back
        parsed = json.loads(json_str)
        
        # Verify structure is preserved
        assert parsed["query_type"] == response["query_type"]
        assert parsed["confidence"] == response["confidence"]
        assert parsed["query_time_ms"] == response["query_time_ms"]
        assert "fact_tree" in parsed
