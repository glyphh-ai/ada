"""
Property-based tests for FactTreeBuilder.

This module contains property-based tests using Hypothesis to verify
that all query operations return SDK FactTree instances.

# Feature: unify-nl-query-facttree, Property 1: All Query Operations Return SDK FactTree

**Validates: Requirements 1.1, 1.2**
- WHEN the NLQueryService executes any query operation, THE System SHALL wrap the result in an SDK FactTree structure.
- WHEN building a FactTree response, THE System SHALL use the SDK's FactTree class from glyphh.fact_tree.builder.
"""

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite
from datetime import datetime
from uuid import uuid4

from glyphh.fact_tree.builder import FactTree

from domains.query.fact_tree_builder import FactTreeBuilder
from domains.models.schemas import GlyphResponse, ScoredGlyph, PredictedState, Delta


# =============================================================================
# Generator Strategies
# =============================================================================

@composite
def glyph_response_strategy(draw) -> GlyphResponse:
    """Generate a valid GlyphResponse for testing."""
    return GlyphResponse(
        id=uuid4(),
        org_id=f"org_{draw(st.text(alphabet='abcdef0123456789', min_size=8, max_size=8))}",
        model_id=f"model_{draw(st.text(alphabet='abcdef0123456789', min_size=8, max_size=8))}",
        concept_text=draw(st.text(min_size=1, max_size=50).filter(lambda s: s.strip())),
        metadata=draw(st.dictionaries(
            keys=st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10),
            values=st.one_of(st.text(max_size=20), st.integers(-100, 100), st.floats(-100, 100, allow_nan=False)),
            max_size=3
        )),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@composite
def scored_glyph_strategy(draw) -> ScoredGlyph:
    """Generate a valid ScoredGlyph for testing."""
    glyph = draw(glyph_response_strategy())
    return ScoredGlyph(
        glyph=glyph,
        similarity_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        final_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        security_weight=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )


@composite
def state_delta_strategy(draw) -> Delta:
    """Generate a valid Delta for testing."""
    return Delta(
        glyph_id=uuid4(),
        change_type=draw(st.sampled_from(["added", "removed", "modified"])),
        magnitude=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )


@composite
def predicted_state_strategy(draw) -> PredictedState:
    """Generate a valid PredictedState for testing."""
    num_deltas = draw(st.integers(min_value=0, max_value=3))
    return PredictedState(
        state_glyphs=[uuid4() for _ in range(draw(st.integers(min_value=1, max_value=5)))],
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        path_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        deltas=[draw(state_delta_strategy()) for _ in range(num_deltas)],
    )


# =============================================================================
# Property Tests
# =============================================================================

class TestFactTreeBuilderReturnsSDKFactTree:
    """
    Property tests for FactTreeBuilder returning SDK FactTree instances.
    
    # Feature: unify-nl-query-facttree, Property 1: All Query Operations Return SDK FactTree
    
    **Validates: Requirements 1.1, 1.2**
    """
    
    @given(st.sampled_from(["similarity_search", "count", "list", "temporal_predict"]))
    @settings(max_examples=50)
    def test_all_operations_return_fact_tree(self, operation: str):
        """
        Property test: For any query operation, result is an SDK FactTree.
        
        **Validates: Requirements 1.1, 1.2**
        """
        if operation == "similarity_search":
            result = FactTreeBuilder.build_similarity_search(
                query="test", results=[], query_time_ms=10.0, total_count=0
            )
        elif operation == "count":
            result = FactTreeBuilder.build_count(count=0, query_time_ms=10.0)
        elif operation == "list":
            result = FactTreeBuilder.build_list(
                glyphs=[], query_time_ms=10.0, total_count=0, limit=100
            )
        else:
            result = FactTreeBuilder.build_temporal_predict(
                predictions=[], prediction_time_ms=10.0, current_state=[], steps_ahead=1
            )
        
        assert isinstance(result, FactTree), \
            f"Expected FactTree instance for {operation}, got {type(result)}"
        assert hasattr(result, 'root'), \
            f"FactTree for {operation} should have 'root' attribute"
        assert hasattr(result, 'to_json'), \
            f"FactTree for {operation} should have 'to_json' method"
    
    @given(
        query=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
        query_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
        total_count=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=50)
    def test_similarity_search_returns_fact_tree(
        self, query: str, query_time_ms: float, total_count: int
    ):
        """
        Property test: build_similarity_search always returns SDK FactTree.
        
        **Validates: Requirements 1.1, 1.2**
        """
        result = FactTreeBuilder.build_similarity_search(
            query=query,
            results=[],
            query_time_ms=query_time_ms,
            total_count=total_count,
        )
        
        assert isinstance(result, FactTree)
        assert result.root.description == "Similarity Search"
    
    @given(
        count=st.integers(min_value=0, max_value=1000000),
        query_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
        filter_applied=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    )
    @settings(max_examples=50)
    def test_count_returns_fact_tree(
        self, count: int, query_time_ms: float, filter_applied: str | None
    ):
        """
        Property test: build_count always returns SDK FactTree.
        
        **Validates: Requirements 1.1, 1.2**
        """
        result = FactTreeBuilder.build_count(
            count=count,
            query_time_ms=query_time_ms,
            filter_applied=filter_applied,
        )
        
        assert isinstance(result, FactTree)
        assert result.root.description == "Count Query"
    
    @given(
        query_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
        total_count=st.integers(min_value=0, max_value=1000),
        limit=st.integers(min_value=1, max_value=1000),
        offset=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=50)
    def test_list_returns_fact_tree(
        self, query_time_ms: float, total_count: int, limit: int, offset: int
    ):
        """
        Property test: build_list always returns SDK FactTree.
        
        **Validates: Requirements 1.1, 1.2**
        """
        result = FactTreeBuilder.build_list(
            glyphs=[],
            query_time_ms=query_time_ms,
            total_count=total_count,
            limit=limit,
            offset=offset,
        )
        
        assert isinstance(result, FactTree)
        assert result.root.description == "List Query"
    
    @given(
        prediction_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
        steps_ahead=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_temporal_predict_returns_fact_tree(
        self, prediction_time_ms: float, steps_ahead: int
    ):
        """
        Property test: build_temporal_predict always returns SDK FactTree.
        
        **Validates: Requirements 1.1, 1.2**
        """
        result = FactTreeBuilder.build_temporal_predict(
            predictions=[],
            prediction_time_ms=prediction_time_ms,
            current_state=[],
            steps_ahead=steps_ahead,
        )
        
        assert isinstance(result, FactTree)
        assert result.root.description == "Temporal Prediction"
    
    @given(
        error_message=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
        error_type=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        query=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    )
    @settings(max_examples=50)
    def test_error_returns_fact_tree(
        self, error_message: str, error_type: str, query: str | None
    ):
        """
        Property test: build_error always returns SDK FactTree.
        
        **Validates: Requirements 1.1, 1.2**
        """
        result = FactTreeBuilder.build_error(
            error_message=error_message,
            error_type=error_type,
            query=query,
        )
        
        assert isinstance(result, FactTree)
        assert result.root.description == "Query Error"
    
    @given(data=st.data())
    @settings(max_examples=30)
    def test_similarity_search_with_results_returns_fact_tree(self, data):
        """
        Property test: build_similarity_search with results returns SDK FactTree.
        
        **Validates: Requirements 1.1, 1.2**
        """
        num_results = data.draw(st.integers(min_value=1, max_value=5))
        results = [data.draw(scored_glyph_strategy()) for _ in range(num_results)]
        
        result = FactTreeBuilder.build_similarity_search(
            query="test query",
            results=results,
            query_time_ms=data.draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
            total_count=num_results,
        )
        
        assert isinstance(result, FactTree)
        assert hasattr(result, 'root')
        assert hasattr(result, 'to_json')
        
        # Verify serialization works
        json_output = result.to_json()
        assert isinstance(json_output, dict)
        assert "description" in json_output
    
    @given(data=st.data())
    @settings(max_examples=30)
    def test_list_with_glyphs_returns_fact_tree(self, data):
        """
        Property test: build_list with glyphs returns SDK FactTree.
        
        **Validates: Requirements 1.1, 1.2**
        """
        num_glyphs = data.draw(st.integers(min_value=1, max_value=5))
        glyphs = [data.draw(glyph_response_strategy()) for _ in range(num_glyphs)]
        
        result = FactTreeBuilder.build_list(
            glyphs=glyphs,
            query_time_ms=data.draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
            total_count=num_glyphs,
            limit=100,
            offset=0,
        )
        
        assert isinstance(result, FactTree)
        assert hasattr(result, 'root')
        assert hasattr(result, 'to_json')
        
        # Verify serialization works
        json_output = result.to_json()
        assert isinstance(json_output, dict)
    
    @given(data=st.data())
    @settings(max_examples=30)
    def test_temporal_predict_with_predictions_returns_fact_tree(self, data):
        """
        Property test: build_temporal_predict with predictions returns SDK FactTree.
        
        **Validates: Requirements 1.1, 1.2**
        """
        num_predictions = data.draw(st.integers(min_value=1, max_value=3))
        predictions = [data.draw(predicted_state_strategy()) for _ in range(num_predictions)]
        current_state = [f"concept_{i}" for i in range(data.draw(st.integers(min_value=1, max_value=3)))]
        
        result = FactTreeBuilder.build_temporal_predict(
            predictions=predictions,
            prediction_time_ms=data.draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
            current_state=current_state,
            steps_ahead=data.draw(st.integers(min_value=1, max_value=5)),
        )
        
        assert isinstance(result, FactTree)
        assert hasattr(result, 'root')
        assert hasattr(result, 'to_json')
        
        # Verify serialization works
        json_output = result.to_json()
        assert isinstance(json_output, dict)
