"""
Property-based tests for QueryService FactTree methods.

This module contains property-based tests using Hypothesis to verify
that FactTree root descriptions match expected query types.

# Feature: unify-nl-query-facttree, Property 2: FactTree Root Description Matches Query Type

**Validates: Requirements 1.4, 3.1, 4.1, 5.1, 6.1**
- THE FactTree root node description SHALL be set to the query type (e.g., "Similarity Search", "Count Query", "List Query").
- WHEN a similarity_search operation completes, THE System SHALL create a FactTree with the query as the root description.
- WHEN a count operation completes, THE System SHALL create a FactTree with "Count Query" as the root description.
- WHEN a list operation completes, THE System SHALL create a FactTree with "List Query" as the root description.
- WHEN a temporal_predict operation completes, THE System SHALL create a FactTree with "Temporal Prediction" as the root description.
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

class TestFactTreeRootDescriptionMatchesQueryType:
    """
    Property tests for FactTree root description matching query type.
    
    # Feature: unify-nl-query-facttree, Property 2: FactTree Root Description Matches Query Type
    
    **Validates: Requirements 1.4, 3.1, 4.1, 5.1, 6.1**
    """
    
    @given(st.sampled_from([
        ("similarity_search", "Similarity Search"),
        ("count", "Count Query"),
        ("list", "List Query"),
        ("temporal_predict", "Temporal Prediction"),
    ]))
    @settings(max_examples=100)
    def test_root_description_matches_query_type(self, operation_and_expected):
        """
        Property test: For any query type, root description matches expected value.
        
        # Feature: unify-nl-query-facttree, Property 2: FactTree Root Description Matches Query Type
        
        **Validates: Requirements 1.4, 3.1, 4.1, 5.1, 6.1**
        """
        operation, expected = operation_and_expected
        
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
        
        assert result.root.description == expected, \
            f"Expected root description '{expected}' for {operation}, got '{result.root.description}'"
    
    @given(
        query=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
        query_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
        total_count=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=50)
    def test_similarity_search_root_description(
        self, query: str, query_time_ms: float, total_count: int
    ):
        """
        Property test: Similarity search FactTree root description is always "Similarity Search".
        
        **Validates: Requirements 1.4, 3.1**
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
    def test_count_root_description(
        self, count: int, query_time_ms: float, filter_applied: str | None
    ):
        """
        Property test: Count FactTree root description is always "Count Query".
        
        **Validates: Requirements 1.4, 4.1**
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
    def test_list_root_description(
        self, query_time_ms: float, total_count: int, limit: int, offset: int
    ):
        """
        Property test: List FactTree root description is always "List Query".
        
        **Validates: Requirements 1.4, 5.1**
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
    def test_temporal_predict_root_description(
        self, prediction_time_ms: float, steps_ahead: int
    ):
        """
        Property test: Temporal predict FactTree root description is always "Temporal Prediction".
        
        **Validates: Requirements 1.4, 6.1**
        """
        result = FactTreeBuilder.build_temporal_predict(
            predictions=[],
            prediction_time_ms=prediction_time_ms,
            current_state=[],
            steps_ahead=steps_ahead,
        )
        
        assert isinstance(result, FactTree)
        assert result.root.description == "Temporal Prediction"
    
    @given(data=st.data())
    @settings(max_examples=30)
    def test_similarity_search_with_results_root_description(self, data):
        """
        Property test: Similarity search with results still has correct root description.
        
        **Validates: Requirements 1.4, 3.1**
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
        assert result.root.description == "Similarity Search"
        
        # Verify serialization preserves root description
        json_output = result.to_json()
        assert json_output["description"] == "Similarity Search"
    
    @given(data=st.data())
    @settings(max_examples=30)
    def test_list_with_glyphs_root_description(self, data):
        """
        Property test: List with glyphs still has correct root description.
        
        **Validates: Requirements 1.4, 5.1**
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
        assert result.root.description == "List Query"
        
        # Verify serialization preserves root description
        json_output = result.to_json()
        assert json_output["description"] == "List Query"
    
    @given(data=st.data())
    @settings(max_examples=30)
    def test_temporal_predict_with_predictions_root_description(self, data):
        """
        Property test: Temporal predict with predictions still has correct root description.
        
        **Validates: Requirements 1.4, 6.1**
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
        assert result.root.description == "Temporal Prediction"
        
        # Verify serialization preserves root description
        json_output = result.to_json()
        assert json_output["description"] == "Temporal Prediction"
    
    @given(st.sampled_from([
        ("similarity_search", "Similarity Search"),
        ("count", "Count Query"),
        ("list", "List Query"),
        ("temporal_predict", "Temporal Prediction"),
    ]))
    @settings(max_examples=100)
    def test_root_description_preserved_in_json_serialization(self, operation_and_expected):
        """
        Property test: Root description is preserved when FactTree is serialized to JSON.
        
        **Validates: Requirements 1.4, 3.1, 4.1, 5.1, 6.1**
        """
        operation, expected = operation_and_expected
        
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
        
        # Verify root description in FactTree object
        assert result.root.description == expected
        
        # Verify root description is preserved in JSON serialization
        json_output = result.to_json()
        assert isinstance(json_output, dict)
        assert "description" in json_output
        assert json_output["description"] == expected
