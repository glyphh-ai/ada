"""
Integration property-based tests for FactTree functionality.

This module contains property-based tests using Hypothesis to verify
FactTree serialization, citation conversion, result node structure,
metadata nodes, and error responses.

# Feature: unify-nl-query-facttree

Properties tested:
- Property 5: FactTree Serialization Round-Trip
- Property 6: GlyphResponse to Citation Conversion
- Property 3: Result Nodes Contain Required Fields and Citations
- Property 4: Metadata Node Contains Execution Timing
- Property 8: Error Responses Contain Error FactTree
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


@composite
def fact_tree_strategy(draw) -> FactTree:
    """Generate a valid FactTree for testing."""
    operation = draw(st.sampled_from(["similarity_search", "count", "list", "temporal_predict"]))
    
    if operation == "similarity_search":
        num_results = draw(st.integers(min_value=0, max_value=5))
        results = [draw(scored_glyph_strategy()) for _ in range(num_results)]
        return FactTreeBuilder.build_similarity_search(
            query=draw(st.text(min_size=1, max_size=50).filter(lambda s: s.strip())),
            results=results,
            query_time_ms=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
            total_count=num_results,
        )
    elif operation == "count":
        return FactTreeBuilder.build_count(
            count=draw(st.integers(min_value=0, max_value=10000)),
            query_time_ms=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
        )
    elif operation == "list":
        num_glyphs = draw(st.integers(min_value=0, max_value=5))
        glyphs = [draw(glyph_response_strategy()) for _ in range(num_glyphs)]
        return FactTreeBuilder.build_list(
            glyphs=glyphs,
            query_time_ms=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
            total_count=num_glyphs,
            limit=100,
        )
    else:
        num_predictions = draw(st.integers(min_value=0, max_value=3))
        predictions = [draw(predicted_state_strategy()) for _ in range(num_predictions)]
        return FactTreeBuilder.build_temporal_predict(
            predictions=predictions,
            prediction_time_ms=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
            current_state=[f"concept_{i}" for i in range(draw(st.integers(min_value=1, max_value=3)))],
            steps_ahead=draw(st.integers(min_value=1, max_value=5)),
        )


# =============================================================================
# Property 5: FactTree Serialization Round-Trip
# =============================================================================

class TestFactTreeSerializationRoundTrip:
    """
    Property tests for FactTree serialization round-trip.
    
    # Feature: unify-nl-query-facttree, Property 5: FactTree Serialization Round-Trip
    
    **Validates: Requirements 7.1, 8.1, 11.3**
    """
    
    @given(fact_tree_strategy())
    @settings(max_examples=50)
    def test_serialization_produces_valid_json(self, fact_tree: FactTree):
        """
        Property test: Serializing FactTree produces valid JSON structure.
        
        **Validates: Requirements 7.1, 8.1, 11.3**
        """
        json_data = fact_tree.to_json()
        
        assert isinstance(json_data, dict)
        assert "description" in json_data
        assert "value" in json_data or json_data.get("value") is None
        assert "children" in json_data
        assert isinstance(json_data["children"], list)
        assert "citations" in json_data
        assert isinstance(json_data["citations"], list)
        assert "data_context" in json_data
        assert isinstance(json_data["data_context"], dict)
    
    @given(fact_tree_strategy())
    @settings(max_examples=50)
    def test_children_have_valid_structure(self, fact_tree: FactTree):
        """
        Property test: All children in serialized FactTree have valid structure.
        
        **Validates: Requirements 7.1, 8.1**
        """
        json_data = fact_tree.to_json()
        
        def check_node(node: dict):
            assert "description" in node
            assert "children" in node
            assert isinstance(node["children"], list)
            assert "citations" in node
            assert isinstance(node["citations"], list)
            assert "data_context" in node
            
            for child in node["children"]:
                check_node(child)
        
        check_node(json_data)
    
    @given(fact_tree_strategy())
    @settings(max_examples=30)
    def test_json_can_be_serialized_to_string(self, fact_tree: FactTree):
        """
        Property test: FactTree JSON can be serialized to string and back.
        
        **Validates: Requirements 11.3**
        """
        import json
        
        json_data = fact_tree.to_json()
        json_str = json.dumps(json_data)
        parsed = json.loads(json_str)
        
        assert parsed["description"] == json_data["description"]
        assert len(parsed["children"]) == len(json_data["children"])


# =============================================================================
# Property 6: GlyphResponse to Citation Conversion
# =============================================================================

class TestGlyphResponseToCitationConversion:
    """
    Property tests for GlyphResponse to Citation conversion.
    
    # Feature: unify-nl-query-facttree, Property 6: GlyphResponse to Citation Conversion
    
    **Validates: Requirements 9.3, 9.4**
    """
    
    @given(glyph_response_strategy())
    @settings(max_examples=100)
    def test_citation_contains_glyph_id(self, glyph: GlyphResponse):
        """
        Property test: Citation glyph_id contains the original glyph UUID.
        
        **Validates: Requirements 9.3**
        """
        citation = FactTreeBuilder.create_citation(glyph)
        
        assert str(glyph.id) in citation.glyph_id
    
    @given(glyph_response_strategy())
    @settings(max_examples=100)
    def test_citation_contains_timestamp(self, glyph: GlyphResponse):
        """
        Property test: Citation glyph_id contains the glyph timestamp.
        
        **Validates: Requirements 9.4**
        """
        citation = FactTreeBuilder.create_citation(glyph)
        
        assert glyph.created_at.isoformat() in citation.glyph_id
    
    @given(glyph_response_strategy())
    @settings(max_examples=100)
    def test_citation_has_default_component(self, glyph: GlyphResponse):
        """
        Property test: Citation component defaults to "cortex".
        
        **Validates: Requirements 9.3**
        """
        citation = FactTreeBuilder.create_citation(glyph)
        
        assert citation.component == "cortex"
    
    @given(glyph_response_strategy())
    @settings(max_examples=100)
    def test_citation_has_non_empty_data_hash(self, glyph: GlyphResponse):
        """
        Property test: Citation has a non-empty data_hash.
        
        **Validates: Requirements 9.4**
        """
        citation = FactTreeBuilder.create_citation(glyph)
        
        assert len(citation.data_hash) > 0
    
    @given(glyph_response_strategy())
    @settings(max_examples=100)
    def test_citation_has_valid_timestamp(self, glyph: GlyphResponse):
        """
        Property test: Citation has a valid datetime timestamp.
        
        **Validates: Requirements 9.4**
        """
        citation = FactTreeBuilder.create_citation(glyph)
        
        assert isinstance(citation.timestamp, datetime)


# =============================================================================
# Property 3: Result Nodes Contain Required Fields and Citations
# =============================================================================

class TestResultNodesContainRequiredFields:
    """
    Property tests for result node structure.
    
    # Feature: unify-nl-query-facttree, Property 3: Result Nodes Contain Required Fields and Citations
    
    **Validates: Requirements 3.2, 3.3, 5.2, 6.2, 6.3**
    """
    
    @given(data=st.data())
    @settings(max_examples=30)
    def test_similarity_search_results_have_required_fields(self, data):
        """
        Property test: Similarity search result nodes have required fields.
        
        **Validates: Requirements 3.2, 3.3**
        """
        num_results = data.draw(st.integers(min_value=1, max_value=5))
        results = [data.draw(scored_glyph_strategy()) for _ in range(num_results)]
        
        fact_tree = FactTreeBuilder.build_similarity_search(
            query="test query",
            results=results,
            query_time_ms=50.0,
            total_count=num_results,
        )
        
        json_data = fact_tree.to_json()
        
        # Find results node
        results_node = None
        for child in json_data["children"]:
            if "results" in child.get("description", "").lower():
                results_node = child
                break
        
        if results_node and results_node["children"]:
            for result_child in results_node["children"]:
                value = result_child.get("value", {})
                if isinstance(value, dict):
                    assert "glyph_id" in value
                    assert "concept_text" in value
                    # Check citations
                    assert len(result_child.get("citations", [])) >= 1
    
    @given(data=st.data())
    @settings(max_examples=30)
    def test_list_results_have_citations(self, data):
        """
        Property test: List result nodes have citations.
        
        **Validates: Requirements 5.2**
        """
        num_glyphs = data.draw(st.integers(min_value=1, max_value=5))
        glyphs = [data.draw(glyph_response_strategy()) for _ in range(num_glyphs)]
        
        fact_tree = FactTreeBuilder.build_list(
            glyphs=glyphs,
            query_time_ms=50.0,
            total_count=num_glyphs,
            limit=100,
        )
        
        json_data = fact_tree.to_json()
        
        # Find results node
        results_node = None
        for child in json_data["children"]:
            if "results" in child.get("description", "").lower():
                results_node = child
                break
        
        if results_node and results_node["children"]:
            for result_child in results_node["children"]:
                # Each glyph should have a citation
                assert len(result_child.get("citations", [])) >= 1


# =============================================================================
# Property 4: Metadata Node Contains Execution Timing
# =============================================================================

class TestMetadataNodeContainsExecutionTiming:
    """
    Property tests for metadata node structure.
    
    # Feature: unify-nl-query-facttree, Property 4: Metadata Node Contains Execution Timing
    
    **Validates: Requirements 3.4, 4.3, 5.3, 6.4**
    """
    
    @given(query_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False))
    @settings(max_examples=50)
    def test_similarity_search_has_query_time(self, query_time_ms: float):
        """
        Property test: Similarity search metadata contains query_time_ms.
        
        **Validates: Requirements 3.4**
        """
        fact_tree = FactTreeBuilder.build_similarity_search(
            query="test",
            results=[],
            query_time_ms=query_time_ms,
            total_count=0,
        )
        
        json_data = fact_tree.to_json()
        
        # Find metadata node
        metadata_node = None
        for child in json_data["children"]:
            if "metadata" in child.get("description", "").lower():
                metadata_node = child
                break
        
        assert metadata_node is not None
        assert "query_time_ms" in metadata_node.get("data_context", {})
        assert metadata_node["data_context"]["query_time_ms"] >= 0
    
    @given(query_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False))
    @settings(max_examples=50)
    def test_count_has_query_time(self, query_time_ms: float):
        """
        Property test: Count metadata contains query_time_ms.
        
        **Validates: Requirements 4.3**
        """
        fact_tree = FactTreeBuilder.build_count(
            count=42,
            query_time_ms=query_time_ms,
        )
        
        json_data = fact_tree.to_json()
        
        # Find metadata node
        metadata_node = None
        for child in json_data["children"]:
            if "metadata" in child.get("description", "").lower():
                metadata_node = child
                break
        
        assert metadata_node is not None
        assert "query_time_ms" in metadata_node.get("data_context", {})
    
    @given(prediction_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False))
    @settings(max_examples=50)
    def test_temporal_predict_has_prediction_time(self, prediction_time_ms: float):
        """
        Property test: Temporal predict metadata contains prediction_time_ms.
        
        **Validates: Requirements 6.4**
        """
        fact_tree = FactTreeBuilder.build_temporal_predict(
            predictions=[],
            prediction_time_ms=prediction_time_ms,
            current_state=["test"],
            steps_ahead=1,
        )
        
        json_data = fact_tree.to_json()
        
        # Find metadata node
        metadata_node = None
        for child in json_data["children"]:
            if "metadata" in child.get("description", "").lower():
                metadata_node = child
                break
        
        assert metadata_node is not None
        assert "prediction_time_ms" in metadata_node.get("data_context", {})


# =============================================================================
# Property 8: Error Responses Contain Error FactTree
# =============================================================================

class TestErrorResponsesContainErrorFactTree:
    """
    Property tests for error response structure.
    
    # Feature: unify-nl-query-facttree, Property 8: Error Responses Contain Error FactTree
    
    **Validates: Requirements 8.3**
    """
    
    @given(
        error_message=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
        error_type=st.sampled_from(["ValueError", "RuntimeError", "NoMatchError", "EncodingError"]),
    )
    @settings(max_examples=50)
    def test_error_fact_tree_has_query_error_root(self, error_message: str, error_type: str):
        """
        Property test: Error FactTree has "Query Error" root description.
        
        **Validates: Requirements 8.3**
        """
        fact_tree = FactTreeBuilder.build_error(
            error_message=error_message,
            error_type=error_type,
        )
        
        assert fact_tree.root.description == "Query Error"
    
    @given(
        error_message=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
        error_type=st.sampled_from(["ValueError", "RuntimeError", "NoMatchError"]),
    )
    @settings(max_examples=50)
    def test_error_fact_tree_contains_error_node(self, error_message: str, error_type: str):
        """
        Property test: Error FactTree contains error node with message.
        
        **Validates: Requirements 8.3**
        """
        fact_tree = FactTreeBuilder.build_error(
            error_message=error_message,
            error_type=error_type,
        )
        
        json_data = fact_tree.to_json()
        
        # Find error node
        error_node = None
        for child in json_data["children"]:
            if "error" in child.get("description", "").lower():
                error_node = child
                break
        
        assert error_node is not None
        assert error_node["value"] == error_message
    
    @given(
        error_message=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
        error_type=st.sampled_from(["ValueError", "RuntimeError", "NoMatchError"]),
    )
    @settings(max_examples=50)
    def test_error_fact_tree_contains_error_type(self, error_message: str, error_type: str):
        """
        Property test: Error FactTree contains error_type in data_context.
        
        **Validates: Requirements 8.3**
        """
        fact_tree = FactTreeBuilder.build_error(
            error_message=error_message,
            error_type=error_type,
        )
        
        json_data = fact_tree.to_json()
        
        # Find error node
        error_node = None
        for child in json_data["children"]:
            if "error" in child.get("description", "").lower():
                error_node = child
                break
        
        assert error_node is not None
        assert "error_type" in error_node.get("data_context", {})
        assert error_node["data_context"]["error_type"] == error_type
    
    @given(
        error_message=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
        error_type=st.sampled_from(["ValueError", "RuntimeError"]),
        query=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
    )
    @settings(max_examples=30)
    def test_error_fact_tree_includes_original_query(self, error_message: str, error_type: str, query: str):
        """
        Property test: Error FactTree includes original query when provided.
        
        **Validates: Requirements 8.3**
        """
        fact_tree = FactTreeBuilder.build_error(
            error_message=error_message,
            error_type=error_type,
            query=query,
        )
        
        json_data = fact_tree.to_json()
        
        # Find query node
        query_node = None
        for child in json_data["children"]:
            if "query" in child.get("description", "").lower():
                query_node = child
                break
        
        assert query_node is not None
        assert query_node["value"] == query
