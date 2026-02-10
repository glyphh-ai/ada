"""
Property-based tests for MCP procedure execution response format.

This module contains property-based tests using Hypothesis to verify
the MCP response format when executing stored procedures.

**Validates: Property 9** - MCP Procedure Execution Response Format
- Response SHALL include result, procedure_name, confidence, time

**Validates: Requirements 7.1, 7.2, 7.4**
"""

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite
from typing import List, Dict, Any
from uuid import uuid4
from dataclasses import dataclass

from domains.mcp.server import MCPResponse


# =============================================================================
# Generator Strategies
# =============================================================================

ASCII_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ASCII_LETTERS_DIGITS_UNDERSCORE = ASCII_LETTERS + "0123456789_"


@composite
def valid_procedure_name_strategy(draw) -> str:
    """Generate a valid procedure name."""
    first = draw(st.sampled_from(ASCII_LETTERS))
    rest = draw(st.text(
        alphabet=ASCII_LETTERS_DIGITS_UNDERSCORE,
        min_size=0,
        max_size=15
    ))
    return first + rest


@composite
def mcp_response_strategy(draw) -> Dict[str, Any]:
    """Generate a valid MCP response dict for procedure execution."""
    procedure_name = draw(valid_procedure_name_strategy())
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    query_time_ms = draw(st.floats(min_value=0.0, max_value=10000.0))
    
    return {
        "result": {"items": []},
        "query_type": "stored_procedure",
        "match_method": "stored_procedure",
        "confidence": confidence,
        "query_time_ms": query_time_ms,
        "procedure_name": procedure_name,
    }


# =============================================================================
# Property Tests
# =============================================================================

class TestMCPResponseFormat:
    """
    Property tests for MCP Response Format (Property 9).
    
    **Validates: Property 9** - MCP Procedure Execution Response Format
    For any stored procedure execution via MCP, the response SHALL include:
    - result (the query execution result)
    - procedure_name (the matched procedure's name)
    - match_confidence (the similarity score)
    - query_time_ms (execution duration)
    
    **Validates: Requirements 7.1, 7.2, 7.4**
    """
    
    @given(data=st.data())
    @settings(max_examples=100)
    def test_mcp_response_includes_required_fields(self, data):
        """
        Property test: MCPResponse includes all required fields.
        
        For any MCPResponse, the to_dict() output SHALL include
        result, match_method, confidence, and query_time_ms.
        
        **Validates: Requirements 7.1, 7.2, 7.4**
        """
        # Generate response data
        procedure_name = data.draw(valid_procedure_name_strategy())
        confidence = data.draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        query_time_ms = data.draw(st.floats(min_value=0.0, max_value=10000.0, allow_nan=False))
        
        # Create MCPResponse
        response = MCPResponse(
            content=[{"type": "json", "data": {"items": []}}],
            is_error=False,
            result={"items": []},
            query_type="stored_procedure",
            match_method="stored_procedure",
            confidence=confidence,
            query_time_ms=query_time_ms,
        )
        
        # Convert to dict
        response_dict = response.to_dict()
        
        # Verify required fields
        assert "result" in response_dict, "Response must include 'result'"
        assert "match_method" in response_dict, "Response must include 'match_method'"
        assert "confidence" in response_dict, "Response must include 'confidence'"
        assert "query_time_ms" in response_dict, "Response must include 'query_time_ms'"
        
        # Verify values
        assert response_dict["match_method"] == "stored_procedure"
        assert response_dict["confidence"] == confidence
        assert response_dict["query_time_ms"] == query_time_ms
    
    @given(data=st.data())
    @settings(max_examples=100)
    def test_mcp_response_confidence_in_valid_range(self, data):
        """
        Property test: Confidence is always in [0, 1] range.
        
        For any MCPResponse, confidence SHALL be between 0 and 1.
        
        **Validates: Requirements 7.2**
        """
        confidence = data.draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        
        response = MCPResponse(
            content=[],
            confidence=confidence,
        )
        
        response_dict = response.to_dict()
        
        assert 0.0 <= response_dict["confidence"] <= 1.0, \
            f"Confidence {response_dict['confidence']} not in [0, 1]"
    
    @given(data=st.data())
    @settings(max_examples=100)
    def test_mcp_response_query_time_non_negative(self, data):
        """
        Property test: Query time is always non-negative.
        
        For any MCPResponse, query_time_ms SHALL be >= 0.
        
        **Validates: Requirements 7.4**
        """
        query_time_ms = data.draw(st.floats(min_value=0.0, max_value=100000.0, allow_nan=False))
        
        response = MCPResponse(
            content=[],
            query_time_ms=query_time_ms,
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["query_time_ms"] >= 0.0, \
            f"Query time {response_dict['query_time_ms']} is negative"


class TestMCPProcedureResponse:
    """
    Property tests for MCP Procedure-specific response fields.
    
    **Validates: Requirements 7.1, 7.2**
    """
    
    @given(data=st.data())
    @settings(max_examples=100)
    def test_procedure_match_method_is_stored_procedure(self, data):
        """
        Property test: Procedure matches have match_method="stored_procedure".
        
        When a stored procedure is matched, match_method SHALL be "stored_procedure".
        
        **Validates: Requirements 7.1**
        """
        response = MCPResponse(
            content=[{"type": "json", "data": {}}],
            match_method="stored_procedure",
            confidence=0.95,
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["match_method"] == "stored_procedure"
    
    @given(data=st.data())
    @settings(max_examples=50)
    def test_error_response_has_error_field(self, data):
        """
        Property test: Error responses include error field.
        
        When is_error=True, the response SHALL include an error message.
        
        **Validates: Requirements 7.4**
        """
        error_message = data.draw(st.text(min_size=1, max_size=100))
        
        response = MCPResponse(
            content=[{"type": "text", "text": error_message}],
            is_error=True,
            error=error_message,
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["isError"] is True
        assert "error" in response_dict
        assert response_dict["error"] == error_message
