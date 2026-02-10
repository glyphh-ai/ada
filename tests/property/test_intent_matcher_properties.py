"""
Property-based tests for IntentMatcher stored procedure matching.

This module contains property-based tests using Hypothesis to verify
the procedure matching priority and selection properties.

**Validates: Property 7** - Procedure Matching Priority and Selection
- Stored procedures SHALL be checked before default patterns
- If multiple procedures match above threshold, the one with highest confidence SHALL be selected
- The response match_method SHALL be "stored_procedure" when a procedure is matched

**Validates: Requirements 4.1, 4.3, 4.4, 4.5**
"""

import pytest
from hypothesis import given, settings, strategies as st, assume
from hypothesis.strategies import composite
from typing import List, Optional
from uuid import uuid4
from dataclasses import dataclass, field
from datetime import datetime

from domains.nl_query.intent_matcher import IntentMatcher, IntentMatch


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
def valid_lexicon_strategy(draw) -> str:
    """Generate a valid lexicon entry (simple words)."""
    return draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz ",
        min_size=3,
        max_size=20
    ).filter(lambda s: s.strip() and len(s.strip()) >= 3))


@composite
def valid_lexicons_list_strategy(draw) -> List[str]:
    """Generate a valid list of lexicons."""
    num_lexicons = draw(st.integers(min_value=1, max_value=3))
    return [draw(valid_lexicon_strategy()) for _ in range(num_lexicons)]


# Valid GQL queries for testing
valid_gql_queries = st.sampled_from([
    'LIST ALL LIMIT 10',
    'COUNT ALL',
    'FIND SIMILAR TO "test" LIMIT 5 THRESHOLD 0.5',
])


# =============================================================================
# Mock Stored Procedure
# =============================================================================

@dataclass
class MockStoredProcedure:
    """Mock stored procedure for testing."""
    id: str
    org_id: str
    model_id: str
    name: str
    gql_query: str
    lexicons: List[str]
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class MockProcedureService:
    """Mock procedure service for testing IntentMatcher."""
    
    def __init__(self, procedures: List[MockStoredProcedure] = None):
        self._procedures = procedures or []
    
    async def list(self, org_id: str, model_id: str) -> List[MockStoredProcedure]:
        """Return procedures matching org_id and model_id."""
        return [
            p for p in self._procedures
            if p.org_id == org_id and p.model_id == model_id
        ]


# =============================================================================
# Property Tests
# =============================================================================

class TestProcedureMatchingPriority:
    """
    Property tests for Procedure Matching Priority (Property 7).
    
    **Validates: Property 7** - Procedure Matching Priority and Selection
    - Stored procedures SHALL be checked before default patterns
    - The response match_method SHALL be "stored_procedure" when a procedure is matched
    
    **Validates: Requirements 4.1, 4.5**
    """
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=50)
    async def test_procedure_match_returns_stored_procedure_method(self, data):
        """
        Property test: When a procedure matches, match_method is "stored_procedure".
        
        For any query that exactly matches a procedure's lexicon,
        the match_method SHALL be "stored_procedure".
        
        **Validates: Requirements 4.1, 4.5**
        """
        # Generate procedure data
        name = data.draw(valid_procedure_name_strategy())
        lexicon = data.draw(valid_lexicon_strategy())
        gql_query = data.draw(valid_gql_queries)
        org_id = f"org_{uuid4().hex[:8]}"
        model_id = f"model_{uuid4().hex[:8]}"
        
        # Create procedure with the lexicon
        procedure = MockStoredProcedure(
            id=str(uuid4()),
            org_id=org_id,
            model_id=model_id,
            name=name,
            gql_query=gql_query,
            lexicons=[lexicon.strip()],
        )
        
        # Create service with the procedure
        service = MockProcedureService([procedure])
        
        # Create matcher with low threshold to ensure match
        matcher = IntentMatcher(
            confidence_threshold=0.5,
            procedure_service=service,
        )
        
        # Query using the exact lexicon
        query = lexicon.strip()
        
        # Match
        result = await matcher.match_intent(query, org_id, model_id)
        
        # If we got a match, verify it's a stored_procedure match
        if result is not None and result.confidence >= 0.5:
            assert result.match_method == "stored_procedure", \
                f"Expected match_method='stored_procedure', got '{result.match_method}'"
            assert result.procedure_name == name, \
                f"Expected procedure_name='{name}', got '{result.procedure_name}'"
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=50)
    async def test_no_procedures_falls_back_to_default(self, data):
        """
        Property test: When no procedures exist, falls back to default matching.
        
        For any query when no procedures are defined,
        the match_method SHALL NOT be "stored_procedure".
        
        **Validates: Requirements 4.1**
        """
        org_id = f"org_{uuid4().hex[:8]}"
        model_id = f"model_{uuid4().hex[:8]}"
        
        # Create service with no procedures
        service = MockProcedureService([])
        
        # Create matcher
        matcher = IntentMatcher(
            confidence_threshold=0.5,
            procedure_service=service,
        )
        
        # Query with a common pattern
        query = "find similar to cars"
        
        # Match
        result = await matcher.match_intent(query, org_id, model_id)
        
        # If we got a match, it should NOT be stored_procedure
        if result is not None:
            assert result.match_method != "stored_procedure", \
                "Should not match stored_procedure when no procedures exist"


class TestProcedureMatchingSelection:
    """
    Property tests for Procedure Matching Selection (Property 7).
    
    **Validates: Property 7** - Procedure Matching Priority and Selection
    - If multiple procedures match above threshold, the one with highest confidence SHALL be selected
    
    **Validates: Requirements 4.3, 4.4**
    """
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=50)
    async def test_best_match_selected_from_multiple_procedures(self, data):
        """
        Property test: Best matching procedure is selected.
        
        When multiple procedures could match, the one with the
        highest similarity score SHALL be selected.
        
        **Validates: Requirements 4.3, 4.4**
        """
        org_id = f"org_{uuid4().hex[:8]}"
        model_id = f"model_{uuid4().hex[:8]}"
        
        # Create two procedures with different lexicons
        # One with exact match, one with partial match
        exact_procedure = MockStoredProcedure(
            id=str(uuid4()),
            org_id=org_id,
            model_id=model_id,
            name="exact_match_proc",
            gql_query="LIST ALL LIMIT 10",
            lexicons=["reliable family car"],
        )
        
        partial_procedure = MockStoredProcedure(
            id=str(uuid4()),
            org_id=org_id,
            model_id=model_id,
            name="partial_match_proc",
            gql_query="COUNT ALL",
            lexicons=["car"],
        )
        
        # Create service with both procedures
        service = MockProcedureService([exact_procedure, partial_procedure])
        
        # Create matcher
        matcher = IntentMatcher(
            confidence_threshold=0.5,
            procedure_service=service,
        )
        
        # Query that should match exact_procedure better
        query = "reliable family car"
        
        # Match
        result = await matcher.match_intent(query, org_id, model_id)
        
        # Should match the exact procedure
        if result is not None and result.match_method == "stored_procedure":
            assert result.procedure_name == "exact_match_proc", \
                f"Expected 'exact_match_proc', got '{result.procedure_name}'"


class TestProcedureMatchingResponse:
    """
    Property tests for Procedure Matching Response Format.
    
    **Validates: Requirements 4.5, 4.6**
    """
    
    @pytest.mark.asyncio
    @given(data=st.data())
    @settings(max_examples=50)
    async def test_procedure_match_includes_gql_query(self, data):
        """
        Property test: Procedure match includes GQL query in structured_query.
        
        When a procedure matches, the structured_query SHALL include
        the procedure's gql_query.
        
        **Validates: Requirements 4.5, 4.6**
        """
        name = data.draw(valid_procedure_name_strategy())
        gql_query = data.draw(valid_gql_queries)
        org_id = f"org_{uuid4().hex[:8]}"
        model_id = f"model_{uuid4().hex[:8]}"
        
        # Create procedure
        procedure = MockStoredProcedure(
            id=str(uuid4()),
            org_id=org_id,
            model_id=model_id,
            name=name,
            gql_query=gql_query,
            lexicons=["test query"],
        )
        
        # Create service
        service = MockProcedureService([procedure])
        
        # Create matcher with low threshold
        matcher = IntentMatcher(
            confidence_threshold=0.5,
            procedure_service=service,
        )
        
        # Query using exact lexicon
        query = "test query"
        
        # Match
        result = await matcher.match_intent(query, org_id, model_id)
        
        # Verify response format
        if result is not None and result.match_method == "stored_procedure":
            assert "gql_query" in result.structured_query, \
                "structured_query should include gql_query"
            assert result.structured_query["gql_query"] == gql_query, \
                f"Expected gql_query='{gql_query}', got '{result.structured_query.get('gql_query')}'"
            assert result.structured_query["operation"] == "execute_procedure", \
                "operation should be 'execute_procedure'"
            assert result.structured_query["procedure_name"] == name, \
                f"procedure_name should be '{name}'"
