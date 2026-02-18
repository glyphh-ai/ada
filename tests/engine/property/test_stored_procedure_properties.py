"""
Property-based tests for StoredProcedure validation.

This module contains property-based tests using Hypothesis to verify
the validation behavior of the StoredProcedure class.

**Validates: Property 2** - StoredProcedure Validation
For any StoredProcedure creation attempt:
- If the name does not match ^[a-zA-Z][a-zA-Z0-9_]*$, creation SHALL fail
- If the gql_query is not valid GQL, creation SHALL fail
- If lexicons is empty or contains only whitespace, creation SHALL fail
- If all validations pass, creation SHALL succeed

**Validates: Requirements 2.2, 2.3, 2.4, 2.5**
"""

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite
from typing import List

from glyphh.gql.stored_procedure import StoredProcedure


# =============================================================================
# Generator Strategies for StoredProcedure Data
# =============================================================================

# ASCII letters and digits for valid names (no Unicode)
ASCII_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ASCII_LETTERS_DIGITS_UNDERSCORE = ASCII_LETTERS + "0123456789_"
ASCII_LEXICON_CHARS = ASCII_LETTERS + "0123456789 -_"


# Strategy for valid procedure names
# Must start with a letter and contain only alphanumeric characters and underscores
@composite
def valid_name_strategy(draw) -> str:
    """Generate a valid procedure name."""
    first = draw(st.sampled_from(ASCII_LETTERS))
    rest = draw(st.text(
        alphabet=ASCII_LETTERS_DIGITS_UNDERSCORE,
        min_size=0,
        max_size=30
    ))
    return first + rest


# Strategy for invalid names - names that should fail validation
@composite
def invalid_name_starting_with_number(draw) -> str:
    """Generate a name starting with a number (invalid)."""
    digit = draw(st.sampled_from("0123456789"))
    rest = draw(st.text(
        alphabet=ASCII_LETTERS_DIGITS_UNDERSCORE,
        min_size=0,
        max_size=20
    ))
    return digit + rest


@composite
def invalid_name_with_special_chars(draw) -> str:
    """Generate a name with special characters (invalid)."""
    # Start with a valid letter
    first = draw(st.sampled_from(ASCII_LETTERS))
    # Add some valid chars
    middle = draw(st.text(
        alphabet=ASCII_LETTERS_DIGITS_UNDERSCORE,
        min_size=0,
        max_size=10
    ))
    # Add a special character
    special = draw(st.sampled_from("!@#$%^&*()-+=[]{}|;:',.<>?/~` "))
    # Add more valid chars
    end = draw(st.text(
        alphabet=ASCII_LETTERS_DIGITS_UNDERSCORE,
        min_size=0,
        max_size=10
    ))
    return first + middle + special + end


# Strategy for valid GQL queries
# Using simple, known-valid GQL query patterns
# Note: Avoid using reserved keywords like 'count' as field names
valid_gql_queries = st.sampled_from([
    'LIST ALL LIMIT 10',
    'LIST ALL LIMIT 100',
    'COUNT ALL',
    'FIND SIMILAR TO "test query" LIMIT 10 THRESHOLD 0.5',
    'FIND SIMILAR TO "search term" LIMIT 5 THRESHOLD 0.8',
    'LIST ALL WHERE status = "active" LIMIT 20',
    'LIST ALL WHERE quantity > 5 LIMIT 10',
    'FIND SIMILAR TO "example" WHERE category = "test" LIMIT 10 THRESHOLD 0.6',
])


# Strategy for invalid GQL queries
invalid_gql_queries = st.sampled_from([
    '',  # Empty query
    'INVALID QUERY',  # Unknown command
    'LIST',  # Incomplete LIST
    'FIND SIMILAR',  # Incomplete FIND SIMILAR
    'LIST ALL LIMIT',  # Missing limit value
    'FIND SIMILAR TO LIMIT 10',  # Missing target
    'WHERE x = 1',  # WHERE without command
    '123 456',  # Random numbers
])


# Strategy for valid lexicons
@composite
def valid_lexicon_strategy(draw) -> str:
    """Generate a valid lexicon entry."""
    return draw(st.text(
        alphabet=ASCII_LEXICON_CHARS,
        min_size=1,
        max_size=30
    ).filter(lambda s: s.strip()))  # Ensure non-empty after stripping


@composite
def valid_lexicons_list_strategy(draw) -> List[str]:
    """Generate a valid list of lexicons (at least one non-empty entry)."""
    num_lexicons = draw(st.integers(min_value=1, max_value=5))
    lexicons = [draw(valid_lexicon_strategy()) for _ in range(num_lexicons)]
    return lexicons


# Strategy for invalid lexicons (empty or whitespace-only)
empty_lexicons_strategy = st.just([])

whitespace_only_lexicons_strategy = st.lists(
    st.sampled_from(["", "   ", "\t", "\n", "  \t  "]),
    min_size=1,
    max_size=5
)


# Strategy for valid description
valid_description_strategy = st.text(
    alphabet=ASCII_LEXICON_CHARS,
    min_size=0,
    max_size=100
)


# =============================================================================
# Composite Strategies for Complete StoredProcedure Data
# =============================================================================

@composite
def valid_stored_procedure_data(draw) -> dict:
    """Generate valid StoredProcedure data that should pass validation."""
    return {
        "name": draw(valid_name_strategy()),
        "gql_query": draw(valid_gql_queries),
        "lexicons": draw(valid_lexicons_list_strategy()),
        "description": draw(valid_description_strategy),
    }


# =============================================================================
# Property Tests
# =============================================================================

class TestStoredProcedureValidation:
    """
    Property tests for StoredProcedure Validation (Property 2).
    
    **Validates: Property 2** - StoredProcedure Validation
    For any StoredProcedure creation attempt:
    - If the name does not match ^[a-zA-Z][a-zA-Z0-9_]*$, creation SHALL fail
    - If the gql_query is not valid GQL, creation SHALL fail
    - If lexicons is empty or contains only whitespace, creation SHALL fail
    - If all validations pass, creation SHALL succeed
    
    **Validates: Requirements 2.2, 2.3, 2.4, 2.5**
    """
    
    @given(data=valid_stored_procedure_data())
    @settings(max_examples=100)
    def test_valid_procedure_creation_succeeds(self, data: dict):
        """
        Property test: Valid procedure data creates successfully.
        
        For any valid name, gql_query, and lexicons, StoredProcedure
        creation SHALL succeed.
        
        **Validates: Requirements 2.2, 2.3, 2.4, 2.5**
        """
        procedure = StoredProcedure(
            name=data["name"],
            gql_query=data["gql_query"],
            lexicons=data["lexicons"],
            description=data["description"],
        )
        
        # Verify the procedure was created with correct values
        assert procedure.name == data["name"]
        assert procedure.gql_query == data["gql_query"]
        # Lexicons are cleaned (lowercased and stripped)
        assert len(procedure.lexicons) > 0
        assert procedure.description == data["description"]
    
    @given(name=invalid_name_starting_with_number())
    @settings(max_examples=100)
    def test_name_starting_with_number_rejected(self, name: str):
        """
        Property test: Names starting with a number are rejected.
        
        For any name starting with a digit, StoredProcedure creation
        SHALL fail with a ValueError.
        
        **Validates: Requirement 2.5**
        """
        with pytest.raises(ValueError) as exc_info:
            StoredProcedure(
                name=name,
                gql_query='LIST ALL LIMIT 10',
                lexicons=["test"],
            )
        
        assert "Invalid procedure name" in str(exc_info.value) or "name" in str(exc_info.value).lower()
    
    @given(name=invalid_name_with_special_chars())
    @settings(max_examples=100)
    def test_name_with_special_chars_rejected(self, name: str):
        """
        Property test: Names with special characters are rejected.
        
        For any name containing special characters (not alphanumeric or underscore),
        StoredProcedure creation SHALL fail with a ValueError.
        
        **Validates: Requirement 2.5**
        """
        with pytest.raises(ValueError) as exc_info:
            StoredProcedure(
                name=name,
                gql_query='LIST ALL LIMIT 10',
                lexicons=["test"],
            )
        
        assert "Invalid procedure name" in str(exc_info.value) or "name" in str(exc_info.value).lower()
    
    @given(invalid_query=invalid_gql_queries)
    @settings(max_examples=100)
    def test_invalid_gql_query_rejected(self, invalid_query: str):
        """
        Property test: Invalid GQL queries are rejected.
        
        For any invalid GQL query string, StoredProcedure creation
        SHALL fail with a ValueError.
        
        **Validates: Requirement 2.3**
        """
        with pytest.raises(ValueError) as exc_info:
            StoredProcedure(
                name="valid_name",
                gql_query=invalid_query,
                lexicons=["test"],
            )
        
        error_msg = str(exc_info.value).lower()
        assert "gql" in error_msg or "query" in error_msg or "empty" in error_msg
    
    @given(lexicons=empty_lexicons_strategy)
    @settings(max_examples=100)
    def test_empty_lexicons_rejected(self, lexicons: List[str]):
        """
        Property test: Empty lexicons list is rejected.
        
        For an empty lexicons list, StoredProcedure creation
        SHALL fail with a ValueError.
        
        **Validates: Requirement 2.4**
        """
        with pytest.raises(ValueError) as exc_info:
            StoredProcedure(
                name="valid_name",
                gql_query='LIST ALL LIMIT 10',
                lexicons=lexicons,
            )
        
        assert "lexicon" in str(exc_info.value).lower()
    
    @given(lexicons=whitespace_only_lexicons_strategy)
    @settings(max_examples=100)
    def test_whitespace_only_lexicons_rejected(self, lexicons: List[str]):
        """
        Property test: Whitespace-only lexicons are rejected.
        
        For a lexicons list containing only whitespace entries,
        StoredProcedure creation SHALL fail with a ValueError.
        
        **Validates: Requirement 2.4**
        """
        with pytest.raises(ValueError) as exc_info:
            StoredProcedure(
                name="valid_name",
                gql_query='LIST ALL LIMIT 10',
                lexicons=lexicons,
            )
        
        assert "lexicon" in str(exc_info.value).lower()


class TestStoredProcedureSerializationRoundTrip:
    """
    Property tests for StoredProcedure serialization round-trip.
    
    For any valid StoredProcedure, serializing to dict and deserializing
    back SHALL produce an equivalent procedure.
    
    **Validates: Requirements 2.2, 2.3, 2.4, 2.5**
    """
    
    @given(data=valid_stored_procedure_data())
    @settings(max_examples=100)
    def test_serialization_round_trip(self, data: dict):
        """
        Property test: to_dict → from_dict produces equivalent procedure.
        
        For any valid StoredProcedure, serializing to dictionary and
        deserializing back SHALL produce a procedure with the same values.
        
        **Validates: Requirements 2.2, 2.3, 2.4, 2.5**
        """
        # Create original procedure
        original = StoredProcedure(
            name=data["name"],
            gql_query=data["gql_query"],
            lexicons=data["lexicons"],
            description=data["description"],
        )
        
        # Serialize to dict
        serialized = original.to_dict()
        
        # Deserialize back
        restored = StoredProcedure.from_dict(serialized)
        
        # Verify equivalence
        assert restored.name == original.name
        assert restored.gql_query == original.gql_query
        assert restored.lexicons == original.lexicons
        assert restored.description == original.description
    
    @given(data=valid_stored_procedure_data())
    @settings(max_examples=100)
    def test_double_round_trip(self, data: dict):
        """
        Property test: Double round-trip produces stable output.
        
        For any valid StoredProcedure, to_dict → from_dict → to_dict → from_dict
        SHALL produce the same result as a single round-trip.
        
        **Validates: Requirements 2.2, 2.3, 2.4, 2.5**
        """
        # Create original procedure
        original = StoredProcedure(
            name=data["name"],
            gql_query=data["gql_query"],
            lexicons=data["lexicons"],
            description=data["description"],
        )
        
        # First round-trip
        dict1 = original.to_dict()
        restored1 = StoredProcedure.from_dict(dict1)
        
        # Second round-trip
        dict2 = restored1.to_dict()
        restored2 = StoredProcedure.from_dict(dict2)
        
        # Dictionaries should be identical
        assert dict1 == dict2
        
        # Procedures should be equivalent
        assert restored1.name == restored2.name
        assert restored1.gql_query == restored2.gql_query
        assert restored1.lexicons == restored2.lexicons
        assert restored1.description == restored2.description


class TestStoredProcedureLexiconCleaning:
    """
    Property tests for lexicon cleaning behavior.
    
    Lexicons should be cleaned (stripped and lowercased) during validation.
    
    **Validates: Requirements 2.4**
    """
    
    @given(data=valid_stored_procedure_data())
    @settings(max_examples=100)
    def test_lexicons_are_lowercased(self, data: dict):
        """
        Property test: Lexicons are lowercased during validation.
        
        For any valid StoredProcedure, all lexicons SHALL be lowercased.
        
        **Validates: Requirement 2.4**
        """
        procedure = StoredProcedure(
            name=data["name"],
            gql_query=data["gql_query"],
            lexicons=data["lexicons"],
            description=data["description"],
        )
        
        for lexicon in procedure.lexicons:
            assert lexicon == lexicon.lower(), \
                f"Lexicon '{lexicon}' should be lowercased"
    
    @given(data=valid_stored_procedure_data())
    @settings(max_examples=100)
    def test_lexicons_are_stripped(self, data: dict):
        """
        Property test: Lexicons are stripped during validation.
        
        For any valid StoredProcedure, all lexicons SHALL have no
        leading or trailing whitespace.
        
        **Validates: Requirement 2.4**
        """
        procedure = StoredProcedure(
            name=data["name"],
            gql_query=data["gql_query"],
            lexicons=data["lexicons"],
            description=data["description"],
        )
        
        for lexicon in procedure.lexicons:
            assert lexicon == lexicon.strip(), \
                f"Lexicon '{lexicon}' should be stripped"


class TestStoredProcedureNameValidation:
    """
    Additional property tests for name validation edge cases.
    
    **Validates: Requirement 2.5**
    """
    
    def test_empty_name_rejected(self):
        """
        Test: Empty name is rejected.
        
        **Validates: Requirement 2.5**
        """
        with pytest.raises(ValueError) as exc_info:
            StoredProcedure(
                name="",
                gql_query='LIST ALL LIMIT 10',
                lexicons=["test"],
            )
        
        assert "name" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()
    
    @given(name=valid_name_strategy())
    @settings(max_examples=100)
    def test_valid_names_accepted(self, name: str):
        """
        Property test: Valid names are accepted.
        
        For any name matching ^[a-zA-Z][a-zA-Z0-9_]*$, StoredProcedure
        creation SHALL succeed.
        
        **Validates: Requirement 2.5**
        """
        procedure = StoredProcedure(
            name=name,
            gql_query='LIST ALL LIMIT 10',
            lexicons=["test"],
        )
        
        assert procedure.name == name
    
    @given(
        first_char=st.sampled_from(ASCII_LETTERS),
        rest=st.text(
            alphabet=ASCII_LETTERS_DIGITS_UNDERSCORE,
            min_size=0,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_names_with_underscores_accepted(self, first_char: str, rest: str):
        """
        Property test: Names with underscores are accepted.
        
        For any name starting with a letter and containing underscores,
        StoredProcedure creation SHALL succeed.
        
        **Validates: Requirement 2.5**
        """
        name = first_char + rest
        procedure = StoredProcedure(
            name=name,
            gql_query='LIST ALL LIMIT 10',
            lexicons=["test"],
        )
        
        assert procedure.name == name
