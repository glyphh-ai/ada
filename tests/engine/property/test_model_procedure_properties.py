"""
Property-based tests for GlyphhModel with stored procedures serialization.

This module contains property-based tests using Hypothesis to verify
the serialization round-trip for GlyphhModel with stored procedures.

**Validates: Property 10** - SDK Procedure Serialization Round-Trip
For any GlyphhModel with stored_procedures, exporting to .glyphh format
and then loading SHALL preserve all procedure data (name, gql_query,
lexicons, description).

**Validates: Requirements 8.2, 8.3, 8.5**
"""

import os
import tempfile
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite
from typing import List

from glyphh.model.package import GlyphhModel
from glyphh.core.config import EncoderConfig, Layer, Segment, Role
from glyphh.gql.stored_procedure import StoredProcedure


# =============================================================================
# Generator Strategies
# =============================================================================

# ASCII letters and digits for valid names
ASCII_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ASCII_LETTERS_DIGITS_UNDERSCORE = ASCII_LETTERS + "0123456789_"
ASCII_LEXICON_CHARS = ASCII_LETTERS + "0123456789 -_"


@composite
def valid_procedure_name_strategy(draw) -> str:
    """Generate a valid procedure name."""
    first = draw(st.sampled_from(ASCII_LETTERS))
    rest = draw(st.text(
        alphabet=ASCII_LETTERS_DIGITS_UNDERSCORE,
        min_size=0,
        max_size=20
    ))
    return first + rest


@composite
def valid_lexicon_strategy(draw) -> str:
    """Generate a valid lexicon entry."""
    return draw(st.text(
        alphabet=ASCII_LEXICON_CHARS,
        min_size=1,
        max_size=20
    ).filter(lambda s: s.strip()))


@composite
def valid_lexicons_list_strategy(draw) -> List[str]:
    """Generate a valid list of lexicons."""
    num_lexicons = draw(st.integers(min_value=1, max_value=5))
    return [draw(valid_lexicon_strategy()) for _ in range(num_lexicons)]


# Valid GQL queries for testing
valid_gql_queries = st.sampled_from([
    'LIST ALL LIMIT 10',
    'LIST ALL LIMIT 100',
    'COUNT ALL',
    'FIND SIMILAR TO "test query" LIMIT 10 THRESHOLD 0.5',
    'FIND SIMILAR TO "search term" LIMIT 5 THRESHOLD 0.8',
    'LIST ALL WHERE status = "active" LIMIT 20',
    'FIND SIMILAR TO "example" WHERE category = "test" LIMIT 10 THRESHOLD 0.6',
])


@composite
def valid_description_strategy(draw) -> str:
    """Generate a valid description."""
    return draw(st.text(
        alphabet=ASCII_LEXICON_CHARS,
        min_size=0,
        max_size=100
    ))


@composite
def stored_procedure_strategy(draw) -> StoredProcedure:
    """Generate a valid StoredProcedure."""
    return StoredProcedure(
        name=draw(valid_procedure_name_strategy()),
        gql_query=draw(valid_gql_queries),
        lexicons=draw(valid_lexicons_list_strategy()),
        description=draw(valid_description_strategy()),
    )


@composite
def stored_procedures_list_strategy(draw, min_size: int = 0, max_size: int = 5) -> List[StoredProcedure]:
    """Generate a list of stored procedures with unique names."""
    num_procedures = draw(st.integers(min_value=min_size, max_value=max_size))
    procedures = []
    used_names = set()
    
    for _ in range(num_procedures):
        # Generate unique name
        name = draw(valid_procedure_name_strategy())
        while name in used_names:
            name = draw(valid_procedure_name_strategy())
        used_names.add(name)
        
        procedures.append(StoredProcedure(
            name=name,
            gql_query=draw(valid_gql_queries),
            lexicons=draw(valid_lexicons_list_strategy()),
            description=draw(valid_description_strategy()),
        ))
    
    return procedures


def create_minimal_encoder_config() -> EncoderConfig:
    """Create a minimal valid encoder config for testing."""
    return EncoderConfig(
        dimension=1024,
        seed=42,
        layers=[
            Layer(
                name="test_layer",
                similarity_weight=1.0,
                security_weight=1.0,
                segments=[
                    Segment(
                        name="test_segment",
                        similarity_weight=1.0,
                        security_weight=1.0,
                        roles=[
                            Role(
                                name="test_role",
                                similarity_weight=1.0,
                                security_weight=1.0
                            )
                        ]
                    )
                ]
            )
        ]
    )


@composite
def glyphh_model_with_procedures_strategy(draw) -> GlyphhModel:
    """Generate a GlyphhModel with stored procedures."""
    procedures = draw(stored_procedures_list_strategy(min_size=1, max_size=5))
    
    return GlyphhModel(
        name=draw(st.text(alphabet=ASCII_LETTERS, min_size=1, max_size=20)),
        version="1.0.0",
        encoder_config=create_minimal_encoder_config(),
        glyphs=[],  # Empty glyphs for simplicity
        stored_procedures=procedures,
        metadata={"test": True},
    )


# =============================================================================
# Property Tests
# =============================================================================

class TestModelProcedureSerializationRoundTrip:
    """
    Property tests for GlyphhModel with stored procedures serialization.
    
    **Validates: Property 10** - SDK Procedure Serialization Round-Trip
    For any GlyphhModel with stored_procedures, exporting to .glyphh format
    and then loading SHALL preserve all procedure data.
    
    **Validates: Requirements 8.2, 8.3, 8.5**
    """
    
    @given(model=glyphh_model_with_procedures_strategy())
    @settings(max_examples=100)
    def test_serialization_round_trip_preserves_procedures(self, model: GlyphhModel):
        """
        Property test: to_file → from_file preserves stored procedures.
        
        For any GlyphhModel with stored_procedures, serializing to file
        and deserializing back SHALL preserve all procedure data.
        
        **Validates: Requirements 8.2, 8.3, 8.5**
        """
        with tempfile.NamedTemporaryFile(suffix='.glyphh', delete=False) as f:
            temp_path = f.name
        
        try:
            # Serialize to file
            model.to_file(temp_path)
            
            # Deserialize from file
            loaded_model = GlyphhModel.from_file(temp_path)
            
            # Verify procedure count
            assert len(loaded_model.stored_procedures) == len(model.stored_procedures), \
                f"Expected {len(model.stored_procedures)} procedures, got {len(loaded_model.stored_procedures)}"
            
            # Verify each procedure
            for original, loaded in zip(model.stored_procedures, loaded_model.stored_procedures):
                assert loaded.name == original.name, \
                    f"Name mismatch: {loaded.name} != {original.name}"
                assert loaded.gql_query == original.gql_query, \
                    f"GQL query mismatch: {loaded.gql_query} != {original.gql_query}"
                assert loaded.lexicons == original.lexicons, \
                    f"Lexicons mismatch: {loaded.lexicons} != {original.lexicons}"
                assert loaded.description == original.description, \
                    f"Description mismatch: {loaded.description} != {original.description}"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @given(procedures=stored_procedures_list_strategy(min_size=1, max_size=5))
    @settings(max_examples=100)
    def test_double_round_trip_produces_stable_output(self, procedures: List[StoredProcedure]):
        """
        Property test: Double round-trip produces stable output.
        
        For any GlyphhModel with stored_procedures, to_file → from_file →
        to_file → from_file SHALL produce the same result.
        
        **Validates: Requirements 8.2, 8.3, 8.5**
        """
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=create_minimal_encoder_config(),
            glyphs=[],
            stored_procedures=procedures,
        )
        
        with tempfile.NamedTemporaryFile(suffix='.glyphh', delete=False) as f:
            temp_path1 = f.name
        with tempfile.NamedTemporaryFile(suffix='.glyphh', delete=False) as f:
            temp_path2 = f.name
        
        try:
            # First round-trip
            model.to_file(temp_path1)
            loaded1 = GlyphhModel.from_file(temp_path1)
            
            # Second round-trip
            loaded1.to_file(temp_path2)
            loaded2 = GlyphhModel.from_file(temp_path2)
            
            # Verify procedures are identical
            assert len(loaded1.stored_procedures) == len(loaded2.stored_procedures)
            
            for proc1, proc2 in zip(loaded1.stored_procedures, loaded2.stored_procedures):
                assert proc1.name == proc2.name
                assert proc1.gql_query == proc2.gql_query
                assert proc1.lexicons == proc2.lexicons
                assert proc1.description == proc2.description
        finally:
            for path in [temp_path1, temp_path2]:
                if os.path.exists(path):
                    os.unlink(path)
    
    @given(model=glyphh_model_with_procedures_strategy())
    @settings(max_examples=100)
    def test_has_stored_procedures_returns_true(self, model: GlyphhModel):
        """
        Property test: has_stored_procedures returns True when procedures exist.
        
        For any GlyphhModel with at least one stored procedure,
        has_stored_procedures() SHALL return True.
        
        **Validates: Requirements 8.2**
        """
        assert model.has_stored_procedures() is True
    
    def test_has_stored_procedures_returns_false_when_empty(self):
        """
        Test: has_stored_procedures returns False when no procedures.
        
        **Validates: Requirements 8.2**
        """
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=create_minimal_encoder_config(),
            glyphs=[],
            stored_procedures=[],
        )
        
        assert model.has_stored_procedures() is False


class TestModelProcedureManagement:
    """
    Property tests for GlyphhModel stored procedure management methods.
    
    **Validates: Requirements 8.1, 8.3**
    """
    
    @given(procedure=stored_procedure_strategy())
    @settings(max_examples=100)
    def test_add_stored_procedure(self, procedure: StoredProcedure):
        """
        Property test: add_stored_procedure adds procedure to model.
        
        For any valid StoredProcedure, add_stored_procedure SHALL add
        it to the model's stored_procedures list.
        
        **Validates: Requirements 8.1**
        """
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=create_minimal_encoder_config(),
            glyphs=[],
        )
        
        model.add_stored_procedure(procedure)
        
        assert len(model.stored_procedures) == 1
        assert model.stored_procedures[0].name == procedure.name
    
    @given(procedure=stored_procedure_strategy())
    @settings(max_examples=100)
    def test_add_duplicate_procedure_raises_error(self, procedure: StoredProcedure):
        """
        Property test: Adding duplicate procedure raises ValueError.
        
        For any StoredProcedure already in the model, adding another
        procedure with the same name SHALL raise ValueError.
        
        **Validates: Requirements 8.1**
        """
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=create_minimal_encoder_config(),
            glyphs=[],
        )
        
        model.add_stored_procedure(procedure)
        
        # Create a duplicate with same name
        duplicate = StoredProcedure(
            name=procedure.name,
            gql_query='LIST ALL LIMIT 5',
            lexicons=["different"],
        )
        
        with pytest.raises(ValueError) as exc_info:
            model.add_stored_procedure(duplicate)
        
        assert procedure.name in str(exc_info.value)
    
    @given(procedure=stored_procedure_strategy())
    @settings(max_examples=100)
    def test_get_stored_procedure_returns_procedure(self, procedure: StoredProcedure):
        """
        Property test: get_stored_procedure returns the correct procedure.
        
        For any StoredProcedure in the model, get_stored_procedure(name)
        SHALL return that procedure.
        
        **Validates: Requirements 8.3**
        """
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=create_minimal_encoder_config(),
            glyphs=[],
            stored_procedures=[procedure],
        )
        
        retrieved = model.get_stored_procedure(procedure.name)
        
        assert retrieved is not None
        assert retrieved.name == procedure.name
        assert retrieved.gql_query == procedure.gql_query
    
    @given(procedure=stored_procedure_strategy())
    @settings(max_examples=100)
    def test_get_stored_procedure_returns_none_for_nonexistent(self, procedure: StoredProcedure):
        """
        Property test: get_stored_procedure returns None for nonexistent name.
        
        For any name not in the model's procedures, get_stored_procedure
        SHALL return None.
        
        **Validates: Requirements 8.3**
        """
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=create_minimal_encoder_config(),
            glyphs=[],
            stored_procedures=[procedure],
        )
        
        retrieved = model.get_stored_procedure("nonexistent_procedure")
        
        assert retrieved is None
    
    @given(procedure=stored_procedure_strategy())
    @settings(max_examples=100)
    def test_remove_stored_procedure(self, procedure: StoredProcedure):
        """
        Property test: remove_stored_procedure removes the procedure.
        
        For any StoredProcedure in the model, remove_stored_procedure(name)
        SHALL remove it and return True.
        
        **Validates: Requirements 8.3**
        """
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=create_minimal_encoder_config(),
            glyphs=[],
            stored_procedures=[procedure],
        )
        
        result = model.remove_stored_procedure(procedure.name)
        
        assert result is True
        assert len(model.stored_procedures) == 0
        assert model.get_stored_procedure(procedure.name) is None
    
    @given(procedure=stored_procedure_strategy())
    @settings(max_examples=100)
    def test_remove_nonexistent_procedure_returns_false(self, procedure: StoredProcedure):
        """
        Property test: remove_stored_procedure returns False for nonexistent.
        
        For any name not in the model's procedures, remove_stored_procedure
        SHALL return False.
        
        **Validates: Requirements 8.3**
        """
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=create_minimal_encoder_config(),
            glyphs=[],
            stored_procedures=[procedure],
        )
        
        result = model.remove_stored_procedure("nonexistent_procedure")
        
        assert result is False
        assert len(model.stored_procedures) == 1


class TestModelWithoutProcedures:
    """
    Tests for GlyphhModel backward compatibility without stored procedures.
    
    **Validates: Requirements 8.5**
    """
    
    def test_model_without_procedures_serializes_correctly(self):
        """
        Test: Model without procedures serializes and deserializes correctly.
        
        **Validates: Requirements 8.5**
        """
        model = GlyphhModel(
            name="test_model",
            version="1.0.0",
            encoder_config=create_minimal_encoder_config(),
            glyphs=[],
        )
        
        with tempfile.NamedTemporaryFile(suffix='.glyphh', delete=False) as f:
            temp_path = f.name
        
        try:
            model.to_file(temp_path)
            loaded = GlyphhModel.from_file(temp_path)
            
            assert loaded.name == model.name
            assert loaded.version == model.version
            assert len(loaded.stored_procedures) == 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
