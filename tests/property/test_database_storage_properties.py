"""
Property-based tests for DatabaseGlyphStorage.

This module contains property-based tests using Hypothesis to verify
correctness properties for the DatabaseGlyphStorage implementation.

Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness

**Validates: Requirements 5.3, 5.6**

For any list of GlyphResponse objects and embeddings dict passed to DatabaseGlyphStorage:
- get_embedding(id) returns the embedding from the dict
- get_embedding_for_scope(id, layer, segment) returns None when layer or segment is specified
  (since database glyphs don't have hierarchical structure)
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from hypothesis import given, settings, strategies as st, assume

from domains.gql.storage import DatabaseGlyphStorage
from domains.models.schemas import GlyphResponse


# =============================================================================
# Test Data Generators (Strategies)
# =============================================================================

@st.composite
def valid_org_id(draw) -> str:
    """Generate a valid org_id string."""
    prefix = draw(st.sampled_from(["org", "test", "prod", "dev"]))
    suffix = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=4,
        max_size=8,
    ))
    return f"{prefix}_{suffix}"


@st.composite
def valid_model_id(draw) -> str:
    """Generate a valid model_id string."""
    prefix = draw(st.sampled_from(["model", "test", "demo"]))
    suffix = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=4,
        max_size=8,
    ))
    return f"{prefix}_{suffix}"


@st.composite
def valid_concept_text(draw) -> str:
    """Generate valid concept text."""
    words = draw(st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz ",
            min_size=3,
            max_size=15,
        ),
        min_size=1,
        max_size=10,
    ))
    return " ".join(words).strip()


@st.composite
def valid_embedding(draw, dim: int = 128) -> List[float]:
    """
    Generate a valid embedding vector.
    
    Uses a smaller dimension (128) for faster test execution while
    still being representative of real embeddings.
    """
    import numpy as np
    
    values = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=dim,
        max_size=dim,
    ))
    # Normalize to unit length for realistic embeddings
    norm = np.linalg.norm(values)
    if norm > 0:
        values = [v / norm for v in values]
    return values


@st.composite
def valid_metadata(draw) -> Dict[str, Any]:
    """Generate valid metadata dictionary."""
    return draw(st.fixed_dictionaries({
        "source": st.sampled_from(["api", "listener", "batch", "import"]),
        "version": st.integers(min_value=1, max_value=100),
    }, optional={
        "tags": st.lists(st.text(min_size=1, max_size=20), max_size=5),
        "category": st.text(min_size=1, max_size=30),
    }))


@st.composite
def glyph_response_strategy(
    draw,
    org_id: Optional[str] = None,
    model_id: Optional[str] = None
) -> GlyphResponse:
    """
    Generate a valid GlyphResponse object for property testing.
    
    Args:
        org_id: Optional fixed org_id (generates one if not provided)
        model_id: Optional fixed model_id (generates one if not provided)
        
    Returns:
        GlyphResponse with random but valid attributes
    """
    return GlyphResponse(
        id=uuid4(),
        org_id=org_id or draw(valid_org_id()),
        model_id=model_id or draw(valid_model_id()),
        concept_text=draw(valid_concept_text()),
        metadata=draw(valid_metadata()),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@st.composite
def glyph_responses_with_embeddings_strategy(
    draw,
    min_glyphs: int = 0,
    max_glyphs: int = 10,
    org_id: Optional[str] = None,
    model_id: Optional[str] = None
) -> tuple[List[GlyphResponse], Dict[str, List[float]]]:
    """
    Generate a list of GlyphResponse objects with corresponding embeddings dict.
    
    Args:
        min_glyphs: Minimum number of glyphs to generate
        max_glyphs: Maximum number of glyphs to generate
        org_id: Optional fixed org_id for all glyphs
        model_id: Optional fixed model_id for all glyphs
        
    Returns:
        Tuple of (list of GlyphResponse, dict mapping glyph_id to embedding)
    """
    # Use fixed org_id and model_id if not provided
    fixed_org_id = org_id or draw(valid_org_id())
    fixed_model_id = model_id or draw(valid_model_id())
    
    num_glyphs = draw(st.integers(min_value=min_glyphs, max_value=max_glyphs))
    
    glyphs: List[GlyphResponse] = []
    embeddings: Dict[str, List[float]] = {}
    
    for _ in range(num_glyphs):
        glyph = draw(glyph_response_strategy(
            org_id=fixed_org_id,
            model_id=fixed_model_id
        ))
        glyphs.append(glyph)
        
        # Generate embedding for this glyph
        embedding = draw(valid_embedding())
        embeddings[str(glyph.id)] = embedding
    
    return glyphs, embeddings


# Layer and segment name generators for scope testing
layer_names = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-'),
    min_size=1,
    max_size=30
).filter(lambda x: x.strip())

segment_names = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-'),
    min_size=1,
    max_size=30
).filter(lambda x: x.strip())


# =============================================================================
# Property 5: DatabaseGlyphStorage Correctness
# =============================================================================

class TestDatabaseGlyphStorageCorrectness:
    """
    Property tests for DatabaseGlyphStorage Correctness (Property 5).
    
    Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
    
    **Validates: Requirements 5.3, 5.6**
    
    For any list of GlyphResponse objects and embeddings dict passed to DatabaseGlyphStorage:
    - get_embedding(id) returns the embedding from the dict
    - get_embedding_for_scope(id, layer, segment) returns None when layer or segment is specified
      (since database glyphs don't have hierarchical structure)
    """
    
    @given(data=glyph_responses_with_embeddings_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_get_embedding_returns_from_embeddings_dict(
        self, data: tuple[List[GlyphResponse], Dict[str, List[float]]]
    ):
        """
        Property test: get_embedding(id) returns the embedding from the embeddings dict.
        
        For any glyph_id in the storage, get_embedding() SHALL return the
        corresponding embedding from the embeddings dict passed at construction.
        
        Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
        **Validates: Requirements 5.3**
        """
        glyphs, embeddings = data
        
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated data
        storage = DatabaseGlyphStorage(
            org_id=glyphs[0].org_id,
            model_id=glyphs[0].model_id,
            glyphs=glyphs,
            embeddings=embeddings,
        )
        
        # For every glyph, verify get_embedding returns the correct embedding
        for glyph in glyphs:
            glyph_id = str(glyph.id)
            retrieved_embedding = storage.get_embedding(glyph_id)
            expected_embedding = embeddings.get(glyph_id)
            
            # Property: get_embedding should return the embedding from the dict
            assert retrieved_embedding == expected_embedding, \
                f"get_embedding('{glyph_id}') returned wrong embedding"
    
    @given(data=glyph_responses_with_embeddings_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_get_embedding_returns_none_for_unknown_id(
        self, data: tuple[List[GlyphResponse], Dict[str, List[float]]]
    ):
        """
        Property test: get_embedding(id) returns None for unknown glyph IDs.
        
        For any glyph_id NOT in the embeddings dict, get_embedding() SHALL
        return None.
        
        Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
        **Validates: Requirements 5.3**
        """
        glyphs, embeddings = data
        
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated data
        storage = DatabaseGlyphStorage(
            org_id=glyphs[0].org_id,
            model_id=glyphs[0].model_id,
            glyphs=glyphs,
            embeddings=embeddings,
        )
        
        # Generate an unknown glyph ID
        unknown_id = str(uuid4())
        
        # Ensure it's not in the embeddings dict
        assume(unknown_id not in embeddings)
        
        # Property: get_embedding should return None for unknown IDs
        retrieved_embedding = storage.get_embedding(unknown_id)
        assert retrieved_embedding is None, \
            f"get_embedding('{unknown_id}') should return None for unknown ID"
    
    @given(
        data=glyph_responses_with_embeddings_strategy(min_glyphs=1, max_glyphs=10),
        layer=layer_names
    )
    @settings(max_examples=100)
    def test_get_embedding_for_scope_returns_none_when_layer_specified(
        self, data: tuple[List[GlyphResponse], Dict[str, List[float]]], layer: str
    ):
        """
        Property test: get_embedding_for_scope(id, layer, None) returns None.
        
        When layer is specified, get_embedding_for_scope() SHALL return None
        since database glyphs don't have hierarchical structure.
        
        Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
        **Validates: Requirements 5.6**
        """
        glyphs, embeddings = data
        
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated data
        storage = DatabaseGlyphStorage(
            org_id=glyphs[0].org_id,
            model_id=glyphs[0].model_id,
            glyphs=glyphs,
            embeddings=embeddings,
        )
        
        # For every glyph, verify get_embedding_for_scope returns None when layer is specified
        for glyph in glyphs:
            glyph_id = str(glyph.id)
            scoped_embedding = storage.get_embedding_for_scope(
                glyph_id, layer=layer, segment=None
            )
            
            # Property: Should return None when layer is specified
            assert scoped_embedding is None, \
                f"get_embedding_for_scope('{glyph_id}', layer='{layer}', segment=None) " \
                f"should return None for database glyphs"
    
    @given(
        data=glyph_responses_with_embeddings_strategy(min_glyphs=1, max_glyphs=10),
        segment=segment_names
    )
    @settings(max_examples=100)
    def test_get_embedding_for_scope_returns_none_when_segment_specified(
        self, data: tuple[List[GlyphResponse], Dict[str, List[float]]], segment: str
    ):
        """
        Property test: get_embedding_for_scope(id, None, segment) returns None.
        
        When segment is specified (even without layer), get_embedding_for_scope()
        SHALL return None since database glyphs don't have hierarchical structure.
        
        Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
        **Validates: Requirements 5.6**
        """
        glyphs, embeddings = data
        
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated data
        storage = DatabaseGlyphStorage(
            org_id=glyphs[0].org_id,
            model_id=glyphs[0].model_id,
            glyphs=glyphs,
            embeddings=embeddings,
        )
        
        # For every glyph, verify get_embedding_for_scope returns None when segment is specified
        for glyph in glyphs:
            glyph_id = str(glyph.id)
            scoped_embedding = storage.get_embedding_for_scope(
                glyph_id, layer=None, segment=segment
            )
            
            # Property: Should return None when segment is specified
            assert scoped_embedding is None, \
                f"get_embedding_for_scope('{glyph_id}', layer=None, segment='{segment}') " \
                f"should return None for database glyphs"
    
    @given(
        data=glyph_responses_with_embeddings_strategy(min_glyphs=1, max_glyphs=10),
        layer=layer_names,
        segment=segment_names
    )
    @settings(max_examples=100)
    def test_get_embedding_for_scope_returns_none_when_both_layer_and_segment_specified(
        self,
        data: tuple[List[GlyphResponse], Dict[str, List[float]]],
        layer: str,
        segment: str
    ):
        """
        Property test: get_embedding_for_scope(id, layer, segment) returns None.
        
        When both layer and segment are specified, get_embedding_for_scope()
        SHALL return None since database glyphs don't have hierarchical structure.
        
        Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
        **Validates: Requirements 5.6**
        """
        glyphs, embeddings = data
        
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated data
        storage = DatabaseGlyphStorage(
            org_id=glyphs[0].org_id,
            model_id=glyphs[0].model_id,
            glyphs=glyphs,
            embeddings=embeddings,
        )
        
        # For every glyph, verify get_embedding_for_scope returns None
        for glyph in glyphs:
            glyph_id = str(glyph.id)
            scoped_embedding = storage.get_embedding_for_scope(
                glyph_id, layer=layer, segment=segment
            )
            
            # Property: Should return None when both layer and segment are specified
            assert scoped_embedding is None, \
                f"get_embedding_for_scope('{glyph_id}', layer='{layer}', segment='{segment}') " \
                f"should return None for database glyphs"
    
    @given(data=glyph_responses_with_embeddings_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_get_embedding_for_scope_returns_primary_when_no_scope(
        self, data: tuple[List[GlyphResponse], Dict[str, List[float]]]
    ):
        """
        Property test: get_embedding_for_scope(id, None, None) returns primary embedding.
        
        When neither layer nor segment is specified, get_embedding_for_scope()
        SHALL return the same result as get_embedding().
        
        Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
        **Validates: Requirements 5.3, 5.6**
        """
        glyphs, embeddings = data
        
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated data
        storage = DatabaseGlyphStorage(
            org_id=glyphs[0].org_id,
            model_id=glyphs[0].model_id,
            glyphs=glyphs,
            embeddings=embeddings,
        )
        
        # For every glyph, verify get_embedding_for_scope with no scope
        for glyph in glyphs:
            glyph_id = str(glyph.id)
            scoped_embedding = storage.get_embedding_for_scope(
                glyph_id, layer=None, segment=None
            )
            primary_embedding = storage.get_embedding(glyph_id)
            
            # Property: Should return the same result as get_embedding
            assert scoped_embedding == primary_embedding, \
                f"get_embedding_for_scope('{glyph_id}', None, None) should equal get_embedding()"
    
    @given(data=glyph_responses_with_embeddings_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_embedding_identity_preserved(
        self, data: tuple[List[GlyphResponse], Dict[str, List[float]]]
    ):
        """
        Property test: Embedding values are preserved exactly.
        
        The embedding returned by get_embedding() SHALL be exactly equal
        to the embedding passed in the embeddings dict (no modification).
        
        Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
        **Validates: Requirements 5.3**
        """
        glyphs, embeddings = data
        
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated data
        storage = DatabaseGlyphStorage(
            org_id=glyphs[0].org_id,
            model_id=glyphs[0].model_id,
            glyphs=glyphs,
            embeddings=embeddings,
        )
        
        # For every glyph, verify embedding values are preserved exactly
        for glyph in glyphs:
            glyph_id = str(glyph.id)
            retrieved_embedding = storage.get_embedding(glyph_id)
            expected_embedding = embeddings[glyph_id]
            
            # Property: Values should be exactly equal
            assert len(retrieved_embedding) == len(expected_embedding), \
                f"Embedding length mismatch for '{glyph_id}'"
            
            for i, (retrieved, expected) in enumerate(zip(retrieved_embedding, expected_embedding)):
                assert retrieved == expected, \
                    f"Embedding value mismatch at index {i} for '{glyph_id}': " \
                    f"got {retrieved}, expected {expected}"


# =============================================================================
# Additional Storage Protocol Consistency Tests for DatabaseGlyphStorage
# =============================================================================

class TestDatabaseGlyphStorageProtocolConsistency:
    """
    Additional property tests for DatabaseGlyphStorage protocol consistency.
    
    These tests verify that DatabaseGlyphStorage correctly implements
    the GlyphStorageProtocol interface.
    
    Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
    **Validates: Requirements 5.3, 5.6**
    """
    
    @given(data=glyph_responses_with_embeddings_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_has_glyph_get_glyph_consistency(
        self, data: tuple[List[GlyphResponse], Dict[str, List[float]]]
    ):
        """
        Property test: has_glyph and get_glyph are consistent.
        
        For any glyph_id, has_glyph(id) == True if and only if
        get_glyph(id) succeeds without raising KeyError.
        
        Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
        **Validates: Requirements 5.3**
        """
        glyphs, embeddings = data
        
        # Handle empty case
        if len(glyphs) == 0:
            storage = DatabaseGlyphStorage(
                org_id="test_org",
                model_id="test_model",
                glyphs=[],
                embeddings={},
            )
            # Verify unknown ID behavior
            unknown_id = str(uuid4())
            assert not storage.has_glyph(unknown_id)
            with pytest.raises(KeyError):
                storage.get_glyph(unknown_id)
            return
        
        # Create storage with the generated data
        storage = DatabaseGlyphStorage(
            org_id=glyphs[0].org_id,
            model_id=glyphs[0].model_id,
            glyphs=glyphs,
            embeddings=embeddings,
        )
        
        # For every known glyph, verify consistency
        for glyph in glyphs:
            glyph_id = str(glyph.id)
            
            # has_glyph should return True
            assert storage.has_glyph(glyph_id), \
                f"has_glyph('{glyph_id}') should return True for known glyph"
            
            # get_glyph should succeed
            try:
                retrieved = storage.get_glyph(glyph_id)
                assert retrieved is not None
                assert retrieved.id == glyph.id
            except KeyError:
                pytest.fail(f"get_glyph('{glyph_id}') raised KeyError for known glyph")
        
        # For unknown ID, verify consistency
        unknown_id = str(uuid4())
        assume(unknown_id not in {str(g.id) for g in glyphs})
        
        assert not storage.has_glyph(unknown_id), \
            f"has_glyph('{unknown_id}') should return False for unknown glyph"
        
        with pytest.raises(KeyError):
            storage.get_glyph(unknown_id)
    
    @given(data=glyph_responses_with_embeddings_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_list_glyphs_consistent_with_has_glyph(
        self, data: tuple[List[GlyphResponse], Dict[str, List[float]]]
    ):
        """
        Property test: list_glyphs is consistent with has_glyph.
        
        For any storage, every glyph_id returned by list_glyphs SHALL have
        has_glyph return True.
        
        Feature: gql-storage-abstraction, Property 5: DatabaseGlyphStorage Correctness
        **Validates: Requirements 5.3**
        """
        glyphs, embeddings = data
        
        # Handle empty case
        if len(glyphs) == 0:
            storage = DatabaseGlyphStorage(
                org_id="test_org",
                model_id="test_model",
                glyphs=[],
                embeddings={},
            )
            assert len(storage.list_glyphs()) == 0
            return
        
        # Create storage with the generated data
        storage = DatabaseGlyphStorage(
            org_id=glyphs[0].org_id,
            model_id=glyphs[0].model_id,
            glyphs=glyphs,
            embeddings=embeddings,
        )
        
        # Get all glyphs from list_glyphs
        listed_glyphs = storage.list_glyphs()
        
        # Verify count matches
        assert len(listed_glyphs) == len(glyphs), \
            f"list_glyphs() returned {len(listed_glyphs)} glyphs, expected {len(glyphs)}"
        
        # For every listed glyph ID, has_glyph must return True
        for glyph_id in listed_glyphs.keys():
            assert storage.has_glyph(glyph_id), \
                f"has_glyph('{glyph_id}') returned False for glyph in list_glyphs"
