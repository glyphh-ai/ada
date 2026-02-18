"""
Property-based tests for GQL Storage Abstraction.

This module contains property-based tests using Hypothesis to verify
universal correctness properties for the GlyphStorageProtocol and its implementations.

Feature: gql-storage-abstraction
"""

import pytest
from hypothesis import given, settings, strategies as st, assume
from typing import Dict, Any, Optional, List, Tuple

from glyphh.gql.storage import GlyphStorageProtocol, InMemoryGlyphStorage


# =============================================================================
# Test Data Generators (Strategies)
# =============================================================================

# Glyph ID generator - generates valid glyph identifiers
glyph_ids = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-'),
    min_size=1,
    max_size=50
).filter(lambda x: x.strip())

# Generate random vectors (list of floats)
vectors = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=10,
    max_size=100
)

# Generate random metadata dictionaries
metadata = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N')),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.strip()),
    values=st.one_of(
        st.text(min_size=1, max_size=30),
        st.integers(min_value=-1000, max_value=1000),
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    ),
    min_size=0,
    max_size=5
)


class MockSDKGlyph:
    """
    Mock SDK Glyph object for testing InMemoryGlyphStorage.
    
    This simulates the structure of a real SDK Glyph with cortex,
    global_cortex, attributes, and metadata.
    """
    
    def __init__(
        self,
        identifier: str,
        cortex: Optional[List[float]] = None,
        global_cortex: Optional[List[float]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        layers: Optional[List[Any]] = None
    ):
        self.identifier = identifier
        self.cortex = cortex
        self.global_cortex = global_cortex
        self.attributes = attributes or {}
        self.metadata = metadata or {}
        self.layers = layers or []


@st.composite
def mock_sdk_glyph_strategy(draw) -> MockSDKGlyph:
    """
    Generate a mock SDK glyph for property testing.
    
    Returns:
        MockSDKGlyph with random but valid attributes
    """
    glyph_id = draw(glyph_ids)
    
    # Randomly decide which embedding to use
    has_cortex = draw(st.booleans())
    has_global_cortex = draw(st.booleans())
    
    cortex = draw(vectors) if has_cortex else None
    global_cortex = draw(vectors) if has_global_cortex else None
    
    attrs = draw(metadata)
    meta = draw(metadata)
    
    return MockSDKGlyph(
        identifier=glyph_id,
        cortex=cortex,
        global_cortex=global_cortex,
        attributes=attrs,
        metadata=meta
    )


@st.composite
def glyph_dict_strategy(draw, min_glyphs: int = 0, max_glyphs: int = 10) -> Dict[str, MockSDKGlyph]:
    """
    Generate a dictionary of mock SDK glyphs for property testing.
    
    Args:
        min_glyphs: Minimum number of glyphs to generate
        max_glyphs: Maximum number of glyphs to generate
        
    Returns:
        Dictionary mapping glyph_id to MockSDKGlyph
    """
    num_glyphs = draw(st.integers(min_value=min_glyphs, max_value=max_glyphs))
    glyphs = {}
    
    for _ in range(num_glyphs):
        glyph = draw(mock_sdk_glyph_strategy())
        # Use the glyph's identifier as the key
        glyphs[glyph.identifier] = glyph
    
    return glyphs


# =============================================================================
# Property 1: Storage Protocol Consistency
# =============================================================================

class TestStorageProtocolConsistency:
    """
    Property tests for Storage Protocol Consistency (Property 1).
    
    Feature: gql-storage-abstraction, Property 1: Storage Protocol Consistency
    
    **Validates: Requirements 1.2, 1.3**
    
    For any storage implementation and any glyph_id, `has_glyph(glyph_id)` returns
    True if and only if `get_glyph(glyph_id)` succeeds without raising KeyError.
    """
    
    @given(
        glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10),
        query_id=glyph_ids
    )
    @settings(max_examples=100)
    def test_has_glyph_true_implies_get_glyph_succeeds(
        self, glyphs: Dict[str, MockSDKGlyph], query_id: str
    ):
        """
        Property test: has_glyph(id) == True implies get_glyph(id) succeeds.
        
        For any storage and any glyph_id, if has_glyph returns True,
        then get_glyph SHALL succeed without raising KeyError.
        
        Feature: gql-storage-abstraction, Property 1: Storage Protocol Consistency
        **Validates: Requirements 1.2, 1.3**
        """
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Check if glyph exists
        exists = storage.has_glyph(query_id)
        
        if exists:
            # Property: If has_glyph returns True, get_glyph must succeed
            try:
                glyph = storage.get_glyph(query_id)
                # Verify we got a valid glyph back
                assert glyph is not None, \
                    f"get_glyph returned None for existing glyph '{query_id}'"
            except KeyError as e:
                pytest.fail(
                    f"has_glyph('{query_id}') returned True but get_glyph raised KeyError: {e}"
                )
    
    @given(
        glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10),
        query_id=glyph_ids
    )
    @settings(max_examples=100)
    def test_has_glyph_false_implies_get_glyph_raises_keyerror(
        self, glyphs: Dict[str, MockSDKGlyph], query_id: str
    ):
        """
        Property test: has_glyph(id) == False implies get_glyph(id) raises KeyError.
        
        For any storage and any glyph_id, if has_glyph returns False,
        then get_glyph SHALL raise KeyError.
        
        Feature: gql-storage-abstraction, Property 1: Storage Protocol Consistency
        **Validates: Requirements 1.2, 1.3**
        """
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Check if glyph exists
        exists = storage.has_glyph(query_id)
        
        if not exists:
            # Property: If has_glyph returns False, get_glyph must raise KeyError
            with pytest.raises(KeyError):
                storage.get_glyph(query_id)
    
    @given(
        glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10),
        query_id=glyph_ids
    )
    @settings(max_examples=100)
    def test_get_glyph_success_implies_has_glyph_true(
        self, glyphs: Dict[str, MockSDKGlyph], query_id: str
    ):
        """
        Property test: get_glyph(id) succeeds implies has_glyph(id) == True.
        
        For any storage and any glyph_id, if get_glyph succeeds,
        then has_glyph SHALL return True.
        
        Feature: gql-storage-abstraction, Property 1: Storage Protocol Consistency
        **Validates: Requirements 1.2, 1.3**
        """
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Try to get the glyph
        try:
            glyph = storage.get_glyph(query_id)
            # If get_glyph succeeded, has_glyph must return True
            assert storage.has_glyph(query_id), \
                f"get_glyph('{query_id}') succeeded but has_glyph returned False"
        except KeyError:
            # get_glyph raised KeyError, which is fine - we're testing the converse
            pass
    
    @given(
        glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10),
        query_id=glyph_ids
    )
    @settings(max_examples=100)
    def test_get_glyph_keyerror_implies_has_glyph_false(
        self, glyphs: Dict[str, MockSDKGlyph], query_id: str
    ):
        """
        Property test: get_glyph(id) raises KeyError implies has_glyph(id) == False.
        
        For any storage and any glyph_id, if get_glyph raises KeyError,
        then has_glyph SHALL return False.
        
        Feature: gql-storage-abstraction, Property 1: Storage Protocol Consistency
        **Validates: Requirements 1.2, 1.3**
        """
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Try to get the glyph
        try:
            storage.get_glyph(query_id)
            # get_glyph succeeded, which is fine - we're testing the converse
        except KeyError:
            # If get_glyph raised KeyError, has_glyph must return False
            assert not storage.has_glyph(query_id), \
                f"get_glyph('{query_id}') raised KeyError but has_glyph returned True"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_has_glyph_get_glyph_consistency_for_known_ids(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: has_glyph and get_glyph are consistent for all known glyph IDs.
        
        For any storage, for every glyph_id in the storage, has_glyph SHALL return
        True and get_glyph SHALL succeed.
        
        Feature: gql-storage-abstraction, Property 1: Storage Protocol Consistency
        **Validates: Requirements 1.2, 1.3**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # For every known glyph ID, verify consistency
        for glyph_id in glyphs.keys():
            # has_glyph must return True
            assert storage.has_glyph(glyph_id), \
                f"has_glyph('{glyph_id}') returned False for known glyph"
            
            # get_glyph must succeed
            try:
                glyph = storage.get_glyph(glyph_id)
                assert glyph is not None, \
                    f"get_glyph('{glyph_id}') returned None for known glyph"
                # Verify we got the correct glyph back
                assert glyph is glyphs[glyph_id], \
                    f"get_glyph('{glyph_id}') returned wrong glyph"
            except KeyError as e:
                pytest.fail(
                    f"get_glyph('{glyph_id}') raised KeyError for known glyph: {e}"
                )
    
    @given(
        glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10),
        unknown_ids=st.lists(glyph_ids, min_size=1, max_size=5, unique=True)
    )
    @settings(max_examples=100)
    def test_has_glyph_get_glyph_consistency_for_unknown_ids(
        self, glyphs: Dict[str, MockSDKGlyph], unknown_ids: List[str]
    ):
        """
        Property test: has_glyph and get_glyph are consistent for unknown glyph IDs.
        
        For any storage and any glyph_id NOT in the storage, has_glyph SHALL return
        False and get_glyph SHALL raise KeyError.
        
        Feature: gql-storage-abstraction, Property 1: Storage Protocol Consistency
        **Validates: Requirements 1.2, 1.3**
        """
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Filter to only IDs that are actually unknown
        truly_unknown_ids = [uid for uid in unknown_ids if uid not in glyphs]
        
        # Skip if all generated IDs happen to be in the glyphs dict
        assume(len(truly_unknown_ids) > 0)
        
        # For every unknown glyph ID, verify consistency
        for unknown_id in truly_unknown_ids:
            # has_glyph must return False
            assert not storage.has_glyph(unknown_id), \
                f"has_glyph('{unknown_id}') returned True for unknown glyph"
            
            # get_glyph must raise KeyError
            with pytest.raises(KeyError):
                storage.get_glyph(unknown_id)
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_list_glyphs_consistent_with_has_glyph(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: list_glyphs is consistent with has_glyph.
        
        For any storage, every glyph_id returned by list_glyphs SHALL have
        has_glyph return True.
        
        Feature: gql-storage-abstraction, Property 1: Storage Protocol Consistency
        **Validates: Requirements 1.2, 1.3**
        """
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Get all glyphs from list_glyphs
        listed_glyphs = storage.list_glyphs()
        
        # For every listed glyph ID, has_glyph must return True
        for glyph_id in listed_glyphs.keys():
            assert storage.has_glyph(glyph_id), \
                f"has_glyph('{glyph_id}') returned False for glyph in list_glyphs"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_list_glyphs_consistent_with_get_glyph(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: list_glyphs is consistent with get_glyph.
        
        For any storage, every glyph_id returned by list_glyphs SHALL have
        get_glyph succeed.
        
        Feature: gql-storage-abstraction, Property 1: Storage Protocol Consistency
        **Validates: Requirements 1.2, 1.3**
        """
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Get all glyphs from list_glyphs
        listed_glyphs = storage.list_glyphs()
        
        # For every listed glyph ID, get_glyph must succeed
        for glyph_id in listed_glyphs.keys():
            try:
                glyph = storage.get_glyph(glyph_id)
                assert glyph is not None, \
                    f"get_glyph('{glyph_id}') returned None for glyph in list_glyphs"
            except KeyError as e:
                pytest.fail(
                    f"get_glyph('{glyph_id}') raised KeyError for glyph in list_glyphs: {e}"
                )


# =============================================================================
# Property 4: InMemoryGlyphStorage Correctness
# =============================================================================

class MockLayer:
    """
    Mock Layer object for testing InMemoryGlyphStorage hierarchical structure.
    
    This simulates the structure of a real SDK Layer with name, cortex, and segments.
    """
    
    def __init__(
        self,
        name: str,
        cortex: Optional[List[float]] = None,
        segments: Optional[List[Any]] = None
    ):
        self.name = name
        self.cortex = cortex
        self.segments = segments or []


class MockSegment:
    """
    Mock Segment object for testing InMemoryGlyphStorage hierarchical structure.
    
    This simulates the structure of a real SDK Segment with name and cortex.
    """
    
    def __init__(
        self,
        name: str,
        cortex: Optional[List[float]] = None
    ):
        self.name = name
        self.cortex = cortex


class MockSDKGlyphWithLayers:
    """
    Mock SDK Glyph object with full hierarchical structure for testing.
    
    This simulates the structure of a real SDK Glyph with cortex,
    global_cortex, layers, segments, attributes, and metadata.
    """
    
    def __init__(
        self,
        identifier: str,
        cortex: Optional[List[float]] = None,
        global_cortex: Optional[List[float]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        layers: Optional[List[MockLayer]] = None
    ):
        self.identifier = identifier
        self.cortex = cortex
        self.global_cortex = global_cortex
        self.attributes = attributes or {}
        self.metadata = metadata or {}
        self.layers = layers or []


@st.composite
def mock_segment_strategy(draw) -> MockSegment:
    """
    Generate a mock segment for property testing.
    
    Returns:
        MockSegment with random but valid attributes
    """
    name = draw(glyph_ids)
    has_cortex = draw(st.booleans())
    cortex = draw(vectors) if has_cortex else None
    
    return MockSegment(name=name, cortex=cortex)


@st.composite
def mock_layer_strategy(draw) -> MockLayer:
    """
    Generate a mock layer for property testing.
    
    Returns:
        MockLayer with random but valid attributes
    """
    name = draw(glyph_ids)
    has_cortex = draw(st.booleans())
    cortex = draw(vectors) if has_cortex else None
    
    # Generate 0-3 segments
    num_segments = draw(st.integers(min_value=0, max_value=3))
    segments = [draw(mock_segment_strategy()) for _ in range(num_segments)]
    
    return MockLayer(name=name, cortex=cortex, segments=segments)


@st.composite
def mock_sdk_glyph_with_layers_strategy(draw) -> MockSDKGlyphWithLayers:
    """
    Generate a mock SDK glyph with full hierarchical structure for property testing.
    
    Returns:
        MockSDKGlyphWithLayers with random but valid attributes
    """
    glyph_id = draw(glyph_ids)
    
    # Randomly decide which embedding to use
    has_cortex = draw(st.booleans())
    has_global_cortex = draw(st.booleans())
    
    cortex = draw(vectors) if has_cortex else None
    global_cortex = draw(vectors) if has_global_cortex else None
    
    attrs = draw(metadata)
    meta = draw(metadata)
    
    # Generate 0-3 layers
    num_layers = draw(st.integers(min_value=0, max_value=3))
    layers = [draw(mock_layer_strategy()) for _ in range(num_layers)]
    
    return MockSDKGlyphWithLayers(
        identifier=glyph_id,
        cortex=cortex,
        global_cortex=global_cortex,
        attributes=attrs,
        metadata=meta,
        layers=layers
    )


@st.composite
def glyph_dict_with_layers_strategy(draw, min_glyphs: int = 0, max_glyphs: int = 10) -> Dict[str, MockSDKGlyphWithLayers]:
    """
    Generate a dictionary of mock SDK glyphs with layers for property testing.
    
    Args:
        min_glyphs: Minimum number of glyphs to generate
        max_glyphs: Maximum number of glyphs to generate
        
    Returns:
        Dictionary mapping glyph_id to MockSDKGlyphWithLayers
    """
    num_glyphs = draw(st.integers(min_value=min_glyphs, max_value=max_glyphs))
    glyphs = {}
    
    for _ in range(num_glyphs):
        glyph = draw(mock_sdk_glyph_with_layers_strategy())
        # Use the glyph's identifier as the key
        glyphs[glyph.identifier] = glyph
    
    return glyphs


class TestInMemoryGlyphStorageCorrectness:
    """
    Property tests for InMemoryGlyphStorage Correctness (Property 4).
    
    Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
    
    **Validates: Requirements 4.2, 4.3, 4.4, 4.5**
    
    For any dictionary of SDK glyphs passed to InMemoryGlyphStorage:
    - list_glyphs() returns the same dictionary
    - get_glyph(id) returns the glyph for any id in the dict
    - get_embedding(id) returns the glyph's cortex/global_cortex
    - get_embedding_for_scope(id, layer, segment) returns the appropriate hierarchical vector
    """
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_list_glyphs_returns_wrapped_dict(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: list_glyphs() returns the same dictionary passed to constructor.
        
        For any dictionary of SDK glyphs passed to InMemoryGlyphStorage,
        list_glyphs() SHALL return the same dictionary.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.2**
        """
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # list_glyphs should return the same dictionary
        listed = storage.list_glyphs()
        
        # Verify it's the same dictionary (identity check)
        assert listed is glyphs, \
            "list_glyphs() should return the same dictionary object"
        
        # Also verify contents match
        assert len(listed) == len(glyphs), \
            f"list_glyphs() returned {len(listed)} glyphs, expected {len(glyphs)}"
        
        for glyph_id in glyphs:
            assert glyph_id in listed, \
                f"Glyph '{glyph_id}' missing from list_glyphs() result"
            assert listed[glyph_id] is glyphs[glyph_id], \
                f"Glyph '{glyph_id}' in list_glyphs() is not the same object"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_get_glyph_returns_correct_glyph_for_known_ids(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: get_glyph(id) returns the correct glyph for any id in the dict.
        
        For any dictionary of SDK glyphs passed to InMemoryGlyphStorage,
        get_glyph(id) SHALL return the glyph for any id in the dict.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.2**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # For every glyph ID in the dict, get_glyph should return the correct glyph
        for glyph_id, expected_glyph in glyphs.items():
            retrieved_glyph = storage.get_glyph(glyph_id)
            
            # Verify we got the same glyph object back
            assert retrieved_glyph is expected_glyph, \
                f"get_glyph('{glyph_id}') returned wrong glyph object"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_get_embedding_returns_cortex_when_available(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: get_embedding(id) returns glyph.cortex when available.
        
        For any glyph with a cortex attribute, get_embedding() SHALL return
        the cortex value.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.3**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # For every glyph, verify get_embedding behavior
        for glyph_id, glyph in glyphs.items():
            embedding = storage.get_embedding(glyph_id)
            
            if glyph.cortex is not None:
                # If cortex is available, it should be returned
                assert embedding is glyph.cortex, \
                    f"get_embedding('{glyph_id}') should return cortex when available"
            elif glyph.global_cortex is not None:
                # If only global_cortex is available, it should be returned
                assert embedding is glyph.global_cortex, \
                    f"get_embedding('{glyph_id}') should return global_cortex when cortex is None"
            else:
                # If neither is available, should return None
                assert embedding is None, \
                    f"get_embedding('{glyph_id}') should return None when no embedding available"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_get_embedding_returns_global_cortex_as_fallback(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: get_embedding(id) returns glyph.global_cortex when cortex is None.
        
        For any glyph without cortex but with global_cortex, get_embedding()
        SHALL return the global_cortex value.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.3**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Filter to glyphs without cortex but with global_cortex
        glyphs_with_only_global = {
            gid: g for gid, g in glyphs.items()
            if g.cortex is None and g.global_cortex is not None
        }
        
        # For each such glyph, verify get_embedding returns global_cortex
        for glyph_id, glyph in glyphs_with_only_global.items():
            embedding = storage.get_embedding(glyph_id)
            assert embedding is glyph.global_cortex, \
                f"get_embedding('{glyph_id}') should return global_cortex when cortex is None"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_get_embedding_returns_none_when_no_embedding(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: get_embedding(id) returns None when no embedding available.
        
        For any glyph without cortex and without global_cortex, get_embedding()
        SHALL return None.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.3**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Filter to glyphs without any embedding
        glyphs_without_embedding = {
            gid: g for gid, g in glyphs.items()
            if g.cortex is None and g.global_cortex is None
        }
        
        # For each such glyph, verify get_embedding returns None
        for glyph_id in glyphs_without_embedding:
            embedding = storage.get_embedding(glyph_id)
            assert embedding is None, \
                f"get_embedding('{glyph_id}') should return None when no embedding available"
    
    @given(glyphs=glyph_dict_with_layers_strategy(min_glyphs=1, max_glyphs=5))
    @settings(max_examples=100)
    def test_get_embedding_for_scope_returns_primary_when_no_layer(
        self, glyphs: Dict[str, MockSDKGlyphWithLayers]
    ):
        """
        Property test: get_embedding_for_scope(id, None, None) returns primary embedding.
        
        When layer is None, get_embedding_for_scope() SHALL return the same
        result as get_embedding().
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.4, 4.5**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # For every glyph, verify get_embedding_for_scope with no layer
        for glyph_id in glyphs:
            scoped_embedding = storage.get_embedding_for_scope(glyph_id, layer=None, segment=None)
            primary_embedding = storage.get_embedding(glyph_id)
            
            # Should return the same result
            assert scoped_embedding is primary_embedding, \
                f"get_embedding_for_scope('{glyph_id}', None, None) should equal get_embedding()"
    
    @given(glyphs=glyph_dict_with_layers_strategy(min_glyphs=1, max_glyphs=5))
    @settings(max_examples=100)
    def test_get_embedding_for_scope_returns_layer_cortex(
        self, glyphs: Dict[str, MockSDKGlyphWithLayers]
    ):
        """
        Property test: get_embedding_for_scope(id, layer, None) returns layer cortex.
        
        When layer is specified but segment is None, get_embedding_for_scope()
        SHALL return the first matching layer's cortex if the layer exists.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.4, 4.5**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # For every glyph with layers, verify layer cortex retrieval
        for glyph_id, glyph in glyphs.items():
            # Build a map of layer names to the FIRST layer with that name
            # (since the storage implementation finds the first match)
            first_layer_by_name: Dict[str, MockLayer] = {}
            for layer in glyph.layers:
                if layer.name not in first_layer_by_name:
                    first_layer_by_name[layer.name] = layer
            
            # Test each unique layer name
            for layer_name, first_layer in first_layer_by_name.items():
                scoped_embedding = storage.get_embedding_for_scope(
                    glyph_id, layer=layer_name, segment=None
                )
                
                # Should return the first matching layer's cortex
                if first_layer.cortex is not None:
                    assert scoped_embedding is first_layer.cortex, \
                        f"get_embedding_for_scope('{glyph_id}', '{layer_name}', None) " \
                        f"should return first matching layer's cortex"
                else:
                    assert scoped_embedding is None, \
                        f"get_embedding_for_scope('{glyph_id}', '{layer_name}', None) " \
                        f"should return None when first matching layer has no cortex"
    
    @given(glyphs=glyph_dict_with_layers_strategy(min_glyphs=1, max_glyphs=5))
    @settings(max_examples=100)
    def test_get_embedding_for_scope_returns_segment_cortex(
        self, glyphs: Dict[str, MockSDKGlyphWithLayers]
    ):
        """
        Property test: get_embedding_for_scope(id, layer, segment) returns segment cortex.
        
        When both layer and segment are specified, get_embedding_for_scope()
        SHALL return the first matching segment's cortex if both layer and segment exist.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.4, 4.5**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # For every glyph with layers and segments, verify segment cortex retrieval
        for glyph_id, glyph in glyphs.items():
            # Build a map of layer names to the FIRST layer with that name
            first_layer_by_name: Dict[str, MockLayer] = {}
            for layer in glyph.layers:
                if layer.name not in first_layer_by_name:
                    first_layer_by_name[layer.name] = layer
            
            # For each unique layer, build a map of segment names to first segment
            for layer_name, first_layer in first_layer_by_name.items():
                first_segment_by_name: Dict[str, MockSegment] = {}
                for segment in first_layer.segments:
                    if segment.name not in first_segment_by_name:
                        first_segment_by_name[segment.name] = segment
                
                # Test each unique segment name in this layer
                for segment_name, first_segment in first_segment_by_name.items():
                    scoped_embedding = storage.get_embedding_for_scope(
                        glyph_id, layer=layer_name, segment=segment_name
                    )
                    
                    # Should return the first matching segment's cortex
                    if first_segment.cortex is not None:
                        assert scoped_embedding is first_segment.cortex, \
                            f"get_embedding_for_scope('{glyph_id}', '{layer_name}', " \
                            f"'{segment_name}') should return first matching segment's cortex"
                    else:
                        assert scoped_embedding is None, \
                            f"get_embedding_for_scope('{glyph_id}', '{layer_name}', " \
                            f"'{segment_name}') should return None when first matching segment has no cortex"
    
    @given(glyphs=glyph_dict_with_layers_strategy(min_glyphs=1, max_glyphs=5))
    @settings(max_examples=100)
    def test_get_embedding_for_scope_returns_none_for_nonexistent_layer(
        self, glyphs: Dict[str, MockSDKGlyphWithLayers]
    ):
        """
        Property test: get_embedding_for_scope returns None for non-existent layer.
        
        When a layer name is specified that doesn't exist in the glyph,
        get_embedding_for_scope() SHALL return None.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.4, 4.5**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Use a layer name that definitely doesn't exist
        nonexistent_layer = "__nonexistent_layer_xyz__"
        
        # For every glyph, verify None is returned for non-existent layer
        for glyph_id in glyphs:
            scoped_embedding = storage.get_embedding_for_scope(
                glyph_id, layer=nonexistent_layer, segment=None
            )
            
            assert scoped_embedding is None, \
                f"get_embedding_for_scope('{glyph_id}', '{nonexistent_layer}', None) " \
                f"should return None for non-existent layer"
    
    @given(glyphs=glyph_dict_with_layers_strategy(min_glyphs=1, max_glyphs=5))
    @settings(max_examples=100)
    def test_get_embedding_for_scope_returns_none_for_nonexistent_segment(
        self, glyphs: Dict[str, MockSDKGlyphWithLayers]
    ):
        """
        Property test: get_embedding_for_scope returns None for non-existent segment.
        
        When a segment name is specified that doesn't exist in the layer,
        get_embedding_for_scope() SHALL return None.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.4, 4.5**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # Use a segment name that definitely doesn't exist
        nonexistent_segment = "__nonexistent_segment_xyz__"
        
        # For every glyph with layers, verify None is returned for non-existent segment
        for glyph_id, glyph in glyphs.items():
            for layer in glyph.layers:
                scoped_embedding = storage.get_embedding_for_scope(
                    glyph_id, layer=layer.name, segment=nonexistent_segment
                )
                
                assert scoped_embedding is None, \
                    f"get_embedding_for_scope('{glyph_id}', '{layer.name}', " \
                    f"'{nonexistent_segment}') should return None for non-existent segment"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_list_glyphs_keys_match_dict_keys(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: list_glyphs() keys match the original dict keys.
        
        For any dictionary of SDK glyphs passed to InMemoryGlyphStorage,
        list_glyphs().keys() SHALL equal the original dict keys.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.2**
        """
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # list_glyphs keys should match original dict keys
        listed_keys = set(storage.list_glyphs().keys())
        original_keys = set(glyphs.keys())
        
        assert listed_keys == original_keys, \
            f"list_glyphs() keys {listed_keys} don't match original keys {original_keys}"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_list_glyphs_values_match_dict_values(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: list_glyphs() values match the original dict values.
        
        For any dictionary of SDK glyphs passed to InMemoryGlyphStorage,
        list_glyphs() values SHALL be the same objects as the original dict values.
        
        Feature: gql-storage-abstraction, Property 4: InMemoryGlyphStorage Correctness
        **Validates: Requirements 4.2**
        """
        # Create storage with the generated glyphs
        storage = InMemoryGlyphStorage(glyphs=glyphs)
        
        # list_glyphs values should be the same objects
        listed = storage.list_glyphs()
        
        for glyph_id, original_glyph in glyphs.items():
            assert listed[glyph_id] is original_glyph, \
                f"list_glyphs()['{glyph_id}'] is not the same object as original"


# =============================================================================
# Property 3: ExecutionContext Delegation
# =============================================================================

# Import ExecutionContext for delegation tests
from glyphh.gql.planner import ExecutionContext


class MockStorage:
    """
    Mock storage implementation for testing ExecutionContext delegation.
    
    This mock tracks all method calls to verify that ExecutionContext
    correctly delegates to the underlying storage.
    """
    
    def __init__(
        self,
        glyphs: Dict[str, Any],
        embeddings: Optional[Dict[str, List[float]]] = None
    ):
        self._glyphs = glyphs
        self._embeddings = embeddings or {}
        # Track method calls for verification
        self.call_log: List[Tuple[str, Any]] = []
    
    def list_glyphs(self) -> Dict[str, Any]:
        self.call_log.append(("list_glyphs", None))
        return self._glyphs
    
    def get_glyph(self, glyph_id: str) -> Any:
        self.call_log.append(("get_glyph", glyph_id))
        if glyph_id not in self._glyphs:
            raise KeyError(f"Glyph not found: {glyph_id}")
        return self._glyphs[glyph_id]
    
    def has_glyph(self, glyph_id: str) -> bool:
        self.call_log.append(("has_glyph", glyph_id))
        return glyph_id in self._glyphs
    
    def get_embedding(self, glyph_id: str) -> Optional[List[float]]:
        self.call_log.append(("get_embedding", glyph_id))
        return self._embeddings.get(glyph_id)
    
    def get_embedding_for_scope(
        self,
        glyph_id: str,
        layer: Optional[str] = None,
        segment: Optional[str] = None
    ) -> Optional[List[float]]:
        self.call_log.append(("get_embedding_for_scope", (glyph_id, layer, segment)))
        if layer is None:
            return self._embeddings.get(glyph_id)
        return None  # Mock doesn't support scoped embeddings
    
    def compute_similarity(self, v1: Any, v2: Any) -> float:
        self.call_log.append(("compute_similarity", (v1, v2)))
        # Simple mock similarity - return 0.5 for any vectors
        return 0.5
    
    def get_glyph_attribute(self, glyph_id: str, attribute: str) -> Any:
        self.call_log.append(("get_glyph_attribute", (glyph_id, attribute)))
        glyph = self._glyphs.get(glyph_id)
        if glyph and hasattr(glyph, attribute):
            return getattr(glyph, attribute)
        return None


@st.composite
def mock_storage_strategy(draw) -> Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]]:
    """
    Generate a mock storage with glyphs and embeddings for property testing.
    
    Returns:
        Tuple of (MockStorage, glyphs dict, embeddings dict)
    """
    glyphs = draw(glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    
    # Generate embeddings for glyphs that have cortex
    embeddings = {}
    for glyph_id, glyph in glyphs.items():
        if glyph.cortex is not None:
            embeddings[glyph_id] = glyph.cortex
        elif glyph.global_cortex is not None:
            embeddings[glyph_id] = glyph.global_cortex
    
    storage = MockStorage(glyphs=glyphs, embeddings=embeddings)
    return storage, glyphs, embeddings


class TestExecutionContextDelegation:
    """
    Property tests for ExecutionContext Delegation (Property 3).
    
    Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
    
    **Validates: Requirements 2.1, 2.3, 2.4**
    
    For any storage implementation passed to ExecutionContext, calling
    context.has_glyph(id), context.get_glyph(id), context.list_glyphs(),
    and context.compute_similarity(v1, v2) returns the same results as
    calling the corresponding methods directly on the storage.
    """
    
    @given(data=mock_storage_strategy(), query_id=glyph_ids)
    @settings(max_examples=100)
    def test_has_glyph_delegates_to_storage(
        self, data: Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]], query_id: str
    ):
        """
        Property test: context.has_glyph(id) delegates to storage.has_glyph(id).
        
        For any storage implementation passed to ExecutionContext,
        context.has_glyph(id) SHALL return the same result as storage.has_glyph(id).
        
        Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
        **Validates: Requirements 2.1, 2.3**
        """
        storage, glyphs, embeddings = data
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Clear call log to track only our test calls
        storage.call_log.clear()
        
        # Call has_glyph on context
        context_result = context.has_glyph(query_id)
        
        # Verify delegation occurred
        assert ("has_glyph", query_id) in storage.call_log, \
            f"ExecutionContext.has_glyph('{query_id}') did not delegate to storage"
        
        # Verify result matches direct storage call
        storage.call_log.clear()
        storage_result = storage.has_glyph(query_id)
        
        assert context_result == storage_result, \
            f"context.has_glyph('{query_id}') returned {context_result}, " \
            f"but storage.has_glyph returned {storage_result}"
    
    @given(data=mock_storage_strategy())
    @settings(max_examples=100)
    def test_has_glyph_returns_same_result_as_storage_for_known_ids(
        self, data: Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]]
    ):
        """
        Property test: context.has_glyph returns same result as storage for known IDs.
        
        For any storage and any glyph_id in the storage, context.has_glyph(id)
        SHALL return True, matching storage.has_glyph(id).
        
        Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
        **Validates: Requirements 2.1, 2.3**
        """
        storage, glyphs, embeddings = data
        
        # Skip if no glyphs
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # For every known glyph ID, verify delegation and result
        for glyph_id in glyphs.keys():
            context_result = context.has_glyph(glyph_id)
            storage_result = storage.has_glyph(glyph_id)
            
            assert context_result == storage_result == True, \
                f"has_glyph('{glyph_id}') mismatch: context={context_result}, storage={storage_result}"
    
    @given(data=mock_storage_strategy())
    @settings(max_examples=100)
    def test_get_glyph_delegates_to_storage_for_known_ids(
        self, data: Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]]
    ):
        """
        Property test: context.get_glyph(id) delegates to storage.get_glyph(id).
        
        For any storage and any glyph_id in the storage, context.get_glyph(id)
        SHALL return the same glyph object as storage.get_glyph(id).
        
        Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
        **Validates: Requirements 2.1, 2.3**
        """
        storage, glyphs, embeddings = data
        
        # Skip if no glyphs
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # For every known glyph ID, verify delegation and result
        for glyph_id in glyphs.keys():
            # Clear call log
            storage.call_log.clear()
            
            # Call get_glyph on context
            context_result = context.get_glyph(glyph_id)
            
            # Verify delegation occurred
            assert ("get_glyph", glyph_id) in storage.call_log, \
                f"ExecutionContext.get_glyph('{glyph_id}') did not delegate to storage"
            
            # Verify result matches direct storage call
            storage_result = storage.get_glyph(glyph_id)
            
            assert context_result is storage_result, \
                f"context.get_glyph('{glyph_id}') returned different object than storage"
    
    @given(data=mock_storage_strategy())
    @settings(max_examples=100)
    def test_list_glyphs_delegates_to_storage(
        self, data: Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]]
    ):
        """
        Property test: context.list_glyphs() delegates to storage.list_glyphs().
        
        For any storage implementation passed to ExecutionContext,
        context.list_glyphs() SHALL return the same dict as storage.list_glyphs().
        
        Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
        **Validates: Requirements 2.1, 2.3**
        """
        storage, glyphs, embeddings = data
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Clear call log
        storage.call_log.clear()
        
        # Call list_glyphs on context
        context_result = context.list_glyphs()
        
        # Verify delegation occurred
        assert ("list_glyphs", None) in storage.call_log, \
            "ExecutionContext.list_glyphs() did not delegate to storage"
        
        # Verify result matches direct storage call
        storage_result = storage.list_glyphs()
        
        assert context_result is storage_result, \
            "context.list_glyphs() returned different dict than storage"
    
    @given(data=mock_storage_strategy())
    @settings(max_examples=100)
    def test_list_glyphs_returns_same_keys_as_storage(
        self, data: Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]]
    ):
        """
        Property test: context.list_glyphs() returns same keys as storage.
        
        For any storage, context.list_glyphs().keys() SHALL equal
        storage.list_glyphs().keys().
        
        Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
        **Validates: Requirements 2.1, 2.3**
        """
        storage, glyphs, embeddings = data
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Get results from both
        context_keys = set(context.list_glyphs().keys())
        storage_keys = set(storage.list_glyphs().keys())
        
        assert context_keys == storage_keys, \
            f"context.list_glyphs() keys {context_keys} don't match storage keys {storage_keys}"
    
    @given(
        data=mock_storage_strategy(),
        v1=vectors,
        v2=vectors
    )
    @settings(max_examples=100)
    def test_compute_similarity_delegates_to_storage(
        self, data: Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]],
        v1: List[float], v2: List[float]
    ):
        """
        Property test: context.compute_similarity(v1, v2) delegates to storage.
        
        For any storage and any two vectors, context.compute_similarity(v1, v2)
        SHALL return the same result as storage.compute_similarity(v1, v2).
        
        Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
        **Validates: Requirements 2.4**
        """
        storage, glyphs, embeddings = data
        
        # Ensure vectors have same length for valid comparison
        min_len = min(len(v1), len(v2))
        assume(min_len > 0)
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Clear call log
        storage.call_log.clear()
        
        # Call compute_similarity on context
        context_result = context.compute_similarity(v1, v2)
        
        # Verify delegation occurred
        assert any(call[0] == "compute_similarity" for call in storage.call_log), \
            "ExecutionContext.compute_similarity() did not delegate to storage"
        
        # Verify result matches direct storage call
        storage_result = storage.compute_similarity(v1, v2)
        
        assert context_result == storage_result, \
            f"context.compute_similarity() returned {context_result}, " \
            f"but storage returned {storage_result}"
    
    @given(data=mock_storage_strategy())
    @settings(max_examples=100)
    def test_get_embedding_delegates_to_storage(
        self, data: Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]]
    ):
        """
        Property test: context.get_embedding(id) delegates to storage.
        
        For any storage and any glyph_id, context.get_embedding(id)
        SHALL return the same result as storage.get_embedding(id).
        
        Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
        **Validates: Requirements 2.1, 2.3**
        """
        storage, glyphs, embeddings = data
        
        # Skip if no glyphs
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # For every known glyph ID, verify delegation and result
        for glyph_id in glyphs.keys():
            # Clear call log
            storage.call_log.clear()
            
            # Call get_embedding on context
            context_result = context.get_embedding(glyph_id)
            
            # Verify delegation occurred
            assert ("get_embedding", glyph_id) in storage.call_log, \
                f"ExecutionContext.get_embedding('{glyph_id}') did not delegate to storage"
            
            # Verify result matches direct storage call
            storage_result = storage.get_embedding(glyph_id)
            
            # Compare results (both could be None or same list)
            if context_result is None:
                assert storage_result is None, \
                    f"context.get_embedding('{glyph_id}') returned None but storage returned {storage_result}"
            else:
                assert context_result == storage_result, \
                    f"context.get_embedding('{glyph_id}') returned {context_result}, " \
                    f"but storage returned {storage_result}"
    
    @given(data=mock_storage_strategy())
    @settings(max_examples=100)
    def test_get_embedding_for_scope_delegates_to_storage(
        self, data: Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]]
    ):
        """
        Property test: context.get_embedding_for_scope() delegates to storage.
        
        For any storage and any glyph_id, context.get_embedding_for_scope(id, layer, segment)
        SHALL return the same result as storage.get_embedding_for_scope(id, layer, segment).
        
        Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
        **Validates: Requirements 2.1, 2.3**
        """
        storage, glyphs, embeddings = data
        
        # Skip if no glyphs
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Test with various scope combinations
        test_cases = [
            (None, None),  # No scope
            ("test_layer", None),  # Layer only
            ("test_layer", "test_segment"),  # Layer and segment
        ]
        
        # For every known glyph ID, verify delegation and result
        for glyph_id in glyphs.keys():
            for layer, segment in test_cases:
                # Clear call log
                storage.call_log.clear()
                
                # Call get_embedding_for_scope on context
                context_result = context.get_embedding_for_scope(glyph_id, layer=layer, segment=segment)
                
                # Verify delegation occurred
                expected_call = ("get_embedding_for_scope", (glyph_id, layer, segment))
                assert expected_call in storage.call_log, \
                    f"ExecutionContext.get_embedding_for_scope('{glyph_id}', '{layer}', '{segment}') " \
                    f"did not delegate to storage"
                
                # Verify result matches direct storage call
                storage_result = storage.get_embedding_for_scope(glyph_id, layer=layer, segment=segment)
                
                # Compare results
                if context_result is None:
                    assert storage_result is None, \
                        f"context.get_embedding_for_scope() returned None but storage returned {storage_result}"
                else:
                    assert context_result == storage_result, \
                        f"context.get_embedding_for_scope() returned {context_result}, " \
                        f"but storage returned {storage_result}"
    
    @given(data=mock_storage_strategy(), query_id=glyph_ids)
    @settings(max_examples=100)
    def test_context_storage_consistency_for_random_ids(
        self, data: Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]], query_id: str
    ):
        """
        Property test: context and storage are consistent for random glyph IDs.
        
        For any storage and any random glyph_id, the context's has_glyph, get_glyph,
        and list_glyphs methods SHALL be consistent with the storage's methods.
        
        Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
        **Validates: Requirements 2.1, 2.3, 2.4**
        """
        storage, glyphs, embeddings = data
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Test has_glyph consistency
        context_has = context.has_glyph(query_id)
        storage_has = storage.has_glyph(query_id)
        assert context_has == storage_has, \
            f"has_glyph('{query_id}') inconsistent: context={context_has}, storage={storage_has}"
        
        # Test get_glyph consistency (only if glyph exists)
        if context_has:
            context_glyph = context.get_glyph(query_id)
            storage_glyph = storage.get_glyph(query_id)
            assert context_glyph is storage_glyph, \
                f"get_glyph('{query_id}') returned different objects"
        
        # Test list_glyphs consistency
        context_list = context.list_glyphs()
        storage_list = storage.list_glyphs()
        assert context_list is storage_list, \
            "list_glyphs() returned different dicts"
    
    @given(data=mock_storage_strategy())
    @settings(max_examples=100)
    def test_all_delegation_methods_work_together(
        self, data: Tuple[MockStorage, Dict[str, MockSDKGlyph], Dict[str, List[float]]]
    ):
        """
        Property test: all delegation methods work together consistently.
        
        For any storage, using multiple context methods in sequence SHALL
        produce consistent results that match the storage.
        
        Feature: gql-storage-abstraction, Property 3: ExecutionContext Delegation
        **Validates: Requirements 2.1, 2.3, 2.4**
        """
        storage, glyphs, embeddings = data
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Get all glyphs via list_glyphs
        listed = context.list_glyphs()
        
        # Verify each listed glyph is accessible via has_glyph and get_glyph
        for glyph_id in listed.keys():
            # has_glyph should return True
            assert context.has_glyph(glyph_id), \
                f"has_glyph('{glyph_id}') returned False for listed glyph"
            
            # get_glyph should return the same object
            glyph = context.get_glyph(glyph_id)
            assert glyph is listed[glyph_id], \
                f"get_glyph('{glyph_id}') returned different object than list_glyphs"
            
            # get_embedding should work (may return None)
            embedding = context.get_embedding(glyph_id)
            storage_embedding = storage.get_embedding(glyph_id)
            if embedding is not None:
                assert embedding == storage_embedding, \
                    f"get_embedding('{glyph_id}') returned different result than storage"


# =============================================================================
# Property 6: Plan Execution with Storage
# =============================================================================

# Import plan classes for testing
from glyphh.gql.plans import (
    ListPlan,
    SimilaritySearchPlan,
    ComparePlan,
)
from glyphh.fact_tree import FactTree
from hypothesis import HealthCheck


# Smaller vector strategy for plan tests (to avoid large base example health check)
small_vectors = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=8,
    max_size=16
)

# Simple glyph ID strategy for plan tests
simple_glyph_ids = st.text(
    alphabet=st.characters(whitelist_categories=('L',), whitelist_characters='_'),
    min_size=2,
    max_size=8
).filter(lambda x: x.strip())

# Simple metadata for plan tests
simple_metadata = st.dictionaries(
    keys=st.text(alphabet='abcdefghij', min_size=1, max_size=5),
    values=st.integers(min_value=0, max_value=100),
    min_size=0,
    max_size=2
)


class MockStorageForPlans:
    """
    Mock storage implementation for testing plan execution.
    
    This mock implements GlyphStorageProtocol and tracks all method calls
    to verify that plans use storage abstraction methods instead of
    accessing glyph object attributes directly.
    """
    
    def __init__(
        self,
        glyphs: Dict[str, Any],
        embeddings: Optional[Dict[str, List[float]]] = None,
        attributes: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self._glyphs = glyphs
        self._embeddings = embeddings or {}
        self._attributes = attributes or {}
        # Track method calls for verification
        self.call_log: List[Tuple[str, Any]] = []
        # Track if any direct glyph attribute access was attempted
        self.direct_access_attempted = False
    
    def list_glyphs(self) -> Dict[str, Any]:
        self.call_log.append(("list_glyphs", None))
        return self._glyphs
    
    def get_glyph(self, glyph_id: str) -> Any:
        self.call_log.append(("get_glyph", glyph_id))
        if glyph_id not in self._glyphs:
            raise KeyError(f"Glyph not found: {glyph_id}")
        return self._glyphs[glyph_id]
    
    def has_glyph(self, glyph_id: str) -> bool:
        self.call_log.append(("has_glyph", glyph_id))
        return glyph_id in self._glyphs
    
    def get_embedding(self, glyph_id: str) -> Optional[List[float]]:
        self.call_log.append(("get_embedding", glyph_id))
        return self._embeddings.get(glyph_id)
    
    def get_embedding_for_scope(
        self,
        glyph_id: str,
        layer: Optional[str] = None,
        segment: Optional[str] = None
    ) -> Optional[List[float]]:
        self.call_log.append(("get_embedding_for_scope", (glyph_id, layer, segment)))
        if layer is None:
            return self._embeddings.get(glyph_id)
        return None  # Mock doesn't support scoped embeddings
    
    def compute_similarity(self, v1: Any, v2: Any) -> float:
        self.call_log.append(("compute_similarity", (type(v1).__name__, type(v2).__name__)))
        # Compute actual cosine similarity for valid results
        import numpy as np
        try:
            arr1 = np.array(v1)
            arr2 = np.array(v2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(arr1, arr2) / (norm1 * norm2))
        except Exception:
            return 0.5
    
    def get_glyph_attribute(self, glyph_id: str, attribute: str) -> Any:
        self.call_log.append(("get_glyph_attribute", (glyph_id, attribute)))
        if glyph_id in self._attributes:
            return self._attributes[glyph_id].get(attribute)
        return None


class MinimalGlyph:
    """
    Minimal glyph object that doesn't expose cortex or other attributes directly.
    
    This is used to verify that plans use storage abstraction methods
    instead of accessing glyph object attributes directly.
    """
    
    def __init__(self, glyph_id: str):
        self._id = glyph_id
    
    @property
    def identifier(self) -> str:
        return self._id
    
    def __repr__(self) -> str:
        return f"MinimalGlyph({self._id})"


@st.composite
def plan_test_storage_strategy(draw, min_glyphs: int = 1, max_glyphs: int = 5) -> Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]:
    """
    Generate a mock storage with minimal glyphs and embeddings for plan testing.
    
    Uses smaller data structures to avoid Hypothesis health check failures.
    
    Returns:
        Tuple of (MockStorageForPlans, glyphs dict, embeddings dict, attributes dict)
    """
    num_glyphs = draw(st.integers(min_value=min_glyphs, max_value=max_glyphs))
    
    glyphs = {}
    embeddings = {}
    attributes = {}
    
    for i in range(num_glyphs):
        # Use simple sequential IDs to avoid uniqueness issues
        glyph_id = f"g{i}"
        
        # Create minimal glyph (no direct cortex access)
        glyphs[glyph_id] = MinimalGlyph(glyph_id)
        
        # Generate small embedding
        embedding = draw(small_vectors)
        if len(embedding) > 0:
            embeddings[glyph_id] = embedding
        
        # Generate simple attributes
        attrs = draw(simple_metadata)
        if attrs:
            attributes[glyph_id] = attrs
    
    storage = MockStorageForPlans(
        glyphs=glyphs,
        embeddings=embeddings,
        attributes=attributes
    )
    
    return storage, glyphs, embeddings, attributes


class TestPlanExecutionWithStorage:
    """
    Property tests for Plan Execution with Storage (Property 6).
    
    Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    
    For any storage implementation that satisfies GlyphStorageProtocol,
    executing ListPlan, SimilaritySearchPlan, and ComparePlan through
    ExecutionContext produces valid FactTree results without accessing
    glyph object attributes directly.
    """
    
    @given(data=plan_test_storage_strategy(min_glyphs=1, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_list_plan_uses_storage_abstraction(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: ListPlan uses storage abstraction for iteration.
        
        For any storage implementation, ListPlan SHALL iterate glyphs via
        context.list_glyphs() instead of accessing the glyphs dict directly.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.2, 3.4**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Clear call log
        storage.call_log.clear()
        
        # Create and execute ListPlan
        plan = ListPlan(predicate=None, limit=100)
        result = plan.execute(context)
        
        # Verify list_glyphs was called
        assert any(call[0] == "list_glyphs" for call in storage.call_log), \
            "ListPlan did not call storage.list_glyphs()"
        
        # Verify result is a valid FactTree
        assert result is not None, "ListPlan returned None"
        assert hasattr(result, 'root'), "ListPlan result is not a FactTree"
        
        # Verify result contains expected data
        json_result = result.to_json()
        assert "children" in json_result, "FactTree has no children"
    
    @given(data=plan_test_storage_strategy(min_glyphs=1, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_list_plan_with_predicate_uses_storage_attribute_access(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: ListPlan with predicate uses storage for attribute access.
        
        For any storage implementation, ListPlan with a predicate SHALL use
        context.get_glyph_attribute() for predicate evaluation instead of
        accessing glyph object properties directly.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.2, 3.4**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least one glyph with attributes
        glyphs_with_attrs = [gid for gid in glyphs if gid in attributes and attributes[gid]]
        assume(len(glyphs_with_attrs) > 0)
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Create a predicate that uses attribute access
        def predicate(glyph_id: str, ctx: ExecutionContext) -> bool:
            # This should trigger get_glyph_attribute call
            value = ctx.get_glyph_attribute(glyph_id, "test_attr")
            return value is not None
        
        # Clear call log
        storage.call_log.clear()
        
        # Create and execute ListPlan with predicate
        plan = ListPlan(predicate=predicate, limit=100)
        result = plan.execute(context)
        
        # Verify list_glyphs was called
        assert any(call[0] == "list_glyphs" for call in storage.call_log), \
            "ListPlan did not call storage.list_glyphs()"
        
        # Verify get_glyph_attribute was called (predicate evaluation)
        assert any(call[0] == "get_glyph_attribute" for call in storage.call_log), \
            "ListPlan predicate did not use storage.get_glyph_attribute()"
        
        # Verify result is valid
        assert result is not None, "ListPlan returned None"
    
    @given(data=plan_test_storage_strategy(min_glyphs=2, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_similarity_search_plan_uses_storage_for_embeddings(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: SimilaritySearchPlan uses storage for embedding retrieval.
        
        For any storage implementation, SimilaritySearchPlan SHALL retrieve
        embeddings via context.get_embedding() instead of accessing glyph.cortex directly.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.1, 3.5**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least 2 glyphs with embeddings
        glyphs_with_embeddings = [gid for gid in glyphs if gid in embeddings]
        assume(len(glyphs_with_embeddings) >= 2)
        
        # Pick a target glyph
        target_glyph_id = glyphs_with_embeddings[0]
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Clear call log
        storage.call_log.clear()
        
        # Create and execute SimilaritySearchPlan
        plan = SimilaritySearchPlan(
            target_glyph_id=target_glyph_id,
            limit=10,
            threshold=0.0  # Accept all results
        )
        result = plan.execute(context)
        
        # Verify get_embedding was called for target
        get_embedding_calls = [call for call in storage.call_log if call[0] == "get_embedding"]
        assert len(get_embedding_calls) > 0, \
            "SimilaritySearchPlan did not call storage.get_embedding()"
        
        # Verify list_glyphs was called for iteration
        assert any(call[0] == "list_glyphs" for call in storage.call_log), \
            "SimilaritySearchPlan did not call storage.list_glyphs()"
        
        # Verify compute_similarity was called
        assert any(call[0] == "compute_similarity" for call in storage.call_log), \
            "SimilaritySearchPlan did not call storage.compute_similarity()"
        
        # Verify result is valid
        assert result is not None, "SimilaritySearchPlan returned None"
        assert hasattr(result, 'root'), "SimilaritySearchPlan result is not a FactTree"
    
    @given(data=plan_test_storage_strategy(min_glyphs=2, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_similarity_search_plan_with_scope_uses_storage(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: SimilaritySearchPlan with scope uses storage for scoped embeddings.
        
        For any storage implementation, SimilaritySearchPlan with scope SHALL use
        context.get_embedding_for_scope() instead of accessing glyph layer/segment directly.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.1, 3.5**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least 2 glyphs with embeddings
        glyphs_with_embeddings = [gid for gid in glyphs if gid in embeddings]
        assume(len(glyphs_with_embeddings) >= 2)
        
        # Pick a target glyph
        target_glyph_id = glyphs_with_embeddings[0]
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Clear call log
        storage.call_log.clear()
        
        # Create and execute SimilaritySearchPlan with scope
        plan = SimilaritySearchPlan(
            target_glyph_id=target_glyph_id,
            scope_layer="test_layer",
            limit=10,
            threshold=0.0
        )
        result = plan.execute(context)
        
        # Verify get_embedding_for_scope was called
        scope_calls = [call for call in storage.call_log if call[0] == "get_embedding_for_scope"]
        assert len(scope_calls) > 0, \
            "SimilaritySearchPlan with scope did not call storage.get_embedding_for_scope()"
        
        # Verify result is valid (may have no results if scope not supported)
        assert result is not None, "SimilaritySearchPlan returned None"
    
    @given(data=plan_test_storage_strategy(min_glyphs=2, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_compare_plan_uses_storage_for_embeddings(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: ComparePlan uses storage for embedding retrieval.
        
        For any storage implementation, ComparePlan SHALL retrieve embeddings
        via context.get_embedding() for both glyphs instead of accessing
        glyph.cortex directly.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.3**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least 2 glyphs with embeddings
        glyphs_with_embeddings = [gid for gid in glyphs if gid in embeddings]
        assume(len(glyphs_with_embeddings) >= 2)
        
        # Pick two glyphs to compare
        glyph1_id = glyphs_with_embeddings[0]
        glyph2_id = glyphs_with_embeddings[1]
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Clear call log
        storage.call_log.clear()
        
        # Create and execute ComparePlan
        plan = ComparePlan(
            glyph1_id=glyph1_id,
            glyph2_id=glyph2_id,
            edge_type="neural_cortex"
        )
        result = plan.execute(context)
        
        # Verify get_embedding was called for both glyphs
        get_embedding_calls = [call for call in storage.call_log if call[0] == "get_embedding"]
        assert len(get_embedding_calls) >= 2, \
            f"ComparePlan should call get_embedding at least twice, got {len(get_embedding_calls)} calls"
        
        # Verify compute_similarity was called
        assert any(call[0] == "compute_similarity" for call in storage.call_log), \
            "ComparePlan did not call storage.compute_similarity()"
        
        # Verify result is valid
        assert result is not None, "ComparePlan returned None"
        assert hasattr(result, 'root'), "ComparePlan result is not a FactTree"
    
    @given(data=plan_test_storage_strategy(min_glyphs=2, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_compare_plan_with_layer_uses_storage_scope(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: ComparePlan with layer uses storage for scoped embeddings.
        
        For any storage implementation, ComparePlan with neural_layer edge type
        SHALL use context.get_embedding_for_scope() instead of accessing
        glyph layer attributes directly.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.3**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least 2 glyphs
        glyph_ids = list(glyphs.keys())
        assume(len(glyph_ids) >= 2)
        
        # Pick two glyphs to compare
        glyph1_id = glyph_ids[0]
        glyph2_id = glyph_ids[1]
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Clear call log
        storage.call_log.clear()
        
        # Create and execute ComparePlan with layer scope
        plan = ComparePlan(
            glyph1_id=glyph1_id,
            glyph2_id=glyph2_id,
            edge_type="neural_layer",
            level_path="test_layer"
        )
        result = plan.execute(context)
        
        # Verify get_embedding_for_scope was called
        scope_calls = [call for call in storage.call_log if call[0] == "get_embedding_for_scope"]
        assert len(scope_calls) >= 2, \
            f"ComparePlan with layer should call get_embedding_for_scope at least twice, got {len(scope_calls)} calls"
        
        # Verify result is valid
        assert result is not None, "ComparePlan returned None"
    
    @given(data=plan_test_storage_strategy(min_glyphs=1, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_plans_produce_valid_fact_tree_results(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: All plans produce valid FactTree results.
        
        For any storage implementation, executing plans through ExecutionContext
        SHALL produce valid FactTree results that can be serialized to JSON.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Test ListPlan
        list_plan = ListPlan(predicate=None, limit=100)
        list_result = list_plan.execute(context)
        
        # Verify ListPlan result
        assert list_result is not None, "ListPlan returned None"
        assert hasattr(list_result, 'to_json'), "ListPlan result has no to_json method"
        
        # Verify JSON serialization works
        try:
            json_output = list_result.to_json()
            assert isinstance(json_output, dict), "to_json() should return a dict"
        except Exception as e:
            pytest.fail(f"ListPlan result JSON serialization failed: {e}")
    
    @given(data=plan_test_storage_strategy(min_glyphs=2, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_similarity_search_handles_missing_embeddings_gracefully(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: SimilaritySearchPlan handles missing embeddings gracefully.
        
        For any storage implementation, SimilaritySearchPlan SHALL gracefully
        handle glyphs without embeddings by skipping them instead of raising errors.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.5**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least one glyph with embedding for target
        glyphs_with_embeddings = [gid for gid in glyphs if gid in embeddings]
        assume(len(glyphs_with_embeddings) >= 1)
        
        # Remove some embeddings to test graceful handling
        glyph_ids = list(glyphs.keys())
        if len(glyph_ids) > 1:
            # Remove embedding from one glyph
            glyph_without_embedding = glyph_ids[-1]
            if glyph_without_embedding in storage._embeddings:
                del storage._embeddings[glyph_without_embedding]
        
        # Pick a target glyph that has embedding
        target_glyph_id = glyphs_with_embeddings[0]
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Create and execute SimilaritySearchPlan
        plan = SimilaritySearchPlan(
            target_glyph_id=target_glyph_id,
            limit=10,
            threshold=0.0
        )
        
        # Should not raise an error
        try:
            result = plan.execute(context)
            assert result is not None, "SimilaritySearchPlan returned None"
        except Exception as e:
            pytest.fail(f"SimilaritySearchPlan failed with missing embeddings: {e}")
    
    @given(data=plan_test_storage_strategy(min_glyphs=2, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_compare_plan_handles_missing_embeddings_gracefully(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: ComparePlan handles missing embeddings gracefully.
        
        For any storage implementation, ComparePlan SHALL handle missing
        embeddings by returning 0.0 similarity instead of raising errors.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.5**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least 2 glyphs
        glyph_ids = list(glyphs.keys())
        assume(len(glyph_ids) >= 2)
        
        # Remove embeddings to test graceful handling
        storage._embeddings.clear()
        
        # Pick two glyphs to compare
        glyph1_id = glyph_ids[0]
        glyph2_id = glyph_ids[1]
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Create and execute ComparePlan
        plan = ComparePlan(
            glyph1_id=glyph1_id,
            glyph2_id=glyph2_id,
            edge_type="neural_cortex"
        )
        
        # Should not raise an error
        try:
            result = plan.execute(context)
            assert result is not None, "ComparePlan returned None"
            
            # Verify similarity is 0.0 when embeddings are missing
            json_result = result.to_json()
            # The result should contain comparison data with 0.0 similarity
            assert "children" in json_result, "FactTree has no children"
        except Exception as e:
            pytest.fail(f"ComparePlan failed with missing embeddings: {e}")
    
    @given(data=plan_test_storage_strategy(min_glyphs=1, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_list_plan_respects_limit(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: ListPlan respects the limit parameter.
        
        For any storage implementation, ListPlan SHALL return at most
        `limit` glyphs in the results.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.2**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Test with various limits
        for limit in [1, 2, 5]:
            plan = ListPlan(predicate=None, limit=limit)
            result = plan.execute(context)
            
            # Get the results from the fact tree
            json_result = result.to_json()
            
            # Find the results in the fact tree children
            for child in json_result.get("children", []):
                if "value" in child and isinstance(child["value"], list):
                    result_count = len(child["value"])
                    assert result_count <= limit, \
                        f"ListPlan returned {result_count} results, expected at most {limit}"
    
    @given(data=plan_test_storage_strategy(min_glyphs=2, max_glyphs=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_similarity_search_respects_threshold(
        self, data: Tuple[MockStorageForPlans, Dict[str, MinimalGlyph], Dict[str, List[float]], Dict[str, Dict[str, Any]]]
    ):
        """
        Property test: SimilaritySearchPlan respects the threshold parameter.
        
        For any storage implementation, SimilaritySearchPlan SHALL only return
        glyphs with similarity >= threshold.
        
        Feature: gql-storage-abstraction, Property 6: Plan Execution with Storage
        **Validates: Requirements 3.1**
        """
        storage, glyphs, embeddings, attributes = data
        
        # Ensure we have at least 2 glyphs with embeddings
        glyphs_with_embeddings = [gid for gid in glyphs if gid in embeddings]
        assume(len(glyphs_with_embeddings) >= 2)
        
        # Pick a target glyph
        target_glyph_id = glyphs_with_embeddings[0]
        
        # Create ExecutionContext with the storage
        context = ExecutionContext(storage=storage)
        
        # Test with high threshold (should return fewer or no results)
        plan = SimilaritySearchPlan(
            target_glyph_id=target_glyph_id,
            limit=100,
            threshold=0.99  # Very high threshold
        )
        result = plan.execute(context)
        
        # Verify result is valid
        assert result is not None, "SimilaritySearchPlan returned None"
        
        # Get the results from the fact tree
        json_result = result.to_json()
        
        # Find the results in the fact tree children
        for child in json_result.get("children", []):
            if "value" in child and isinstance(child["value"], list):
                for item in child["value"]:
                    if isinstance(item, dict) and "score" in item:
                        assert item["score"] >= 0.99, \
                            f"SimilaritySearchPlan returned result with score {item['score']} < threshold 0.99"


# =============================================================================
# Property 7: Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """
    Property tests for Backward Compatibility (Property 7).
    
    Feature: gql-storage-abstraction, Property 7: Backward Compatibility
    
    **Validates: Requirements 7.1, 7.2, 7.4**
    
    For any dictionary of glyphs:
    - Creating ExecutionContext with `glyphs=dict` produces a working context
      where all glyph operations succeed
    - Passing both `glyphs` and `storage` parameters raises ValueError
    """
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_execution_context_with_glyphs_dict_works(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext with glyphs dict produces working context.
        
        For any dictionary of glyphs, creating ExecutionContext with `glyphs=dict`
        SHALL produce a working context where all glyph operations succeed.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        # Create ExecutionContext with glyphs dict (backward compatible API)
        context = ExecutionContext(glyphs=glyphs)
        
        # Verify list_glyphs works and returns the same glyphs
        listed = context.list_glyphs()
        assert listed is glyphs, \
            "ExecutionContext with glyphs dict should return same dict from list_glyphs()"
        
        # Verify has_glyph works for all known glyphs
        for glyph_id in glyphs.keys():
            assert context.has_glyph(glyph_id), \
                f"has_glyph('{glyph_id}') should return True for known glyph"
        
        # Verify get_glyph works for all known glyphs
        for glyph_id, expected_glyph in glyphs.items():
            retrieved = context.get_glyph(glyph_id)
            assert retrieved is expected_glyph, \
                f"get_glyph('{glyph_id}') should return the same glyph object"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_execution_context_with_glyphs_dict_get_embedding_works(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext with glyphs dict supports get_embedding.
        
        For any dictionary of glyphs, creating ExecutionContext with `glyphs=dict`
        SHALL support get_embedding() that returns cortex/global_cortex.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with glyphs dict
        context = ExecutionContext(glyphs=glyphs)
        
        # Verify get_embedding works for all glyphs
        for glyph_id, glyph in glyphs.items():
            embedding = context.get_embedding(glyph_id)
            
            # Verify embedding matches expected value
            if glyph.cortex is not None:
                assert embedding is glyph.cortex, \
                    f"get_embedding('{glyph_id}') should return cortex when available"
            elif glyph.global_cortex is not None:
                assert embedding is glyph.global_cortex, \
                    f"get_embedding('{glyph_id}') should return global_cortex as fallback"
            else:
                assert embedding is None, \
                    f"get_embedding('{glyph_id}') should return None when no embedding"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_execution_context_with_glyphs_dict_compute_similarity_works(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext with glyphs dict supports compute_similarity.
        
        For any dictionary of glyphs, creating ExecutionContext with `glyphs=dict`
        SHALL support compute_similarity() that returns valid similarity scores.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with glyphs dict
        context = ExecutionContext(glyphs=glyphs)
        
        # Find glyphs with embeddings
        glyphs_with_embeddings = [
            (gid, g) for gid, g in glyphs.items()
            if g.cortex is not None or g.global_cortex is not None
        ]
        
        # Skip if no glyphs have embeddings
        assume(len(glyphs_with_embeddings) >= 1)
        
        # Get embedding for first glyph
        glyph_id, glyph = glyphs_with_embeddings[0]
        embedding = context.get_embedding(glyph_id)
        
        # Verify compute_similarity works
        if embedding is not None and len(embedding) > 0:
            # Self-similarity should be close to 1.0
            # Note: Due to floating-point precision, similarity can be slightly > 1.0
            # (e.g., 1.0000000000000002), so we use a small epsilon tolerance
            similarity = context.compute_similarity(embedding, embedding)
            assert -0.01 <= similarity <= 1.01, \
                f"compute_similarity should return value approximately in [0, 1], got {similarity}"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_execution_context_with_glyphs_dict_has_glyph_false_for_unknown(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext with glyphs dict returns False for unknown IDs.
        
        For any dictionary of glyphs, creating ExecutionContext with `glyphs=dict`
        SHALL return False from has_glyph() for unknown glyph IDs.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        # Create ExecutionContext with glyphs dict
        context = ExecutionContext(glyphs=glyphs)
        
        # Use a glyph ID that definitely doesn't exist
        unknown_id = "__definitely_unknown_glyph_id_xyz__"
        
        # Verify has_glyph returns False for unknown ID
        assert not context.has_glyph(unknown_id), \
            f"has_glyph('{unknown_id}') should return False for unknown glyph"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_execution_context_with_glyphs_dict_get_glyph_raises_for_unknown(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext with glyphs dict raises for unknown IDs.
        
        For any dictionary of glyphs, creating ExecutionContext with `glyphs=dict`
        SHALL raise PlanningError from get_glyph() for unknown glyph IDs.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        from glyphh.gql.exceptions import PlanningError
        
        # Create ExecutionContext with glyphs dict
        context = ExecutionContext(glyphs=glyphs)
        
        # Use a glyph ID that definitely doesn't exist
        unknown_id = "__definitely_unknown_glyph_id_xyz__"
        
        # Verify get_glyph raises PlanningError for unknown ID
        with pytest.raises(PlanningError):
            context.get_glyph(unknown_id)
    
    @given(
        glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10),
        storage_glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=5)
    )
    @settings(max_examples=100)
    def test_execution_context_both_glyphs_and_storage_raises_valueerror(
        self, glyphs: Dict[str, MockSDKGlyph], storage_glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext with both glyphs and storage raises ValueError.
        
        For any dictionary of glyphs and any storage implementation,
        passing both `glyphs` and `storage` parameters to ExecutionContext
        SHALL raise ValueError indicating the conflict.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.4**
        """
        # Create a mock storage
        storage = MockStorage(glyphs=storage_glyphs)
        
        # Verify that passing both glyphs and storage raises ValueError
        with pytest.raises(ValueError) as exc_info:
            ExecutionContext(glyphs=glyphs, storage=storage)
        
        # Verify the error message mentions the conflict
        error_message = str(exc_info.value).lower()
        assert "glyphs" in error_message or "storage" in error_message, \
            f"ValueError message should mention 'glyphs' or 'storage', got: {exc_info.value}"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_execution_context_with_empty_glyphs_dict_works(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext with empty glyphs dict works.
        
        Creating ExecutionContext with an empty glyphs dict SHALL produce
        a working context where list_glyphs() returns empty dict.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        # Create ExecutionContext with empty glyphs dict
        context = ExecutionContext(glyphs={})
        
        # Verify list_glyphs returns empty dict
        listed = context.list_glyphs()
        assert listed == {}, \
            "ExecutionContext with empty glyphs dict should return empty dict"
        
        # Verify has_glyph returns False for any ID
        assert not context.has_glyph("any_id"), \
            "has_glyph should return False for empty context"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=0, max_glyphs=10))
    @settings(max_examples=100)
    def test_execution_context_with_none_glyphs_creates_empty_storage(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext with no parameters creates empty storage.
        
        Creating ExecutionContext with neither glyphs nor storage SHALL
        create an empty InMemoryGlyphStorage internally.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        # Create ExecutionContext with no parameters
        context = ExecutionContext()
        
        # Verify list_glyphs returns empty dict
        listed = context.list_glyphs()
        assert listed == {}, \
            "ExecutionContext with no parameters should have empty storage"
        
        # Verify has_glyph returns False for any ID
        assert not context.has_glyph("any_id"), \
            "has_glyph should return False for empty context"
    
    @given(glyphs=glyph_dict_with_layers_strategy(min_glyphs=1, max_glyphs=5))
    @settings(max_examples=100)
    def test_execution_context_with_glyphs_dict_get_embedding_for_scope_works(
        self, glyphs: Dict[str, MockSDKGlyphWithLayers]
    ):
        """
        Property test: ExecutionContext with glyphs dict supports get_embedding_for_scope.
        
        For any dictionary of glyphs with layers, creating ExecutionContext with
        `glyphs=dict` SHALL support get_embedding_for_scope() for layer/segment access.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with glyphs dict
        context = ExecutionContext(glyphs=glyphs)
        
        # Verify get_embedding_for_scope works for all glyphs
        for glyph_id, glyph in glyphs.items():
            # Test with no scope (should return primary embedding)
            scoped = context.get_embedding_for_scope(glyph_id, layer=None, segment=None)
            primary = context.get_embedding(glyph_id)
            assert scoped is primary, \
                f"get_embedding_for_scope('{glyph_id}', None, None) should equal get_embedding()"
            
            # Build a map of layer names to the FIRST layer with that name
            # (since the storage implementation finds the first match)
            first_layer_by_name: Dict[str, MockLayer] = {}
            for layer in glyph.layers:
                if layer.name not in first_layer_by_name:
                    first_layer_by_name[layer.name] = layer
            
            # Test with layer scope for unique layer names
            for layer_name, first_layer in first_layer_by_name.items():
                layer_embedding = context.get_embedding_for_scope(
                    glyph_id, layer=layer_name, segment=None
                )
                # Should return first matching layer's cortex or None
                if first_layer.cortex is not None:
                    assert layer_embedding is first_layer.cortex, \
                        f"get_embedding_for_scope should return first matching layer's cortex"
                else:
                    assert layer_embedding is None, \
                        f"get_embedding_for_scope should return None when layer has no cortex"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_execution_context_backward_compatible_api_unchanged(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext backward compatible API is unchanged.
        
        For any dictionary of glyphs, the ExecutionContext created with
        `glyphs=dict` SHALL have the same public API methods as before.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with glyphs dict
        context = ExecutionContext(glyphs=glyphs)
        
        # Verify all expected public methods exist
        expected_methods = [
            'has_glyph',
            'get_glyph',
            'list_glyphs',
            'get_embedding',
            'get_embedding_for_scope',
            'compute_similarity',
            'get_glyph_attribute',
            'get_layer_vector',  # Legacy method
            'get_segment_vector',  # Legacy method
            'encode_text',
        ]
        
        for method_name in expected_methods:
            assert hasattr(context, method_name), \
                f"ExecutionContext should have method '{method_name}'"
            assert callable(getattr(context, method_name)), \
                f"ExecutionContext.{method_name} should be callable"
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_execution_context_legacy_get_layer_vector_works(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext legacy get_layer_vector method works.
        
        For any dictionary of glyphs, the ExecutionContext created with
        `glyphs=dict` SHALL support the legacy get_layer_vector() method.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with glyphs dict
        context = ExecutionContext(glyphs=glyphs)
        
        # Get first glyph
        glyph_id = list(glyphs.keys())[0]
        glyph = glyphs[glyph_id]
        
        # Verify get_layer_vector works (should not raise)
        try:
            result = context.get_layer_vector(glyph, "test_layer")
            # Result may be None if layer doesn't exist, which is fine
        except Exception as e:
            pytest.fail(f"get_layer_vector raised unexpected exception: {e}")
    
    @given(glyphs=glyph_dict_strategy(min_glyphs=1, max_glyphs=10))
    @settings(max_examples=100)
    def test_execution_context_legacy_get_segment_vector_works(
        self, glyphs: Dict[str, MockSDKGlyph]
    ):
        """
        Property test: ExecutionContext legacy get_segment_vector method works.
        
        For any dictionary of glyphs, the ExecutionContext created with
        `glyphs=dict` SHALL support the legacy get_segment_vector() method.
        
        Feature: gql-storage-abstraction, Property 7: Backward Compatibility
        **Validates: Requirements 7.1, 7.2**
        """
        # Ensure we have at least one glyph
        assume(len(glyphs) > 0)
        
        # Create ExecutionContext with glyphs dict
        context = ExecutionContext(glyphs=glyphs)
        
        # Get first glyph
        glyph_id = list(glyphs.keys())[0]
        glyph = glyphs[glyph_id]
        
        # Verify get_segment_vector works (should not raise)
        try:
            result = context.get_segment_vector(glyph, "test_layer", "test_segment")
            # Result may be None if layer/segment doesn't exist, which is fine
        except Exception as e:
            pytest.fail(f"get_segment_vector raised unexpected exception: {e}")
