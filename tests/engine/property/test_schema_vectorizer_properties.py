"""
Property-based tests for SchemaVectorizer.

This module contains property-based tests using Hypothesis to verify
universal correctness properties for schema vector generation.

**Validates: Property 1** - Schema Vector Bipolarity
For any role name or value in a model config, the generated schema vector
SHALL be bipolar (all elements in {-1, +1}) and have the same dimension
and space_id as the encoder.

**Validates: Requirements 1.1, 1.2, 1.3**
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from glyphh.encoder.base import Encoder
from glyphh.core.config import EncoderConfig, Layer, Segment, Role
from glyphh.nl.schema_vectorizer import SchemaVectorizer, SchemaVector


# Generator Strategies (from design doc)

# Role name generator - generates valid role names with letters, numbers, and underscores
role_names = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
    min_size=1, max_size=50
)

# Value generator (including multi-word) - generates arbitrary text values
values = st.text(min_size=1, max_size=100)


class TestSchemaVectorBipolarity:
    """
    Property tests for Schema Vector Bipolarity (Property 1).
    
    **Validates: Property 1** - Schema Vector Bipolarity
    For any role name or value in a model config, the generated schema vector
    SHALL be bipolar (all elements in {-1, +1}) and have the same dimension
    and space_id as the encoder.
    
    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    
    @given(role_name=role_names)
    @settings(max_examples=100)
    def test_vectorize_role_produces_bipolar_vector(self, role_name: str):
        """
        Property test: vectorize_role produces bipolar vectors.
        
        For any valid role name, the generated vector SHALL be bipolar
        (all elements in {-1, +1}).
        
        **Validates: Requirements 1.1**
        """
        # Setup encoder and vectorizer with fixed config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate vector for the role name
        schema_vector = vectorizer.vectorize_role(role_name)
        
        # Property: All vector elements must be in {-1, +1}
        assert np.all(np.isin(schema_vector.vector.data, [-1, 1])), \
            f"Vector for role '{role_name}' contains non-bipolar values"
    
    @given(role_name=role_names)
    @settings(max_examples=100)
    def test_vectorize_role_dimension_matches_encoder(self, role_name: str):
        """
        Property test: vectorize_role produces vectors with correct dimension.
        
        For any valid role name, the generated vector dimension SHALL match
        the encoder config dimension.
        
        **Validates: Requirements 1.1, 1.3**
        """
        # Setup encoder and vectorizer with fixed config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate vector for the role name
        schema_vector = vectorizer.vectorize_role(role_name)
        
        # Property: Vector dimension must match encoder config
        assert schema_vector.vector.dimension == config.dimension, \
            f"Vector dimension {schema_vector.vector.dimension} does not match " \
            f"encoder config dimension {config.dimension}"
    
    @given(role_name=role_names)
    @settings(max_examples=100)
    def test_vectorize_role_space_id_matches_encoder(self, role_name: str):
        """
        Property test: vectorize_role produces vectors with correct space_id.
        
        For any valid role name, the generated vector space_id SHALL match
        the encoder space_id.
        
        **Validates: Requirements 1.1, 1.3**
        """
        # Setup encoder and vectorizer with fixed config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate vector for the role name
        schema_vector = vectorizer.vectorize_role(role_name)
        
        # Property: Vector space_id must match encoder space_id
        assert schema_vector.vector.space_id == encoder.space_id, \
            f"Vector space_id '{schema_vector.vector.space_id}' does not match " \
            f"encoder space_id '{encoder.space_id}'"
    
    @given(value=values.filter(lambda x: x.strip()))  # Filter out whitespace-only
    @settings(max_examples=100)
    def test_vectorize_value_produces_bipolar_vector(self, value: str):
        """
        Property test: vectorize_value produces bipolar vectors.
        
        For any valid value, the generated vector SHALL be bipolar
        (all elements in {-1, +1}).
        
        **Validates: Requirements 1.2**
        """
        # Setup encoder and vectorizer with fixed config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate vector for the value
        schema_vector = vectorizer.vectorize_value("test_role", value)
        
        # Property: All vector elements must be in {-1, +1}
        assert np.all(np.isin(schema_vector.vector.data, [-1, 1])), \
            f"Vector for value '{value}' contains non-bipolar values"
    
    @given(value=values.filter(lambda x: x.strip()))  # Filter out whitespace-only
    @settings(max_examples=100)
    def test_vectorize_value_dimension_matches_encoder(self, value: str):
        """
        Property test: vectorize_value produces vectors with correct dimension.
        
        For any valid value, the generated vector dimension SHALL match
        the encoder config dimension.
        
        **Validates: Requirements 1.2, 1.3**
        """
        # Setup encoder and vectorizer with fixed config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate vector for the value
        schema_vector = vectorizer.vectorize_value("test_role", value)
        
        # Property: Vector dimension must match encoder config
        assert schema_vector.vector.dimension == config.dimension, \
            f"Vector dimension {schema_vector.vector.dimension} does not match " \
            f"encoder config dimension {config.dimension}"
    
    @given(value=values.filter(lambda x: x.strip()))  # Filter out whitespace-only
    @settings(max_examples=100)
    def test_vectorize_value_space_id_matches_encoder(self, value: str):
        """
        Property test: vectorize_value produces vectors with correct space_id.
        
        For any valid value, the generated vector space_id SHALL match
        the encoder space_id.
        
        **Validates: Requirements 1.2, 1.3**
        """
        # Setup encoder and vectorizer with fixed config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate vector for the value
        schema_vector = vectorizer.vectorize_value("test_role", value)
        
        # Property: Vector space_id must match encoder space_id
        assert schema_vector.vector.space_id == encoder.space_id, \
            f"Vector space_id '{schema_vector.vector.space_id}' does not match " \
            f"encoder space_id '{encoder.space_id}'"
    
    @given(
        role_name=role_names,
        dimension=st.integers(min_value=100, max_value=5000),
        seed=st.integers(min_value=1, max_value=1000000)
    )
    @settings(max_examples=100)
    def test_vectorize_role_bipolarity_across_configs(
        self, role_name: str, dimension: int, seed: int
    ):
        """
        Property test: vectorize_role produces bipolar vectors across different configs.
        
        For any valid role name and any valid encoder configuration, the generated
        vector SHALL be bipolar (all elements in {-1, +1}) and have the correct
        dimension and space_id.
        
        **Validates: Requirements 1.1, 1.3**
        """
        # Setup encoder and vectorizer with generated config
        config = EncoderConfig(dimension=dimension, seed=seed)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate vector for the role name
        schema_vector = vectorizer.vectorize_role(role_name)
        
        # Property 1: All vector elements must be in {-1, +1}
        assert np.all(np.isin(schema_vector.vector.data, [-1, 1])), \
            f"Vector for role '{role_name}' with config (dim={dimension}, seed={seed}) " \
            f"contains non-bipolar values"
        
        # Property 2: Vector dimension must match encoder config
        assert schema_vector.vector.dimension == dimension, \
            f"Vector dimension {schema_vector.vector.dimension} does not match " \
            f"config dimension {dimension}"
        
        # Property 3: Vector space_id must match encoder space_id
        assert schema_vector.vector.space_id == encoder.space_id, \
            f"Vector space_id does not match encoder space_id"
    
    @given(
        value=values.filter(lambda x: x.strip()),  # Filter out whitespace-only
        dimension=st.integers(min_value=100, max_value=5000),
        seed=st.integers(min_value=1, max_value=1000000)
    )
    @settings(max_examples=100)
    def test_vectorize_value_bipolarity_across_configs(
        self, value: str, dimension: int, seed: int
    ):
        """
        Property test: vectorize_value produces bipolar vectors across different configs.
        
        For any valid value and any valid encoder configuration, the generated
        vector SHALL be bipolar (all elements in {-1, +1}) and have the correct
        dimension and space_id.
        
        **Validates: Requirements 1.2, 1.3**
        """
        # Setup encoder and vectorizer with generated config
        config = EncoderConfig(dimension=dimension, seed=seed)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate vector for the value
        schema_vector = vectorizer.vectorize_value("test_role", value)
        
        # Property 1: All vector elements must be in {-1, +1}
        assert np.all(np.isin(schema_vector.vector.data, [-1, 1])), \
            f"Vector for value '{value}' with config (dim={dimension}, seed={seed}) " \
            f"contains non-bipolar values"
        
        # Property 2: Vector dimension must match encoder config
        assert schema_vector.vector.dimension == dimension, \
            f"Vector dimension {schema_vector.vector.dimension} does not match " \
            f"config dimension {dimension}"
        
        # Property 3: Vector space_id must match encoder space_id
        assert schema_vector.vector.space_id == encoder.space_id, \
            f"Vector space_id does not match encoder space_id"
    
    @given(words=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
            min_size=1, max_size=20
        ).filter(lambda x: x.strip()),
        min_size=1, max_size=5
    ))
    @settings(max_examples=100)
    def test_vectorize_compound_produces_bipolar_vector(self, words: list):
        """
        Property test: vectorize_compound produces bipolar vectors.
        
        For any list of valid words, the generated compound vector SHALL be
        bipolar (all elements in {-1, +1}).
        
        **Validates: Requirements 10.1** (compound values as single units)
        """
        # Setup encoder and vectorizer with fixed config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate compound vector
        compound_vector = vectorizer.vectorize_compound(words)
        
        # Property: All vector elements must be in {-1, +1}
        assert np.all(np.isin(compound_vector.data, [-1, 1])), \
            f"Compound vector for words {words} contains non-bipolar values"
    
    @given(words=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
            min_size=1, max_size=20
        ).filter(lambda x: x.strip()),
        min_size=1, max_size=5
    ))
    @settings(max_examples=100)
    def test_vectorize_compound_dimension_matches_encoder(self, words: list):
        """
        Property test: vectorize_compound produces vectors with correct dimension.
        
        For any list of valid words, the generated compound vector dimension
        SHALL match the encoder config dimension.
        
        **Validates: Requirements 10.1, 1.3**
        """
        # Setup encoder and vectorizer with fixed config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate compound vector
        compound_vector = vectorizer.vectorize_compound(words)
        
        # Property: Vector dimension must match encoder config
        assert compound_vector.dimension == config.dimension, \
            f"Compound vector dimension {compound_vector.dimension} does not match " \
            f"encoder config dimension {config.dimension}"
    
    @given(words=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
            min_size=1, max_size=20
        ).filter(lambda x: x.strip()),
        min_size=1, max_size=5
    ))
    @settings(max_examples=100)
    def test_vectorize_compound_space_id_matches_encoder(self, words: list):
        """
        Property test: vectorize_compound produces vectors with correct space_id.
        
        For any list of valid words, the generated compound vector space_id
        SHALL match the encoder space_id.
        
        **Validates: Requirements 10.1, 1.3**
        """
        # Setup encoder and vectorizer with fixed config
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate compound vector
        compound_vector = vectorizer.vectorize_compound(words)
        
        # Property: Vector space_id must match encoder space_id
        assert compound_vector.space_id == encoder.space_id, \
            f"Compound vector space_id '{compound_vector.space_id}' does not match " \
            f"encoder space_id '{encoder.space_id}'"



class TestSchemaVectorCachingConsistency:
    """
    Property tests for Schema Vector Caching Consistency (Property 2).
    
    **Validates: Property 2** - Schema Vector Caching Consistency
    For any model config, calling `vectorize_schema()` multiple times without
    config changes SHALL return identical vectors (same data, same cache hits).
    
    **Validates: Requirements 1.4**
    """
    
    # Strategy for generating valid layer names
    layer_names = st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
        min_size=1, max_size=20
    ).filter(lambda x: x.strip())
    
    # Strategy for generating valid segment names
    segment_names = st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
        min_size=1, max_size=20
    ).filter(lambda x: x.strip())
    
    # Strategy for generating valid role names
    role_name_strategy = st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
        min_size=1, max_size=20
    ).filter(lambda x: x.strip())
    
    @given(
        layer_name=layer_names,
        segment_name=segment_names,
        role_name=role_name_strategy
    )
    @settings(max_examples=100)
    def test_vectorize_schema_returns_same_reference_on_multiple_calls(
        self, layer_name: str, segment_name: str, role_name: str
    ):
        """
        Property test: vectorize_schema returns the same dictionary reference.
        
        For any model config, calling vectorize_schema() multiple times SHALL
        return the same dictionary reference (cached result).
        
        **Validates: Requirements 1.4**
        """
        # Create config with generated names
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name=layer_name,
                    segments=[
                        Segment(
                            name=segment_name,
                            roles=[Role(name=role_name)]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Call vectorize_schema multiple times
        result1 = vectorizer.vectorize_schema()
        result2 = vectorizer.vectorize_schema()
        result3 = vectorizer.vectorize_schema()
        
        # Property: All calls should return the same dictionary reference
        assert result1 is result2, \
            "Second call to vectorize_schema() should return same reference"
        assert result2 is result3, \
            "Third call to vectorize_schema() should return same reference"
    
    @given(
        layer_name=layer_names,
        segment_name=segment_names,
        role_name=role_name_strategy,
        num_calls=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=100)
    def test_vectorize_schema_vectors_identical_across_calls(
        self, layer_name: str, segment_name: str, role_name: str, num_calls: int
    ):
        """
        Property test: vectors are identical across multiple vectorize_schema calls.
        
        For any model config, calling vectorize_schema() multiple times SHALL
        return dictionaries with identical vectors (same numpy array data).
        
        **Validates: Requirements 1.4**
        """
        # Create config with generated names
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name=layer_name,
                    segments=[
                        Segment(
                            name=segment_name,
                            roles=[Role(name=role_name)]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Get first result
        first_result = vectorizer.vectorize_schema()
        first_vectors = {k: v.vector.data.copy() for k, v in first_result.items()}
        
        # Call vectorize_schema multiple times and verify vectors are identical
        for i in range(num_calls - 1):
            current_result = vectorizer.vectorize_schema()
            
            # Property: Same keys should be present
            assert set(current_result.keys()) == set(first_vectors.keys()), \
                f"Call {i+2} has different keys than first call"
            
            # Property: All vectors should have identical data
            for key in first_vectors:
                assert np.array_equal(
                    current_result[key].vector.data,
                    first_vectors[key]
                ), f"Vector for key '{key}' differs on call {i+2}"
    
    @given(
        num_roles=st.integers(min_value=1, max_value=5),
        dimension=st.integers(min_value=100, max_value=2000),
        seed=st.integers(min_value=1, max_value=1000000)
    )
    @settings(max_examples=100)
    def test_vectorize_schema_caching_with_multiple_roles(
        self, num_roles: int, dimension: int, seed: int
    ):
        """
        Property test: caching works correctly with multiple roles.
        
        For any model config with multiple roles, calling vectorize_schema()
        multiple times SHALL return identical vectors for all roles.
        
        **Validates: Requirements 1.4**
        """
        # Create config with multiple roles
        roles = [Role(name=f"role_{i}") for i in range(num_roles)]
        config = EncoderConfig(
            dimension=dimension,
            seed=seed,
            layers=[
                Layer(
                    name="test_layer",
                    segments=[
                        Segment(
                            name="test_segment",
                            roles=roles
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # First call
        result1 = vectorizer.vectorize_schema()
        
        # Second call
        result2 = vectorizer.vectorize_schema()
        
        # Property: Same reference
        assert result1 is result2, \
            "Multiple calls should return same dictionary reference"
        
        # Property: All role vectors should be present
        for i in range(num_roles):
            role_key = f"role_{i}"
            assert role_key in result1, \
                f"Role '{role_key}' should be in schema vectors"
    
    @given(
        num_segments=st.integers(min_value=1, max_value=3),
        roles_per_segment=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=100)
    def test_vectorize_schema_caching_with_multiple_segments(
        self, num_segments: int, roles_per_segment: int
    ):
        """
        Property test: caching works correctly with multiple segments.
        
        For any model config with multiple segments, calling vectorize_schema()
        multiple times SHALL return identical vectors for all roles across segments.
        
        **Validates: Requirements 1.4**
        """
        # Create config with multiple segments
        segments = []
        for seg_idx in range(num_segments):
            roles = [Role(name=f"role_s{seg_idx}_r{r}") for r in range(roles_per_segment)]
            segments.append(Segment(name=f"segment_{seg_idx}", roles=roles))
        
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(name="test_layer", segments=segments)
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # First call - store vector data
        result1 = vectorizer.vectorize_schema()
        stored_vectors = {k: v.vector.data.copy() for k, v in result1.items()}
        
        # Second call
        result2 = vectorizer.vectorize_schema()
        
        # Property: Same reference
        assert result1 is result2, \
            "Multiple calls should return same dictionary reference"
        
        # Property: All vectors should be identical
        for key, stored_data in stored_vectors.items():
            assert np.array_equal(result2[key].vector.data, stored_data), \
                f"Vector for '{key}' should be identical across calls"
    
    @given(
        layer_name=layer_names,
        segment_name=segment_names,
        role_name=role_name_strategy
    )
    @settings(max_examples=100)
    def test_vectorize_schema_config_hash_unchanged_on_multiple_calls(
        self, layer_name: str, segment_name: str, role_name: str
    ):
        """
        Property test: config hash remains unchanged across multiple calls.
        
        For any model config, calling vectorize_schema() multiple times without
        config changes SHALL maintain the same config hash.
        
        **Validates: Requirements 1.4**
        """
        # Create config with generated names
        config = EncoderConfig(
            dimension=1000,
            seed=42,
            layers=[
                Layer(
                    name=layer_name,
                    segments=[
                        Segment(
                            name=segment_name,
                            roles=[Role(name=role_name)]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # First call
        vectorizer.vectorize_schema()
        hash1 = vectorizer.get_config_hash()
        
        # Second call
        vectorizer.vectorize_schema()
        hash2 = vectorizer.get_config_hash()
        
        # Third call
        vectorizer.vectorize_schema()
        hash3 = vectorizer.get_config_hash()
        
        # Property: Config hash should remain unchanged
        assert hash1 == hash2 == hash3, \
            "Config hash should remain unchanged across multiple calls"
    
    @given(
        dimension=st.integers(min_value=100, max_value=5000),
        seed=st.integers(min_value=1, max_value=1000000)
    )
    @settings(max_examples=100)
    def test_vectorize_schema_vector_properties_preserved_on_cache_hit(
        self, dimension: int, seed: int
    ):
        """
        Property test: vector properties are preserved on cache hits.
        
        For any model config, cached vectors SHALL maintain their properties
        (bipolarity, dimension, space_id) across multiple calls.
        
        **Validates: Requirements 1.4**
        """
        # Create config
        config = EncoderConfig(
            dimension=dimension,
            seed=seed,
            layers=[
                Layer(
                    name="test_layer",
                    segments=[
                        Segment(
                            name="test_segment",
                            roles=[Role(name="test_role")]
                        )
                    ]
                )
            ]
        )
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # First call
        result1 = vectorizer.vectorize_schema()
        
        # Second call (cache hit)
        result2 = vectorizer.vectorize_schema()
        
        # Property: All vectors should maintain their properties
        for key in result2:
            vector = result2[key].vector
            
            # Bipolarity preserved
            assert np.all(np.isin(vector.data, [-1, 1])), \
                f"Vector for '{key}' should remain bipolar on cache hit"
            
            # Dimension preserved
            assert vector.dimension == dimension, \
                f"Vector for '{key}' should maintain dimension on cache hit"
            
            # Space ID preserved
            assert vector.space_id == encoder.space_id, \
                f"Vector for '{key}' should maintain space_id on cache hit"
    
    @given(
        num_layers=st.integers(min_value=1, max_value=3),
        segments_per_layer=st.integers(min_value=1, max_value=2),
        roles_per_segment=st.integers(min_value=1, max_value=2)
    )
    @settings(max_examples=100)
    def test_vectorize_schema_caching_with_complex_hierarchy(
        self, num_layers: int, segments_per_layer: int, roles_per_segment: int
    ):
        """
        Property test: caching works correctly with complex config hierarchies.
        
        For any model config with multiple layers, segments, and roles,
        calling vectorize_schema() multiple times SHALL return identical results.
        
        **Validates: Requirements 1.4**
        """
        # Create complex config hierarchy
        layers = []
        for layer_idx in range(num_layers):
            segments = []
            for seg_idx in range(segments_per_layer):
                roles = [
                    Role(name=f"role_l{layer_idx}_s{seg_idx}_r{r}")
                    for r in range(roles_per_segment)
                ]
                segments.append(Segment(name=f"segment_{layer_idx}_{seg_idx}", roles=roles))
            layers.append(Layer(name=f"layer_{layer_idx}", segments=segments))
        
        config = EncoderConfig(dimension=1000, seed=42, layers=layers)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # First call - store all vector data
        result1 = vectorizer.vectorize_schema()
        stored_data = {k: v.vector.data.copy() for k, v in result1.items()}
        
        # Multiple subsequent calls
        for call_num in range(3):
            result = vectorizer.vectorize_schema()
            
            # Property: Same reference
            assert result is result1, \
                f"Call {call_num + 2} should return same reference"
            
            # Property: Same keys
            assert set(result.keys()) == set(stored_data.keys()), \
                f"Call {call_num + 2} should have same keys"
            
            # Property: Identical vector data
            for key in stored_data:
                assert np.array_equal(result[key].vector.data, stored_data[key]), \
                    f"Vector for '{key}' should be identical on call {call_num + 2}"
