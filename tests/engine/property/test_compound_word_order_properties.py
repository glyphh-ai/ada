"""
Property-based tests for Compound Match Word Order Invariance.

This module contains property-based tests using Hypothesis to verify
that compound vectors are order-independent (bundling is commutative).

**Validates: Property 16** - Compound Match Word Order Invariance
For any multi-word schema value, queries with words in different orders
(e.g., "pads brake" vs "brake pads") SHALL both match the value with
similar confidence.

**Validates: Requirements 10.5**
"""

import itertools
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from glyphh.encoder.base import Encoder
from glyphh.core.config import EncoderConfig
from glyphh.core.types import Vector
from glyphh.nl.schema_vectorizer import SchemaVectorizer, SchemaVector


# Generator Strategies

# Word generator - generates valid words with letters and numbers
word_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
    min_size=1, max_size=20
).filter(lambda x: x.strip())

# List of 2-3 words for compound values
compound_words_strategy = st.lists(
    word_strategy,
    min_size=2,
    max_size=3,
    unique=True  # Ensure unique words to make permutations meaningful
)


class TestCompoundMatchWordOrderInvariance:
    """
    Property tests for Compound Match Word Order Invariance (Property 16).
    
    **Validates: Property 16** - Compound Match Word Order Invariance
    For any multi-word schema value, queries with words in different orders
    (e.g., "pads brake" vs "brake pads") SHALL both match the value with
    similar confidence.
    
    **Validates: Requirements 10.5**
    """
    
    @given(words=compound_words_strategy)
    @settings(max_examples=100)
    def test_compound_vectors_identical_for_all_permutations(self, words: list):
        """
        Property test: Compound vectors are identical for all word order permutations.
        
        For any list of 2-3 words, generating compound vectors with different
        word orders (permutations) SHALL produce identical vectors because
        HDC bundling is commutative.
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate all permutations of the words
        permutations = list(itertools.permutations(words))
        
        # Generate compound vector for the first permutation (reference)
        reference_vector = vectorizer.vectorize_compound(list(permutations[0]))
        
        # Property: All permutations should produce identical vectors
        for perm in permutations[1:]:
            perm_vector = vectorizer.vectorize_compound(list(perm))
            
            assert np.array_equal(reference_vector.data, perm_vector.data), \
                f"Compound vector for {list(perm)} differs from {list(permutations[0])}"
    
    @given(words=compound_words_strategy)
    @settings(max_examples=100)
    def test_compound_vectors_same_similarity_for_all_permutations(self, words: list):
        """
        Property test: Matching against any permutation produces the same similarity score.
        
        For any list of 2-3 words, matching a query against compound vectors
        created from different word orders SHALL produce identical similarity
        scores because the vectors are identical.
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate all permutations of the words
        permutations = list(itertools.permutations(words))
        
        # Create a query vector (using the first word as query)
        query_vector = encoder.generate_symbol(words[0])
        
        # Compute similarity for each permutation's compound vector
        similarities = []
        for perm in permutations:
            compound_vector = vectorizer.vectorize_compound(list(perm))
            # Compute cosine similarity
            similarity = np.dot(query_vector.data, compound_vector.data) / (
                np.linalg.norm(query_vector.data) * np.linalg.norm(compound_vector.data)
            )
            similarities.append(similarity)
        
        # Property: All similarities should be identical
        reference_similarity = similarities[0]
        for i, sim in enumerate(similarities[1:], start=1):
            assert np.isclose(sim, reference_similarity, rtol=1e-10), \
                f"Similarity for permutation {list(permutations[i])} ({sim}) " \
                f"differs from reference ({reference_similarity})"
    
    @given(
        words=compound_words_strategy,
        dimension=st.integers(min_value=100, max_value=5000),
        seed=st.integers(min_value=1, max_value=1000000)
    )
    @settings(max_examples=100)
    def test_compound_vectors_identical_across_configs(
        self, words: list, dimension: int, seed: int
    ):
        """
        Property test: Compound vector order invariance holds across different configs.
        
        For any list of 2-3 words and any valid encoder configuration,
        compound vectors with different word orders SHALL be identical.
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Setup encoder and vectorizer with generated config
        config = EncoderConfig(dimension=dimension, seed=seed)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate all permutations of the words
        permutations = list(itertools.permutations(words))
        
        # Generate compound vector for the first permutation (reference)
        reference_vector = vectorizer.vectorize_compound(list(permutations[0]))
        
        # Property: All permutations should produce identical vectors
        for perm in permutations[1:]:
            perm_vector = vectorizer.vectorize_compound(list(perm))
            
            assert np.array_equal(reference_vector.data, perm_vector.data), \
                f"Compound vector for {list(perm)} differs from {list(permutations[0])} " \
                f"with config (dim={dimension}, seed={seed})"
    
    @given(words=compound_words_strategy)
    @settings(max_examples=100)
    def test_reversed_word_order_produces_identical_vector(self, words: list):
        """
        Property test: Reversed word order produces identical compound vector.
        
        For any list of 2-3 words, the compound vector for the original order
        SHALL be identical to the compound vector for the reversed order.
        
        This is a specific case of the general permutation invariance property.
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate compound vectors for original and reversed order
        original_vector = vectorizer.vectorize_compound(words)
        reversed_vector = vectorizer.vectorize_compound(list(reversed(words)))
        
        # Property: Original and reversed should produce identical vectors
        assert np.array_equal(original_vector.data, reversed_vector.data), \
            f"Compound vector for {words} differs from reversed {list(reversed(words))}"
    
    @given(words=compound_words_strategy)
    @settings(max_examples=100)
    def test_compound_vector_bipolarity_preserved_across_permutations(self, words: list):
        """
        Property test: Compound vectors remain bipolar across all permutations.
        
        For any list of 2-3 words and any permutation, the resulting compound
        vector SHALL be bipolar (all elements in {-1, +1}).
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate all permutations of the words
        permutations = list(itertools.permutations(words))
        
        # Property: All permutations should produce bipolar vectors
        for perm in permutations:
            compound_vector = vectorizer.vectorize_compound(list(perm))
            
            assert np.all(np.isin(compound_vector.data, [-1, 1])), \
                f"Compound vector for {list(perm)} contains non-bipolar values"
    
    @given(words=compound_words_strategy)
    @settings(max_examples=100)
    def test_compound_vector_dimension_preserved_across_permutations(self, words: list):
        """
        Property test: Compound vector dimension is preserved across all permutations.
        
        For any list of 2-3 words and any permutation, the resulting compound
        vector SHALL have the same dimension as the encoder config.
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate all permutations of the words
        permutations = list(itertools.permutations(words))
        
        # Property: All permutations should produce vectors with correct dimension
        for perm in permutations:
            compound_vector = vectorizer.vectorize_compound(list(perm))
            
            assert compound_vector.dimension == config.dimension, \
                f"Compound vector for {list(perm)} has dimension " \
                f"{compound_vector.dimension} instead of {config.dimension}"
    
    @given(words=compound_words_strategy)
    @settings(max_examples=100)
    def test_compound_vector_space_id_preserved_across_permutations(self, words: list):
        """
        Property test: Compound vector space_id is preserved across all permutations.
        
        For any list of 2-3 words and any permutation, the resulting compound
        vector SHALL have the same space_id as the encoder.
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate all permutations of the words
        permutations = list(itertools.permutations(words))
        
        # Property: All permutations should produce vectors with correct space_id
        for perm in permutations:
            compound_vector = vectorizer.vectorize_compound(list(perm))
            
            assert compound_vector.space_id == encoder.space_id, \
                f"Compound vector for {list(perm)} has space_id " \
                f"'{compound_vector.space_id}' instead of '{encoder.space_id}'"
    
    @given(
        word1=word_strategy,
        word2=word_strategy
    )
    @settings(max_examples=100)
    def test_two_word_compound_order_invariance(self, word1: str, word2: str):
        """
        Property test: Two-word compounds are order-invariant.
        
        For any two distinct words, the compound vector for [word1, word2]
        SHALL be identical to the compound vector for [word2, word1].
        
        This tests the specific case mentioned in the design doc:
        "pads brake" vs "brake pads".
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Skip if words are identical (permutation would be the same)
        assume(word1 != word2)
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate compound vectors for both orders
        vector_12 = vectorizer.vectorize_compound([word1, word2])
        vector_21 = vectorizer.vectorize_compound([word2, word1])
        
        # Property: Both orders should produce identical vectors
        assert np.array_equal(vector_12.data, vector_21.data), \
            f"Compound vector for [{word1}, {word2}] differs from [{word2}, {word1}]"
    
    @given(
        word1=word_strategy,
        word2=word_strategy,
        word3=word_strategy
    )
    @settings(max_examples=100)
    def test_three_word_compound_order_invariance(
        self, word1: str, word2: str, word3: str
    ):
        """
        Property test: Three-word compounds are order-invariant.
        
        For any three distinct words, all 6 permutations SHALL produce
        identical compound vectors.
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Skip if any words are identical (permutations would overlap)
        assume(word1 != word2 and word2 != word3 and word1 != word3)
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate all 6 permutations
        words = [word1, word2, word3]
        permutations = list(itertools.permutations(words))
        
        # Generate reference vector
        reference_vector = vectorizer.vectorize_compound(list(permutations[0]))
        
        # Property: All 6 permutations should produce identical vectors
        for perm in permutations[1:]:
            perm_vector = vectorizer.vectorize_compound(list(perm))
            
            assert np.array_equal(reference_vector.data, perm_vector.data), \
                f"Compound vector for {list(perm)} differs from {list(permutations[0])}"
    
    @given(words=compound_words_strategy)
    @settings(max_examples=100)
    def test_value_vectorization_order_invariance(self, words: list):
        """
        Property test: Value vectorization is order-invariant for multi-word values.
        
        For any multi-word value, vectorize_value() with different word orders
        SHALL produce identical vectors because it uses vectorize_compound()
        internally for multi-word values.
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create multi-word value strings from words
        value_original = " ".join(words)
        value_reversed = " ".join(reversed(words))
        
        # Generate value vectors
        vector_original = vectorizer.vectorize_value("test_role", value_original)
        vector_reversed = vectorizer.vectorize_value("test_role", value_reversed)
        
        # Property: Both word orders should produce identical vectors
        assert np.array_equal(vector_original.vector.data, vector_reversed.vector.data), \
            f"Value vector for '{value_original}' differs from '{value_reversed}'"
    
    @given(words=compound_words_strategy)
    @settings(max_examples=100)
    def test_matching_similarity_identical_for_permuted_queries(self, words: list):
        """
        Property test: Matching similarity is identical for permuted query words.
        
        For any multi-word schema value, queries with words in different orders
        SHALL produce identical similarity scores when matched against the
        schema value's compound vector.
        
        **Validates: Property 16**
        **Validates: Requirements 10.5**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema value compound vector (reference)
        schema_value = " ".join(words)
        schema_vector = vectorizer.vectorize_value("test_role", schema_value)
        
        # Generate all permutations of the words
        permutations = list(itertools.permutations(words))
        
        # Compute similarity for each permutation as a query
        similarities = []
        for perm in permutations:
            # Create query compound vector
            query_vector = vectorizer.vectorize_compound(list(perm))
            
            # Compute cosine similarity
            similarity = np.dot(query_vector.data, schema_vector.vector.data) / (
                np.linalg.norm(query_vector.data) * np.linalg.norm(schema_vector.vector.data)
            )
            similarities.append(similarity)
        
        # Property: All similarities should be identical (since vectors are identical)
        reference_similarity = similarities[0]
        for i, sim in enumerate(similarities[1:], start=1):
            assert np.isclose(sim, reference_similarity, rtol=1e-10), \
                f"Similarity for query permutation {list(permutations[i])} ({sim}) " \
                f"differs from reference ({reference_similarity})"

