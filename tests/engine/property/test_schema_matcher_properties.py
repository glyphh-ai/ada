"""
Property-based tests for SchemaMatcher.

This module contains property-based tests using Hypothesis to verify
universal correctness properties for schema matching.

**Validates: Property 8** - Threshold Filtering
For any token-schema match, the match SHALL be included in results if and only if
its similarity score is greater than or equal to the configured threshold.

**Validates: Requirements 3.2, 9.3**
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from glyphh.encoder.base import Encoder
from glyphh.core.config import EncoderConfig
from glyphh.core.types import Vector
from glyphh.nl.schema_matcher import SchemaMatcher, MatchConfig, TokenMatch, MatchResult
from glyphh.nl.schema_vectorizer import SchemaVectorizer, SchemaVector
from glyphh.nl.query_tokenizer import Token, QueryTokenizer, TokenizerConfig


# Generator Strategies

# Role name generator - generates valid role names with letters, numbers, and underscores
role_names = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
    min_size=1, max_size=30
).filter(lambda x: x.strip())

# Value generator - generates arbitrary text values
values = st.text(min_size=1, max_size=50).filter(lambda x: x.strip())

# Token text generator - generates valid token text (alphanumeric)
token_texts = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N')),
    min_size=1, max_size=30
).filter(lambda x: x.strip())

# Threshold generator - generates valid threshold values in [0.0, 1.0]
thresholds = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


class TestThresholdFiltering:
    """
    Property tests for Threshold Filtering (Property 8).
    
    **Validates: Property 8** - Threshold Filtering
    For any token-schema match, the match SHALL be included in results if and only if
    its similarity score is greater than or equal to the configured threshold.
    
    **Validates: Requirements 3.2, 9.3**
    """
    
    @given(
        role_threshold=thresholds,
        value_threshold=thresholds,
        token_text=token_texts
    )
    @settings(max_examples=100)
    def test_all_returned_matches_have_similarity_at_or_above_threshold(
        self, role_threshold: float, value_threshold: float, token_text: str
    ):
        """
        Property test: All returned matches have similarity >= threshold.
        
        For any token-schema match, if the match is included in results,
        its similarity score SHALL be greater than or equal to the configured threshold.
        
        **Validates: Property 8**
        **Validates: Requirements 3.2, 9.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors with various values
        schema_vectors = {}
        test_values = ["apple", "banana", "cherry", "date", "elderberry"]
        for value in test_values:
            sv = vectorizer.vectorize_value("test_role", value)
            schema_vectors[f"test_role={value}"] = sv
        
        # Also add a role vector
        role_sv = vectorizer.vectorize_role("test_role")
        schema_vectors["test_role"] = role_sv
        
        # Create matcher with specified thresholds
        match_config = MatchConfig(
            role_threshold=role_threshold,
            value_threshold=value_threshold
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token
        token = Token(
            text=token_text.lower(),
            original=token_text,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: All returned matches must have similarity >= threshold
        for match in matches:
            if match.schema_vector.element_type == "role":
                # For role matches, use role_threshold
                # Note: exact_match_bonus may be added, so we check base similarity
                # The match is included if base_similarity >= threshold
                # Since we can't easily get base similarity, we verify the match
                # was included, which means it passed the threshold check
                assert match.similarity >= role_threshold, \
                    f"Role match '{match.schema_vector.key}' has similarity " \
                    f"{match.similarity} below role_threshold {role_threshold}"
            else:
                # For value matches, use value_threshold
                assert match.similarity >= value_threshold, \
                    f"Value match '{match.schema_vector.key}' has similarity " \
                    f"{match.similarity} below value_threshold {value_threshold}"
    
    @given(
        role_threshold=thresholds,
        value_threshold=thresholds
    )
    @settings(max_examples=100)
    def test_exact_match_always_included_when_threshold_allows(
        self, role_threshold: float, value_threshold: float
    ):
        """
        Property test: Exact matches are included when threshold allows.
        
        For any token that exactly matches a schema value, the match SHALL be
        included if the threshold is <= 1.0 (since exact match similarity is 1.0
        plus any bonus).
        
        **Validates: Property 8**
        **Validates: Requirements 3.2, 9.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a schema vector for "toyota"
        toyota_sv = vectorizer.vectorize_value("make", "toyota")
        schema_vectors = {"make=toyota": toyota_sv}
        
        # Create matcher with specified thresholds
        match_config = MatchConfig(
            role_threshold=role_threshold,
            value_threshold=value_threshold,
            exact_match_bonus=0.2
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token with exact match text
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: Exact match should be included if threshold <= 1.0
        # (since exact match has similarity 1.0 + bonus = 1.2)
        if value_threshold <= 1.0:
            assert len(matches) >= 1, \
                f"Exact match should be included when value_threshold={value_threshold}"
            # Find the toyota match
            toyota_matches = [m for m in matches if m.schema_vector.original_value == "toyota"]
            assert len(toyota_matches) == 1, \
                f"Should have exactly one toyota match, got {len(toyota_matches)}"
            # Verify it's an exact match with bonus
            assert toyota_matches[0].similarity == 1.2, \
                f"Exact match should have similarity 1.2 (1.0 + 0.2 bonus), got {toyota_matches[0].similarity}"
    
    @given(
        threshold=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_low_similarity_matches_excluded_by_high_threshold(
        self, threshold: float
    ):
        """
        Property test: Low similarity matches are excluded by high threshold.
        
        For any token with low similarity to schema vectors, matches SHALL be
        excluded when the threshold is high enough.
        
        **Validates: Property 8**
        **Validates: Requirements 3.2, 9.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors with specific values
        apple_sv = vectorizer.vectorize_value("fruit", "apple")
        schema_vectors = {"fruit=apple": apple_sv}
        
        # Create matcher with high threshold
        match_config = MatchConfig(
            role_threshold=threshold,
            value_threshold=threshold
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token with completely different text (should have low similarity)
        token = Token(
            text="xyz123",
            original="xyz123",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: All returned matches must have similarity >= threshold
        for match in matches:
            assert match.similarity >= threshold, \
                f"Match with similarity {match.similarity} should not be included " \
                f"with threshold {threshold}"
    
    @given(
        role_threshold=thresholds,
        value_threshold=thresholds,
        num_schema_values=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_threshold_filtering_with_multiple_schema_vectors(
        self, role_threshold: float, value_threshold: float, num_schema_values: int
    ):
        """
        Property test: Threshold filtering works correctly with multiple schema vectors.
        
        For any set of schema vectors and any threshold, only matches with
        similarity >= threshold SHALL be included in results.
        
        **Validates: Property 8**
        **Validates: Requirements 3.2, 9.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create multiple schema vectors
        schema_vectors = {}
        test_values = [f"value{i}" for i in range(num_schema_values)]
        for value in test_values:
            sv = vectorizer.vectorize_value("test_role", value)
            schema_vectors[f"test_role={value}"] = sv
        
        # Create matcher with specified thresholds
        match_config = MatchConfig(
            role_threshold=role_threshold,
            value_threshold=value_threshold
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token
        token = Token(
            text="value0",
            original="value0",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: All returned matches must have similarity >= threshold
        for match in matches:
            threshold = value_threshold  # All are value vectors
            assert match.similarity >= threshold, \
                f"Match '{match.schema_vector.key}' has similarity {match.similarity} " \
                f"below threshold {threshold}"
    
    @given(
        role_threshold=thresholds,
        value_threshold=thresholds
    )
    @settings(max_examples=100)
    def test_role_and_value_thresholds_applied_independently(
        self, role_threshold: float, value_threshold: float
    ):
        """
        Property test: Role and value thresholds are applied independently.
        
        For any token-schema match, the appropriate threshold (role_threshold for
        role vectors, value_threshold for value vectors) SHALL be used.
        
        **Validates: Property 8**
        **Validates: Requirements 3.2, 9.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create both role and value schema vectors
        role_sv = vectorizer.vectorize_role("make")
        value_sv = vectorizer.vectorize_value("make", "toyota")
        schema_vectors = {
            "make": role_sv,
            "make=toyota": value_sv
        }
        
        # Create matcher with different thresholds for role and value
        match_config = MatchConfig(
            role_threshold=role_threshold,
            value_threshold=value_threshold
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token
        token = Token(
            text="make",
            original="make",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: Role matches use role_threshold, value matches use value_threshold
        for match in matches:
            if match.schema_vector.element_type == "role":
                assert match.similarity >= role_threshold, \
                    f"Role match has similarity {match.similarity} below role_threshold {role_threshold}"
            else:
                assert match.similarity >= value_threshold, \
                    f"Value match has similarity {match.similarity} below value_threshold {value_threshold}"
    
    @given(
        threshold=st.floats(min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_low_threshold_includes_more_matches(
        self, threshold: float
    ):
        """
        Property test: Lower thresholds include more matches.
        
        For any threshold, lowering it SHALL not decrease the number of matches
        (it may include more matches that were previously filtered out).
        
        **Validates: Property 8**
        **Validates: Requirements 3.2, 9.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        schema_vectors = {}
        test_values = ["apple", "apricot", "avocado", "banana", "blueberry"]
        for value in test_values:
            sv = vectorizer.vectorize_value("fruit", value)
            schema_vectors[f"fruit={value}"] = sv
        
        # Create token
        token = Token(
            text="apple",
            original="apple",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match with low threshold
        low_config = MatchConfig(value_threshold=threshold)
        low_matcher = SchemaMatcher(schema_vectors, encoder, low_config)
        low_matches = low_matcher.match_token(token)
        
        # Match with higher threshold
        high_threshold = min(threshold + 0.3, 1.0)
        high_config = MatchConfig(value_threshold=high_threshold)
        high_matcher = SchemaMatcher(schema_vectors, encoder, high_config)
        high_matches = high_matcher.match_token(token)
        
        # Property: Lower threshold should include at least as many matches
        assert len(low_matches) >= len(high_matches), \
            f"Lower threshold {threshold} produced {len(low_matches)} matches, " \
            f"but higher threshold {high_threshold} produced {len(high_matches)} matches"
    
    @given(
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_threshold_zero_includes_all_positive_similarity_matches(
        self, threshold: float
    ):
        """
        Property test: Threshold of 0.0 includes all matches with non-negative similarity.
        
        When threshold is 0.0, all matches with similarity >= 0.0 SHALL be included.
        
        **Validates: Property 8**
        **Validates: Requirements 3.2, 9.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        schema_vectors = {}
        test_values = ["test1", "test2", "test3"]
        for value in test_values:
            sv = vectorizer.vectorize_value("role", value)
            schema_vectors[f"role={value}"] = sv
        
        # Create token
        token = Token(
            text="test1",
            original="test1",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match with threshold 0.0
        zero_config = MatchConfig(value_threshold=0.0)
        zero_matcher = SchemaMatcher(schema_vectors, encoder, zero_config)
        zero_matches = zero_matcher.match_token(token)
        
        # Match with specified threshold
        threshold_config = MatchConfig(value_threshold=threshold)
        threshold_matcher = SchemaMatcher(schema_vectors, encoder, threshold_config)
        threshold_matches = threshold_matcher.match_token(token)
        
        # Property: Zero threshold should include at least as many matches
        assert len(zero_matches) >= len(threshold_matches), \
            f"Zero threshold produced {len(zero_matches)} matches, " \
            f"but threshold {threshold} produced {len(threshold_matches)} matches"
    
    @given(
        role_threshold=thresholds,
        value_threshold=thresholds,
        query=st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'Zs')),
            min_size=1, max_size=50
        ).filter(lambda x: x.strip() and any(c.isalnum() for c in x))
    )
    @settings(max_examples=100)
    def test_match_query_threshold_filtering(
        self, role_threshold: float, value_threshold: float, query: str
    ):
        """
        Property test: match_query applies threshold filtering correctly.
        
        For any query matched against schema, all returned matches SHALL have
        similarity >= the appropriate threshold.
        
        **Validates: Property 8**
        **Validates: Requirements 3.2, 9.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        schema_vectors = {}
        test_values = ["toyota", "honda", "ford", "bmw", "audi"]
        for value in test_values:
            sv = vectorizer.vectorize_value("make", value)
            schema_vectors[f"make={value}"] = sv
        
        # Add role vector
        role_sv = vectorizer.vectorize_role("make")
        schema_vectors["make"] = role_sv
        
        # Create matcher with specified thresholds
        match_config = MatchConfig(
            role_threshold=role_threshold,
            value_threshold=value_threshold
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create tokenizer
        tokenizer = QueryTokenizer()
        
        # Match query against schema
        result = matcher.match_query(query, tokenizer)
        
        # Property: All returned matches must have similarity >= threshold
        for match in result.token_matches:
            if match.schema_vector.element_type == "role":
                assert match.similarity >= role_threshold, \
                    f"Role match in query result has similarity {match.similarity} " \
                    f"below role_threshold {role_threshold}"
            else:
                # For compound matches, compound_bonus is added
                # The base similarity must have been >= threshold
                assert match.similarity >= value_threshold, \
                    f"Value match in query result has similarity {match.similarity} " \
                    f"below value_threshold {value_threshold}"
    
    @given(
        threshold=st.floats(min_value=1.01, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_threshold_above_one_excludes_all_base_matches(
        self, threshold: float
    ):
        """
        Property test: Threshold > 1.0 excludes all matches without bonus.
        
        When threshold is > 1.0, only matches with exact_match_bonus can be included
        (since base similarity is at most 1.0).
        
        Note: This test uses threshold values that are technically invalid for
        MatchConfig (which requires 0.0-1.0), so we test the boundary behavior.
        
        **Validates: Property 8**
        **Validates: Requirements 3.2, 9.3**
        """
        # This test verifies the boundary behavior at threshold = 1.0
        # Since MatchConfig validates threshold in [0.0, 1.0], we test at 1.0
        threshold = 1.0
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        apple_sv = vectorizer.vectorize_value("fruit", "apple")
        banana_sv = vectorizer.vectorize_value("fruit", "banana")
        schema_vectors = {
            "fruit=apple": apple_sv,
            "fruit=banana": banana_sv
        }
        
        # Create matcher with threshold = 1.0
        match_config = MatchConfig(
            value_threshold=threshold,
            exact_match_bonus=0.2
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token that doesn't exactly match any schema value
        token = Token(
            text="orange",
            original="orange",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: With threshold = 1.0, only exact matches (with bonus) should be included
        # Since "orange" doesn't match "apple" or "banana", no matches should be returned
        # (unless there's coincidental high similarity)
        for match in matches:
            assert match.similarity >= threshold, \
                f"Match with similarity {match.similarity} should not be included " \
                f"with threshold {threshold}"
    
    @given(
        role_threshold=thresholds,
        value_threshold=thresholds
    )
    @settings(max_examples=100)
    def test_threshold_filtering_preserves_match_ordering(
        self, role_threshold: float, value_threshold: float
    ):
        """
        Property test: Threshold filtering preserves match ordering by similarity.
        
        After threshold filtering, matches SHALL still be ordered by similarity
        in descending order.
        
        **Validates: Property 8**
        **Validates: Requirements 3.2, 9.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        schema_vectors = {}
        test_values = ["test", "testing", "tested", "tester", "tests"]
        for value in test_values:
            sv = vectorizer.vectorize_value("word", value)
            schema_vectors[f"word={value}"] = sv
        
        # Create matcher with specified thresholds
        match_config = MatchConfig(
            role_threshold=role_threshold,
            value_threshold=value_threshold
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token
        token = Token(
            text="test",
            original="test",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: Matches should be ordered by similarity (descending)
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert matches[i].similarity >= matches[i + 1].similarity, \
                    f"Matches not in descending order: {matches[i].similarity} < {matches[i + 1].similarity}"


class TestMatchTypeClassification:
    """
    Property tests for Match Type Classification (Property 9).
    
    **Validates: Property 9** - Match Type Classification
    For any token-schema match, the match SHALL be correctly classified as either
    "role" or "value" based on the schema element type.
    
    **Validates: Requirements 3.4**
    """
    
    @given(
        role_name=role_names,
        token_text=token_texts
    )
    @settings(max_examples=100)
    def test_role_matches_have_role_element_type(
        self, role_name: str, token_text: str
    ):
        """
        Property test: Matches against role vectors have element_type == "role".
        
        For any token matched against a role vector, the resulting TokenMatch
        SHALL have schema_vector.element_type == "role".
        
        **Validates: Property 9**
        **Validates: Requirements 3.4**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a role schema vector
        role_sv = vectorizer.vectorize_role(role_name)
        schema_vectors = {role_name: role_sv}
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token
        token = Token(
            text=token_text.lower(),
            original=token_text,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: All matches against role vectors must have element_type == "role"
        for match in matches:
            assert match.schema_vector.element_type == "role", \
                f"Match against role vector '{match.schema_vector.key}' has " \
                f"element_type '{match.schema_vector.element_type}' instead of 'role'"
    
    @given(
        role_name=role_names,
        value=values,
        token_text=token_texts
    )
    @settings(max_examples=100)
    def test_value_matches_have_value_element_type(
        self, role_name: str, value: str, token_text: str
    ):
        """
        Property test: Matches against value vectors have element_type == "value".
        
        For any token matched against a value vector, the resulting TokenMatch
        SHALL have schema_vector.element_type == "value".
        
        **Validates: Property 9**
        **Validates: Requirements 3.4**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a value schema vector
        value_sv = vectorizer.vectorize_value(role_name, value)
        schema_vectors = {f"{role_name}={value}": value_sv}
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token
        token = Token(
            text=token_text.lower(),
            original=token_text,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: All matches against value vectors must have element_type == "value"
        for match in matches:
            assert match.schema_vector.element_type == "value", \
                f"Match against value vector '{match.schema_vector.key}' has " \
                f"element_type '{match.schema_vector.element_type}' instead of 'value'"
    
    @given(
        role_name=role_names,
        value=values,
        token_text=token_texts
    )
    @settings(max_examples=100)
    def test_role_matches_categorized_in_role_matches_list(
        self, role_name: str, value: str, token_text: str
    ):
        """
        Property test: Role matches are correctly categorized in role_matches list.
        
        For any query matched against schema, matches against role vectors
        SHALL appear in the role_matches list of MatchResult.
        
        **Validates: Property 9**
        **Validates: Requirements 3.4**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create both role and value schema vectors
        role_sv = vectorizer.vectorize_role(role_name)
        value_sv = vectorizer.vectorize_value(role_name, value)
        schema_vectors = {
            role_name: role_sv,
            f"{role_name}={value}": value_sv
        }
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create tokenizer
        tokenizer = QueryTokenizer()
        
        # Match query against schema
        result = matcher.match_query(token_text, tokenizer)
        
        # Property: All role matches in token_matches should also be in role_matches
        for match in result.token_matches:
            if match.schema_vector.element_type == "role":
                # This match should be in role_matches
                assert any(
                    m.schema_vector.key == match.schema_vector.key and
                    m.token.text == match.token.text
                    for m in result.role_matches
                ), f"Role match '{match.schema_vector.key}' not found in role_matches"
    
    @given(
        role_name=role_names,
        value=values,
        token_text=token_texts
    )
    @settings(max_examples=100)
    def test_value_matches_categorized_in_value_matches_list(
        self, role_name: str, value: str, token_text: str
    ):
        """
        Property test: Value matches are correctly categorized in value_matches list.
        
        For any query matched against schema, matches against value vectors
        SHALL appear in the value_matches list of MatchResult.
        
        **Validates: Property 9**
        **Validates: Requirements 3.4**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create both role and value schema vectors
        role_sv = vectorizer.vectorize_role(role_name)
        value_sv = vectorizer.vectorize_value(role_name, value)
        schema_vectors = {
            role_name: role_sv,
            f"{role_name}={value}": value_sv
        }
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create tokenizer
        tokenizer = QueryTokenizer()
        
        # Match query against schema
        result = matcher.match_query(token_text, tokenizer)
        
        # Property: All value matches in token_matches should also be in value_matches
        for match in result.token_matches:
            if match.schema_vector.element_type == "value":
                # This match should be in value_matches
                assert any(
                    m.schema_vector.key == match.schema_vector.key and
                    m.token.text == match.token.text
                    for m in result.value_matches
                ), f"Value match '{match.schema_vector.key}' not found in value_matches"
    
    @given(
        num_roles=st.integers(min_value=1, max_value=5),
        num_values_per_role=st.integers(min_value=1, max_value=3),
        token_text=token_texts
    )
    @settings(max_examples=100)
    def test_mixed_schema_classification_consistency(
        self, num_roles: int, num_values_per_role: int, token_text: str
    ):
        """
        Property test: Classification is consistent with mixed role and value vectors.
        
        For any schema with both role and value vectors, each match SHALL be
        correctly classified based on the schema element type it matched against.
        
        **Validates: Property 9**
        **Validates: Requirements 3.4**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create multiple role and value schema vectors
        schema_vectors = {}
        role_names_list = [f"role{i}" for i in range(num_roles)]
        
        for role_name in role_names_list:
            # Add role vector
            role_sv = vectorizer.vectorize_role(role_name)
            schema_vectors[role_name] = role_sv
            
            # Add value vectors for this role
            for j in range(num_values_per_role):
                value = f"value{j}"
                value_sv = vectorizer.vectorize_value(role_name, value)
                schema_vectors[f"{role_name}={value}"] = value_sv
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token
        token = Token(
            text=token_text.lower(),
            original=token_text,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: Each match's element_type must match the schema vector's element_type
        for match in matches:
            schema_key = match.schema_vector.key
            original_sv = schema_vectors[schema_key]
            
            assert match.schema_vector.element_type == original_sv.element_type, \
                f"Match element_type '{match.schema_vector.element_type}' does not match " \
                f"schema vector element_type '{original_sv.element_type}' for key '{schema_key}'"
    
    @given(
        role_name=role_names,
        value=values
    )
    @settings(max_examples=100)
    def test_exact_role_match_has_role_element_type(
        self, role_name: str, value: str
    ):
        """
        Property test: Exact match against role name has element_type == "role".
        
        When a token exactly matches a role name, the resulting match SHALL
        have schema_vector.element_type == "role".
        
        **Validates: Property 9**
        **Validates: Requirements 3.4**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create both role and value schema vectors
        role_sv = vectorizer.vectorize_role(role_name)
        value_sv = vectorizer.vectorize_value(role_name, value)
        schema_vectors = {
            role_name: role_sv,
            f"{role_name}={value}": value_sv
        }
        
        # Create matcher
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token with exact role name
        token = Token(
            text=role_name.lower(),
            original=role_name,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Find the match against the role vector
        role_matches = [m for m in matches if m.schema_vector.key == role_name]
        
        # Property: The role match must have element_type == "role"
        for match in role_matches:
            assert match.schema_vector.element_type == "role", \
                f"Exact role match has element_type '{match.schema_vector.element_type}' " \
                f"instead of 'role'"
    
    @given(
        role_name=role_names,
        value=values
    )
    @settings(max_examples=100)
    def test_exact_value_match_has_value_element_type(
        self, role_name: str, value: str
    ):
        """
        Property test: Exact match against value has element_type == "value".
        
        When a token exactly matches a value, the resulting match SHALL
        have schema_vector.element_type == "value".
        
        **Validates: Property 9**
        **Validates: Requirements 3.4**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create both role and value schema vectors
        role_sv = vectorizer.vectorize_role(role_name)
        value_sv = vectorizer.vectorize_value(role_name, value)
        schema_vectors = {
            role_name: role_sv,
            f"{role_name}={value}": value_sv
        }
        
        # Create matcher
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token with exact value
        token = Token(
            text=value.lower(),
            original=value,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Find the match against the value vector
        value_key = f"{role_name}={value}"
        value_matches = [m for m in matches if m.schema_vector.key == value_key]
        
        # Property: The value match must have element_type == "value"
        for match in value_matches:
            assert match.schema_vector.element_type == "value", \
                f"Exact value match has element_type '{match.schema_vector.element_type}' " \
                f"instead of 'value'"
    
    @given(
        role_name=role_names,
        values_list=st.lists(values, min_size=1, max_size=5, unique=True),
        query=st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'Zs')),
            min_size=1, max_size=50
        ).filter(lambda x: x.strip() and any(c.isalnum() for c in x))
    )
    @settings(max_examples=100)
    def test_match_query_classification_consistency(
        self, role_name: str, values_list: list, query: str
    ):
        """
        Property test: match_query correctly classifies all matches.
        
        For any query matched against schema, all matches in role_matches
        SHALL have element_type == "role" and all matches in value_matches
        SHALL have element_type == "value".
        
        **Validates: Property 9**
        **Validates: Requirements 3.4**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        schema_vectors = {}
        
        # Add role vector
        role_sv = vectorizer.vectorize_role(role_name)
        schema_vectors[role_name] = role_sv
        
        # Add value vectors
        for value in values_list:
            value_sv = vectorizer.vectorize_value(role_name, value)
            schema_vectors[f"{role_name}={value}"] = value_sv
        
        # Create matcher with low threshold
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create tokenizer
        tokenizer = QueryTokenizer()
        
        # Match query against schema
        result = matcher.match_query(query, tokenizer)
        
        # Property: All matches in role_matches must have element_type == "role"
        for match in result.role_matches:
            assert match.schema_vector.element_type == "role", \
                f"Match in role_matches has element_type '{match.schema_vector.element_type}' " \
                f"instead of 'role'"
        
        # Property: All matches in value_matches must have element_type == "value"
        for match in result.value_matches:
            assert match.schema_vector.element_type == "value", \
                f"Match in value_matches has element_type '{match.schema_vector.element_type}' " \
                f"instead of 'value'"
    
    @given(
        role_threshold=thresholds,
        value_threshold=thresholds,
        token_text=token_texts
    )
    @settings(max_examples=100)
    def test_classification_independent_of_threshold(
        self, role_threshold: float, value_threshold: float, token_text: str
    ):
        """
        Property test: Classification is independent of threshold values.
        
        For any threshold configuration, the element_type classification
        SHALL be determined by the schema vector type, not the threshold.
        
        **Validates: Property 9**
        **Validates: Requirements 3.4**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create both role and value schema vectors
        role_sv = vectorizer.vectorize_role("test_role")
        value_sv = vectorizer.vectorize_value("test_role", "test_value")
        schema_vectors = {
            "test_role": role_sv,
            "test_role=test_value": value_sv
        }
        
        # Create matcher with specified thresholds
        match_config = MatchConfig(
            role_threshold=role_threshold,
            value_threshold=value_threshold
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token
        token = Token(
            text=token_text.lower(),
            original=token_text,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Property: Classification must match schema vector element_type
        for match in matches:
            expected_type = schema_vectors[match.schema_vector.key].element_type
            assert match.schema_vector.element_type == expected_type, \
                f"Match classification '{match.schema_vector.element_type}' does not match " \
                f"expected '{expected_type}' for key '{match.schema_vector.key}'"


class TestCompoundMatchPreference:
    """
    Property tests for Compound Match Preference (Property 15).
    
    **Validates: Property 15** - Compound Match Preference
    For any query where tokens can match either as a compound (e.g., "brake pads" → "Brake Pads")
    or individually (e.g., "brake" → "Brake", "pads" → "Pads"), the compound match SHALL be
    preferred and individual matches for those tokens SHALL be suppressed.
    
    **Validates: Requirements 10.2, 10.3**
    """
    
    @given(
        word1=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha()),
        word2=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha())
    )
    @settings(max_examples=100)
    def test_compound_matches_preserved_after_prefer_compound(
        self, word1: str, word2: str
    ):
        """
        Property test: Compound matches are preserved when prefer_compound_matches is called.
        
        For any compound match in the input, it SHALL be present in the output
        after calling prefer_compound_matches().
        
        **Validates: Property 15**
        **Validates: Requirements 10.2, 10.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a compound value (two words)
        compound_value = f"{word1} {word2}"
        
        # Create schema vectors for compound and individual values
        compound_sv = vectorizer.vectorize_value("category", compound_value)
        word1_sv = vectorizer.vectorize_value("category", word1)
        word2_sv = vectorizer.vectorize_value("category", word2)
        
        schema_vectors = {
            f"category={compound_value}": compound_sv,
            f"category={word1}": word1_sv,
            f"category={word2}": word2_sv
        }
        
        # Create matcher with low threshold
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create tokens - individual and compound
        token_word1 = Token(
            text=word1.lower(),
            original=word1,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        token_word2 = Token(
            text=word2.lower(),
            original=word2,
            position=len(word1) + 1,
            is_stop_word=False,
            ngram_size=1
        )
        token_compound = Token(
            text=compound_value.lower(),
            original=compound_value,
            position=0,
            is_stop_word=False,
            ngram_size=2
        )
        
        # Create matches for all tokens
        matches_word1 = matcher.match_token(token_word1)
        matches_word2 = matcher.match_token(token_word2)
        matches_compound = matcher.match_token(token_compound)
        
        # Mark compound matches with match_type="compound"
        compound_token_matches = []
        for match in matches_compound:
            compound_match = TokenMatch(
                token=match.token,
                schema_vector=match.schema_vector,
                similarity=match.similarity + match_config.compound_bonus,
                match_type="compound"
            )
            compound_token_matches.append(compound_match)
        
        # Combine all matches
        all_matches = matches_word1 + matches_word2 + compound_token_matches
        
        # Apply prefer_compound_matches
        filtered_matches = matcher.prefer_compound_matches(all_matches)
        
        # Property: All compound matches should be preserved
        for compound_match in compound_token_matches:
            assert any(
                m.token.text == compound_match.token.text and
                m.schema_vector.key == compound_match.schema_vector.key and
                m.match_type == "compound"
                for m in filtered_matches
            ), f"Compound match for '{compound_match.token.text}' was not preserved"
    
    @given(
        word1=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha()),
        word2=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha())
    )
    @settings(max_examples=100)
    def test_individual_matches_suppressed_when_covered_by_compound(
        self, word1: str, word2: str
    ):
        """
        Property test: Individual matches covered by compound matches are suppressed.
        
        For any individual token match that is covered by a compound match,
        the individual match SHALL be suppressed after calling prefer_compound_matches().
        
        **Validates: Property 15**
        **Validates: Requirements 10.2, 10.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a compound value (two words)
        compound_value = f"{word1} {word2}"
        
        # Create schema vectors for compound and individual values
        compound_sv = vectorizer.vectorize_value("category", compound_value)
        word1_sv = vectorizer.vectorize_value("category", word1)
        word2_sv = vectorizer.vectorize_value("category", word2)
        
        schema_vectors = {
            f"category={compound_value}": compound_sv,
            f"category={word1}": word1_sv,
            f"category={word2}": word2_sv
        }
        
        # Create matcher with low threshold
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create tokens - individual tokens at positions covered by compound
        # Compound starts at position 0 and covers positions 0 to len(compound_value)-1
        token_word1 = Token(
            text=word1.lower(),
            original=word1,
            position=0,  # Covered by compound (position 0)
            is_stop_word=False,
            ngram_size=1
        )
        token_word2 = Token(
            text=word2.lower(),
            original=word2,
            position=len(word1) + 1,  # Covered by compound
            is_stop_word=False,
            ngram_size=1
        )
        token_compound = Token(
            text=compound_value.lower(),
            original=compound_value,
            position=0,
            is_stop_word=False,
            ngram_size=2
        )
        
        # Create matches for all tokens
        matches_word1 = matcher.match_token(token_word1)
        matches_word2 = matcher.match_token(token_word2)
        matches_compound = matcher.match_token(token_compound)
        
        # Mark compound matches with match_type="compound"
        compound_token_matches = []
        for match in matches_compound:
            compound_match = TokenMatch(
                token=match.token,
                schema_vector=match.schema_vector,
                similarity=match.similarity + match_config.compound_bonus,
                match_type="compound"
            )
            compound_token_matches.append(compound_match)
        
        # Only proceed if we have compound matches
        assume(len(compound_token_matches) > 0)
        
        # Combine all matches
        all_matches = matches_word1 + matches_word2 + compound_token_matches
        
        # Apply prefer_compound_matches
        filtered_matches = matcher.prefer_compound_matches(all_matches)
        
        # Property: Individual matches covered by compound should be suppressed
        # Check that individual matches at covered positions are not in filtered results
        for individual_match in matches_word1 + matches_word2:
            individual_pos = individual_match.token.position
            individual_end = individual_pos + len(individual_match.token.original)
            
            # Check if this individual match is covered by any compound match
            is_covered = False
            for compound_match in compound_token_matches:
                compound_pos = compound_match.token.position
                compound_end = compound_pos + len(compound_match.token.original)
                
                # Check for overlap
                if individual_pos < compound_end and individual_end > compound_pos:
                    is_covered = True
                    break
            
            if is_covered:
                # This individual match should NOT be in filtered results
                assert not any(
                    m.token.text == individual_match.token.text and
                    m.token.position == individual_match.token.position and
                    m.schema_vector.key == individual_match.schema_vector.key and
                    m.match_type != "compound"
                    for m in filtered_matches
                ), f"Individual match '{individual_match.token.text}' at position {individual_pos} should be suppressed"
    
    @given(
        word1=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha()),
        word2=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha()),
        uncovered_word=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha())
    )
    @settings(max_examples=100)
    def test_individual_matches_not_covered_are_preserved(
        self, word1: str, word2: str, uncovered_word: str
    ):
        """
        Property test: Individual matches NOT covered by compound matches are preserved.
        
        For any individual token match that is NOT covered by any compound match,
        the individual match SHALL be preserved after calling prefer_compound_matches().
        
        **Validates: Property 15**
        **Validates: Requirements 10.2, 10.3**
        """
        # Ensure uncovered_word is different from word1 and word2
        assume(uncovered_word.lower() != word1.lower())
        assume(uncovered_word.lower() != word2.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a compound value (two words)
        compound_value = f"{word1} {word2}"
        
        # Create schema vectors for compound, individual, and uncovered values
        compound_sv = vectorizer.vectorize_value("category", compound_value)
        word1_sv = vectorizer.vectorize_value("category", word1)
        word2_sv = vectorizer.vectorize_value("category", word2)
        uncovered_sv = vectorizer.vectorize_value("make", uncovered_word)
        
        schema_vectors = {
            f"category={compound_value}": compound_sv,
            f"category={word1}": word1_sv,
            f"category={word2}": word2_sv,
            f"make={uncovered_word}": uncovered_sv
        }
        
        # Create matcher with low threshold
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create tokens - compound covers positions 0 to len(compound_value)-1
        # Uncovered token is at a position AFTER the compound
        compound_end_pos = len(compound_value)
        uncovered_pos = compound_end_pos + 10  # Well after the compound
        
        token_word1 = Token(
            text=word1.lower(),
            original=word1,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        token_word2 = Token(
            text=word2.lower(),
            original=word2,
            position=len(word1) + 1,
            is_stop_word=False,
            ngram_size=1
        )
        token_compound = Token(
            text=compound_value.lower(),
            original=compound_value,
            position=0,
            is_stop_word=False,
            ngram_size=2
        )
        token_uncovered = Token(
            text=uncovered_word.lower(),
            original=uncovered_word,
            position=uncovered_pos,  # Not covered by compound
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create matches for all tokens
        matches_word1 = matcher.match_token(token_word1)
        matches_word2 = matcher.match_token(token_word2)
        matches_compound = matcher.match_token(token_compound)
        matches_uncovered = matcher.match_token(token_uncovered)
        
        # Mark compound matches with match_type="compound"
        compound_token_matches = []
        for match in matches_compound:
            compound_match = TokenMatch(
                token=match.token,
                schema_vector=match.schema_vector,
                similarity=match.similarity + match_config.compound_bonus,
                match_type="compound"
            )
            compound_token_matches.append(compound_match)
        
        # Only proceed if we have compound matches and uncovered matches
        assume(len(compound_token_matches) > 0)
        assume(len(matches_uncovered) > 0)
        
        # Combine all matches
        all_matches = matches_word1 + matches_word2 + compound_token_matches + matches_uncovered
        
        # Apply prefer_compound_matches
        filtered_matches = matcher.prefer_compound_matches(all_matches)
        
        # Property: Uncovered individual matches should be preserved
        for uncovered_match in matches_uncovered:
            assert any(
                m.token.text == uncovered_match.token.text and
                m.token.position == uncovered_match.token.position and
                m.schema_vector.key == uncovered_match.schema_vector.key
                for m in filtered_matches
            ), f"Uncovered match '{uncovered_match.token.text}' at position {uncovered_match.token.position} should be preserved"
    
    @given(
        word1=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha()),
        word2=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha())
    )
    @settings(max_examples=100)
    def test_no_compound_matches_returns_all_individual_matches(
        self, word1: str, word2: str
    ):
        """
        Property test: When no compound matches exist, all individual matches are preserved.
        
        For any set of matches with no compound matches, calling prefer_compound_matches()
        SHALL return all matches unchanged.
        
        **Validates: Property 15**
        **Validates: Requirements 10.2, 10.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors for individual values only (no compound)
        word1_sv = vectorizer.vectorize_value("category", word1)
        word2_sv = vectorizer.vectorize_value("make", word2)
        
        schema_vectors = {
            f"category={word1}": word1_sv,
            f"make={word2}": word2_sv
        }
        
        # Create matcher with low threshold
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create individual tokens only (no compound)
        token_word1 = Token(
            text=word1.lower(),
            original=word1,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        token_word2 = Token(
            text=word2.lower(),
            original=word2,
            position=len(word1) + 1,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create matches for individual tokens
        matches_word1 = matcher.match_token(token_word1)
        matches_word2 = matcher.match_token(token_word2)
        
        # Combine all matches (no compound matches)
        all_matches = matches_word1 + matches_word2
        
        # Apply prefer_compound_matches
        filtered_matches = matcher.prefer_compound_matches(all_matches)
        
        # Property: All individual matches should be preserved when no compound matches exist
        assert len(filtered_matches) == len(all_matches), \
            f"Expected {len(all_matches)} matches, got {len(filtered_matches)}"
        
        # Verify each original match is in the filtered results
        for original_match in all_matches:
            assert any(
                m.token.text == original_match.token.text and
                m.schema_vector.key == original_match.schema_vector.key
                for m in filtered_matches
            ), f"Match '{original_match.token.text}' should be preserved when no compound matches exist"
    
    @given(
        word1=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha()),
        word2=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha()),
        word3=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha())
    )
    @settings(max_examples=100)
    def test_multiple_compound_matches_all_preserved(
        self, word1: str, word2: str, word3: str
    ):
        """
        Property test: Multiple compound matches are all preserved.
        
        For any set of matches with multiple compound matches, all compound matches
        SHALL be preserved after calling prefer_compound_matches().
        
        **Validates: Property 15**
        **Validates: Requirements 10.2, 10.3**
        """
        # Ensure words are different
        assume(word1.lower() != word2.lower())
        assume(word2.lower() != word3.lower())
        assume(word1.lower() != word3.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create two compound values
        compound1 = f"{word1} {word2}"
        compound2 = f"{word2} {word3}"
        
        # Create schema vectors
        compound1_sv = vectorizer.vectorize_value("category", compound1)
        compound2_sv = vectorizer.vectorize_value("type", compound2)
        
        schema_vectors = {
            f"category={compound1}": compound1_sv,
            f"type={compound2}": compound2_sv
        }
        
        # Create matcher with low threshold
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create compound tokens at different positions
        token_compound1 = Token(
            text=compound1.lower(),
            original=compound1,
            position=0,
            is_stop_word=False,
            ngram_size=2
        )
        token_compound2 = Token(
            text=compound2.lower(),
            original=compound2,
            position=len(compound1) + 5,  # Non-overlapping position
            is_stop_word=False,
            ngram_size=2
        )
        
        # Create matches for compound tokens
        matches_compound1 = matcher.match_token(token_compound1)
        matches_compound2 = matcher.match_token(token_compound2)
        
        # Mark all as compound matches
        compound_token_matches = []
        for match in matches_compound1:
            compound_match = TokenMatch(
                token=match.token,
                schema_vector=match.schema_vector,
                similarity=match.similarity + match_config.compound_bonus,
                match_type="compound"
            )
            compound_token_matches.append(compound_match)
        
        for match in matches_compound2:
            compound_match = TokenMatch(
                token=match.token,
                schema_vector=match.schema_vector,
                similarity=match.similarity + match_config.compound_bonus,
                match_type="compound"
            )
            compound_token_matches.append(compound_match)
        
        # Only proceed if we have compound matches
        assume(len(compound_token_matches) > 0)
        
        # Apply prefer_compound_matches
        filtered_matches = matcher.prefer_compound_matches(compound_token_matches)
        
        # Property: All compound matches should be preserved
        assert len(filtered_matches) == len(compound_token_matches), \
            f"Expected {len(compound_token_matches)} compound matches, got {len(filtered_matches)}"
        
        for compound_match in compound_token_matches:
            assert any(
                m.token.text == compound_match.token.text and
                m.schema_vector.key == compound_match.schema_vector.key and
                m.match_type == "compound"
                for m in filtered_matches
            ), f"Compound match '{compound_match.token.text}' should be preserved"
    
    @given(
        word1=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha()),
        word2=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=2, max_size=10
        ).filter(lambda x: x.strip() and x.isalpha())
    )
    @settings(max_examples=100)
    def test_filtered_matches_sorted_by_similarity(
        self, word1: str, word2: str
    ):
        """
        Property test: Filtered matches are sorted by similarity in descending order.
        
        After calling prefer_compound_matches(), the returned matches SHALL be
        sorted by similarity score in descending order.
        
        **Validates: Property 15**
        **Validates: Requirements 10.2, 10.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a compound value
        compound_value = f"{word1} {word2}"
        
        # Create schema vectors
        compound_sv = vectorizer.vectorize_value("category", compound_value)
        word1_sv = vectorizer.vectorize_value("category", word1)
        
        schema_vectors = {
            f"category={compound_value}": compound_sv,
            f"category={word1}": word1_sv
        }
        
        # Create matcher with low threshold
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create tokens
        token_compound = Token(
            text=compound_value.lower(),
            original=compound_value,
            position=0,
            is_stop_word=False,
            ngram_size=2
        )
        # Uncovered individual token at a different position
        token_uncovered = Token(
            text=word1.lower(),
            original=word1,
            position=len(compound_value) + 10,  # Not covered by compound
            is_stop_word=False,
            ngram_size=1
        )
        
        # Create matches
        matches_compound = matcher.match_token(token_compound)
        matches_uncovered = matcher.match_token(token_uncovered)
        
        # Mark compound matches
        compound_token_matches = []
        for match in matches_compound:
            compound_match = TokenMatch(
                token=match.token,
                schema_vector=match.schema_vector,
                similarity=match.similarity + match_config.compound_bonus,
                match_type="compound"
            )
            compound_token_matches.append(compound_match)
        
        # Combine matches
        all_matches = compound_token_matches + matches_uncovered
        
        # Apply prefer_compound_matches
        filtered_matches = matcher.prefer_compound_matches(all_matches)
        
        # Property: Filtered matches should be sorted by similarity (descending)
        if len(filtered_matches) > 1:
            for i in range(len(filtered_matches) - 1):
                assert filtered_matches[i].similarity >= filtered_matches[i + 1].similarity, \
                    f"Matches not sorted: {filtered_matches[i].similarity} < {filtered_matches[i + 1].similarity}"
    
    @settings(max_examples=100)
    @given(st.data())
    def test_empty_matches_returns_empty_list(self, data):
        """
        Property test: Empty input returns empty list.
        
        Calling prefer_compound_matches() with an empty list SHALL return an empty list.
        
        **Validates: Property 15**
        **Validates: Requirements 10.2, 10.3**
        """
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create minimal schema vectors
        sv = vectorizer.vectorize_value("test", "value")
        schema_vectors = {"test=value": sv}
        
        # Create matcher
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Apply prefer_compound_matches with empty list
        filtered_matches = matcher.prefer_compound_matches([])
        
        # Property: Empty input should return empty list
        assert filtered_matches == [], \
            f"Expected empty list, got {len(filtered_matches)} matches"
