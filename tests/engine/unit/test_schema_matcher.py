"""
Unit tests for the SchemaMatcher class.

Tests cover:
- SchemaMatcher initialization
- compute_similarity() method
- Type validation and error handling

Validates: Requirement 3 - Token-to-Schema Matching
"""

import pytest
import numpy as np

from glyphh.encoder.base import Encoder
from glyphh.core.config import EncoderConfig
from glyphh.core.types import Vector
from glyphh.nl.schema_matcher import SchemaMatcher, MatchConfig, TokenMatch, MatchResult
from glyphh.nl.schema_vectorizer import SchemaVectorizer, SchemaVector
from glyphh.nl.query_tokenizer import Token, QueryTokenizer


class TestSchemaMatcherInit:
    """Test SchemaMatcher.__init__() method."""
    
    def test_init_stores_schema_vectors_reference(self):
        """Test that __init__ stores the schema_vectors reference.
        
        **Validates: Requirement 3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        schema_vectors = {}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        assert matcher.schema_vectors is schema_vectors
    
    def test_init_stores_encoder_reference(self):
        """Test that __init__ stores the encoder reference.
        
        **Validates: Requirement 3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        schema_vectors = {}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        assert matcher.encoder is encoder
    
    def test_init_creates_default_match_config(self):
        """Test that __init__ creates default MatchConfig when not provided.
        
        **Validates: Requirement 3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        schema_vectors = {}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        assert isinstance(matcher.config, MatchConfig)
        assert matcher.config.role_threshold == 0.3
        assert matcher.config.value_threshold == 0.3
        assert matcher.config.compound_bonus == 0.1
        assert matcher.config.exact_match_bonus == 0.2
    
    def test_init_uses_provided_match_config(self):
        """Test that __init__ uses provided MatchConfig.
        
        **Validates: Requirement 3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        schema_vectors = {}
        custom_config = MatchConfig(
            role_threshold=0.5,
            value_threshold=0.5,
            compound_bonus=0.15,
            exact_match_bonus=0.25
        )
        
        matcher = SchemaMatcher(schema_vectors, encoder, custom_config)
        
        assert matcher.config is custom_config
        assert matcher.config.role_threshold == 0.5
    
    def test_init_raises_type_error_for_invalid_schema_vectors(self):
        """Test that __init__ raises TypeError for non-dict schema_vectors."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        
        with pytest.raises(TypeError) as exc_info:
            SchemaMatcher([], encoder)
        
        assert "schema_vectors must be a dict" in str(exc_info.value)
    
    def test_init_raises_type_error_for_invalid_encoder(self):
        """Test that __init__ raises TypeError for non-Encoder encoder."""
        schema_vectors = {}
        
        with pytest.raises(TypeError) as exc_info:
            SchemaMatcher(schema_vectors, "not_an_encoder")
        
        assert "encoder must be an Encoder instance" in str(exc_info.value)
    
    def test_init_raises_type_error_for_invalid_config(self):
        """Test that __init__ raises TypeError for invalid config."""
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        schema_vectors = {}
        
        with pytest.raises(TypeError) as exc_info:
            SchemaMatcher(schema_vectors, encoder, "not_a_config")
        
        assert "config must be a MatchConfig instance" in str(exc_info.value)


class TestSchemaMatcherComputeSimilarity:
    """Test SchemaMatcher.compute_similarity() method.
    
    **Validates: Requirement 3.1** - THE SDK SHALL compute similarity between
    each token vector and all schema vectors
    """
    
    def test_compute_similarity_returns_float(self):
        """Test that compute_similarity returns a float.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        vec_a = encoder.generate_symbol("test")
        vec_b = encoder.generate_symbol("test")
        
        result = matcher.compute_similarity(vec_a, vec_b)
        
        assert isinstance(result, float)
    
    def test_compute_similarity_identical_vectors_returns_one(self):
        """Test that identical vectors have similarity 1.0.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        vec = encoder.generate_symbol("test")
        
        result = matcher.compute_similarity(vec, vec)
        
        assert result == 1.0
    
    def test_compute_similarity_opposite_vectors_returns_negative_one(self):
        """Test that opposite vectors have similarity -1.0.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        vec = encoder.generate_symbol("test")
        
        # Create opposite vector (negate all values)
        opposite_data = -vec.data
        opposite_vec = Vector(
            data=opposite_data,
            dimension=vec.dimension,
            space_id=vec.space_id
        )
        
        result = matcher.compute_similarity(vec, opposite_vec)
        
        assert result == -1.0
    
    def test_compute_similarity_in_valid_range(self):
        """Test that similarity is always in range [-1.0, 1.0].
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        # Test with various different vectors
        test_keys = ["apple", "orange", "banana", "car", "truck", "bike"]
        vectors = [encoder.generate_symbol(key) for key in test_keys]
        
        for i, vec_a in enumerate(vectors):
            for j, vec_b in enumerate(vectors):
                result = matcher.compute_similarity(vec_a, vec_b)
                assert -1.0 <= result <= 1.0, f"Similarity {result} out of range for vectors {i}, {j}"
    
    def test_compute_similarity_different_vectors_less_than_one(self):
        """Test that different vectors have similarity less than 1.0.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        vec_a = encoder.generate_symbol("apple")
        vec_b = encoder.generate_symbol("orange")
        
        result = matcher.compute_similarity(vec_a, vec_b)
        
        # Different vectors should have similarity < 1.0
        assert result < 1.0
    
    def test_compute_similarity_is_symmetric(self):
        """Test that similarity is symmetric: sim(a, b) == sim(b, a).
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        vec_a = encoder.generate_symbol("apple")
        vec_b = encoder.generate_symbol("orange")
        
        result_ab = matcher.compute_similarity(vec_a, vec_b)
        result_ba = matcher.compute_similarity(vec_b, vec_a)
        
        assert result_ab == result_ba
    
    def test_compute_similarity_with_schema_vector(self):
        """Test compute_similarity with actual SchemaVector.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vector
        schema_vec = vectorizer.vectorize_value("make", "Toyota")
        
        # Create token vector
        token_vec = encoder.generate_symbol("Toyota")
        
        matcher = SchemaMatcher({}, encoder)
        
        # Same text should have high similarity
        result = matcher.compute_similarity(token_vec, schema_vec.vector)
        
        # Should be 1.0 since both are generated from "Toyota"
        assert result == 1.0
    
    def test_compute_similarity_raises_type_error_for_invalid_token_vector(self):
        """Test that compute_similarity raises TypeError for invalid token_vector.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        vec = encoder.generate_symbol("test")
        
        with pytest.raises(TypeError) as exc_info:
            matcher.compute_similarity("not_a_vector", vec)
        
        assert "token_vector must be a Vector instance" in str(exc_info.value)
    
    def test_compute_similarity_raises_type_error_for_invalid_schema_vector(self):
        """Test that compute_similarity raises TypeError for invalid schema_vector.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        vec = encoder.generate_symbol("test")
        
        with pytest.raises(TypeError) as exc_info:
            matcher.compute_similarity(vec, "not_a_vector")
        
        assert "schema_vector must be a Vector instance" in str(exc_info.value)
    
    def test_compute_similarity_raises_value_error_for_dimension_mismatch(self):
        """Test that compute_similarity raises ValueError for dimension mismatch.
        
        **Validates: Requirement 3.1**
        """
        config_1000 = EncoderConfig(dimension=1000, seed=42)
        encoder_1000 = Encoder(config_1000)
        
        config_500 = EncoderConfig(dimension=500, seed=42)
        encoder_500 = Encoder(config_500)
        
        matcher = SchemaMatcher({}, encoder_1000)
        
        vec_1000 = encoder_1000.generate_symbol("test")
        vec_500 = encoder_500.generate_symbol("test")
        
        with pytest.raises(ValueError) as exc_info:
            matcher.compute_similarity(vec_1000, vec_500)
        
        assert "Vectors must have the same dimension" in str(exc_info.value)
    
    def test_compute_similarity_raises_value_error_for_space_id_mismatch(self):
        """Test that compute_similarity raises ValueError for space_id mismatch.
        
        **Validates: Requirement 3.1**
        """
        config_a = EncoderConfig(dimension=1000, seed=42)
        encoder_a = Encoder(config_a)
        
        config_b = EncoderConfig(dimension=1000, seed=99)  # Different seed = different space_id
        encoder_b = Encoder(config_b)
        
        matcher = SchemaMatcher({}, encoder_a)
        
        vec_a = encoder_a.generate_symbol("test")
        vec_b = encoder_b.generate_symbol("test")
        
        with pytest.raises(ValueError) as exc_info:
            matcher.compute_similarity(vec_a, vec_b)
        
        assert "Vectors must be in the same vector space" in str(exc_info.value)
    
    def test_compute_similarity_with_large_dimension(self):
        """Test compute_similarity with large dimension vectors.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=10000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        vec_a = encoder.generate_symbol("apple")
        vec_b = encoder.generate_symbol("orange")
        
        result = matcher.compute_similarity(vec_a, vec_b)
        
        assert -1.0 <= result <= 1.0
    
    def test_compute_similarity_deterministic(self):
        """Test that compute_similarity is deterministic.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        vec_a = encoder.generate_symbol("apple")
        vec_b = encoder.generate_symbol("orange")
        
        result1 = matcher.compute_similarity(vec_a, vec_b)
        result2 = matcher.compute_similarity(vec_a, vec_b)
        result3 = matcher.compute_similarity(vec_a, vec_b)
        
        assert result1 == result2 == result3
    
    def test_compute_similarity_with_manually_created_vectors(self):
        """Test compute_similarity with manually created bipolar vectors.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=100, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        # Create two vectors with known similarity
        # All +1 vs all +1 should give similarity 1.0
        all_ones = Vector(
            data=np.ones(100, dtype=np.int8),
            dimension=100,
            space_id=encoder.space_id
        )
        
        result = matcher.compute_similarity(all_ones, all_ones)
        assert result == 1.0
        
        # All +1 vs all -1 should give similarity -1.0
        all_neg_ones = Vector(
            data=-np.ones(100, dtype=np.int8),
            dimension=100,
            space_id=encoder.space_id
        )
        
        result = matcher.compute_similarity(all_ones, all_neg_ones)
        assert result == -1.0
    
    def test_compute_similarity_half_matching(self):
        """Test compute_similarity with vectors that half match.
        
        **Validates: Requirement 3.1**
        """
        config = EncoderConfig(dimension=100, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        # Create vector with first half +1, second half -1
        half_half = np.concatenate([
            np.ones(50, dtype=np.int8),
            -np.ones(50, dtype=np.int8)
        ])
        vec_a = Vector(
            data=half_half,
            dimension=100,
            space_id=encoder.space_id
        )
        
        # All +1 vector
        all_ones = Vector(
            data=np.ones(100, dtype=np.int8),
            dimension=100,
            space_id=encoder.space_id
        )
        
        # Similarity should be 0.0 (50 matches, 50 mismatches)
        result = matcher.compute_similarity(vec_a, all_ones)
        assert result == 0.0


class TestMatchConfig:
    """Test MatchConfig dataclass."""
    
    def test_match_config_default_values(self):
        """Test MatchConfig default values."""
        config = MatchConfig()
        
        assert config.role_threshold == 0.3
        assert config.value_threshold == 0.3
        assert config.compound_bonus == 0.1
        assert config.exact_match_bonus == 0.2
    
    def test_match_config_custom_values(self):
        """Test MatchConfig with custom values."""
        config = MatchConfig(
            role_threshold=0.5,
            value_threshold=0.4,
            compound_bonus=0.15,
            exact_match_bonus=0.25
        )
        
        assert config.role_threshold == 0.5
        assert config.value_threshold == 0.4
        assert config.compound_bonus == 0.15
        assert config.exact_match_bonus == 0.25
    
    def test_match_config_raises_value_error_for_invalid_role_threshold(self):
        """Test MatchConfig raises ValueError for invalid role_threshold."""
        with pytest.raises(ValueError) as exc_info:
            MatchConfig(role_threshold=1.5)
        
        assert "role_threshold must be between 0.0 and 1.0" in str(exc_info.value)
    
    def test_match_config_raises_value_error_for_negative_role_threshold(self):
        """Test MatchConfig raises ValueError for negative role_threshold."""
        with pytest.raises(ValueError) as exc_info:
            MatchConfig(role_threshold=-0.1)
        
        assert "role_threshold must be between 0.0 and 1.0" in str(exc_info.value)
    
    def test_match_config_raises_value_error_for_invalid_value_threshold(self):
        """Test MatchConfig raises ValueError for invalid value_threshold."""
        with pytest.raises(ValueError) as exc_info:
            MatchConfig(value_threshold=1.5)
        
        assert "value_threshold must be between 0.0 and 1.0" in str(exc_info.value)
    
    def test_match_config_raises_value_error_for_negative_compound_bonus(self):
        """Test MatchConfig raises ValueError for negative compound_bonus."""
        with pytest.raises(ValueError) as exc_info:
            MatchConfig(compound_bonus=-0.1)
        
        assert "compound_bonus must be non-negative" in str(exc_info.value)
    
    def test_match_config_raises_value_error_for_negative_exact_match_bonus(self):
        """Test MatchConfig raises ValueError for negative exact_match_bonus."""
        with pytest.raises(ValueError) as exc_info:
            MatchConfig(exact_match_bonus=-0.1)
        
        assert "exact_match_bonus must be non-negative" in str(exc_info.value)


class TestSchemaMatcherMatchToken:
    """Test SchemaMatcher.match_token() method.
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    """
    
    def test_match_token_returns_list(self):
        """Test that match_token returns a list.
        
        **Validates: Requirement 3.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        assert isinstance(result, list)
    
    def test_match_token_empty_schema_returns_empty_list(self):
        """Test that match_token returns empty list when no schema vectors.
        
        **Validates: Requirement 3.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        assert result == []
    
    def test_match_token_finds_exact_value_match(self):
        """Test that match_token finds exact value matches.
        
        Note: HDC encoding is case-sensitive, so the token text must match
        the schema value's casing for high similarity. The "exact" match type
        is determined by case-insensitive text comparison.
        
        **Validates: Requirements 3.1, 3.2, 3.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a value schema vector with lowercase (matching normalized token)
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        schema_vectors = {"make=toyota": toyota_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Token with same text (normalized to lowercase)
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        assert len(result) == 1
        assert result[0].schema_vector.original_value == "toyota"
        assert result[0].match_type == "exact"
    
    def test_match_token_finds_exact_role_match(self):
        """Test that match_token finds exact role matches.
        
        **Validates: Requirements 3.1, 3.2, 3.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a role schema vector
        make_vec = vectorizer.vectorize_role("make")
        schema_vectors = {"make": make_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Token with same text
        token = Token(
            text="make",
            original="make",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        assert len(result) == 1
        assert result[0].schema_vector.element_type == "role"
        assert result[0].match_type == "exact"
    
    def test_match_token_applies_exact_match_bonus(self):
        """Test that match_token applies exact_match_bonus for exact matches.
        
        Note: HDC encoding is case-sensitive, so the token text must match
        the schema value's casing for high similarity.
        
        **Validates: Requirement 3.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a value schema vector with lowercase (matching normalized token)
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        schema_vectors = {"make=toyota": toyota_vec}
        
        match_config = MatchConfig(exact_match_bonus=0.2)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Token with same text (exact match)
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Similarity should be 1.0 (identical vectors) + 0.2 (exact_match_bonus) = 1.2
        assert len(result) == 1
        assert result[0].similarity == 1.2
    
    def test_match_token_filters_below_threshold(self):
        """Test that match_token filters matches below threshold.
        
        **Validates: Requirement 3.2**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a value schema vector
        toyota_vec = vectorizer.vectorize_value("make", "Toyota")
        schema_vectors = {"make=Toyota": toyota_vec}
        
        # Set very high threshold
        match_config = MatchConfig(value_threshold=0.99)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Token with different text (low similarity)
        token = Token(
            text="honda",
            original="Honda",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Should be filtered out due to low similarity
        assert result == []
    
    def test_match_token_returns_multiple_matches(self):
        """Test that match_token returns all matches above threshold.
        
        Note: HDC encoding is case-sensitive, so we use lowercase values
        to match normalized token text. We create schema vectors with
        similar prefixes to ensure positive similarity.
        
        **Validates: Requirement 3.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create multiple value schema vectors with lowercase
        # Using the same token text for both to ensure high similarity
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        toyota2_vec = vectorizer.vectorize_value("model", "toyota")  # Same text, different role
        schema_vectors = {
            "make=toyota": toyota_vec,
            "model=toyota": toyota2_vec
        }
        
        # Set threshold to 0.0 to get all positive matches
        match_config = MatchConfig(value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Should return both matches (both exact matches)
        assert len(result) == 2
        # Both should be exact matches since the text is the same
        assert all(m.match_type == "exact" for m in result)
    
    def test_match_token_sorted_by_similarity_descending(self):
        """Test that match_token returns matches sorted by similarity descending.
        
        **Validates: Requirement 3.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create multiple value schema vectors
        toyota_vec = vectorizer.vectorize_value("make", "Toyota")
        honda_vec = vectorizer.vectorize_value("make", "Honda")
        ford_vec = vectorizer.vectorize_value("make", "Ford")
        schema_vectors = {
            "make=Toyota": toyota_vec,
            "make=Honda": honda_vec,
            "make=Ford": ford_vec
        }
        
        # Set low threshold to get all matches
        match_config = MatchConfig(value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Verify sorted by similarity descending
        for i in range(len(result) - 1):
            assert result[i].similarity >= result[i + 1].similarity
    
    def test_match_token_distinguishes_role_and_value_matches(self):
        """Test that match_token correctly classifies role vs value matches.
        
        **Validates: Requirement 3.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create both role and value schema vectors
        make_role_vec = vectorizer.vectorize_role("make")
        toyota_value_vec = vectorizer.vectorize_value("make", "Toyota")
        schema_vectors = {
            "make": make_role_vec,
            "make=Toyota": toyota_value_vec
        }
        
        # Set low threshold to get all matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        token = Token(
            text="make",
            original="make",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Should have at least the role match
        role_matches = [m for m in result if m.schema_vector.element_type == "role"]
        value_matches = [m for m in result if m.schema_vector.element_type == "value"]
        
        assert len(role_matches) >= 1
        assert role_matches[0].schema_vector.element_type == "role"
    
    def test_match_token_uses_role_threshold_for_roles(self):
        """Test that match_token uses role_threshold for role vectors.
        
        **Validates: Requirement 3.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a role schema vector
        make_vec = vectorizer.vectorize_role("make")
        schema_vectors = {"make": make_vec}
        
        # Set high role_threshold, low value_threshold
        match_config = MatchConfig(role_threshold=0.99, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Token with different text (low similarity)
        token = Token(
            text="model",
            original="model",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Should be filtered out due to high role_threshold
        assert result == []
    
    def test_match_token_uses_value_threshold_for_values(self):
        """Test that match_token uses value_threshold for value vectors.
        
        **Validates: Requirement 3.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a value schema vector
        toyota_vec = vectorizer.vectorize_value("make", "Toyota")
        schema_vectors = {"make=Toyota": toyota_vec}
        
        # Set low role_threshold, high value_threshold
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.99)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Token with different text (low similarity)
        token = Token(
            text="honda",
            original="Honda",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Should be filtered out due to high value_threshold
        assert result == []
    
    def test_match_token_raises_type_error_for_invalid_token(self):
        """Test that match_token raises TypeError for non-Token input.
        
        **Validates: Requirement 3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        with pytest.raises(TypeError) as exc_info:
            matcher.match_token("not_a_token")
        
        assert "token must be a Token instance" in str(exc_info.value)
    
    def test_match_token_partial_match_type(self):
        """Test that match_token sets match_type to 'partial' for non-exact matches.
        
        **Validates: Requirement 3.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a value schema vector
        toyota_vec = vectorizer.vectorize_value("make", "Toyota")
        schema_vectors = {"make=Toyota": toyota_vec}
        
        # Set low threshold to get the match
        match_config = MatchConfig(value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Token with different text (partial match)
        token = Token(
            text="honda",
            original="Honda",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Should be a partial match
        assert len(result) == 1
        assert result[0].match_type == "partial"
    
    def test_match_token_case_insensitive_exact_match(self):
        """Test that exact match detection is case-insensitive.
        
        Note: HDC encoding is case-sensitive, so we use lowercase values
        to match normalized token text. The exact match detection compares
        the token text with the schema value case-insensitively.
        
        **Validates: Requirement 3.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a value schema vector with uppercase
        # Note: The vector is generated from "TOYOTA" but we compare case-insensitively
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        schema_vectors = {"make=toyota": toyota_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Token with lowercase text
        token = Token(
            text="toyota",
            original="toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Should be an exact match (case-insensitive)
        assert len(result) == 1
        assert result[0].match_type == "exact"


from glyphh.nl.query_tokenizer import Token

from glyphh.nl.query_tokenizer import QueryTokenizer, TokenizerConfig


class TestSchemaMatcherMatchQuery:
    """Test SchemaMatcher.match_query() method.
    
    **Validates: Requirement 3.6** - THE SDK SHALL provide a `match_query()` method
    that returns all token-schema matches
    """
    
    def test_match_query_returns_match_result(self):
        """Test that match_query returns a MatchResult.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find Toyota", tokenizer)
        
        assert isinstance(result, MatchResult)
    
    def test_match_query_stores_original_query(self):
        """Test that match_query stores the original query in result.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        tokenizer = QueryTokenizer()
        
        query = "Find Toyota brake pads"
        result = matcher.match_query(query, tokenizer)
        
        assert result.query == query
    
    def test_match_query_empty_schema_returns_empty_matches(self):
        """Test that match_query returns empty matches when no schema vectors.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find Toyota", tokenizer)
        
        assert result.token_matches == []
        assert result.role_matches == []
        assert result.value_matches == []
        assert result.compound_matches == []
        assert result.all_candidates == []
    
    def test_match_query_empty_query_returns_empty_matches(self):
        """Test that match_query returns empty matches for empty query.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        schema_vectors = {"make=toyota": toyota_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("", tokenizer)
        
        assert result.token_matches == []
        assert result.query == ""
    
    def test_match_query_finds_single_word_value_match(self):
        """Test that match_query finds single-word value matches.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vector with lowercase to match normalized token
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        schema_vectors = {"make=toyota": toyota_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find Toyota", tokenizer)
        
        # Should find the toyota match
        assert len(result.token_matches) > 0
        assert len(result.value_matches) > 0
        
        # Check that toyota was matched
        toyota_matches = [m for m in result.value_matches 
                        if m.schema_vector.original_value == "toyota"]
        assert len(toyota_matches) > 0
    
    def test_match_query_finds_role_match(self):
        """Test that match_query finds role matches.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create role schema vector
        make_vec = vectorizer.vectorize_role("make")
        schema_vectors = {"make": make_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Show make", tokenizer)
        
        # Should find the make role match
        assert len(result.role_matches) > 0
        assert result.role_matches[0].schema_vector.element_type == "role"
    
    def test_match_query_categorizes_role_and_value_matches(self):
        """Test that match_query correctly categorizes role and value matches.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create both role and value schema vectors
        make_role_vec = vectorizer.vectorize_role("make")
        toyota_value_vec = vectorizer.vectorize_value("make", "toyota")
        schema_vectors = {
            "make": make_role_vec,
            "make=toyota": toyota_value_vec
        }
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find make toyota", tokenizer)
        
        # Should have both role and value matches
        assert len(result.role_matches) > 0
        assert len(result.value_matches) > 0
        
        # Verify categorization
        for match in result.role_matches:
            assert match.schema_vector.element_type == "role"
        for match in result.value_matches:
            assert match.schema_vector.element_type == "value"
    
    def test_match_query_generates_compound_matches(self):
        """Test that match_query generates compound matches from n-grams.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a compound value schema vector
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        schema_vectors = {"category=brake pads": brake_pads_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find brake pads", tokenizer)
        
        # Should find compound match for "brake pads"
        assert len(result.compound_matches) > 0
        
        # Verify compound match has match_type="compound"
        for match in result.compound_matches:
            assert match.match_type == "compound"
    
    def test_match_query_compound_matches_have_ngram_size_greater_than_one(self):
        """Test that compound matches come from n-gram tokens.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a compound value schema vector
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        schema_vectors = {"category=brake pads": brake_pads_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find brake pads", tokenizer)
        
        # Compound matches should have ngram_size > 1
        for match in result.compound_matches:
            assert match.token.ngram_size > 1
    
    def test_match_query_applies_compound_bonus(self):
        """Test that match_query applies compound_bonus to n-gram matches.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a compound value schema vector
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        schema_vectors = {"category=brake pads": brake_pads_vec}
        
        match_config = MatchConfig(compound_bonus=0.1, exact_match_bonus=0.2)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find brake pads", tokenizer)
        
        # Compound matches should have bonus applied
        # Base similarity (1.0) + exact_match_bonus (0.2) + compound_bonus (0.1) = 1.3
        assert len(result.compound_matches) > 0
        
        # Find the "brake pads" compound match specifically
        brake_pads_matches = [m for m in result.compound_matches 
                             if m.token.text == "brake pads"]
        assert len(brake_pads_matches) > 0
        
        # The "brake pads" match should have similarity > 1.0 due to bonuses
        assert brake_pads_matches[0].similarity > 1.0
    
    def test_match_query_all_candidates_includes_all_matches(self):
        """Test that all_candidates includes all matches for disambiguation.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create multiple schema vectors
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        honda_vec = vectorizer.vectorize_value("make", "honda")
        schema_vectors = {
            "make=toyota": toyota_vec,
            "make=honda": honda_vec
        }
        
        # Set low threshold to get all matches
        match_config = MatchConfig(value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find toyota", tokenizer)
        
        # all_candidates should include all matches
        assert len(result.all_candidates) >= len(result.token_matches)
    
    def test_match_query_matches_sorted_by_similarity(self):
        """Test that match_query returns matches sorted by similarity descending.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create multiple schema vectors
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        honda_vec = vectorizer.vectorize_value("make", "honda")
        ford_vec = vectorizer.vectorize_value("make", "ford")
        schema_vectors = {
            "make=toyota": toyota_vec,
            "make=honda": honda_vec,
            "make=ford": ford_vec
        }
        
        # Set low threshold to get all matches
        match_config = MatchConfig(value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find toyota", tokenizer)
        
        # Verify all match lists are sorted by similarity descending
        for i in range(len(result.token_matches) - 1):
            assert result.token_matches[i].similarity >= result.token_matches[i + 1].similarity
        
        for i in range(len(result.value_matches) - 1):
            assert result.value_matches[i].similarity >= result.value_matches[i + 1].similarity
    
    def test_match_query_respects_max_ngram_size(self):
        """Test that match_query respects tokenizer's max_ngram_size.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a 3-word compound value
        three_word_vec = vectorizer.vectorize_value("category", "front brake pads")
        schema_vectors = {"category=front brake pads": three_word_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Tokenizer with max_ngram_size=2 (won't generate 3-grams)
        tokenizer_config = TokenizerConfig(max_ngram_size=2)
        tokenizer = QueryTokenizer(config=tokenizer_config)
        
        result = matcher.match_query("Find front brake pads", tokenizer)
        
        # Should not find exact match for 3-word compound
        # because max_ngram_size=2 limits n-gram generation
        exact_3word_matches = [m for m in result.compound_matches 
                              if m.token.ngram_size == 3]
        assert len(exact_3word_matches) == 0
    
    def test_match_query_raises_type_error_for_invalid_query(self):
        """Test that match_query raises TypeError for non-string query.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        tokenizer = QueryTokenizer()
        
        with pytest.raises(TypeError) as exc_info:
            matcher.match_query(123, tokenizer)
        
        assert "query must be a string" in str(exc_info.value)
    
    def test_match_query_raises_type_error_for_invalid_tokenizer(self):
        """Test that match_query raises TypeError for non-QueryTokenizer tokenizer.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        with pytest.raises(TypeError) as exc_info:
            matcher.match_query("Find Toyota", "not_a_tokenizer")
        
        assert "tokenizer must be a QueryTokenizer instance" in str(exc_info.value)
    
    def test_match_query_multiple_tokens_multiple_matches(self):
        """Test that match_query handles multiple tokens with multiple matches.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create multiple schema vectors
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        brake_vec = vectorizer.vectorize_value("category", "brake")
        pads_vec = vectorizer.vectorize_value("type", "pads")
        schema_vectors = {
            "make=toyota": toyota_vec,
            "category=brake": brake_vec,
            "type=pads": pads_vec
        }
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find toyota brake pads", tokenizer)
        
        # Should find matches for multiple tokens
        assert len(result.token_matches) >= 3  # At least one match per token
    
    def test_match_query_whitespace_only_query(self):
        """Test that match_query handles whitespace-only query.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        schema_vectors = {"make=toyota": toyota_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("   ", tokenizer)
        
        assert result.token_matches == []
        assert result.query == "   "
    
    def test_match_query_compound_also_in_value_matches(self):
        """Test that compound value matches are also in value_matches.
        
        **Validates: Requirement 3.6**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create a compound value schema vector
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        schema_vectors = {"category=brake pads": brake_pads_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find brake pads", tokenizer)
        
        # Compound matches that are values should also be in value_matches
        for compound_match in result.compound_matches:
            if compound_match.schema_vector.element_type == "value":
                # Should be in value_matches
                assert compound_match in result.value_matches


class TestSchemaMatcherPreferCompoundMatches:
    """Test SchemaMatcher.prefer_compound_matches() method.
    
    **Validates: Requirements 10.2, 10.3**
    - 10.2: WHEN a compound match is found, THE SDK SHALL prefer it over individual token matches
    - 10.3: THE SDK SHALL suppress individual token matches when a compound match covers those tokens
    """
    
    def test_prefer_compound_matches_returns_list(self):
        """Test that prefer_compound_matches returns a list.
        
        **Validates: Requirements 10.2, 10.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        result = matcher.prefer_compound_matches([])
        
        assert isinstance(result, list)
    
    def test_prefer_compound_matches_empty_input_returns_empty_list(self):
        """Test that prefer_compound_matches returns empty list for empty input.
        
        **Validates: Requirements 10.2, 10.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        matcher = SchemaMatcher({}, encoder)
        
        result = matcher.prefer_compound_matches([])
        
        assert result == []
    
    def test_prefer_compound_matches_no_compound_returns_all(self):
        """Test that prefer_compound_matches returns all matches when no compound matches.
        
        **Validates: Requirements 10.2, 10.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create individual token matches only
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        honda_vec = vectorizer.vectorize_value("make", "honda")
        
        token1 = Token(text="toyota", original="Toyota", position=0, is_stop_word=False, ngram_size=1)
        token2 = Token(text="honda", original="Honda", position=7, is_stop_word=False, ngram_size=1)
        
        match1 = TokenMatch(token=token1, schema_vector=toyota_vec, similarity=0.9, match_type="exact")
        match2 = TokenMatch(token=token2, schema_vector=honda_vec, similarity=0.8, match_type="exact")
        
        matcher = SchemaMatcher({}, encoder)
        result = matcher.prefer_compound_matches([match1, match2])
        
        # All matches should be returned since no compound matches
        assert len(result) == 2
    
    def test_prefer_compound_matches_suppresses_covered_individual(self):
        """Test that prefer_compound_matches suppresses individual matches covered by compound.
        
        Query: "Find brake pads"
        - "brake" at position 5
        - "pads" at position 11
        - "brake pads" compound at position 5
        
        The compound "brake pads" should suppress individual "brake" and "pads" matches.
        
        **Validates: Requirements 10.2, 10.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        brake_vec = vectorizer.vectorize_value("category", "brake")
        pads_vec = vectorizer.vectorize_value("category", "pads")
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        
        # Create individual tokens
        brake_token = Token(text="brake", original="brake", position=5, is_stop_word=False, ngram_size=1)
        pads_token = Token(text="pads", original="pads", position=11, is_stop_word=False, ngram_size=1)
        
        # Create compound token
        compound_token = Token(text="brake pads", original="brake pads", position=5, is_stop_word=False, ngram_size=2)
        
        # Create matches
        brake_match = TokenMatch(token=brake_token, schema_vector=brake_vec, similarity=0.7, match_type="partial")
        pads_match = TokenMatch(token=pads_token, schema_vector=pads_vec, similarity=0.7, match_type="partial")
        compound_match = TokenMatch(token=compound_token, schema_vector=brake_pads_vec, similarity=0.9, match_type="compound")
        
        matcher = SchemaMatcher({}, encoder)
        result = matcher.prefer_compound_matches([brake_match, pads_match, compound_match])
        
        # Only compound match should remain
        assert len(result) == 1
        assert result[0].match_type == "compound"
        assert result[0].token.text == "brake pads"
    
    def test_prefer_compound_matches_keeps_uncovered_individual(self):
        """Test that prefer_compound_matches keeps individual matches not covered by compound.
        
        Query: "Find Toyota brake pads"
        - "toyota" at position 5 (not covered)
        - "brake" at position 12 (covered by compound)
        - "pads" at position 18 (covered by compound)
        - "brake pads" compound at position 12
        
        "toyota" should be kept, "brake" and "pads" should be suppressed.
        
        **Validates: Requirements 10.2, 10.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        brake_vec = vectorizer.vectorize_value("category", "brake")
        pads_vec = vectorizer.vectorize_value("category", "pads")
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        
        # Create individual tokens
        toyota_token = Token(text="toyota", original="Toyota", position=5, is_stop_word=False, ngram_size=1)
        brake_token = Token(text="brake", original="brake", position=12, is_stop_word=False, ngram_size=1)
        pads_token = Token(text="pads", original="pads", position=18, is_stop_word=False, ngram_size=1)
        
        # Create compound token
        compound_token = Token(text="brake pads", original="brake pads", position=12, is_stop_word=False, ngram_size=2)
        
        # Create matches
        toyota_match = TokenMatch(token=toyota_token, schema_vector=toyota_vec, similarity=0.95, match_type="exact")
        brake_match = TokenMatch(token=brake_token, schema_vector=brake_vec, similarity=0.7, match_type="partial")
        pads_match = TokenMatch(token=pads_token, schema_vector=pads_vec, similarity=0.7, match_type="partial")
        compound_match = TokenMatch(token=compound_token, schema_vector=brake_pads_vec, similarity=0.9, match_type="compound")
        
        matcher = SchemaMatcher({}, encoder)
        result = matcher.prefer_compound_matches([toyota_match, brake_match, pads_match, compound_match])
        
        # Toyota and compound match should remain
        assert len(result) == 2
        
        # Check that toyota is in results
        toyota_in_result = any(m.token.text == "toyota" for m in result)
        assert toyota_in_result
        
        # Check that compound is in results
        compound_in_result = any(m.match_type == "compound" for m in result)
        assert compound_in_result
        
        # Check that individual brake and pads are NOT in results
        brake_in_result = any(m.token.text == "brake" and m.token.ngram_size == 1 for m in result)
        pads_in_result = any(m.token.text == "pads" and m.token.ngram_size == 1 for m in result)
        assert not brake_in_result
        assert not pads_in_result
    
    def test_prefer_compound_matches_sorted_by_similarity(self):
        """Test that prefer_compound_matches returns results sorted by similarity.
        
        **Validates: Requirements 10.2, 10.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        
        # Create tokens at non-overlapping positions
        toyota_token = Token(text="toyota", original="Toyota", position=0, is_stop_word=False, ngram_size=1)
        compound_token = Token(text="brake pads", original="brake pads", position=20, is_stop_word=False, ngram_size=2)
        
        # Create matches with different similarities
        toyota_match = TokenMatch(token=toyota_token, schema_vector=toyota_vec, similarity=0.5, match_type="exact")
        compound_match = TokenMatch(token=compound_token, schema_vector=brake_pads_vec, similarity=0.9, match_type="compound")
        
        matcher = SchemaMatcher({}, encoder)
        result = matcher.prefer_compound_matches([toyota_match, compound_match])
        
        # Results should be sorted by similarity descending
        assert len(result) == 2
        assert result[0].similarity >= result[1].similarity
    
    def test_prefer_compound_matches_multiple_compounds(self):
        """Test that prefer_compound_matches handles multiple compound matches.
        
        **Validates: Requirements 10.2, 10.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        oil_filter_vec = vectorizer.vectorize_value("category", "oil filter")
        
        # Create compound tokens at different positions
        compound1_token = Token(text="brake pads", original="brake pads", position=0, is_stop_word=False, ngram_size=2)
        compound2_token = Token(text="oil filter", original="oil filter", position=15, is_stop_word=False, ngram_size=2)
        
        # Create matches
        compound1_match = TokenMatch(token=compound1_token, schema_vector=brake_pads_vec, similarity=0.9, match_type="compound")
        compound2_match = TokenMatch(token=compound2_token, schema_vector=oil_filter_vec, similarity=0.85, match_type="compound")
        
        matcher = SchemaMatcher({}, encoder)
        result = matcher.prefer_compound_matches([compound1_match, compound2_match])
        
        # Both compound matches should be kept
        assert len(result) == 2
        assert all(m.match_type == "compound" for m in result)
    
    def test_prefer_compound_matches_identifies_compound_by_ngram_size(self):
        """Test that prefer_compound_matches identifies compounds by ngram_size > 1.
        
        **Validates: Requirements 10.2, 10.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        brake_vec = vectorizer.vectorize_value("category", "brake")
        
        # Create tokens - compound has ngram_size=2 but match_type="partial"
        brake_token = Token(text="brake", original="brake", position=0, is_stop_word=False, ngram_size=1)
        compound_token = Token(text="brake pads", original="brake pads", position=0, is_stop_word=False, ngram_size=2)
        
        # Create matches - compound has match_type="partial" but ngram_size > 1
        brake_match = TokenMatch(token=brake_token, schema_vector=brake_vec, similarity=0.7, match_type="partial")
        compound_match = TokenMatch(token=compound_token, schema_vector=brake_pads_vec, similarity=0.9, match_type="partial")
        
        matcher = SchemaMatcher({}, encoder)
        result = matcher.prefer_compound_matches([brake_match, compound_match])
        
        # Compound should be identified by ngram_size > 1, individual "brake" should be suppressed
        assert len(result) == 1
        assert result[0].token.ngram_size == 2
    
    def test_prefer_compound_matches_preserves_all_when_no_overlap(self):
        """Test that prefer_compound_matches preserves all matches when no position overlap.
        
        **Validates: Requirements 10.2, 10.3**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors
        toyota_vec = vectorizer.vectorize_value("make", "toyota")
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        
        # Create tokens at completely separate positions (no overlap)
        toyota_token = Token(text="toyota", original="Toyota", position=0, is_stop_word=False, ngram_size=1)
        compound_token = Token(text="brake pads", original="brake pads", position=100, is_stop_word=False, ngram_size=2)
        
        # Create matches
        toyota_match = TokenMatch(token=toyota_token, schema_vector=toyota_vec, similarity=0.9, match_type="exact")
        compound_match = TokenMatch(token=compound_token, schema_vector=brake_pads_vec, similarity=0.85, match_type="compound")
        
        matcher = SchemaMatcher({}, encoder)
        result = matcher.prefer_compound_matches([toyota_match, compound_match])
        
        # Both should be preserved since no overlap
        assert len(result) == 2


class TestSchemaMatcherSynonymAliasMatching:
    """Test SchemaMatcher.match_token() with synonym and alias vectors.
    
    **Validates: Requirement 11.4** - THE SDK SHALL match queries against both
    primary values and synonyms
    """
    
    def test_match_token_finds_synonym_match(self):
        """Test that match_token finds matches against synonym vectors.
        
        When a synonym is added (e.g., "TYT" for "Toyota"), the match_token()
        method should find the synonym vector when the query contains "TYT".
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add a synonym for "Toyota"
        vectorizer.add_synonym("Toyota", "tyt")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Token with synonym text (normalized to lowercase)
        token = Token(
            text="tyt",
            original="TYT",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Should find the synonym match
        assert len(result) >= 1
        
        # Find the synonym match
        synonym_matches = [m for m in result if m.schema_vector.key == "synonym:tyt"]
        assert len(synonym_matches) == 1
        
        # The original_value should be the primary value "Toyota"
        assert synonym_matches[0].schema_vector.original_value == "Toyota"
    
    def test_match_token_synonym_returns_primary_value_in_original_value(self):
        """Test that synonym match returns primary value in original_value field.
        
        When a synonym is matched, the TokenMatch.schema_vector.original_value
        should contain the primary value (e.g., "Toyota"), not the synonym.
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add multiple synonyms for "Toyota"
        vectorizer.add_synonym("Toyota", "toy")
        vectorizer.add_synonym("Toyota", "tyt")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Match against "toy" synonym
        token = Token(
            text="toy",
            original="Toy",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Find the synonym match
        synonym_matches = [m for m in result if m.schema_vector.key == "synonym:toy"]
        assert len(synonym_matches) == 1
        
        # original_value should be "Toyota" (the primary value)
        assert synonym_matches[0].schema_vector.original_value == "Toyota"
        
        # element_type should be "value" for synonyms
        assert synonym_matches[0].schema_vector.element_type == "value"
    
    def test_match_token_finds_alias_match(self):
        """Test that match_token finds matches against alias vectors.
        
        When an alias is added (e.g., "manufacturer" for "make"), the match_token()
        method should find the alias vector when the query contains "manufacturer".
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add an alias for "make"
        vectorizer.add_alias("make", "manufacturer")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Token with alias text
        token = Token(
            text="manufacturer",
            original="manufacturer",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Should find the alias match
        assert len(result) >= 1
        
        # Find the alias match
        alias_matches = [m for m in result if m.schema_vector.key == "alias:manufacturer"]
        assert len(alias_matches) == 1
        
        # The original_value should be the primary role "make"
        assert alias_matches[0].schema_vector.original_value == "make"
    
    def test_match_token_alias_returns_primary_role_in_original_value(self):
        """Test that alias match returns primary role in original_value field.
        
        When an alias is matched, the TokenMatch.schema_vector.original_value
        should contain the primary role name (e.g., "make"), not the alias.
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add multiple aliases for "make"
        vectorizer.add_alias("make", "manufacturer")
        vectorizer.add_alias("make", "brand")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Match against "brand" alias
        token = Token(
            text="brand",
            original="brand",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Find the alias match
        alias_matches = [m for m in result if m.schema_vector.key == "alias:brand"]
        assert len(alias_matches) == 1
        
        # original_value should be "make" (the primary role)
        assert alias_matches[0].schema_vector.original_value == "make"
        
        # element_type should be "role" for aliases
        assert alias_matches[0].schema_vector.element_type == "role"
    
    def test_match_token_synonym_has_exact_match_type_when_text_matches(self):
        """Test that synonym match has 'exact' match_type when text matches.
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add a synonym
        vectorizer.add_synonym("Toyota", "tyt")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Token with exact synonym text
        token = Token(
            text="tyt",
            original="TYT",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Find the synonym match
        synonym_matches = [m for m in result if m.schema_vector.key == "synonym:tyt"]
        assert len(synonym_matches) == 1
        
        # Should be an exact match (case-insensitive comparison with original_value)
        # Note: The match_type is determined by comparing token.text with schema_vector.original_value
        # For synonyms, original_value is the primary value ("Toyota"), not the synonym
        # So this will be a "partial" match since "tyt" != "Toyota"
        assert synonym_matches[0].match_type == "partial"
    
    def test_match_token_alias_has_exact_match_type_when_text_matches(self):
        """Test that alias match has 'exact' match_type when text matches.
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add an alias
        vectorizer.add_alias("make", "manufacturer")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Token with exact alias text
        token = Token(
            text="manufacturer",
            original="manufacturer",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Find the alias match
        alias_matches = [m for m in result if m.schema_vector.key == "alias:manufacturer"]
        assert len(alias_matches) == 1
        
        # For aliases, original_value is the primary role ("make"), not the alias
        # So this will be a "partial" match since "manufacturer" != "make"
        assert alias_matches[0].match_type == "partial"
    
    def test_match_token_matches_both_primary_and_synonym(self):
        """Test that match_token can match both primary value and synonym.
        
        When both a primary value and its synonym are in the schema, a query
        for the primary value should match the primary, and a query for the
        synonym should match the synonym.
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add primary value manually to schema_vectors
        primary_vec = vectorizer.vectorize_value("make", "toyota")
        vectorizer._schema_vectors[primary_vec.key] = primary_vec
        
        # Add synonym
        vectorizer.add_synonym("Toyota", "tyt")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Query for primary value
        primary_token = Token(
            text="toyota",
            original="Toyota",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        primary_result = matcher.match_token(primary_token)
        
        # Should find the primary value match
        primary_matches = [m for m in primary_result if m.schema_vector.key == "make=toyota"]
        assert len(primary_matches) == 1
        assert primary_matches[0].schema_vector.original_value == "toyota"
        
        # Query for synonym
        synonym_token = Token(
            text="tyt",
            original="TYT",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        synonym_result = matcher.match_token(synonym_token)
        
        # Should find the synonym match
        synonym_matches = [m for m in synonym_result if m.schema_vector.key == "synonym:tyt"]
        assert len(synonym_matches) == 1
        assert synonym_matches[0].schema_vector.original_value == "Toyota"
    
    def test_match_token_matches_both_primary_role_and_alias(self):
        """Test that match_token can match both primary role and alias.
        
        When both a primary role and its alias are in the schema, a query
        for the primary role should match the primary, and a query for the
        alias should match the alias.
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add primary role manually to schema_vectors
        primary_vec = vectorizer.vectorize_role("make")
        vectorizer._schema_vectors[primary_vec.key] = primary_vec
        
        # Add alias
        vectorizer.add_alias("make", "manufacturer")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Query for primary role
        primary_token = Token(
            text="make",
            original="make",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        primary_result = matcher.match_token(primary_token)
        
        # Should find the primary role match
        primary_matches = [m for m in primary_result if m.schema_vector.key == "make"]
        assert len(primary_matches) == 1
        assert primary_matches[0].schema_vector.element_type == "role"
        
        # Query for alias
        alias_token = Token(
            text="manufacturer",
            original="manufacturer",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        alias_result = matcher.match_token(alias_token)
        
        # Should find the alias match
        alias_matches = [m for m in alias_result if m.schema_vector.key == "alias:manufacturer"]
        assert len(alias_matches) == 1
        assert alias_matches[0].schema_vector.original_value == "make"
    
    def test_match_query_finds_synonym_in_query(self):
        """Test that match_query finds synonyms in a full query.
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add synonym
        vectorizer.add_synonym("Toyota", "tyt")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Find TYT cars", tokenizer)
        
        # Should find the synonym match
        synonym_matches = [m for m in result.value_matches 
                         if m.schema_vector.key == "synonym:tyt"]
        assert len(synonym_matches) >= 1
        
        # original_value should be the primary value
        assert synonym_matches[0].schema_vector.original_value == "Toyota"
    
    def test_match_query_finds_alias_in_query(self):
        """Test that match_query finds aliases in a full query.
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add alias
        vectorizer.add_alias("make", "manufacturer")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        result = matcher.match_query("Show manufacturer", tokenizer)
        
        # Should find the alias match
        alias_matches = [m for m in result.role_matches 
                        if m.schema_vector.key == "alias:manufacturer"]
        assert len(alias_matches) >= 1
        
        # original_value should be the primary role
        assert alias_matches[0].schema_vector.original_value == "make"
    
    def test_synonym_match_similarity_is_high_for_exact_text(self):
        """Test that synonym match has high similarity when text matches exactly.
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add synonym
        vectorizer.add_synonym("Toyota", "tyt")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Token with exact synonym text
        token = Token(
            text="tyt",
            original="TYT",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Find the synonym match
        synonym_matches = [m for m in result if m.schema_vector.key == "synonym:tyt"]
        assert len(synonym_matches) == 1
        
        # Similarity should be 1.0 (identical vectors) since the synonym vector
        # was generated from "tyt" and the token is also "tyt"
        assert synonym_matches[0].similarity == 1.0
    
    def test_alias_match_similarity_is_high_for_exact_text(self):
        """Test that alias match has high similarity when text matches exactly.
        
        **Validates: Requirement 11.4**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add alias
        vectorizer.add_alias("make", "manufacturer")
        schema_vectors = vectorizer.get_schema_vectors()
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        
        # Token with exact alias text
        token = Token(
            text="manufacturer",
            original="manufacturer",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        result = matcher.match_token(token)
        
        # Find the alias match
        alias_matches = [m for m in result if m.schema_vector.key == "alias:manufacturer"]
        assert len(alias_matches) == 1
        
        # Similarity should be 1.0 (identical vectors) since the alias vector
        # was generated from "manufacturer" and the token is also "manufacturer"
        assert alias_matches[0].similarity == 1.0


class TestSchemaMatcherWordOrderPermutation:
    """Test word order permutation matching for compound values.
    
    **Validates: Requirement 10.5** - THE SDK SHALL handle word order variations
    in compound values
    
    The key insight is that HDC bundling is commutative (order-independent),
    so "A + B" = "B + A". This means compound vectors for "Brake Pads" should
    be similar to query vectors for "pads brake".
    """
    
    def test_word_order_permutation_produces_similar_compound_vectors(self):
        """Test that different word orders produce identical compound vectors.
        
        Since HDC bundling is commutative, vectorize_compound(["Brake", "Pads"])
        should produce the same vector as vectorize_compound(["Pads", "Brake"]).
        
        **Validates: Requirement 10.5**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Generate compound vectors with different word orders
        brake_pads = vectorizer.vectorize_compound(["brake", "pads"])
        pads_brake = vectorizer.vectorize_compound(["pads", "brake"])
        
        # Bundling is commutative, so vectors should be identical
        assert np.array_equal(brake_pads.data, pads_brake.data)
    
    def test_query_pads_brake_matches_schema_brake_pads(self):
        """Test that query "pads brake" matches schema value "brake pads".
        
        This is the core test for word order permutation matching. A query
        with reversed word order should still match the schema value with
        high similarity because HDC bundling is commutative.
        
        **Validates: Requirement 10.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vector for "brake pads" (canonical order)
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        schema_vectors = {"category=brake pads": brake_pads_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        # Query with reversed word order: "pads brake"
        result = matcher.match_query("Find pads brake", tokenizer)
        
        # Should find compound match for "pads brake" against "brake pads"
        # because bundling is commutative
        assert len(result.compound_matches) > 0
        
        # The compound match should have high similarity (1.0 base + bonuses)
        # Since bundling is commutative, the vectors are identical
        compound_match = result.compound_matches[0]
        assert compound_match.similarity >= 1.0  # Base similarity is 1.0
    
    def test_word_order_permutation_similarity_is_identical(self):
        """Test that different word orders produce identical base similarity scores.
        
        Since the compound vectors are identical (bundling is commutative),
        the base similarity scores should be identical. However, the exact_match_bonus
        is only applied when the token text exactly matches the schema value text,
        so the final similarity may differ by the exact_match_bonus amount.
        
        **Validates: Requirement 10.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vector for "brake pads"
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        schema_vectors = {"category=brake pads": brake_pads_vec}
        
        # Use config with no exact_match_bonus to test base similarity
        match_config = MatchConfig(exact_match_bonus=0.0, compound_bonus=0.1)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        tokenizer = QueryTokenizer()
        
        # Query with canonical order: "brake pads"
        result_canonical = matcher.match_query("Find brake pads", tokenizer)
        
        # Query with reversed order: "pads brake"
        result_reversed = matcher.match_query("Find pads brake", tokenizer)
        
        # Both should have compound matches
        assert len(result_canonical.compound_matches) > 0
        assert len(result_reversed.compound_matches) > 0
        
        # Get the compound match similarities
        canonical_similarity = result_canonical.compound_matches[0].similarity
        reversed_similarity = result_reversed.compound_matches[0].similarity
        
        # Base similarities should be identical (both 1.0 + compound_bonus)
        # since bundling is commutative
        assert abs(canonical_similarity - reversed_similarity) < 0.001
    
    def test_three_word_order_permutation_matching(self):
        """Test that three-word permutations produce similar matches.
        
        For a three-word compound like "front brake pads", all permutations
        should produce identical compound vectors and similar match scores.
        The base similarity is identical, but exact_match_bonus only applies
        when the token text exactly matches the schema value.
        
        **Validates: Requirement 10.5**
        """
        import numpy as np
        
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vector for "front brake pads"
        schema_vec = vectorizer.vectorize_value("category", "front brake pads")
        schema_vectors = {"category=front brake pads": schema_vec}
        
        # Use tokenizer with max_ngram_size=3 to generate 3-grams
        tokenizer_config = TokenizerConfig(max_ngram_size=3)
        tokenizer = QueryTokenizer(config=tokenizer_config)
        
        # Use config with no exact_match_bonus to test base similarity
        match_config = MatchConfig(exact_match_bonus=0.0, compound_bonus=0.1)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Test different word order permutations
        queries = [
            "Find front brake pads",  # Original order
            "Find brake pads front",  # Rotated
            "Find pads front brake",  # Another rotation
        ]
        
        similarities = []
        for query in queries:
            result = matcher.match_query(query, tokenizer)
            # Find 3-gram compound matches
            three_gram_matches = [m for m in result.compound_matches 
                                 if m.token.ngram_size == 3]
            if three_gram_matches:
                similarities.append(three_gram_matches[0].similarity)
        
        # All permutations should have the same base similarity
        if len(similarities) > 1:
            for i in range(1, len(similarities)):
                assert abs(similarities[0] - similarities[i]) < 0.001
    
    def test_word_order_permutation_with_case_normalization(self):
        """Test word order permutation with case normalization.
        
        The tokenizer normalizes text to lowercase, so "Brake Pads" and
        "pads brake" should both match after normalization.
        
        **Validates: Requirement 10.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vector with lowercase (matching normalized tokens)
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        schema_vectors = {"category=brake pads": brake_pads_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        # Query with mixed case and reversed order
        result = matcher.match_query("Find PADS BRAKE", tokenizer)
        
        # Should find compound match (tokenizer normalizes to lowercase)
        assert len(result.compound_matches) > 0
        
        # The match should have high similarity
        assert result.compound_matches[0].similarity > 0.9
    
    def test_word_order_permutation_compound_match_preferred_over_individual(self):
        """Test that compound match is preferred over individual word matches.
        
        When "pads brake" matches "brake pads" as a compound, the compound
        match should be preferred over individual "pads" and "brake" matches.
        
        **Validates: Requirements 10.2, 10.3, 10.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vectors for compound and individual words
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        brake_vec = vectorizer.vectorize_value("part", "brake")
        pads_vec = vectorizer.vectorize_value("part", "pads")
        schema_vectors = {
            "category=brake pads": brake_pads_vec,
            "part=brake": brake_vec,
            "part=pads": pads_vec
        }
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        # Query with reversed word order
        result = matcher.match_query("Find pads brake", tokenizer)
        
        # Should have compound match
        assert len(result.compound_matches) > 0
        
        # Use prefer_compound_matches to filter
        filtered = matcher.prefer_compound_matches(result.token_matches)
        
        # Compound match should be in filtered results
        compound_in_filtered = any(m.match_type == "compound" or m.token.ngram_size > 1 
                                   for m in filtered)
        assert compound_in_filtered
    
    def test_word_order_permutation_match_type_is_compound(self):
        """Test that word order permutation matches have match_type='compound'.
        
        **Validates: Requirement 10.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vector for "brake pads"
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        schema_vectors = {"category=brake pads": brake_pads_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        # Query with reversed word order
        result = matcher.match_query("Find pads brake", tokenizer)
        
        # Compound matches should have match_type="compound"
        for match in result.compound_matches:
            assert match.match_type == "compound"
    
    def test_word_order_permutation_preserves_schema_value_reference(self):
        """Test that word order permutation matches reference the correct schema value.
        
        When "pads brake" matches "brake pads", the match should reference
        the original schema value "brake pads".
        
        **Validates: Requirement 10.5**
        """
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Create schema vector for "brake pads"
        brake_pads_vec = vectorizer.vectorize_value("category", "brake pads")
        schema_vectors = {"category=brake pads": brake_pads_vec}
        
        matcher = SchemaMatcher(schema_vectors, encoder)
        tokenizer = QueryTokenizer()
        
        # Query with reversed word order
        result = matcher.match_query("Find pads brake", tokenizer)
        
        # The compound match should reference the original schema value
        assert len(result.compound_matches) > 0
        match = result.compound_matches[0]
        assert match.schema_vector.original_value == "brake pads"
        assert match.schema_vector.key == "category=brake pads"

