"""
Property-based tests for Synonym Matching Equivalence.

This module contains property-based tests using Hypothesis to verify
that synonyms produce the same role association as primary values.

**Validates: Property 17** - Synonym Matching Equivalence
For any schema value with defined synonyms, queries using the synonym SHALL
produce matches with the same role association as queries using the primary value.

**Validates: Requirements 11.2, 11.4**
"""

import pytest
from hypothesis import given, settings, strategies as st, assume

from glyphh.encoder.base import Encoder
from glyphh.core.config import EncoderConfig
from glyphh.nl.schema_matcher import SchemaMatcher, MatchConfig
from glyphh.nl.schema_vectorizer import SchemaVectorizer
from glyphh.nl.query_tokenizer import Token, QueryTokenizer


# Generator Strategies

# Primary value generator - generates valid primary values (ASCII alphanumeric with spaces)
# Using ASCII only to avoid case-sensitivity issues with Unicode characters
primary_values = st.text(
    alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '),
    min_size=2, max_size=20
).filter(lambda x: x.strip() and len(x.strip()) >= 2)

# Synonym generator - generates valid synonym strings (ASCII alphanumeric)
# Using ASCII only to ensure consistent case handling
synonyms = st.text(
    alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
    min_size=2, max_size=15
).filter(lambda x: x.strip() and len(x.strip()) >= 2)

# Role name generator - generates valid role names (ASCII alphanumeric with underscore)
role_names = st.text(
    alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'),
    min_size=1, max_size=20
).filter(lambda x: x.strip())


class TestSynonymMatchingEquivalence:
    """
    Property tests for Synonym Matching Equivalence (Property 17).
    
    **Validates: Property 17** - Synonym Matching Equivalence
    For any schema value with defined synonyms, queries using the synonym SHALL
    produce matches with the same role association as queries using the primary value.
    
    **Validates: Requirements 11.2, 11.4**
    """
    
    @given(
        primary=primary_values,
        synonym=synonyms
    )
    @settings(max_examples=100)
    def test_synonym_match_returns_primary_value_in_original_value(
        self, primary: str, synonym: str
    ):
        """
        Property test: Synonym matches return the primary value in original_value.
        
        When a synonym is matched, the `original_value` field of the matched
        SchemaVector SHALL contain the primary value, not the synonym.
        
        **Validates: Property 17**
        **Validates: Requirements 11.2, 11.4**
        """
        # Normalize inputs
        primary_norm = primary.strip()
        synonym_norm = synonym.strip()
        
        # Ensure primary and synonym are different
        assume(primary_norm.lower() != synonym_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add synonym for the primary value
        vectorizer.add_synonym(primary_norm, synonym_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token from the synonym text
        # Use the same text as the synonym (the encoder generates vectors from exact strings)
        token = Token(
            text=synonym_norm,
            original=synonym_norm,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Find the synonym match (should have highest similarity since it's exact)
        synonym_matches = [
            m for m in matches 
            if m.schema_vector.key == f"synonym:{synonym_norm}"
        ]
        
        # Property: The synonym match should exist and have original_value == primary
        assert len(synonym_matches) == 1, \
            f"Expected exactly one synonym match for '{synonym_norm}', got {len(synonym_matches)}"
        
        synonym_match = synonym_matches[0]
        assert synonym_match.schema_vector.original_value == primary_norm, \
            f"Synonym match original_value should be '{primary_norm}', " \
            f"got '{synonym_match.schema_vector.original_value}'"
    
    @given(
        primary=primary_values,
        synonym=synonyms,
        role_name=role_names
    )
    @settings(max_examples=100)
    def test_synonym_and_primary_produce_same_role_association(
        self, primary: str, synonym: str, role_name: str
    ):
        """
        Property test: Synonyms produce the same role association as primary.
        
        When both a primary value and its synonym are added to the schema,
        matching against the synonym SHALL produce a match with the same
        role association (via original_value) as matching against the primary.
        
        **Validates: Property 17**
        **Validates: Requirements 11.2, 11.4**
        """
        # Normalize inputs
        primary_norm = primary.strip()
        synonym_norm = synonym.strip()
        role_norm = role_name.strip()
        
        # Ensure primary and synonym are different
        assume(primary_norm.lower() != synonym_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add the primary value as a schema value
        vectorizer.vectorize_value(role_norm, primary_norm)
        
        # Add synonym for the primary value
        vectorizer.add_synonym(primary_norm, synonym_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token from the synonym text
        # Use the same text as the synonym (the encoder generates vectors from exact strings)
        synonym_token = Token(
            text=synonym_norm,
            original=synonym_norm,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match synonym token against schema
        synonym_matches = matcher.match_token(synonym_token)
        
        # Find the synonym match
        synonym_match_list = [
            m for m in synonym_matches 
            if m.schema_vector.key == f"synonym:{synonym_norm}"
        ]
        
        # Property: The synonym match should have original_value == primary
        assert len(synonym_match_list) == 1, \
            f"Expected exactly one synonym match, got {len(synonym_match_list)}"
        
        synonym_match = synonym_match_list[0]
        
        # The original_value should be the primary value
        assert synonym_match.schema_vector.original_value == primary_norm, \
            f"Synonym match should have original_value='{primary_norm}', " \
            f"got '{synonym_match.schema_vector.original_value}'"
        
        # The element_type should be "value" (synonyms are value vectors)
        assert synonym_match.schema_vector.element_type == "value", \
            f"Synonym match should have element_type='value', " \
            f"got '{synonym_match.schema_vector.element_type}'"
    
    @given(
        primary=primary_values,
        synonyms_list=st.lists(synonyms, min_size=1, max_size=5, unique=True)
    )
    @settings(max_examples=100)
    def test_multiple_synonyms_all_map_to_same_primary(
        self, primary: str, synonyms_list: list
    ):
        """
        Property test: Multiple synonyms all map to the same primary value.
        
        When multiple synonyms are defined for a primary value, matching
        against any of them SHALL produce a match with original_value
        equal to the primary value.
        
        **Validates: Property 17**
        **Validates: Requirements 11.2, 11.4**
        """
        # Normalize primary
        primary_norm = primary.strip()
        
        # Filter out synonyms that match the primary
        filtered_synonyms = [
            s.strip() for s in synonyms_list 
            if s.strip().lower() != primary_norm.lower()
        ]
        assume(len(filtered_synonyms) >= 1)
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add all synonyms for the primary value
        for syn in filtered_synonyms:
            vectorizer.add_synonym(primary_norm, syn)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Property: Each synonym should map to the same primary value
        for syn in filtered_synonyms:
            # Create token from the synonym text
            # Use the same text as the synonym (the encoder generates vectors from exact strings)
            token = Token(
                text=syn,
                original=syn,
                position=0,
                is_stop_word=False,
                ngram_size=1
            )
            
            # Match token against schema
            matches = matcher.match_token(token)
            
            # Find the synonym match
            synonym_match_list = [
                m for m in matches 
                if m.schema_vector.key == f"synonym:{syn}"
            ]
            
            assert len(synonym_match_list) == 1, \
                f"Expected exactly one match for synonym '{syn}', " \
                f"got {len(synonym_match_list)}"
            
            # Verify original_value is the primary
            assert synonym_match_list[0].schema_vector.original_value == primary_norm, \
                f"Synonym '{syn}' should map to primary '{primary_norm}', " \
                f"got '{synonym_match_list[0].schema_vector.original_value}'"
    
    @given(
        primary=primary_values,
        synonym=synonyms
    )
    @settings(max_examples=100)
    def test_synonym_match_has_value_element_type(
        self, primary: str, synonym: str
    ):
        """
        Property test: Synonym matches have element_type == "value".
        
        Synonyms represent alternative names for values, so their
        SchemaVector SHALL have element_type == "value".
        
        **Validates: Property 17**
        **Validates: Requirements 11.2, 11.4**
        """
        # Normalize inputs
        primary_norm = primary.strip()
        synonym_norm = synonym.strip()
        
        # Ensure primary and synonym are different
        assume(primary_norm.lower() != synonym_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add synonym for the primary value
        vectorizer.add_synonym(primary_norm, synonym_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token from the synonym text
        # Use the same text as the synonym (the encoder generates vectors from exact strings)
        token = Token(
            text=synonym_norm,
            original=synonym_norm,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Find the synonym match
        synonym_matches = [
            m for m in matches 
            if m.schema_vector.key == f"synonym:{synonym_norm}"
        ]
        
        # Property: The synonym match should have element_type == "value"
        assert len(synonym_matches) == 1, \
            f"Expected exactly one synonym match, got {len(synonym_matches)}"
        
        assert synonym_matches[0].schema_vector.element_type == "value", \
            f"Synonym match should have element_type='value', " \
            f"got '{synonym_matches[0].schema_vector.element_type}'"
    
    @given(
        primary=primary_values,
        synonym=synonyms
    )
    @settings(max_examples=100)
    def test_synonym_stored_with_correct_key_format(
        self, primary: str, synonym: str
    ):
        """
        Property test: Synonyms are stored with "synonym:" prefix in key.
        
        Synonym vectors SHALL be stored with a key format of "synonym:synonym_text"
        to distinguish them from regular value vectors.
        
        **Validates: Property 17**
        **Validates: Requirements 11.2, 11.4**
        """
        # Normalize inputs
        primary_norm = primary.strip()
        synonym_norm = synonym.strip()
        
        # Ensure primary and synonym are different
        assume(primary_norm.lower() != synonym_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add synonym for the primary value
        vectorizer.add_synonym(primary_norm, synonym_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Property: The synonym should be stored with "synonym:" prefix
        expected_key = f"synonym:{synonym_norm}"
        assert expected_key in schema_vectors, \
            f"Expected synonym key '{expected_key}' in schema vectors, " \
            f"got keys: {list(schema_vectors.keys())}"
        
        # Verify the stored vector has correct properties
        stored_vector = schema_vectors[expected_key]
        assert stored_vector.original_value == primary_norm, \
            f"Stored synonym should have original_value='{primary_norm}', " \
            f"got '{stored_vector.original_value}'"
    
    @given(
        primary=primary_values,
        synonym=synonyms
    )
    @settings(max_examples=100)
    def test_synonym_mapping_tracked_in_get_synonyms(
        self, primary: str, synonym: str
    ):
        """
        Property test: Synonym mappings are tracked in get_synonyms().
        
        After adding a synonym, the get_synonyms() method SHALL return
        a dictionary containing the synonym → primary mapping.
        
        **Validates: Property 17**
        **Validates: Requirements 11.2, 11.4**
        """
        # Normalize inputs
        primary_norm = primary.strip()
        synonym_norm = synonym.strip()
        
        # Ensure primary and synonym are different
        assume(primary_norm.lower() != synonym_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add synonym for the primary value
        vectorizer.add_synonym(primary_norm, synonym_norm)
        
        # Get synonyms mapping
        synonyms_dict = vectorizer.get_synonyms()
        
        # Property: The synonym → primary mapping should be tracked
        assert synonym_norm in synonyms_dict, \
            f"Expected synonym '{synonym_norm}' in synonyms dict, " \
            f"got keys: {list(synonyms_dict.keys())}"
        
        assert synonyms_dict[synonym_norm] == primary_norm, \
            f"Synonym '{synonym_norm}' should map to '{primary_norm}', " \
            f"got '{synonyms_dict[synonym_norm]}'"
    
    @given(
        primary=primary_values,
        synonym=synonyms
    )
    @settings(max_examples=100)
    def test_synonym_vector_is_bipolar(
        self, primary: str, synonym: str
    ):
        """
        Property test: Synonym vectors are bipolar.
        
        The vector generated for a synonym SHALL be bipolar (all elements
        in {-1, +1}) with the same dimension as the encoder.
        
        **Validates: Property 17**
        **Validates: Requirements 11.2, 11.4**
        """
        import numpy as np
        
        # Normalize inputs
        primary_norm = primary.strip()
        synonym_norm = synonym.strip()
        
        # Ensure primary and synonym are different
        assume(primary_norm.lower() != synonym_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add synonym for the primary value
        vectorizer.add_synonym(primary_norm, synonym_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Get the synonym vector
        synonym_key = f"synonym:{synonym_norm}"
        synonym_vector = schema_vectors[synonym_key]
        
        # Property: Vector should be bipolar (all values in {-1, +1})
        assert np.all(np.isin(synonym_vector.vector.data, [-1, 1])), \
            f"Synonym vector should be bipolar, but contains values " \
            f"outside {{-1, +1}}"
        
        # Property: Vector should have correct dimension
        assert synonym_vector.vector.dimension == config.dimension, \
            f"Synonym vector dimension should be {config.dimension}, " \
            f"got {synonym_vector.vector.dimension}"
    
    @given(
        primary=primary_values,
        synonym=synonyms,
        role_name=role_names
    )
    @settings(max_examples=100)
    def test_synonym_match_similarity_is_high_for_exact_match(
        self, primary: str, synonym: str, role_name: str
    ):
        """
        Property test: Synonym match has high similarity when token matches synonym.
        
        When a token matches a synonym, the similarity score SHALL be high
        (1.0 for exact vector match). The match_type may be "partial" because
        the exact match check compares against original_value (the primary),
        not the synonym text.
        
        **Validates: Property 17**
        **Validates: Requirements 11.2, 11.4**
        """
        # Normalize inputs
        primary_norm = primary.strip()
        synonym_norm = synonym.strip()
        
        # Ensure primary and synonym are different
        assume(primary_norm.lower() != synonym_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add synonym for the primary value
        vectorizer.add_synonym(primary_norm, synonym_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Create matcher with exact match bonus
        match_config = MatchConfig(
            role_threshold=0.0, 
            value_threshold=0.0,
            exact_match_bonus=0.2
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token from the synonym text (exact match to synonym vector)
        # Use the same text as the synonym (the encoder generates vectors from exact strings)
        token = Token(
            text=synonym_norm,
            original=synonym_norm,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Find the synonym match
        synonym_matches = [
            m for m in matches 
            if m.schema_vector.key == f"synonym:{synonym_norm}"
        ]
        
        # Property: Should have exactly one synonym match
        assert len(synonym_matches) == 1, \
            f"Expected exactly one synonym match, got {len(synonym_matches)}"
        
        # The base similarity should be 1.0 (exact vector match)
        # Note: match_type may be "partial" because exact match check compares
        # token text against original_value (the primary), not the synonym text.
        # This is expected behavior - the important property is that the
        # original_value contains the primary value.
        
        # The similarity should be at least 1.0 (exact vector match)
        # If match_type is "exact" (token matches primary), bonus is added
        assert synonym_matches[0].similarity >= 1.0, \
            f"Synonym match should have similarity >= 1.0, " \
            f"got {synonym_matches[0].similarity}"
        
        # The key property: original_value should be the primary value
        assert synonym_matches[0].schema_vector.original_value == primary_norm, \
            f"Synonym match should have original_value='{primary_norm}', " \
            f"got '{synonym_matches[0].schema_vector.original_value}'"


# Alias generator - generates valid alias strings (ASCII alphanumeric)
# Using ASCII only to ensure consistent case handling
aliases = st.text(
    alphabet=st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
    min_size=2, max_size=15
).filter(lambda x: x.strip() and len(x.strip()) >= 2)


class TestAliasMatchingEquivalence:
    """
    Property tests for Alias Matching Equivalence (Property 18).
    
    **Validates: Property 18** - Alias Matching Equivalence
    For any role with defined aliases, queries using the alias SHALL produce
    matches with the same role as queries using the primary role name.
    
    **Validates: Requirements 11.1, 11.4**
    """
    
    @given(
        primary_role=role_names,
        alias=aliases
    )
    @settings(max_examples=100)
    def test_alias_match_returns_primary_role_in_original_value(
        self, primary_role: str, alias: str
    ):
        """
        Property test: Alias matches return the primary role in original_value.
        
        When an alias is matched, the `original_value` field of the matched
        SchemaVector SHALL contain the primary role name, not the alias.
        
        **Validates: Property 18**
        **Validates: Requirements 11.1, 11.4**
        """
        # Normalize inputs
        primary_norm = primary_role.strip()
        alias_norm = alias.strip()
        
        # Ensure primary and alias are different
        assume(primary_norm.lower() != alias_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add alias for the primary role
        vectorizer.add_alias(primary_norm, alias_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token from the alias text
        token = Token(
            text=alias_norm,
            original=alias_norm,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Find the alias match (should have highest similarity since it's exact)
        alias_matches = [
            m for m in matches 
            if m.schema_vector.key == f"alias:{alias_norm}"
        ]
        
        # Property: The alias match should exist and have original_value == primary role
        assert len(alias_matches) == 1, \
            f"Expected exactly one alias match for '{alias_norm}', got {len(alias_matches)}"
        
        alias_match = alias_matches[0]
        assert alias_match.schema_vector.original_value == primary_norm, \
            f"Alias match original_value should be '{primary_norm}', " \
            f"got '{alias_match.schema_vector.original_value}'"
    
    @given(
        primary_role=role_names,
        alias=aliases
    )
    @settings(max_examples=100)
    def test_alias_match_has_role_element_type(
        self, primary_role: str, alias: str
    ):
        """
        Property test: Alias matches have element_type == "role".
        
        Aliases represent alternative names for roles, so their
        SchemaVector SHALL have element_type == "role".
        
        **Validates: Property 18**
        **Validates: Requirements 11.1, 11.4**
        """
        # Normalize inputs
        primary_norm = primary_role.strip()
        alias_norm = alias.strip()
        
        # Ensure primary and alias are different
        assume(primary_norm.lower() != alias_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add alias for the primary role
        vectorizer.add_alias(primary_norm, alias_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token from the alias text
        token = Token(
            text=alias_norm,
            original=alias_norm,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Find the alias match
        alias_matches = [
            m for m in matches 
            if m.schema_vector.key == f"alias:{alias_norm}"
        ]
        
        # Property: The alias match should have element_type == "role"
        assert len(alias_matches) == 1, \
            f"Expected exactly one alias match, got {len(alias_matches)}"
        
        assert alias_matches[0].schema_vector.element_type == "role", \
            f"Alias match should have element_type='role', " \
            f"got '{alias_matches[0].schema_vector.element_type}'"
    
    @given(
        primary_role=role_names,
        alias=aliases
    )
    @settings(max_examples=100)
    def test_alias_and_primary_produce_same_role(
        self, primary_role: str, alias: str
    ):
        """
        Property test: Aliases produce the same role as primary.
        
        When both a primary role and its alias are added to the schema,
        matching against the alias SHALL produce a match with the same
        role (via original_value) as the primary role.
        
        **Validates: Property 18**
        **Validates: Requirements 11.1, 11.4**
        """
        # Normalize inputs
        primary_norm = primary_role.strip()
        alias_norm = alias.strip()
        
        # Ensure primary and alias are different
        assume(primary_norm.lower() != alias_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add the primary role as a schema role
        vectorizer.vectorize_role(primary_norm)
        
        # Add alias for the primary role
        vectorizer.add_alias(primary_norm, alias_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token from the alias text
        alias_token = Token(
            text=alias_norm,
            original=alias_norm,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match alias token against schema
        alias_matches = matcher.match_token(alias_token)
        
        # Find the alias match
        alias_match_list = [
            m for m in alias_matches 
            if m.schema_vector.key == f"alias:{alias_norm}"
        ]
        
        # Property: The alias match should have original_value == primary role
        assert len(alias_match_list) == 1, \
            f"Expected exactly one alias match, got {len(alias_match_list)}"
        
        alias_match = alias_match_list[0]
        
        # The original_value should be the primary role
        assert alias_match.schema_vector.original_value == primary_norm, \
            f"Alias match should have original_value='{primary_norm}', " \
            f"got '{alias_match.schema_vector.original_value}'"
        
        # The element_type should be "role" (aliases are role vectors)
        assert alias_match.schema_vector.element_type == "role", \
            f"Alias match should have element_type='role', " \
            f"got '{alias_match.schema_vector.element_type}'"
    
    @given(
        primary_role=role_names,
        aliases_list=st.lists(aliases, min_size=1, max_size=5, unique=True)
    )
    @settings(max_examples=100)
    def test_multiple_aliases_all_map_to_same_primary_role(
        self, primary_role: str, aliases_list: list
    ):
        """
        Property test: Multiple aliases all map to the same primary role.
        
        When multiple aliases are defined for a primary role, matching
        against any of them SHALL produce a match with original_value
        equal to the primary role.
        
        **Validates: Property 18**
        **Validates: Requirements 11.1, 11.4**
        """
        # Normalize primary
        primary_norm = primary_role.strip()
        
        # Filter out aliases that match the primary
        filtered_aliases = [
            a.strip() for a in aliases_list 
            if a.strip().lower() != primary_norm.lower()
        ]
        assume(len(filtered_aliases) >= 1)
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add all aliases for the primary role
        for alias in filtered_aliases:
            vectorizer.add_alias(primary_norm, alias)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Create matcher with low threshold to ensure matches
        match_config = MatchConfig(role_threshold=0.0, value_threshold=0.0)
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Property: Each alias should map to the same primary role
        for alias in filtered_aliases:
            # Create token from the alias text
            token = Token(
                text=alias,
                original=alias,
                position=0,
                is_stop_word=False,
                ngram_size=1
            )
            
            # Match token against schema
            matches = matcher.match_token(token)
            
            # Find the alias match
            alias_match_list = [
                m for m in matches 
                if m.schema_vector.key == f"alias:{alias}"
            ]
            
            assert len(alias_match_list) == 1, \
                f"Expected exactly one match for alias '{alias}', " \
                f"got {len(alias_match_list)}"
            
            # Verify original_value is the primary role
            assert alias_match_list[0].schema_vector.original_value == primary_norm, \
                f"Alias '{alias}' should map to primary role '{primary_norm}', " \
                f"got '{alias_match_list[0].schema_vector.original_value}'"
    
    @given(
        primary_role=role_names,
        alias=aliases
    )
    @settings(max_examples=100)
    def test_alias_stored_with_correct_key_format(
        self, primary_role: str, alias: str
    ):
        """
        Property test: Aliases are stored with "alias:" prefix in key.
        
        Alias vectors SHALL be stored with a key format of "alias:alias_text"
        to distinguish them from regular role vectors.
        
        **Validates: Property 18**
        **Validates: Requirements 11.1, 11.4**
        """
        # Normalize inputs
        primary_norm = primary_role.strip()
        alias_norm = alias.strip()
        
        # Ensure primary and alias are different
        assume(primary_norm.lower() != alias_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add alias for the primary role
        vectorizer.add_alias(primary_norm, alias_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Property: The alias should be stored with "alias:" prefix
        expected_key = f"alias:{alias_norm}"
        assert expected_key in schema_vectors, \
            f"Expected alias key '{expected_key}' in schema vectors, " \
            f"got keys: {list(schema_vectors.keys())}"
        
        # Verify the stored vector has correct properties
        stored_vector = schema_vectors[expected_key]
        assert stored_vector.original_value == primary_norm, \
            f"Stored alias should have original_value='{primary_norm}', " \
            f"got '{stored_vector.original_value}'"
    
    @given(
        primary_role=role_names,
        alias=aliases
    )
    @settings(max_examples=100)
    def test_alias_mapping_tracked_in_get_aliases(
        self, primary_role: str, alias: str
    ):
        """
        Property test: Alias mappings are tracked in get_aliases().
        
        After adding an alias, the get_aliases() method SHALL return
        a dictionary containing the alias → primary role mapping.
        
        **Validates: Property 18**
        **Validates: Requirements 11.1, 11.4**
        """
        # Normalize inputs
        primary_norm = primary_role.strip()
        alias_norm = alias.strip()
        
        # Ensure primary and alias are different
        assume(primary_norm.lower() != alias_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add alias for the primary role
        vectorizer.add_alias(primary_norm, alias_norm)
        
        # Get aliases mapping
        aliases_dict = vectorizer.get_aliases()
        
        # Property: The alias → primary role mapping should be tracked
        assert alias_norm in aliases_dict, \
            f"Expected alias '{alias_norm}' in aliases dict, " \
            f"got keys: {list(aliases_dict.keys())}"
        
        assert aliases_dict[alias_norm] == primary_norm, \
            f"Alias '{alias_norm}' should map to '{primary_norm}', " \
            f"got '{aliases_dict[alias_norm]}'"
    
    @given(
        primary_role=role_names,
        alias=aliases
    )
    @settings(max_examples=100)
    def test_alias_vector_is_bipolar(
        self, primary_role: str, alias: str
    ):
        """
        Property test: Alias vectors are bipolar.
        
        The vector generated for an alias SHALL be bipolar (all elements
        in {-1, +1}) with the same dimension as the encoder.
        
        **Validates: Property 18**
        **Validates: Requirements 11.1, 11.4**
        """
        import numpy as np
        
        # Normalize inputs
        primary_norm = primary_role.strip()
        alias_norm = alias.strip()
        
        # Ensure primary and alias are different
        assume(primary_norm.lower() != alias_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add alias for the primary role
        vectorizer.add_alias(primary_norm, alias_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Get the alias vector
        alias_key = f"alias:{alias_norm}"
        alias_vector = schema_vectors[alias_key]
        
        # Property: Vector should be bipolar (all values in {-1, +1})
        assert np.all(np.isin(alias_vector.vector.data, [-1, 1])), \
            f"Alias vector should be bipolar, but contains values " \
            f"outside {{-1, +1}}"
        
        # Property: Vector should have correct dimension
        assert alias_vector.vector.dimension == config.dimension, \
            f"Alias vector dimension should be {config.dimension}, " \
            f"got {alias_vector.vector.dimension}"
    
    @given(
        primary_role=role_names,
        alias=aliases
    )
    @settings(max_examples=100)
    def test_alias_match_similarity_is_high_for_exact_match(
        self, primary_role: str, alias: str
    ):
        """
        Property test: Alias match has high similarity when token matches alias.
        
        When a token matches an alias, the similarity score SHALL be high
        (1.0 for exact vector match). The match_type may be "partial" because
        the exact match check compares against original_value (the primary role),
        not the alias text.
        
        **Validates: Property 18**
        **Validates: Requirements 11.1, 11.4**
        """
        # Normalize inputs
        primary_norm = primary_role.strip()
        alias_norm = alias.strip()
        
        # Ensure primary and alias are different
        assume(primary_norm.lower() != alias_norm.lower())
        
        # Setup encoder and vectorizer
        config = EncoderConfig(dimension=1000, seed=42)
        encoder = Encoder(config)
        vectorizer = SchemaVectorizer(encoder, config)
        
        # Add alias for the primary role
        vectorizer.add_alias(primary_norm, alias_norm)
        
        # Get schema vectors
        schema_vectors = vectorizer.get_schema_vectors()
        
        # Create matcher with exact match bonus
        match_config = MatchConfig(
            role_threshold=0.0, 
            value_threshold=0.0,
            exact_match_bonus=0.2
        )
        matcher = SchemaMatcher(schema_vectors, encoder, match_config)
        
        # Create token from the alias text (exact match to alias vector)
        token = Token(
            text=alias_norm,
            original=alias_norm,
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        
        # Match token against schema
        matches = matcher.match_token(token)
        
        # Find the alias match
        alias_matches = [
            m for m in matches 
            if m.schema_vector.key == f"alias:{alias_norm}"
        ]
        
        # Property: Should have exactly one alias match
        assert len(alias_matches) == 1, \
            f"Expected exactly one alias match, got {len(alias_matches)}"
        
        # The base similarity should be 1.0 (exact vector match)
        # Note: match_type may be "partial" because exact match check compares
        # token text against original_value (the primary role), not the alias text.
        # This is expected behavior - the important property is that the
        # original_value contains the primary role name.
        
        # The similarity should be at least 1.0 (exact vector match)
        assert alias_matches[0].similarity >= 1.0, \
            f"Alias match should have similarity >= 1.0, " \
            f"got {alias_matches[0].similarity}"
        
        # The key property: original_value should be the primary role
        assert alias_matches[0].schema_vector.original_value == primary_norm, \
            f"Alias match should have original_value='{primary_norm}', " \
            f"got '{alias_matches[0].schema_vector.original_value}'"
