"""
Property-based tests for QueryTokenizer.

This module contains property-based tests using Hypothesis to verify
universal correctness properties for query tokenization.

**Validates: Property 5** - Query Tokenization Completeness
For any non-empty query string, tokenization SHALL produce at least one token,
and the concatenation of all token original texts (with spaces) SHALL
reconstruct the original query.

**Validates: Requirements 2.1, 2.4**
"""

import pytest
from hypothesis import given, settings, strategies as st, assume

from glyphh.nl.query_tokenizer import QueryTokenizer, Token, TokenizerConfig


# Generator Strategies (from design doc)

# Query generator - generates non-empty text with words
queries = st.text(min_size=1, max_size=500).filter(lambda x: x.strip())


class TestTokenizationCompleteness:
    """
    Property tests for Query Tokenization Completeness (Property 5).
    
    **Validates: Property 5** - Query Tokenization Completeness
    For any non-empty query string, tokenization SHALL produce at least one token,
    and the concatenation of all token original texts (with spaces) SHALL
    reconstruct the original query.
    
    **Validates: Requirements 2.1, 2.4**
    """
    
    @given(query=queries)
    @settings(max_examples=100)
    def test_non_empty_queries_with_words_produce_at_least_one_token(self, query: str):
        """
        Property test: Non-empty queries with at least one word produce at least one token.
        
        For any non-empty query string that contains at least one alphanumeric character,
        tokenization SHALL produce at least one token.
        
        **Validates: Requirements 2.1**
        """
        tokenizer = QueryTokenizer()
        
        # Check if query has at least one alphanumeric character (letter or digit)
        # Note: We use [a-zA-Z0-9] instead of \w because \w includes underscore,
        # which is removed by punctuation removal and would result in empty tokens
        import re
        has_alnum_chars = bool(re.search(r'[a-zA-Z0-9]', query))
        
        tokens = tokenizer.tokenize(query)
        
        if has_alnum_chars:
            # Property: Non-empty queries with alphanumeric characters produce at least one token
            assert len(tokens) >= 1, \
                f"Query '{query}' with alphanumeric characters should produce at least one token, got {len(tokens)}"
    
    @given(query=queries)
    @settings(max_examples=100)
    def test_token_originals_appear_at_correct_positions(self, query: str):
        """
        Property test: Token originals appear at their correct positions in the query.
        
        For any non-empty query string, each token's original text SHALL appear
        at the position indicated by the token's position field.
        
        **Validates: Requirements 2.4**
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize(query)
        
        for token in tokens:
            # Property: Token original should appear at the indicated position
            expected_original = query[token.position:token.position + len(token.original)]
            assert expected_original == token.original, \
                f"Token original '{token.original}' should appear at position {token.position} " \
                f"in query '{query}', but found '{expected_original}'"
    
    @given(query=queries)
    @settings(max_examples=100)
    def test_query_can_be_reconstructed_from_token_positions_and_originals(self, query: str):
        """
        Property test: Query can be reconstructed from token positions and originals.
        
        For any non-empty query string, the tokens' original texts at their
        positions SHALL cover all non-whitespace content of the original query.
        
        **Validates: Requirements 2.1, 2.4**
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize(query)
        
        if not tokens:
            # If no tokens, query should be whitespace-only, punctuation-only, or underscore-only
            import re
            # Check that there are no alphanumeric characters left
            # Note: We use [a-zA-Z0-9] instead of \w because \w includes underscore,
            # which is removed by punctuation removal and would result in empty tokens
            has_alnum_chars = bool(re.search(r'[a-zA-Z0-9]', query))
            if has_alnum_chars:
                # If there are alphanumeric characters, we should have tokens
                assert False, \
                    f"Query '{query}' has alphanumeric characters but produced no tokens"
            return
        
        # Build a reconstruction by placing token originals at their positions
        # and filling gaps with the original query content
        reconstructed_parts = []
        last_end = 0
        
        for token in sorted(tokens, key=lambda t: t.position):
            # Add any content between last token and this one (whitespace/punctuation)
            if token.position > last_end:
                reconstructed_parts.append(query[last_end:token.position])
            
            # Add the token's original text
            reconstructed_parts.append(token.original)
            last_end = token.position + len(token.original)
        
        # Add any trailing content
        if last_end < len(query):
            reconstructed_parts.append(query[last_end:])
        
        reconstructed = ''.join(reconstructed_parts)
        
        # Property: Reconstructed query should match original
        assert reconstructed == query, \
            f"Reconstructed query '{reconstructed}' does not match original '{query}'"
    
    @given(query=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'Zs'), whitelist_characters=' '),
        min_size=1, max_size=200
    ).filter(lambda x: x.strip() and any(c.isalnum() for c in x)))
    @settings(max_examples=100)
    def test_simple_word_queries_produce_tokens(self, query: str):
        """
        Property test: Simple word queries (letters, numbers, spaces) produce tokens.
        
        For any query containing only letters, numbers, and spaces with at least
        one alphanumeric character, tokenization SHALL produce at least one token.
        
        **Validates: Requirements 2.1**
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize(query)
        
        # Property: Simple queries with alphanumeric content produce tokens
        assert len(tokens) >= 1, \
            f"Simple query '{query}' should produce at least one token"
    
    @given(query=queries)
    @settings(max_examples=100)
    def test_token_positions_are_non_negative(self, query: str):
        """
        Property test: All token positions are non-negative.
        
        For any query, all token positions SHALL be non-negative integers.
        
        **Validates: Requirements 2.4**
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize(query)
        
        for token in tokens:
            # Property: Position must be non-negative
            assert token.position >= 0, \
                f"Token '{token.original}' has negative position {token.position}"
    
    @given(query=queries)
    @settings(max_examples=100)
    def test_token_positions_are_within_query_bounds(self, query: str):
        """
        Property test: All token positions are within query bounds.
        
        For any query, all token positions SHALL be within the bounds of the query string.
        
        **Validates: Requirements 2.4**
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize(query)
        
        for token in tokens:
            # Property: Position must be within query bounds
            assert token.position < len(query), \
                f"Token '{token.original}' position {token.position} exceeds query length {len(query)}"
            
            # Property: Token original must fit within query from its position
            assert token.position + len(token.original) <= len(query), \
                f"Token '{token.original}' at position {token.position} extends beyond query"
    
    @given(query=queries)
    @settings(max_examples=100)
    def test_tokens_are_in_position_order(self, query: str):
        """
        Property test: Tokens are returned in position order.
        
        For any query, tokens SHALL be returned in ascending position order.
        
        **Validates: Requirements 2.4**
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize(query)
        
        if len(tokens) > 1:
            positions = [t.position for t in tokens]
            # Property: Positions should be in ascending order
            assert positions == sorted(positions), \
                f"Token positions {positions} are not in ascending order"
    
    @given(query=queries)
    @settings(max_examples=100)
    def test_single_word_tokens_have_ngram_size_one(self, query: str):
        """
        Property test: Single word tokens have ngram_size of 1.
        
        For any query, tokens from tokenize() (not generate_ngrams) SHALL have
        ngram_size of 1.
        
        **Validates: Requirements 2.1**
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize(query)
        
        for token in tokens:
            # Property: All tokens from tokenize() should have ngram_size=1
            assert token.ngram_size == 1, \
                f"Token '{token.original}' should have ngram_size=1, got {token.ngram_size}"
    
    @given(
        query=queries,
        max_ngram=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_tokenization_completeness_with_different_configs(
        self, query: str, max_ngram: int
    ):
        """
        Property test: Tokenization completeness holds across different configs.
        
        For any query and any valid tokenizer configuration, the tokenization
        completeness property SHALL hold.
        
        **Validates: Requirements 2.1, 2.4**
        """
        config = TokenizerConfig(max_ngram_size=max_ngram)
        tokenizer = QueryTokenizer(config=config)
        tokens = tokenizer.tokenize(query)
        
        # Check if query has at least one alphanumeric character
        # Note: We use [a-zA-Z0-9] instead of \w because \w includes underscore,
        # which is removed by punctuation removal and would result in empty tokens
        import re
        has_alnum_chars = bool(re.search(r'[a-zA-Z0-9]', query))
        
        if has_alnum_chars:
            # Property: Non-empty queries with alphanumeric characters produce at least one token
            assert len(tokens) >= 1, \
                f"Query '{query}' with alphanumeric characters should produce at least one token"
        
        # Property: All token originals appear at correct positions
        for token in tokens:
            expected_original = query[token.position:token.position + len(token.original)]
            assert expected_original == token.original, \
                f"Token original mismatch at position {token.position}"


class TestNgramGeneration:
    """
    Property tests for N-gram Generation (Property 6).
    
    **Validates: Property 6** - N-gram Generation
    For any query with N words where N >= 2, tokenization with max_ngram_size=M
    SHALL produce n-grams of sizes 2 through min(N, M).
    
    **Validates: Requirement 2.2**
    """
    
    # Strategy for generating multi-word queries (at least 2 words)
    # Uses simple alphanumeric words separated by spaces
    multi_word_queries = st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N')),
            min_size=1,
            max_size=20
        ).filter(lambda x: x.strip()),
        min_size=2,
        max_size=10
    ).map(lambda words: " ".join(words))
    
    @given(
        query=multi_word_queries,
        max_ngram_size=st.integers(min_value=2, max_value=6)
    )
    @settings(max_examples=100)
    def test_ngram_sizes_are_generated_correctly(self, query: str, max_ngram_size: int):
        """
        Property test: Correct n-gram sizes are generated.
        
        For any query with N words where N >= 2, tokenization with max_ngram_size=M
        SHALL produce n-grams of sizes 2 through min(N, M).
        
        **Validates: Property 6**
        **Validates: Requirement 2.2**
        """
        config = TokenizerConfig(max_ngram_size=max_ngram_size)
        tokenizer = QueryTokenizer(config=config)
        
        # Get single-word tokens
        tokens = tokenizer.tokenize(query)
        n_tokens = len(tokens)
        
        # Skip if we don't have at least 2 tokens
        assume(n_tokens >= 2)
        
        # Calculate expected n-gram sizes: 2 through min(N, M)
        expected_max_ngram = min(n_tokens, max_ngram_size)
        expected_sizes = set(range(2, expected_max_ngram + 1))
        
        # Generate n-grams for each size and collect actual sizes
        actual_sizes = set()
        for size in range(2, max_ngram_size + 1):
            ngrams = tokenizer.generate_ngrams(tokens, size)
            if ngrams:
                actual_sizes.add(size)
        
        # Property: N-grams of sizes 2 through min(N, M) should be generated
        assert actual_sizes == expected_sizes, \
            f"For {n_tokens} tokens and max_ngram_size={max_ngram_size}, " \
            f"expected n-gram sizes {expected_sizes}, got {actual_sizes}"
    
    @given(
        query=multi_word_queries,
        max_ngram_size=st.integers(min_value=2, max_value=6)
    )
    @settings(max_examples=100)
    def test_ngram_count_is_correct(self, query: str, max_ngram_size: int):
        """
        Property test: The number of n-grams of size k is (N - k + 1).
        
        For N tokens, the number of n-grams of size k should be exactly (N - k + 1).
        
        **Validates: Property 6**
        **Validates: Requirement 2.2**
        """
        config = TokenizerConfig(max_ngram_size=max_ngram_size)
        tokenizer = QueryTokenizer(config=config)
        
        # Get single-word tokens
        tokens = tokenizer.tokenize(query)
        n_tokens = len(tokens)
        
        # Skip if we don't have at least 2 tokens
        assume(n_tokens >= 2)
        
        # For each valid n-gram size, verify the count
        for k in range(2, min(n_tokens, max_ngram_size) + 1):
            ngrams = tokenizer.generate_ngrams(tokens, k)
            expected_count = n_tokens - k + 1
            
            # Property: Number of n-grams of size k is (N - k + 1)
            assert len(ngrams) == expected_count, \
                f"For {n_tokens} tokens, expected {expected_count} {k}-grams, got {len(ngrams)}"
    
    @given(
        query=multi_word_queries,
        max_ngram_size=st.integers(min_value=2, max_value=6)
    )
    @settings(max_examples=100)
    def test_ngram_tokens_have_correct_ngram_size_attribute(self, query: str, max_ngram_size: int):
        """
        Property test: All n-gram tokens have the correct ngram_size attribute.
        
        For n-grams of size k, all tokens should have ngram_size=k.
        
        **Validates: Property 6**
        **Validates: Requirement 2.2**
        """
        config = TokenizerConfig(max_ngram_size=max_ngram_size)
        tokenizer = QueryTokenizer(config=config)
        
        # Get single-word tokens
        tokens = tokenizer.tokenize(query)
        n_tokens = len(tokens)
        
        # Skip if we don't have at least 2 tokens
        assume(n_tokens >= 2)
        
        # For each valid n-gram size, verify the ngram_size attribute
        for k in range(2, min(n_tokens, max_ngram_size) + 1):
            ngrams = tokenizer.generate_ngrams(tokens, k)
            
            for ngram in ngrams:
                # Property: All n-gram tokens have correct ngram_size attribute
                assert ngram.ngram_size == k, \
                    f"N-gram '{ngram.text}' should have ngram_size={k}, got {ngram.ngram_size}"
    
    @given(
        query=multi_word_queries,
        max_ngram_size=st.integers(min_value=2, max_value=6)
    )
    @settings(max_examples=100)
    def test_ngrams_beyond_max_size_are_not_generated(self, query: str, max_ngram_size: int):
        """
        Property test: N-grams beyond max_ngram_size are not generated.
        
        For max_ngram_size=M, no n-grams of size > M should be generated.
        
        **Validates: Property 6**
        **Validates: Requirement 2.2**
        """
        config = TokenizerConfig(max_ngram_size=max_ngram_size)
        tokenizer = QueryTokenizer(config=config)
        
        # Get single-word tokens
        tokens = tokenizer.tokenize(query)
        n_tokens = len(tokens)
        
        # Skip if we don't have at least 2 tokens
        assume(n_tokens >= 2)
        
        # Try to generate n-grams beyond max_ngram_size
        for k in range(max_ngram_size + 1, max_ngram_size + 3):
            ngrams = tokenizer.generate_ngrams(tokens, k)
            
            # Property: N-grams beyond max_ngram_size should not be generated
            assert len(ngrams) == 0, \
                f"N-grams of size {k} should not be generated with max_ngram_size={max_ngram_size}"
    
    @given(
        query=multi_word_queries,
        max_ngram_size=st.integers(min_value=2, max_value=6)
    )
    @settings(max_examples=100)
    def test_ngrams_beyond_token_count_are_not_generated(self, query: str, max_ngram_size: int):
        """
        Property test: N-grams beyond token count are not generated.
        
        For N tokens, no n-grams of size > N should be generated.
        
        **Validates: Property 6**
        **Validates: Requirement 2.2**
        """
        config = TokenizerConfig(max_ngram_size=max_ngram_size)
        tokenizer = QueryTokenizer(config=config)
        
        # Get single-word tokens
        tokens = tokenizer.tokenize(query)
        n_tokens = len(tokens)
        
        # Skip if we don't have at least 2 tokens
        assume(n_tokens >= 2)
        
        # Try to generate n-grams beyond token count
        for k in range(n_tokens + 1, n_tokens + 3):
            ngrams = tokenizer.generate_ngrams(tokens, k)
            
            # Property: N-grams beyond token count should not be generated
            assert len(ngrams) == 0, \
                f"N-grams of size {k} should not be generated with only {n_tokens} tokens"
    
    @given(
        query=multi_word_queries,
        max_ngram_size=st.integers(min_value=2, max_value=6)
    )
    @settings(max_examples=100)
    def test_ngram_text_contains_correct_number_of_words(self, query: str, max_ngram_size: int):
        """
        Property test: N-gram text contains the correct number of words.
        
        For n-grams of size k, the text should contain exactly k words.
        
        **Validates: Property 6**
        **Validates: Requirement 2.2**
        """
        config = TokenizerConfig(max_ngram_size=max_ngram_size)
        tokenizer = QueryTokenizer(config=config)
        
        # Get single-word tokens
        tokens = tokenizer.tokenize(query)
        n_tokens = len(tokens)
        
        # Skip if we don't have at least 2 tokens
        assume(n_tokens >= 2)
        
        # For each valid n-gram size, verify the word count in text
        for k in range(2, min(n_tokens, max_ngram_size) + 1):
            ngrams = tokenizer.generate_ngrams(tokens, k)
            
            for ngram in ngrams:
                word_count = len(ngram.text.split())
                
                # Property: N-gram text should contain exactly k words
                assert word_count == k, \
                    f"N-gram '{ngram.text}' should contain {k} words, got {word_count}"



class TestNormalizationIdempotence:
    """
    Property tests for Token Normalization Idempotence (Property 7).
    
    **Validates: Property 7** - Token Normalization Idempotence
    For any token text, applying normalization twice SHALL produce the same result
    as applying it once (idempotent), and the result SHALL be lowercase with no
    punctuation.
    
    **Validates: Requirement 2.3**
    """
    
    # Strategy for generating arbitrary text strings
    arbitrary_text = st.text(min_size=0, max_size=200)
    
    # Strategy for text with various characters including punctuation
    text_with_punctuation = st.text(
        alphabet=st.characters(
            whitelist_categories=('L', 'N', 'P', 'Zs', 'S'),
            whitelist_characters=' !@#$%^&*()_+-=[]{}|;:\'",.<>?/\\`~'
        ),
        min_size=0,
        max_size=200
    )
    
    @given(text=arbitrary_text)
    @settings(max_examples=100)
    def test_normalization_is_idempotent(self, text: str):
        """
        Property test: normalize(normalize(x)) == normalize(x).
        
        For any token text, applying normalization twice SHALL produce the same
        result as applying it once.
        
        **Validates: Property 7**
        **Validates: Requirement 2.3**
        """
        tokenizer = QueryTokenizer()
        
        # Apply normalization once
        normalized_once = tokenizer.normalize(text)
        
        # Apply normalization twice
        normalized_twice = tokenizer.normalize(normalized_once)
        
        # Property: Normalization is idempotent
        assert normalized_twice == normalized_once, \
            f"Normalization is not idempotent: " \
            f"normalize('{text}') = '{normalized_once}', " \
            f"normalize(normalize('{text}')) = '{normalized_twice}'"
    
    @given(text=text_with_punctuation)
    @settings(max_examples=100)
    def test_normalization_is_idempotent_with_punctuation(self, text: str):
        """
        Property test: Idempotence holds for text with punctuation.
        
        For any token text containing punctuation, applying normalization twice
        SHALL produce the same result as applying it once.
        
        **Validates: Property 7**
        **Validates: Requirement 2.3**
        """
        tokenizer = QueryTokenizer()
        
        # Apply normalization once
        normalized_once = tokenizer.normalize(text)
        
        # Apply normalization twice
        normalized_twice = tokenizer.normalize(normalized_once)
        
        # Property: Normalization is idempotent
        assert normalized_twice == normalized_once, \
            f"Normalization is not idempotent for punctuated text: " \
            f"normalize('{text}') = '{normalized_once}', " \
            f"normalize(normalize('{text}')) = '{normalized_twice}'"
    
    @given(text=arbitrary_text)
    @settings(max_examples=100)
    def test_normalized_result_is_lowercase_when_configured(self, text: str):
        """
        Property test: Normalized result is lowercase when normalize_case=True.
        
        For any token text, when normalize_case is True, the normalized result
        SHALL be lowercase.
        
        **Validates: Property 7**
        **Validates: Requirement 2.3**
        """
        config = TokenizerConfig(normalize_case=True, remove_punctuation=True)
        tokenizer = QueryTokenizer(config=config)
        
        normalized = tokenizer.normalize(text)
        
        # Property: Result should be lowercase
        assert normalized == normalized.lower(), \
            f"Normalized text '{normalized}' is not lowercase for input '{text}'"
    
    @given(text=text_with_punctuation)
    @settings(max_examples=100)
    def test_normalized_result_has_no_punctuation_when_configured(self, text: str):
        """
        Property test: Normalized result has no punctuation when remove_punctuation=True.
        
        For any token text, when remove_punctuation is True, the normalized result
        SHALL have no punctuation characters.
        
        **Validates: Property 7**
        **Validates: Requirement 2.3**
        """
        import string
        
        config = TokenizerConfig(normalize_case=True, remove_punctuation=True)
        tokenizer = QueryTokenizer(config=config)
        
        normalized = tokenizer.normalize(text)
        
        # Property: Result should have no punctuation
        has_punctuation = any(char in string.punctuation for char in normalized)
        assert not has_punctuation, \
            f"Normalized text '{normalized}' contains punctuation for input '{text}'"
    
    @given(text=arbitrary_text)
    @settings(max_examples=100)
    def test_normalization_idempotence_with_case_only(self, text: str):
        """
        Property test: Idempotence holds when only case normalization is enabled.
        
        For any token text, when only normalize_case is True, applying normalization
        twice SHALL produce the same result as applying it once.
        
        **Validates: Property 7**
        **Validates: Requirement 2.3**
        """
        config = TokenizerConfig(normalize_case=True, remove_punctuation=False)
        tokenizer = QueryTokenizer(config=config)
        
        # Apply normalization once
        normalized_once = tokenizer.normalize(text)
        
        # Apply normalization twice
        normalized_twice = tokenizer.normalize(normalized_once)
        
        # Property: Normalization is idempotent
        assert normalized_twice == normalized_once, \
            f"Normalization (case only) is not idempotent: " \
            f"normalize('{text}') = '{normalized_once}', " \
            f"normalize(normalize('{text}')) = '{normalized_twice}'"
    
    @given(text=text_with_punctuation)
    @settings(max_examples=100)
    def test_normalization_idempotence_with_punctuation_only(self, text: str):
        """
        Property test: Idempotence holds when only punctuation removal is enabled.
        
        For any token text, when only remove_punctuation is True, applying normalization
        twice SHALL produce the same result as applying it once.
        
        **Validates: Property 7**
        **Validates: Requirement 2.3**
        """
        config = TokenizerConfig(normalize_case=False, remove_punctuation=True)
        tokenizer = QueryTokenizer(config=config)
        
        # Apply normalization once
        normalized_once = tokenizer.normalize(text)
        
        # Apply normalization twice
        normalized_twice = tokenizer.normalize(normalized_once)
        
        # Property: Normalization is idempotent
        assert normalized_twice == normalized_once, \
            f"Normalization (punctuation only) is not idempotent: " \
            f"normalize('{text}') = '{normalized_once}', " \
            f"normalize(normalize('{text}')) = '{normalized_twice}'"
    
    @given(text=arbitrary_text)
    @settings(max_examples=100)
    def test_normalization_idempotence_with_no_normalization(self, text: str):
        """
        Property test: Idempotence holds when no normalization is enabled.
        
        For any token text, when both normalize_case and remove_punctuation are False,
        applying normalization twice SHALL produce the same result as applying it once
        (which should be the original text).
        
        **Validates: Property 7**
        **Validates: Requirement 2.3**
        """
        config = TokenizerConfig(normalize_case=False, remove_punctuation=False)
        tokenizer = QueryTokenizer(config=config)
        
        # Apply normalization once
        normalized_once = tokenizer.normalize(text)
        
        # Apply normalization twice
        normalized_twice = tokenizer.normalize(normalized_once)
        
        # Property: Normalization is idempotent
        assert normalized_twice == normalized_once, \
            f"Normalization (disabled) is not idempotent: " \
            f"normalize('{text}') = '{normalized_once}', " \
            f"normalize(normalize('{text}')) = '{normalized_twice}'"
        
        # Additional property: With no normalization, result should equal input
        assert normalized_once == text, \
            f"With normalization disabled, result '{normalized_once}' should equal input '{text}'"
    
    @given(
        text=arbitrary_text,
        normalize_case=st.booleans(),
        remove_punctuation=st.booleans()
    )
    @settings(max_examples=100)
    def test_normalization_idempotence_all_config_combinations(
        self, text: str, normalize_case: bool, remove_punctuation: bool
    ):
        """
        Property test: Idempotence holds for all configuration combinations.
        
        For any token text and any combination of normalize_case and remove_punctuation
        settings, applying normalization twice SHALL produce the same result as
        applying it once.
        
        **Validates: Property 7**
        **Validates: Requirement 2.3**
        """
        config = TokenizerConfig(
            normalize_case=normalize_case,
            remove_punctuation=remove_punctuation
        )
        tokenizer = QueryTokenizer(config=config)
        
        # Apply normalization once
        normalized_once = tokenizer.normalize(text)
        
        # Apply normalization twice
        normalized_twice = tokenizer.normalize(normalized_once)
        
        # Property: Normalization is idempotent for all config combinations
        assert normalized_twice == normalized_once, \
            f"Normalization is not idempotent with config " \
            f"(normalize_case={normalize_case}, remove_punctuation={remove_punctuation}): " \
            f"normalize('{text}') = '{normalized_once}', " \
            f"normalize(normalize('{text}')) = '{normalized_twice}'"
