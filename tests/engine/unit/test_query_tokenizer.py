"""
Unit tests for the QueryTokenizer class.

Tests the normalize() method and other tokenization functionality.
"""

import pytest
from glyphh.nl.query_tokenizer import QueryTokenizer, TokenizerConfig, Token


class TestQueryTokenizerNormalize:
    """Tests for the normalize() method.
    
    Validates: Requirement 2.3 - THE SDK SHALL normalize tokens (lowercase, remove punctuation) before vectorization
    """
    
    def test_normalize_lowercase(self):
        """Test that normalize converts text to lowercase when configured."""
        tokenizer = QueryTokenizer()
        assert tokenizer.normalize("HELLO") == "hello"
        assert tokenizer.normalize("Hello World") == "hello world"
        assert tokenizer.normalize("TOYOTA") == "toyota"
    
    def test_normalize_removes_punctuation(self):
        """Test that normalize removes punctuation when configured."""
        tokenizer = QueryTokenizer()
        assert tokenizer.normalize("hello!") == "hello"
        assert tokenizer.normalize("hello, world") == "hello world"
        assert tokenizer.normalize("what's up?") == "whats up"
        assert tokenizer.normalize("brake-pads") == "brakepads"
        assert tokenizer.normalize("test...") == "test"
    
    def test_normalize_combined(self):
        """Test that normalize applies both lowercase and punctuation removal."""
        tokenizer = QueryTokenizer()
        assert tokenizer.normalize("Hello, World!") == "hello world"
        assert tokenizer.normalize("Toyota's BRAKE-PADS") == "toyotas brakepads"
        assert tokenizer.normalize("What's UP?!") == "whats up"
    
    def test_normalize_preserves_spaces(self):
        """Test that normalize preserves spaces between words."""
        tokenizer = QueryTokenizer()
        assert tokenizer.normalize("hello world") == "hello world"
        assert tokenizer.normalize("  multiple   spaces  ") == "  multiple   spaces  "
    
    def test_normalize_empty_string(self):
        """Test that normalize handles empty strings."""
        tokenizer = QueryTokenizer()
        assert tokenizer.normalize("") == ""
    
    def test_normalize_only_punctuation(self):
        """Test that normalize handles strings with only punctuation."""
        tokenizer = QueryTokenizer()
        assert tokenizer.normalize("!!!") == ""
        assert tokenizer.normalize("...") == ""
        assert tokenizer.normalize("?!,") == ""
    
    def test_normalize_idempotent(self):
        """Test that normalize is idempotent: normalize(normalize(x)) == normalize(x).
        
        Validates: Property 7 - Token Normalization Idempotence
        """
        tokenizer = QueryTokenizer()
        test_cases = [
            "Hello, World!",
            "TOYOTA",
            "brake-pads",
            "What's up?",
            "already lowercase",
            "",
            "   spaces   ",
            "MiXeD CaSe!!!",
        ]
        for text in test_cases:
            once = tokenizer.normalize(text)
            twice = tokenizer.normalize(once)
            assert once == twice, f"normalize is not idempotent for '{text}': '{once}' != '{twice}'"
    
    def test_normalize_with_case_disabled(self):
        """Test normalize with normalize_case=False."""
        config = TokenizerConfig(normalize_case=False, remove_punctuation=True)
        tokenizer = QueryTokenizer(config=config)
        assert tokenizer.normalize("Hello, World!") == "Hello World"
        assert tokenizer.normalize("TOYOTA") == "TOYOTA"
    
    def test_normalize_with_punctuation_disabled(self):
        """Test normalize with remove_punctuation=False."""
        config = TokenizerConfig(normalize_case=True, remove_punctuation=False)
        tokenizer = QueryTokenizer(config=config)
        assert tokenizer.normalize("Hello, World!") == "hello, world!"
        assert tokenizer.normalize("TOYOTA") == "toyota"
    
    def test_normalize_with_both_disabled(self):
        """Test normalize with both options disabled."""
        config = TokenizerConfig(normalize_case=False, remove_punctuation=False)
        tokenizer = QueryTokenizer(config=config)
        assert tokenizer.normalize("Hello, World!") == "Hello, World!"
        assert tokenizer.normalize("TOYOTA") == "TOYOTA"
    
    def test_normalize_numbers_preserved(self):
        """Test that normalize preserves numbers."""
        tokenizer = QueryTokenizer()
        assert tokenizer.normalize("123") == "123"
        assert tokenizer.normalize("test123") == "test123"
        assert tokenizer.normalize("123test") == "123test"
        assert tokenizer.normalize("test 123 test") == "test 123 test"
    
    def test_normalize_unicode_letters(self):
        """Test that normalize handles unicode letters."""
        tokenizer = QueryTokenizer()
        # Unicode letters should be lowercased
        assert tokenizer.normalize("CAFÉ") == "café"
        assert tokenizer.normalize("Naïve") == "naïve"


class TestTokenDataclass:
    """Tests for the Token dataclass."""
    
    def test_token_creation(self):
        """Test basic Token creation."""
        token = Token(
            text="hello",
            original="Hello",
            position=0,
            is_stop_word=False,
            ngram_size=1
        )
        assert token.text == "hello"
        assert token.original == "Hello"
        assert token.position == 0
        assert token.is_stop_word is False
        assert token.ngram_size == 1
    
    def test_token_empty_text_raises(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Token text cannot be empty"):
            Token(text="", original="Hello", position=0, is_stop_word=False, ngram_size=1)
    
    def test_token_empty_original_raises(self):
        """Test that empty original raises ValueError."""
        with pytest.raises(ValueError, match="Token original text cannot be empty"):
            Token(text="hello", original="", position=0, is_stop_word=False, ngram_size=1)
    
    def test_token_negative_position_raises(self):
        """Test that negative position raises ValueError."""
        with pytest.raises(ValueError, match="Token position must be non-negative"):
            Token(text="hello", original="Hello", position=-1, is_stop_word=False, ngram_size=1)
    
    def test_token_invalid_ngram_size_raises(self):
        """Test that ngram_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Token ngram_size must be at least 1"):
            Token(text="hello", original="Hello", position=0, is_stop_word=False, ngram_size=0)


class TestTokenizerConfig:
    """Tests for the TokenizerConfig dataclass."""
    
    def test_default_config(self):
        """Test default TokenizerConfig values."""
        config = TokenizerConfig()
        assert config.max_ngram_size == 3
        assert config.stop_words == {"the", "a", "an", "is", "are"}
        assert config.normalize_case is True
        assert config.remove_punctuation is True
    
    def test_custom_config(self):
        """Test custom TokenizerConfig values."""
        config = TokenizerConfig(
            max_ngram_size=5,
            stop_words={"the", "a"},
            normalize_case=False,
            remove_punctuation=False
        )
        assert config.max_ngram_size == 5
        assert config.stop_words == {"the", "a"}
        assert config.normalize_case is False
        assert config.remove_punctuation is False
    
    def test_invalid_max_ngram_size_raises(self):
        """Test that max_ngram_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_ngram_size must be at least 1"):
            TokenizerConfig(max_ngram_size=0)
    
    def test_stop_words_list_converted_to_set(self):
        """Test that stop_words list is converted to set."""
        config = TokenizerConfig(stop_words=["the", "a", "an"])
        assert isinstance(config.stop_words, set)
        assert config.stop_words == {"the", "a", "an"}


class TestQueryTokenizerTokenize:
    """Tests for the tokenize() method.
    
    Validates: Requirements 2.1, 2.4, 2.5
    """
    
    def test_tokenize_simple_query(self):
        """Test tokenization of a simple query.
        
        Validates: Requirement 2.1 - THE SDK SHALL tokenize queries into individual words
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("Find Toyota brake pads")
        
        assert len(tokens) == 4
        assert tokens[0].text == "find"
        assert tokens[1].text == "toyota"
        assert tokens[2].text == "brake"
        assert tokens[3].text == "pads"
    
    def test_tokenize_preserves_original_text(self):
        """Test that tokenize preserves original text before normalization.
        
        Validates: Requirement 2.4 - THE SDK SHALL preserve original token positions
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("Find Toyota BRAKE-PADS")
        
        assert tokens[0].original == "Find"
        assert tokens[1].original == "Toyota"
        assert tokens[2].original == "BRAKE-PADS"
    
    def test_tokenize_preserves_positions(self):
        """Test that tokenize preserves character positions.
        
        Validates: Requirement 2.4 - THE SDK SHALL preserve original token positions for parameter extraction
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("Find Toyota brake pads")
        
        # "Find" starts at position 0
        assert tokens[0].position == 0
        # "Toyota" starts at position 5
        assert tokens[1].position == 5
        # "brake" starts at position 12
        assert tokens[2].position == 12
        # "pads" starts at position 18
        assert tokens[3].position == 18
    
    def test_tokenize_marks_stop_words(self):
        """Test that tokenize correctly marks stop words.
        
        Validates: Requirement 2.5 - THE SDK SHALL handle common stop words (the, a, an) appropriately
        """
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("the Toyota is a car")
        
        # "the" is a stop word
        assert tokens[0].text == "the"
        assert tokens[0].is_stop_word is True
        
        # "Toyota" is not a stop word
        assert tokens[1].text == "toyota"
        assert tokens[1].is_stop_word is False
        
        # "is" is a stop word
        assert tokens[2].text == "is"
        assert tokens[2].is_stop_word is True
        
        # "a" is a stop word
        assert tokens[3].text == "a"
        assert tokens[3].is_stop_word is True
        
        # "car" is not a stop word
        assert tokens[4].text == "car"
        assert tokens[4].is_stop_word is False
    
    def test_tokenize_all_tokens_have_ngram_size_1(self):
        """Test that all tokens from tokenize() have ngram_size=1."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("Find Toyota brake pads")
        
        for token in tokens:
            assert token.ngram_size == 1
    
    def test_tokenize_empty_query(self):
        """Test that tokenize handles empty queries."""
        tokenizer = QueryTokenizer()
        
        assert tokenizer.tokenize("") == []
        assert tokenizer.tokenize("   ") == []
        assert tokenizer.tokenize("\t\n") == []
    
    def test_tokenize_single_word(self):
        """Test tokenization of a single word."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("Toyota")
        
        assert len(tokens) == 1
        assert tokens[0].text == "toyota"
        assert tokens[0].original == "Toyota"
        assert tokens[0].position == 0
    
    def test_tokenize_with_punctuation(self):
        """Test tokenization with punctuation in words."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("Hello, World! What's up?")
        
        assert len(tokens) == 4
        assert tokens[0].text == "hello"
        assert tokens[1].text == "world"
        assert tokens[2].text == "whats"
        assert tokens[3].text == "up"
    
    def test_tokenize_skips_punctuation_only_tokens(self):
        """Test that tokenize skips tokens that become empty after normalization."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("Hello ... World")
        
        # "..." should be skipped as it becomes empty after normalization
        assert len(tokens) == 2
        assert tokens[0].text == "hello"
        assert tokens[1].text == "world"
    
    def test_tokenize_with_multiple_spaces(self):
        """Test tokenization with multiple spaces between words."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("Find   Toyota    brake")
        
        assert len(tokens) == 3
        assert tokens[0].text == "find"
        assert tokens[0].position == 0
        assert tokens[1].text == "toyota"
        assert tokens[1].position == 7  # After "Find   "
        assert tokens[2].text == "brake"
        assert tokens[2].position == 17  # After "Find   Toyota    "
    
    def test_tokenize_with_leading_trailing_spaces(self):
        """Test tokenization with leading and trailing spaces."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("  Find Toyota  ")
        
        assert len(tokens) == 2
        assert tokens[0].text == "find"
        assert tokens[0].position == 2  # After "  "
        assert tokens[1].text == "toyota"
        assert tokens[1].position == 7  # After "  Find "
    
    def test_tokenize_with_custom_stop_words(self):
        """Test tokenization with custom stop words."""
        config = TokenizerConfig(stop_words={"find", "show"})
        tokenizer = QueryTokenizer(config=config)
        tokens = tokenizer.tokenize("Find Toyota show cars")
        
        assert tokens[0].text == "find"
        assert tokens[0].is_stop_word is True
        
        assert tokens[1].text == "toyota"
        assert tokens[1].is_stop_word is False
        
        assert tokens[2].text == "show"
        assert tokens[2].is_stop_word is True
        
        assert tokens[3].text == "cars"
        assert tokens[3].is_stop_word is False
    
    def test_tokenize_with_numbers(self):
        """Test tokenization with numbers."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("Find 2023 Toyota")
        
        assert len(tokens) == 3
        assert tokens[0].text == "find"
        assert tokens[1].text == "2023"
        assert tokens[2].text == "toyota"
    
    def test_tokenize_normalizes_text(self):
        """Test that tokenize applies normalization."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("FIND Toyota's BRAKE-PADS")
        
        assert tokens[0].text == "find"
        assert tokens[1].text == "toyotas"
        assert tokens[2].text == "brakepads"
    
    def test_tokenize_with_normalization_disabled(self):
        """Test tokenization with normalization disabled."""
        config = TokenizerConfig(normalize_case=False, remove_punctuation=False)
        tokenizer = QueryTokenizer(config=config)
        tokens = tokenizer.tokenize("Find Toyota's")
        
        assert tokens[0].text == "Find"
        assert tokens[0].original == "Find"
        assert tokens[1].text == "Toyota's"
        assert tokens[1].original == "Toyota's"
    
    def test_tokenize_reconstructs_query(self):
        """Test that token originals can reconstruct the query structure.
        
        Validates: Property 5 - Query Tokenization Completeness
        """
        tokenizer = QueryTokenizer()
        query = "Find Toyota brake pads"
        tokens = tokenizer.tokenize(query)
        
        # Verify each original text appears at its position in the query
        for token in tokens:
            assert query[token.position:token.position + len(token.original)] == token.original


class TestQueryTokenizerGenerateNgrams:
    """Tests for the generate_ngrams() method.
    
    Validates: Requirement 2.2 - THE SDK SHALL generate n-grams (2-word, 3-word combinations) for compound value matching
    """
    
    def test_generate_bigrams(self):
        """Test generation of 2-word n-grams (bigrams)."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("find toyota brake pads")
        bigrams = tokenizer.generate_ngrams(tokens, 2)
        
        assert len(bigrams) == 3
        assert bigrams[0].text == "find toyota"
        assert bigrams[1].text == "toyota brake"
        assert bigrams[2].text == "brake pads"
    
    def test_generate_trigrams(self):
        """Test generation of 3-word n-grams (trigrams)."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("find toyota brake pads")
        trigrams = tokenizer.generate_ngrams(tokens, 3)
        
        assert len(trigrams) == 2
        assert trigrams[0].text == "find toyota brake"
        assert trigrams[1].text == "toyota brake pads"
    
    def test_ngram_preserves_original_text(self):
        """Test that n-grams preserve original text (space-separated)."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("Find Toyota BRAKE")
        bigrams = tokenizer.generate_ngrams(tokens, 2)
        
        assert bigrams[0].original == "Find Toyota"
        assert bigrams[1].original == "Toyota BRAKE"
    
    def test_ngram_position_is_first_token_position(self):
        """Test that n-gram position is the position of the first token."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("find toyota brake pads")
        bigrams = tokenizer.generate_ngrams(tokens, 2)
        
        # "find" is at position 0
        assert bigrams[0].position == 0
        # "toyota" is at position 5
        assert bigrams[1].position == 5
        # "brake" is at position 12
        assert bigrams[2].position == 12
    
    def test_ngram_is_not_stop_word(self):
        """Test that n-grams are never marked as stop words."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("the toyota is great")
        bigrams = tokenizer.generate_ngrams(tokens, 2)
        
        # Even though "the" and "is" are stop words, the n-grams are not
        for bigram in bigrams:
            assert bigram.is_stop_word is False
    
    def test_ngram_has_correct_ngram_size(self):
        """Test that n-grams have the correct ngram_size attribute."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("find toyota brake pads")
        
        bigrams = tokenizer.generate_ngrams(tokens, 2)
        for bigram in bigrams:
            assert bigram.ngram_size == 2
        
        trigrams = tokenizer.generate_ngrams(tokens, 3)
        for trigram in trigrams:
            assert trigram.ngram_size == 3
    
    def test_ngram_empty_tokens_returns_empty(self):
        """Test that generate_ngrams returns empty list for empty tokens."""
        tokenizer = QueryTokenizer()
        ngrams = tokenizer.generate_ngrams([], 2)
        assert ngrams == []
    
    def test_ngram_n_greater_than_tokens_returns_empty(self):
        """Test that generate_ngrams returns empty list when n > len(tokens)."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("find toyota")
        
        # Can't generate 3-grams from 2 tokens
        trigrams = tokenizer.generate_ngrams(tokens, 3)
        assert trigrams == []
        
        # Can't generate 5-grams from 2 tokens
        fivegrams = tokenizer.generate_ngrams(tokens, 5)
        assert fivegrams == []
    
    def test_ngram_n_equals_tokens_returns_single_ngram(self):
        """Test that generate_ngrams returns single n-gram when n == len(tokens)."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("find toyota")
        
        bigrams = tokenizer.generate_ngrams(tokens, 2)
        assert len(bigrams) == 1
        assert bigrams[0].text == "find toyota"
    
    def test_ngram_respects_max_ngram_size(self):
        """Test that generate_ngrams respects config.max_ngram_size."""
        config = TokenizerConfig(max_ngram_size=2)
        tokenizer = QueryTokenizer(config=config)
        tokens = tokenizer.tokenize("find toyota brake pads")
        
        # Bigrams should work (n=2 <= max_ngram_size=2)
        bigrams = tokenizer.generate_ngrams(tokens, 2)
        assert len(bigrams) == 3
        
        # Trigrams should return empty (n=3 > max_ngram_size=2)
        trigrams = tokenizer.generate_ngrams(tokens, 3)
        assert trigrams == []
    
    def test_ngram_n_less_than_1_returns_empty(self):
        """Test that generate_ngrams returns empty list for n < 1."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("find toyota brake pads")
        
        assert tokenizer.generate_ngrams(tokens, 0) == []
        assert tokenizer.generate_ngrams(tokens, -1) == []
    
    def test_ngram_unigrams(self):
        """Test that generate_ngrams can generate unigrams (n=1)."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("find toyota brake")
        
        unigrams = tokenizer.generate_ngrams(tokens, 1)
        assert len(unigrams) == 3
        assert unigrams[0].text == "find"
        assert unigrams[1].text == "toyota"
        assert unigrams[2].text == "brake"
        
        # Unigrams should have ngram_size=1
        for unigram in unigrams:
            assert unigram.ngram_size == 1
    
    def test_ngram_single_token(self):
        """Test generate_ngrams with a single token."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("toyota")
        
        # Unigram should work
        unigrams = tokenizer.generate_ngrams(tokens, 1)
        assert len(unigrams) == 1
        assert unigrams[0].text == "toyota"
        
        # Bigram should return empty
        bigrams = tokenizer.generate_ngrams(tokens, 2)
        assert bigrams == []
    
    def test_ngram_count_formula(self):
        """Test that the number of n-grams follows the formula: len(tokens) - n + 1."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("a b c d e")  # 5 tokens
        
        # For 5 tokens:
        # 1-grams: 5 - 1 + 1 = 5
        assert len(tokenizer.generate_ngrams(tokens, 1)) == 5
        # 2-grams: 5 - 2 + 1 = 4
        assert len(tokenizer.generate_ngrams(tokens, 2)) == 4
        # 3-grams: 5 - 3 + 1 = 3
        assert len(tokenizer.generate_ngrams(tokens, 3)) == 3
    
    def test_ngram_with_normalized_text(self):
        """Test that n-grams use normalized text."""
        tokenizer = QueryTokenizer()
        tokens = tokenizer.tokenize("FIND Toyota's BRAKE-PADS")
        bigrams = tokenizer.generate_ngrams(tokens, 2)
        
        # Text should be normalized (lowercase, no punctuation)
        assert bigrams[0].text == "find toyotas"
        assert bigrams[1].text == "toyotas brakepads"
        
        # Original should preserve original text
        assert bigrams[0].original == "FIND Toyota's"
        assert bigrams[1].original == "Toyota's BRAKE-PADS"
