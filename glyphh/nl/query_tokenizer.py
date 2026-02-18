"""
Query Tokenizer module for the Glyphh SDK.

This module provides tokenization functionality for natural language queries,
including n-gram support for compound value matching.

Classes:
    Token: Represents a single token from a query
    TokenizerConfig: Configuration for tokenization behavior
"""

from dataclasses import dataclass, field
import re
import string
from typing import List, Set


@dataclass
class Token:
    """A token from a query.
    
    Represents a single token extracted from a natural language query,
    preserving both the normalized and original text along with metadata
    about its position and characteristics.
    
    Attributes:
        text: Normalized token text (lowercase, no punctuation if configured)
        original: Original text before normalization
        position: Position in original query (0-indexed character position)
        is_stop_word: Whether this is a stop word (e.g., "the", "a", "an")
        ngram_size: 1 for single word, 2+ for n-grams (multi-word combinations)
    
    Example:
        >>> token = Token(
        ...     text="toyota",
        ...     original="Toyota",
        ...     position=5,
        ...     is_stop_word=False,
        ...     ngram_size=1
        ... )
        >>> token.text
        'toyota'
        >>> token.is_stop_word
        False
    
    Validates: Requirement 2 - Query Tokenization
    """
    text: str
    original: str
    position: int
    is_stop_word: bool
    ngram_size: int
    
    def __post_init__(self) -> None:
        """Validate token fields after initialization.
        
        Raises:
            ValueError: If text is empty
            ValueError: If original is empty
            ValueError: If position is negative
            ValueError: If ngram_size is less than 1
        """
        if not self.text:
            raise ValueError("Token text cannot be empty")
        if not self.original:
            raise ValueError("Token original text cannot be empty")
        if self.position < 0:
            raise ValueError(f"Token position must be non-negative, got {self.position}")
        if self.ngram_size < 1:
            raise ValueError(f"Token ngram_size must be at least 1, got {self.ngram_size}")


@dataclass
class TokenizerConfig:
    """Configuration for tokenization.
    
    Controls how queries are tokenized, including n-gram generation,
    stop word handling, and text normalization.
    
    Attributes:
        max_ngram_size: Maximum size of n-grams to generate (default: 3).
            For example, with max_ngram_size=3, the tokenizer will generate
            unigrams (1-word), bigrams (2-word), and trigrams (3-word).
        stop_words: Set of words to mark as stop words (default: common English
            stop words like "the", "a", "an", "is", "are"). Stop words are still
            tokenized but marked with is_stop_word=True.
        normalize_case: Whether to convert tokens to lowercase (default: True)
        remove_punctuation: Whether to remove punctuation from tokens (default: True)
    
    Example:
        >>> config = TokenizerConfig(
        ...     max_ngram_size=2,
        ...     stop_words={"the", "a"},
        ...     normalize_case=True,
        ...     remove_punctuation=True
        ... )
        >>> config.max_ngram_size
        2
    
    Validates: Requirement 2 - Query Tokenization
    """
    max_ngram_size: int = 3
    stop_words: Set[str] = field(default_factory=lambda: {"the", "a", "an", "is", "are"})
    normalize_case: bool = True
    remove_punctuation: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration fields after initialization.
        
        Raises:
            ValueError: If max_ngram_size is less than 1
            TypeError: If stop_words is not a set
        """
        if self.max_ngram_size < 1:
            raise ValueError(f"max_ngram_size must be at least 1, got {self.max_ngram_size}")
        if not isinstance(self.stop_words, set):
            # Convert to set if it's a list or other iterable
            if hasattr(self.stop_words, '__iter__'):
                self.stop_words = set(self.stop_words)
            else:
                raise TypeError(f"stop_words must be a set, got {type(self.stop_words).__name__}")


class QueryTokenizer:
    """Tokenizes queries with n-gram support.
    
    This class provides tokenization functionality for natural language queries,
    breaking them into individual words and generating n-grams for compound
    value matching. It handles stop words, normalization, and preserves
    original token positions for parameter extraction.
    
    The tokenizer is a key component in the auto-schema NL query matching
    pipeline, preparing queries for matching against schema vectors.
    
    Attributes:
        config: TokenizerConfig controlling tokenization behavior
    
    Example:
        >>> tokenizer = QueryTokenizer()
        >>> tokenizer.config.max_ngram_size
        3
        >>> tokenizer.config.stop_words
        {'the', 'a', 'an', 'is', 'are'}
        
        >>> # With custom config
        >>> custom_config = TokenizerConfig(
        ...     max_ngram_size=2,
        ...     stop_words={"the", "a"}
        ... )
        >>> tokenizer = QueryTokenizer(config=custom_config)
        >>> tokenizer.config.max_ngram_size
        2
    
    Validates: Requirement 2.5 - THE SDK SHALL handle common stop words (the, a, an) appropriately
    """
    
    def __init__(self, config: TokenizerConfig = None) -> None:
        """Initialize the QueryTokenizer with configuration.
        
        Creates a new QueryTokenizer instance with the specified configuration.
        If no configuration is provided, uses default settings:
        - Default stop words: the, a, an, is, are
        - Default max_ngram_size: 3
        - normalize_case: True
        - remove_punctuation: True
        
        Args:
            config: Optional TokenizerConfig to control tokenization behavior.
                If None, a default TokenizerConfig is created with standard
                English stop words and max_ngram_size of 3.
        
        Example:
            >>> # Default configuration
            >>> tokenizer = QueryTokenizer()
            >>> tokenizer.config.stop_words
            {'the', 'a', 'an', 'is', 'are'}
            >>> tokenizer.config.max_ngram_size
            3
            
            >>> # Custom configuration
            >>> config = TokenizerConfig(max_ngram_size=5)
            >>> tokenizer = QueryTokenizer(config=config)
            >>> tokenizer.config.max_ngram_size
            5
        
        Validates: Requirement 2.5 - THE SDK SHALL handle common stop words (the, a, an) appropriately
        """
        self.config = config or TokenizerConfig()

    def normalize(self, text: str) -> str:
        """Normalize token text.
        
        Applies normalization transformations to the input text based on
        the tokenizer configuration. This method is idempotent - applying
        it multiple times produces the same result as applying it once.
        
        Normalization steps (applied in order):
        1. Convert to lowercase if config.normalize_case is True
        2. Remove punctuation if config.remove_punctuation is True
        
        Args:
            text: The text to normalize
        
        Returns:
            The normalized text. If both normalize_case and remove_punctuation
            are False, returns the original text unchanged.
        
        Example:
            >>> tokenizer = QueryTokenizer()
            >>> tokenizer.normalize("Hello, World!")
            'hello world'
            >>> tokenizer.normalize("Toyota's")
            'toyotas'
            >>> tokenizer.normalize("BRAKE-PADS")
            'brakepads'
            
            >>> # Idempotence: normalize(normalize(x)) == normalize(x)
            >>> text = "Hello, World!"
            >>> tokenizer.normalize(tokenizer.normalize(text)) == tokenizer.normalize(text)
            True
            
            >>> # With normalization disabled
            >>> config = TokenizerConfig(normalize_case=False, remove_punctuation=False)
            >>> tokenizer = QueryTokenizer(config=config)
            >>> tokenizer.normalize("Hello, World!")
            'Hello, World!'
        
        Validates: Requirement 2.3 - THE SDK SHALL normalize tokens (lowercase, remove punctuation) before vectorization
        """
        result = text
        
        # Step 1: Convert to lowercase if configured
        if self.config.normalize_case:
            result = result.lower()
        
        # Step 2: Remove punctuation if configured
        # We replace punctuation with empty string to remove it
        if self.config.remove_punctuation:
            result = result.translate(str.maketrans('', '', string.punctuation))
        
        return result

    def tokenize(self, query: str) -> List[Token]:
        """Tokenize query into words.
        
        Splits the query into individual words and creates Token objects for each.
        This method only creates single-word tokens (ngram_size=1). N-gram generation
        is handled separately by the generate_ngrams() method.
        
        The tokenization process:
        1. Split the query into words using whitespace as delimiter
        2. For each word, create a Token with:
           - text: normalized text (lowercase, no punctuation if configured)
           - original: original text before normalization
           - position: character position in the original query
           - is_stop_word: True if normalized text is in config.stop_words
           - ngram_size: 1 (single word)
        
        Args:
            query: The natural language query to tokenize
        
        Returns:
            A list of Token objects representing each word in the query.
            Returns an empty list if the query is empty or contains only whitespace.
            Tokens with empty normalized text (e.g., punctuation-only words) are skipped.
        
        Example:
            >>> tokenizer = QueryTokenizer()
            >>> tokens = tokenizer.tokenize("Find Toyota brake pads")
            >>> len(tokens)
            4
            >>> tokens[0].text
            'find'
            >>> tokens[0].original
            'Find'
            >>> tokens[0].position
            0
            >>> tokens[1].text
            'toyota'
            >>> tokens[1].position
            5
            
            >>> # Stop words are marked
            >>> tokens = tokenizer.tokenize("the Toyota")
            >>> tokens[0].is_stop_word
            True
            >>> tokens[1].is_stop_word
            False
            
            >>> # Empty query returns empty list
            >>> tokenizer.tokenize("")
            []
            >>> tokenizer.tokenize("   ")
            []
        
        Validates: Requirements 2.1, 2.4, 2.5
        """
        if not query or not query.strip():
            return []
        
        tokens: List[Token] = []
        
        # Use regex to find all word sequences and their positions
        # This pattern matches sequences of non-whitespace characters
        for match in re.finditer(r'\S+', query):
            original = match.group()
            position = match.start()
            
            # Normalize the text
            normalized = self.normalize(original)
            
            # Skip tokens that become empty after normalization (e.g., punctuation-only)
            if not normalized:
                continue
            
            # Check if this is a stop word
            is_stop_word = normalized in self.config.stop_words
            
            # Create the token
            token = Token(
                text=normalized,
                original=original,
                position=position,
                is_stop_word=is_stop_word,
                ngram_size=1
            )
            tokens.append(token)
        
        return tokens

    def generate_ngrams(self, tokens: List[Token], n: int) -> List[Token]:
        """Generate n-grams from token list.
        
        Creates n-gram tokens by combining consecutive tokens from the input list.
        N-grams are multi-word combinations useful for matching compound values
        like "brake pads" or "Toyota Camry".
        
        The method respects the config.max_ngram_size setting - if n exceeds
        max_ngram_size, an empty list is returned.
        
        For each n-gram, a new Token is created with:
        - text: combined normalized text (space-separated)
        - original: combined original text (space-separated)
        - position: position of the first token in the n-gram
        - is_stop_word: False (n-grams are not considered stop words)
        - ngram_size: n
        
        Args:
            tokens: List of Token objects to generate n-grams from.
                These should typically be single-word tokens (ngram_size=1)
                from the tokenize() method.
            n: The size of n-grams to generate (e.g., 2 for bigrams, 3 for trigrams).
                Must be at least 1.
        
        Returns:
            A list of Token objects representing the n-grams.
            Returns an empty list if:
            - n > len(tokens) (not enough tokens to form n-grams)
            - n > config.max_ngram_size (exceeds configured maximum)
            - n < 1 (invalid n-gram size)
            - tokens is empty
        
        Example:
            >>> tokenizer = QueryTokenizer()
            >>> tokens = tokenizer.tokenize("find toyota brake pads")
            >>> bigrams = tokenizer.generate_ngrams(tokens, 2)
            >>> len(bigrams)
            3
            >>> bigrams[0].text
            'find toyota'
            >>> bigrams[0].original
            'find toyota'
            >>> bigrams[0].ngram_size
            2
            >>> bigrams[1].text
            'toyota brake'
            >>> bigrams[2].text
            'brake pads'
            
            >>> trigrams = tokenizer.generate_ngrams(tokens, 3)
            >>> len(trigrams)
            2
            >>> trigrams[0].text
            'find toyota brake'
            >>> trigrams[1].text
            'toyota brake pads'
            
            >>> # Not enough tokens for 5-grams
            >>> tokenizer.generate_ngrams(tokens, 5)
            []
            
            >>> # Respects max_ngram_size config
            >>> config = TokenizerConfig(max_ngram_size=2)
            >>> tokenizer = QueryTokenizer(config=config)
            >>> tokens = tokenizer.tokenize("find toyota brake pads")
            >>> tokenizer.generate_ngrams(tokens, 3)  # Exceeds max_ngram_size
            []
        
        Validates: Requirement 2.2 - THE SDK SHALL generate n-grams (2-word, 3-word combinations) for compound value matching
        """
        # Validate inputs
        if n < 1:
            return []
        
        if not tokens:
            return []
        
        # Respect max_ngram_size configuration
        if n > self.config.max_ngram_size:
            return []
        
        # Not enough tokens to form n-grams of size n
        if n > len(tokens):
            return []
        
        ngrams: List[Token] = []
        
        # Generate all consecutive n-word combinations
        # For n tokens, we can generate (len(tokens) - n + 1) n-grams
        for i in range(len(tokens) - n + 1):
            # Get the n consecutive tokens starting at position i
            ngram_tokens = tokens[i:i + n]
            
            # Combine the normalized text (space-separated)
            combined_text = " ".join(token.text for token in ngram_tokens)
            
            # Combine the original text (space-separated)
            combined_original = " ".join(token.original for token in ngram_tokens)
            
            # Position is the position of the first token in the n-gram
            position = ngram_tokens[0].position
            
            # Create the n-gram token
            ngram_token = Token(
                text=combined_text,
                original=combined_original,
                position=position,
                is_stop_word=False,  # N-grams are not considered stop words
                ngram_size=n
            )
            ngrams.append(ngram_token)
        
        return ngrams
