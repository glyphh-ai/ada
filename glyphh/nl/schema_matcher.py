from __future__ import annotations

"""
Schema Matcher module for the Glyphh SDK.

This module provides dataclasses and functionality for matching query tokens
against schema vectors using HDC similarity. It is a key component in the
auto-schema NL query matching pipeline.

Classes:
    TokenMatch: Represents a match between a token and a schema element
    MatchConfig: Configuration for matching thresholds and bonuses
    MatchResult: Complete result of matching a query against schema

The SchemaMatcher class (to be implemented in subsequent tasks) will use these
dataclasses to perform the actual matching operations.

Validates: Requirement 3 - Token-to-Schema Matching
Validates: Requirement 13.1 - THE SDK SHALL use approximate nearest neighbor search for large schema indexes
Validates: Requirement 13.6 - THE SDK SHALL optimize vector comparison using SIMD operations where available
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from glyphh.nl.query_tokenizer import Token
from glyphh.nl.schema_vectorizer import SchemaVector

if TYPE_CHECKING:
    from glyphh.nl.performance import ANNIndex

# Module logger
logger = logging.getLogger(__name__)


@dataclass
class TokenMatch:
    """
    A match between a token and a schema element.
    
    Represents a single match found when comparing a query token's vector
    against schema vectors. Contains the matched token, the schema element
    it matched, the similarity score, and the type of match.
    
    Attributes:
        token: The Token from the query that was matched.
            Contains the normalized text, original text, position, and
            other metadata about the token.
        schema_vector: The SchemaVector that the token matched against.
            Contains the key, vector, element_type (role/value), role_path,
            and original_value for value vectors.
        similarity: The similarity score between the token vector and
            schema vector. Should be in the range [-1.0, 1.0] for cosine
            similarity, with higher values indicating better matches.
        match_type: The type of match, one of:
            - "exact": Token text exactly matches the schema element
            - "partial": Token has high similarity but not exact match
            - "compound": Match was found using compound (multi-word) matching
    
    Example:
        >>> from glyphh.nl.query_tokenizer import Token
        >>> from glyphh.nl.schema_vectorizer import SchemaVector
        >>> 
        >>> # Create a token
        >>> token = Token(
        ...     text="toyota",
        ...     original="Toyota",
        ...     position=5,
        ...     is_stop_word=False,
        ...     ngram_size=1
        ... )
        >>> 
        >>> # Create a schema vector (simplified - vector would be actual HDC vector)
        >>> schema_vec = SchemaVector(
        ...     key="make=Toyota",
        ...     vector=some_vector,
        ...     element_type="value",
        ...     role_path="vehicle.identity.make",
        ...     original_value="Toyota"
        ... )
        >>> 
        >>> # Create a token match
        >>> match = TokenMatch(
        ...     token=token,
        ...     schema_vector=schema_vec,
        ...     similarity=0.85,
        ...     match_type="exact"
        ... )
        >>> match.similarity
        0.85
        >>> match.match_type
        'exact'
    
    Notes:
        - Similarity scores are typically computed using cosine similarity
        - The match_type helps distinguish between different matching strategies
        - Compound matches typically receive a bonus in scoring
        - Exact matches may also receive a bonus for precise text matching
    
    Validates: Requirement 3 - Token-to-Schema Matching
    """
    token: Token
    schema_vector: SchemaVector
    similarity: float
    match_type: str
    
    def __post_init__(self) -> None:
        """Validate TokenMatch fields after initialization.
        
        Validates that:
        - token is a Token instance
        - schema_vector is a SchemaVector instance
        - similarity is a valid float
        - match_type is one of the allowed values
        
        Raises:
            TypeError: If token is not a Token instance
            TypeError: If schema_vector is not a SchemaVector instance
            ValueError: If match_type is not one of "exact", "partial", "compound"
        """
        # Validate token type
        if not isinstance(self.token, Token):
            raise TypeError(
                f"token must be a Token instance, got {type(self.token).__name__}"
            )
        
        # Validate schema_vector type
        if not isinstance(self.schema_vector, SchemaVector):
            raise TypeError(
                f"schema_vector must be a SchemaVector instance, "
                f"got {type(self.schema_vector).__name__}"
            )
        
        # Validate similarity is a number (int or float)
        if not isinstance(self.similarity, (int, float)):
            raise TypeError(
                f"similarity must be a number, got {type(self.similarity).__name__}"
            )
        
        # Validate match_type is one of the allowed values
        valid_match_types = {"exact", "partial", "compound"}
        if self.match_type not in valid_match_types:
            raise ValueError(
                f"match_type must be one of {valid_match_types}, "
                f"got '{self.match_type}'"
            )
    
    def __repr__(self) -> str:
        """Return a string representation of the TokenMatch."""
        return (
            f"TokenMatch(token='{self.token.text}', "
            f"schema_key='{self.schema_vector.key}', "
            f"similarity={self.similarity:.4f}, "
            f"match_type='{self.match_type}')"
        )


@dataclass
class MatchConfig:
    """
    Configuration for matching.
    
    Controls the thresholds and bonuses used when matching query tokens
    against schema vectors. These settings allow tuning the precision
    and recall of the matching process.
    
    Attributes:
        role_threshold: Minimum similarity score required for a token to
            match a role vector. Matches below this threshold are excluded
            from results. Default: 0.3
        value_threshold: Minimum similarity score required for a token to
            match a value vector. Matches below this threshold are excluded
            from results. Default: 0.3
        compound_bonus: Bonus added to similarity scores for compound
            (multi-word) matches. This encourages preferring compound
            matches over individual token matches. Default: 0.1
        exact_match_bonus: Bonus added to similarity scores when the
            token text exactly matches the schema element text (case-
            insensitive). Default: 0.2
    
    Example:
        >>> # Default configuration
        >>> config = MatchConfig()
        >>> config.role_threshold
        0.3
        >>> config.value_threshold
        0.3
        >>> config.compound_bonus
        0.1
        >>> config.exact_match_bonus
        0.2
        
        >>> # Custom configuration for stricter matching
        >>> strict_config = MatchConfig(
        ...     role_threshold=0.5,
        ...     value_threshold=0.5,
        ...     compound_bonus=0.15,
        ...     exact_match_bonus=0.25
        ... )
        >>> strict_config.role_threshold
        0.5
        
        >>> # Custom configuration for more lenient matching
        >>> lenient_config = MatchConfig(
        ...     role_threshold=0.2,
        ...     value_threshold=0.2
        ... )
        >>> lenient_config.role_threshold
        0.2
    
    Notes:
        - Thresholds should be in the range [0.0, 1.0]
        - Lower thresholds increase recall but may decrease precision
        - Higher thresholds increase precision but may miss valid matches
        - Bonuses are added to the base similarity score
        - The compound_bonus helps prefer "brake pads" over separate
          "brake" and "pads" matches
        - The exact_match_bonus rewards precise text matching
    
    Validates: Requirement 3 - Token-to-Schema Matching
    Validates: Requirement 9 - Match Confidence Tuning
    """
    role_threshold: float = 0.3
    value_threshold: float = 0.3
    compound_bonus: float = 0.1
    exact_match_bonus: float = 0.2
    
    def __post_init__(self) -> None:
        """Validate MatchConfig fields after initialization.
        
        Validates that:
        - role_threshold is in range [0.0, 1.0]
        - value_threshold is in range [0.0, 1.0]
        - compound_bonus is non-negative
        - exact_match_bonus is non-negative
        
        Raises:
            ValueError: If role_threshold is not in range [0.0, 1.0]
            ValueError: If value_threshold is not in range [0.0, 1.0]
            ValueError: If compound_bonus is negative
            ValueError: If exact_match_bonus is negative
        """
        # Validate role_threshold is in valid range
        if not isinstance(self.role_threshold, (int, float)):
            raise TypeError(
                f"role_threshold must be a number, "
                f"got {type(self.role_threshold).__name__}"
            )
        if not 0.0 <= self.role_threshold <= 1.0:
            raise ValueError(
                f"role_threshold must be between 0.0 and 1.0, "
                f"got {self.role_threshold}"
            )
        
        # Validate value_threshold is in valid range
        if not isinstance(self.value_threshold, (int, float)):
            raise TypeError(
                f"value_threshold must be a number, "
                f"got {type(self.value_threshold).__name__}"
            )
        if not 0.0 <= self.value_threshold <= 1.0:
            raise ValueError(
                f"value_threshold must be between 0.0 and 1.0, "
                f"got {self.value_threshold}"
            )
        
        # Validate compound_bonus is non-negative
        if not isinstance(self.compound_bonus, (int, float)):
            raise TypeError(
                f"compound_bonus must be a number, "
                f"got {type(self.compound_bonus).__name__}"
            )
        if self.compound_bonus < 0:
            raise ValueError(
                f"compound_bonus must be non-negative, "
                f"got {self.compound_bonus}"
            )
        
        # Validate exact_match_bonus is non-negative
        if not isinstance(self.exact_match_bonus, (int, float)):
            raise TypeError(
                f"exact_match_bonus must be a number, "
                f"got {type(self.exact_match_bonus).__name__}"
            )
        if self.exact_match_bonus < 0:
            raise ValueError(
                f"exact_match_bonus must be non-negative, "
                f"got {self.exact_match_bonus}"
            )
    
    def __repr__(self) -> str:
        """Return a string representation of the MatchConfig."""
        return (
            f"MatchConfig(role_threshold={self.role_threshold}, "
            f"value_threshold={self.value_threshold}, "
            f"compound_bonus={self.compound_bonus}, "
            f"exact_match_bonus={self.exact_match_bonus})"
        )


@dataclass
class MatchResult:
    """
    Result of matching a query against schema.
    
    Contains the complete results of matching all tokens in a query against
    the schema vectors. Organizes matches by type (role, value, compound)
    and provides all candidates for disambiguation when needed.
    
    Attributes:
        query: The original query string that was matched.
        token_matches: All token matches found, regardless of type.
            This is the complete list of matches above threshold.
        role_matches: Matches where the token matched a role vector
            (schema_vector.element_type == "role").
        value_matches: Matches where the token matched a value vector
            (schema_vector.element_type == "value").
        compound_matches: Matches found using compound (multi-word)
            matching (match_type == "compound").
        all_candidates: All potential matches for disambiguation purposes.
            May include matches below threshold or alternative interpretations.
            Used when the query is ambiguous and multiple interpretations
            are possible.
    
    Example:
        >>> # Create a match result
        >>> result = MatchResult(
        ...     query="Find Toyota brake pads",
        ...     token_matches=[toyota_match, brake_pads_match],
        ...     role_matches=[],
        ...     value_matches=[toyota_match],
        ...     compound_matches=[brake_pads_match],
        ...     all_candidates=[toyota_match, brake_pads_match, brake_match, pads_match]
        ... )
        >>> result.query
        'Find Toyota brake pads'
        >>> len(result.value_matches)
        1
        >>> len(result.compound_matches)
        1
        
        >>> # Empty result for query with no matches
        >>> empty_result = MatchResult(
        ...     query="xyz123",
        ...     token_matches=[],
        ...     role_matches=[],
        ...     value_matches=[],
        ...     compound_matches=[],
        ...     all_candidates=[]
        ... )
        >>> len(empty_result.token_matches)
        0
    
    Notes:
        - token_matches contains all matches (union of role, value, compound)
        - role_matches and value_matches are mutually exclusive based on
          schema_vector.element_type
        - compound_matches may overlap with value_matches (a compound match
          is typically also a value match)
        - all_candidates is used for disambiguation and may include matches
          that were filtered out of the main results
        - The lists are not automatically populated - the SchemaMatcher
          is responsible for categorizing matches correctly
    
    Validates: Requirement 3 - Token-to-Schema Matching
    Validates: Requirement 12 - Query Disambiguation
    """
    query: str
    token_matches: List[TokenMatch] = field(default_factory=list)
    role_matches: List[TokenMatch] = field(default_factory=list)
    value_matches: List[TokenMatch] = field(default_factory=list)
    compound_matches: List[TokenMatch] = field(default_factory=list)
    all_candidates: List[TokenMatch] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate MatchResult fields after initialization.
        
        Validates that:
        - query is a string (can be empty)
        - All match lists contain only TokenMatch instances
        
        Raises:
            TypeError: If query is not a string
            TypeError: If any match list contains non-TokenMatch items
        """
        # Validate query is a string
        if not isinstance(self.query, str):
            raise TypeError(
                f"query must be a string, got {type(self.query).__name__}"
            )
        
        # Validate all match lists contain only TokenMatch instances
        match_lists = [
            ("token_matches", self.token_matches),
            ("role_matches", self.role_matches),
            ("value_matches", self.value_matches),
            ("compound_matches", self.compound_matches),
            ("all_candidates", self.all_candidates),
        ]
        
        for list_name, match_list in match_lists:
            if not isinstance(match_list, list):
                raise TypeError(
                    f"{list_name} must be a list, "
                    f"got {type(match_list).__name__}"
                )
            for i, item in enumerate(match_list):
                if not isinstance(item, TokenMatch):
                    raise TypeError(
                        f"{list_name}[{i}] must be a TokenMatch instance, "
                        f"got {type(item).__name__}"
                    )
    
    def __repr__(self) -> str:
        """Return a string representation of the MatchResult."""
        return (
            f"MatchResult(query='{self.query}', "
            f"token_matches={len(self.token_matches)}, "
            f"role_matches={len(self.role_matches)}, "
            f"value_matches={len(self.value_matches)}, "
            f"compound_matches={len(self.compound_matches)}, "
            f"all_candidates={len(self.all_candidates)})"
        )
    
    def has_matches(self) -> bool:
        """Check if any matches were found.
        
        Returns:
            True if token_matches is non-empty, False otherwise.
        
        Example:
            >>> result = MatchResult(query="test", token_matches=[some_match])
            >>> result.has_matches()
            True
            >>> empty_result = MatchResult(query="test")
            >>> empty_result.has_matches()
            False
        """
        return len(self.token_matches) > 0
    
    def get_best_match(self) -> TokenMatch | None:
        """Get the match with the highest similarity score.
        
        Returns:
            The TokenMatch with the highest similarity score from
            token_matches, or None if no matches exist.
        
        Example:
            >>> result = MatchResult(
            ...     query="test",
            ...     token_matches=[match1, match2, match3]
            ... )
            >>> best = result.get_best_match()
            >>> # Returns the match with highest similarity
        """
        if not self.token_matches:
            return None
        return max(self.token_matches, key=lambda m: m.similarity)


class SchemaMatcher:
    """
    Matches query tokens against schema vectors using HDC similarity.
    
    The SchemaMatcher is a key component in the auto-schema NL query matching
    pipeline. It takes tokenized query input and compares each token's vector
    against all schema vectors to find matches above a configurable threshold.
    
    The matcher uses the same encoder that was used to generate the schema
    vectors, ensuring that query tokens are encoded in the same vector space
    and can be meaningfully compared using cosine similarity.
    
    Attributes:
        schema_vectors: Dictionary mapping keys to SchemaVector objects.
            Keys are typically in the format "role_name" for role vectors
            or "role_name=value" for value vectors.
        encoder: The Encoder instance used to encode query tokens.
            Must be the same encoder (or compatible) used to generate
            the schema vectors.
        config: MatchConfig controlling thresholds and bonuses for matching.
            If not provided, uses default MatchConfig values.
    
    Example:
        >>> from glyphh.encoder.base import Encoder
        >>> from glyphh.core.config import EncoderConfig
        >>> from glyphh.nl.schema_vectorizer import SchemaVectorizer
        >>> 
        >>> # Create encoder and vectorizer
        >>> config = EncoderConfig(dimension=10000, seed=42)
        >>> encoder = Encoder(config)
        >>> vectorizer = SchemaVectorizer(encoder, config)
        >>> 
        >>> # Generate schema vectors
        >>> schema_vectors = vectorizer.get_schema_vectors()
        >>> 
        >>> # Create matcher with default config
        >>> matcher = SchemaMatcher(schema_vectors, encoder)
        >>> matcher.config.role_threshold
        0.3
        >>> 
        >>> # Create matcher with custom config
        >>> custom_config = MatchConfig(role_threshold=0.5, value_threshold=0.5)
        >>> matcher = SchemaMatcher(schema_vectors, encoder, custom_config)
        >>> matcher.config.role_threshold
        0.5
    
    Notes:
        - The encoder must be compatible with the one used to generate
          schema vectors (same dimension, seed, and space_id)
        - Schema vectors should be pre-generated using SchemaVectorizer
        - The matcher validates inputs during initialization
        - Subsequent tasks will implement match_query(), match_token(),
          compute_similarity(), and prefer_compound_matches() methods
    
    Validates: Requirement 3 - Token-to-Schema Matching
        THE SDK SHALL compute similarity between each token vector and all
        schema vectors.
    """
    
    def __init__(
        self,
        schema_vectors: dict[str, SchemaVector],
        encoder: 'Encoder',
        config: MatchConfig = None
    ):
        """
        Initialize the SchemaMatcher with schema vectors, encoder, and config.
        
        Stores references to the schema vectors and encoder, and initializes
        the match configuration. If no config is provided, creates a default
        MatchConfig with standard threshold values.
        
        Args:
            schema_vectors: Dictionary mapping keys to SchemaVector objects.
                Keys are typically "role_name" for role vectors or
                "role_name=value" for value vectors. Must be a dict.
            encoder: The Encoder instance used to encode query tokens.
                Must be an Encoder instance (or subclass). This encoder
                should be compatible with the one used to generate the
                schema vectors.
            config: Optional MatchConfig controlling thresholds and bonuses.
                If None, a default MatchConfig is created with:
                - role_threshold: 0.3
                - value_threshold: 0.3
                - compound_bonus: 0.1
                - exact_match_bonus: 0.2
        
        Raises:
            TypeError: If schema_vectors is not a dict
            TypeError: If encoder is not an Encoder instance
            TypeError: If config is provided but not a MatchConfig instance
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> # Setup
            >>> config = EncoderConfig(dimension=10000, seed=42)
            >>> encoder = Encoder(config)
            >>> schema_vectors = {}  # Would normally come from SchemaVectorizer
            >>> 
            >>> # Create with default config
            >>> matcher = SchemaMatcher(schema_vectors, encoder)
            >>> isinstance(matcher.config, MatchConfig)
            True
            >>> matcher.config.role_threshold
            0.3
            >>> 
            >>> # Create with custom config
            >>> custom_config = MatchConfig(
            ...     role_threshold=0.4,
            ...     value_threshold=0.4,
            ...     compound_bonus=0.15,
            ...     exact_match_bonus=0.25
            ... )
            >>> matcher = SchemaMatcher(schema_vectors, encoder, custom_config)
            >>> matcher.config.role_threshold
            0.4
            >>> 
            >>> # Validation errors
            >>> SchemaMatcher([], encoder)  # TypeError: not a dict
            >>> SchemaMatcher({}, "not_encoder")  # TypeError: not an Encoder
        
        Validates: Requirement 3 - Token-to-Schema Matching
        """
        # Import Encoder here to avoid circular imports
        from glyphh.encoder.base import Encoder
        
        # Validate schema_vectors is a dict
        if not isinstance(schema_vectors, dict):
            raise TypeError(
                f"schema_vectors must be a dict, "
                f"got {type(schema_vectors).__name__}"
            )
        
        # Validate encoder is an Encoder instance
        if not isinstance(encoder, Encoder):
            raise TypeError(
                f"encoder must be an Encoder instance, "
                f"got {type(encoder).__name__}"
            )
        
        # Validate config if provided
        if config is not None and not isinstance(config, MatchConfig):
            raise TypeError(
                f"config must be a MatchConfig instance or None, "
                f"got {type(config).__name__}"
            )
        
        # Store references
        self.schema_vectors = schema_vectors
        self.encoder = encoder
        
        # Initialize config with default if not provided
        self.config = config if config is not None else MatchConfig()
        
        # Initialize ANN index for large schemas
        # Validates: Requirement 13.1 - THE SDK SHALL use approximate nearest
        # neighbor search for large schema indexes
        self._ann_index: Optional['ANNIndex'] = None
        self._use_ann = False
        
        # Build ANN index if schema is large enough
        self._maybe_build_ann_index()
    
    def __repr__(self) -> str:
        """Return a string representation of the SchemaMatcher."""
        return (
            f"SchemaMatcher(schema_vectors={len(self.schema_vectors)}, "
            f"encoder_dim={self.encoder.dimension}, "
            f"config={self.config}, "
            f"use_ann={self._use_ann})"
        )
    
    def _maybe_build_ann_index(self) -> None:
        """
        Build ANN index if schema is large enough.
        
        For schemas with > 1000 vectors, builds an ANN index for faster
        nearest neighbor search. For smaller schemas, uses brute force
        search which is fast enough.
        
        Validates: Requirement 13.1 - THE SDK SHALL use approximate nearest
        neighbor search for large schema indexes
        """
        from glyphh.nl.performance import ANNIndex, should_use_ann_index
        
        # Check if we should use ANN index
        if not should_use_ann_index(len(self.schema_vectors)):
            self._use_ann = False
            self._ann_index = None
            return
        
        # Build ANN index
        if not self.schema_vectors:
            self._use_ann = False
            self._ann_index = None
            return
        
        # Get dimension from first vector
        first_key = next(iter(self.schema_vectors))
        first_vector = self.schema_vectors[first_key].vector
        if hasattr(first_vector, 'dimension'):
            dimension = first_vector.dimension
        else:
            dimension = len(first_vector.data if hasattr(first_vector, 'data') else first_vector)
        
        # Create and build index
        self._ann_index = ANNIndex(dimension=dimension)
        
        for key, schema_vector in self.schema_vectors.items():
            self._ann_index.add_vector(key, schema_vector.vector)
        
        metrics = self._ann_index.build()
        self._use_ann = True
        
        logger.info(
            "Built ANN index for SchemaMatcher: %d vectors, %.2f ms build time",
            metrics.num_vectors,
            metrics.build_time_ms
        )
    
    def rebuild_ann_index(self) -> None:
        """
        Rebuild the ANN index after schema vectors have changed.
        
        Call this method after adding or removing schema vectors to
        ensure the ANN index is up to date.
        """
        self._maybe_build_ann_index()
    
    def get_ann_metrics(self) -> Optional['ANNIndexMetrics']:
        """
        Get metrics for the ANN index.
        
        Returns:
            ANNIndexMetrics if ANN index is built, None otherwise.
        """
        if self._ann_index is not None:
            return self._ann_index.get_metrics()
        return None
    
    def compute_similarity(self, token_vector: 'Vector', schema_vector: 'Vector') -> float:
        """
        Compute cosine similarity between token and schema vectors.
        
        Computes the cosine similarity between two bipolar HDC vectors. For bipolar
        vectors (values in {-1, +1}), the cosine similarity simplifies to:
        
            similarity = dot_product / dimension
        
        This is because bipolar vectors have a magnitude of sqrt(dimension), so:
        
            cosine_similarity = dot(a, b) / (||a|| * ||b||)
                              = dot(a, b) / (sqrt(dim) * sqrt(dim))
                              = dot(a, b) / dimension
        
        The result is in the range [-1.0, 1.0]:
        - 1.0: Vectors are identical
        - 0.0: Vectors are orthogonal (uncorrelated)
        - -1.0: Vectors are opposite (anti-correlated)
        
        Args:
            token_vector: Vector representing the encoded query token.
                Must be a bipolar Vector with the same dimension and space_id
                as the encoder.
            schema_vector: Vector representing the schema element.
                Must be a bipolar Vector with the same dimension and space_id
                as the encoder.
        
        Returns:
            Cosine similarity as a float in the range [-1.0, 1.0].
            Higher values indicate greater similarity between the token
            and schema element.
        
        Raises:
            TypeError: If token_vector is not a Vector instance
            TypeError: If schema_vector is not a Vector instance
            ValueError: If vectors have different dimensions
            ValueError: If vectors have different space_ids
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> from glyphh.nl.schema_vectorizer import SchemaVectorizer
            >>> 
            >>> # Setup encoder and vectorizer
            >>> config = EncoderConfig(dimension=10000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Create schema vectors
            >>> schema_vectors = vectorizer.get_schema_vectors()
            >>> 
            >>> # Create matcher
            >>> matcher = SchemaMatcher(schema_vectors, encoder)
            >>> 
            >>> # Generate vectors for comparison
            >>> token_vec = encoder.generate_symbol("toyota")
            >>> schema_vec = encoder.generate_symbol("Toyota")
            >>> 
            >>> # Compute similarity
            >>> similarity = matcher.compute_similarity(token_vec, schema_vec)
            >>> print(f"Similarity: {similarity:.4f}")
            >>> 
            >>> # Identical vectors have similarity 1.0
            >>> same_vec = encoder.generate_symbol("test")
            >>> self_similarity = matcher.compute_similarity(same_vec, same_vec)
            >>> assert self_similarity == 1.0
            >>> 
            >>> # Different vectors typically have lower similarity
            >>> vec_a = encoder.generate_symbol("apple")
            >>> vec_b = encoder.generate_symbol("orange")
            >>> diff_similarity = matcher.compute_similarity(vec_a, vec_b)
            >>> assert -1.0 <= diff_similarity <= 1.0
        
        Notes:
            - Both vectors must be bipolar (values in {-1, +1})
            - Both vectors must have the same dimension
            - Both vectors must belong to the same vector space (same space_id)
            - The computation uses numpy's dot product for efficiency
            - For high-dimensional vectors (e.g., 10000), this is very fast
        
        Validates: Requirement 3.1 - THE SDK SHALL compute similarity between
        each token vector and all schema vectors
        """
        # Import Vector type for validation
        from glyphh.core.types import Vector
        
        # Validate token_vector is a Vector instance
        if not isinstance(token_vector, Vector):
            raise TypeError(
                f"token_vector must be a Vector instance, "
                f"got {type(token_vector).__name__}"
            )
        
        # Validate schema_vector is a Vector instance
        if not isinstance(schema_vector, Vector):
            raise TypeError(
                f"schema_vector must be a Vector instance, "
                f"got {type(schema_vector).__name__}"
            )
        
        # Validate vectors have the same dimension
        if token_vector.dimension != schema_vector.dimension:
            raise ValueError(
                f"Vectors must have the same dimension. "
                f"token_vector has dimension {token_vector.dimension}, "
                f"schema_vector has dimension {schema_vector.dimension}"
            )
        
        # Validate vectors are in the same vector space
        if token_vector.space_id != schema_vector.space_id:
            raise ValueError(
                f"Vectors must be in the same vector space. "
                f"token_vector has space_id '{token_vector.space_id}', "
                f"schema_vector has space_id '{schema_vector.space_id}'"
            )
        
        # Import numpy for efficient computation
        # Uses BLAS/LAPACK for SIMD acceleration
        # Validates: Requirement 13.6 - THE SDK SHALL optimize vector comparison
        # using SIMD operations where available
        import numpy as np
        
        # Compute dot product between the two vectors using SIMD-optimized operations
        # For bipolar vectors, this is the sum of element-wise products
        # Each product is either +1 (same sign) or -1 (different sign)
        # Using float64 for numerical precision, numpy's dot uses BLAS for SIMD
        dot_product = np.dot(
            token_vector.data.astype(np.float64),
            schema_vector.data.astype(np.float64)
        )
        
        # Compute cosine similarity for bipolar vectors
        # For bipolar vectors with values in {-1, +1}:
        # - Magnitude of each vector = sqrt(dimension)
        # - Cosine similarity = dot_product / (sqrt(dim) * sqrt(dim))
        #                     = dot_product / dimension
        dimension = token_vector.dimension
        similarity = dot_product / dimension
        
        # Return as Python float (not numpy scalar)
        return float(similarity)
    
    def match_token(self, token: Token) -> List[TokenMatch]:
        """
        Match single token against all schema vectors.
        
        Encodes the token text as an HDC vector and compares it against all
        schema vectors using cosine similarity. Returns matches above the
        configured threshold, with appropriate bonuses applied for exact matches.
        
        The method:
        1. Encodes the token text using the encoder
        2. Compares against all schema vectors using compute_similarity()
        3. Filters matches based on threshold (role_threshold for roles,
           value_threshold for values)
        4. Determines match_type ("exact" or "partial")
        5. Applies exact_match_bonus for exact text matches
        6. Returns matches sorted by similarity (descending)
        
        Args:
            token: The Token from the query to match against schema vectors.
                The token's text attribute is used for encoding and matching.
        
        Returns:
            List of TokenMatch objects for all matches above threshold,
            sorted by similarity score in descending order. Each TokenMatch
            contains:
            - token: The input token
            - schema_vector: The matched SchemaVector
            - similarity: The similarity score (with bonus if applicable)
            - match_type: "exact" if text matches exactly, "partial" otherwise
        
        Raises:
            TypeError: If token is not a Token instance
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> from glyphh.nl.schema_vectorizer import SchemaVectorizer
            >>> from glyphh.nl.query_tokenizer import Token
            >>> 
            >>> # Setup encoder and vectorizer
            >>> config = EncoderConfig(dimension=1000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Create schema vectors
            >>> vectorizer.vectorize_value("make", "Toyota")
            >>> vectorizer.vectorize_value("make", "Honda")
            >>> schema_vectors = vectorizer.get_schema_vectors()
            >>> 
            >>> # Create matcher
            >>> matcher = SchemaMatcher(schema_vectors, encoder)
            >>> 
            >>> # Create a token
            >>> token = Token(
            ...     text="toyota",
            ...     original="Toyota",
            ...     position=5,
            ...     is_stop_word=False,
            ...     ngram_size=1
            ... )
            >>> 
            >>> # Match token against schema
            >>> matches = matcher.match_token(token)
            >>> 
            >>> # Should find the Toyota value match
            >>> assert len(matches) > 0
            >>> assert matches[0].schema_vector.original_value == "Toyota"
            >>> assert matches[0].match_type == "exact"  # Case-insensitive match
            >>> 
            >>> # Matches are sorted by similarity (descending)
            >>> for i in range(len(matches) - 1):
            ...     assert matches[i].similarity >= matches[i + 1].similarity
        
        Notes:
            - The token text is encoded using the same encoder as the schema
              vectors, ensuring vectors are in the same space
            - Exact match detection is case-insensitive
            - For role vectors, the role name is compared against token text
            - For value vectors, the original_value is compared against token text
            - The exact_match_bonus is added to the base similarity score
            - Matches below threshold are excluded from results
            - An empty list is returned if no matches are above threshold
        
        Validates: Requirements 3.1, 3.2, 3.3, 3.4
        - 3.1: THE SDK SHALL compute similarity between each token vector and all schema vectors
        - 3.2: THE SDK SHALL return matches above a configurable threshold (default: 0.3)
        - 3.3: WHEN multiple schema vectors match a token, THE SDK SHALL return all matches with scores
        - 3.4: THE SDK SHALL distinguish between role matches and value matches
        """
        # Validate token is a Token instance
        if not isinstance(token, Token):
            raise TypeError(
                f"token must be a Token instance, got {type(token).__name__}"
            )
        
        # Encode the token text using the encoder
        # For n-gram tokens (ngram_size > 1), use compound vectorization to enable
        # word order permutation matching. This ensures "pads brake" produces the
        # same vector as "brake pads" because HDC bundling is commutative.
        # 
        # For single-word tokens, use generate_symbol directly.
        #
        # Validates: Requirement 10.5 - THE SDK SHALL handle word order variations
        # in compound values
        if token.ngram_size > 1:
            # N-gram token: split into words and use compound vectorization
            words = token.text.split()
            # Import SchemaVectorizer to use vectorize_compound
            from glyphh.nl.schema_vectorizer import SchemaVectorizer
            # Create a temporary vectorizer to use vectorize_compound
            # Note: We need to use the same encoder to ensure vectors are in the same space
            temp_vectorizer = SchemaVectorizer(self.encoder, self.encoder.config)
            token_vector = temp_vectorizer.vectorize_compound(words)
        else:
            # Single-word token: use generate_symbol directly
            token_vector = self.encoder.generate_symbol(token.text)
        
        # List to collect all matches above threshold
        matches: List[TokenMatch] = []
        
        # Compare token vector against all schema vectors
        for key, schema_vector in self.schema_vectors.items():
            # Compute cosine similarity between token and schema vectors
            similarity = self.compute_similarity(token_vector, schema_vector.vector)
            
            # Determine the appropriate threshold based on element type
            # Requirement 3.4: Distinguish between role matches and value matches
            if schema_vector.element_type == "role":
                threshold = self.config.role_threshold
            else:  # element_type == "value"
                threshold = self.config.value_threshold
            
            # Skip matches below threshold
            # Requirement 3.2: Return matches above configurable threshold
            if similarity < threshold:
                continue
            
            # Determine match_type by comparing token text with schema element
            # Case-insensitive comparison for exact match detection
            # For roles: compare with the role name (key)
            # For values: compare with the original_value
            if schema_vector.element_type == "role":
                # For role vectors, compare token text with role name
                schema_text = schema_vector.key
            else:
                # For value vectors, compare token text with original value
                schema_text = schema_vector.original_value
            
            # Case-insensitive exact match check
            is_exact_match = token.text.lower() == schema_text.lower()
            
            # Determine match_type
            match_type = "exact" if is_exact_match else "partial"
            
            # Apply exact_match_bonus if applicable
            # Requirement 3.5: Support weighted matching where exact matches score higher
            final_similarity = similarity
            if is_exact_match:
                final_similarity = similarity + self.config.exact_match_bonus
            
            # Create TokenMatch and add to results
            # Requirement 3.3: Return all matches with scores
            token_match = TokenMatch(
                token=token,
                schema_vector=schema_vector,
                similarity=final_similarity,
                match_type=match_type
            )
            matches.append(token_match)
        
        # Sort matches by similarity in descending order
        # Higher similarity scores come first
        matches.sort(key=lambda m: m.similarity, reverse=True)
        
        return matches
    
    def match_token_with_ann(self, token: Token, k: int = 50) -> List[TokenMatch]:
        """
        Match token against schema vectors using ANN index.
        
        Uses the ANN index for fast approximate nearest neighbor search.
        This is more efficient for large schemas (> 1000 vectors) while
        maintaining high accuracy.
        
        Args:
            token: The Token from the query to match.
            k: Number of nearest neighbors to retrieve from ANN index.
                Default is 50 to ensure we don't miss relevant matches.
        
        Returns:
            List of TokenMatch objects, sorted by similarity (descending).
        
        Notes:
            - Falls back to brute force if ANN index is not built
            - Uses minimum threshold for ANN search to filter results
        
        Validates: Requirement 13.1 - THE SDK SHALL use approximate nearest
        neighbor search for large schema indexes
        """
        # Fall back to brute force if ANN index is not available
        if not self._use_ann or self._ann_index is None:
            return self.match_token(token)
        
        # Validate token
        if not isinstance(token, Token):
            raise TypeError(
                f"token must be a Token instance, got {type(token).__name__}"
            )
        
        # Encode the token
        if token.ngram_size > 1:
            words = token.text.split()
            from glyphh.nl.schema_vectorizer import SchemaVectorizer
            temp_vectorizer = SchemaVectorizer(self.encoder, self.encoder.config)
            token_vector = temp_vectorizer.vectorize_compound(words)
        else:
            token_vector = self.encoder.generate_symbol(token.text)
        
        # Use minimum threshold for ANN search
        min_threshold = min(self.config.role_threshold, self.config.value_threshold)
        
        # Search using ANN index
        ann_results = self._ann_index.search(
            token_vector,
            k=k,
            threshold=min_threshold
        )
        
        # Convert ANN results to TokenMatch objects
        matches: List[TokenMatch] = []
        
        for ann_result in ann_results:
            key = ann_result.key
            if key not in self.schema_vectors:
                continue
            
            schema_vector = self.schema_vectors[key]
            similarity = ann_result.similarity
            
            # Apply threshold based on element type
            if schema_vector.element_type == "role":
                threshold = self.config.role_threshold
            else:
                threshold = self.config.value_threshold
            
            if similarity < threshold:
                continue
            
            # Determine match type
            if schema_vector.element_type == "role":
                schema_text = schema_vector.key
            else:
                schema_text = schema_vector.original_value
            
            is_exact_match = token.text.lower() == schema_text.lower()
            match_type = "exact" if is_exact_match else "partial"
            
            # Apply exact match bonus
            final_similarity = similarity
            if is_exact_match:
                final_similarity = similarity + self.config.exact_match_bonus
            
            token_match = TokenMatch(
                token=token,
                schema_vector=schema_vector,
                similarity=final_similarity,
                match_type=match_type
            )
            matches.append(token_match)
        
        # Sort by similarity
        matches.sort(key=lambda m: m.similarity, reverse=True)
        
        return matches
    
    def match_query(self, query: str, tokenizer: 'QueryTokenizer') -> MatchResult:
        """
        Match all tokens in query against schema.
        
        Tokenizes the query using the provided tokenizer, generates n-grams for
        compound matching, and matches all tokens (single words and n-grams)
        against schema vectors. Returns a MatchResult with categorized matches.
        
        The method:
        1. Tokenizes the query into single-word tokens
        2. Generates n-grams (2-word, 3-word, etc.) up to max_ngram_size
        3. Matches each token (single and n-gram) against schema vectors
        4. Categorizes matches into role_matches, value_matches, compound_matches
        5. Collects all matches for disambiguation purposes
        
        Args:
            query: The natural language query string to match against schema.
                Can be any non-empty string.
            tokenizer: A QueryTokenizer instance used to tokenize the query
                and generate n-grams. The tokenizer's config controls
                max_ngram_size and other tokenization settings.
        
        Returns:
            MatchResult containing:
            - query: The original query string
            - token_matches: All matches found (union of role, value, compound)
            - role_matches: Matches where schema_vector.element_type == "role"
            - value_matches: Matches where schema_vector.element_type == "value"
            - compound_matches: Matches from n-gram tokens (ngram_size > 1)
            - all_candidates: All matches for disambiguation purposes
        
        Raises:
            TypeError: If query is not a string
            TypeError: If tokenizer is not a QueryTokenizer instance
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> from glyphh.nl.schema_vectorizer import SchemaVectorizer
            >>> from glyphh.nl.query_tokenizer import QueryTokenizer
            >>> 
            >>> # Setup encoder and vectorizer
            >>> config = EncoderConfig(dimension=1000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Create schema vectors
            >>> vectorizer.vectorize_value("make", "toyota")
            >>> vectorizer.vectorize_value("category", "brake pads")
            >>> schema_vectors = vectorizer.get_schema_vectors()
            >>> 
            >>> # Create matcher and tokenizer
            >>> matcher = SchemaMatcher(schema_vectors, encoder)
            >>> tokenizer = QueryTokenizer()
            >>> 
            >>> # Match a query
            >>> result = matcher.match_query("Find Toyota brake pads", tokenizer)
            >>> 
            >>> # Check results
            >>> result.query
            'Find Toyota brake pads'
            >>> result.has_matches()
            True
            >>> len(result.value_matches) > 0
            True
            >>> 
            >>> # Compound matches for "brake pads"
            >>> compound = [m for m in result.compound_matches if "brake pads" in m.token.text]
            >>> len(compound) > 0  # If "brake pads" is in schema
            True
        
        Notes:
            - Single-word tokens are matched first, then n-grams
            - N-grams are generated for sizes 2 through max_ngram_size
            - Compound matches (n-gram matches) have match_type="compound"
            - Role matches and value matches are mutually exclusive based on
              schema_vector.element_type
            - all_candidates includes all matches for disambiguation
            - Empty queries return a MatchResult with empty match lists
        
        Validates: Requirement 3.6 - THE SDK SHALL provide a `match_query()` method
        that returns all token-schema matches
        """
        # Import QueryTokenizer here to avoid circular imports
        from glyphh.nl.query_tokenizer import QueryTokenizer
        
        # Validate query is a string
        if not isinstance(query, str):
            raise TypeError(
                f"query must be a string, got {type(query).__name__}"
            )
        
        # Validate tokenizer is a QueryTokenizer instance
        if not isinstance(tokenizer, QueryTokenizer):
            raise TypeError(
                f"tokenizer must be a QueryTokenizer instance, "
                f"got {type(tokenizer).__name__}"
            )
        
        # Initialize result lists
        all_token_matches: List[TokenMatch] = []
        role_matches: List[TokenMatch] = []
        value_matches: List[TokenMatch] = []
        compound_matches: List[TokenMatch] = []
        all_candidates: List[TokenMatch] = []
        
        # Step 1: Tokenize the query into single-word tokens
        single_tokens = tokenizer.tokenize(query)
        
        # Step 2: Match single-word tokens against schema
        for token in single_tokens:
            matches = self.match_token(token)
            
            for match in matches:
                # Add to all_token_matches
                all_token_matches.append(match)
                
                # Categorize by element_type
                if match.schema_vector.element_type == "role":
                    role_matches.append(match)
                else:  # element_type == "value"
                    value_matches.append(match)
                
                # Add to all_candidates for disambiguation
                all_candidates.append(match)
        
        # Step 3: Generate n-grams and match them
        # Generate n-grams for sizes 2 through max_ngram_size
        max_ngram_size = tokenizer.config.max_ngram_size
        
        for n in range(2, max_ngram_size + 1):
            ngram_tokens = tokenizer.generate_ngrams(single_tokens, n)
            
            for ngram_token in ngram_tokens:
                matches = self.match_token(ngram_token)
                
                for match in matches:
                    # Create a new TokenMatch with match_type="compound"
                    # since this is an n-gram match
                    compound_match = TokenMatch(
                        token=match.token,
                        schema_vector=match.schema_vector,
                        similarity=match.similarity + self.config.compound_bonus,
                        match_type="compound"
                    )
                    
                    # Add to all_token_matches
                    all_token_matches.append(compound_match)
                    
                    # Add to compound_matches
                    compound_matches.append(compound_match)
                    
                    # Also categorize by element_type
                    if compound_match.schema_vector.element_type == "role":
                        role_matches.append(compound_match)
                    else:  # element_type == "value"
                        value_matches.append(compound_match)
                    
                    # Add to all_candidates for disambiguation
                    all_candidates.append(compound_match)
        
        # Sort all match lists by similarity (descending)
        all_token_matches.sort(key=lambda m: m.similarity, reverse=True)
        role_matches.sort(key=lambda m: m.similarity, reverse=True)
        value_matches.sort(key=lambda m: m.similarity, reverse=True)
        compound_matches.sort(key=lambda m: m.similarity, reverse=True)
        all_candidates.sort(key=lambda m: m.similarity, reverse=True)
        
        # Create and return MatchResult
        return MatchResult(
            query=query,
            token_matches=all_token_matches,
            role_matches=role_matches,
            value_matches=value_matches,
            compound_matches=compound_matches,
            all_candidates=all_candidates
        )
    
    def prefer_compound_matches(self, matches: List[TokenMatch]) -> List[TokenMatch]:
        """
        Filter to prefer compound matches over individual token matches.
        
        When a compound match (ngram_size > 1) is found, this method suppresses
        individual token matches that are covered by the compound match. This
        ensures that "brake pads" as a compound match takes precedence over
        separate "brake" and "pads" individual matches.
        
        The method identifies which individual tokens are covered by compound
        matches based on their position in the original query. An individual
        token is considered "covered" if its position falls within the range
        of positions covered by a compound match.
        
        Args:
            matches: List of TokenMatch objects to filter. May contain both
                individual matches (ngram_size=1) and compound matches
                (ngram_size > 1).
        
        Returns:
            Filtered list of TokenMatch objects where:
            - All compound matches are preserved
            - Individual matches covered by compound matches are suppressed
            - Individual matches not covered by any compound match are preserved
        
        Example:
            >>> # Query: "Find Toyota brake pads"
            >>> # Tokens: ["find", "toyota", "brake", "pads", "brake pads"]
            >>> # If "brake pads" matches a schema value:
            >>> # - Keep "toyota" match (not covered by compound)
            >>> # - Keep "brake pads" compound match
            >>> # - Suppress "brake" individual match (covered by compound)
            >>> # - Suppress "pads" individual match (covered by compound)
            >>> 
            >>> matcher = SchemaMatcher(schema_vectors, encoder)
            >>> filtered = matcher.prefer_compound_matches(all_matches)
            >>> # filtered contains "toyota" and "brake pads" matches only
        
        Notes:
            - Position-based coverage: An individual token at position P is
              covered by a compound match if P falls within the compound's
              position range (from compound.position to compound.position +
              length of compound text)
            - Multiple compound matches: If multiple compound matches exist,
              individual tokens covered by ANY compound match are suppressed
            - Empty input: Returns empty list if input is empty
            - No compound matches: Returns all matches unchanged if no
              compound matches exist
            - Compound matches are identified by match_type="compound" or
              token.ngram_size > 1
        
        Validates: Requirements 10.2, 10.3
        - 10.2: WHEN a compound match is found, THE SDK SHALL prefer it over
                individual token matches
        - 10.3: THE SDK SHALL suppress individual token matches when a compound
                match covers those tokens
        """
        if not matches:
            return []
        
        # Separate compound matches from individual matches
        # Compound matches have ngram_size > 1 or match_type == "compound"
        compound_matches: List[TokenMatch] = []
        individual_matches: List[TokenMatch] = []
        
        for match in matches:
            if match.token.ngram_size > 1 or match.match_type == "compound":
                compound_matches.append(match)
            else:
                individual_matches.append(match)
        
        # If no compound matches, return all matches unchanged
        if not compound_matches:
            return matches
        
        # Build a set of positions covered by compound matches
        # For each compound match, we need to determine which individual
        # token positions it covers
        covered_positions: set[int] = set()
        
        for compound_match in compound_matches:
            # Get the compound token's position and text
            compound_token = compound_match.token
            compound_start = compound_token.position
            compound_text = compound_token.original
            
            # The compound covers positions from compound_start to
            # compound_start + len(compound_text) - 1
            # However, we need to track which individual token positions
            # are covered, not character positions
            
            # For position-based coverage, we consider an individual token
            # covered if its position falls within the compound's range
            compound_end = compound_start + len(compound_text)
            
            # Add all character positions in the compound's range
            for pos in range(compound_start, compound_end):
                covered_positions.add(pos)
        
        # Filter individual matches to exclude those covered by compound matches
        filtered_individual: List[TokenMatch] = []
        
        for individual_match in individual_matches:
            individual_token = individual_match.token
            individual_pos = individual_token.position
            individual_text = individual_token.original
            
            # Check if this individual token's position range overlaps with
            # any covered positions
            individual_end = individual_pos + len(individual_text)
            
            # An individual token is covered if ANY of its character positions
            # fall within the covered range
            is_covered = False
            for pos in range(individual_pos, individual_end):
                if pos in covered_positions:
                    is_covered = True
                    break
            
            # Only keep individual matches that are NOT covered
            if not is_covered:
                filtered_individual.append(individual_match)
        
        # Combine compound matches with non-covered individual matches
        result = compound_matches + filtered_individual
        
        # Sort by similarity (descending) to maintain consistent ordering
        result.sort(key=lambda m: m.similarity, reverse=True)
        
        return result
