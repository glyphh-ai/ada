"""
Schema Vectorizer for the Glyphh SDK.

This module provides the SchemaVector dataclass and SchemaVectorizer class
for generating bipolar HDC vectors for all roles and values in the model config.
These vectors are used to match query tokens against the schema.

The SchemaVectorizer uses the same encoder as the model to ensure vectors are
in the same vector space, enabling accurate similarity comparisons.
"""

import csv
from dataclasses import dataclass
import hashlib
import io
import json
from typing import Dict, List, Optional, Union

from glyphh.core.config import EncoderConfig
from glyphh.core.types import Vector
from glyphh.encoder.base import Encoder


@dataclass
class SchemaVector:
    """
    A vector representing a schema element.
    
    SchemaVector encapsulates a bipolar HDC vector generated from a role name
    or value in the model config. These vectors are used to match query tokens
    against the schema during NL query processing.
    
    Attributes:
        key: Unique identifier for this schema element.
            For roles: the role name (e.g., "make")
            For values: "role=value" format (e.g., "make=Toyota")
        vector: Bipolar HDC vector in {-1, +1} with the same dimension
            and space_id as the encoder.
        element_type: Type of schema element, either "role" or "value".
            - "role": Vector represents a role name
            - "value": Vector represents a specific value for a role
        role_path: Full hierarchical path to the role in the schema.
            Format: "layer.segment.role" (e.g., "vehicle.identity.make")
        original_value: The original value string for value vectors.
            None for role vectors.
    
    Example:
        >>> # Role vector
        >>> make_role = SchemaVector(
        ...     key="make",
        ...     vector=encoder.generate_symbol("make"),
        ...     element_type="role",
        ...     role_path="vehicle.identity.make",
        ...     original_value=None
        ... )
        >>> 
        >>> # Value vector
        >>> toyota_value = SchemaVector(
        ...     key="make=Toyota",
        ...     vector=encoder.generate_symbol("Toyota"),
        ...     element_type="value",
        ...     role_path="vehicle.identity.make",
        ...     original_value="Toyota"
        ... )
    
    Notes:
        - All vectors must be bipolar (values in {-1, +1})
        - Vectors must have the same dimension as the encoder
        - Vectors must have the same space_id as the encoder
        - element_type must be either "role" or "value"
    
    Validates: Requirement 1 - Schema Vector Generation
    """
    key: str
    vector: Vector
    element_type: str
    role_path: str
    original_value: Optional[str]
    
    def __post_init__(self):
        """Validate SchemaVector fields after initialization."""
        # Validate element_type
        if self.element_type not in ("role", "value"):
            raise ValueError(
                f"element_type must be 'role' or 'value', got '{self.element_type}'"
            )
        
        # Validate key is non-empty
        if not self.key:
            raise ValueError("key must be a non-empty string")
        
        # Validate role_path is non-empty
        if not self.role_path:
            raise ValueError("role_path must be a non-empty string")
        
        # Validate original_value consistency with element_type
        # Special case: alias role vectors can have original_value to store
        # the primary role name they map to
        if self.element_type == "role" and self.original_value is not None:
            if self.role_path != "alias":
                raise ValueError(
                    "original_value must be None for role vectors (except aliases)"
                )
        if self.element_type == "value" and self.original_value is None:
            raise ValueError(
                "original_value must be provided for value vectors"
            )
    
    def __repr__(self) -> str:
        """Return a string representation of the SchemaVector."""
        return (
            f"SchemaVector(key='{self.key}', element_type='{self.element_type}', "
            f"role_path='{self.role_path}', original_value={self.original_value!r})"
        )
    
    def __eq__(self, other) -> bool:
        """Check equality based on key and element_type."""
        if not isinstance(other, SchemaVector):
            return False
        return (
            self.key == other.key and
            self.element_type == other.element_type and
            self.role_path == other.role_path and
            self.original_value == other.original_value and
            self.vector == other.vector
        )
    
    def __hash__(self) -> int:
        """Compute hash for use in sets and dicts."""
        return hash((self.key, self.element_type, self.role_path, self.original_value))


class SchemaVectorizer:
    """
    Generates vectors for schema roles and values.
    
    The SchemaVectorizer creates bipolar HDC vectors for all roles and values
    in the model config. These vectors are used to match query tokens against
    the schema during NL query processing.
    
    Uses the same encoder as the model to ensure vectors are in the same space,
    enabling accurate similarity comparisons between query tokens and schema elements.
    
    Attributes:
        encoder: The Encoder instance used for vector generation.
            Must be the same encoder used by the model to ensure vectors
            are in the same vector space.
        config: The EncoderConfig containing the model schema definition.
            Defines layers, segments, and roles for vectorization.
        _schema_vectors: Cache of generated schema vectors.
            Maps schema element keys to their SchemaVector instances.
    
    Example:
        >>> from glyphh.encoder.base import Encoder
        >>> from glyphh.core.config import EncoderConfig, Layer, Segment, Role
        >>> 
        >>> # Create encoder config with schema
        >>> config = EncoderConfig(
        ...     dimension=10000,
        ...     seed=42,
        ...     layers=[
        ...         Layer(
        ...             name="vehicle",
        ...             segments=[
        ...                 Segment(
        ...                     name="identity",
        ...                     roles=[
        ...                         Role(name="make"),
        ...                         Role(name="model"),
        ...                     ]
        ...                 )
        ...             ]
        ...         )
        ...     ]
        ... )
        >>> 
        >>> # Create encoder and vectorizer
        >>> encoder = Encoder(config)
        >>> vectorizer = SchemaVectorizer(encoder, config)
        >>> 
        >>> # Access schema vectors (empty until vectorize_schema() is called)
        >>> vectors = vectorizer.get_schema_vectors()
        >>> print(len(vectors))
        0
    
    Notes:
        - The encoder and config must be compatible (same dimension, seed)
        - Schema vectors are cached in _schema_vectors for performance
        - Call vectorize_schema() to populate the cache
        - The vectorizer uses the same encoder configuration as the model
          to ensure vectors are in the same vector space
    
    Validates: Requirement 1.3 - WHEN generating schema vectors, THE SDK SHALL
    use the same encoder configuration as the model
    """
    
    def __init__(self, encoder: Encoder, config: EncoderConfig):
        """
        Initialize the SchemaVectorizer with encoder and config.
        
        Stores references to the encoder and config for use in vector generation.
        Initializes an empty cache for schema vectors.
        
        Args:
            encoder: The Encoder instance to use for generating vectors.
                Must be the same encoder used by the model to ensure
                vectors are in the same vector space.
            config: The EncoderConfig containing the model schema definition.
                Defines the layers, segments, and roles to vectorize.
        
        Raises:
            TypeError: If encoder is not an Encoder instance.
            TypeError: If config is not an EncoderConfig instance.
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> config = EncoderConfig(dimension=10000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Encoder and config are stored
            >>> assert vectorizer.encoder is encoder
            >>> assert vectorizer.config is config
            >>> 
            >>> # Schema vectors cache is empty initially
            >>> assert len(vectorizer._schema_vectors) == 0
        
        Validates: Requirement 1.3 - WHEN generating schema vectors, THE SDK SHALL
        use the same encoder configuration as the model
        """
        # Validate encoder type
        if not isinstance(encoder, Encoder):
            raise TypeError(
                f"encoder must be an Encoder instance, got {type(encoder).__name__}"
            )
        
        # Validate config type
        if not isinstance(config, EncoderConfig):
            raise TypeError(
                f"config must be an EncoderConfig instance, got {type(config).__name__}"
            )
        
        # Store encoder reference - ensures vectors are in the same space as the model
        self.encoder = encoder
        
        # Store config reference - contains schema definition (layers, segments, roles)
        self.config = config
        
        # Initialize empty schema vectors cache
        # Key: schema element key (e.g., "make" or "make=Toyota")
        # Value: SchemaVector instance
        self._schema_vectors: Dict[str, SchemaVector] = {}
        
        # Initialize config hash for cache invalidation
        # This hash is computed when vectorize_schema() is first called
        # and compared on subsequent calls to detect config changes
        # Validates: Requirements 1.4, 1.5
        self._config_hash: Optional[str] = None
        
        # Initialize synonyms dictionary for synonym → primary value mappings
        # Key: synonym string (e.g., "TYT")
        # Value: primary value string (e.g., "Toyota")
        # Validates: Requirements 11.2, 11.3
        self._synonyms: Dict[str, str] = {}
        
        # Initialize aliases dictionary for alias → primary role mappings
        # Key: alias string (e.g., "manufacturer")
        # Value: primary role string (e.g., "make")
        # Validates: Requirements 11.1, 11.3
        self._aliases: Dict[str, str] = {}
    
    def get_schema_vectors(self) -> Dict[str, SchemaVector]:
        """
        Return all cached schema vectors.
        
        Returns the dictionary of schema vectors that have been generated
        and cached. This will be empty until vectorize_schema() is called.
        
        Returns:
            Dictionary mapping schema element keys to SchemaVector instances.
            Keys are either role names (e.g., "make") or role=value pairs
            (e.g., "make=Toyota").
        
        Example:
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> vectors = vectorizer.get_schema_vectors()
            >>> print(len(vectors))
            0
            >>> 
            >>> # After vectorize_schema() is called, vectors will be populated
            >>> # vectorizer.vectorize_schema()
            >>> # vectors = vectorizer.get_schema_vectors()
            >>> # print(len(vectors))  # Will show number of schema elements
        """
        return self._schema_vectors

    def _compute_config_hash(self) -> str:
        """
        Compute a hash of the config for cache invalidation.
        
        Computes a SHA-256 hash of the config's JSON representation to detect
        when the config has changed. The hash includes:
        - The config's dimension and seed
        - The layers, segments, and roles structure
        - Any values lists defined in roles
        
        The JSON representation is sorted by keys to ensure deterministic
        hashing regardless of dictionary ordering.
        
        Returns:
            A hexadecimal string representing the SHA-256 hash of the config.
        
        Example:
            >>> config = EncoderConfig(dimension=1000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> hash1 = vectorizer._compute_config_hash()
            >>> hash2 = vectorizer._compute_config_hash()
            >>> assert hash1 == hash2  # Same config produces same hash
            >>> 
            >>> # Different config produces different hash
            >>> config2 = EncoderConfig(dimension=2000, seed=42)
            >>> encoder2 = Encoder(config2)
            >>> vectorizer2 = SchemaVectorizer(encoder2, config2)
            >>> hash3 = vectorizer2._compute_config_hash()
            >>> assert hash1 != hash3
        
        Notes:
            - Uses SHA-256 for collision resistance
            - JSON is sorted by keys for deterministic output
            - The hash is used to detect config changes for cache invalidation
        
        Validates: Requirements 1.4, 1.5
        - 1.4: THE SDK SHALL cache schema vectors to avoid regeneration
        - 1.5: THE SDK SHALL regenerate schema vectors when the model config changes
        """
        # Get the config as a dictionary
        config_dict = self.config.to_dict()
        
        # Convert to JSON with sorted keys for deterministic hashing
        config_json = json.dumps(config_dict, sort_keys=True)
        
        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(config_json.encode('utf-8'))
        
        return hash_obj.hexdigest()

    def vectorize_role(self, role_path: str) -> SchemaVector:
        """
        Generate vector for a single role name.
        
        Generates a bipolar HDC vector using the encoder. The vector is created
        from the full role path to ensure distinct vectors for nested paths.
        This supports paths of any depth (e.g., "a.b.c.d.e").
        
        Args:
            role_path: Full hierarchical path to the role in the schema.
                Format: "layer.segment.role" (e.g., "vehicle.identity.make")
                Can also be a simple role name (e.g., "make")
                Supports any depth (e.g., "a.b.c.d.e")
        
        Returns:
            SchemaVector with:
                - key: the role name (last component of path)
                - vector: bipolar HDC vector generated from the full path
                - element_type: "role"
                - role_path: the full path provided
                - original_value: None (roles don't have values)
        
        Raises:
            ValueError: If role_path is empty or contains only whitespace
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> config = EncoderConfig(dimension=10000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Generate vector for a role with full path
            >>> make_vector = vectorizer.vectorize_role("vehicle.identity.make")
            >>> print(make_vector.key)
            'make'
            >>> print(make_vector.element_type)
            'role'
            >>> print(make_vector.role_path)
            'vehicle.identity.make'
            >>> print(make_vector.original_value)
            None
            >>> 
            >>> # Vector is bipolar (all values in {-1, +1})
            >>> import numpy as np
            >>> assert np.all(np.isin(make_vector.vector.data, [-1, 1]))
            >>> 
            >>> # Vector has correct dimension
            >>> assert make_vector.vector.dimension == 10000
            >>> 
            >>> # Simple role name also works
            >>> model_vector = vectorizer.vectorize_role("model")
            >>> print(model_vector.key)
            'model'
            >>> print(model_vector.role_path)
            'model'
            >>> 
            >>> # Nested paths produce distinct vectors
            >>> path1 = vectorizer.vectorize_role("vehicle.identity.make")
            >>> path2 = vectorizer.vectorize_role("product.details.make")
            >>> # Different paths produce different vectors
            >>> assert not np.array_equal(path1.vector.data, path2.vector.data)
        
        Notes:
            - The role name is extracted as the last component of the path
              (after the last '.' if present)
            - The vector is generated from the FULL PATH to ensure distinct
              vectors for the same role name in different paths
            - This supports paths of any depth (e.g., "a.b.c.d.e")
            - The generated vector uses the same encoder as the model,
              ensuring vectors are in the same vector space
            - The vector is deterministic: same full path always produces
              the same vector for a given encoder configuration
        
        Validates: Requirements 1.1, 1.6
        - 1.1: THE SDK SHALL generate a bipolar vector for each role name
        - 1.6: THE SDK SHALL support generating vectors for nested role paths
        """
        # Validate role_path is not empty
        if not role_path or not role_path.strip():
            raise ValueError("role_path must be a non-empty string")
        
        # Extract the role name from the path (last component after '.')
        # e.g., "vehicle.identity.make" → "make"
        # e.g., "make" → "make"
        role_name = role_path.split('.')[-1]
        
        # Generate bipolar vector for the FULL PATH using the encoder
        # This ensures distinct vectors for nested paths (Requirement 1.6)
        # e.g., "vehicle.identity.make" and "product.details.make" will have
        # different vectors even though they share the same role name "make"
        # The encoder.generate_symbol() method ensures:
        # - Deterministic generation (same key → same vector)
        # - Bipolar constraint (all values in {-1, +1})
        # - Correct dimension matching encoder config
        # - Same space_id as the encoder
        vector = self.encoder.generate_symbol(role_path)
        
        # Create and return SchemaVector
        schema_vector = SchemaVector(
            key=role_name,
            vector=vector,
            element_type="role",
            role_path=role_path,
            original_value=None
        )
        
        return schema_vector

    def vectorize_value(self, role_path: str, value: str) -> SchemaVector:
        """
        Generate vector for a role-value pair.
        
        Extracts the role name from the role_path and generates a bipolar
        HDC vector for the value using the encoder. The vector is created
        for the value itself, enabling matching of query tokens against
        known values in the schema.
        
        Args:
            role_path: Full hierarchical path to the role in the schema.
                Format: "layer.segment.role" (e.g., "vehicle.identity.make")
                Can also be a simple role name (e.g., "make")
            value: The value string to vectorize (e.g., "Toyota", "Brake Pads")
        
        Returns:
            SchemaVector with:
                - key: "role=value" format (e.g., "make=Toyota")
                - vector: bipolar HDC vector generated from the value
                - element_type: "value"
                - role_path: the full path provided
                - original_value: the value string provided
        
        Raises:
            ValueError: If role_path is empty or contains only whitespace
            ValueError: If value is empty or contains only whitespace
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> config = EncoderConfig(dimension=10000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Generate vector for a value with full path
            >>> toyota_vector = vectorizer.vectorize_value("vehicle.identity.make", "Toyota")
            >>> print(toyota_vector.key)
            'make=Toyota'
            >>> print(toyota_vector.element_type)
            'value'
            >>> print(toyota_vector.role_path)
            'vehicle.identity.make'
            >>> print(toyota_vector.original_value)
            'Toyota'
            >>> 
            >>> # Vector is bipolar (all values in {-1, +1})
            >>> import numpy as np
            >>> assert np.all(np.isin(toyota_vector.vector.data, [-1, 1]))
            >>> 
            >>> # Vector has correct dimension
            >>> assert toyota_vector.vector.dimension == 10000
            >>> 
            >>> # Multi-word values also work
            >>> brake_pads = vectorizer.vectorize_value("parts.category", "Brake Pads")
            >>> print(brake_pads.key)
            'category=Brake Pads'
            >>> print(brake_pads.original_value)
            'Brake Pads'
        
        Notes:
            - The role name is extracted as the last component of the path
              (after the last '.' if present)
            - The key format is "role=value" for easy identification
            - The generated vector uses the same encoder as the model,
              ensuring vectors are in the same vector space
            - The vector is deterministic: same value always produces
              the same vector for a given encoder configuration
            - The vector is generated from the value string, not the role name
        
        Validates: Requirement 1.2 - THE SDK SHALL generate a bipolar vector
        for each unique value defined in the model config
        """
        # Validate role_path is not empty
        if not role_path or not role_path.strip():
            raise ValueError("role_path must be a non-empty string")
        
        # Validate value is not empty
        if not value or not value.strip():
            raise ValueError("value must be a non-empty string")
        
        # Extract the role name from the path (last component after '.')
        # e.g., "vehicle.identity.make" → "make"
        # e.g., "make" → "make"
        role_name = role_path.split('.')[-1]
        
        # Create the key in "role=value" format
        # e.g., "make=Toyota", "category=Brake Pads"
        key = f"{role_name}={value}"
        
        # Generate bipolar vector for the value using the encoder
        # For multi-word values, use compound vectorization (bundling) to enable
        # word order permutation matching. This ensures "Brake Pads" produces
        # the same vector as "Pads Brake" because HDC bundling is commutative.
        # 
        # For single-word values, use generate_symbol directly.
        # 
        # The encoder methods ensure:
        # - Deterministic generation (same key → same vector)
        # - Bipolar constraint (all values in {-1, +1})
        # - Correct dimension matching encoder config
        # - Same space_id as the encoder
        #
        # Validates: Requirement 10.5 - THE SDK SHALL handle word order variations
        # in compound values
        words = value.split()
        if len(words) > 1:
            # Multi-word value: use compound vectorization for order-independent matching
            vector = self.vectorize_compound(words)
        else:
            # Single-word value: use generate_symbol directly
            vector = self.encoder.generate_symbol(value)
        
        # Create and return SchemaVector
        schema_vector = SchemaVector(
            key=key,
            vector=vector,
            element_type="value",
            role_path=role_path,
            original_value=value
        )
        
        return schema_vector

    def vectorize_compound(self, words: List[str]) -> Vector:
        """
        Generate vector for multi-word compound value.
        
        Encodes each word as a bipolar HDC vector using the encoder, then
        bundles all word vectors together into a single compound vector.
        This enables matching of multi-word values like "Brake Pads" as
        single units rather than individual tokens.
        
        The bundling operation uses majority-vote aggregation, which is
        commutative (word order doesn't affect the result). This means
        "Brake Pads" and "Pads Brake" will produce the same compound vector.
        
        Args:
            words: List of words to combine into a compound vector.
                Each word should be a non-empty string.
                Example: ["Brake", "Pads"], ["Toyota", "Camry", "XLE"]
        
        Returns:
            Bundled bipolar Vector representing the compound value.
            The vector has the same dimension and space_id as the encoder.
        
        Raises:
            ValueError: If words list is empty
            ValueError: If any word in the list is empty or whitespace-only
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> config = EncoderConfig(dimension=1000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Generate compound vector for "Brake Pads"
            >>> compound_vector = vectorizer.vectorize_compound(["Brake", "Pads"])
            >>> 
            >>> # Vector is bipolar (all values in {-1, +1})
            >>> import numpy as np
            >>> assert np.all(np.isin(compound_vector.data, [-1, 1]))
            >>> 
            >>> # Vector has correct dimension
            >>> assert compound_vector.dimension == 1000
            >>> 
            >>> # Vector has correct space_id
            >>> assert compound_vector.space_id == encoder.space_id
            >>> 
            >>> # Word order doesn't affect result (bundling is commutative)
            >>> compound_vector_reversed = vectorizer.vectorize_compound(["Pads", "Brake"])
            >>> assert np.array_equal(compound_vector.data, compound_vector_reversed.data)
            >>> 
            >>> # Single word also works
            >>> single_word = vectorizer.vectorize_compound(["Toyota"])
            >>> assert single_word.dimension == 1000
        
        Notes:
            - The bundling operation is commutative, so word order doesn't matter
            - Each word is encoded using encoder.generate_symbol()
            - The bundle operation uses majority-vote aggregation
            - For a single word, the result is equivalent to generate_symbol(word)
            - Empty words or whitespace-only words are not allowed
        
        Validates: Requirement 10.1 - THE SDK SHALL generate vectors for
        multi-word values as single units
        """
        # Validate words list is not empty
        if not words:
            raise ValueError("words list cannot be empty")
        
        # Validate each word is non-empty
        for i, word in enumerate(words):
            if not word or not word.strip():
                raise ValueError(
                    f"word at index {i} must be a non-empty string"
                )
        
        # Generate a bipolar vector for each word using the encoder
        word_vectors = []
        for word in words:
            # Use generate_symbol to create a deterministic bipolar vector
            # for each word. This ensures:
            # - Deterministic generation (same word → same vector)
            # - Bipolar constraint (all values in {-1, +1})
            # - Correct dimension matching encoder config
            # - Same space_id as the encoder
            word_vector = self.encoder.generate_symbol(word)
            word_vectors.append(word_vector)
        
        # Bundle all word vectors together using majority-vote aggregation
        # The bundle operation is commutative, so word order doesn't affect
        # the result. This is useful for matching compound values where
        # word order may vary (e.g., "brake pads" vs "pads brake")
        compound_vector = self.encoder.bundle(word_vectors)
        
        return compound_vector

    def add_synonym(self, primary: str, synonym: str) -> None:
        """
        Add synonym vector for a value.
        
        Generates a bipolar HDC vector for the synonym and stores it in the
        schema vectors cache with a link to the primary value. When the synonym
        is matched during query processing, it will return the primary value.
        
        The synonym vector is stored with a key format of "synonym:synonym_text"
        to distinguish it from regular value vectors. The SchemaVector's
        original_value field stores the primary value, enabling the matcher
        to return the primary value when the synonym is matched.
        
        Args:
            primary: The primary value string that the synonym maps to.
                This is the canonical value in the schema (e.g., "Toyota").
            synonym: The synonym string to add (e.g., "TYT", "Toy").
                This is an alternative way to refer to the primary value.
        
        Raises:
            ValueError: If primary is empty or contains only whitespace.
            ValueError: If synonym is empty or contains only whitespace.
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> config = EncoderConfig(dimension=1000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Add a synonym for "Toyota"
            >>> vectorizer.add_synonym("Toyota", "TYT")
            >>> 
            >>> # The synonym vector is stored in schema vectors
            >>> vectors = vectorizer.get_schema_vectors()
            >>> assert "synonym:TYT" in vectors
            >>> 
            >>> # The synonym vector links to the primary value
            >>> synonym_vector = vectors["synonym:TYT"]
            >>> assert synonym_vector.original_value == "Toyota"
            >>> assert synonym_vector.element_type == "value"
            >>> 
            >>> # The synonym mapping is tracked
            >>> assert vectorizer.get_synonyms()["TYT"] == "Toyota"
            >>> 
            >>> # Multiple synonyms can be added for the same primary value
            >>> vectorizer.add_synonym("Toyota", "Toy")
            >>> assert "synonym:Toy" in vectorizer.get_schema_vectors()
            >>> 
            >>> # Vector is bipolar
            >>> import numpy as np
            >>> assert np.all(np.isin(synonym_vector.vector.data, [-1, 1]))
        
        Notes:
            - The synonym vector is generated from the synonym text itself,
              not the primary value. This allows queries containing the synonym
              to match against the synonym vector.
            - The original_value field stores the primary value, so when the
              synonym is matched, the matcher can return the primary value.
            - Synonyms are stored with a "synonym:" prefix in the key to
              distinguish them from regular value vectors.
            - The _synonyms dict tracks synonym → primary mappings for lookup.
            - Adding the same synonym twice will overwrite the previous entry.
        
        Validates: Requirements 11.2, 11.3
        - 11.2: THE SDK SHALL support defining synonyms for values
        - 11.3: WHEN generating schema vectors, THE SDK SHALL include
                alias/synonym vectors
        """
        # Validate primary is not empty
        if not primary or not primary.strip():
            raise ValueError("primary must be a non-empty string")
        
        # Validate synonym is not empty
        if not synonym or not synonym.strip():
            raise ValueError("synonym must be a non-empty string")
        
        # Generate bipolar vector for the synonym using the encoder
        # The vector is generated from the synonym text itself, so queries
        # containing the synonym will match against this vector
        synonym_vector = self.encoder.generate_symbol(synonym)
        
        # Create the key with "synonym:" prefix to distinguish from regular values
        key = f"synonym:{synonym}"
        
        # Create SchemaVector for the synonym
        # - element_type is "value" since synonyms represent values
        # - role_path is "synonym" to indicate this is a synonym entry
        # - original_value stores the PRIMARY value, enabling the matcher
        #   to return the primary value when the synonym is matched
        schema_vector = SchemaVector(
            key=key,
            vector=synonym_vector,
            element_type="value",
            role_path="synonym",
            original_value=primary  # Link to primary value
        )
        
        # Store the synonym vector in the schema vectors cache
        self._schema_vectors[key] = schema_vector
        
        # Track the synonym → primary mapping for lookup
        self._synonyms[synonym] = primary

    def get_synonyms(self) -> Dict[str, str]:
        """
        Return all synonym → primary value mappings.
        
        Returns the dictionary of synonyms that have been added via
        add_synonym(). This maps synonym strings to their primary values.
        
        Returns:
            Dictionary mapping synonym strings to primary value strings.
            Example: {"TYT": "Toyota", "Toy": "Toyota"}
        
        Example:
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> vectorizer.add_synonym("Toyota", "TYT")
            >>> vectorizer.add_synonym("Toyota", "Toy")
            >>> 
            >>> synonyms = vectorizer.get_synonyms()
            >>> assert synonyms["TYT"] == "Toyota"
            >>> assert synonyms["Toy"] == "Toyota"
        
        Validates: Requirements 11.2, 11.3
        """
        return self._synonyms

    def add_alias(self, role: str, alias: str) -> None:
        """
        Add alias vector for a role name.
        
        Generates a bipolar HDC vector for the alias and stores it in the
        schema vectors cache with a link to the primary role. When the alias
        is matched during query processing, it will return the primary role.
        
        The alias vector is stored with a key format of "alias:alias_text"
        to distinguish it from regular role vectors. The SchemaVector's
        original_value field stores the primary role name, enabling the matcher
        to return the primary role when the alias is matched.
        
        Args:
            role: The primary role name that the alias maps to.
                This is the canonical role in the schema (e.g., "make").
            alias: The alias string to add (e.g., "manufacturer", "brand").
                This is an alternative way to refer to the primary role.
        
        Raises:
            ValueError: If role is empty or contains only whitespace.
            ValueError: If alias is empty or contains only whitespace.
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> config = EncoderConfig(dimension=1000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Add an alias for "make"
            >>> vectorizer.add_alias("make", "manufacturer")
            >>> 
            >>> # The alias vector is stored in schema vectors
            >>> vectors = vectorizer.get_schema_vectors()
            >>> assert "alias:manufacturer" in vectors
            >>> 
            >>> # The alias vector links to the primary role
            >>> alias_vector = vectors["alias:manufacturer"]
            >>> assert alias_vector.original_value == "make"
            >>> assert alias_vector.element_type == "role"
            >>> 
            >>> # The alias mapping is tracked
            >>> assert vectorizer.get_aliases()["manufacturer"] == "make"
            >>> 
            >>> # Multiple aliases can be added for the same primary role
            >>> vectorizer.add_alias("make", "brand")
            >>> assert "alias:brand" in vectorizer.get_schema_vectors()
            >>> 
            >>> # Vector is bipolar
            >>> import numpy as np
            >>> assert np.all(np.isin(alias_vector.vector.data, [-1, 1]))
        
        Notes:
            - The alias vector is generated from the alias text itself,
              not the primary role. This allows queries containing the alias
              to match against the alias vector.
            - The original_value field stores the primary role name, so when the
              alias is matched, the matcher can return the primary role.
            - Aliases are stored with an "alias:" prefix in the key to
              distinguish them from regular role vectors.
            - The _aliases dict tracks alias → primary role mappings for lookup.
            - Adding the same alias twice will overwrite the previous entry.
        
        Validates: Requirements 11.1, 11.3
        - 11.1: THE SDK SHALL support defining aliases for role names
        - 11.3: WHEN generating schema vectors, THE SDK SHALL include
                alias/synonym vectors
        """
        # Validate role is not empty
        if not role or not role.strip():
            raise ValueError("role must be a non-empty string")
        
        # Validate alias is not empty
        if not alias or not alias.strip():
            raise ValueError("alias must be a non-empty string")
        
        # Generate bipolar vector for the alias using the encoder
        # The vector is generated from the alias text itself, so queries
        # containing the alias will match against this vector
        alias_vector = self.encoder.generate_symbol(alias)
        
        # Create the key with "alias:" prefix to distinguish from regular roles
        key = f"alias:{alias}"
        
        # Create SchemaVector for the alias
        # - element_type is "role" since aliases represent role names
        # - role_path is "alias" to indicate this is an alias entry
        # - original_value stores the PRIMARY role name, enabling the matcher
        #   to return the primary role when the alias is matched
        schema_vector = SchemaVector(
            key=key,
            vector=alias_vector,
            element_type="role",
            role_path="alias",
            original_value=role  # Link to primary role
        )
        
        # Store the alias vector in the schema vectors cache
        self._schema_vectors[key] = schema_vector
        
        # Track the alias → primary role mapping for lookup
        self._aliases[alias] = role

    def get_aliases(self) -> Dict[str, str]:
        """
        Return all alias → primary role mappings.
        
        Returns the dictionary of aliases that have been added via
        add_alias(). This maps alias strings to their primary role names.
        
        Returns:
            Dictionary mapping alias strings to primary role strings.
            Example: {"manufacturer": "make", "brand": "make"}
        
        Example:
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> vectorizer.add_alias("make", "manufacturer")
            >>> vectorizer.add_alias("make", "brand")
            >>> 
            >>> aliases = vectorizer.get_aliases()
            >>> assert aliases["manufacturer"] == "make"
            >>> assert aliases["brand"] == "make"
        
        Validates: Requirements 11.1, 11.3
        """
        return self._aliases

    def vectorize_schema(self) -> Dict[str, SchemaVector]:
        """
        Generate vectors for all roles and values in config.
        
        Iterates through the model config hierarchy (layers → segments → roles)
        and generates bipolar HDC vectors for:
        - Each role name in the schema
        - Each known value defined for roles (from values lists if available)
        
        All generated vectors are cached in `_schema_vectors` to avoid
        regeneration on subsequent calls. If the config changes (detected via
        hash comparison), the cache is cleared and vectors are regenerated.
        
        Returns:
            Dictionary mapping schema element keys to SchemaVector instances.
            Keys are either:
            - Role names (e.g., "make") for role vectors
            - "role=value" format (e.g., "make=Toyota") for value vectors
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig, Layer, Segment, Role
            >>> 
            >>> # Create config with schema
            >>> config = EncoderConfig(
            ...     dimension=1000,
            ...     seed=42,
            ...     layers=[
            ...         Layer(
            ...             name="vehicle",
            ...             segments=[
            ...                 Segment(
            ...                     name="identity",
            ...                     roles=[
            ...                         Role(name="make"),
            ...                         Role(name="model"),
            ...                     ]
            ...                 )
            ...             ]
            ...         )
            ...     ]
            ... )
            >>> 
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Generate vectors for all schema elements
            >>> vectors = vectorizer.vectorize_schema()
            >>> 
            >>> # Role vectors are generated
            >>> assert "make" in vectors
            >>> assert "model" in vectors
            >>> assert vectors["make"].element_type == "role"
            >>> assert vectors["model"].element_type == "role"
            >>> 
            >>> # Full paths are preserved
            >>> assert vectors["make"].role_path == "vehicle.identity.make"
            >>> assert vectors["model"].role_path == "vehicle.identity.model"
            >>> 
            >>> # Vectors are cached
            >>> vectors2 = vectorizer.vectorize_schema()
            >>> assert vectors is vectors2  # Same dict reference
        
        Notes:
            - The method iterates through: config.layers → layer.segments → segment.roles
            - For each role, a vector is generated using vectorize_role()
            - If a role has known values (via a 'values' attribute), vectors are
              generated for each value using vectorize_value()
            - Results are cached in _schema_vectors for performance
            - Subsequent calls return the cached dictionary without regeneration
            - The cache is invalidated when the config changes (detected via hash)
        
        Validates: Requirements 1.1, 1.2, 1.4, 1.5
        - 1.1: THE SDK SHALL generate a bipolar vector for each role name
        - 1.2: THE SDK SHALL generate a bipolar vector for each unique value
        - 1.4: THE SDK SHALL cache schema vectors to avoid regeneration
        - 1.5: THE SDK SHALL regenerate schema vectors when the model config changes
        """
        # Compute current config hash for change detection (Requirement 1.5)
        current_hash = self._compute_config_hash()
        
        # Check if config has changed since last vectorization
        # If hash differs, clear cache and regenerate vectors
        if self._config_hash is not None and self._config_hash != current_hash:
            # Config has changed - clear the cache to force regeneration
            self._schema_vectors.clear()
        
        # If vectors are already cached and config hasn't changed, return them
        # (Requirement 1.4)
        if self._schema_vectors:
            return self._schema_vectors
        
        # Store the current config hash for future comparisons
        self._config_hash = current_hash
        
        # Iterate through the config hierarchy: layers → segments → roles
        for layer in self.config.layers:
            for segment in layer.segments:
                for role in segment.roles:
                    # Build the full role path: "layer.segment.role"
                    role_path = f"{layer.name}.{segment.name}.{role.name}"
                    
                    # Generate vector for the role name (Requirement 1.1)
                    role_vector = self.vectorize_role(role_path)
                    
                    # Cache the role vector using the role name as key
                    # Note: If the same role name appears in multiple paths,
                    # the last one will be stored (same vector, different path)
                    self._schema_vectors[role_vector.key] = role_vector
                    
                    # Generate vectors for known values if the role has them
                    # (Requirement 1.2)
                    # Check if the role has a 'values' attribute with known values
                    if hasattr(role, 'values') and role.values:
                        for value in role.values:
                            # Skip empty or whitespace-only values
                            if value and str(value).strip():
                                value_vector = self.vectorize_value(
                                    role_path, str(value)
                                )
                                # Cache the value vector using "role=value" as key
                                self._schema_vectors[value_vector.key] = value_vector
        
        return self._schema_vectors

    def get_config_hash(self) -> Optional[str]:
        """
        Get the stored config hash used for cache invalidation.
        
        Returns the hash that was computed when vectorize_schema() was last
        called. This can be useful for debugging and testing cache behavior.
        
        Returns:
            The stored config hash string, or None if vectorize_schema()
            has not been called yet.
        
        Example:
            >>> config = EncoderConfig(dimension=1000, seed=42, layers=[...])
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Before vectorize_schema(), hash is None
            >>> assert vectorizer.get_config_hash() is None
            >>> 
            >>> # After vectorize_schema(), hash is set
            >>> vectorizer.vectorize_schema()
            >>> hash_value = vectorizer.get_config_hash()
            >>> assert hash_value is not None
            >>> assert isinstance(hash_value, str)
            >>> assert len(hash_value) == 64  # SHA-256 hex digest length
        
        Notes:
            - Returns None if vectorize_schema() has not been called
            - The hash is a 64-character hexadecimal string (SHA-256)
            - Same config always produces the same hash
        
        Validates: Requirements 1.4, 1.5
        - 1.4: THE SDK SHALL cache schema vectors to avoid regeneration
        - 1.5: THE SDK SHALL regenerate schema vectors when the model config changes
        """
        return self._config_hash

    def import_synonyms_from_csv(
        self, source: Union[str, io.IOBase]
    ) -> int:
        """
        Import synonyms from a CSV file or file-like object.
        
        Reads synonym definitions from a CSV source and adds them to the
        schema vectorizer using add_synonym(). The CSV must have columns
        'primary' and 'synonym' (header row required).
        
        CSV Format:
            primary,synonym
            Toyota,TYT
            Toyota,Toy
            Honda,HND
        
        Args:
            source: Either a file path (str) to a CSV file, or a file-like
                object (e.g., io.StringIO) containing CSV data.
        
        Returns:
            The count of successfully imported synonyms.
        
        Raises:
            ValueError: If the CSV is missing required columns ('primary', 'synonym').
            ValueError: If the CSV is empty (no data rows).
            FileNotFoundError: If the file path does not exist.
            csv.Error: If the CSV format is invalid.
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> import io
            >>> 
            >>> config = EncoderConfig(dimension=1000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Import from a file path
            >>> count = vectorizer.import_synonyms_from_csv("synonyms.csv")
            >>> print(f"Imported {count} synonyms")
            >>> 
            >>> # Import from a file-like object
            >>> csv_data = io.StringIO("primary,synonym\\nToyota,TYT\\nToyota,Toy")
            >>> count = vectorizer.import_synonyms_from_csv(csv_data)
            >>> assert count == 2
            >>> 
            >>> # Synonyms are now available
            >>> synonyms = vectorizer.get_synonyms()
            >>> assert synonyms["TYT"] == "Toyota"
            >>> assert synonyms["Toy"] == "Toyota"
        
        Notes:
            - The CSV must have a header row with 'primary' and 'synonym' columns.
            - Column order does not matter as long as both columns are present.
            - Empty rows are skipped.
            - Rows with empty primary or synonym values are skipped.
            - Each valid row calls add_synonym(primary, synonym).
            - Duplicate synonyms will overwrite previous entries.
        
        Validates: Requirement 11.6 - THE SDK SHALL support importing synonyms
        from external sources (CSV, JSON)
        """
        imported_count = 0
        
        # Handle file path vs file-like object
        if isinstance(source, str):
            # Source is a file path - open and read
            with open(source, 'r', newline='', encoding='utf-8') as f:
                imported_count = self._import_synonyms_from_csv_reader(f)
        else:
            # Source is a file-like object
            imported_count = self._import_synonyms_from_csv_reader(source)
        
        return imported_count

    def _import_synonyms_from_csv_reader(self, file_obj: io.IOBase) -> int:
        """
        Internal helper to import synonyms from a CSV reader.
        
        Args:
            file_obj: A file-like object containing CSV data.
        
        Returns:
            The count of successfully imported synonyms.
        
        Raises:
            ValueError: If required columns are missing or CSV is empty.
        """
        imported_count = 0
        
        # Create CSV reader with DictReader for column access by name
        reader = csv.DictReader(file_obj)
        
        # Validate required columns exist
        if reader.fieldnames is None:
            raise ValueError(
                "CSV is empty or has no header row. "
                "Required columns: 'primary', 'synonym'"
            )
        
        # Normalize fieldnames to lowercase for case-insensitive matching
        fieldnames_lower = [f.lower().strip() for f in reader.fieldnames]
        
        if 'primary' not in fieldnames_lower or 'synonym' not in fieldnames_lower:
            raise ValueError(
                f"CSV missing required columns. Found: {reader.fieldnames}. "
                "Required columns: 'primary', 'synonym'"
            )
        
        # Find the actual column names (preserving original case)
        primary_col = None
        synonym_col = None
        for fname in reader.fieldnames:
            if fname.lower().strip() == 'primary':
                primary_col = fname
            elif fname.lower().strip() == 'synonym':
                synonym_col = fname
        
        # Process each row
        has_data = False
        for row in reader:
            has_data = True
            primary = row.get(primary_col, '').strip()
            synonym = row.get(synonym_col, '').strip()
            
            # Skip rows with empty values
            if not primary or not synonym:
                continue
            
            # Add the synonym
            self.add_synonym(primary, synonym)
            imported_count += 1
        
        if not has_data:
            raise ValueError(
                "CSV has no data rows. At least one synonym entry is required."
            )
        
        return imported_count

    def import_synonyms_from_json(
        self, source: Union[str, io.IOBase]
    ) -> int:
        """
        Import synonyms from a JSON file or file-like object.
        
        Reads synonym definitions from a JSON source and adds them to the
        schema vectorizer using add_synonym(). The JSON must be an object
        mapping primary values to arrays of synonyms.
        
        JSON Format:
            {
                "Toyota": ["TYT", "Toy"],
                "Honda": ["HND"]
            }
        
        Args:
            source: Either a file path (str) to a JSON file, or a file-like
                object (e.g., io.StringIO) containing JSON data.
        
        Returns:
            The count of successfully imported synonyms.
        
        Raises:
            ValueError: If the JSON is not a valid object/dict.
            ValueError: If synonym values are not arrays/lists.
            ValueError: If the JSON is empty (no entries).
            FileNotFoundError: If the file path does not exist.
            json.JSONDecodeError: If the JSON format is invalid.
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> import io
            >>> 
            >>> config = EncoderConfig(dimension=1000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Import from a file path
            >>> count = vectorizer.import_synonyms_from_json("synonyms.json")
            >>> print(f"Imported {count} synonyms")
            >>> 
            >>> # Import from a file-like object
            >>> json_data = io.StringIO('{"Toyota": ["TYT", "Toy"], "Honda": ["HND"]}')
            >>> count = vectorizer.import_synonyms_from_json(json_data)
            >>> assert count == 3
            >>> 
            >>> # Synonyms are now available
            >>> synonyms = vectorizer.get_synonyms()
            >>> assert synonyms["TYT"] == "Toyota"
            >>> assert synonyms["Toy"] == "Toyota"
            >>> assert synonyms["HND"] == "Honda"
        
        Notes:
            - The JSON must be an object (dict) at the top level.
            - Keys are primary values, values are arrays of synonyms.
            - Empty arrays are allowed but contribute 0 to the count.
            - Empty string synonyms are skipped.
            - Each valid synonym calls add_synonym(primary, synonym).
            - Duplicate synonyms will overwrite previous entries.
        
        Validates: Requirement 11.6 - THE SDK SHALL support importing synonyms
        from external sources (CSV, JSON)
        """
        imported_count = 0
        
        # Handle file path vs file-like object
        if isinstance(source, str):
            # Source is a file path - open and read
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Source is a file-like object
            data = json.load(source)
        
        # Validate JSON structure
        if not isinstance(data, dict):
            raise ValueError(
                f"JSON must be an object/dict mapping primary values to synonym arrays. "
                f"Got: {type(data).__name__}"
            )
        
        if not data:
            raise ValueError(
                "JSON is empty. At least one primary value with synonyms is required."
            )
        
        # Process each primary value and its synonyms
        for primary, synonyms in data.items():
            # Validate primary is a non-empty string
            if not isinstance(primary, str) or not primary.strip():
                continue
            
            # Validate synonyms is a list
            if not isinstance(synonyms, list):
                raise ValueError(
                    f"Synonyms for '{primary}' must be an array/list. "
                    f"Got: {type(synonyms).__name__}"
                )
            
            # Add each synonym
            for synonym in synonyms:
                # Skip non-string or empty synonyms
                if not isinstance(synonym, str) or not synonym.strip():
                    continue
                
                self.add_synonym(primary.strip(), synonym.strip())
                imported_count += 1
        
        return imported_count

    def import_aliases_from_csv(
        self, source: Union[str, io.IOBase]
    ) -> int:
        """
        Import aliases from a CSV file or file-like object.
        
        Reads alias definitions from a CSV source and adds them to the
        schema vectorizer using add_alias(). The CSV must have columns
        'role' and 'alias' (header row required).
        
        CSV Format:
            role,alias
            make,manufacturer
            make,brand
            model,product_name
        
        Args:
            source: Either a file path (str) to a CSV file, or a file-like
                object (e.g., io.StringIO) containing CSV data.
        
        Returns:
            The count of successfully imported aliases.
        
        Raises:
            ValueError: If the CSV is missing required columns ('role', 'alias').
            ValueError: If the CSV is empty (no data rows).
            FileNotFoundError: If the file path does not exist.
            csv.Error: If the CSV format is invalid.
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> import io
            >>> 
            >>> config = EncoderConfig(dimension=1000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Import from a file path
            >>> count = vectorizer.import_aliases_from_csv("aliases.csv")
            >>> print(f"Imported {count} aliases")
            >>> 
            >>> # Import from a file-like object
            >>> csv_data = io.StringIO("role,alias\\nmake,manufacturer\\nmake,brand")
            >>> count = vectorizer.import_aliases_from_csv(csv_data)
            >>> assert count == 2
            >>> 
            >>> # Aliases are now available
            >>> aliases = vectorizer.get_aliases()
            >>> assert aliases["manufacturer"] == "make"
            >>> assert aliases["brand"] == "make"
        
        Notes:
            - The CSV must have a header row with 'role' and 'alias' columns.
            - Column order does not matter as long as both columns are present.
            - Empty rows are skipped.
            - Rows with empty role or alias values are skipped.
            - Each valid row calls add_alias(role, alias).
            - Duplicate aliases will overwrite previous entries.
        
        Validates: Requirement 11.6 - THE SDK SHALL support importing synonyms
        from external sources (CSV, JSON)
        """
        imported_count = 0
        
        # Handle file path vs file-like object
        if isinstance(source, str):
            # Source is a file path - open and read
            with open(source, 'r', newline='', encoding='utf-8') as f:
                imported_count = self._import_aliases_from_csv_reader(f)
        else:
            # Source is a file-like object
            imported_count = self._import_aliases_from_csv_reader(source)
        
        return imported_count

    def _import_aliases_from_csv_reader(self, file_obj: io.IOBase) -> int:
        """
        Internal helper to import aliases from a CSV reader.
        
        Args:
            file_obj: A file-like object containing CSV data.
        
        Returns:
            The count of successfully imported aliases.
        
        Raises:
            ValueError: If required columns are missing or CSV is empty.
        """
        imported_count = 0
        
        # Create CSV reader with DictReader for column access by name
        reader = csv.DictReader(file_obj)
        
        # Validate required columns exist
        if reader.fieldnames is None:
            raise ValueError(
                "CSV is empty or has no header row. "
                "Required columns: 'role', 'alias'"
            )
        
        # Normalize fieldnames to lowercase for case-insensitive matching
        fieldnames_lower = [f.lower().strip() for f in reader.fieldnames]
        
        if 'role' not in fieldnames_lower or 'alias' not in fieldnames_lower:
            raise ValueError(
                f"CSV missing required columns. Found: {reader.fieldnames}. "
                "Required columns: 'role', 'alias'"
            )
        
        # Find the actual column names (preserving original case)
        role_col = None
        alias_col = None
        for fname in reader.fieldnames:
            if fname.lower().strip() == 'role':
                role_col = fname
            elif fname.lower().strip() == 'alias':
                alias_col = fname
        
        # Process each row
        has_data = False
        for row in reader:
            has_data = True
            role = row.get(role_col, '').strip()
            alias = row.get(alias_col, '').strip()
            
            # Skip rows with empty values
            if not role or not alias:
                continue
            
            # Add the alias
            self.add_alias(role, alias)
            imported_count += 1
        
        if not has_data:
            raise ValueError(
                "CSV has no data rows. At least one alias entry is required."
            )
        
        return imported_count

    def import_aliases_from_json(
        self, source: Union[str, io.IOBase]
    ) -> int:
        """
        Import aliases from a JSON file or file-like object.
        
        Reads alias definitions from a JSON source and adds them to the
        schema vectorizer using add_alias(). The JSON must be an object
        mapping role names to arrays of aliases.
        
        JSON Format:
            {
                "make": ["manufacturer", "brand"],
                "model": ["product_name"]
            }
        
        Args:
            source: Either a file path (str) to a JSON file, or a file-like
                object (e.g., io.StringIO) containing JSON data.
        
        Returns:
            The count of successfully imported aliases.
        
        Raises:
            ValueError: If the JSON is not a valid object/dict.
            ValueError: If alias values are not arrays/lists.
            ValueError: If the JSON is empty (no entries).
            FileNotFoundError: If the file path does not exist.
            json.JSONDecodeError: If the JSON format is invalid.
        
        Example:
            >>> from glyphh.encoder.base import Encoder
            >>> from glyphh.core.config import EncoderConfig
            >>> import io
            >>> 
            >>> config = EncoderConfig(dimension=1000, seed=42)
            >>> encoder = Encoder(config)
            >>> vectorizer = SchemaVectorizer(encoder, config)
            >>> 
            >>> # Import from a file path
            >>> count = vectorizer.import_aliases_from_json("aliases.json")
            >>> print(f"Imported {count} aliases")
            >>> 
            >>> # Import from a file-like object
            >>> json_data = io.StringIO('{"make": ["manufacturer", "brand"], "model": ["product_name"]}')
            >>> count = vectorizer.import_aliases_from_json(json_data)
            >>> assert count == 3
            >>> 
            >>> # Aliases are now available
            >>> aliases = vectorizer.get_aliases()
            >>> assert aliases["manufacturer"] == "make"
            >>> assert aliases["brand"] == "make"
            >>> assert aliases["product_name"] == "model"
        
        Notes:
            - The JSON must be an object (dict) at the top level.
            - Keys are role names, values are arrays of aliases.
            - Empty arrays are allowed but contribute 0 to the count.
            - Empty string aliases are skipped.
            - Each valid alias calls add_alias(role, alias).
            - Duplicate aliases will overwrite previous entries.
        
        Validates: Requirement 11.6 - THE SDK SHALL support importing synonyms
        from external sources (CSV, JSON)
        """
        imported_count = 0
        
        # Handle file path vs file-like object
        if isinstance(source, str):
            # Source is a file path - open and read
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Source is a file-like object
            data = json.load(source)
        
        # Validate JSON structure
        if not isinstance(data, dict):
            raise ValueError(
                f"JSON must be an object/dict mapping role names to alias arrays. "
                f"Got: {type(data).__name__}"
            )
        
        if not data:
            raise ValueError(
                "JSON is empty. At least one role with aliases is required."
            )
        
        # Process each role and its aliases
        for role, aliases in data.items():
            # Validate role is a non-empty string
            if not isinstance(role, str) or not role.strip():
                continue
            
            # Validate aliases is a list
            if not isinstance(aliases, list):
                raise ValueError(
                    f"Aliases for '{role}' must be an array/list. "
                    f"Got: {type(aliases).__name__}"
                )
            
            # Add each alias
            for alias in aliases:
                # Skip non-string or empty aliases
                if not isinstance(alias, str) or not alias.strip():
                    continue
                
                self.add_alias(role.strip(), alias.strip())
                imported_count += 1
        
        return imported_count
