from __future__ import annotations

"""
Parameter Extractor module for the Glyphh SDK.

This module provides parameter extraction functionality for natural language queries,
extracting matched values as structured query parameters for GQL generation.

Classes:
    ExtractedParameter: An extracted parameter from the query
    ExtractionResult: Result of parameter extraction
    ParameterExtractor: Extracts parameters from matched tokens

The ParameterExtractor class uses the schema configuration to perform type inference
and extract structured parameters from natural language query matches.

Validates: Requirement 5 - Parameter Extraction
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from glyphh.nl.query_tokenizer import Token
    from glyphh.nl.schema_matcher import MatchResult
    from glyphh.core.config import EncoderConfig


# Valid value types supported by the system
VALID_VALUE_TYPES = {"string", "number", "boolean"}


@dataclass
class ExtractedParameter:
    """
    An extracted parameter from the query.
    
    Represents a single parameter extracted from a natural language query,
    including the role it belongs to, the extracted value, the original text
    from the query, a confidence score, and the value type.
    
    Attributes:
        role: Role name (e.g., "make", "category", "price"). This is the
            schema role that the extracted value belongs to. Must be a
            non-empty string.
        value: Extracted value. Can be any type (string, number, boolean)
            depending on the schema definition. The actual type should match
            the value_type attribute.
        original_text: Original text from the query that was matched to
            extract this parameter. Preserves the user's original wording
            for reference and debugging.
        confidence: Confidence score for this extraction, in the range
            [0.0, 1.0]. Higher values indicate greater confidence in the
            extraction accuracy.
            - 1.0: Very high confidence (exact match)
            - 0.7-0.9: High confidence (strong similarity)
            - 0.4-0.6: Medium confidence (partial match)
            - 0.1-0.3: Low confidence (weak match)
        value_type: The type of the extracted value. Must be one of:
            - "string": Text value (e.g., "Toyota", "Brake Pads")
            - "number": Numeric value (e.g., 100, 3.14)
            - "boolean": Boolean value (e.g., True, False)
    
    Example:
        >>> # String parameter extraction
        >>> param = ExtractedParameter(
        ...     role="make",
        ...     value="Toyota",
        ...     original_text="Toyota",
        ...     confidence=0.95,
        ...     value_type="string"
        ... )
        >>> param.role
        'make'
        >>> param.value
        'Toyota'
        >>> param.confidence
        0.95
        >>> param.value_type
        'string'
        
        >>> # Number parameter extraction
        >>> price_param = ExtractedParameter(
        ...     role="price",
        ...     value=25000,
        ...     original_text="25000",
        ...     confidence=0.9,
        ...     value_type="number"
        ... )
        >>> price_param.value_type
        'number'
        
        >>> # Boolean parameter extraction
        >>> available_param = ExtractedParameter(
        ...     role="available",
        ...     value=True,
        ...     original_text="available",
        ...     confidence=0.85,
        ...     value_type="boolean"
        ... )
        >>> available_param.value_type
        'boolean'
        
        >>> # Invalid role raises ValueError
        >>> ExtractedParameter(
        ...     role="",
        ...     value="Toyota",
        ...     original_text="Toyota",
        ...     confidence=0.9,
        ...     value_type="string"
        ... )
        Traceback (most recent call last):
            ...
        ValueError: role must be a non-empty string
        
        >>> # Invalid confidence raises ValueError
        >>> ExtractedParameter(
        ...     role="make",
        ...     value="Toyota",
        ...     original_text="Toyota",
        ...     confidence=1.5,
        ...     value_type="string"
        ... )
        Traceback (most recent call last):
            ...
        ValueError: confidence must be between 0.0 and 1.0, got 1.5
        
        >>> # Invalid value_type raises ValueError
        >>> ExtractedParameter(
        ...     role="make",
        ...     value="Toyota",
        ...     original_text="Toyota",
        ...     confidence=0.9,
        ...     value_type="invalid"
        ... )
        Traceback (most recent call last):
            ...
        ValueError: value_type must be one of {'string', 'number', 'boolean'}, got 'invalid'
    
    Notes:
        - The role must be a non-empty string matching a schema role
        - Confidence must be in the range [0.0, 1.0]
        - value_type must be one of "string", "number", or "boolean"
        - The value can be None if the extraction is uncertain
        - original_text preserves the user's exact wording
    
    Validates: Requirement 5 - Parameter Extraction
    Validates: Requirement 5.1 - THE SDK SHALL extract matched values as query parameters
    Validates: Requirement 5.2 - WHEN a value is matched, THE SDK SHALL associate it with its role
    Validates: Requirement 5.4 - THE SDK SHALL preserve value types (string, number, boolean) from the schema
    """
    role: str
    value: Any
    original_text: str
    confidence: float
    value_type: str
    
    def __post_init__(self) -> None:
        """Validate ExtractedParameter fields after initialization.
        
        Validates that:
        - role is a non-empty string
        - confidence is in the range [0.0, 1.0]
        - value_type is one of the valid types ("string", "number", "boolean")
        
        Raises:
            ValueError: If role is empty or not a string
            ValueError: If confidence is not in range [0.0, 1.0]
            ValueError: If value_type is not one of the valid types
            TypeError: If confidence is not a number
        """
        # Validate role is a non-empty string
        if not isinstance(self.role, str):
            raise TypeError(
                f"role must be a string, got {type(self.role).__name__}"
            )
        if not self.role:
            raise ValueError("role must be a non-empty string")
        
        # Validate confidence is a number
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(
                f"confidence must be a number, "
                f"got {type(self.confidence).__name__}"
            )
        
        # Validate confidence is in valid range [0.0, 1.0]
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, "
                f"got {self.confidence}"
            )
        
        # Validate value_type is one of the valid types
        if self.value_type not in VALID_VALUE_TYPES:
            raise ValueError(
                f"value_type must be one of {VALID_VALUE_TYPES}, "
                f"got '{self.value_type}'"
            )
    
    def __repr__(self) -> str:
        """Return a string representation of the ExtractedParameter."""
        return (
            f"ExtractedParameter("
            f"role='{self.role}', "
            f"value={self.value!r}, "
            f"original_text='{self.original_text}', "
            f"confidence={self.confidence:.4f}, "
            f"value_type='{self.value_type}')"
        )
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if this parameter has high confidence.
        
        Args:
            threshold: The confidence threshold for "high confidence".
                Default is 0.7.
        
        Returns:
            True if confidence >= threshold, False otherwise.
        
        Example:
            >>> param = ExtractedParameter(
            ...     role="make",
            ...     value="Toyota",
            ...     original_text="Toyota",
            ...     confidence=0.85,
            ...     value_type="string"
            ... )
            >>> param.is_high_confidence()
            True
            >>> param.is_high_confidence(threshold=0.9)
            False
        """
        return self.confidence >= threshold
    
    def is_string(self) -> bool:
        """Check if this parameter is a string type.
        
        Returns:
            True if value_type is "string", False otherwise.
        
        Example:
            >>> param = ExtractedParameter(
            ...     role="make",
            ...     value="Toyota",
            ...     original_text="Toyota",
            ...     confidence=0.9,
            ...     value_type="string"
            ... )
            >>> param.is_string()
            True
        """
        return self.value_type == "string"
    
    def is_number(self) -> bool:
        """Check if this parameter is a number type.
        
        Returns:
            True if value_type is "number", False otherwise.
        
        Example:
            >>> param = ExtractedParameter(
            ...     role="price",
            ...     value=25000,
            ...     original_text="25000",
            ...     confidence=0.9,
            ...     value_type="number"
            ... )
            >>> param.is_number()
            True
        """
        return self.value_type == "number"
    
    def is_boolean(self) -> bool:
        """Check if this parameter is a boolean type.
        
        Returns:
            True if value_type is "boolean", False otherwise.
        
        Example:
            >>> param = ExtractedParameter(
            ...     role="available",
            ...     value=True,
            ...     original_text="available",
            ...     confidence=0.9,
            ...     value_type="boolean"
            ... )
            >>> param.is_boolean()
            True
        """
        return self.value_type == "boolean"


@dataclass
class ExtractionResult:
    """
    Result of parameter extraction.
    
    Contains all extracted parameters from a query, including single-value
    parameters, multi-value parameters (for "X or Y" patterns), and any
    tokens that could not be matched to schema values.
    
    Attributes:
        parameters: Dictionary mapping role names to ExtractedParameter objects.
            Contains single-value parameters where each role has exactly one
            extracted value. For example: {"make": ExtractedParameter(...)}
        multi_value_params: Dictionary mapping role names to lists of
            ExtractedParameter objects. Used for "X or Y" patterns where
            multiple values are extracted for the same role.
            For example: {"make": [ExtractedParameter(value="Toyota"), 
                                   ExtractedParameter(value="Honda")]}
        unmatched_tokens: List of Token objects that could not be matched
            to any schema value. These tokens may represent unknown values,
            typos, or domain-specific terminology not in the schema.
    
    Example:
        >>> from glyphh.nl.query_tokenizer import Token
        >>> 
        >>> # Simple extraction result with single parameters
        >>> result = ExtractionResult(
        ...     parameters={
        ...         "make": ExtractedParameter(
        ...             role="make",
        ...             value="Toyota",
        ...             original_text="Toyota",
        ...             confidence=0.95,
        ...             value_type="string"
        ...         ),
        ...         "category": ExtractedParameter(
        ...             role="category",
        ...             value="Brake Pads",
        ...             original_text="brake pads",
        ...             confidence=0.9,
        ...             value_type="string"
        ...         )
        ...     },
        ...     multi_value_params={},
        ...     unmatched_tokens=[]
        ... )
        >>> "make" in result.parameters
        True
        >>> result.parameters["make"].value
        'Toyota'
        
        >>> # Extraction result with multi-value parameters
        >>> result_multi = ExtractionResult(
        ...     parameters={},
        ...     multi_value_params={
        ...         "make": [
        ...             ExtractedParameter(
        ...                 role="make",
        ...                 value="Toyota",
        ...                 original_text="Toyota",
        ...                 confidence=0.95,
        ...                 value_type="string"
        ...             ),
        ...             ExtractedParameter(
        ...                 role="make",
        ...                 value="Honda",
        ...                 original_text="Honda",
        ...                 confidence=0.92,
        ...                 value_type="string"
        ...             )
        ...         ]
        ...     },
        ...     unmatched_tokens=[]
        ... )
        >>> len(result_multi.multi_value_params["make"])
        2
        
        >>> # Extraction result with unmatched tokens
        >>> unmatched_token = Token(
        ...     text="xyz",
        ...     original="xyz",
        ...     position=10,
        ...     is_stop_word=False,
        ...     ngram_size=1
        ... )
        >>> result_unmatched = ExtractionResult(
        ...     parameters={},
        ...     multi_value_params={},
        ...     unmatched_tokens=[unmatched_token]
        ... )
        >>> len(result_unmatched.unmatched_tokens)
        1
    
    Notes:
        - A role should appear in either parameters OR multi_value_params, not both
        - unmatched_tokens can be used for error reporting or suggestions
        - Empty dictionaries/lists are valid for any field
        - The extraction result is typically created by ParameterExtractor
    
    Validates: Requirement 5 - Parameter Extraction
    Validates: Requirement 5.3 - THE SDK SHALL handle multiple values for the same role (e.g., "Toyota or Honda")
    Validates: Requirement 5.6 - THE SDK SHALL return extracted parameters in a structured format
    """
    parameters: Dict[str, ExtractedParameter]
    multi_value_params: Dict[str, List[ExtractedParameter]]
    unmatched_tokens: List['Token']
    
    def __post_init__(self) -> None:
        """Validate ExtractionResult fields after initialization.
        
        Validates that:
        - parameters is a dictionary
        - multi_value_params is a dictionary
        - unmatched_tokens is a list
        
        Raises:
            TypeError: If parameters is not a dictionary
            TypeError: If multi_value_params is not a dictionary
            TypeError: If unmatched_tokens is not a list
        """
        # Validate parameters is a dictionary
        if not isinstance(self.parameters, dict):
            raise TypeError(
                f"parameters must be a dictionary, "
                f"got {type(self.parameters).__name__}"
            )
        
        # Validate multi_value_params is a dictionary
        if not isinstance(self.multi_value_params, dict):
            raise TypeError(
                f"multi_value_params must be a dictionary, "
                f"got {type(self.multi_value_params).__name__}"
            )
        
        # Validate unmatched_tokens is a list
        if not isinstance(self.unmatched_tokens, list):
            raise TypeError(
                f"unmatched_tokens must be a list, "
                f"got {type(self.unmatched_tokens).__name__}"
            )
    
    def __repr__(self) -> str:
        """Return a string representation of the ExtractionResult."""
        return (
            f"ExtractionResult("
            f"parameters={len(self.parameters)} params, "
            f"multi_value_params={len(self.multi_value_params)} multi-params, "
            f"unmatched_tokens={len(self.unmatched_tokens)} unmatched)"
        )
    
    def has_parameters(self) -> bool:
        """Check if any parameters were extracted.
        
        Returns:
            True if either parameters or multi_value_params is non-empty,
            False otherwise.
        
        Example:
            >>> result = ExtractionResult(
            ...     parameters={"make": ExtractedParameter(...)},
            ...     multi_value_params={},
            ...     unmatched_tokens=[]
            ... )
            >>> result.has_parameters()
            True
            
            >>> empty_result = ExtractionResult(
            ...     parameters={},
            ...     multi_value_params={},
            ...     unmatched_tokens=[]
            ... )
            >>> empty_result.has_parameters()
            False
        """
        return bool(self.parameters) or bool(self.multi_value_params)
    
    def has_unmatched(self) -> bool:
        """Check if there are unmatched tokens.
        
        Returns:
            True if unmatched_tokens is non-empty, False otherwise.
        
        Example:
            >>> result = ExtractionResult(
            ...     parameters={},
            ...     multi_value_params={},
            ...     unmatched_tokens=[Token(...)]
            ... )
            >>> result.has_unmatched()
            True
        """
        return bool(self.unmatched_tokens)
    
    def get_all_roles(self) -> List[str]:
        """Get all role names that have extracted parameters.
        
        Returns:
            A list of role names from both parameters and multi_value_params.
            The list contains unique role names (no duplicates).
        
        Example:
            >>> result = ExtractionResult(
            ...     parameters={"make": ExtractedParameter(...)},
            ...     multi_value_params={"category": [ExtractedParameter(...)]},
            ...     unmatched_tokens=[]
            ... )
            >>> sorted(result.get_all_roles())
            ['category', 'make']
        """
        roles = set(self.parameters.keys())
        roles.update(self.multi_value_params.keys())
        return list(roles)
    
    def get_parameter(self, role: str) -> ExtractedParameter | None:
        """Get a single-value parameter by role name.
        
        Args:
            role: The role name to look up.
        
        Returns:
            The ExtractedParameter for the role, or None if not found.
        
        Example:
            >>> result = ExtractionResult(
            ...     parameters={"make": ExtractedParameter(...)},
            ...     multi_value_params={},
            ...     unmatched_tokens=[]
            ... )
            >>> result.get_parameter("make")
            ExtractedParameter(...)
            >>> result.get_parameter("unknown")
            None
        """
        return self.parameters.get(role)
    
    def get_multi_value_parameter(self, role: str) -> List[ExtractedParameter]:
        """Get a multi-value parameter by role name.
        
        Args:
            role: The role name to look up.
        
        Returns:
            A list of ExtractedParameter objects for the role, or an empty
            list if not found.
        
        Example:
            >>> result = ExtractionResult(
            ...     parameters={},
            ...     multi_value_params={"make": [ExtractedParameter(...), ExtractedParameter(...)]},
            ...     unmatched_tokens=[]
            ... )
            >>> len(result.get_multi_value_parameter("make"))
            2
            >>> result.get_multi_value_parameter("unknown")
            []
        """
        return self.multi_value_params.get(role, [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the extraction result to a dictionary.
        
        Returns:
            A dictionary representation of the extraction result suitable
            for serialization or API responses.
        
        Example:
            >>> result = ExtractionResult(
            ...     parameters={"make": ExtractedParameter(
            ...         role="make",
            ...         value="Toyota",
            ...         original_text="Toyota",
            ...         confidence=0.95,
            ...         value_type="string"
            ...     )},
            ...     multi_value_params={},
            ...     unmatched_tokens=[]
            ... )
            >>> d = result.to_dict()
            >>> d["parameters"]["make"]["value"]
            'Toyota'
        """
        return {
            "parameters": {
                role: {
                    "role": param.role,
                    "value": param.value,
                    "original_text": param.original_text,
                    "confidence": param.confidence,
                    "value_type": param.value_type
                }
                for role, param in self.parameters.items()
            },
            "multi_value_params": {
                role: [
                    {
                        "role": param.role,
                        "value": param.value,
                        "original_text": param.original_text,
                        "confidence": param.confidence,
                        "value_type": param.value_type
                    }
                    for param in params
                ]
                for role, params in self.multi_value_params.items()
            },
            "unmatched_tokens": [
                {
                    "text": token.text,
                    "original": token.original,
                    "position": token.position
                }
                for token in self.unmatched_tokens
            ]
        }


class ParameterExtractor:
    """
    Extracts parameters from matched tokens.
    
    The ParameterExtractor uses the schema configuration to perform type inference
    and extract structured parameters from natural language query matches. It
    associates matched values with their corresponding roles and preserves value
    types (string, number, boolean) from the schema.
    
    The extractor is initialized with an EncoderConfig that contains the schema
    definition (layers, segments, roles) which is used to:
    - Determine the expected type for each role's values
    - Validate extracted parameters against the schema
    - Support compound value extraction from multi-word matches
    
    Attributes:
        schema_config: The EncoderConfig containing the schema definition.
            This config is used for type inference and parameter validation.
    
    Example:
        >>> from glyphh.core.config import EncoderConfig, Layer, Segment, Role
        >>> 
        >>> # Create a schema config
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
        ...                         Role(name="year")
        ...                     ]
        ...                 )
        ...             ]
        ...         )
        ...     ]
        ... )
        >>> 
        >>> # Create the parameter extractor
        >>> extractor = ParameterExtractor(config)
        >>> extractor.schema_config.dimension
        10000
        >>> extractor.schema_config.seed
        42
        
        >>> # Invalid config type raises TypeError
        >>> ParameterExtractor("not a config")
        Traceback (most recent call last):
            ...
        TypeError: schema_config must be an EncoderConfig instance, got str
        
        >>> # None config raises TypeError
        >>> ParameterExtractor(None)
        Traceback (most recent call last):
            ...
        TypeError: schema_config must be an EncoderConfig instance, got NoneType
    
    Notes:
        - The schema_config is stored for use in type inference operations
        - The config must be a valid EncoderConfig instance
        - The extractor uses the config's layer/segment/role structure for
          determining value types and validating extracted parameters
    
    Validates: Requirement 5 - Parameter Extraction
    Validates: Requirement 5.1 - THE SDK SHALL extract matched values as query parameters
    Validates: Requirement 5.2 - WHEN a value is matched, THE SDK SHALL associate it with its role
    Validates: Requirement 5.4 - THE SDK SHALL preserve value types (string, number, boolean) from the schema
    """
    
    def __init__(self, schema_config: 'EncoderConfig') -> None:
        """
        Initialize the ParameterExtractor with a schema configuration.
        
        The schema configuration contains the model's layer/segment/role structure
        which is used for type inference when extracting parameters from matched
        tokens. The config is validated to ensure it is a proper EncoderConfig
        instance.
        
        Args:
            schema_config: The EncoderConfig containing the schema definition.
                Must be a valid EncoderConfig instance with dimension, seed,
                and optionally layers defining the schema structure.
        
        Raises:
            TypeError: If schema_config is not an EncoderConfig instance.
        
        Example:
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> # Basic initialization
            >>> config = EncoderConfig(dimension=10000, seed=42)
            >>> extractor = ParameterExtractor(config)
            >>> extractor.schema_config is config
            True
            
            >>> # With full schema
            >>> from glyphh.core.config import Layer, Segment, Role
            >>> full_config = EncoderConfig(
            ...     dimension=10000,
            ...     seed=42,
            ...     layers=[
            ...         Layer(
            ...             name="data",
            ...             segments=[
            ...                 Segment(
            ...                     name="attributes",
            ...                     roles=[Role(name="category")]
            ...                 )
            ...             ]
            ...         )
            ...     ]
            ... )
            >>> extractor = ParameterExtractor(full_config)
            >>> len(extractor.schema_config.layers)
            1
        
        Validates: Requirement 5 - Parameter Extraction
        """
        # Import here to avoid circular dependency at module level
        from glyphh.core.config import EncoderConfig
        
        # Validate that schema_config is an EncoderConfig instance
        if not isinstance(schema_config, EncoderConfig):
            raise TypeError(
                f"schema_config must be an EncoderConfig instance, "
                f"got {type(schema_config).__name__}"
            )
        
        # Store the config for use in type inference
        self.schema_config = schema_config
    
    def __repr__(self) -> str:
        """Return a string representation of the ParameterExtractor."""
        layer_count = len(self.schema_config.layers) if self.schema_config.layers else 0
        return (
            f"ParameterExtractor("
            f"dimension={self.schema_config.dimension}, "
            f"layers={layer_count})"
        )
    
    def infer_value_type(self, value: str, role_path: str) -> str:
        """
        Infer the type of a value based on schema hints and content analysis.
        
        This method determines whether a value should be treated as a string,
        number, or boolean. It first checks if the schema provides type hints
        for the given role path, and if not, analyzes the value content to
        determine the most appropriate type.
        
        Type inference priority:
        1. Schema hints (if the role has a defined type in the config)
        2. Boolean detection: "true", "false", "yes", "no" (case-insensitive)
        3. Number detection: try to parse as int or float
        4. Default to "string" if type cannot be determined
        
        Args:
            value: The string value to analyze for type inference. This is
                typically the original text from a query token that matched
                a schema value.
            role_path: The full path to the role in the schema, e.g.,
                "vehicle.identity.make" or just "make". Used to look up
                schema type hints if available.
        
        Returns:
            One of: "string", "number", or "boolean"
        
        Example:
            >>> from glyphh.core.config import EncoderConfig
            >>> config = EncoderConfig(dimension=10000, seed=42)
            >>> extractor = ParameterExtractor(config)
            >>> 
            >>> # Boolean detection (case-insensitive)
            >>> extractor.infer_value_type("true", "available")
            'boolean'
            >>> extractor.infer_value_type("FALSE", "active")
            'boolean'
            >>> extractor.infer_value_type("yes", "enabled")
            'boolean'
            >>> extractor.infer_value_type("No", "visible")
            'boolean'
            >>> 
            >>> # Number detection - integers
            >>> extractor.infer_value_type("42", "count")
            'number'
            >>> extractor.infer_value_type("-100", "offset")
            'number'
            >>> extractor.infer_value_type("0", "index")
            'number'
            >>> 
            >>> # Number detection - floats
            >>> extractor.infer_value_type("3.14", "price")
            'number'
            >>> extractor.infer_value_type("-0.5", "discount")
            'number'
            >>> extractor.infer_value_type("1.0", "rate")
            'number'
            >>> 
            >>> # String detection (default)
            >>> extractor.infer_value_type("Toyota", "make")
            'string'
            >>> extractor.infer_value_type("Brake Pads", "category")
            'string'
            >>> extractor.infer_value_type("", "empty")
            'string'
            >>> 
            >>> # Edge cases - strings that look like numbers but aren't
            >>> extractor.infer_value_type("12abc", "code")
            'string'
            >>> extractor.infer_value_type("1.2.3", "version")
            'string'
        
        Notes:
            - Boolean keywords are case-insensitive: "TRUE", "True", "true" all work
            - Number parsing handles both integers and floating-point values
            - Scientific notation (e.g., "1e10") is treated as a number
            - Empty strings default to "string" type
            - The role_path is reserved for future schema type hint support
        
        Validates: Requirement 5.4 - THE SDK SHALL preserve value types
            (string, number, boolean) from the schema
        """
        # Boolean keywords (case-insensitive)
        boolean_keywords = {"true", "false", "yes", "no"}
        
        # Check for boolean values first (case-insensitive)
        if value.lower() in boolean_keywords:
            return "boolean"
        
        # Try to parse as a number (int or float)
        try:
            # Try integer first
            int(value)
            return "number"
        except ValueError:
            pass
        
        try:
            # Try float (handles scientific notation too)
            float(value)
            return "number"
        except ValueError:
            pass
        
        # Default to string
        return "string"
    
    def extract_parameters(self, match_result: 'MatchResult') -> ExtractionResult:
        """
        Extract parameters from match result.
        
        Processes the value matches from a MatchResult and extracts structured
        parameters. Each matched value is associated with its role (extracted
        from the schema_vector.role_path), and the value type is inferred using
        the infer_value_type() method.
        
        For duplicate roles (when multiple values match the same role), the
        match with the highest confidence is kept.
        
        Args:
            match_result: A MatchResult from schema matching containing
                value_matches, token_matches, and other match information.
                Must be a valid MatchResult instance.
        
        Returns:
            ExtractionResult containing:
            - parameters: Dict mapping role names to ExtractedParameter objects
            - multi_value_params: Empty dict (multi-value handled in task 5.5)
            - unmatched_tokens: List of tokens that didn't match any schema value
        
        Example:
            >>> from glyphh.nl.schema_matcher import MatchResult, TokenMatch
            >>> from glyphh.nl.schema_vectorizer import SchemaVector
            >>> from glyphh.nl.query_tokenizer import Token
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> # Create a match result with value matches
            >>> token = Token(
            ...     text="toyota",
            ...     original="Toyota",
            ...     position=5,
            ...     is_stop_word=False,
            ...     ngram_size=1
            ... )
            >>> schema_vec = SchemaVector(
            ...     key="make=Toyota",
            ...     vector=some_vector,
            ...     element_type="value",
            ...     role_path="vehicle.identity.make",
            ...     original_value="Toyota"
            ... )
            >>> match = TokenMatch(
            ...     token=token,
            ...     schema_vector=schema_vec,
            ...     similarity=0.85,
            ...     match_type="exact"
            ... )
            >>> match_result = MatchResult(
            ...     query="Find Toyota",
            ...     token_matches=[match],
            ...     value_matches=[match]
            ... )
            >>> 
            >>> # Extract parameters
            >>> config = EncoderConfig(dimension=10000, seed=42)
            >>> extractor = ParameterExtractor(config)
            >>> result = extractor.extract_parameters(match_result)
            >>> 
            >>> # Check extracted parameter
            >>> "make" in result.parameters
            True
            >>> result.parameters["make"].value
            'Toyota'
            >>> result.parameters["make"].confidence
            0.85
        
        Notes:
            - Only value matches are processed (role matches are not parameters)
            - The role is extracted from the last component of role_path
            - Duplicate roles keep the highest confidence match
            - Unmatched tokens are tokens from the query that didn't match
              any schema value
            - multi_value_params is populated by handle_multi_value() (task 5.5)
        
        Validates: Requirement 5.1 - THE SDK SHALL extract matched values as
            query parameters
        Validates: Requirement 5.2 - WHEN a value is matched, THE SDK SHALL
            associate it with its role
        Validates: Requirement 5.6 - THE SDK SHALL return extracted parameters
            in a structured format
        """
        # Import MatchResult here to avoid circular dependency at module level
        from glyphh.nl.schema_matcher import MatchResult
        
        # Validate that match_result is a MatchResult instance
        if not isinstance(match_result, MatchResult):
            raise TypeError(
                f"match_result must be a MatchResult instance, "
                f"got {type(match_result).__name__}"
            )
        
        # Dictionary to store extracted parameters (role -> ExtractedParameter)
        parameters: Dict[str, ExtractedParameter] = {}
        
        # Track which tokens have been matched to identify unmatched tokens
        matched_token_positions: set = set()
        
        # Process each value match
        for token_match in match_result.value_matches:
            # Extract the role from the role_path (last component)
            # e.g., "vehicle.identity.make" -> "make"
            role_path = token_match.schema_vector.role_path
            role = role_path.split(".")[-1] if "." in role_path else role_path
            
            # Get the original value from the schema vector
            value = token_match.schema_vector.original_value
            
            # Get the original text from the token
            original_text = token_match.token.original
            
            # Get the confidence (similarity score)
            confidence = token_match.similarity
            
            # Infer the value type
            value_type = self.infer_value_type(str(value), role_path)
            
            # Track this token as matched
            matched_token_positions.add(token_match.token.position)
            
            # Handle duplicate roles by keeping the highest confidence match
            if role in parameters:
                # Only replace if new match has higher confidence
                if confidence > parameters[role].confidence:
                    parameters[role] = ExtractedParameter(
                        role=role,
                        value=value,
                        original_text=original_text,
                        confidence=confidence,
                        value_type=value_type
                    )
            else:
                # First match for this role
                parameters[role] = ExtractedParameter(
                    role=role,
                    value=value,
                    original_text=original_text,
                    confidence=confidence,
                    value_type=value_type
                )
        
        # Identify unmatched tokens
        # These are tokens from token_matches that didn't result in value matches
        # We need to find tokens that appear in token_matches but not in value_matches
        unmatched_tokens: List['Token'] = []
        
        # Get all unique tokens from token_matches
        all_token_positions = set()
        for token_match in match_result.token_matches:
            all_token_positions.add(token_match.token.position)
        
        # Find tokens that weren't matched to values
        # We need to look at the original tokens, but we only have access to
        # tokens through the matches. Unmatched tokens are those in token_matches
        # but not in value_matches (by position)
        value_match_positions = set()
        for token_match in match_result.value_matches:
            value_match_positions.add(token_match.token.position)
        
        # Collect unmatched tokens from token_matches
        seen_positions = set()
        for token_match in match_result.token_matches:
            pos = token_match.token.position
            if pos not in value_match_positions and pos not in seen_positions:
                unmatched_tokens.append(token_match.token)
                seen_positions.add(pos)
        
        # Return the extraction result
        return ExtractionResult(
            parameters=parameters,
            multi_value_params={},  # Will be populated by handle_multi_value() in task 5.5
            unmatched_tokens=unmatched_tokens
        )
    
    def handle_multi_value(self, tokens: List['Token']) -> List[str]:
        """
        Handle "X or Y" patterns in a list of tokens.
        
        Detects patterns where values are separated by the word "or" and extracts
        all values for the same role. This supports queries like "Toyota or Honda"
        or "red or blue or green" where multiple values should be extracted.
        
        The method scans through the token list looking for the word "or" (case-
        insensitive). When found, it extracts the tokens before and after "or"
        as values. Multiple consecutive "or" patterns are handled correctly,
        so "X or Y or Z" extracts ["X", "Y", "Z"].
        
        Args:
            tokens: A list of Token objects to scan for "X or Y" patterns.
                These should typically be single-word tokens (ngram_size=1)
                from the tokenize() method. The tokens are expected to be
                in order as they appear in the original query.
        
        Returns:
            A list of extracted value strings. If no "or" pattern is found,
            returns an empty list. If "or" patterns are found, returns all
            values that were separated by "or".
            
            Examples:
            - ["Toyota", "or", "Honda"] → ["Toyota", "Honda"]
            - ["red", "or", "blue", "or", "green"] → ["red", "blue", "green"]
            - ["Toyota"] → [] (no "or" pattern)
            - [] → [] (empty input)
        
        Example:
            >>> from glyphh.nl.query_tokenizer import Token
            >>> from glyphh.core.config import EncoderConfig
            >>> 
            >>> # Create tokens for "Toyota or Honda"
            >>> tokens = [
            ...     Token(text="toyota", original="Toyota", position=0, 
            ...           is_stop_word=False, ngram_size=1),
            ...     Token(text="or", original="or", position=7, 
            ...           is_stop_word=False, ngram_size=1),
            ...     Token(text="honda", original="Honda", position=10, 
            ...           is_stop_word=False, ngram_size=1)
            ... ]
            >>> 
            >>> config = EncoderConfig(dimension=10000, seed=42)
            >>> extractor = ParameterExtractor(config)
            >>> values = extractor.handle_multi_value(tokens)
            >>> values
            ['Toyota', 'Honda']
            
            >>> # Multiple "or" patterns: "red or blue or green"
            >>> tokens = [
            ...     Token(text="red", original="red", position=0,
            ...           is_stop_word=False, ngram_size=1),
            ...     Token(text="or", original="or", position=4,
            ...           is_stop_word=False, ngram_size=1),
            ...     Token(text="blue", original="blue", position=7,
            ...           is_stop_word=False, ngram_size=1),
            ...     Token(text="or", original="or", position=12,
            ...           is_stop_word=False, ngram_size=1),
            ...     Token(text="green", original="green", position=15,
            ...           is_stop_word=False, ngram_size=1)
            ... ]
            >>> values = extractor.handle_multi_value(tokens)
            >>> values
            ['red', 'blue', 'green']
            
            >>> # No "or" pattern
            >>> tokens = [
            ...     Token(text="toyota", original="Toyota", position=0,
            ...           is_stop_word=False, ngram_size=1)
            ... ]
            >>> values = extractor.handle_multi_value(tokens)
            >>> values
            []
            
            >>> # Empty input
            >>> extractor.handle_multi_value([])
            []
            
            >>> # Case-insensitive "or" detection
            >>> tokens = [
            ...     Token(text="toyota", original="Toyota", position=0,
            ...           is_stop_word=False, ngram_size=1),
            ...     Token(text="or", original="OR", position=7,
            ...           is_stop_word=False, ngram_size=1),
            ...     Token(text="honda", original="Honda", position=10,
            ...           is_stop_word=False, ngram_size=1)
            ... ]
            >>> values = extractor.handle_multi_value(tokens)
            >>> values
            ['Toyota', 'Honda']
        
        Notes:
            - The "or" keyword is detected case-insensitively (matches "or", "Or", "OR")
            - Values are extracted using the token's original text (preserving case)
            - Stop words are not filtered out - they can be values if around "or"
            - N-gram tokens (ngram_size > 1) are supported as values
            - If "or" appears at the start or end without adjacent values, those
              positions are skipped
            - Consecutive "or" tokens (e.g., "or or") are handled gracefully
        
        Validates: Requirement 5.3 - THE SDK SHALL handle multiple values for
            the same role (e.g., "Toyota or Honda")
        """
        # Handle empty input
        if not tokens:
            return []
        
        # Find all positions of "or" tokens (case-insensitive)
        or_positions: List[int] = []
        for i, token in enumerate(tokens):
            if token.text.lower() == "or":
                or_positions.append(i)
        
        # If no "or" found, return empty list (no multi-value pattern)
        if not or_positions:
            return []
        
        # Extract values around "or" tokens
        # We need to collect all values that are separated by "or"
        values: List[str] = []
        seen_positions: set = set()  # Track which token positions we've added
        
        for or_pos in or_positions:
            # Get the token before "or" (if exists and not already added)
            if or_pos > 0:
                before_pos = or_pos - 1
                # Skip if the token before is also "or"
                if tokens[before_pos].text.lower() != "or":
                    if before_pos not in seen_positions:
                        values.append(tokens[before_pos].original)
                        seen_positions.add(before_pos)
            
            # Get the token after "or" (if exists and not already added)
            if or_pos < len(tokens) - 1:
                after_pos = or_pos + 1
                # Skip if the token after is also "or"
                if tokens[after_pos].text.lower() != "or":
                    if after_pos not in seen_positions:
                        values.append(tokens[after_pos].original)
                        seen_positions.add(after_pos)
        
        return values
