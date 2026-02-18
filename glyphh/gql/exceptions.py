"""
GQL Exception Hierarchy.

All GQL exceptions inherit from GQLError, which itself inherits from GlyphhException.
This allows catching all GQL-specific errors while maintaining compatibility with
the broader SDK exception hierarchy.

Exception Hierarchy:
====================

GlyphhException (from glyphh.exceptions)
└── GQLError (base for all GQL errors)
    ├── LexerError - Tokenization failures
    ├── ParseError - Syntax errors
    ├── PlanningError - Execution plan generation failures
    ├── ExecutionError - Query execution failures
    ├── SlotValidationError - NL slot extraction/validation failures
    ├── HierarchyError - Invalid hierarchy level references
    └── PredictionError - Temporal prediction failures

Usage Examples:
===============

1. Lexer Errors:
   >>> raise LexerError(
   ...     message="Unexpected character '@'",
   ...     line=1, column=15,
   ...     source_snippet='FIND SIMILAR @ "test"'
   ... )

2. Parse Errors:
   >>> raise ParseError(
   ...     message="Expected LIMIT or THRESHOLD",
   ...     line=1, column=25,
   ...     expected=["LIMIT", "THRESHOLD"],
   ...     actual="WHERE"
   ... )

3. Execution Errors:
   >>> raise ExecutionError(
   ...     message="Similarity search failed",
   ...     operation="similarity_search",
   ...     partial_results=[{"id": "glyph-1", "score": 0.9}]
   ... )
"""

from typing import Any, Dict, List, Optional
from glyphh.exceptions import GlyphhException


class GQLError(GlyphhException):
    """
    Base exception for all GQL errors.
    
    All GQL-specific exceptions inherit from this class to enable
    easy catching of GQL errors while allowing other SDK exceptions
    to propagate normally.
    
    Attributes:
        message: Human-readable error message
        line: Line number where error occurred (if applicable)
        column: Column number where error occurred (if applicable)
        suggestion: Optional suggested fix
    
    Example:
        >>> try:
        ...     result = gql.execute(query, model)
        ... except GQLError as e:
        ...     print(f"GQL error at {e.line}:{e.column}: {e.message}")
        ...     if e.suggestion:
        ...         print(f"Suggestion: {e.suggestion}")
    """
    
    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        suggestion: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.line = line
        self.column = column
        self.suggestion = suggestion
        
        # Build position string
        position = ""
        if line is not None:
            position = f" at line {line}"
            if column is not None:
                position += f", column {column}"
        
        full_message = f"{message}{position}"
        if suggestion:
            full_message += f". Suggestion: {suggestion}"
        
        ctx = context or {}
        ctx.update({
            "line": line,
            "column": column,
            "suggestion": suggestion
        })
        
        super().__init__(full_message, ctx)


class LexerError(GQLError):
    """
    Exception raised when tokenization fails.
    
    This exception is raised when the lexer encounters:
    - Invalid characters
    - Unterminated string literals
    - Invalid numeric literals
    - Invalid duration formats
    
    Attributes:
        message: Description of the lexer error
        line: Line number where error occurred
        column: Column number where error occurred
        source_snippet: Snippet of source around the error
        invalid_char: The invalid character (if applicable)
    
    Example:
        >>> raise LexerError(
        ...     message="Unterminated string literal",
        ...     line=1, column=15,
        ...     source_snippet='FIND SIMILAR TO "test',
        ...     suggestion="Add closing quote"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        line: int,
        column: int,
        source_snippet: Optional[str] = None,
        invalid_char: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        self.source_snippet = source_snippet
        self.invalid_char = invalid_char
        
        context = {
            "source_snippet": source_snippet,
            "invalid_char": invalid_char
        }
        
        super().__init__(
            message=message,
            line=line,
            column=column,
            suggestion=suggestion,
            context=context
        )


class ParseError(GQLError):
    """
    Exception raised when parsing fails.
    
    This exception is raised when the parser encounters:
    - Unexpected tokens
    - Missing required clauses
    - Invalid constraint values (e.g., negative LIMIT)
    - Malformed expressions
    
    Attributes:
        message: Description of the parse error
        line: Line number where error occurred
        column: Column number where error occurred
        expected: List of expected token types or keywords
        actual: The actual token that was found
    
    Example:
        >>> raise ParseError(
        ...     message="Unexpected token",
        ...     line=1, column=25,
        ...     expected=["LIMIT", "THRESHOLD", "IN"],
        ...     actual="WHERE",
        ...     suggestion="FIND SIMILAR queries don't support WHERE clauses"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        line: int,
        column: int,
        expected: Optional[List[str]] = None,
        actual: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        self.expected = expected or []
        self.actual = actual
        
        # Build detailed message
        if expected and actual:
            expected_str = ", ".join(expected)
            message = f"{message}: expected one of [{expected_str}], got '{actual}'"
        elif expected:
            expected_str = ", ".join(expected)
            message = f"{message}: expected one of [{expected_str}]"
        elif actual:
            message = f"{message}: unexpected '{actual}'"
        
        context = {
            "expected": expected,
            "actual": actual
        }
        
        super().__init__(
            message=message,
            line=line,
            column=column,
            suggestion=suggestion,
            context=context
        )


class PlanningError(GQLError):
    """
    Exception raised when execution plan generation fails.
    
    This exception is raised when the planner encounters:
    - References to non-existent glyphs
    - Invalid hierarchy paths
    - Unsupported operation combinations
    
    Attributes:
        message: Description of the planning error
        reference: The invalid reference (glyph ID, path, etc.)
        available: List of available/valid options
    
    Example:
        >>> raise PlanningError(
        ...     message="Glyph not found",
        ...     reference="product-xyz",
        ...     available=["product-001", "product-002", "product-003"],
        ...     suggestion="Check glyph ID spelling"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        reference: Optional[str] = None,
        available: Optional[List[str]] = None,
        suggestion: Optional[str] = None
    ):
        self.reference = reference
        self.available = available or []
        
        # Build detailed message
        if reference:
            message = f"{message}: '{reference}'"
        if available:
            # Show first 5 available options
            sample = available[:5]
            message += f". Available: {sample}"
            if len(available) > 5:
                message += f" (and {len(available) - 5} more)"
        
        context = {
            "reference": reference,
            "available": available
        }
        
        super().__init__(
            message=message,
            suggestion=suggestion,
            context=context
        )


class ExecutionError(GQLError):
    """
    Exception raised when query execution fails.
    
    This exception is raised when:
    - SDK operations fail
    - Resource limits are exceeded
    - Unexpected errors occur during execution
    
    Attributes:
        message: Description of the execution error
        operation: The operation that failed
        partial_results: Any partial results before failure
    
    Example:
        >>> raise ExecutionError(
        ...     message="Similarity search timed out",
        ...     operation="similarity_search",
        ...     partial_results=[{"id": "glyph-1", "score": 0.95}],
        ...     suggestion="Try reducing LIMIT or increasing THRESHOLD"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        partial_results: Optional[List[Any]] = None,
        suggestion: Optional[str] = None
    ):
        self.operation = operation
        self.partial_results = partial_results or []
        
        if operation:
            message = f"Operation '{operation}' failed: {message}"
        
        context = {
            "operation": operation,
            "partial_results": partial_results,
            "partial_count": len(partial_results) if partial_results else 0
        }
        
        super().__init__(
            message=message,
            suggestion=suggestion,
            context=context
        )


class SlotValidationError(GQLError):
    """
    Exception raised when NL slot extraction or validation fails.
    
    This exception is raised when:
    - Required slots are missing from the query
    - Slot values don't match expected types
    - Slot values fail validation rules
    
    Attributes:
        message: Description of the validation error
        slot_name: Name of the slot that failed
        slot_type: Expected type of the slot
        actual_value: The actual value that failed validation
        pattern_name: Name of the matched pattern
    
    Example:
        >>> raise SlotValidationError(
        ...     message="Invalid slot value",
        ...     slot_name="price",
        ...     slot_type="number",
        ...     actual_value="expensive",
        ...     pattern_name="find_by_price",
        ...     suggestion="Price must be a numeric value like '50' or '99.99'"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        slot_name: str,
        slot_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        pattern_name: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        self.slot_name = slot_name
        self.slot_type = slot_type
        self.actual_value = actual_value
        self.pattern_name = pattern_name
        
        # Build detailed message
        full_message = f"Slot '{slot_name}' validation failed: {message}"
        if slot_type and actual_value is not None:
            full_message += f" (expected {slot_type}, got '{actual_value}')"
        if pattern_name:
            full_message = f"[Pattern: {pattern_name}] {full_message}"
        
        context = {
            "slot_name": slot_name,
            "slot_type": slot_type,
            "actual_value": actual_value,
            "pattern_name": pattern_name
        }
        
        super().__init__(
            message=full_message,
            suggestion=suggestion,
            context=context
        )


class HierarchyError(GQLError):
    """
    Exception raised when hierarchy level references are invalid.
    
    This exception is raised when:
    - Specified layer doesn't exist in the glyph
    - Specified segment doesn't exist in the layer
    - Specified role doesn't exist in the segment
    
    Attributes:
        message: Description of the hierarchy error
        level: The hierarchy level (cortex, layer, segment, role)
        path: The invalid path
        available_paths: List of valid paths at this level
    
    Example:
        >>> raise HierarchyError(
        ...     message="Layer not found",
        ...     level="layer",
        ...     path="semantic.attributes",
        ...     available_paths=["semantic.concept", "semantic.relations"],
        ...     suggestion="Check layer and segment names in model config"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        level: str,
        path: Optional[str] = None,
        available_paths: Optional[List[str]] = None,
        suggestion: Optional[str] = None
    ):
        self.level = level
        self.path = path
        self.available_paths = available_paths or []
        
        # Build detailed message
        full_message = f"Hierarchy error at {level} level: {message}"
        if path:
            full_message += f" (path: '{path}')"
        if available_paths:
            sample = available_paths[:5]
            full_message += f". Available: {sample}"
            if len(available_paths) > 5:
                full_message += f" (and {len(available_paths) - 5} more)"
        
        context = {
            "level": level,
            "path": path,
            "available_paths": available_paths
        }
        
        super().__init__(
            message=full_message,
            suggestion=suggestion,
            context=context
        )


class PredictionError(GQLError):
    """
    Exception raised when temporal prediction fails.
    
    This exception is raised when:
    - Insufficient temporal data for prediction
    - Invalid time window specification
    - Prediction model fails
    
    Attributes:
        message: Description of the prediction error
        glyph_id: ID of the glyph being predicted
        window: The time window requested
        available_data_points: Number of available temporal data points
    
    Example:
        >>> raise PredictionError(
        ...     message="Insufficient temporal data",
        ...     glyph_id="customer-123",
        ...     window="30d",
        ...     available_data_points=2,
        ...     suggestion="Need at least 5 data points for prediction"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        glyph_id: Optional[str] = None,
        window: Optional[str] = None,
        available_data_points: Optional[int] = None,
        suggestion: Optional[str] = None
    ):
        self.glyph_id = glyph_id
        self.window = window
        self.available_data_points = available_data_points
        
        # Build detailed message
        full_message = f"Prediction failed: {message}"
        if glyph_id:
            full_message += f" (glyph: '{glyph_id}')"
        if window:
            full_message += f" (window: {window})"
        if available_data_points is not None:
            full_message += f" (available data points: {available_data_points})"
        
        context = {
            "glyph_id": glyph_id,
            "window": window,
            "available_data_points": available_data_points
        }
        
        super().__init__(
            message=full_message,
            suggestion=suggestion,
            context=context
        )
