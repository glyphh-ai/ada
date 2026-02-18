"""
GQL Generator module for the Glyphh SDK.

This module provides GQL query generation functionality from inferred intents
and extracted parameters. It uses the existing GQLPattern infrastructure to
map intents to GQL templates and fill them with extracted parameter values.

Classes:
    GQLGenerator: Generates GQL queries from intent and parameters

The GQLGenerator class uses the existing GQLPattern infrastructure from
glyphh.gql.patterns to map intents to GQL templates.

Validates: Requirement 4 - Intent Inference from Matches
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from glyphh.gql.patterns import GQLPattern, SlotDefinition, SlotType, DEFAULT_GQL_PATTERNS

if TYPE_CHECKING:
    from glyphh.nl.intent_inferrer import InferredIntent
    from glyphh.nl.parameter_extractor import ExtractionResult


class GQLGenerator:
    """
    Generates GQL queries from intent and parameters.
    
    Uses existing GQLPattern infrastructure to map intents to GQL templates.
    The generator maintains a collection of GQL patterns and provides methods
    to find the appropriate pattern for an intent, validate required slots,
    fill templates with parameter values, and generate executable GQL strings.
    
    The GQL generation process works as follows:
    1. Find the best matching GQL pattern for the inferred intent
    2. Validate that all required slots are filled by extracted parameters
    3. Fill the GQL template with parameter values
    4. Return the executable GQL string
    
    Attributes:
        patterns: List of GQLPattern objects defining available GQL templates.
            If not provided, DEFAULT_GQL_PATTERNS from glyphh.gql.patterns is used.
        _pattern_map: Internal dictionary mapping pattern names to GQLPattern
            objects for fast lookup.
    
    Example:
        >>> # Create generator with default patterns
        >>> generator = GQLGenerator()
        >>> len(generator.patterns) > 0
        True
        >>> "find_similar" in generator._pattern_map
        True
        
        >>> # Create generator with custom patterns
        >>> custom_patterns = [
        ...     GQLPattern(
        ...         name="custom_find",
        ...         phrases=["find {query}"],
        ...         slots=[SlotDefinition(name="query", type=SlotType.STRING)],
        ...         gql_template='FIND "{query}"'
        ...     )
        ... ]
        >>> generator = GQLGenerator(patterns=custom_patterns)
        >>> len(generator.patterns)
        1
        >>> "custom_find" in generator._pattern_map
        True
        
        >>> # Verify default patterns are used when None is passed
        >>> generator = GQLGenerator(patterns=None)
        >>> len(generator.patterns) > 0
        True
    
    Notes:
        - The generator is stateless - it does not cache results between calls
        - Pattern matching is based on intent type and pattern name
        - Required slots must be filled for successful GQL generation
        - Optional slots use default values if not provided
        - The generator works in conjunction with IntentInferrer and
          ParameterExtractor to complete the NL-to-GQL pipeline
    
    Validates: Requirement 4 - Intent Inference from Matches
    """
    
    def __init__(self, patterns: Optional[List[GQLPattern]] = None) -> None:
        """
        Initialize the GQLGenerator with optional custom patterns.
        
        Creates a new GQLGenerator instance with the specified GQL patterns.
        If no patterns are provided, DEFAULT_GQL_PATTERNS from glyphh.gql.patterns
        is used, which includes comprehensive patterns for all GQL operations:
        
        Default patterns include:
            - find_similar: Find glyphs similar to a text query
            - find_similar_by_role: Find glyphs with similar role values
            - list_all: List all glyphs
            - count_all: Count all glyphs
            - compare: Compare two glyphs
            - trend: Track trends over time
            - introspect: Introspect a glyph's structure
            - And many more...
        
        Args:
            patterns: Optional list of GQLPattern objects defining available
                GQL templates. If None, DEFAULT_GQL_PATTERNS is used.
        
        Raises:
            TypeError: If patterns is provided but is not a list.
            TypeError: If any element in patterns is not a GQLPattern instance.
        
        Example:
            >>> # Default patterns
            >>> generator = GQLGenerator()
            >>> "find_similar" in generator._pattern_map
            True
            >>> "count_all" in generator._pattern_map
            True
            
            >>> # Custom patterns
            >>> custom = [
            ...     GQLPattern(
            ...         name="my_pattern",
            ...         phrases=["my query"],
            ...         gql_template='MY GQL'
            ...     )
            ... ]
            >>> generator = GQLGenerator(patterns=custom)
            >>> "my_pattern" in generator._pattern_map
            True
            >>> "find_similar" in generator._pattern_map
            False
            
            >>> # None uses defaults
            >>> generator = GQLGenerator(patterns=None)
            >>> len(generator.patterns) > 0
            True
            
            >>> # Invalid type raises TypeError
            >>> generator = GQLGenerator(patterns="invalid")
            Traceback (most recent call last):
                ...
            TypeError: patterns must be a list or None, got str
            
            >>> # Invalid element type raises TypeError
            >>> generator = GQLGenerator(patterns=["not a pattern"])
            Traceback (most recent call last):
                ...
            TypeError: All patterns must be GQLPattern instances, got str at index 0
        
        Validates: Requirement 4
        """
        # Validate patterns parameter type
        if patterns is not None:
            if not isinstance(patterns, list):
                raise TypeError(
                    f"patterns must be a list or None, "
                    f"got {type(patterns).__name__}"
                )
            
            # Validate all elements are GQLPattern instances
            for i, pattern in enumerate(patterns):
                if not isinstance(pattern, GQLPattern):
                    raise TypeError(
                        f"All patterns must be GQLPattern instances, "
                        f"got {type(pattern).__name__} at index {i}"
                    )
        
        # Use provided patterns or default patterns
        self.patterns: List[GQLPattern] = patterns if patterns is not None else DEFAULT_GQL_PATTERNS
        
        # Build pattern map for fast lookup by name
        self._pattern_map: Dict[str, GQLPattern] = {p.name: p for p in self.patterns}
    
    def __repr__(self) -> str:
        """Return a string representation of the GQLGenerator."""
        return (
            f"GQLGenerator("
            f"patterns={len(self.patterns)}, "
            f"pattern_names={list(self._pattern_map.keys())[:5]}...)"
        )
    
    def get_pattern_by_name(self, name: str) -> Optional[GQLPattern]:
        """
        Get a pattern by its name.
        
        Args:
            name: The name of the pattern to retrieve.
        
        Returns:
            The GQLPattern with the given name, or None if not found.
        
        Example:
            >>> generator = GQLGenerator()
            >>> pattern = generator.get_pattern_by_name("find_similar")
            >>> pattern is not None
            True
            >>> pattern.name
            'find_similar'
            >>> generator.get_pattern_by_name("nonexistent") is None
            True
        """
        return self._pattern_map.get(name)
    
    def get_patterns_for_intent(self, intent_type: str) -> List[GQLPattern]:
        """
        Get all patterns that match an intent type.
        
        Searches through all patterns and returns those whose names contain
        the intent type or are commonly associated with it.
        
        Args:
            intent_type: The intent type to match (e.g., "find", "count", "filter").
        
        Returns:
            A list of GQLPattern objects that match the intent type,
            sorted by priority (highest first).
        
        Example:
            >>> generator = GQLGenerator()
            >>> find_patterns = generator.get_patterns_for_intent("find")
            >>> len(find_patterns) > 0
            True
            >>> all("find" in p.name or "similar" in p.name or "list" in p.name 
            ...     for p in find_patterns)
            True
        """
        # Intent type to pattern name mappings
        intent_pattern_mappings = {
            "find": ["find_similar", "find_similar_by_role", "list_all", "list_where", "list_where_role"],
            "count": ["count_all", "aggregate_count", "aggregate_count_role"],
            "filter": ["list_where", "list_where_role"],
            "similar": ["find_similar", "find_similar_by_role"],
            "compare": ["compare", "compare_by_role"],
            "trend": ["trend", "trend_by_role"],
            "introspect": ["introspect", "introspect_layer", "introspect_segment", "introspect_role"],
        }
        
        # Get pattern names for this intent type
        pattern_names = intent_pattern_mappings.get(intent_type, [])
        
        # Also include patterns whose names contain the intent type
        matching_patterns = []
        for pattern in self.patterns:
            if pattern.name in pattern_names or intent_type in pattern.name.lower():
                matching_patterns.append(pattern)
        
        # Sort by priority (highest first)
        matching_patterns.sort(key=lambda p: p.priority, reverse=True)
        
        return matching_patterns
    
    def get_required_slots(self, pattern: GQLPattern) -> List[str]:
        """
        Get the names of required slots for a pattern.
        
        Args:
            pattern: The GQLPattern to get required slots for.
        
        Returns:
            A list of slot names that are required for this pattern.
        
        Example:
            >>> generator = GQLGenerator()
            >>> pattern = generator.get_pattern_by_name("find_similar")
            >>> required = generator.get_required_slots(pattern)
            >>> "query" in required
            True
        """
        return [slot.name for slot in pattern.slots if slot.required]
    
    def get_optional_slots(self, pattern: GQLPattern) -> List[str]:
        """
        Get the names of optional slots for a pattern.
        
        Args:
            pattern: The GQLPattern to get optional slots for.
        
        Returns:
            A list of slot names that are optional for this pattern.
        
        Example:
            >>> generator = GQLGenerator()
            >>> pattern = generator.get_pattern_by_name("find_similar")
            >>> optional = generator.get_optional_slots(pattern)
            >>> "limit" in optional
            True
        """
        return [slot.name for slot in pattern.slots if not slot.required]
    
    def get_slot_defaults(self, pattern: GQLPattern) -> Dict[str, Any]:
        """
        Get default values for all slots in a pattern.
        
        Args:
            pattern: The GQLPattern to get slot defaults for.
        
        Returns:
            A dictionary mapping slot names to their default values.
            Only slots with default values are included.
        
        Example:
            >>> generator = GQLGenerator()
            >>> pattern = generator.get_pattern_by_name("find_similar")
            >>> defaults = generator.get_slot_defaults(pattern)
            >>> "limit" in defaults
            True
            >>> defaults["limit"]
            10
        """
        return {
            slot.name: slot.default
            for slot in pattern.slots
            if slot.default is not None
        }
    
    def find_pattern_for_intent(self, intent_type: str) -> Optional[GQLPattern]:
        """
        Find the best GQL pattern for an intent type.
        
        Maps an intent type (e.g., "find", "count", "filter", "similar") to the
        best matching GQL pattern. The "best" pattern is determined by priority,
        with higher priority patterns being preferred.
        
        This method uses `get_patterns_for_intent()` internally, which returns
        patterns sorted by priority (highest first). This method simply returns
        the first (highest priority) pattern from that list, or None if no
        patterns match the intent type.
        
        Args:
            intent_type: The intent type to find a pattern for. Common values
                include:
                - "find": Find/search operations
                - "count": Counting operations
                - "filter": Filtering/where operations
                - "similar": Similarity search operations
                - "compare": Comparison operations
                - "trend": Trend analysis operations
                - "introspect": Introspection operations
        
        Returns:
            The GQLPattern with the highest priority that matches the intent
            type, or None if no patterns match.
        
        Example:
            >>> generator = GQLGenerator()
            >>> 
            >>> # Find pattern for 'find' intent
            >>> pattern = generator.find_pattern_for_intent("find")
            >>> pattern is not None
            True
            >>> "find" in pattern.name or "similar" in pattern.name or "list" in pattern.name
            True
            >>> 
            >>> # Find pattern for 'count' intent
            >>> pattern = generator.find_pattern_for_intent("count")
            >>> pattern is not None
            True
            >>> "count" in pattern.name
            True
            >>> 
            >>> # Find pattern for 'similar' intent
            >>> pattern = generator.find_pattern_for_intent("similar")
            >>> pattern is not None
            True
            >>> "similar" in pattern.name
            True
            >>> 
            >>> # Unknown intent returns None
            >>> pattern = generator.find_pattern_for_intent("unknown_xyz_intent")
            >>> pattern is None
            True
            >>> 
            >>> # Returns highest priority pattern
            >>> generator = GQLGenerator()
            >>> find_patterns = generator.get_patterns_for_intent("find")
            >>> best_pattern = generator.find_pattern_for_intent("find")
            >>> if find_patterns:
            ...     best_pattern == find_patterns[0]
            ... else:
            ...     best_pattern is None
            True
        
        Notes:
            - The method is case-sensitive for intent type matching
            - If multiple patterns match, the one with highest priority is returned
            - An empty string intent type will return None (no patterns match)
            - The method delegates to `get_patterns_for_intent()` for pattern lookup
        
        Validates: Requirement 4 - Intent Inference from Matches
        """
        # Get all patterns for this intent type, sorted by priority (highest first)
        matching_patterns = self.get_patterns_for_intent(intent_type)
        
        # Return the highest priority pattern, or None if no matches
        if matching_patterns:
            return matching_patterns[0]
        
        return None
    
    def validate_slots(
        self,
        pattern: GQLPattern,
        parameters: Dict[str, Any]
    ) -> List[str]:
        """
        Validate that required slots are filled. Returns missing slot names.
        
        Checks that all required slots defined in the pattern have corresponding
        values in the parameters dictionary. A slot is considered "filled" if
        its name exists as a key in the parameters dictionary and the value is
        not None.
        
        This method is used during GQL generation to ensure that all required
        parameters have been extracted from the user's query before attempting
        to fill the GQL template.
        
        Args:
            pattern: The GQLPattern to validate against. The pattern's slots
                define which parameters are required vs optional.
            parameters: A dictionary mapping slot names to their values. This
                typically comes from the ParameterExtractor's extraction result.
        
        Returns:
            A list of slot names that are required but missing from parameters.
            Returns an empty list if all required slots are filled.
        
        Example:
            >>> generator = GQLGenerator()
            >>> 
            >>> # Get a pattern with required slots
            >>> pattern = generator.get_pattern_by_name("find_similar")
            >>> 
            >>> # All required slots filled - returns empty list
            >>> missing = generator.validate_slots(pattern, {"query": "Toyota"})
            >>> missing
            []
            >>> 
            >>> # Missing required slot - returns list with missing slot name
            >>> missing = generator.validate_slots(pattern, {})
            >>> "query" in missing
            True
            >>> 
            >>> # Optional slots don't need to be filled
            >>> missing = generator.validate_slots(pattern, {"query": "Toyota"})
            >>> "limit" in missing  # limit is optional
            False
            >>> 
            >>> # Pattern with no required slots
            >>> count_pattern = generator.get_pattern_by_name("count_all")
            >>> missing = generator.validate_slots(count_pattern, {})
            >>> missing
            []
            >>> 
            >>> # Pattern with multiple required slots
            >>> compare_pattern = generator.get_pattern_by_name("compare")
            >>> missing = generator.validate_slots(compare_pattern, {"glyph1": "abc"})
            >>> "glyph2" in missing
            True
            >>> "glyph1" in missing
            False
            >>> 
            >>> # All required slots filled for compare
            >>> missing = generator.validate_slots(compare_pattern, {"glyph1": "abc", "glyph2": "xyz"})
            >>> missing
            []
            >>> 
            >>> # None values are treated as missing
            >>> missing = generator.validate_slots(pattern, {"query": None})
            >>> "query" in missing
            True
        
        Notes:
            - Only required slots (slot.required=True) are validated
            - Optional slots are not included in the missing list even if absent
            - A slot with a None value is considered missing
            - Empty strings are considered valid values (not missing)
            - The order of missing slot names in the returned list is not guaranteed
        
        Validates: Requirement 5 - Parameter Extraction
        """
        missing_slots: List[str] = []
        
        for slot in pattern.slots:
            if slot.required:
                # Check if the slot name exists in parameters and has a non-None value
                if slot.name not in parameters or parameters[slot.name] is None:
                    missing_slots.append(slot.name)
        
        return missing_slots
    
    def fill_template(
        self,
        pattern: GQLPattern,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Fill GQL template with parameter values.
        
        Takes a GQL pattern and a dictionary of parameters, and returns the
        GQL template with all placeholders replaced by their corresponding
        parameter values. Missing optional parameters are filled with their
        default values from the pattern's slot definitions.
        
        The method handles different value types appropriately:
        - Strings: Inserted as-is (templates already include quotes where needed)
        - Numbers: Converted to string representation
        - Lists: Joined with commas
        - Booleans: Converted to lowercase string ("true" or "false")
        - None: Treated as missing (uses default if available)
        
        Args:
            pattern: The GQLPattern containing the template and slot definitions.
                The pattern's gql_template contains placeholders in the format
                {slot_name} that will be replaced with parameter values.
            parameters: A dictionary mapping slot names to their values. This
                typically comes from the ParameterExtractor's extraction result.
        
        Returns:
            The filled GQL string with all placeholders replaced by values.
        
        Raises:
            ValueError: If a required slot is missing from parameters and has
                no default value.
        
        Example:
            >>> generator = GQLGenerator()
            >>> 
            >>> # Basic template filling
            >>> pattern = generator.get_pattern_by_name("find_similar")
            >>> gql = generator.fill_template(pattern, {"query": "Toyota"})
            >>> "Toyota" in gql
            True
            >>> "LIMIT 10" in gql  # Default limit applied
            True
            >>> 
            >>> # Override default values
            >>> gql = generator.fill_template(pattern, {"query": "Honda", "limit": 5})
            >>> "Honda" in gql
            True
            >>> "LIMIT 5" in gql
            True
            >>> 
            >>> # Pattern with no slots
            >>> count_pattern = generator.get_pattern_by_name("count_all")
            >>> gql = generator.fill_template(count_pattern, {})
            >>> gql
            'COUNT ALL'
            >>> 
            >>> # Pattern with multiple required slots
            >>> compare_pattern = generator.get_pattern_by_name("compare")
            >>> gql = generator.fill_template(compare_pattern, {"glyph1": "abc", "glyph2": "xyz"})
            >>> "abc" in gql
            True
            >>> "xyz" in gql
            True
            >>> 
            >>> # Missing required slot raises ValueError
            >>> try:
            ...     generator.fill_template(pattern, {})
            ... except ValueError as e:
            ...     "query" in str(e)
            True
            >>> 
            >>> # List values are joined with commas
            >>> custom_pattern = GQLPattern(
            ...     name="test",
            ...     phrases=[],
            ...     slots=[SlotDefinition(name="roles", type=SlotType.STRING)],
            ...     gql_template='FIND IN {roles}'
            ... )
            >>> gql = generator.fill_template(custom_pattern, {"roles": ["make", "model"]})
            >>> "make, model" in gql
            True
        
        Notes:
            - Placeholders use the format {slot_name} in the template
            - Default values are obtained from get_slot_defaults()
            - The method does not validate the resulting GQL syntax
            - Empty strings are valid values and will be inserted as-is
            - The template's existing quotes are preserved (don't double-quote)
        
        Validates: Requirements 4, 5 - Intent Inference and Parameter Extraction
        """
        # Start with the template
        template = pattern.gql_template
        
        # Get default values for optional slots
        defaults = self.get_slot_defaults(pattern)
        
        # Merge defaults with provided parameters (provided values take precedence)
        merged_params: Dict[str, Any] = {}
        merged_params.update(defaults)
        
        # Only update with non-None values from parameters
        for key, value in parameters.items():
            if value is not None:
                merged_params[key] = value
        
        # Check for missing required slots
        missing_required = self.validate_slots(pattern, merged_params)
        if missing_required:
            raise ValueError(
                f"Missing required slot(s) for pattern '{pattern.name}': "
                f"{', '.join(missing_required)}"
            )
        
        # Replace each placeholder in the template
        result = template
        for slot in pattern.slots:
            placeholder = "{" + slot.name + "}"
            if placeholder in result:
                value = merged_params.get(slot.name)
                
                # Format the value appropriately
                formatted_value = self._format_value(value, slot.type)
                
                # Replace the placeholder
                result = result.replace(placeholder, formatted_value)
        
        return result
    
    def _format_value(self, value: Any, slot_type: SlotType) -> str:
        """
        Format a value for insertion into a GQL template.
        
        Converts different Python types to their appropriate string
        representations for GQL templates.
        
        Args:
            value: The value to format.
            slot_type: The slot type, used for type-specific formatting.
        
        Returns:
            The formatted string representation of the value.
        
        Example:
            >>> generator = GQLGenerator()
            >>> generator._format_value("Toyota", SlotType.STRING)
            'Toyota'
            >>> generator._format_value(10, SlotType.NUMBER)
            '10'
            >>> generator._format_value(3.14, SlotType.NUMBER)
            '3.14'
            >>> generator._format_value(True, SlotType.STRING)
            'true'
            >>> generator._format_value(False, SlotType.STRING)
            'false'
            >>> generator._format_value(["a", "b", "c"], SlotType.STRING)
            'a, b, c'
            >>> generator._format_value(None, SlotType.STRING)
            ''
        """
        if value is None:
            return ""
        
        if isinstance(value, bool):
            # Boolean values: convert to lowercase string
            return "true" if value else "false"
        
        if isinstance(value, (int, float)):
            # Numeric values: convert to string
            return str(value)
        
        if isinstance(value, (list, tuple)):
            # List values: join with comma and space
            return ", ".join(str(item) for item in value)
        
        # String and other values: convert to string
        return str(value)
    
    def generate_gql(
        self,
        intent: 'InferredIntent',
        parameters: 'ExtractionResult'
    ) -> str:
        """
        Generate GQL query from intent and extracted parameters.
        
        This is the main method for converting natural language query analysis
        results into an executable GQL string. It combines the inferred intent
        with extracted parameters to produce a complete GQL query.
        
        The generation process:
        1. Find all matching GQL patterns for the intent type
        2. Select the best pattern that can be satisfied by the available parameters
        3. Convert extracted parameters to a dictionary of {slot_name: value}
        4. Fill the GQL template with parameter values
        5. Return the executable GQL string
        
        Args:
            intent: An InferredIntent object containing the intent type
                (e.g., "find", "count", "filter", "similar") and confidence
                information. The intent_type is used to find the appropriate
                GQL pattern.
            parameters: An ExtractionResult object containing extracted
                parameters from the query. The parameters dict maps role names
                to ExtractedParameter objects, which are converted to a simple
                {role: value} dict for template filling.
        
        Returns:
            An executable GQL string with all placeholders filled with
            parameter values.
        
        Raises:
            ValueError: If no matching GQL pattern is found for the intent type.
            ValueError: If required slots are missing from the parameters
                (raised by fill_template()).
            TypeError: If intent is not an InferredIntent instance.
            TypeError: If parameters is not an ExtractionResult instance.
        
        Example:
            >>> from glyphh.nl.intent_inferrer import InferredIntent
            >>> from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter
            >>> 
            >>> generator = GQLGenerator()
            >>> 
            >>> # Create an inferred intent
            >>> intent = InferredIntent(
            ...     intent_type="find",
            ...     confidence=0.85,
            ...     matched_keywords=["find"],
            ...     supporting_matches=[]
            ... )
            >>> 
            >>> # Create extraction result with parameters
            >>> params = ExtractionResult(
            ...     parameters={
            ...         "query": ExtractedParameter(
            ...             role="query",
            ...             value="Toyota",
            ...             original_text="Toyota",
            ...             confidence=0.9,
            ...             value_type="string"
            ...         )
            ...     },
            ...     multi_value_params={},
            ...     unmatched_tokens=[]
            ... )
            >>> 
            >>> # Generate GQL
            >>> gql = generator.generate_gql(intent, params)
            >>> "Toyota" in gql
            True
            >>> "FIND SIMILAR TO" in gql
            True
            >>> 
            >>> # Count intent with no parameters
            >>> count_intent = InferredIntent(
            ...     intent_type="count",
            ...     confidence=0.9,
            ...     matched_keywords=["count"],
            ...     supporting_matches=[]
            ... )
            >>> empty_params = ExtractionResult(
            ...     parameters={},
            ...     multi_value_params={},
            ...     unmatched_tokens=[]
            ... )
            >>> gql = generator.generate_gql(count_intent, empty_params)
            >>> gql
            'COUNT ALL'
            >>> 
            >>> # Unknown intent raises ValueError
            >>> unknown_intent = InferredIntent(
            ...     intent_type="find",  # Valid type but...
            ...     confidence=0.5,
            ...     matched_keywords=[],
            ...     supporting_matches=[]
            ... )
            >>> # If we use a generator with no patterns:
            >>> empty_generator = GQLGenerator(patterns=[])
            >>> try:
            ...     empty_generator.generate_gql(unknown_intent, empty_params)
            ... except ValueError as e:
            ...     "No matching GQL pattern" in str(e)
            True
        
        Notes:
            - The method uses get_patterns_for_intent() to get all matching patterns
            - It selects the best pattern whose required slots can be satisfied
            - Parameters are converted from ExtractedParameter objects to simple values
            - The role names from parameters are used as slot names for template filling
            - Multi-value parameters are not currently supported (uses single values)
            - Missing required slots will raise ValueError from fill_template()
        
        Validates: Requirement 4 - Intent Inference from Matches
        Validates: Requirement 5 - Parameter Extraction
        """
        # Import types here to avoid circular imports at module level
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Validate intent is an InferredIntent instance
        if not isinstance(intent, InferredIntent):
            raise TypeError(
                f"intent must be an InferredIntent instance, "
                f"got {type(intent).__name__}"
            )
        
        # Validate parameters is an ExtractionResult instance
        if not isinstance(parameters, ExtractionResult):
            raise TypeError(
                f"parameters must be an ExtractionResult instance, "
                f"got {type(parameters).__name__}"
            )
        
        # Step 1: Convert ExtractionResult.parameters to a dict of {role: value}
        # The parameters dict maps role names to ExtractedParameter objects
        # We need to extract the actual values for template filling
        param_dict: Dict[str, Any] = {}
        
        for role, extracted_param in parameters.parameters.items():
            # Use the extracted value directly
            param_dict[role] = extracted_param.value
        
        # Step 2: Get all patterns for this intent type, sorted by priority
        matching_patterns = self.get_patterns_for_intent(intent.intent_type)
        
        if not matching_patterns:
            raise ValueError(
                f"No matching GQL pattern found for intent type '{intent.intent_type}'"
            )
        
        # Step 3: Find the best pattern that can be satisfied by available parameters
        # Try patterns in priority order (highest first)
        best_pattern = None
        for pattern in matching_patterns:
            missing_slots = self.validate_slots(pattern, param_dict)
            if not missing_slots:
                # This pattern can be satisfied with available parameters
                best_pattern = pattern
                break
        
        # If no pattern can be satisfied, use the first pattern and let fill_template
        # raise an appropriate error about missing slots
        if best_pattern is None:
            best_pattern = matching_patterns[0]
        
        # Step 4: Fill the template with parameter values
        # This will raise ValueError if required slots are missing
        gql = self.fill_template(best_pattern, param_dict)
        
        return gql
