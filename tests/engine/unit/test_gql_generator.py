"""
Unit tests for the GQLGenerator class.

Tests the GQL generation functionality including:
- Initialization with default and custom patterns
- Pattern lookup by name and intent type
- Slot retrieval (required, optional, defaults)
- Input validation

Validates: Requirement 4 - Intent Inference from Matches
"""

import pytest
from glyphh.nl.gql_generator import GQLGenerator
from glyphh.gql.patterns import GQLPattern, SlotDefinition, SlotType, DEFAULT_GQL_PATTERNS


class TestGQLGeneratorInit:
    """Tests for GQLGenerator initialization."""
    
    def test_init_with_default_patterns(self):
        """Test initialization with default patterns."""
        generator = GQLGenerator()
        
        # Should have default patterns
        assert len(generator.patterns) > 0
        assert generator.patterns == DEFAULT_GQL_PATTERNS
        
        # Pattern map should be built
        assert len(generator._pattern_map) == len(generator.patterns)
        
        # Should have common patterns
        assert "find_similar" in generator._pattern_map
        assert "count_all" in generator._pattern_map
        assert "compare" in generator._pattern_map
    
    def test_init_with_none_uses_defaults(self):
        """Test that passing None uses default patterns."""
        generator = GQLGenerator(patterns=None)
        
        assert len(generator.patterns) > 0
        assert generator.patterns == DEFAULT_GQL_PATTERNS
    
    def test_init_with_custom_patterns(self):
        """Test initialization with custom patterns."""
        custom_patterns = [
            GQLPattern(
                name="custom_find",
                phrases=["find {query}"],
                slots=[SlotDefinition(name="query", type=SlotType.STRING)],
                gql_template='FIND "{query}"'
            ),
            GQLPattern(
                name="custom_count",
                phrases=["count all"],
                slots=[],
                gql_template='COUNT ALL'
            )
        ]
        
        generator = GQLGenerator(patterns=custom_patterns)
        
        assert len(generator.patterns) == 2
        assert "custom_find" in generator._pattern_map
        assert "custom_count" in generator._pattern_map
        assert "find_similar" not in generator._pattern_map  # Default not included
    
    def test_init_with_empty_list(self):
        """Test initialization with empty pattern list."""
        generator = GQLGenerator(patterns=[])
        
        assert len(generator.patterns) == 0
        assert len(generator._pattern_map) == 0
    
    def test_init_with_invalid_type_raises_error(self):
        """Test that invalid patterns type raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            GQLGenerator(patterns="invalid")
        
        assert "patterns must be a list or None" in str(exc_info.value)
        assert "got str" in str(exc_info.value)
    
    def test_init_with_invalid_element_raises_error(self):
        """Test that invalid pattern element raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            GQLGenerator(patterns=["not a pattern"])
        
        assert "All patterns must be GQLPattern instances" in str(exc_info.value)
        assert "got str at index 0" in str(exc_info.value)
    
    def test_init_with_mixed_invalid_elements(self):
        """Test that mixed valid/invalid elements raises error at first invalid."""
        valid_pattern = GQLPattern(
            name="valid",
            phrases=["test"],
            gql_template='TEST'
        )
        
        with pytest.raises(TypeError) as exc_info:
            GQLGenerator(patterns=[valid_pattern, "invalid", 123])
        
        assert "got str at index 1" in str(exc_info.value)


class TestGQLGeneratorRepr:
    """Tests for GQLGenerator string representation."""
    
    def test_repr_with_default_patterns(self):
        """Test repr with default patterns."""
        generator = GQLGenerator()
        repr_str = repr(generator)
        
        assert "GQLGenerator" in repr_str
        assert f"patterns={len(generator.patterns)}" in repr_str
    
    def test_repr_with_custom_patterns(self):
        """Test repr with custom patterns."""
        custom_patterns = [
            GQLPattern(name="test1", phrases=[], gql_template='TEST1'),
            GQLPattern(name="test2", phrases=[], gql_template='TEST2'),
        ]
        generator = GQLGenerator(patterns=custom_patterns)
        repr_str = repr(generator)
        
        assert "patterns=2" in repr_str


class TestGetPatternByName:
    """Tests for get_pattern_by_name method."""
    
    def test_get_existing_pattern(self):
        """Test getting an existing pattern by name."""
        generator = GQLGenerator()
        
        pattern = generator.get_pattern_by_name("find_similar")
        
        assert pattern is not None
        assert pattern.name == "find_similar"
        assert isinstance(pattern, GQLPattern)
    
    def test_get_nonexistent_pattern(self):
        """Test getting a nonexistent pattern returns None."""
        generator = GQLGenerator()
        
        pattern = generator.get_pattern_by_name("nonexistent_pattern")
        
        assert pattern is None
    
    def test_get_pattern_case_sensitive(self):
        """Test that pattern lookup is case-sensitive."""
        generator = GQLGenerator()
        
        # Exact case should work
        pattern = generator.get_pattern_by_name("find_similar")
        assert pattern is not None
        
        # Different case should not work
        pattern = generator.get_pattern_by_name("FIND_SIMILAR")
        assert pattern is None
        
        pattern = generator.get_pattern_by_name("Find_Similar")
        assert pattern is None


class TestGetPatternsForIntent:
    """Tests for get_patterns_for_intent method."""
    
    def test_get_find_patterns(self):
        """Test getting patterns for 'find' intent."""
        generator = GQLGenerator()
        
        patterns = generator.get_patterns_for_intent("find")
        
        assert len(patterns) > 0
        # Should include find_similar and related patterns
        pattern_names = [p.name for p in patterns]
        assert "find_similar" in pattern_names or any("find" in name for name in pattern_names)
    
    def test_get_count_patterns(self):
        """Test getting patterns for 'count' intent."""
        generator = GQLGenerator()
        
        patterns = generator.get_patterns_for_intent("count")
        
        assert len(patterns) > 0
        pattern_names = [p.name for p in patterns]
        assert "count_all" in pattern_names or any("count" in name for name in pattern_names)
    
    def test_get_similar_patterns(self):
        """Test getting patterns for 'similar' intent."""
        generator = GQLGenerator()
        
        patterns = generator.get_patterns_for_intent("similar")
        
        assert len(patterns) > 0
        pattern_names = [p.name for p in patterns]
        assert "find_similar" in pattern_names or any("similar" in name for name in pattern_names)
    
    def test_get_filter_patterns(self):
        """Test getting patterns for 'filter' intent."""
        generator = GQLGenerator()
        
        patterns = generator.get_patterns_for_intent("filter")
        
        assert len(patterns) > 0
        pattern_names = [p.name for p in patterns]
        assert any("where" in name or "filter" in name for name in pattern_names)
    
    def test_patterns_sorted_by_priority(self):
        """Test that returned patterns are sorted by priority."""
        generator = GQLGenerator()
        
        patterns = generator.get_patterns_for_intent("find")
        
        # Check that patterns are sorted by priority (descending)
        for i in range(len(patterns) - 1):
            assert patterns[i].priority >= patterns[i + 1].priority
    
    def test_unknown_intent_returns_empty_or_partial(self):
        """Test that unknown intent returns empty list or partial matches."""
        generator = GQLGenerator()
        
        patterns = generator.get_patterns_for_intent("unknown_intent_xyz")
        
        # Should return empty list or only patterns containing the intent name
        assert isinstance(patterns, list)


class TestGetRequiredSlots:
    """Tests for get_required_slots method."""
    
    def test_get_required_slots_find_similar(self):
        """Test getting required slots for find_similar pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        required = generator.get_required_slots(pattern)
        
        assert isinstance(required, list)
        assert "query" in required
    
    def test_get_required_slots_count_all(self):
        """Test getting required slots for count_all pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("count_all")
        
        required = generator.get_required_slots(pattern)
        
        assert isinstance(required, list)
        # count_all has no required slots
        assert len(required) == 0
    
    def test_get_required_slots_compare(self):
        """Test getting required slots for compare pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("compare")
        
        required = generator.get_required_slots(pattern)
        
        assert isinstance(required, list)
        assert "glyph1" in required
        assert "glyph2" in required


class TestGetOptionalSlots:
    """Tests for get_optional_slots method."""
    
    def test_get_optional_slots_find_similar(self):
        """Test getting optional slots for find_similar pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        optional = generator.get_optional_slots(pattern)
        
        assert isinstance(optional, list)
        assert "limit" in optional
    
    def test_get_optional_slots_count_all(self):
        """Test getting optional slots for count_all pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("count_all")
        
        optional = generator.get_optional_slots(pattern)
        
        assert isinstance(optional, list)
        # count_all has no optional slots
        assert len(optional) == 0


class TestGetSlotDefaults:
    """Tests for get_slot_defaults method."""
    
    def test_get_slot_defaults_find_similar(self):
        """Test getting slot defaults for find_similar pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        defaults = generator.get_slot_defaults(pattern)
        
        assert isinstance(defaults, dict)
        assert "limit" in defaults
        assert defaults["limit"] == 10
    
    def test_get_slot_defaults_list_all(self):
        """Test getting slot defaults for list_all pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("list_all")
        
        defaults = generator.get_slot_defaults(pattern)
        
        assert isinstance(defaults, dict)
        assert "limit" in defaults
        assert defaults["limit"] == 100
    
    def test_get_slot_defaults_count_all(self):
        """Test getting slot defaults for count_all pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("count_all")
        
        defaults = generator.get_slot_defaults(pattern)
        
        assert isinstance(defaults, dict)
        # count_all has no slots with defaults
        assert len(defaults) == 0


class TestFindPatternForIntent:
    """Tests for find_pattern_for_intent method."""
    
    def test_find_pattern_for_find_intent(self):
        """Test finding pattern for 'find' intent."""
        generator = GQLGenerator()
        
        pattern = generator.find_pattern_for_intent("find")
        
        assert pattern is not None
        assert isinstance(pattern, GQLPattern)
        # Should be a find-related pattern
        assert "find" in pattern.name or "similar" in pattern.name or "list" in pattern.name
    
    def test_find_pattern_for_count_intent(self):
        """Test finding pattern for 'count' intent."""
        generator = GQLGenerator()
        
        pattern = generator.find_pattern_for_intent("count")
        
        assert pattern is not None
        assert isinstance(pattern, GQLPattern)
        # Should be a count-related pattern
        assert "count" in pattern.name or "aggregate" in pattern.name
    
    def test_find_pattern_for_similar_intent(self):
        """Test finding pattern for 'similar' intent."""
        generator = GQLGenerator()
        
        pattern = generator.find_pattern_for_intent("similar")
        
        assert pattern is not None
        assert isinstance(pattern, GQLPattern)
        # Should be a similarity-related pattern
        assert "similar" in pattern.name
    
    def test_find_pattern_for_filter_intent(self):
        """Test finding pattern for 'filter' intent."""
        generator = GQLGenerator()
        
        pattern = generator.find_pattern_for_intent("filter")
        
        assert pattern is not None
        assert isinstance(pattern, GQLPattern)
        # Should be a filter-related pattern
        assert "where" in pattern.name or "filter" in pattern.name or "list" in pattern.name
    
    def test_find_pattern_for_compare_intent(self):
        """Test finding pattern for 'compare' intent."""
        generator = GQLGenerator()
        
        pattern = generator.find_pattern_for_intent("compare")
        
        assert pattern is not None
        assert isinstance(pattern, GQLPattern)
        assert "compare" in pattern.name
    
    def test_find_pattern_for_trend_intent(self):
        """Test finding pattern for 'trend' intent."""
        generator = GQLGenerator()
        
        pattern = generator.find_pattern_for_intent("trend")
        
        assert pattern is not None
        assert isinstance(pattern, GQLPattern)
        assert "trend" in pattern.name
    
    def test_find_pattern_for_introspect_intent(self):
        """Test finding pattern for 'introspect' intent."""
        generator = GQLGenerator()
        
        pattern = generator.find_pattern_for_intent("introspect")
        
        assert pattern is not None
        assert isinstance(pattern, GQLPattern)
        assert "introspect" in pattern.name
    
    def test_find_pattern_for_unknown_intent_returns_none(self):
        """Test that unknown intent returns None."""
        generator = GQLGenerator()
        
        pattern = generator.find_pattern_for_intent("unknown_xyz_intent")
        
        assert pattern is None
    
    def test_find_pattern_for_empty_intent_returns_highest_priority(self):
        """Test that empty intent returns highest priority pattern (matches all)."""
        generator = GQLGenerator()
        
        # Empty string matches all patterns because "" is in every string
        # This is consistent with get_patterns_for_intent behavior
        pattern = generator.find_pattern_for_intent("")
        
        # Should return the highest priority pattern from all patterns
        all_patterns = generator.get_patterns_for_intent("")
        if all_patterns:
            assert pattern == all_patterns[0]
        else:
            assert pattern is None
    
    def test_find_pattern_returns_highest_priority(self):
        """Test that find_pattern_for_intent returns the highest priority pattern."""
        generator = GQLGenerator()
        
        # Get all patterns for 'find' intent
        all_patterns = generator.get_patterns_for_intent("find")
        
        # Get the best pattern
        best_pattern = generator.find_pattern_for_intent("find")
        
        # If there are patterns, the best should be the first (highest priority)
        if all_patterns:
            assert best_pattern == all_patterns[0]
            # Verify it has the highest priority among matching patterns
            for pattern in all_patterns:
                assert best_pattern.priority >= pattern.priority
        else:
            assert best_pattern is None
    
    def test_find_pattern_with_custom_patterns(self):
        """Test find_pattern_for_intent with custom patterns."""
        custom_patterns = [
            GQLPattern(
                name="custom_find_low",
                phrases=["find {query}"],
                slots=[SlotDefinition(name="query", type=SlotType.STRING)],
                gql_template='FIND "{query}"',
                priority=5
            ),
            GQLPattern(
                name="custom_find_high",
                phrases=["search {query}"],
                slots=[SlotDefinition(name="query", type=SlotType.STRING)],
                gql_template='SEARCH "{query}"',
                priority=15
            ),
        ]
        
        generator = GQLGenerator(patterns=custom_patterns)
        
        pattern = generator.find_pattern_for_intent("find")
        
        # Should return the higher priority pattern
        assert pattern is not None
        assert pattern.name == "custom_find_high"
        assert pattern.priority == 15
    
    def test_find_pattern_with_empty_patterns_list(self):
        """Test find_pattern_for_intent with empty patterns list."""
        generator = GQLGenerator(patterns=[])
        
        pattern = generator.find_pattern_for_intent("find")
        
        assert pattern is None
    
    def test_find_pattern_case_sensitivity(self):
        """Test that intent type matching is case-sensitive."""
        generator = GQLGenerator()
        
        # Lowercase should work
        pattern_lower = generator.find_pattern_for_intent("find")
        assert pattern_lower is not None
        
        # Uppercase should not match (unless pattern names contain uppercase)
        pattern_upper = generator.find_pattern_for_intent("FIND")
        # This may or may not return a pattern depending on pattern names
        # The key is that it's consistent with get_patterns_for_intent behavior
        all_upper_patterns = generator.get_patterns_for_intent("FIND")
        if all_upper_patterns:
            assert pattern_upper == all_upper_patterns[0]
        else:
            assert pattern_upper is None


class TestValidateSlots:
    """Tests for validate_slots method."""
    
    def test_validate_slots_all_required_filled(self):
        """Test that all required slots filled returns empty list."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        # find_similar requires "query"
        missing = generator.validate_slots(pattern, {"query": "Toyota"})
        
        assert missing == []
    
    def test_validate_slots_missing_required(self):
        """Test that missing required slot is returned."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        # find_similar requires "query", but we don't provide it
        missing = generator.validate_slots(pattern, {})
        
        assert "query" in missing
    
    def test_validate_slots_optional_not_required(self):
        """Test that optional slots are not required."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        # find_similar has optional "limit", but we only provide "query"
        missing = generator.validate_slots(pattern, {"query": "Toyota"})
        
        # "limit" should not be in missing since it's optional
        assert "limit" not in missing
        assert missing == []
    
    def test_validate_slots_no_required_slots(self):
        """Test pattern with no required slots returns empty list."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("count_all")
        
        # count_all has no required slots
        missing = generator.validate_slots(pattern, {})
        
        assert missing == []
    
    def test_validate_slots_multiple_required(self):
        """Test pattern with multiple required slots."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("compare")
        
        # compare requires both "glyph1" and "glyph2"
        # Only provide one
        missing = generator.validate_slots(pattern, {"glyph1": "abc"})
        
        assert "glyph2" in missing
        assert "glyph1" not in missing
    
    def test_validate_slots_all_multiple_required_filled(self):
        """Test all multiple required slots filled returns empty list."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("compare")
        
        # Provide both required slots
        missing = generator.validate_slots(pattern, {"glyph1": "abc", "glyph2": "xyz"})
        
        assert missing == []
    
    def test_validate_slots_none_value_is_missing(self):
        """Test that None values are treated as missing."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        # Provide query but with None value
        missing = generator.validate_slots(pattern, {"query": None})
        
        assert "query" in missing
    
    def test_validate_slots_empty_string_is_valid(self):
        """Test that empty string is considered a valid value."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        # Empty string is a valid value (not None)
        missing = generator.validate_slots(pattern, {"query": ""})
        
        assert missing == []
    
    def test_validate_slots_extra_parameters_ignored(self):
        """Test that extra parameters are ignored."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        # Provide required slot plus extra parameters
        missing = generator.validate_slots(pattern, {
            "query": "Toyota",
            "extra_param": "value",
            "another_extra": 123
        })
        
        assert missing == []
    
    def test_validate_slots_with_custom_pattern(self):
        """Test validate_slots with custom pattern."""
        custom_pattern = GQLPattern(
            name="custom_test",
            phrases=["test {a} {b} {c}"],
            slots=[
                SlotDefinition(name="a", type=SlotType.STRING, required=True),
                SlotDefinition(name="b", type=SlotType.STRING, required=True),
                SlotDefinition(name="c", type=SlotType.STRING, required=False),
            ],
            gql_template='TEST {a} {b} {c}'
        )
        
        generator = GQLGenerator(patterns=[custom_pattern])
        
        # Missing both required slots
        missing = generator.validate_slots(custom_pattern, {})
        assert "a" in missing
        assert "b" in missing
        assert "c" not in missing  # optional
        
        # Missing one required slot
        missing = generator.validate_slots(custom_pattern, {"a": "value_a"})
        assert "a" not in missing
        assert "b" in missing
        
        # All required filled
        missing = generator.validate_slots(custom_pattern, {"a": "value_a", "b": "value_b"})
        assert missing == []
    
    def test_validate_slots_find_similar_by_role(self):
        """Test validate_slots with find_similar_by_role pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar_by_role")
        
        # find_similar_by_role requires "role" and "value"
        # Missing both
        missing = generator.validate_slots(pattern, {})
        assert "role" in missing
        assert "value" in missing
        
        # Missing one
        missing = generator.validate_slots(pattern, {"role": "make"})
        assert "role" not in missing
        assert "value" in missing
        
        # All filled
        missing = generator.validate_slots(pattern, {"role": "make", "value": "Toyota"})
        assert missing == []
    
    def test_validate_slots_trend_pattern(self):
        """Test validate_slots with trend pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("trend")
        
        # trend requires "glyph", "window" is optional
        missing = generator.validate_slots(pattern, {})
        assert "glyph" in missing
        assert "window" not in missing  # optional
        
        # Glyph provided
        missing = generator.validate_slots(pattern, {"glyph": "my-glyph-id"})
        assert missing == []


class TestGQLGeneratorIntegration:
    """Integration tests for GQLGenerator."""
    
    def test_pattern_map_consistency(self):
        """Test that pattern map is consistent with patterns list."""
        generator = GQLGenerator()
        
        # All patterns should be in the map
        for pattern in generator.patterns:
            assert pattern.name in generator._pattern_map
            assert generator._pattern_map[pattern.name] == pattern
        
        # Map size should match patterns list size
        assert len(generator._pattern_map) == len(generator.patterns)
    
    def test_all_default_patterns_have_templates(self):
        """Test that all default patterns have GQL templates."""
        generator = GQLGenerator()
        
        for pattern in generator.patterns:
            assert pattern.gql_template, f"Pattern {pattern.name} has no template"
    
    def test_all_default_patterns_have_names(self):
        """Test that all default patterns have unique names."""
        generator = GQLGenerator()
        
        names = [p.name for p in generator.patterns]
        assert len(names) == len(set(names)), "Duplicate pattern names found"


class TestFillTemplate:
    """Tests for fill_template method."""
    
    def test_fill_template_basic(self):
        """Test basic template filling with required slot."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        gql = generator.fill_template(pattern, {"query": "Toyota"})
        
        assert "Toyota" in gql
        assert "FIND SIMILAR TO" in gql
    
    def test_fill_template_with_default_values(self):
        """Test that default values are applied for missing optional slots."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        # Only provide required slot, limit should use default (10)
        gql = generator.fill_template(pattern, {"query": "Toyota"})
        
        assert "LIMIT 10" in gql
    
    def test_fill_template_override_defaults(self):
        """Test that provided values override defaults."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        gql = generator.fill_template(pattern, {"query": "Honda", "limit": 5})
        
        assert "Honda" in gql
        assert "LIMIT 5" in gql
        assert "LIMIT 10" not in gql
    
    def test_fill_template_no_slots(self):
        """Test template with no slots."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("count_all")
        
        gql = generator.fill_template(pattern, {})
        
        assert gql == "COUNT ALL"
    
    def test_fill_template_multiple_required_slots(self):
        """Test template with multiple required slots."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("compare")
        
        gql = generator.fill_template(pattern, {"glyph1": "abc123", "glyph2": "xyz789"})
        
        assert "abc123" in gql
        assert "xyz789" in gql
        assert "COMPARE" in gql
    
    def test_fill_template_missing_required_raises_error(self):
        """Test that missing required slot raises ValueError."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        with pytest.raises(ValueError) as exc_info:
            generator.fill_template(pattern, {})
        
        assert "query" in str(exc_info.value)
        assert "Missing required slot" in str(exc_info.value)
    
    def test_fill_template_none_value_for_required_raises_error(self):
        """Test that None value for required slot raises ValueError."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        with pytest.raises(ValueError) as exc_info:
            generator.fill_template(pattern, {"query": None})
        
        assert "query" in str(exc_info.value)
    
    def test_fill_template_empty_string_is_valid(self):
        """Test that empty string is a valid value."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        # Empty string should not raise an error
        gql = generator.fill_template(pattern, {"query": ""})
        
        assert 'FIND SIMILAR TO ""' in gql
    
    def test_fill_template_numeric_value(self):
        """Test template filling with numeric values."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        gql = generator.fill_template(pattern, {"query": "test", "limit": 25})
        
        assert "LIMIT 25" in gql
    
    def test_fill_template_float_value(self):
        """Test template filling with float values."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("drift")
        
        gql = generator.fill_template(pattern, {"glyph": "my-glyph", "threshold": 0.25})
        
        assert "0.25" in gql
    
    def test_fill_template_list_value(self):
        """Test template filling with list values."""
        custom_pattern = GQLPattern(
            name="test_list",
            phrases=["test"],
            slots=[
                SlotDefinition(name="roles", type=SlotType.STRING, required=True),
            ],
            gql_template='FIND IN {roles}'
        )
        
        generator = GQLGenerator(patterns=[custom_pattern])
        
        gql = generator.fill_template(custom_pattern, {"roles": ["make", "model", "year"]})
        
        assert "make, model, year" in gql
    
    def test_fill_template_boolean_true(self):
        """Test template filling with boolean True value."""
        custom_pattern = GQLPattern(
            name="test_bool",
            phrases=["test"],
            slots=[
                SlotDefinition(name="flag", type=SlotType.STRING, required=True),
            ],
            gql_template='FIND WHERE active = {flag}'
        )
        
        generator = GQLGenerator(patterns=[custom_pattern])
        
        gql = generator.fill_template(custom_pattern, {"flag": True})
        
        assert "active = true" in gql
    
    def test_fill_template_boolean_false(self):
        """Test template filling with boolean False value."""
        custom_pattern = GQLPattern(
            name="test_bool",
            phrases=["test"],
            slots=[
                SlotDefinition(name="flag", type=SlotType.STRING, required=True),
            ],
            gql_template='FIND WHERE active = {flag}'
        )
        
        generator = GQLGenerator(patterns=[custom_pattern])
        
        gql = generator.fill_template(custom_pattern, {"flag": False})
        
        assert "active = false" in gql
    
    def test_fill_template_find_similar_by_role(self):
        """Test fill_template with find_similar_by_role pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar_by_role")
        
        gql = generator.fill_template(pattern, {"role": "make", "value": "Toyota"})
        
        assert "Toyota" in gql
        assert "make" in gql
        assert "FIND SIMILAR TO" in gql
        assert "LIMIT 10" in gql  # Default
    
    def test_fill_template_trend_pattern(self):
        """Test fill_template with trend pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("trend")
        
        gql = generator.fill_template(pattern, {"glyph": "my-glyph-id"})
        
        assert "my-glyph-id" in gql
        assert "TREND" in gql
        assert "7d" in gql  # Default window
    
    def test_fill_template_trend_with_custom_window(self):
        """Test fill_template with trend pattern and custom window."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("trend")
        
        gql = generator.fill_template(pattern, {"glyph": "my-glyph-id", "window": "30d"})
        
        assert "my-glyph-id" in gql
        assert "30d" in gql
        assert "7d" not in gql
    
    def test_fill_template_list_where_role(self):
        """Test fill_template with list_where_role pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("list_where_role")
        
        gql = generator.fill_template(pattern, {"role": "category", "value": "Brake Pads"})
        
        assert "category" in gql
        assert "Brake Pads" in gql
        assert "LIST ALL WHERE" in gql
        assert "LIMIT 100" in gql  # Default
    
    def test_fill_template_introspect_role(self):
        """Test fill_template with introspect_role pattern."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("introspect_role")
        
        gql = generator.fill_template(pattern, {"glyph": "glyph-123", "role": "make"})
        
        assert "glyph-123" in gql
        assert "make" in gql
        assert "INTROSPECT" in gql
    
    def test_fill_template_extra_parameters_ignored(self):
        """Test that extra parameters not in template are ignored."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        gql = generator.fill_template(pattern, {
            "query": "Toyota",
            "extra_param": "ignored",
            "another_extra": 123
        })
        
        assert "Toyota" in gql
        assert "ignored" not in gql
        assert "123" not in gql or "LIMIT" in gql  # 123 shouldn't appear except maybe in limit
    
    def test_fill_template_preserves_template_structure(self):
        """Test that template structure is preserved."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        gql = generator.fill_template(pattern, {"query": "test"})
        
        # Should match the template structure
        assert gql == 'FIND SIMILAR TO "test" LIMIT 10 THRESHOLD 0.5'
    
    def test_fill_template_compare_preserves_structure(self):
        """Test that compare template structure is preserved."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("compare")
        
        gql = generator.fill_template(pattern, {"glyph1": "a", "glyph2": "b"})
        
        assert gql == 'COMPARE glyph("a") TO glyph("b")'
    
    def test_fill_template_with_special_characters_in_value(self):
        """Test template filling with special characters in values."""
        generator = GQLGenerator()
        pattern = generator.get_pattern_by_name("find_similar")
        
        # Value with special characters
        gql = generator.fill_template(pattern, {"query": "Toyota's Brake-Pads (2024)"})
        
        assert "Toyota's Brake-Pads (2024)" in gql


class TestFormatValue:
    """Tests for _format_value helper method."""
    
    def test_format_string(self):
        """Test formatting string values."""
        generator = GQLGenerator()
        
        result = generator._format_value("Toyota", SlotType.STRING)
        
        assert result == "Toyota"
    
    def test_format_integer(self):
        """Test formatting integer values."""
        generator = GQLGenerator()
        
        result = generator._format_value(10, SlotType.NUMBER)
        
        assert result == "10"
    
    def test_format_float(self):
        """Test formatting float values."""
        generator = GQLGenerator()
        
        result = generator._format_value(3.14, SlotType.NUMBER)
        
        assert result == "3.14"
    
    def test_format_boolean_true(self):
        """Test formatting boolean True."""
        generator = GQLGenerator()
        
        result = generator._format_value(True, SlotType.STRING)
        
        assert result == "true"
    
    def test_format_boolean_false(self):
        """Test formatting boolean False."""
        generator = GQLGenerator()
        
        result = generator._format_value(False, SlotType.STRING)
        
        assert result == "false"
    
    def test_format_list(self):
        """Test formatting list values."""
        generator = GQLGenerator()
        
        result = generator._format_value(["a", "b", "c"], SlotType.STRING)
        
        assert result == "a, b, c"
    
    def test_format_tuple(self):
        """Test formatting tuple values."""
        generator = GQLGenerator()
        
        result = generator._format_value(("x", "y", "z"), SlotType.STRING)
        
        assert result == "x, y, z"
    
    def test_format_none(self):
        """Test formatting None value."""
        generator = GQLGenerator()
        
        result = generator._format_value(None, SlotType.STRING)
        
        assert result == ""
    
    def test_format_empty_list(self):
        """Test formatting empty list."""
        generator = GQLGenerator()
        
        result = generator._format_value([], SlotType.STRING)
        
        assert result == ""
    
    def test_format_list_with_numbers(self):
        """Test formatting list with numeric values."""
        generator = GQLGenerator()
        
        result = generator._format_value([1, 2, 3], SlotType.NUMBER)
        
        assert result == "1, 2, 3"
    
    def test_format_mixed_list(self):
        """Test formatting list with mixed types."""
        generator = GQLGenerator()
        
        result = generator._format_value(["a", 1, True], SlotType.STRING)
        
        assert result == "a, 1, True"


class TestGenerateGQL:
    """Tests for generate_gql method."""
    
    def test_generate_gql_find_intent(self):
        """Test generating GQL for find intent."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter
        
        generator = GQLGenerator()
        
        intent = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        params = ExtractionResult(
            parameters={
                "query": ExtractedParameter(
                    role="query",
                    value="Toyota",
                    original_text="Toyota",
                    confidence=0.9,
                    value_type="string"
                )
            },
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        gql = generator.generate_gql(intent, params)
        
        assert "Toyota" in gql
        assert "FIND SIMILAR TO" in gql
        assert "LIMIT 10" in gql  # Default limit
    
    def test_generate_gql_count_intent(self):
        """Test generating GQL for count intent with no parameters."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        generator = GQLGenerator()
        
        intent = InferredIntent(
            intent_type="count",
            confidence=0.9,
            matched_keywords=["count"],
            supporting_matches=[]
        )
        
        params = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        gql = generator.generate_gql(intent, params)
        
        assert gql == "COUNT ALL"
    
    def test_generate_gql_similar_intent(self):
        """Test generating GQL for similar intent."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter
        
        generator = GQLGenerator()
        
        intent = InferredIntent(
            intent_type="similar",
            confidence=0.8,
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        params = ExtractionResult(
            parameters={
                "query": ExtractedParameter(
                    role="query",
                    value="Honda Civic",
                    original_text="Honda Civic",
                    confidence=0.85,
                    value_type="string"
                )
            },
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        gql = generator.generate_gql(intent, params)
        
        assert "Honda Civic" in gql
        assert "FIND SIMILAR TO" in gql
    
    def test_generate_gql_filter_intent(self):
        """Test generating GQL for filter intent."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter
        
        generator = GQLGenerator()
        
        intent = InferredIntent(
            intent_type="filter",
            confidence=0.8,
            matched_keywords=["greater"],
            supporting_matches=[]
        )
        
        params = ExtractionResult(
            parameters={
                "role": ExtractedParameter(
                    role="role",
                    value="price",
                    original_text="price",
                    confidence=0.9,
                    value_type="string"
                ),
                "value": ExtractedParameter(
                    role="value",
                    value="10000",
                    original_text="10000",
                    confidence=0.85,
                    value_type="string"
                )
            },
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        gql = generator.generate_gql(intent, params)
        
        assert "price" in gql
        assert "10000" in gql
        assert "LIST ALL WHERE" in gql or "WHERE" in gql
    
    def test_generate_gql_no_matching_pattern_raises_error(self):
        """Test that unknown intent type raises ValueError."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Create generator with empty patterns
        generator = GQLGenerator(patterns=[])
        
        intent = InferredIntent(
            intent_type="find",
            confidence=0.5,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        params = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        with pytest.raises(ValueError) as exc_info:
            generator.generate_gql(intent, params)
        
        assert "No matching GQL pattern" in str(exc_info.value)
        assert "find" in str(exc_info.value)
    
    def test_generate_gql_missing_required_slot_raises_error(self):
        """Test that missing required slot raises ValueError when no fallback pattern exists."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        # Create a generator with only patterns that have required slots
        custom_patterns = [
            GQLPattern(
                name="custom_find",
                phrases=["find {query}"],
                slots=[
                    SlotDefinition(name="query", type=SlotType.STRING, required=True),
                ],
                gql_template='FIND "{query}"',
                priority=10
            )
        ]
        generator = GQLGenerator(patterns=custom_patterns)
        
        intent = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        # Empty parameters - missing required "query" slot
        params = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        with pytest.raises(ValueError) as exc_info:
            generator.generate_gql(intent, params)
        
        assert "Missing required slot" in str(exc_info.value)
        assert "query" in str(exc_info.value)
    
    def test_generate_gql_invalid_intent_type_raises_error(self):
        """Test that invalid intent type raises TypeError."""
        from glyphh.nl.parameter_extractor import ExtractionResult
        
        generator = GQLGenerator()
        
        params = ExtractionResult(
            parameters={},
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        with pytest.raises(TypeError) as exc_info:
            generator.generate_gql("not an intent", params)
        
        assert "intent must be an InferredIntent instance" in str(exc_info.value)
    
    def test_generate_gql_invalid_parameters_type_raises_error(self):
        """Test that invalid parameters type raises TypeError."""
        from glyphh.nl.intent_inferrer import InferredIntent
        
        generator = GQLGenerator()
        
        intent = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        with pytest.raises(TypeError) as exc_info:
            generator.generate_gql(intent, {"query": "Toyota"})
        
        assert "parameters must be an ExtractionResult instance" in str(exc_info.value)
    
    def test_generate_gql_with_numeric_parameter(self):
        """Test generating GQL with numeric parameter value."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter
        
        generator = GQLGenerator()
        
        intent = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        params = ExtractionResult(
            parameters={
                "query": ExtractedParameter(
                    role="query",
                    value="Toyota",
                    original_text="Toyota",
                    confidence=0.9,
                    value_type="string"
                ),
                "limit": ExtractedParameter(
                    role="limit",
                    value=25,
                    original_text="25",
                    confidence=0.95,
                    value_type="number"
                )
            },
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        gql = generator.generate_gql(intent, params)
        
        assert "Toyota" in gql
        assert "LIMIT 25" in gql
    
    def test_generate_gql_extra_parameters_ignored(self):
        """Test that extra parameters not in template are ignored."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter
        
        generator = GQLGenerator()
        
        intent = InferredIntent(
            intent_type="count",
            confidence=0.9,
            matched_keywords=["count"],
            supporting_matches=[]
        )
        
        # count_all has no slots, so extra params should be ignored
        params = ExtractionResult(
            parameters={
                "extra_param": ExtractedParameter(
                    role="extra_param",
                    value="ignored",
                    original_text="ignored",
                    confidence=0.5,
                    value_type="string"
                )
            },
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        gql = generator.generate_gql(intent, params)
        
        assert gql == "COUNT ALL"
        assert "ignored" not in gql
    
    def test_generate_gql_with_special_characters(self):
        """Test generating GQL with special characters in parameter values."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter
        
        generator = GQLGenerator()
        
        intent = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        params = ExtractionResult(
            parameters={
                "query": ExtractedParameter(
                    role="query",
                    value="Toyota's Brake-Pads (2024)",
                    original_text="Toyota's Brake-Pads (2024)",
                    confidence=0.9,
                    value_type="string"
                )
            },
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        gql = generator.generate_gql(intent, params)
        
        assert "Toyota's Brake-Pads (2024)" in gql
    
    def test_generate_gql_find_similar_by_role(self):
        """Test generating GQL for find intent with role and value parameters."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter
        
        generator = GQLGenerator()
        
        intent = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        # Provide role and value to match find_similar_by_role pattern
        params = ExtractionResult(
            parameters={
                "role": ExtractedParameter(
                    role="role",
                    value="make",
                    original_text="make",
                    confidence=0.9,
                    value_type="string"
                ),
                "value": ExtractedParameter(
                    role="value",
                    value="Toyota",
                    original_text="Toyota",
                    confidence=0.9,
                    value_type="string"
                )
            },
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        gql = generator.generate_gql(intent, params)
        
        assert "Toyota" in gql
        assert "make" in gql
        assert "FIND SIMILAR TO" in gql
    
    def test_generate_gql_with_custom_patterns(self):
        """Test generating GQL with custom patterns."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter
        
        custom_patterns = [
            GQLPattern(
                name="custom_find",
                phrases=["find {item}"],
                slots=[
                    SlotDefinition(name="item", type=SlotType.STRING, required=True),
                ],
                gql_template='CUSTOM FIND "{item}"',
                priority=10
            )
        ]
        
        generator = GQLGenerator(patterns=custom_patterns)
        
        intent = InferredIntent(
            intent_type="find",
            confidence=0.85,
            matched_keywords=["find"],
            supporting_matches=[]
        )
        
        params = ExtractionResult(
            parameters={
                "item": ExtractedParameter(
                    role="item",
                    value="test_item",
                    original_text="test_item",
                    confidence=0.9,
                    value_type="string"
                )
            },
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        gql = generator.generate_gql(intent, params)
        
        assert gql == 'CUSTOM FIND "test_item"'
    
    def test_generate_gql_similar_with_role_value(self):
        """Test generating GQL for similar intent with role and value."""
        from glyphh.nl.intent_inferrer import InferredIntent
        from glyphh.nl.parameter_extractor import ExtractionResult, ExtractedParameter
        
        generator = GQLGenerator()
        
        intent = InferredIntent(
            intent_type="similar",
            confidence=0.9,
            matched_keywords=["similar"],
            supporting_matches=[]
        )
        
        params = ExtractionResult(
            parameters={
                "role": ExtractedParameter(
                    role="role",
                    value="category",
                    original_text="category",
                    confidence=0.9,
                    value_type="string"
                ),
                "value": ExtractedParameter(
                    role="value",
                    value="Brake Pads",
                    original_text="Brake Pads",
                    confidence=0.85,
                    value_type="string"
                )
            },
            multi_value_params={},
            unmatched_tokens=[]
        )
        
        gql = generator.generate_gql(intent, params)
        
        assert "Brake Pads" in gql
        assert "category" in gql
        assert "FIND SIMILAR TO" in gql
