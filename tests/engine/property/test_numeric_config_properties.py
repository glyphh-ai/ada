"""
Property-based tests for NumericConfig validation.

This module contains property-based tests using Hypothesis to verify
universal correctness properties for NumericConfig validation.

Feature: numeric-binning-role-config
"""

import pytest
from hypothesis import given, settings, strategies as st, assume
from typing import Optional

from glyphh.core.config import NumericConfig, EncodingStrategy
from glyphh.exceptions import ConfigurationException


# =============================================================================
# Test Data Generators (Strategies)
# =============================================================================

# Generate positive bin widths (valid)
positive_bin_widths = st.floats(
    min_value=0.001,
    max_value=1000.0,
    allow_nan=False,
    allow_infinity=False
)

# Generate non-positive bin widths (invalid)
non_positive_bin_widths = st.floats(
    max_value=0.0,
    allow_nan=False,
    allow_infinity=False
)

# Generate encoding strategies
encoding_strategies = st.sampled_from([
    EncodingStrategy.THERMOMETER,
    EncodingStrategy.BINARY,
    EncodingStrategy.GRAY
])

# Generate optional min/max values
optional_bounds = st.one_of(
    st.none(),
    st.floats(min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
)


@st.composite
def valid_min_max_pair(draw):
    """
    Generate a valid (min_value, max_value) pair where min < max.
    
    Returns:
        Tuple of (min_value, max_value) where min_value < max_value
    """
    min_val = draw(st.floats(min_value=-10000.0, max_value=9999.0, allow_nan=False, allow_infinity=False))
    # Ensure max is strictly greater than min
    max_val = draw(st.floats(min_value=min_val + 0.001, max_value=10000.0, allow_nan=False, allow_infinity=False))
    return (min_val, max_val)


@st.composite
def invalid_min_max_pair(draw):
    """
    Generate an invalid (min_value, max_value) pair where min >= max.
    
    Returns:
        Tuple of (min_value, max_value) where min_value >= max_value
    """
    max_val = draw(st.floats(min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    # min_value is >= max_value
    min_val = draw(st.floats(min_value=max_val, max_value=10000.0, allow_nan=False, allow_infinity=False))
    return (min_val, max_val)


# =============================================================================
# Property 1: Bin Width Validation
# =============================================================================

class TestBinWidthValidation:
    """
    Property tests for Bin Width Validation (Property 1).
    
    Feature: numeric-binning-role-config, Property 1: Bin Width Validation
    
    **Validates: Requirements 1.2, 3.5**
    
    For any NumericConfig creation attempt with a bin_width value that is zero
    or negative, the system should reject the configuration with a validation error.
    """
    
    @given(bin_width=non_positive_bin_widths)
    @settings(max_examples=100)
    def test_non_positive_bin_width_rejected(self, bin_width: float):
        """
        Property test: Non-positive bin_width values are rejected.
        
        For any bin_width <= 0, NumericConfig creation SHALL raise
        ConfigurationException.
        
        Feature: numeric-binning-role-config, Property 1: Bin Width Validation
        **Validates: Requirements 1.2, 3.5**
        """
        with pytest.raises(ConfigurationException) as exc_info:
            NumericConfig(bin_width=bin_width)
        
        # Verify the error message mentions bin_width
        assert "bin_width" in str(exc_info.value).lower(), \
            f"Error message should mention bin_width, got: {exc_info.value}"
    
    @given(bin_width=st.just(0.0))
    @settings(max_examples=10)
    def test_zero_bin_width_rejected(self, bin_width: float):
        """
        Property test: Zero bin_width is rejected.
        
        bin_width = 0 SHALL raise ConfigurationException.
        
        Feature: numeric-binning-role-config, Property 1: Bin Width Validation
        **Validates: Requirements 1.2, 3.5**
        """
        with pytest.raises(ConfigurationException) as exc_info:
            NumericConfig(bin_width=bin_width)
        
        assert "positive" in str(exc_info.value).lower() or "bin_width" in str(exc_info.value).lower(), \
            f"Error message should indicate bin_width must be positive, got: {exc_info.value}"
    
    @given(bin_width=positive_bin_widths)
    @settings(max_examples=100)
    def test_positive_bin_width_accepted(self, bin_width: float):
        """
        Property test: Positive bin_width values are accepted.
        
        For any bin_width > 0, NumericConfig creation SHALL succeed.
        
        Feature: numeric-binning-role-config, Property 1: Bin Width Validation
        **Validates: Requirements 1.2, 3.5**
        """
        config = NumericConfig(bin_width=bin_width)
        
        assert config.bin_width == bin_width, \
            f"bin_width should be {bin_width}, got {config.bin_width}"
    
    @given(
        bin_width=positive_bin_widths,
        encoding_strategy=encoding_strategies
    )
    @settings(max_examples=100)
    def test_positive_bin_width_with_strategy_accepted(
        self, bin_width: float, encoding_strategy: EncodingStrategy
    ):
        """
        Property test: Positive bin_width with any encoding strategy is accepted.
        
        For any bin_width > 0 and any valid encoding_strategy,
        NumericConfig creation SHALL succeed.
        
        Feature: numeric-binning-role-config, Property 1: Bin Width Validation
        **Validates: Requirements 1.2, 3.5**
        """
        config = NumericConfig(
            bin_width=bin_width,
            encoding_strategy=encoding_strategy
        )
        
        assert config.bin_width == bin_width
        assert config.encoding_strategy == encoding_strategy


# =============================================================================
# Property 2: Min/Max Validation
# =============================================================================

class TestMinMaxValidation:
    """
    Property tests for Min/Max Validation (Property 2).
    
    Feature: numeric-binning-role-config, Property 2: Min/Max Validation
    
    **Validates: Requirements 3.6**
    
    For any NumericConfig where both min_value and max_value are provided,
    if min_value >= max_value, the system should reject the configuration
    with a validation error.
    """
    
    @given(min_max=invalid_min_max_pair())
    @settings(max_examples=100)
    def test_min_greater_than_or_equal_max_rejected(self, min_max):
        """
        Property test: min_value >= max_value is rejected.
        
        For any (min_value, max_value) where min_value >= max_value,
        NumericConfig creation SHALL raise ConfigurationException.
        
        Feature: numeric-binning-role-config, Property 2: Min/Max Validation
        **Validates: Requirements 3.6**
        """
        min_val, max_val = min_max
        
        with pytest.raises(ConfigurationException) as exc_info:
            NumericConfig(
                bin_width=1.0,
                min_value=min_val,
                max_value=max_val
            )
        
        # Verify the error message mentions min/max
        error_msg = str(exc_info.value).lower()
        assert "min" in error_msg or "max" in error_msg, \
            f"Error message should mention min/max, got: {exc_info.value}"
    
    @given(min_max=valid_min_max_pair())
    @settings(max_examples=100)
    def test_min_less_than_max_accepted(self, min_max):
        """
        Property test: min_value < max_value is accepted.
        
        For any (min_value, max_value) where min_value < max_value,
        NumericConfig creation SHALL succeed.
        
        Feature: numeric-binning-role-config, Property 2: Min/Max Validation
        **Validates: Requirements 3.6**
        """
        min_val, max_val = min_max
        
        config = NumericConfig(
            bin_width=1.0,
            min_value=min_val,
            max_value=max_val
        )
        
        assert config.min_value == min_val, \
            f"min_value should be {min_val}, got {config.min_value}"
        assert config.max_value == max_val, \
            f"max_value should be {max_val}, got {config.max_value}"
    
    @given(
        bin_width=positive_bin_widths,
        min_value=optional_bounds
    )
    @settings(max_examples=100)
    def test_min_only_accepted(self, bin_width: float, min_value: Optional[float]):
        """
        Property test: min_value without max_value is accepted.
        
        For any min_value (or None) without max_value,
        NumericConfig creation SHALL succeed.
        
        Feature: numeric-binning-role-config, Property 2: Min/Max Validation
        **Validates: Requirements 3.6**
        """
        config = NumericConfig(
            bin_width=bin_width,
            min_value=min_value
        )
        
        assert config.min_value == min_value
        assert config.max_value is None
    
    @given(
        bin_width=positive_bin_widths,
        max_value=optional_bounds
    )
    @settings(max_examples=100)
    def test_max_only_accepted(self, bin_width: float, max_value: Optional[float]):
        """
        Property test: max_value without min_value is accepted.
        
        For any max_value (or None) without min_value,
        NumericConfig creation SHALL succeed.
        
        Feature: numeric-binning-role-config, Property 2: Min/Max Validation
        **Validates: Requirements 3.6**
        """
        config = NumericConfig(
            bin_width=bin_width,
            max_value=max_value
        )
        
        assert config.min_value is None
        assert config.max_value == max_value
    
    @given(bin_width=positive_bin_widths)
    @settings(max_examples=100)
    def test_no_bounds_accepted(self, bin_width: float):
        """
        Property test: No min/max bounds is accepted.
        
        NumericConfig with neither min_value nor max_value SHALL succeed.
        
        Feature: numeric-binning-role-config, Property 2: Min/Max Validation
        **Validates: Requirements 3.6**
        """
        config = NumericConfig(bin_width=bin_width)
        
        assert config.min_value is None
        assert config.max_value is None


# =============================================================================
# Additional Validation Tests
# =============================================================================

class TestEncodingStrategyValidation:
    """
    Additional property tests for encoding strategy validation.
    """
    
    @given(
        bin_width=positive_bin_widths,
        strategy=encoding_strategies
    )
    @settings(max_examples=100)
    def test_valid_encoding_strategies_accepted(
        self, bin_width: float, strategy: EncodingStrategy
    ):
        """
        Property test: All valid encoding strategies are accepted.
        
        For any valid EncodingStrategy enum value, NumericConfig creation
        SHALL succeed.
        """
        config = NumericConfig(
            bin_width=bin_width,
            encoding_strategy=strategy
        )
        
        assert config.encoding_strategy == strategy
    
    @given(
        bin_width=positive_bin_widths,
        strategy_str=st.sampled_from(['thermometer', 'binary', 'gray'])
    )
    @settings(max_examples=100)
    def test_string_encoding_strategies_accepted(
        self, bin_width: float, strategy_str: str
    ):
        """
        Property test: String encoding strategy values are accepted and converted.
        
        For any valid encoding strategy string, NumericConfig creation
        SHALL succeed and convert to EncodingStrategy enum.
        """
        config = NumericConfig(
            bin_width=bin_width,
            encoding_strategy=strategy_str
        )
        
        assert config.encoding_strategy == EncodingStrategy(strategy_str)
    
    @given(
        bin_width=positive_bin_widths,
        invalid_strategy=st.text(min_size=1, max_size=20).filter(
            lambda x: x not in ['thermometer', 'binary', 'gray']
        )
    )
    @settings(max_examples=50)
    def test_invalid_encoding_strategies_rejected(
        self, bin_width: float, invalid_strategy: str
    ):
        """
        Property test: Invalid encoding strategy strings are rejected.
        
        For any string that is not a valid encoding strategy,
        NumericConfig creation SHALL raise ConfigurationException.
        """
        with pytest.raises((ConfigurationException, ValueError)):
            NumericConfig(
                bin_width=bin_width,
                encoding_strategy=invalid_strategy
            )


class TestNumericConfigSerialization:
    """
    Property tests for NumericConfig serialization round-trip.
    """
    
    @given(
        bin_width=positive_bin_widths,
        encoding_strategy=encoding_strategies,
        min_max=st.one_of(
            st.just((None, None)),
            valid_min_max_pair().map(lambda x: (x[0], None)),
            valid_min_max_pair().map(lambda x: (None, x[1])),
            valid_min_max_pair()
        )
    )
    @settings(max_examples=100)
    def test_serialization_round_trip(
        self,
        bin_width: float,
        encoding_strategy: EncodingStrategy,
        min_max
    ):
        """
        Property test: NumericConfig serialization round-trip preserves values.
        
        For any valid NumericConfig, to_dict() followed by from_dict()
        SHALL produce an equivalent config.
        """
        min_value, max_value = min_max
        
        original = NumericConfig(
            bin_width=bin_width,
            encoding_strategy=encoding_strategy,
            min_value=min_value,
            max_value=max_value
        )
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = NumericConfig.from_dict(serialized)
        
        # Verify all fields match
        assert restored.bin_width == original.bin_width, \
            f"bin_width mismatch: {restored.bin_width} != {original.bin_width}"
        assert restored.encoding_strategy == original.encoding_strategy, \
            f"encoding_strategy mismatch: {restored.encoding_strategy} != {original.encoding_strategy}"
        assert restored.min_value == original.min_value, \
            f"min_value mismatch: {restored.min_value} != {original.min_value}"
        assert restored.max_value == original.max_value, \
            f"max_value mismatch: {restored.max_value} != {original.max_value}"



# =============================================================================
# Property 13: Role Serialization Round-Trip
# =============================================================================

class TestRoleSerializationRoundTrip:
    """
    Property tests for Role Serialization Round-Trip (Property 13).
    
    Feature: numeric-binning-role-config, Property 13: Role Serialization Round-Trip
    
    **Validates: Requirements 7.1, 7.2**
    
    For any Role with numeric_config, serializing (exporting) and then
    deserializing (importing) should produce an equivalent Role with
    the same numeric_config values.
    """
    
    @given(
        name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        similarity_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        security_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        key_part=st.booleans(),
        bin_width=positive_bin_widths,
        encoding_strategy=encoding_strategies,
        min_max=st.one_of(
            st.just((None, None)),
            st.tuples(
                st.floats(min_value=-500.0, max_value=400.0, allow_nan=False, allow_infinity=False),
                st.none()
            ),
            st.tuples(
                st.none(),
                st.floats(min_value=-400.0, max_value=500.0, allow_nan=False, allow_infinity=False)
            ),
        )
    )
    @settings(max_examples=100)
    def test_role_with_numeric_config_round_trip(
        self,
        name: str,
        similarity_weight: float,
        security_weight: float,
        key_part: bool,
        bin_width: float,
        encoding_strategy: EncodingStrategy,
        min_max
    ):
        """
        Property test: Role with numeric_config serialization round-trip.
        
        For any Role with numeric_config, to_dict() followed by from_dict()
        SHALL produce an equivalent Role.
        
        Feature: numeric-binning-role-config, Property 13: Role Serialization Round-Trip
        **Validates: Requirements 7.1, 7.2**
        """
        from glyphh.core.config import Role
        
        min_value, max_value = min_max
        
        # Create numeric config
        numeric_config = NumericConfig(
            bin_width=bin_width,
            encoding_strategy=encoding_strategy,
            min_value=min_value,
            max_value=max_value
        )
        
        # Create role with numeric config
        original = Role(
            name=name,
            similarity_weight=similarity_weight,
            security_weight=security_weight,
            key_part=key_part,
            numeric_config=numeric_config
        )
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = Role.from_dict(serialized)
        
        # Verify all fields match
        assert restored.name == original.name, \
            f"name mismatch: {restored.name} != {original.name}"
        assert restored.similarity_weight == original.similarity_weight, \
            f"similarity_weight mismatch: {restored.similarity_weight} != {original.similarity_weight}"
        assert restored.security_weight == original.security_weight, \
            f"security_weight mismatch: {restored.security_weight} != {original.security_weight}"
        assert restored.key_part == original.key_part, \
            f"key_part mismatch: {restored.key_part} != {original.key_part}"
        
        # Verify numeric_config
        assert restored.numeric_config is not None, \
            "numeric_config should not be None after round-trip"
        assert restored.numeric_config.bin_width == original.numeric_config.bin_width, \
            f"bin_width mismatch: {restored.numeric_config.bin_width} != {original.numeric_config.bin_width}"
        assert restored.numeric_config.encoding_strategy == original.numeric_config.encoding_strategy, \
            f"encoding_strategy mismatch: {restored.numeric_config.encoding_strategy} != {original.numeric_config.encoding_strategy}"
        assert restored.numeric_config.min_value == original.numeric_config.min_value, \
            f"min_value mismatch: {restored.numeric_config.min_value} != {original.numeric_config.min_value}"
        assert restored.numeric_config.max_value == original.numeric_config.max_value, \
            f"max_value mismatch: {restored.numeric_config.max_value} != {original.numeric_config.max_value}"
    
    @given(
        name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        similarity_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        security_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        key_part=st.booleans()
    )
    @settings(max_examples=100)
    def test_role_without_numeric_config_round_trip(
        self,
        name: str,
        similarity_weight: float,
        security_weight: float,
        key_part: bool
    ):
        """
        Property test: Role without numeric_config serialization round-trip.
        
        For any Role without numeric_config, to_dict() followed by from_dict()
        SHALL produce an equivalent Role with numeric_config=None.
        
        Feature: numeric-binning-role-config, Property 13: Role Serialization Round-Trip
        **Validates: Requirements 7.1, 7.2**
        """
        from glyphh.core.config import Role
        
        # Create role without numeric config
        original = Role(
            name=name,
            similarity_weight=similarity_weight,
            security_weight=security_weight,
            key_part=key_part
        )
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = Role.from_dict(serialized)
        
        # Verify all fields match
        assert restored.name == original.name
        assert restored.similarity_weight == original.similarity_weight
        assert restored.security_weight == original.security_weight
        assert restored.key_part == original.key_part
        assert restored.numeric_config is None, \
            "numeric_config should be None for role without numeric config"
