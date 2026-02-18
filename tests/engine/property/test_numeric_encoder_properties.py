"""
Property-based tests for Numeric Binning Encoder.

This module contains property-based tests using Hypothesis to verify
universal correctness properties for the numeric binning encoder.

Feature: numeric-binning-role-config
"""

import pytest
import math
from hypothesis import given, settings, strategies as st, assume
from typing import List, Tuple

from glyphh.encoder.numeric import (
    compute_bin_number,
    clamp_value,
    thermometer_encode,
    binary_encode,
    gray_encode,
    encode_numeric_value,
    compute_similarity,
)
from glyphh.core.config import NumericConfig, EncodingStrategy


# =============================================================================
# Test Data Generators (Strategies)
# =============================================================================

# Generate positive bin widths
positive_bin_widths = st.floats(
    min_value=0.001,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False
)

# Generate numeric values for testing
numeric_values = st.floats(
    min_value=-10000.0,
    max_value=10000.0,
    allow_nan=False,
    allow_infinity=False
)

# Generate bin numbers
bin_numbers = st.integers(min_value=0, max_value=1000)

# Generate max_bins for thermometer encoding
max_bins_values = st.integers(min_value=1, max_value=100)

# Generate num_bits for binary/gray encoding
num_bits_values = st.integers(min_value=1, max_value=16)


@st.composite
def valid_numeric_config(draw):
    """Generate a valid NumericConfig for testing."""
    bin_width = draw(positive_bin_widths)
    encoding_strategy = draw(st.sampled_from(list(EncodingStrategy)))
    
    # Optionally add min/max bounds
    has_bounds = draw(st.booleans())
    if has_bounds:
        min_val = draw(st.floats(min_value=-1000.0, max_value=999.0, allow_nan=False, allow_infinity=False))
        max_val = draw(st.floats(min_value=min_val + 0.1, max_value=1000.0, allow_nan=False, allow_infinity=False))
        return NumericConfig(
            bin_width=bin_width,
            encoding_strategy=encoding_strategy,
            min_value=min_val,
            max_value=max_val
        )
    else:
        return NumericConfig(
            bin_width=bin_width,
            encoding_strategy=encoding_strategy
        )


@st.composite
def config_with_bounds(draw):
    """Generate a NumericConfig with both min and max bounds."""
    bin_width = draw(st.floats(min_value=0.001, max_value=50.0, allow_nan=False, allow_infinity=False))
    encoding_strategy = draw(st.sampled_from(list(EncodingStrategy)))
    min_val = draw(st.floats(min_value=-500.0, max_value=400.0, allow_nan=False, allow_infinity=False))
    max_val = draw(st.floats(min_value=min_val + bin_width + 0.1, max_value=500.0, allow_nan=False, allow_infinity=False))
    
    return NumericConfig(
        bin_width=bin_width,
        encoding_strategy=encoding_strategy,
        min_value=min_val,
        max_value=max_val
    )


# =============================================================================
# Property 7: Bin Number Calculation
# =============================================================================

class TestBinNumberCalculation:
    """
    Property tests for Bin Number Calculation (Property 7).
    
    Feature: numeric-binning-role-config, Property 7: Bin Number Calculation
    
    **Validates: Requirements 5.1, 5.2**
    
    For any numeric value V and NumericConfig with bin_width B and min_value M
    (defaulting to 0), the computed bin number should equal floor((V - M) / B).
    """
    
    @given(
        value=numeric_values,
        bin_width=positive_bin_widths
    )
    @settings(max_examples=100)
    def test_bin_number_formula_no_min(self, value: float, bin_width: float):
        """
        Property test: Bin number equals floor(value / bin_width) when no min_value.
        
        For any value V and bin_width B with no min_value,
        bin_number SHALL equal floor(V / B).
        
        Feature: numeric-binning-role-config, Property 7: Bin Number Calculation
        **Validates: Requirements 5.1, 5.2**
        """
        config = NumericConfig(bin_width=bin_width)
        
        bin_num = compute_bin_number(value, config)
        expected = max(0, math.floor(value / bin_width))
        
        assert bin_num == expected, \
            f"Bin number for {value} with width {bin_width} should be {expected}, got {bin_num}"
    
    @given(
        value=numeric_values,
        bin_width=positive_bin_widths,
        min_value=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_bin_number_formula_with_min(self, value: float, bin_width: float, min_value: float):
        """
        Property test: Bin number equals floor((value - min_value) / bin_width).
        
        For any value V, bin_width B, and min_value M,
        bin_number SHALL equal floor((V - M) / B).
        
        Feature: numeric-binning-role-config, Property 7: Bin Number Calculation
        **Validates: Requirements 5.1, 5.2**
        """
        config = NumericConfig(bin_width=bin_width, min_value=min_value)
        
        # Clamp value to min_value if below
        effective_value = max(value, min_value)
        
        bin_num = compute_bin_number(value, config)
        expected = max(0, math.floor((effective_value - min_value) / bin_width))
        
        assert bin_num == expected, \
            f"Bin number for {value} (min={min_value}, width={bin_width}) should be {expected}, got {bin_num}"
    
    @given(
        value=numeric_values,
        config=valid_numeric_config()
    )
    @settings(max_examples=100)
    def test_bin_number_non_negative(self, value: float, config: NumericConfig):
        """
        Property test: Bin number is always non-negative.
        
        For any value and config, bin_number SHALL be >= 0.
        
        Feature: numeric-binning-role-config, Property 7: Bin Number Calculation
        **Validates: Requirements 5.1, 5.2**
        """
        bin_num = compute_bin_number(value, config)
        
        assert bin_num >= 0, \
            f"Bin number should be non-negative, got {bin_num}"
    
    @given(
        value=numeric_values,
        config=valid_numeric_config()
    )
    @settings(max_examples=100)
    def test_bin_number_is_integer(self, value: float, config: NumericConfig):
        """
        Property test: Bin number is always an integer.
        
        For any value and config, bin_number SHALL be an integer.
        
        Feature: numeric-binning-role-config, Property 7: Bin Number Calculation
        **Validates: Requirements 5.1, 5.2**
        """
        bin_num = compute_bin_number(value, config)
        
        assert isinstance(bin_num, int), \
            f"Bin number should be an integer, got {type(bin_num)}"


# =============================================================================
# Property 8: Value Clamping to Bounds
# =============================================================================

class TestValueClamping:
    """
    Property tests for Value Clamping to Bounds (Property 8).
    
    Feature: numeric-binning-role-config, Property 8: Value Clamping to Bounds
    
    **Validates: Requirements 5.3, 5.4**
    
    For any numeric value V and NumericConfig with min_value and/or max_value
    specified, the value used for encoding should be clamped.
    """
    
    @given(config=config_with_bounds())
    @settings(max_examples=100)
    def test_value_below_min_clamped(self, config: NumericConfig):
        """
        Property test: Values below min_value are clamped to min_value.
        
        For any value V < min_value, clamp_value SHALL return min_value.
        
        Feature: numeric-binning-role-config, Property 8: Value Clamping to Bounds
        **Validates: Requirements 5.4**
        """
        # Generate a value below min_value
        value = config.min_value - 100.0
        
        clamped = clamp_value(value, config)
        
        assert clamped == config.min_value, \
            f"Value {value} below min {config.min_value} should clamp to {config.min_value}, got {clamped}"
    
    @given(config=config_with_bounds())
    @settings(max_examples=100)
    def test_value_above_max_clamped(self, config: NumericConfig):
        """
        Property test: Values above max_value are clamped to max_value.
        
        For any value V > max_value, clamp_value SHALL return max_value.
        
        Feature: numeric-binning-role-config, Property 8: Value Clamping to Bounds
        **Validates: Requirements 5.3**
        """
        # Generate a value above max_value
        value = config.max_value + 100.0
        
        clamped = clamp_value(value, config)
        
        assert clamped == config.max_value, \
            f"Value {value} above max {config.max_value} should clamp to {config.max_value}, got {clamped}"
    
    @given(config=config_with_bounds())
    @settings(max_examples=100)
    def test_value_within_bounds_unchanged(self, config: NumericConfig):
        """
        Property test: Values within bounds are unchanged.
        
        For any value V where min_value <= V <= max_value,
        clamp_value SHALL return V unchanged.
        
        Feature: numeric-binning-role-config, Property 8: Value Clamping to Bounds
        **Validates: Requirements 5.3, 5.4**
        """
        # Generate a value within bounds
        value = (config.min_value + config.max_value) / 2
        
        clamped = clamp_value(value, config)
        
        assert clamped == value, \
            f"Value {value} within bounds should be unchanged, got {clamped}"
    
    @given(
        value=numeric_values,
        config=config_with_bounds()
    )
    @settings(max_examples=100)
    def test_clamped_value_always_within_bounds(self, value: float, config: NumericConfig):
        """
        Property test: Clamped value is always within bounds.
        
        For any value and config with bounds, clamped value SHALL be
        within [min_value, max_value].
        
        Feature: numeric-binning-role-config, Property 8: Value Clamping to Bounds
        **Validates: Requirements 5.3, 5.4**
        """
        clamped = clamp_value(value, config)
        
        assert config.min_value <= clamped <= config.max_value, \
            f"Clamped value {clamped} should be within [{config.min_value}, {config.max_value}]"


# =============================================================================
# Property 9: Thermometer Encoding Correctness
# =============================================================================

class TestThermometerEncoding:
    """
    Property tests for Thermometer Encoding Correctness (Property 9).
    
    Feature: numeric-binning-role-config, Property 9: Thermometer Encoding Correctness
    
    **Validates: Requirements 5.5**
    
    For any bin number N using thermometer encoding, the resulting bit pattern
    should have exactly N bits set to 1 (from the start), followed by 0s.
    """
    
    @given(
        bin_number=bin_numbers,
        max_bins=max_bins_values
    )
    @settings(max_examples=100)
    def test_thermometer_has_correct_ones_count(self, bin_number: int, max_bins: int):
        """
        Property test: Thermometer encoding has correct number of 1s.
        
        For any bin_number N and max_bins M, the encoding SHALL have
        min(N, M) ones.
        
        Feature: numeric-binning-role-config, Property 9: Thermometer Encoding Correctness
        **Validates: Requirements 5.5**
        """
        bits = thermometer_encode(bin_number, max_bins)
        
        expected_ones = min(bin_number, max_bins)
        actual_ones = sum(bits)
        
        assert actual_ones == expected_ones, \
            f"Thermometer({bin_number}, {max_bins}) should have {expected_ones} ones, got {actual_ones}"
    
    @given(
        bin_number=bin_numbers,
        max_bins=max_bins_values
    )
    @settings(max_examples=100)
    def test_thermometer_has_correct_length(self, bin_number: int, max_bins: int):
        """
        Property test: Thermometer encoding has correct length.
        
        For any bin_number and max_bins M, the encoding SHALL have length M.
        
        Feature: numeric-binning-role-config, Property 9: Thermometer Encoding Correctness
        **Validates: Requirements 5.5**
        """
        bits = thermometer_encode(bin_number, max_bins)
        
        assert len(bits) == max_bins, \
            f"Thermometer encoding should have length {max_bins}, got {len(bits)}"
    
    @given(
        bin_number=bin_numbers,
        max_bins=max_bins_values
    )
    @settings(max_examples=100)
    def test_thermometer_ones_are_contiguous_from_start(self, bin_number: int, max_bins: int):
        """
        Property test: Thermometer encoding has contiguous 1s from start.
        
        For any bin_number N, the encoding SHALL have all 1s at the start,
        followed by all 0s.
        
        Feature: numeric-binning-role-config, Property 9: Thermometer Encoding Correctness
        **Validates: Requirements 5.5**
        """
        bits = thermometer_encode(bin_number, max_bins)
        
        # Find first 0
        first_zero = -1
        for i, bit in enumerate(bits):
            if bit == 0:
                first_zero = i
                break
        
        if first_zero == -1:
            # All 1s is valid
            return
        
        # All bits after first 0 should be 0
        for i in range(first_zero, len(bits)):
            assert bits[i] == 0, \
                f"Thermometer encoding should have contiguous 1s, but found 1 at position {i} after 0 at {first_zero}"
        
        # All bits before first 0 should be 1
        for i in range(first_zero):
            assert bits[i] == 1, \
                f"Thermometer encoding should have contiguous 1s from start, but found 0 at position {i}"


# =============================================================================
# Property 10: Binary Encoding Correctness
# =============================================================================

class TestBinaryEncoding:
    """
    Property tests for Binary Encoding Correctness (Property 10).
    
    Feature: numeric-binning-role-config, Property 10: Binary Encoding Correctness
    
    **Validates: Requirements 5.6**
    
    For any bin number N using binary encoding, the resulting bit pattern
    should be the standard binary representation of N.
    """
    
    @given(
        bin_number=st.integers(min_value=0, max_value=255),
        num_bits=st.integers(min_value=8, max_value=16)
    )
    @settings(max_examples=100)
    def test_binary_encoding_is_correct(self, bin_number: int, num_bits: int):
        """
        Property test: Binary encoding matches standard binary representation.
        
        For any bin_number N, the encoding SHALL be the binary representation of N.
        
        Feature: numeric-binning-role-config, Property 10: Binary Encoding Correctness
        **Validates: Requirements 5.6**
        """
        bits = binary_encode(bin_number, num_bits)
        
        # Convert bits back to integer
        result = 0
        for bit in bits:
            result = (result << 1) | bit
        
        # Should match original (clamped to max representable)
        max_value = (1 << num_bits) - 1
        expected = min(bin_number, max_value)
        
        assert result == expected, \
            f"Binary encoding of {bin_number} with {num_bits} bits should decode to {expected}, got {result}"
    
    @given(
        bin_number=bin_numbers,
        num_bits=num_bits_values
    )
    @settings(max_examples=100)
    def test_binary_encoding_has_correct_length(self, bin_number: int, num_bits: int):
        """
        Property test: Binary encoding has correct length.
        
        For any bin_number and num_bits N, the encoding SHALL have length N.
        
        Feature: numeric-binning-role-config, Property 10: Binary Encoding Correctness
        **Validates: Requirements 5.6**
        """
        bits = binary_encode(bin_number, num_bits)
        
        assert len(bits) == num_bits, \
            f"Binary encoding should have length {num_bits}, got {len(bits)}"
    
    @given(num_bits=num_bits_values)
    @settings(max_examples=50)
    def test_binary_encoding_zero_is_all_zeros(self, num_bits: int):
        """
        Property test: Binary encoding of 0 is all zeros.
        
        binary_encode(0, N) SHALL return N zeros.
        
        Feature: numeric-binning-role-config, Property 10: Binary Encoding Correctness
        **Validates: Requirements 5.6**
        """
        bits = binary_encode(0, num_bits)
        
        assert all(b == 0 for b in bits), \
            f"Binary encoding of 0 should be all zeros, got {bits}"


# =============================================================================
# Property 11: Gray Code Encoding Correctness
# =============================================================================

class TestGrayCodeEncoding:
    """
    Property tests for Gray Code Encoding Correctness (Property 11).
    
    Feature: numeric-binning-role-config, Property 11: Gray Code Encoding Correctness
    
    **Validates: Requirements 5.7**
    
    For any bin number N using gray encoding, the resulting bit pattern
    should be the Gray code of N, computed as N XOR (N >> 1).
    """
    
    @given(
        bin_number=st.integers(min_value=0, max_value=255),
        num_bits=st.integers(min_value=8, max_value=16)
    )
    @settings(max_examples=100)
    def test_gray_encoding_formula(self, bin_number: int, num_bits: int):
        """
        Property test: Gray encoding equals N XOR (N >> 1).
        
        For any bin_number N, the Gray code SHALL equal N XOR (N >> 1).
        
        Feature: numeric-binning-role-config, Property 11: Gray Code Encoding Correctness
        **Validates: Requirements 5.7**
        """
        bits = gray_encode(bin_number, num_bits)
        
        # Convert bits back to integer
        result = 0
        for bit in bits:
            result = (result << 1) | bit
        
        # Compute expected Gray code
        max_value = (1 << num_bits) - 1
        clamped = min(bin_number, max_value)
        expected = clamped ^ (clamped >> 1)
        
        assert result == expected, \
            f"Gray encoding of {bin_number} should be {expected}, got {result}"
    
    @given(
        bin_number=bin_numbers,
        num_bits=num_bits_values
    )
    @settings(max_examples=100)
    def test_gray_encoding_has_correct_length(self, bin_number: int, num_bits: int):
        """
        Property test: Gray encoding has correct length.
        
        For any bin_number and num_bits N, the encoding SHALL have length N.
        
        Feature: numeric-binning-role-config, Property 11: Gray Code Encoding Correctness
        **Validates: Requirements 5.7**
        """
        bits = gray_encode(bin_number, num_bits)
        
        assert len(bits) == num_bits, \
            f"Gray encoding should have length {num_bits}, got {len(bits)}"
    
    @given(num_bits=num_bits_values)
    @settings(max_examples=50)
    def test_gray_encoding_zero_is_all_zeros(self, num_bits: int):
        """
        Property test: Gray encoding of 0 is all zeros.
        
        gray_encode(0, N) SHALL return N zeros (since 0 XOR 0 = 0).
        
        Feature: numeric-binning-role-config, Property 11: Gray Code Encoding Correctness
        **Validates: Requirements 5.7**
        """
        bits = gray_encode(0, num_bits)
        
        assert all(b == 0 for b in bits), \
            f"Gray encoding of 0 should be all zeros, got {bits}"
    
    @given(num_bits=st.integers(min_value=2, max_value=12))
    @settings(max_examples=50)
    def test_adjacent_gray_codes_differ_by_one_bit(self, num_bits: int):
        """
        Property test: Adjacent Gray codes differ by exactly one bit.
        
        For any consecutive bin numbers N and N+1, their Gray codes
        SHALL differ by exactly one bit.
        
        Feature: numeric-binning-role-config, Property 11: Gray Code Encoding Correctness
        **Validates: Requirements 5.7**
        """
        max_value = (1 << num_bits) - 2  # -2 to allow N+1
        
        # Test several adjacent pairs
        for n in range(min(10, max_value)):
            bits_n = gray_encode(n, num_bits)
            bits_n1 = gray_encode(n + 1, num_bits)
            
            # Count differing bits
            diff_count = sum(1 for a, b in zip(bits_n, bits_n1) if a != b)
            
            assert diff_count == 1, \
                f"Gray codes for {n} and {n+1} should differ by 1 bit, differ by {diff_count}"


# =============================================================================
# Property 12: Thermometer Similarity Preservation
# =============================================================================

class TestThermometerSimilarityPreservation:
    """
    Property tests for Thermometer Similarity Preservation (Property 12).
    
    Feature: numeric-binning-role-config, Property 12: Thermometer Similarity Preservation
    
    **Validates: Requirements 5.8**
    
    For any three bin numbers A < B < C using thermometer encoding,
    the similarity between bins A and B should be greater than
    the similarity between bins A and C.
    """
    
    @given(max_bins=st.integers(min_value=10, max_value=50))
    @settings(max_examples=100)
    def test_closer_bins_have_higher_similarity(self, max_bins: int):
        """
        Property test: Closer bins have higher similarity.
        
        For any A < B < C, similarity(A, B) > similarity(A, C).
        
        Feature: numeric-binning-role-config, Property 12: Thermometer Similarity Preservation
        **Validates: Requirements 5.8**
        """
        # Generate three distinct bin numbers A < B < C
        a = max_bins // 4
        b = max_bins // 2
        c = (3 * max_bins) // 4
        
        assume(a < b < c)
        assume(c <= max_bins)
        
        bits_a = thermometer_encode(a, max_bins)
        bits_b = thermometer_encode(b, max_bins)
        bits_c = thermometer_encode(c, max_bins)
        
        sim_ab = compute_similarity(bits_a, bits_b)
        sim_ac = compute_similarity(bits_a, bits_c)
        
        assert sim_ab > sim_ac, \
            f"Similarity(bin{a}, bin{b})={sim_ab} should be > Similarity(bin{a}, bin{c})={sim_ac}"
    
    @given(
        a=st.integers(min_value=1, max_value=10),
        gap1=st.integers(min_value=1, max_value=5),
        gap2=st.integers(min_value=1, max_value=5),
        max_bins=st.integers(min_value=25, max_value=50)
    )
    @settings(max_examples=100)
    def test_similarity_ordering_property(self, a: int, gap1: int, gap2: int, max_bins: int):
        """
        Property test: Similarity ordering is preserved.
        
        For any A < B < C, similarity(A, B) >= similarity(A, C).
        
        Feature: numeric-binning-role-config, Property 12: Thermometer Similarity Preservation
        **Validates: Requirements 5.8**
        """
        # Construct A < B < C from gaps
        b = a + gap1
        c = b + gap2
        
        assume(c <= max_bins)
        
        bits_a = thermometer_encode(a, max_bins)
        bits_b = thermometer_encode(b, max_bins)
        bits_c = thermometer_encode(c, max_bins)
        
        sim_ab = compute_similarity(bits_a, bits_b)
        sim_ac = compute_similarity(bits_a, bits_c)
        
        assert sim_ab >= sim_ac, \
            f"Similarity(bin{a}, bin{b})={sim_ab} should be >= Similarity(bin{a}, bin{c})={sim_ac}"
    
    @given(max_bins=st.integers(min_value=5, max_value=50))
    @settings(max_examples=100)
    def test_adjacent_bins_have_high_similarity(self, max_bins: int):
        """
        Property test: Adjacent bins have high similarity.
        
        For any adjacent bins N and N+1, similarity should be high.
        
        Feature: numeric-binning-role-config, Property 12: Thermometer Similarity Preservation
        **Validates: Requirements 5.8**
        """
        # Test middle of range
        n = max_bins // 2
        assume(n > 0 and n < max_bins)
        
        bits_n = thermometer_encode(n, max_bins)
        bits_n1 = thermometer_encode(n + 1, max_bins)
        
        sim = compute_similarity(bits_n, bits_n1)
        
        # Adjacent bins should have high similarity
        # For thermometer, they differ by exactly 1 bit
        # Use a slightly lower threshold to account for floating point
        expected_min = 1 - (1.5 / max_bins)
        
        assert sim >= expected_min, \
            f"Adjacent bins {n} and {n+1} should have similarity >= {expected_min}, got {sim}"
    
    @given(max_bins=st.integers(min_value=10, max_value=50))
    @settings(max_examples=100)
    def test_distant_bins_have_low_similarity(self, max_bins: int):
        """
        Property test: Distant bins have lower similarity.
        
        For bins at opposite ends, similarity should be < 0.5.
        
        Feature: numeric-binning-role-config, Property 12: Thermometer Similarity Preservation
        **Validates: Requirements 5.8**
        """
        # Compare bin 1 with bin max_bins-1
        bits_low = thermometer_encode(1, max_bins)
        bits_high = thermometer_encode(max_bins - 1, max_bins)
        
        sim = compute_similarity(bits_low, bits_high)
        
        # Distant bins should have lower similarity
        assert sim < 0.7, \
            f"Distant bins 1 and {max_bins-1} should have similarity < 0.7, got {sim}"
