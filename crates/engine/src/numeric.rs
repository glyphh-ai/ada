//! Numeric binning: thermometer, binary, and gray code encoding.

use crate::config::{EncodingStrategy, NumericConfig};

/// Compute the bin number for a numeric value.
pub fn compute_bin_number(value: f64, config: &NumericConfig) -> i64 {
    let clamped = match (config.min_value, config.max_value) {
        (Some(min), Some(max)) => value.clamp(min, max),
        (Some(min), None) => value.max(min),
        (None, Some(max)) => value.min(max),
        (None, None) => value,
    };
    let offset = config.min_value.unwrap_or(0.0);
    ((clamped - offset) / config.bin_width).floor() as i64
}

/// Encode a bin number into a bit pattern using the configured strategy.
pub fn encode_bin(bin: i64, max_bins: usize, strategy: &EncodingStrategy) -> Vec<u8> {
    match strategy {
        EncodingStrategy::Thermometer => {
            let n = bin.max(0) as usize;
            (0..max_bins).map(|i| if i < n { 1 } else { 0 }).collect()
        }
        EncodingStrategy::Binary => {
            let n = bin.max(0) as usize;
            let bits = (max_bins as f64).log2().ceil() as usize;
            (0..bits).rev().map(|i| ((n >> i) & 1) as u8).collect()
        }
        EncodingStrategy::Gray => {
            let n = bin.max(0) as usize;
            let gray = n ^ (n >> 1);
            let bits = (max_bins as f64).log2().ceil() as usize;
            (0..bits).rev().map(|i| ((gray >> i) & 1) as u8).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermometer_encoding() {
        let bits = encode_bin(3, 5, &EncodingStrategy::Thermometer);
        assert_eq!(bits, vec![1, 1, 1, 0, 0]);
    }

    #[test]
    fn bin_number_with_bounds() {
        let config = NumericConfig {
            bin_width: 1.0,
            encoding_strategy: EncodingStrategy::Thermometer,
            min_value: Some(0.0),
            max_value: Some(10.0),
        };
        assert_eq!(compute_bin_number(7.2, &config), 7);
        assert_eq!(compute_bin_number(7.8, &config), 7);
        assert_eq!(compute_bin_number(0.0, &config), 0);
        assert_eq!(compute_bin_number(-5.0, &config), 0); // clamped
    }
}
