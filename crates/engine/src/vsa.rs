//! VSA (Vector Symbolic Architecture) operations.
//!
//! Core hyperdimensional computing primitives:
//! - `bind`: element-wise multiplication (role-value association)
//! - `bundle`: majority-vote aggregation (concept combination)
//! - `weighted_bundle`: weighted majority-vote
//! - `cosine_similarity`: normalized dot product [-1, 1]
//! - `hamming_similarity`: proportion of agreeing dimensions [0, 1]
//! - `generate_symbol`: deterministic bipolar vector generation

use crate::error::{EngineError, EngineResult};
use crate::types::Vector;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Bind: element-wise multiplication of two bipolar vectors.
///
/// Properties:
/// - Inverse: bind(bind(r, v), r) = v
/// - Commutative: bind(r, v) = bind(v, r)
/// - Self-inverse: bind(v, v) = identity (all 1s)
pub fn bind(r: &Vector, v: &Vector) -> EngineResult<Vector> {
    if r.dimension != v.dimension {
        return Err(EngineError::DimensionMismatch {
            expected: r.dimension,
            actual: v.dimension,
        });
    }
    if r.space_id != v.space_id {
        return Err(EngineError::VectorSpaceMismatch {
            expected: r.space_id.clone(),
            actual: v.space_id.clone(),
        });
    }
    let data: Vec<i8> = r.data.iter().zip(v.data.iter()).map(|(a, b)| a * b).collect();
    Ok(Vector {
        data,
        dimension: r.dimension,
        space_id: r.space_id.clone(),
    })
}

/// Bundle: majority-vote aggregation of multiple bipolar vectors.
pub fn bundle(vectors: &[&Vector]) -> EngineResult<Vector> {
    if vectors.is_empty() {
        return Err(EngineError::Other("cannot bundle empty vector list".into()));
    }
    let dim = vectors[0].dimension;
    let space_id = &vectors[0].space_id;

    for (_i, v) in vectors.iter().enumerate() {
        if v.dimension != dim {
            return Err(EngineError::DimensionMismatch {
                expected: dim,
                actual: v.dimension,
            });
        }
        if v.space_id != *space_id {
            return Err(EngineError::VectorSpaceMismatch {
                expected: space_id.clone(),
                actual: v.space_id.clone(),
            });
        }
    }

    let mut sums = vec![0i32; dim];
    for v in vectors {
        for (i, &val) in v.data.iter().enumerate() {
            sums[i] += val as i32;
        }
    }
    let data: Vec<i8> = sums.iter().map(|&s| if s >= 0 { 1 } else { -1 }).collect();
    Ok(Vector {
        data,
        dimension: dim,
        space_id: space_id.clone(),
    })
}

/// Weighted bundle: weighted majority-vote aggregation.
pub fn weighted_bundle(weighted: &[(&Vector, f64)]) -> EngineResult<Vector> {
    if weighted.is_empty() {
        return Err(EngineError::Other(
            "cannot bundle empty vector list".into(),
        ));
    }
    let dim = weighted[0].0.dimension;
    let space_id = &weighted[0].0.space_id;

    let mut sums = vec![0.0f64; dim];
    for (v, w) in weighted {
        if v.dimension != dim {
            return Err(EngineError::DimensionMismatch {
                expected: dim,
                actual: v.dimension,
            });
        }
        for (i, &val) in v.data.iter().enumerate() {
            sums[i] += val as f64 * w;
        }
    }
    let data: Vec<i8> = sums.iter().map(|&s| if s >= 0.0 { 1 } else { -1 }).collect();
    Ok(Vector {
        data,
        dimension: dim,
        space_id: space_id.clone(),
    })
}

/// Cosine similarity for bipolar vectors. Range: [-1, 1].
pub fn cosine_similarity(v1: &Vector, v2: &Vector) -> EngineResult<f64> {
    if v1.dimension != v2.dimension {
        return Err(EngineError::DimensionMismatch {
            expected: v1.dimension,
            actual: v2.dimension,
        });
    }
    let dot: i64 = v1
        .data
        .iter()
        .zip(v2.data.iter())
        .map(|(&a, &b)| a as i64 * b as i64)
        .sum();
    let norm = v1.dimension as f64; // bipolar: ||v|| = sqrt(dim), so norm product = dim
    Ok(dot as f64 / norm)
}

/// Hamming similarity for bipolar vectors. Range: [0, 1].
/// Relationship: hamming = (cosine + 1) / 2
pub fn hamming_similarity(v1: &Vector, v2: &Vector) -> EngineResult<f64> {
    if v1.dimension != v2.dimension {
        return Err(EngineError::DimensionMismatch {
            expected: v1.dimension,
            actual: v2.dimension,
        });
    }
    let agree: usize = v1
        .data
        .iter()
        .zip(v2.data.iter())
        .filter(|(&a, &b)| a == b)
        .count();
    Ok(agree as f64 / v1.dimension as f64)
}

/// Generate a deterministic bipolar symbol vector from seed + key.
pub fn generate_symbol(seed: u64, key: &str, dimension: usize, space_id: &str) -> Vector {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    key.hash(&mut hasher);
    let combined_seed = hasher.finish();

    let mut rng = ChaCha8Rng::seed_from_u64(combined_seed);
    let data: Vec<i8> = (0..dimension)
        .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
        .collect();

    Vector {
        data,
        dimension,
        space_id: space_id.to_string(),
    }
}

/// Compute a deterministic space_id from dimension, seed, and config JSON.
pub fn compute_space_id(dimension: usize, seed: u64, config_json: &str) -> String {
    let input = format!("{dimension}:{seed}:{config_json}");
    let hash = Sha256::digest(input.as_bytes());
    hex::encode(&hash[..8]) // 16 hex chars
}

// We need hex encoding — inline it to avoid an extra dep
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(data: Vec<i8>, space: &str) -> Vector {
        let dim = data.len();
        Vector {
            data,
            dimension: dim,
            space_id: space.to_string(),
        }
    }

    #[test]
    fn bind_inverse_property() {
        let r = make_vec(vec![1, -1, 1, -1, 1], "test");
        let v = make_vec(vec![-1, 1, -1, 1, -1], "test");
        let bound = bind(&r, &v).unwrap();
        let recovered = bind(&bound, &r).unwrap();
        assert_eq!(recovered.data, v.data);
    }

    #[test]
    fn bundle_commutativity() {
        let a = make_vec(vec![1, -1, 1, -1, 1], "test");
        let b = make_vec(vec![-1, 1, -1, 1, -1], "test");
        let c = make_vec(vec![1, 1, -1, -1, 1], "test");
        let b1 = bundle(&[&a, &b, &c]).unwrap();
        let b2 = bundle(&[&c, &a, &b]).unwrap();
        assert_eq!(b1.data, b2.data);
    }

    #[test]
    fn cosine_identical_vectors() {
        let v = make_vec(vec![1, -1, 1, -1], "test");
        let score = cosine_similarity(&v, &v).unwrap();
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let v1 = make_vec(vec![1, -1, 1, -1], "test");
        let v2 = make_vec(vec![-1, 1, -1, 1], "test");
        let score = cosine_similarity(&v1, &v2).unwrap();
        assert!((score + 1.0).abs() < 1e-10);
    }

    #[test]
    fn hamming_cosine_relationship() {
        let v1 = make_vec(vec![1, -1, 1, 1], "test");
        let v2 = make_vec(vec![1, 1, 1, -1], "test");
        let cos = cosine_similarity(&v1, &v2).unwrap();
        let ham = hamming_similarity(&v1, &v2).unwrap();
        assert!((ham - (cos + 1.0) / 2.0).abs() < 1e-10);
    }

    #[test]
    fn generate_symbol_deterministic() {
        let v1 = generate_symbol(42, "color", 100, "test");
        let v2 = generate_symbol(42, "color", 100, "test");
        assert_eq!(v1.data, v2.data);
    }

    #[test]
    fn generate_symbol_different_keys() {
        let v1 = generate_symbol(42, "color", 1000, "test");
        let v2 = generate_symbol(42, "shape", 1000, "test");
        assert_ne!(v1.data, v2.data);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let v1 = make_vec(vec![1, -1, 1], "test");
        let v2 = make_vec(vec![1, -1], "test");
        assert!(bind(&v1, &v2).is_err());
        assert!(cosine_similarity(&v1, &v2).is_err());
    }

    #[test]
    fn space_mismatch_errors() {
        let v1 = make_vec(vec![1, -1], "space_a");
        let v2 = make_vec(vec![1, -1], "space_b");
        assert!(bind(&v1, &v2).is_err());
    }
}
