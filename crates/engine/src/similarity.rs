//! Weighted similarity calculator with dual weighting (importance + security).

use crate::error::EngineResult;
use crate::types::Vector;
use crate::vsa;

/// Result of a similarity computation.
pub struct SimilarityResult {
    pub score: f64,
    pub visible: bool,
    pub raw_score: f64,
    pub similarity_weight: f64,
    pub security_weight: f64,
    pub metric: String,
}

/// Similarity calculator with configurable threshold and metric.
pub struct SimilarityCalculator {
    pub threshold: f64,
    pub default_metric: Metric,
}

#[derive(Clone, Copy, Debug)]
pub enum Metric {
    Cosine,
    Hamming,
}

impl SimilarityCalculator {
    pub fn new(threshold: f64, metric: Metric) -> Self {
        Self {
            threshold,
            default_metric: metric,
        }
    }

    /// Compute raw cosine similarity between two vectors.
    pub fn cosine(&self, v1: &Vector, v2: &Vector) -> EngineResult<f64> {
        vsa::cosine_similarity(v1, v2)
    }

    /// Compute raw hamming similarity between two vectors.
    pub fn hamming(&self, v1: &Vector, v2: &Vector) -> EngineResult<f64> {
        vsa::hamming_similarity(v1, v2)
    }

    /// Convert cosine [-1,1] to hamming [0,1].
    pub fn cosine_to_hamming(cosine: f64) -> f64 {
        (cosine + 1.0) / 2.0
    }

    /// Convert hamming [0,1] to cosine [-1,1].
    pub fn hamming_to_cosine(hamming: f64) -> f64 {
        2.0 * hamming - 1.0
    }

    // TODO: compute_similarity(glyph1, glyph2, edge_type, clearance, metric) -> SimilarityResult
    // TODO: _extract_edge_data(glyph, edge_type) -> (Vector, weights, security_level)
}
