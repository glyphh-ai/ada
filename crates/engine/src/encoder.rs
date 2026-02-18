//! Hierarchical encoder: concepts → glyphs.
//!
//! Encodes structured concepts into bipolar hyperdimensional vectors
//! using the layer → segment → role hierarchy defined in EncoderConfig.

use crate::config::EncoderConfig;
use crate::error::EngineResult;
use crate::types::Vector;
use crate::vsa;
use std::collections::HashMap;

/// The core encoder that transforms concepts into glyphs.
pub struct Encoder {
    pub dimension: usize,
    pub seed: u64,
    pub space_id: String,
    pub config: EncoderConfig,
    symbol_cache: HashMap<String, Vector>,
}

impl Encoder {
    pub fn new(config: EncoderConfig) -> EngineResult<Self> {
        config.validate()?;
        let space_id = vsa::compute_space_id(config.dimension, config.seed, &config.to_json());
        Ok(Self {
            dimension: config.dimension,
            seed: config.seed,
            space_id,
            config,
            symbol_cache: HashMap::new(),
        })
    }

    /// Generate (or retrieve from cache) a deterministic bipolar symbol vector.
    pub fn generate_symbol(&mut self, key: &str) -> Vector {
        if let Some(v) = self.symbol_cache.get(key) {
            return v.clone();
        }
        let v = vsa::generate_symbol(self.seed, key, self.dimension, &self.space_id);
        self.symbol_cache.insert(key.to_string(), v.clone());
        v
    }

    /// Bind a role vector with a value vector.
    pub fn bind(&self, role: &Vector, value: &Vector) -> EngineResult<Vector> {
        vsa::bind(role, value)
    }

    /// Bundle multiple vectors via majority vote.
    pub fn bundle(&self, vectors: &[&Vector]) -> EngineResult<Vector> {
        vsa::bundle(vectors)
    }

    /// Clear the symbol cache.
    pub fn clear_cache(&mut self) {
        self.symbol_cache.clear();
    }

    // TODO: encode(concept) -> Glyph
    // TODO: encode_segment(name, attributes, weights) -> Segment
    // TODO: _encode_numeric_value(value, numeric_config) -> Vector
}
