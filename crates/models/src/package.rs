//! .glyphh model file format: serialize/deserialize compressed JSON model packs.

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use glyphh_engine::config::EncoderConfig;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::Path;

/// A packaged model for deployment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlyphhModel {
    pub name: String,
    pub version: String,
    pub encoder_config: EncoderConfig,
    /// Serialized glyphs (kept as JSON values to avoid full deserialization on load)
    #[serde(default)]
    pub glyphs: Vec<serde_json::Value>,
    #[serde(default)]
    pub intent_patterns: Option<serde_json::Value>,
    #[serde(default)]
    pub stored_procedures: Vec<serde_json::Value>,
    #[serde(default)]
    pub concepts: Option<Vec<serde_json::Value>>,
    pub readme: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub created_at: String,
}

impl GlyphhModel {
    /// Load a model from a .glyphh file (gzipped JSON).
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        let mut json_str = String::new();
        decoder.read_to_string(&mut json_str)?;
        let model: Self = serde_json::from_str(&json_str)?;
        Ok(model)
    }

    /// Save a model to a .glyphh file (gzipped JSON).
    pub fn to_file(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json_str = serde_json::to_string_pretty(self)?;
        let file = std::fs::File::create(path)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(json_str.as_bytes())?;
        encoder.finish()?;
        Ok(())
    }
}
