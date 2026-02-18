//! Encoder configuration: Role, Segment, Layer, NumericConfig, EncoderConfig.

use crate::error::{EngineError, EngineResult};
use serde::{Deserialize, Serialize};

/// Encoding strategy for numeric bins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingStrategy {
    Thermometer,
    Binary,
    Gray,
}

/// Configuration for numeric binning on a role.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericConfig {
    pub bin_width: f64,
    #[serde(default = "default_thermometer")]
    pub encoding_strategy: EncodingStrategy,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
}

fn default_thermometer() -> EncodingStrategy {
    EncodingStrategy::Thermometer
}

impl NumericConfig {
    pub fn validate(&self) -> EngineResult<()> {
        if self.bin_width <= 0.0 {
            return Err(EngineError::Configuration {
                field: "bin_width".into(),
                reason: format!("must be positive, got {}", self.bin_width),
            });
        }
        if let (Some(min), Some(max)) = (self.min_value, self.max_value) {
            if min >= max {
                return Err(EngineError::Configuration {
                    field: "min_value".into(),
                    reason: format!("min ({min}) must be less than max ({max})"),
                });
            }
        }
        Ok(())
    }
}

/// A role definition within a segment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoleConfig {
    pub name: String,
    #[serde(default = "one")]
    pub similarity_weight: f64,
    #[serde(default = "one")]
    pub security_weight: f64,
    #[serde(default)]
    pub key_part: bool,
    pub numeric_config: Option<NumericConfig>,
    pub lexicons: Option<Vec<String>>,
}

/// A segment definition within a layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentConfig {
    pub name: String,
    #[serde(default = "one")]
    pub similarity_weight: f64,
    #[serde(default = "one")]
    pub security_weight: f64,
    #[serde(default)]
    pub roles: Vec<RoleConfig>,
}

/// A layer definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerConfig {
    pub name: String,
    #[serde(default = "one")]
    pub similarity_weight: f64,
    #[serde(default = "one")]
    pub security_weight: f64,
    #[serde(default)]
    pub segments: Vec<SegmentConfig>,
}

/// Top-level encoder configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub dimension: usize,
    pub seed: u64,
    #[serde(default)]
    pub layers: Vec<LayerConfig>,
    #[serde(default = "default_temporal_source")]
    pub temporal_source: String,
}

fn one() -> f64 {
    1.0
}
fn default_temporal_source() -> String {
    "auto".into()
}

impl EncoderConfig {
    pub fn validate(&self) -> EngineResult<()> {
        if self.dimension == 0 {
            return Err(EngineError::Configuration {
                field: "dimension".into(),
                reason: "must be > 0".into(),
            });
        }
        for layer in &self.layers {
            for seg in &layer.segments {
                for role in &seg.roles {
                    if let Some(nc) = &role.numeric_config {
                        nc.validate()?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}
