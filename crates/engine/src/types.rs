//! Core data types: Vector, Concept, Segment, Layer, Glyph, Edge.

use crate::error::{EngineError, EngineResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bipolar vector in {-1, +1}.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vector {
    pub data: Vec<i8>,
    pub dimension: usize,
    pub space_id: String,
}

impl Vector {
    pub fn new(data: Vec<i8>, dimension: usize, space_id: String) -> EngineResult<Self> {
        if data.len() != dimension {
            return Err(EngineError::DimensionMismatch {
                expected: dimension,
                actual: data.len(),
            });
        }
        if !data.iter().all(|&v| v == 1 || v == -1) {
            return Err(EngineError::BipolarConstraint);
        }
        Ok(Self {
            data,
            dimension,
            space_id,
        })
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        self.space_id == other.space_id
            && self.dimension == other.dimension
            && self.data == other.data
    }
}
impl Eq for Vector {}

/// Structured concept for encoding input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Concept {
    pub name: String,
    pub attributes: HashMap<String, serde_json::Value>,
    pub relationships: Vec<(String, String)>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Segment-level encoding structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub name: String,
    pub cortex: Vector,
    pub roles: HashMap<String, Vector>,
    pub role_values: HashMap<String, serde_json::Value>,
    pub weights: HashMap<String, f64>,
}

/// Layer-level encoding structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
    pub name: String,
    pub cortex: Vector,
    pub segments: HashMap<String, Segment>,
    pub weights: HashMap<String, f64>,
}

/// Encoded concept — the fundamental unit of encoded knowledge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Glyph {
    /// Composite identifier: primary_key@timestamp#version
    pub identifier: String,
    pub name: String,
    pub space_id: String,
    pub global_cortex: Vector,
    pub layers: HashMap<String, Layer>,
    pub security_levels: HashMap<String, f64>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    pub version: String,
}

/// Edge types for similarity computation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    NeuralCortex,
    NeuralLayer,
    NeuralSegment,
    NeuralRole,
    TemporalCortex,
    TemporalLayer,
    TemporalSegment,
    TemporalRole,
}

impl EdgeType {
    pub fn from_str(s: &str) -> EngineResult<Self> {
        match s {
            "neural_cortex" => Ok(Self::NeuralCortex),
            "neural_layer" => Ok(Self::NeuralLayer),
            "neural_segment" => Ok(Self::NeuralSegment),
            "neural_role" => Ok(Self::NeuralRole),
            "temporal_cortex" => Ok(Self::TemporalCortex),
            "temporal_layer" => Ok(Self::TemporalLayer),
            "temporal_segment" => Ok(Self::TemporalSegment),
            "temporal_role" => Ok(Self::TemporalRole),
            _ => Err(EngineError::Other(format!("invalid edge type: {s}"))),
        }
    }
}

/// Edge connecting glyphs for similarity computation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub edge_type: EdgeType,
    pub source: String,
    pub target: Option<String>,
    pub vector: Vector,
    pub weights: HashMap<String, f64>,
    pub security_level: f64,
    pub layer: Option<usize>,
    pub segment: Option<usize>,
    pub role: Option<String>,
}
