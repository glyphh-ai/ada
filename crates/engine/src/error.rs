//! Engine error types.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("bipolar constraint violated: found values outside {{-1, +1}}")]
    BipolarConstraint,

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("vector space mismatch: expected {expected}, got {actual}")]
    VectorSpaceMismatch { expected: String, actual: String },

    #[error("configuration error: {field} — {reason}")]
    Configuration { field: String, reason: String },

    #[error("encoding error in '{concept}' at stage '{stage}': {reason}")]
    Encoding {
        concept: String,
        stage: String,
        reason: String,
    },

    #[error("GQL parse error at {line}:{column}: {message}")]
    GqlParse {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("model validation: {0}")]
    ModelValidation(String),

    #[error("{0}")]
    Other(String),
}

pub type EngineResult<T> = Result<T, EngineError>;
