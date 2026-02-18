//! Glyphh Engine — pure glyph execution, VSA math, encoding, similarity, GQL parsing.
//!
//! This crate contains the core hyperdimensional computing primitives:
//! - Bipolar vector operations (bind, bundle, similarity)
//! - Deterministic symbol generation
//! - Hierarchical encoder (layers → segments → roles)
//! - Similarity calculator with dual weighting (importance + security)
//! - GQL lexer, parser, and AST
//! - Fact tree construction
//! - Numeric binning with thermometer/binary/gray encoding

pub mod types;
pub mod config;
pub mod vsa;
pub mod encoder;
pub mod similarity;
pub mod gql;
pub mod fact_tree;
pub mod numeric;
pub mod error;
