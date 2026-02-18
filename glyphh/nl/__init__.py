"""
Natural Language (NL) module for the Glyphh SDK.

This module provides auto-schema NL query matching functionality:
- SchemaVectorizer: Generates vectors for schema roles and values
- QueryTokenizer: Tokenizes queries with n-gram support
- SchemaMatcher: Matches query tokens against schema vectors
- IntentInferrer: Infers query intent from matches
- ParameterExtractor: Extracts parameters from matched tokens
- GQLGenerator: Generates GQL queries from intent and parameters
- AutoSchemaMatcher: Main interface combining all components

The auto-schema approach uses the existing HDC encoder to create bipolar vectors
for schema elements (role names, value labels), then matches incoming query tokens
against these vectors to understand what the user is asking about.
"""

from glyphh.nl.schema_vectorizer import SchemaVector
from glyphh.nl.query_tokenizer import Token, TokenizerConfig, QueryTokenizer
from glyphh.nl.schema_matcher import TokenMatch, MatchConfig, MatchResult, SchemaMatcher
from glyphh.nl.intent_inferrer import IntentKeywords, InferredIntent
from glyphh.nl.parameter_extractor import ExtractedParameter, ExtractionResult, ParameterExtractor
from glyphh.nl.gql_generator import GQLGenerator
from glyphh.nl.auto_schema_matcher import AutoMatchConfig, AutoMatchResult, AutoSchemaMatcher
from glyphh.nl.performance import (
    ANNIndex,
    ANNSearchResult,
    ANNIndexMetrics,
    OptimizedVectorOps,
    should_use_ann_index,
    create_ann_index_from_schema_vectors,
    ANN_THRESHOLD,
)

__all__ = [
    "SchemaVector",
    "Token",
    "TokenizerConfig",
    "QueryTokenizer",
    "TokenMatch",
    "MatchConfig",
    "MatchResult",
    "SchemaMatcher",
    "IntentKeywords",
    "InferredIntent",
    "ExtractedParameter",
    "ExtractionResult",
    "ParameterExtractor",
    "GQLGenerator",
    "AutoMatchConfig",
    "AutoMatchResult",
    "AutoSchemaMatcher",
    # Performance optimizations
    "ANNIndex",
    "ANNSearchResult",
    "ANNIndexMetrics",
    "OptimizedVectorOps",
    "should_use_ann_index",
    "create_ann_index_from_schema_vectors",
    "ANN_THRESHOLD",
]
