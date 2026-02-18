"""
GQL Query Encoder - Encodes GQL queries as hypervectors.

The query encoder transforms AST nodes into hypervectors using the SDK's
HDC primitives. This enables semantic query caching, similarity-based
optimization, and query comparison.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from glyphh.gql.ast import (
    ASTNode,
    ASTVisitor,
    HierarchyLevel,
    GlyphRef,
    Condition,
    ComparisonCondition,
    LogicalCondition,
    SimilaritySearchNode,
    ListNode,
    CountNode,
    CompareNode,
    PredictNode,
    DriftDetectionNode,
    IntrospectNode,
    TrendNode,
    AggregateNode,
)
from glyphh.core import Vector
from glyphh.core.rust_ops import bind, bundle, generate_symbol, cosine_similarity


class QueryEncoder(ASTVisitor):
    """
    Encodes GQL queries as hypervectors for semantic operations.
    
    The encoder creates a "query cortex" - a hypervector representation
    of the query that captures its semantic meaning. Similar queries
    will have similar vectors.
    
    Encoding strategy:
    1. Encode operation type as a role binding
    2. Encode each parameter as a role binding
    3. Bundle all bindings into a query cortex
    
    Example:
        >>> from glyphh.gql import parse, QueryEncoder
        >>> encoder = QueryEncoder(dimension=10000, seed=42)
        >>> ast = parse('FIND SIMILAR TO "red car" LIMIT 10')
        >>> vector = encoder.encode(ast)
        >>> 
        >>> # Compare two queries
        >>> ast2 = parse('FIND SIMILAR TO "blue car" LIMIT 10')
        >>> vector2 = encoder.encode(ast2)
        >>> similarity = encoder.similarity(vector, vector2)
    """
    
    def __init__(self, dimension: int = 10000, seed: int = 42):
        """
        Initialize the query encoder.
        
        Args:
            dimension: Vector dimension (default 10000)
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.seed = seed
        self._symbol_cache: Dict[str, np.ndarray] = {}
    
    def encode(self, node: ASTNode) -> Vector:
        """
        Encode an AST node as a hypervector (query cortex).
        
        Args:
            node: The AST node to encode
        
        Returns:
            Vector representing the query
        """
        return node.accept(self)
    
    def similarity(self, v1: Vector, v2: Vector) -> float:
        """
        Compute semantic similarity between two query vectors.
        
        Args:
            v1: First query vector
            v2: Second query vector
        
        Returns:
            Cosine similarity score between 0 and 1
        """
        return cosine_similarity(v1.data, v2.data)
    
    def _get_symbol(self, name: str) -> np.ndarray:
        """Get or create a symbol vector for a name."""
        if name not in self._symbol_cache:
            self._symbol_cache[name] = generate_symbol(
                self.seed,
                name,
                self.dimension
            )
        return self._symbol_cache[name]
    
    def _bind_role(self, role_name: str, value_name: str) -> np.ndarray:
        """Create a role-value binding."""
        role_vec = self._get_symbol(f"role:{role_name}")
        value_vec = self._get_symbol(f"value:{value_name}")
        return bind(role_vec, value_vec)
    
    def _bundle_bindings(self, bindings: List[np.ndarray]) -> Vector:
        """Bundle multiple bindings into a cortex vector."""
        if not bindings:
            # Return zero vector if no bindings
            return Vector(
                data=np.zeros(self.dimension, dtype=np.int8),
                dimension=self.dimension,
                space_id="gql_query"
            )
        
        result = bundle(bindings)
        return Vector(
            data=result,
            dimension=self.dimension,
            space_id="gql_query"
        )
    
    def visit_similarity_search(self, node: SimilaritySearchNode) -> Vector:
        """Encode FIND SIMILAR query."""
        bindings = [
            self._bind_role("operation", "similarity_search"),
        ]
        
        # Encode target
        if isinstance(node.target, str):
            bindings.append(self._bind_role("target_type", "text"))
            # Encode text content (simplified - just use hash)
            bindings.append(self._bind_role("target_text", node.target))
        else:
            bindings.append(self._bind_role("target_type", "glyph_ref"))
            bindings.append(self._bind_role("target_id", node.target.identifier))
        
        # Encode scope
        if node.scope:
            bindings.append(self._bind_role("scope", node.scope))
        
        # Encode limit (quantized)
        limit_bucket = self._quantize_limit(node.limit)
        bindings.append(self._bind_role("limit", limit_bucket))
        
        # Encode threshold (quantized)
        threshold_bucket = self._quantize_threshold(node.threshold)
        bindings.append(self._bind_role("threshold", threshold_bucket))
        
        return self._bundle_bindings(bindings)
    
    def visit_list(self, node: ListNode) -> Vector:
        """Encode LIST query."""
        bindings = [
            self._bind_role("operation", "list"),
        ]
        
        if node.where:
            bindings.extend(self._encode_condition(node.where, "where"))
        
        limit_bucket = self._quantize_limit(node.limit)
        bindings.append(self._bind_role("limit", limit_bucket))
        
        return self._bundle_bindings(bindings)
    
    def visit_count(self, node: CountNode) -> Vector:
        """Encode COUNT query."""
        bindings = [
            self._bind_role("operation", "count"),
        ]
        
        if node.where:
            bindings.extend(self._encode_condition(node.where, "where"))
        
        return self._bundle_bindings(bindings)
    
    def visit_compare(self, node: CompareNode) -> Vector:
        """Encode COMPARE query."""
        bindings = [
            self._bind_role("operation", "compare"),
            self._bind_role("glyph1", node.glyph1.identifier),
            self._bind_role("glyph2", node.glyph2.identifier),
        ]
        
        bindings.extend(self._encode_hierarchy_level(node.level))
        
        return self._bundle_bindings(bindings)
    
    def visit_predict(self, node: PredictNode) -> Vector:
        """Encode PREDICT query."""
        bindings = [
            self._bind_role("operation", "predict"),
            self._bind_role("glyph", node.glyph.identifier),
        ]
        
        if node.role:
            bindings.append(self._bind_role("role_path", node.role))
        
        if node.window:
            bindings.append(self._bind_role("window", node.window))
        
        return self._bundle_bindings(bindings)
    
    def visit_drift_detection(self, node: DriftDetectionNode) -> Vector:
        """Encode DETECT DRIFT query."""
        bindings = [
            self._bind_role("operation", "drift_detection"),
            self._bind_role("from_glyph", node.from_glyph.identifier),
            self._bind_role("to_glyph", node.to_glyph.identifier),
        ]
        
        bindings.extend(self._encode_hierarchy_level(node.level))
        
        if node.threshold is not None:
            threshold_bucket = self._quantize_threshold(node.threshold)
            bindings.append(self._bind_role("threshold", threshold_bucket))
        
        return self._bundle_bindings(bindings)
    
    def visit_introspect(self, node: IntrospectNode) -> Vector:
        """Encode INTROSPECT query."""
        bindings = [
            self._bind_role("operation", "introspect"),
            self._bind_role("glyph", node.glyph.identifier),
            self._bind_role("decompose_by", node.decompose_by),
        ]
        
        return self._bundle_bindings(bindings)
    
    def visit_trend(self, node: TrendNode) -> Vector:
        """Encode TREND query."""
        bindings = [
            self._bind_role("operation", "trend"),
            self._bind_role("glyph", node.glyph.identifier),
            self._bind_role("window", node.window),
        ]
        
        bindings.extend(self._encode_hierarchy_level(node.level))
        
        if node.alert_condition:
            bindings.append(self._bind_role("alert", node.alert_condition))
        
        return self._bundle_bindings(bindings)
    
    def visit_aggregate(self, node: AggregateNode) -> Vector:
        """Encode AGGREGATE query."""
        bindings = [
            self._bind_role("operation", "aggregate"),
            self._bind_role("function", node.function),
        ]
        
        if node.field:
            bindings.append(self._bind_role("field", node.field))
        
        if node.where:
            bindings.extend(self._encode_condition(node.where, "where"))
        
        if node.group_by:
            for i, field in enumerate(node.group_by):
                bindings.append(self._bind_role(f"group_by_{i}", field))
        
        return self._bundle_bindings(bindings)
    
    def visit_comparison(self, node: ComparisonCondition) -> Vector:
        """Encode comparison condition (not typically called directly)."""
        bindings = self._encode_condition(node, "cond")
        return self._bundle_bindings(bindings)
    
    def visit_logical(self, node: LogicalCondition) -> Vector:
        """Encode logical condition (not typically called directly)."""
        bindings = self._encode_condition(node, "cond")
        return self._bundle_bindings(bindings)
    
    def _encode_condition(self, cond: Condition, prefix: str) -> List[np.ndarray]:
        """Encode a condition into role bindings."""
        bindings = []
        
        if isinstance(cond, ComparisonCondition):
            bindings.append(self._bind_role(f"{prefix}_type", "comparison"))
            bindings.append(self._bind_role(f"{prefix}_field", cond.field))
            bindings.append(self._bind_role(f"{prefix}_op", cond.operator))
            bindings.append(self._bind_role(f"{prefix}_value", str(cond.value)))
        
        elif isinstance(cond, LogicalCondition):
            bindings.append(self._bind_role(f"{prefix}_type", "logical"))
            bindings.append(self._bind_role(f"{prefix}_op", cond.operator))
            for i, operand in enumerate(cond.operands):
                bindings.extend(self._encode_condition(operand, f"{prefix}_{i}"))
        
        return bindings
    
    def _encode_hierarchy_level(self, level: HierarchyLevel) -> List[np.ndarray]:
        """Encode hierarchy level into role bindings."""
        bindings = [
            self._bind_role("level", level.level),
        ]
        if level.path:
            bindings.append(self._bind_role("level_path", level.path))
        return bindings
    
    def _quantize_limit(self, limit: int) -> str:
        """Quantize limit value into buckets for better similarity matching."""
        if limit <= 5:
            return "tiny"
        elif limit <= 10:
            return "small"
        elif limit <= 50:
            return "medium"
        elif limit <= 100:
            return "large"
        else:
            return "huge"
    
    def _quantize_threshold(self, threshold: float) -> str:
        """Quantize threshold value into buckets for better similarity matching."""
        if threshold <= 0.3:
            return "low"
        elif threshold <= 0.5:
            return "medium_low"
        elif threshold <= 0.7:
            return "medium"
        elif threshold <= 0.9:
            return "high"
        else:
            return "very_high"
