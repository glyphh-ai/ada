"""
GQL Abstract Syntax Tree (AST) Node Types.

Defines all AST node types for representing parsed GQL queries.
Each node type corresponds to a GQL query operation or sub-expression.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


class ASTVisitor(ABC):
    """
    Visitor interface for AST traversal.
    
    Implement this interface to traverse and process AST nodes.
    Used by the pretty printer, planner, and other AST processors.
    """
    
    @abstractmethod
    def visit_similarity_search(self, node: 'SimilaritySearchNode') -> Any:
        """Visit a FIND SIMILAR node."""
        pass
    
    @abstractmethod
    def visit_list(self, node: 'ListNode') -> Any:
        """Visit a LIST node."""
        pass
    
    @abstractmethod
    def visit_count(self, node: 'CountNode') -> Any:
        """Visit a COUNT node."""
        pass
    
    @abstractmethod
    def visit_compare(self, node: 'CompareNode') -> Any:
        """Visit a COMPARE node."""
        pass
    
    @abstractmethod
    def visit_predict(self, node: 'PredictNode') -> Any:
        """Visit a PREDICT node."""
        pass
    
    @abstractmethod
    def visit_drift_detection(self, node: 'DriftDetectionNode') -> Any:
        """Visit a DETECT DRIFT node."""
        pass
    
    @abstractmethod
    def visit_introspect(self, node: 'IntrospectNode') -> Any:
        """Visit an INTROSPECT node."""
        pass
    
    @abstractmethod
    def visit_trend(self, node: 'TrendNode') -> Any:
        """Visit a TREND node."""
        pass
    
    @abstractmethod
    def visit_aggregate(self, node: 'AggregateNode') -> Any:
        """Visit an AGGREGATE node."""
        pass
    
    @abstractmethod
    def visit_comparison(self, node: 'ComparisonCondition') -> Any:
        """Visit a comparison condition."""
        pass
    
    @abstractmethod
    def visit_logical(self, node: 'LogicalCondition') -> Any:
        """Visit a logical condition."""
        pass


class ASTNode(ABC):
    """
    Base class for all AST nodes.
    
    All AST nodes must implement:
    - accept(): For visitor pattern traversal
    - to_dict(): For JSON serialization
    """
    
    @abstractmethod
    def accept(self, visitor: ASTVisitor) -> Any:
        """Accept a visitor for traversal."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        pass


# =============================================================================
# Helper Types
# =============================================================================

@dataclass(frozen=True)
class HierarchyLevel:
    """
    Specifies a hierarchy level (cortex, layer, segment, role).
    
    Attributes:
        level: The hierarchy level name
        path: Optional path within the level (e.g., "semantic.attributes")
    
    Example:
        >>> level = HierarchyLevel("layer", "semantic")
        >>> level = HierarchyLevel("segment", "semantic.attributes")
        >>> level = HierarchyLevel("cortex")  # No path needed for cortex
    """
    level: str  # "cortex", "layer", "segment", "role"
    path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "path": self.path}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HierarchyLevel':
        return cls(level=data["level"], path=data.get("path"))


@dataclass(frozen=True)
class GlyphRef:
    """
    Reference to a glyph by identifier.
    
    Used in queries that operate on specific glyphs (COMPARE, PREDICT, etc.)
    
    Attributes:
        identifier: The glyph's unique identifier
    
    Example:
        >>> ref = GlyphRef("product-123")
    """
    identifier: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {"identifier": self.identifier}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GlyphRef':
        return cls(identifier=data["identifier"])


# =============================================================================
# Condition Nodes (for WHERE clauses)
# =============================================================================

@dataclass
class Condition(ASTNode):
    """Base class for WHERE clause conditions."""
    pass


@dataclass
class ComparisonCondition(Condition):
    """
    A comparison condition (e.g., price < 100).
    
    Attributes:
        field: The field name to compare
        operator: Comparison operator (=, <, >, <=, >=, !=)
        value: The value to compare against
    
    Example:
        >>> cond = ComparisonCondition("price", "<", 100)
        >>> cond = ComparisonCondition("category", "=", "electronics")
    """
    field: str
    operator: str  # "=", "<", ">", "<=", ">=", "!="
    value: Any
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_comparison(self)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "comparison",
            "field": self.field,
            "operator": self.operator,
            "value": self.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComparisonCondition':
        return cls(
            field=data["field"],
            operator=data["operator"],
            value=data["value"]
        )


@dataclass
class LogicalCondition(Condition):
    """
    A logical condition (AND, OR, NOT).
    
    Attributes:
        operator: Logical operator (AND, OR, NOT)
        operands: List of child conditions
    
    Example:
        >>> # price < 100 AND category = "electronics"
        >>> cond = LogicalCondition("AND", [
        ...     ComparisonCondition("price", "<", 100),
        ...     ComparisonCondition("category", "=", "electronics")
        ... ])
        >>> # NOT (status = "inactive")
        >>> cond = LogicalCondition("NOT", [
        ...     ComparisonCondition("status", "=", "inactive")
        ... ])
    """
    operator: str  # "AND", "OR", "NOT"
    operands: List[Condition]
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_logical(self)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "logical",
            "operator": self.operator,
            "operands": [op.to_dict() for op in self.operands]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogicalCondition':
        operands = []
        for op_data in data["operands"]:
            if op_data["type"] == "comparison":
                operands.append(ComparisonCondition.from_dict(op_data))
            else:
                operands.append(LogicalCondition.from_dict(op_data))
        return cls(operator=data["operator"], operands=operands)


def condition_from_dict(data: Dict[str, Any]) -> Condition:
    """Factory function to create Condition from dict."""
    if data["type"] == "comparison":
        return ComparisonCondition.from_dict(data)
    else:
        return LogicalCondition.from_dict(data)


# =============================================================================
# Query AST Nodes
# =============================================================================

@dataclass
class SimilaritySearchNode(ASTNode):
    """
    FIND SIMILAR TO query.
    
    Finds glyphs similar to a text query or another glyph.
    
    Attributes:
        target: Text query string or GlyphRef
        scope: Optional scope (layer.segment path)
        where: Optional WHERE clause condition for filtering results
        limit: Maximum results to return (default 10)
        threshold: Minimum similarity score (default 0.5)
    
    Example GQL:
        FIND SIMILAR TO "red sports car" IN semantic.concept LIMIT 10 THRESHOLD 0.7
        FIND SIMILAR TO glyph("product-123") LIMIT 5
        FIND SIMILAR TO "family car" WHERE make = "Toyota" LIMIT 10
    """
    target: Union[str, GlyphRef]
    scope: Optional[str] = None
    where: Optional[Condition] = None
    limit: int = 10
    threshold: float = 0.5
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_similarity_search(self)
    
    def to_dict(self) -> Dict[str, Any]:
        target_dict = self.target if isinstance(self.target, str) else self.target.to_dict()
        result = {
            "type": "similarity_search",
            "target": target_dict,
            "scope": self.scope,
            "limit": self.limit,
            "threshold": self.threshold
        }
        if self.where:
            result["where"] = self.where.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimilaritySearchNode':
        target = data["target"]
        if isinstance(target, dict):
            target = GlyphRef.from_dict(target)
        where = None
        if data.get("where"):
            where = condition_from_dict(data["where"])
        return cls(
            target=target,
            scope=data.get("scope"),
            where=where,
            limit=data.get("limit", 10),
            threshold=data.get("threshold", 0.5)
        )


@dataclass
class ListNode(ASTNode):
    """
    LIST query with optional WHERE clause.
    
    Lists glyphs, optionally filtered by conditions.
    
    Attributes:
        where: Optional filter condition
        limit: Maximum results to return (default 100)
    
    Example GQL:
        LIST ALL LIMIT 50
        LIST ALL WHERE category = "electronics" AND price < 100
    """
    where: Optional[Condition] = None
    limit: int = 100
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_list(self)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "list",
            "where": self.where.to_dict() if self.where else None,
            "limit": self.limit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ListNode':
        where = None
        if data.get("where"):
            where = condition_from_dict(data["where"])
        return cls(where=where, limit=data.get("limit", 100))


@dataclass
class CountNode(ASTNode):
    """
    COUNT query with optional WHERE clause.
    
    Counts glyphs, optionally filtered by conditions.
    
    Attributes:
        where: Optional filter condition
    
    Example GQL:
        COUNT ALL
        COUNT ALL WHERE status = "active"
    """
    where: Optional[Condition] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_count(self)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "count",
            "where": self.where.to_dict() if self.where else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CountNode':
        where = None
        if data.get("where"):
            where = condition_from_dict(data["where"])
        return cls(where=where)


@dataclass
class CompareNode(ASTNode):
    """
    COMPARE two glyphs.
    
    Computes similarity between two glyphs at a specified hierarchy level.
    
    Attributes:
        glyph1: First glyph reference
        glyph2: Second glyph reference
        level: Hierarchy level for comparison (default: cortex)
    
    Example GQL:
        COMPARE glyph("product-123") TO glyph("product-456")
        COMPARE glyph("a") TO glyph("b") AT LAYER semantic
    """
    glyph1: GlyphRef
    glyph2: GlyphRef
    level: HierarchyLevel = field(default_factory=lambda: HierarchyLevel("cortex"))
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_compare(self)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "compare",
            "glyph1": self.glyph1.to_dict(),
            "glyph2": self.glyph2.to_dict(),
            "level": self.level.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompareNode':
        return cls(
            glyph1=GlyphRef.from_dict(data["glyph1"]),
            glyph2=GlyphRef.from_dict(data["glyph2"]),
            level=HierarchyLevel.from_dict(data.get("level", {"level": "cortex"}))
        )


@dataclass
class PredictNode(ASTNode):
    """
    PREDICT future state of a glyph.
    
    Uses temporal prediction to forecast future values.
    
    Attributes:
        glyph: Glyph to predict
        role: Optional role path to predict
        window: Optional time window (e.g., "30d")
    
    Example GQL:
        PREDICT glyph("customer-123")
        PREDICT glyph("metric-xyz") AT ROLE value WINDOW 30d
    """
    glyph: GlyphRef
    role: Optional[str] = None
    window: Optional[str] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_predict(self)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "predict",
            "glyph": self.glyph.to_dict(),
            "role": self.role,
            "window": self.window
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictNode':
        return cls(
            glyph=GlyphRef.from_dict(data["glyph"]),
            role=data.get("role"),
            window=data.get("window")
        )


@dataclass
class DriftDetectionNode(ASTNode):
    """
    DETECT DRIFT between two glyph states.
    
    Computes semantic drift between baseline and current states.
    
    Attributes:
        from_glyph: Baseline glyph reference
        to_glyph: Current glyph reference
        level: Hierarchy level for comparison (default: cortex)
        threshold: Optional drift threshold for alerting
    
    Example GQL:
        DETECT DRIFT FROM glyph("baseline") TO glyph("current")
        DETECT DRIFT FROM glyph("v1") TO glyph("v2") AT CORTEX THRESHOLD 0.15
    """
    from_glyph: GlyphRef
    to_glyph: GlyphRef
    level: HierarchyLevel = field(default_factory=lambda: HierarchyLevel("cortex"))
    threshold: Optional[float] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_drift_detection(self)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "drift_detection",
            "from_glyph": self.from_glyph.to_dict(),
            "to_glyph": self.to_glyph.to_dict(),
            "level": self.level.to_dict(),
            "threshold": self.threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DriftDetectionNode':
        return cls(
            from_glyph=GlyphRef.from_dict(data["from_glyph"]),
            to_glyph=GlyphRef.from_dict(data["to_glyph"]),
            level=HierarchyLevel.from_dict(data.get("level", {"level": "cortex"})),
            threshold=data.get("threshold")
        )


@dataclass
class IntrospectNode(ASTNode):
    """
    INTROSPECT a glyph's structure.
    
    Decomposes a glyph to show its internal structure.
    
    Attributes:
        glyph: Glyph to introspect
        decompose_by: Level to decompose by (layer, segment, role)
    
    Example GQL:
        INTROSPECT glyph("product-123") DECOMPOSE BY layer
        INTROSPECT glyph("item-001") DECOMPOSE BY segment
    """
    glyph: GlyphRef
    decompose_by: str  # "layer", "segment", "role"
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_introspect(self)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "introspect",
            "glyph": self.glyph.to_dict(),
            "decompose_by": self.decompose_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntrospectNode':
        return cls(
            glyph=GlyphRef.from_dict(data["glyph"]),
            decompose_by=data["decompose_by"]
        )


@dataclass
class TrendNode(ASTNode):
    """
    TREND analysis over time.
    
    Tracks semantic changes over a time window with optional alerting.
    
    Attributes:
        glyph: Glyph to track
        window: Time window (e.g., "7d", "24h")
        level: Hierarchy level for analysis (default: cortex)
        alert_condition: Optional alert condition (e.g., "drift > 0.2")
    
    Example GQL:
        TREND glyph("metric-xyz") OVER 7d
        TREND glyph("model-v1") OVER 30d AT LAYER semantic ALERT ON drift > 0.2
    """
    glyph: GlyphRef
    window: str
    level: HierarchyLevel = field(default_factory=lambda: HierarchyLevel("cortex"))
    alert_condition: Optional[str] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_trend(self)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "trend",
            "glyph": self.glyph.to_dict(),
            "window": self.window,
            "level": self.level.to_dict(),
            "alert_condition": self.alert_condition
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrendNode':
        return cls(
            glyph=GlyphRef.from_dict(data["glyph"]),
            window=data["window"],
            level=HierarchyLevel.from_dict(data.get("level", {"level": "cortex"})),
            alert_condition=data.get("alert_condition")
        )


@dataclass
class AggregateNode(ASTNode):
    """
    AGGREGATE with GROUP BY.
    
    Aggregates glyph data with optional grouping.
    
    Attributes:
        function: Aggregation function (COUNT, AVG, MIN, MAX, SUM)
        field: Optional field to aggregate
        group_by: Optional list of fields to group by
        where: Optional filter condition
    
    Example GQL:
        AGGREGATE COUNT
        AGGREGATE AVG price GROUP BY category
        AGGREGATE SUM quantity WHERE status = "active" GROUP BY region
    """
    function: str  # "COUNT", "AVG", "MIN", "MAX", "SUM"
    field: Optional[str] = None
    group_by: Optional[List[str]] = None
    where: Optional[Condition] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_aggregate(self)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "aggregate",
            "function": self.function,
            "field": self.field,
            "group_by": self.group_by,
            "where": self.where.to_dict() if self.where else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregateNode':
        where = None
        if data.get("where"):
            where = condition_from_dict(data["where"])
        return cls(
            function=data["function"],
            field=data.get("field"),
            group_by=data.get("group_by"),
            where=where
        )


# =============================================================================
# Factory function for deserializing any AST node
# =============================================================================

def ast_from_dict(data: Dict[str, Any]) -> ASTNode:
    """
    Factory function to create any AST node from a dictionary.
    
    Args:
        data: Dictionary with 'type' field indicating node type
    
    Returns:
        The appropriate ASTNode subclass instance
    
    Raises:
        ValueError: If type is unknown
    """
    node_types = {
        "similarity_search": SimilaritySearchNode,
        "list": ListNode,
        "count": CountNode,
        "compare": CompareNode,
        "predict": PredictNode,
        "drift_detection": DriftDetectionNode,
        "introspect": IntrospectNode,
        "trend": TrendNode,
        "aggregate": AggregateNode,
    }
    
    node_type = data.get("type")
    if node_type not in node_types:
        raise ValueError(f"Unknown AST node type: {node_type}")
    
    return node_types[node_type].from_dict(data)
