"""
GQL Pretty Printer - Converts AST nodes back to GQL syntax.

The pretty printer implements the ASTVisitor interface to traverse
AST nodes and generate normalized GQL strings.
"""

from typing import Any
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


class GQLPrettyPrinter(ASTVisitor):
    """
    Converts AST nodes back to GQL syntax.
    
    The pretty printer produces normalized GQL that can be parsed
    back into an equivalent AST (round-trip property).
    
    Example:
        >>> from glyphh.gql import parse, GQLPrettyPrinter
        >>> ast = parse('FIND SIMILAR TO "red car" LIMIT 10')
        >>> printer = GQLPrettyPrinter()
        >>> gql = printer.print(ast)
        >>> print(gql)
        FIND SIMILAR TO "red car" LIMIT 10 THRESHOLD 0.5
    """
    
    def print(self, node: ASTNode) -> str:
        """
        Convert AST node to GQL string.
        
        Args:
            node: The AST node to print
        
        Returns:
            GQL string representation
        """
        return node.accept(self)
    
    def visit_similarity_search(self, node: SimilaritySearchNode) -> str:
        """Print FIND SIMILAR query with optional WHERE clause."""
        # Target
        if isinstance(node.target, str):
            target = f'"{self._escape_string(node.target)}"'
        else:
            target = f'glyph("{self._escape_string(node.target.identifier)}")'
        
        result = f"FIND SIMILAR TO {target}"
        
        # Optional WHERE clause (after target, before other clauses)
        if node.where:
            result += f" WHERE {self._print_condition(node.where)}"
        
        # Optional scope
        if node.scope:
            result += f" IN {node.scope}"
        
        # Limit and threshold (always include for normalization)
        result += f" LIMIT {node.limit}"
        result += f" THRESHOLD {node.threshold}"
        
        return result
    
    def visit_list(self, node: ListNode) -> str:
        """Print LIST query."""
        result = "LIST ALL"
        
        if node.where:
            result += f" WHERE {self._print_condition(node.where)}"
        
        result += f" LIMIT {node.limit}"
        
        return result
    
    def visit_count(self, node: CountNode) -> str:
        """Print COUNT query."""
        result = "COUNT ALL"
        
        if node.where:
            result += f" WHERE {self._print_condition(node.where)}"
        
        return result
    
    def visit_compare(self, node: CompareNode) -> str:
        """Print COMPARE query."""
        g1 = f'glyph("{self._escape_string(node.glyph1.identifier)}")'
        g2 = f'glyph("{self._escape_string(node.glyph2.identifier)}")'
        
        result = f"COMPARE {g1} TO {g2}"
        
        # Include hierarchy level
        result += self._print_hierarchy_level(node.level)
        
        return result
    
    def visit_predict(self, node: PredictNode) -> str:
        """Print PREDICT query."""
        glyph = f'glyph("{self._escape_string(node.glyph.identifier)}")'
        result = f"PREDICT {glyph}"
        
        if node.role:
            result += f" AT ROLE {node.role}"
        
        if node.window:
            result += f" WINDOW {node.window}"
        
        return result
    
    def visit_drift_detection(self, node: DriftDetectionNode) -> str:
        """Print DETECT DRIFT query."""
        from_g = f'glyph("{self._escape_string(node.from_glyph.identifier)}")'
        to_g = f'glyph("{self._escape_string(node.to_glyph.identifier)}")'
        
        result = f"DETECT DRIFT FROM {from_g} TO {to_g}"
        
        # Include hierarchy level
        result += self._print_hierarchy_level(node.level)
        
        if node.threshold is not None:
            result += f" THRESHOLD {node.threshold}"
        
        return result
    
    def visit_introspect(self, node: IntrospectNode) -> str:
        """Print INTROSPECT query."""
        glyph = f'glyph("{self._escape_string(node.glyph.identifier)}")'
        result = f"INTROSPECT {glyph} DECOMPOSE BY {node.decompose_by.upper()}"
        return result
    
    def visit_trend(self, node: TrendNode) -> str:
        """Print TREND query."""
        glyph = f'glyph("{self._escape_string(node.glyph.identifier)}")'
        result = f"TREND {glyph} OVER {node.window}"
        
        # Include hierarchy level
        result += self._print_hierarchy_level(node.level)
        
        if node.alert_condition:
            result += f" ALERT ON {node.alert_condition}"
        
        return result
    
    def visit_aggregate(self, node: AggregateNode) -> str:
        """Print AGGREGATE query."""
        result = f"AGGREGATE {node.function}"
        
        if node.field:
            result += f" {node.field}"
        
        if node.where:
            result += f" WHERE {self._print_condition(node.where)}"
        
        if node.group_by:
            result += f" GROUP BY {', '.join(node.group_by)}"
        
        return result
    
    def visit_comparison(self, node: ComparisonCondition) -> str:
        """Print comparison condition."""
        return self._print_comparison(node)
    
    def visit_logical(self, node: LogicalCondition) -> str:
        """Print logical condition."""
        return self._print_logical(node)
    
    def _print_condition(self, cond: Condition) -> str:
        """Print a condition expression."""
        if isinstance(cond, ComparisonCondition):
            return self._print_comparison(cond)
        elif isinstance(cond, LogicalCondition):
            return self._print_logical(cond)
        return ""
    
    def _print_comparison(self, cond: ComparisonCondition) -> str:
        """Print a comparison condition."""
        value = cond.value
        if isinstance(value, str):
            value = f'"{self._escape_string(value)}"'
        return f"{cond.field} {cond.operator} {value}"
    
    def _print_logical(self, cond: LogicalCondition) -> str:
        """Print a logical condition."""
        if cond.operator == "NOT":
            inner = self._print_condition(cond.operands[0])
            # Wrap in parens if it's a logical expression
            if isinstance(cond.operands[0], LogicalCondition):
                inner = f"({inner})"
            return f"NOT {inner}"
        else:
            parts = []
            for op in cond.operands:
                part = self._print_condition(op)
                # Wrap in parens if it's an OR inside AND, or vice versa
                if isinstance(op, LogicalCondition) and op.operator != cond.operator:
                    part = f"({part})"
                parts.append(part)
            return f" {cond.operator} ".join(parts)
    
    def _print_hierarchy_level(self, level: HierarchyLevel) -> str:
        """Print hierarchy level clause."""
        if level.level == "cortex":
            return " AT CORTEX"
        
        result = f" AT {level.level.upper()}"
        if level.path:
            result += f" {level.path}"
        return result
    
    def _escape_string(self, s: str) -> str:
        """Escape special characters in a string."""
        return (s
            .replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\n', '\\n')
            .replace('\t', '\\t')
            .replace('\r', '\\r'))


def pretty_print(node: ASTNode) -> str:
    """
    Convenience function to pretty print an AST node.
    
    Args:
        node: The AST node to print
    
    Returns:
        GQL string representation
    
    Example:
        >>> from glyphh.gql import parse, pretty_print
        >>> ast = parse('FIND SIMILAR TO "test"')
        >>> print(pretty_print(ast))
    """
    printer = GQLPrettyPrinter()
    return printer.print(node)
