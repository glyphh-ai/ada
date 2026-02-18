"""
Property-based tests for FIND SIMILAR with WHERE clause parser round-trip.

This module contains property-based tests using Hypothesis to verify
the round-trip property for FIND SIMILAR queries with WHERE clauses.

**Validates: Property 1** - Parser Round-Trip for FIND SIMILAR with WHERE
For any valid SimilaritySearchNode AST with a WHERE clause, printing the AST
to a GQL string and then parsing that string back SHALL produce an equivalent AST.

**Validates: Requirements 1.1, 1.2, 1.5, 1.6, 1.7**
"""

import pytest
from hypothesis import given, settings, strategies as st, assume
from hypothesis.strategies import composite
from typing import Optional, Union

from glyphh.gql import parse, pretty_print
from glyphh.gql.ast import (
    SimilaritySearchNode,
    GlyphRef,
    Condition,
    ComparisonCondition,
    LogicalCondition,
)
from typing import Tuple, Dict, Any, List


# =============================================================================
# Generator Strategies for SimilaritySearchNode with WHERE clauses
# =============================================================================

# Strategy for valid identifiers (field names, glyph IDs)
# Must start with a letter and contain only alphanumeric characters and underscores
valid_identifier = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
    min_size=1,
    max_size=1
).flatmap(
    lambda first: st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
        min_size=0,
        max_size=15
    ).map(lambda rest: first + rest)
)

# Strategy for string values (for target text and comparison values)
# Avoid problematic characters that could break parsing
safe_string_chars = st.characters(
    whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
    whitelist_characters=" -_.,!?:;'",
    blacklist_characters='"\\',
)

safe_string = st.text(
    alphabet=safe_string_chars,
    min_size=1,
    max_size=50
).filter(lambda s: s.strip())  # Ensure non-empty after stripping

# Strategy for comparison operators
comparison_operators = st.sampled_from(["=", "!=", "<", ">", "<=", ">="])

# Strategy for comparison values (string or number)
# Note: GQL lexer doesn't support negative numbers directly, so we use non-negative values
comparison_value = st.one_of(
    safe_string,
    st.integers(min_value=0, max_value=10000),
    st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False).map(lambda x: round(x, 2))
)


@composite
def comparison_condition_strategy(draw) -> ComparisonCondition:
    """Generate a valid ComparisonCondition."""
    field = draw(valid_identifier)
    operator = draw(comparison_operators)
    value = draw(comparison_value)
    return ComparisonCondition(field=field, operator=operator, value=value)


@composite
def condition_strategy(draw, max_depth: int = 3) -> Condition:
    """
    Generate a valid Condition (comparison or logical).
    
    Uses max_depth to limit recursion for nested logical conditions.
    """
    if max_depth <= 1:
        # Base case: only generate comparison conditions
        return draw(comparison_condition_strategy())
    
    # Choose between comparison and logical conditions
    is_logical = draw(st.booleans())
    
    if not is_logical:
        return draw(comparison_condition_strategy())
    
    # Generate logical condition
    operator = draw(st.sampled_from(["AND", "OR", "NOT"]))
    
    if operator == "NOT":
        # NOT has exactly one operand
        operand = draw(condition_strategy(max_depth=max_depth - 1))
        return LogicalCondition(operator="NOT", operands=[operand])
    else:
        # AND/OR have 2+ operands
        num_operands = draw(st.integers(min_value=2, max_value=3))
        operands = [
            draw(condition_strategy(max_depth=max_depth - 1))
            for _ in range(num_operands)
        ]
        return LogicalCondition(operator=operator, operands=operands)


@composite
def glyph_ref_strategy(draw) -> GlyphRef:
    """Generate a valid GlyphRef."""
    identifier = draw(valid_identifier)
    return GlyphRef(identifier=identifier)


@composite
def target_strategy(draw) -> Union[str, GlyphRef]:
    """Generate a valid target (string or GlyphRef)."""
    is_string = draw(st.booleans())
    if is_string:
        return draw(safe_string)
    else:
        return draw(glyph_ref_strategy())


@composite
def scope_strategy(draw) -> Optional[str]:
    """Generate a valid scope (dotted path or None)."""
    has_scope = draw(st.booleans())
    if not has_scope:
        return None
    
    # Generate 1-3 path segments
    num_segments = draw(st.integers(min_value=1, max_value=3))
    segments = [draw(valid_identifier) for _ in range(num_segments)]
    return ".".join(segments)


@composite
def similarity_search_node_with_where_strategy(draw) -> SimilaritySearchNode:
    """
    Generate a valid SimilaritySearchNode with a WHERE clause.
    
    This strategy generates random but valid ASTs that should round-trip
    through print → parse.
    """
    target = draw(target_strategy())
    scope = draw(scope_strategy())
    where = draw(condition_strategy(max_depth=3))  # Always include WHERE
    limit = draw(st.integers(min_value=1, max_value=1000))
    threshold = draw(st.floats(min_value=0.0, max_value=1.0).map(lambda x: round(x, 2)))
    
    return SimilaritySearchNode(
        target=target,
        scope=scope,
        where=where,
        limit=limit,
        threshold=threshold
    )


@composite
def similarity_search_node_optional_where_strategy(draw) -> SimilaritySearchNode:
    """
    Generate a valid SimilaritySearchNode with optional WHERE clause.
    
    This tests both with and without WHERE clauses.
    """
    target = draw(target_strategy())
    scope = draw(scope_strategy())
    has_where = draw(st.booleans())
    where = draw(condition_strategy(max_depth=3)) if has_where else None
    limit = draw(st.integers(min_value=1, max_value=1000))
    threshold = draw(st.floats(min_value=0.0, max_value=1.0).map(lambda x: round(x, 2)))
    
    return SimilaritySearchNode(
        target=target,
        scope=scope,
        where=where,
        limit=limit,
        threshold=threshold
    )


# =============================================================================
# Helper Functions for AST Comparison
# =============================================================================

def conditions_equal(c1: Optional[Condition], c2: Optional[Condition]) -> bool:
    """
    Check if two conditions are semantically equivalent.
    
    Note: The parser may flatten nested AND/OR conditions, so we need to
    compare the flattened forms. For example:
    - Original: A AND (B AND C)
    - Parsed: A AND B AND C
    These are semantically equivalent.
    """
    if c1 is None and c2 is None:
        return True
    if c1 is None or c2 is None:
        return False
    
    if isinstance(c1, ComparisonCondition) and isinstance(c2, ComparisonCondition):
        # Compare field, operator, and value
        # Handle numeric value comparison with tolerance for floats
        if c1.field != c2.field or c1.operator != c2.operator:
            return False
        
        v1, v2 = c1.value, c2.value
        if isinstance(v1, float) and isinstance(v2, float):
            return abs(v1 - v2) < 0.001
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return abs(float(v1) - float(v2)) < 0.001
        return v1 == v2
    
    if isinstance(c1, LogicalCondition) and isinstance(c2, LogicalCondition):
        if c1.operator != c2.operator:
            return False
        
        # Flatten operands for comparison (parser may flatten nested same-operator conditions)
        flat1 = _flatten_logical(c1)
        flat2 = _flatten_logical(c2)
        
        if len(flat1) != len(flat2):
            return False
        
        return all(
            conditions_equal(op1, op2)
            for op1, op2 in zip(flat1, flat2)
        )
    
    return False


def _flatten_logical(cond: LogicalCondition) -> list:
    """
    Flatten nested logical conditions with the same operator.
    
    For example, AND(a, AND(b, c)) becomes [a, b, c].
    This handles the parser's behavior of flattening nested same-operator conditions.
    """
    result = []
    for operand in cond.operands:
        if isinstance(operand, LogicalCondition) and operand.operator == cond.operator:
            # Recursively flatten nested same-operator conditions
            result.extend(_flatten_logical(operand))
        else:
            result.append(operand)
    return result


def similarity_nodes_equal(n1: SimilaritySearchNode, n2: SimilaritySearchNode) -> bool:
    """Check if two SimilaritySearchNode ASTs are semantically equivalent."""
    # Compare targets
    if isinstance(n1.target, str) and isinstance(n2.target, str):
        if n1.target != n2.target:
            return False
    elif isinstance(n1.target, GlyphRef) and isinstance(n2.target, GlyphRef):
        if n1.target.identifier != n2.target.identifier:
            return False
    else:
        return False
    
    # Compare scope
    if n1.scope != n2.scope:
        return False
    
    # Compare WHERE conditions
    if not conditions_equal(n1.where, n2.where):
        return False
    
    # Compare limit
    if n1.limit != n2.limit:
        return False
    
    # Compare threshold (with tolerance for float comparison)
    if abs(n1.threshold - n2.threshold) > 0.001:
        return False
    
    return True


# =============================================================================
# Property Tests
# =============================================================================

class TestParserRoundTripWithWhere:
    """
    Property tests for Parser Round-Trip with WHERE clause (Property 1).
    
    **Validates: Property 1** - Parser Round-Trip for FIND SIMILAR with WHERE
    For any valid SimilaritySearchNode AST with a WHERE clause, printing the AST
    to a GQL string and then parsing that string back SHALL produce an equivalent AST.
    
    **Validates: Requirements 1.1, 1.2, 1.5, 1.6, 1.7**
    """
    
    @given(node=similarity_search_node_with_where_strategy())
    @settings(max_examples=100)
    def test_round_trip_with_where_clause(self, node: SimilaritySearchNode):
        """
        Property test: print → parse produces equivalent AST for FIND SIMILAR with WHERE.
        
        For any valid SimilaritySearchNode AST with a WHERE clause, printing to GQL
        and parsing back SHALL produce an equivalent AST.
        
        **Validates: Requirements 1.1, 1.2, 1.5, 1.6, 1.7**
        """
        # Print AST to GQL string
        gql_string = pretty_print(node)
        
        # Parse GQL string back to AST
        parsed_node = parse(gql_string)
        
        # Verify the parsed node is a SimilaritySearchNode
        assert isinstance(parsed_node, SimilaritySearchNode), \
            f"Expected SimilaritySearchNode, got {type(parsed_node).__name__}"
        
        # Verify the ASTs are equivalent
        assert similarity_nodes_equal(node, parsed_node), \
            f"Round-trip failed:\n" \
            f"  Original: {node.to_dict()}\n" \
            f"  GQL: {gql_string}\n" \
            f"  Parsed: {parsed_node.to_dict()}"
    
    @given(node=similarity_search_node_optional_where_strategy())
    @settings(max_examples=100)
    def test_round_trip_with_optional_where(self, node: SimilaritySearchNode):
        """
        Property test: print → parse produces equivalent AST with or without WHERE.
        
        For any valid SimilaritySearchNode AST (with or without WHERE clause),
        printing to GQL and parsing back SHALL produce an equivalent AST.
        
        **Validates: Requirements 1.1, 1.2, 1.5, 1.6, 1.7**
        """
        # Print AST to GQL string
        gql_string = pretty_print(node)
        
        # Parse GQL string back to AST
        parsed_node = parse(gql_string)
        
        # Verify the parsed node is a SimilaritySearchNode
        assert isinstance(parsed_node, SimilaritySearchNode), \
            f"Expected SimilaritySearchNode, got {type(parsed_node).__name__}"
        
        # Verify the ASTs are equivalent
        assert similarity_nodes_equal(node, parsed_node), \
            f"Round-trip failed:\n" \
            f"  Original: {node.to_dict()}\n" \
            f"  GQL: {gql_string}\n" \
            f"  Parsed: {parsed_node.to_dict()}"
    
    @given(condition=condition_strategy(max_depth=3))
    @settings(max_examples=100)
    def test_round_trip_various_where_conditions(self, condition: Condition):
        """
        Property test: Various WHERE condition types round-trip correctly.
        
        For any valid WHERE condition (simple comparison, AND, OR, NOT, nested),
        embedding it in a FIND SIMILAR query and round-tripping SHALL preserve
        the condition structure.
        
        **Validates: Requirements 1.1, 1.5, 1.7**
        """
        # Create a simple SimilaritySearchNode with the generated condition
        node = SimilaritySearchNode(
            target="test query",
            scope=None,
            where=condition,
            limit=10,
            threshold=0.5
        )
        
        # Print AST to GQL string
        gql_string = pretty_print(node)
        
        # Parse GQL string back to AST
        parsed_node = parse(gql_string)
        
        # Verify the parsed node is a SimilaritySearchNode
        assert isinstance(parsed_node, SimilaritySearchNode), \
            f"Expected SimilaritySearchNode, got {type(parsed_node).__name__}"
        
        # Verify the WHERE conditions are equivalent
        assert conditions_equal(node.where, parsed_node.where), \
            f"WHERE condition round-trip failed:\n" \
            f"  Original condition: {condition.to_dict()}\n" \
            f"  GQL: {gql_string}\n" \
            f"  Parsed condition: {parsed_node.where.to_dict() if parsed_node.where else None}"
    
    @given(
        field=valid_identifier,
        operator=comparison_operators,
        value=comparison_value
    )
    @settings(max_examples=100)
    def test_round_trip_simple_comparison(self, field: str, operator: str, value):
        """
        Property test: Simple comparison conditions round-trip correctly.
        
        For any valid field, operator, and value combination, the comparison
        condition SHALL round-trip correctly.
        
        **Validates: Requirements 1.1, 1.5**
        """
        condition = ComparisonCondition(field=field, operator=operator, value=value)
        node = SimilaritySearchNode(
            target="test",
            where=condition,
            limit=10,
            threshold=0.5
        )
        
        gql_string = pretty_print(node)
        parsed_node = parse(gql_string)
        
        assert isinstance(parsed_node, SimilaritySearchNode)
        assert parsed_node.where is not None
        assert isinstance(parsed_node.where, ComparisonCondition)
        assert parsed_node.where.field == field
        assert parsed_node.where.operator == operator
        
        # Compare values with tolerance for floats
        if isinstance(value, float) and isinstance(parsed_node.where.value, float):
            assert abs(value - parsed_node.where.value) < 0.001
        elif isinstance(value, (int, float)) and isinstance(parsed_node.where.value, (int, float)):
            assert abs(float(value) - float(parsed_node.where.value)) < 0.001
        else:
            assert parsed_node.where.value == value
    
    @given(node=similarity_search_node_with_where_strategy())
    @settings(max_examples=100)
    def test_double_round_trip(self, node: SimilaritySearchNode):
        """
        Property test: Double round-trip produces stable output.
        
        For any valid SimilaritySearchNode, print → parse → print → parse
        SHALL produce the same result as a single round-trip.
        
        **Validates: Requirements 1.6, 1.7**
        """
        # First round-trip
        gql1 = pretty_print(node)
        parsed1 = parse(gql1)
        
        # Second round-trip
        gql2 = pretty_print(parsed1)
        parsed2 = parse(gql2)
        
        # The GQL strings should be identical after first normalization
        assert gql1 == gql2, \
            f"Double round-trip produced different GQL:\n" \
            f"  First: {gql1}\n" \
            f"  Second: {gql2}"
        
        # The ASTs should be equivalent
        assert similarity_nodes_equal(parsed1, parsed2), \
            f"Double round-trip produced different ASTs"


class TestWhereClauseFormatting:
    """
    Property tests for WHERE clause formatting in FIND SIMILAR queries.
    
    **Validates: Requirement 1.6** - Pretty printer format
    """
    
    @given(node=similarity_search_node_with_where_strategy())
    @settings(max_examples=100)
    def test_where_clause_appears_in_output(self, node: SimilaritySearchNode):
        """
        Property test: WHERE clause appears in printed output.
        
        For any SimilaritySearchNode with a WHERE clause, the printed GQL
        SHALL contain the WHERE keyword.
        
        **Validates: Requirement 1.6**
        """
        gql_string = pretty_print(node)
        
        assert "WHERE" in gql_string, \
            f"WHERE clause missing from output: {gql_string}"
    
    @given(node=similarity_search_node_with_where_strategy())
    @settings(max_examples=100)
    def test_output_format_order(self, node: SimilaritySearchNode):
        """
        Property test: Output follows expected format order.
        
        The printed GQL SHALL follow the format:
        FIND SIMILAR TO target WHERE condition [IN scope] LIMIT n THRESHOLD t
        
        **Validates: Requirement 1.6**
        """
        gql_string = pretty_print(node)
        
        # Check that FIND SIMILAR TO appears first
        assert gql_string.startswith("FIND SIMILAR TO"), \
            f"Output should start with 'FIND SIMILAR TO': {gql_string}"
        
        # Check that WHERE appears before LIMIT
        where_pos = gql_string.find("WHERE")
        limit_pos = gql_string.find("LIMIT")
        assert where_pos < limit_pos, \
            f"WHERE should appear before LIMIT: {gql_string}"
        
        # Check that LIMIT appears before THRESHOLD
        threshold_pos = gql_string.find("THRESHOLD")
        assert limit_pos < threshold_pos, \
            f"LIMIT should appear before THRESHOLD: {gql_string}"
        
        # If scope is present, check it appears after WHERE and before LIMIT
        if node.scope:
            in_pos = gql_string.find(" IN ")
            assert where_pos < in_pos < limit_pos, \
                f"IN scope should appear between WHERE and LIMIT: {gql_string}"
    
    @given(node=similarity_search_node_optional_where_strategy())
    @settings(max_examples=100)
    def test_output_is_parseable(self, node: SimilaritySearchNode):
        """
        Property test: Printed output is always parseable.
        
        For any valid SimilaritySearchNode, the printed GQL SHALL be
        parseable without errors.
        
        **Validates: Requirements 1.1, 1.6**
        """
        gql_string = pretty_print(node)
        
        # This should not raise any exceptions
        try:
            parsed = parse(gql_string)
            assert parsed is not None
        except Exception as e:
            pytest.fail(f"Failed to parse printed GQL: {gql_string}\nError: {e}")


# =============================================================================
# Property Tests for Predicate Filtering (Property 11)
# =============================================================================

from dataclasses import dataclass, field as dataclass_field
from glyphh.gql.planner import ExecutionContext, ExecutionPlanner
from glyphh.gql.plans import SimilaritySearchPlan
from glyphh.gql.storage import GlyphStorageProtocol


@dataclass
class MockGlyph:
    """Mock glyph for testing with attributes."""
    identifier: str
    attributes: Dict[str, Any] = dataclass_field(default_factory=dict)
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    cortex: Any = None  # For embedding


class MockGlyphStorage:
    """
    Custom storage implementation for testing predicate filtering.
    
    This storage allows us to control similarity scores independently
    of the actual glyph data, making it easier to test predicate filtering.
    """
    
    def __init__(
        self,
        glyphs: Dict[str, MockGlyph],
        scores: Dict[str, float]
    ):
        """
        Initialize test storage.
        
        Args:
            glyphs: Dictionary mapping glyph_id to MockGlyph objects
            scores: Dictionary mapping glyph_id to similarity scores
        """
        self._glyphs = glyphs
        self._scores = scores
    
    def list_glyphs(self) -> Dict[str, Any]:
        """List all glyphs."""
        return self._glyphs
    
    def get_glyph(self, glyph_id: str) -> Any:
        """Get a glyph by ID."""
        if glyph_id not in self._glyphs:
            raise KeyError(f"Glyph not found: {glyph_id}")
        return self._glyphs[glyph_id]
    
    def has_glyph(self, glyph_id: str) -> bool:
        """Check if a glyph exists."""
        return glyph_id in self._glyphs
    
    def get_embedding(self, glyph_id: str) -> Optional[Any]:
        """Get the primary embedding for a glyph (returns glyph_id as marker)."""
        if glyph_id in self._glyphs:
            return glyph_id  # Use glyph_id as the "embedding" for score lookup
        return None
    
    def get_embedding_for_scope(
        self,
        glyph_id: str,
        layer: Optional[str] = None,
        segment: Optional[str] = None
    ) -> Optional[Any]:
        """Get embedding for a specific scope."""
        return self.get_embedding(glyph_id)
    
    def compute_similarity(self, v1: Any, v2: Any) -> float:
        """
        Compute similarity between two vectors.
        
        v1 is the query vector (ignored), v2 is the glyph_id.
        Returns the predefined score for that glyph.
        """
        # v2 is the glyph_id (from get_embedding)
        if isinstance(v2, str) and v2 in self._scores:
            return self._scores[v2]
        return 0.0
    
    def get_glyph_attribute(self, glyph_id: str, attribute: str) -> Any:
        """Get an attribute value from a glyph."""
        glyph = self.get_glyph(glyph_id)
        
        # Check direct object attribute
        if hasattr(glyph, attribute):
            return getattr(glyph, attribute)
        
        # Check attributes dict
        if hasattr(glyph, 'attributes') and isinstance(glyph.attributes, dict):
            if attribute in glyph.attributes:
                return glyph.attributes[attribute]
        
        # Check metadata dict
        if hasattr(glyph, 'metadata') and isinstance(glyph.metadata, dict):
            if attribute in glyph.metadata:
                return glyph.metadata[attribute]
        
        return None


class MockEncoder:
    """Mock encoder for testing."""
    
    def encode(self, concept: Any) -> Any:
        """Return a mock vector."""
        class MockVector:
            cortex = "query"  # Simple marker for query vector
        return MockVector()


# Strategy for generating glyph attributes
@composite
def glyph_attributes_strategy(draw) -> Dict[str, Any]:
    """Generate random glyph attributes."""
    num_attrs = draw(st.integers(min_value=1, max_value=5))
    attrs = {}
    
    for i in range(num_attrs):
        field_name = draw(valid_identifier)
        # Generate different types of values
        value_type = draw(st.sampled_from(["string", "int", "float"]))
        if value_type == "string":
            attrs[field_name] = draw(safe_string)
        elif value_type == "int":
            attrs[field_name] = draw(st.integers(min_value=0, max_value=1000))
        else:
            attrs[field_name] = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False).map(lambda x: round(x, 2)))
    
    return attrs


@composite
def glyphs_with_scores_strategy(draw, min_glyphs: int = 5, max_glyphs: int = 20) -> Tuple[Dict[str, MockGlyph], Dict[str, float]]:
    """
    Generate a set of glyphs with random attributes and similarity scores.
    
    Returns:
        Tuple of (glyphs dict, scores dict)
    """
    num_glyphs = draw(st.integers(min_value=min_glyphs, max_value=max_glyphs))
    glyphs = {}
    scores = {}
    
    for i in range(num_glyphs):
        glyph_id = f"glyph_{i}"
        attrs = draw(glyph_attributes_strategy())
        glyphs[glyph_id] = MockGlyph(identifier=glyph_id, attributes=attrs, metadata={})
        # Generate random similarity score
        scores[glyph_id] = draw(st.floats(min_value=0.0, max_value=1.0).map(lambda x: round(x, 3)))
    
    return glyphs, scores


@composite
def predicate_for_glyphs_strategy(draw, glyphs: Dict[str, MockGlyph]) -> Tuple[Condition, str]:
    """
    Generate a predicate that can be evaluated against the given glyphs.
    
    Picks a field that exists in at least one glyph and generates a
    comparison condition for it.
    
    Returns:
        Tuple of (condition, field_name)
    """
    # Collect all fields from all glyphs
    all_fields = {}
    for glyph in glyphs.values():
        for field_name, value in glyph.attributes.items():
            if field_name not in all_fields:
                all_fields[field_name] = []
            all_fields[field_name].append(value)
    
    # If no fields, create a simple condition that won't match
    if not all_fields:
        return ComparisonCondition(field="nonexistent", operator="=", value="none"), "nonexistent"
    
    # Pick a random field
    field_name = draw(st.sampled_from(list(all_fields.keys())))
    values = all_fields[field_name]
    
    # Pick a value from the existing values or generate a new one
    use_existing = draw(st.booleans())
    if use_existing and values:
        value = draw(st.sampled_from(values))
    else:
        # Generate a value of the same type
        sample_value = values[0] if values else "test"
        if isinstance(sample_value, str):
            value = draw(safe_string)
        elif isinstance(sample_value, int):
            value = draw(st.integers(min_value=0, max_value=1000))
        else:
            value = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False).map(lambda x: round(x, 2)))
    
    # Pick an operator
    if isinstance(value, str):
        operator = draw(st.sampled_from(["=", "!="]))
    else:
        operator = draw(comparison_operators)
    
    return ComparisonCondition(field=field_name, operator=operator, value=value), field_name


class TestPredicateFiltering:
    """
    Property tests for Similarity Search Predicate Filtering (Property 11).
    
    **Validates: Property 11** - Similarity Search Predicate Filtering
    For any SimilaritySearchPlan with a predicate, execution SHALL:
    - Return only results that satisfy the predicate
    - Return results sorted by similarity score (descending)
    - Apply the limit AFTER filtering
    
    **Validates: Requirements 1.4**
    """
    
    def _get_results_from_fact_tree(self, fact_tree) -> List[Dict[str, Any]]:
        """
        Extract results from a FactTree.
        
        The SimilaritySearchPlan adds results at path ["results"], so we need
        to find the child node with that description.
        """
        for child in fact_tree.root.children:
            # The description contains "Found X similar glyphs"
            if "similar glyphs" in child.description or child.description.startswith("Found"):
                return child.value if child.value else []
        return []
    
    @given(data=st.data())
    @settings(max_examples=100)
    def test_results_satisfy_predicate(self, data):
        """
        Property test: All returned results satisfy the predicate.
        
        For any SimilaritySearchPlan with a predicate, all returned results
        SHALL satisfy the predicate condition.
        
        **Validates: Requirements 1.4**
        """
        # Generate glyphs with scores
        glyphs, scores = data.draw(glyphs_with_scores_strategy(min_glyphs=5, max_glyphs=15))
        
        # Skip if no glyphs
        assume(len(glyphs) > 0)
        
        # Generate a predicate based on existing glyph attributes
        condition, field_name = data.draw(predicate_for_glyphs_strategy(glyphs))
        
        # Create storage
        storage = MockGlyphStorage(glyphs=glyphs, scores=scores)
        
        # Create context
        context = ExecutionContext(
            storage=storage,
            encoder=MockEncoder()
        )
        
        # Build predicate from condition
        planner = ExecutionPlanner(context)
        predicate = planner._build_predicate(condition)
        
        # Create and execute plan
        plan = SimilaritySearchPlan(
            target_vector="query",
            limit=100,  # High limit to get all matching results
            threshold=0.0,  # Low threshold to include all
            predicate=predicate,
            predicate_desc=f"{condition.field} {condition.operator} {condition.value}"
        )
        
        result = plan.execute(context)
        
        # Get results from fact tree
        results = self._get_results_from_fact_tree(result)
        if not results:
            return  # No results is valid
        
        result_ids = [r["id"] for r in results]
        
        # Verify all results satisfy the predicate
        for glyph_id in result_ids:
            assert predicate(glyph_id, context), \
                f"Result {glyph_id} does not satisfy predicate: " \
                f"{condition.field} {condition.operator} {condition.value}"
    
    @given(data=st.data())
    @settings(max_examples=100)
    def test_results_sorted_by_score_descending(self, data):
        """
        Property test: Results are sorted by similarity score (descending).
        
        For any SimilaritySearchPlan execution, results SHALL be sorted
        by similarity score in descending order.
        
        **Validates: Requirements 1.4**
        """
        # Generate glyphs with distinct scores
        num_glyphs = data.draw(st.integers(min_value=5, max_value=15))
        glyphs = {}
        scores = {}
        
        for i in range(num_glyphs):
            glyph_id = f"glyph_{i}"
            # Use distinct scores to avoid ties
            score = 0.1 + (i * 0.05)  # 0.1, 0.15, 0.2, ...
            glyphs[glyph_id] = MockGlyph(
                identifier=glyph_id,
                attributes={"category": "test", "value": i}
            )
            scores[glyph_id] = round(score, 3)
        
        # Create storage
        storage = MockGlyphStorage(glyphs=glyphs, scores=scores)
        
        # Create context
        context = ExecutionContext(
            storage=storage,
            encoder=MockEncoder()
        )
        
        # Create a simple predicate that matches all
        def match_all(glyph_id: str, ctx: ExecutionContext) -> bool:
            return True
        
        # Create and execute plan
        plan = SimilaritySearchPlan(
            target_vector="query",
            limit=100,
            threshold=0.0,
            predicate=match_all,
            predicate_desc="match all"
        )
        
        result = plan.execute(context)
        
        # Get results from fact tree
        results = self._get_results_from_fact_tree(result)
        assert len(results) > 0, "Expected results in fact tree"
        
        result_scores = [r["score"] for r in results]
        
        # Verify scores are in descending order
        for i in range(len(result_scores) - 1):
            assert result_scores[i] >= result_scores[i + 1], \
                f"Results not sorted by score descending: {result_scores}"
    
    @given(data=st.data())
    @settings(max_examples=100)
    def test_limit_applied_after_filtering(self, data):
        """
        Property test: Limit is applied AFTER filtering.
        
        For any SimilaritySearchPlan with a predicate and limit, the limit
        SHALL be applied after the predicate filter, not before.
        
        **Validates: Requirements 1.4**
        """
        # Generate glyphs - half will match predicate, half won't
        num_glyphs = data.draw(st.integers(min_value=10, max_value=20))
        glyphs = {}
        scores = {}
        
        for i in range(num_glyphs):
            glyph_id = f"glyph_{i}"
            # Alternate between matching and non-matching
            category = "match" if i % 2 == 0 else "nomatch"
            glyphs[glyph_id] = MockGlyph(
                identifier=glyph_id,
                attributes={"category": category, "index": i}
            )
            # Higher scores for matching glyphs to ensure they appear first
            scores[glyph_id] = 0.9 - (i * 0.01) if category == "match" else 0.5 - (i * 0.01)
        
        # Count matching glyphs
        matching_count = sum(1 for g in glyphs.values() if g.attributes.get("category") == "match")
        
        # Create storage
        storage = MockGlyphStorage(glyphs=glyphs, scores=scores)
        
        # Create context
        context = ExecutionContext(
            storage=storage,
            encoder=MockEncoder()
        )
        
        # Create predicate that only matches "match" category
        def match_category(glyph_id: str, ctx: ExecutionContext) -> bool:
            attr = ctx.get_glyph_attribute(glyph_id, "category")
            return attr == "match"
        
        # Use a limit smaller than matching count
        limit = min(3, matching_count)
        
        # Create and execute plan
        plan = SimilaritySearchPlan(
            target_vector="query",
            limit=limit,
            threshold=0.0,
            predicate=match_category,
            predicate_desc="category = match"
        )
        
        result = plan.execute(context)
        
        # Get results from fact tree
        results = self._get_results_from_fact_tree(result)
        assert results is not None, "Expected results in fact tree"
        
        result_ids = [r["id"] for r in results]
        
        # Verify limit is respected
        assert len(result_ids) <= limit, \
            f"Expected at most {limit} results, got {len(result_ids)}"
        
        # Verify all results match the predicate
        for glyph_id in result_ids:
            glyph = glyphs[glyph_id]
            assert glyph.attributes.get("category") == "match", \
                f"Result {glyph_id} should have category='match'"
    
    @given(data=st.data())
    @settings(max_examples=100)
    def test_predicate_excludes_non_matching(self, data):
        """
        Property test: Predicate correctly excludes non-matching glyphs.
        
        For any SimilaritySearchPlan with a predicate, glyphs that do not
        satisfy the predicate SHALL NOT appear in results.
        
        **Validates: Requirements 1.4**
        """
        # Generate glyphs with known attributes
        num_glyphs = data.draw(st.integers(min_value=5, max_value=15))
        threshold_value = data.draw(st.integers(min_value=20, max_value=80))
        
        glyphs = {}
        scores = {}
        expected_matches = set()
        expected_non_matches = set()
        
        for i in range(num_glyphs):
            glyph_id = f"glyph_{i}"
            value = data.draw(st.integers(min_value=0, max_value=100))
            glyphs[glyph_id] = MockGlyph(
                identifier=glyph_id,
                attributes={"value": value}
            )
            scores[glyph_id] = 0.8  # All have same score
            
            # Track which should match
            if value > threshold_value:
                expected_matches.add(glyph_id)
            else:
                expected_non_matches.add(glyph_id)
        
        # Create storage
        storage = MockGlyphStorage(glyphs=glyphs, scores=scores)
        
        # Create context
        context = ExecutionContext(
            storage=storage,
            encoder=MockEncoder()
        )
        
        # Create predicate: value > threshold_value
        def value_greater_than(glyph_id: str, ctx: ExecutionContext) -> bool:
            attr = ctx.get_glyph_attribute(glyph_id, "value")
            return attr is not None and attr > threshold_value
        
        # Create and execute plan
        plan = SimilaritySearchPlan(
            target_vector="query",
            limit=100,
            threshold=0.0,
            predicate=value_greater_than,
            predicate_desc=f"value > {threshold_value}"
        )
        
        result = plan.execute(context)
        
        # Get results from fact tree
        results = self._get_results_from_fact_tree(result)
        result_ids = set(r["id"] for r in results) if results else set()
        
        # Verify no non-matching glyphs in results
        for glyph_id in expected_non_matches:
            assert glyph_id not in result_ids, \
                f"Non-matching glyph {glyph_id} should not be in results"
        
        # Verify all matching glyphs are in results (if any exist)
        for glyph_id in expected_matches:
            assert glyph_id in result_ids, \
                f"Matching glyph {glyph_id} should be in results"
    
    @given(data=st.data())
    @settings(max_examples=100)
    def test_complex_predicate_and_or(self, data):
        """
        Property test: Complex predicates with AND/OR work correctly.
        
        For any SimilaritySearchPlan with a complex predicate (AND/OR),
        results SHALL correctly satisfy the logical condition.
        
        **Validates: Requirements 1.4**
        """
        # Generate glyphs with two attributes
        num_glyphs = data.draw(st.integers(min_value=10, max_value=20))
        
        glyphs = {}
        scores = {}
        
        for i in range(num_glyphs):
            glyph_id = f"glyph_{i}"
            # Generate two attributes
            attr_a = data.draw(st.sampled_from(["red", "blue", "green"]))
            attr_b = data.draw(st.integers(min_value=0, max_value=100))
            
            glyphs[glyph_id] = MockGlyph(
                identifier=glyph_id,
                attributes={"color": attr_a, "size": attr_b}
            )
            scores[glyph_id] = 0.8
        
        # Create storage
        storage = MockGlyphStorage(glyphs=glyphs, scores=scores)
        
        # Create context
        context = ExecutionContext(
            storage=storage,
            encoder=MockEncoder()
        )
        
        # Create complex predicate: color = "red" AND size > 50
        condition = LogicalCondition(
            operator="AND",
            operands=[
                ComparisonCondition(field="color", operator="=", value="red"),
                ComparisonCondition(field="size", operator=">", value=50)
            ]
        )
        
        planner = ExecutionPlanner(context)
        predicate = planner._build_predicate(condition)
        
        # Create and execute plan
        plan = SimilaritySearchPlan(
            target_vector="query",
            limit=100,
            threshold=0.0,
            predicate=predicate,
            predicate_desc="color = red AND size > 50"
        )
        
        result = plan.execute(context)
        
        # Get results from fact tree
        results = self._get_results_from_fact_tree(result)
        result_ids = [r["id"] for r in results] if results else []
        
        # Verify all results satisfy both conditions
        for glyph_id in result_ids:
            glyph = glyphs[glyph_id]
            assert glyph.attributes.get("color") == "red", \
                f"Result {glyph_id} should have color='red'"
            assert glyph.attributes.get("size") > 50, \
                f"Result {glyph_id} should have size > 50"
        
        # Verify glyphs that should match are included
        for glyph_id, glyph in glyphs.items():
            if glyph.attributes.get("color") == "red" and glyph.attributes.get("size") > 50:
                assert glyph_id in result_ids, \
                    f"Glyph {glyph_id} should be in results"
    
    @given(data=st.data())
    @settings(max_examples=100)
    def test_empty_results_when_no_matches(self, data):
        """
        Property test: Empty results when no glyphs match predicate.
        
        For any SimilaritySearchPlan with a predicate that matches no glyphs,
        the result SHALL be empty.
        
        **Validates: Requirements 1.4**
        """
        # Generate glyphs with known attributes
        num_glyphs = data.draw(st.integers(min_value=5, max_value=10))
        
        glyphs = {}
        scores = {}
        
        for i in range(num_glyphs):
            glyph_id = f"glyph_{i}"
            glyphs[glyph_id] = MockGlyph(
                identifier=glyph_id,
                attributes={"status": "inactive"}  # All inactive
            )
            scores[glyph_id] = 0.8
        
        # Create storage
        storage = MockGlyphStorage(glyphs=glyphs, scores=scores)
        
        # Create context
        context = ExecutionContext(
            storage=storage,
            encoder=MockEncoder()
        )
        
        # Create predicate that matches nothing: status = "active"
        def match_active(glyph_id: str, ctx: ExecutionContext) -> bool:
            attr = ctx.get_glyph_attribute(glyph_id, "status")
            return attr == "active"
        
        # Create and execute plan
        plan = SimilaritySearchPlan(
            target_vector="query",
            limit=100,
            threshold=0.0,
            predicate=match_active,
            predicate_desc="status = active"
        )
        
        result = plan.execute(context)
        
        # Get results from fact tree
        results = self._get_results_from_fact_tree(result)
        assert results is not None, "Expected results fact"
        assert len(results) == 0, \
            f"Expected empty results, got {len(results)}"
