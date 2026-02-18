"""
GQL Parser - Parses GQL tokens into an Abstract Syntax Tree.

The parser implements a recursive descent parser for the GQL grammar.
It validates syntax and constraint values (e.g., LIMIT > 0, THRESHOLD in [0,1]).
"""

from typing import List, Optional, Union
from glyphh.gql.tokens import Token, TokenType
from glyphh.gql.lexer import GQLLexer
from glyphh.gql.ast import (
    ASTNode,
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
from glyphh.gql.exceptions import ParseError


class GQLParser:
    """
    Parses GQL tokens into an AST.
    
    The parser validates:
    - Syntax according to GQL grammar
    - LIMIT values are positive integers
    - THRESHOLD values are between 0.0 and 1.0
    
    Example:
        >>> from glyphh.gql import GQLLexer, GQLParser
        >>> lexer = GQLLexer('FIND SIMILAR TO "red car" LIMIT 10')
        >>> parser = GQLParser(lexer.tokenize())
        >>> ast = parser.parse()
        >>> print(ast.to_dict())
    """
    
    def __init__(self, tokens: List[Token]):
        """
        Initialize parser with tokens.
        
        Args:
            tokens: List of tokens from the lexer
        """
        self.tokens = tokens
        self.pos = 0
    
    @property
    def current(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # Return EOF
    
    def _advance(self) -> Token:
        """Advance to next token and return the previous one."""
        token = self.current
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token
    
    def _peek(self, offset: int = 1) -> Token:
        """Peek at a future token without advancing."""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]
    
    def _expect(self, *types: TokenType) -> Token:
        """
        Expect current token to be one of the given types.
        
        Raises:
            ParseError: If current token doesn't match
        """
        if self.current.type not in types:
            expected = [t.name for t in types]
            raise ParseError(
                message="Unexpected token",
                line=self.current.line,
                column=self.current.column,
                expected=expected,
                actual=self.current.type.name
            )
        return self._advance()
    
    def _match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        return self.current.type in types
    
    def _match_and_advance(self, *types: TokenType) -> Optional[Token]:
        """If current token matches, advance and return it. Otherwise return None."""
        if self._match(*types):
            return self._advance()
        return None
    
    def parse(self) -> ASTNode:
        """
        Parse tokens and return the root AST node.
        
        Returns:
            The parsed AST node
        
        Raises:
            ParseError: If syntax is invalid
        """
        # Determine query type from first token
        if self._match(TokenType.FIND):
            return self._parse_find_similar()
        elif self._match(TokenType.LIST):
            return self._parse_list()
        elif self._match(TokenType.COUNT):
            return self._parse_count()
        elif self._match(TokenType.COMPARE):
            return self._parse_compare()
        elif self._match(TokenType.PREDICT):
            return self._parse_predict()
        elif self._match(TokenType.DETECT):
            return self._parse_drift()
        elif self._match(TokenType.INTROSPECT):
            return self._parse_introspect()
        elif self._match(TokenType.TREND):
            return self._parse_trend()
        elif self._match(TokenType.AGGREGATE):
            return self._parse_aggregate()
        else:
            raise ParseError(
                message="Expected query keyword",
                line=self.current.line,
                column=self.current.column,
                expected=["FIND", "LIST", "COUNT", "COMPARE", "PREDICT", 
                         "DETECT", "INTROSPECT", "TREND", "AGGREGATE"],
                actual=self.current.type.name
            )
    
    def _parse_find_similar(self) -> SimilaritySearchNode:
        """
        Parse FIND SIMILAR TO query with optional WHERE clause.
        
        Grammar:
            FIND SIMILAR TO (STRING | glyph_ref) [WHERE condition] [IN scope] [LIMIT num] [THRESHOLD num]
        
        Note: Optional clauses can appear in any order.
        """
        self._expect(TokenType.FIND)
        self._expect(TokenType.SIMILAR)
        self._expect(TokenType.TO)
        
        # Parse target (string or glyph reference)
        target: Union[str, GlyphRef]
        if self._match(TokenType.STRING):
            target = self._advance().value
        elif self._match(TokenType.GLYPH_FUNC):
            target = self._parse_glyph_ref()
        else:
            raise ParseError(
                message="Expected string or glyph reference",
                line=self.current.line,
                column=self.current.column,
                expected=["STRING", "glyph(...)"],
                actual=self.current.type.name
            )
        
        # Parse optional clauses (order-independent)
        scope = None
        where = None
        limit = 10
        threshold = 0.5
        
        while not self._match(TokenType.EOF):
            if self._match_and_advance(TokenType.WHERE):
                where = self._parse_condition()
            elif self._match_and_advance(TokenType.IN):
                scope = self._parse_scope()
            elif self._match_and_advance(TokenType.LIMIT):
                limit = self._parse_positive_int("LIMIT")
            elif self._match_and_advance(TokenType.THRESHOLD):
                threshold = self._parse_threshold()
            else:
                break
        
        return SimilaritySearchNode(
            target=target,
            scope=scope,
            where=where,
            limit=limit,
            threshold=threshold
        )
    
    def _parse_list(self) -> ListNode:
        """
        Parse LIST query.
        
        Grammar:
            LIST ALL [WHERE condition] [LIMIT num]
        """
        self._expect(TokenType.LIST)
        self._expect(TokenType.ALL)
        
        where = None
        limit = 100
        
        while not self._match(TokenType.EOF):
            if self._match_and_advance(TokenType.WHERE):
                where = self._parse_condition()
            elif self._match_and_advance(TokenType.LIMIT):
                limit = self._parse_positive_int("LIMIT")
            else:
                break
        
        return ListNode(where=where, limit=limit)
    
    def _parse_count(self) -> CountNode:
        """
        Parse COUNT query.
        
        Grammar:
            COUNT ALL [WHERE condition]
        """
        self._expect(TokenType.COUNT)
        self._expect(TokenType.ALL)
        
        where = None
        if self._match_and_advance(TokenType.WHERE):
            where = self._parse_condition()
        
        return CountNode(where=where)
    
    def _parse_compare(self) -> CompareNode:
        """
        Parse COMPARE query.
        
        Grammar:
            COMPARE glyph_ref TO glyph_ref [AT hierarchy_level]
        """
        self._expect(TokenType.COMPARE)
        glyph1 = self._parse_glyph_ref()
        self._expect(TokenType.TO)
        glyph2 = self._parse_glyph_ref()
        
        level = HierarchyLevel("cortex")
        if self._match_and_advance(TokenType.AT):
            level = self._parse_hierarchy_level()
        
        return CompareNode(glyph1=glyph1, glyph2=glyph2, level=level)
    
    def _parse_predict(self) -> PredictNode:
        """
        Parse PREDICT query.
        
        Grammar:
            PREDICT glyph_ref [AT ROLE path] [WINDOW duration]
        """
        self._expect(TokenType.PREDICT)
        glyph = self._parse_glyph_ref()
        
        role = None
        window = None
        
        while not self._match(TokenType.EOF):
            if self._match_and_advance(TokenType.AT):
                self._expect(TokenType.ROLE)
                role = self._parse_path()
            elif self._match_and_advance(TokenType.WINDOW):
                window = self._expect(TokenType.DURATION).value
            else:
                break
        
        return PredictNode(glyph=glyph, role=role, window=window)
    
    def _parse_drift(self) -> DriftDetectionNode:
        """
        Parse DETECT DRIFT query.
        
        Grammar:
            DETECT DRIFT FROM glyph_ref TO glyph_ref [AT hierarchy_level] [THRESHOLD num]
        """
        self._expect(TokenType.DETECT)
        self._expect(TokenType.DRIFT)
        self._expect(TokenType.FROM)
        from_glyph = self._parse_glyph_ref()
        self._expect(TokenType.TO)
        to_glyph = self._parse_glyph_ref()
        
        level = HierarchyLevel("cortex")
        threshold = None
        
        while not self._match(TokenType.EOF):
            if self._match_and_advance(TokenType.AT):
                level = self._parse_hierarchy_level()
            elif self._match_and_advance(TokenType.THRESHOLD):
                threshold = self._parse_threshold()
            else:
                break
        
        return DriftDetectionNode(
            from_glyph=from_glyph,
            to_glyph=to_glyph,
            level=level,
            threshold=threshold
        )
    
    def _parse_introspect(self) -> IntrospectNode:
        """
        Parse INTROSPECT query.
        
        Grammar:
            INTROSPECT glyph_ref DECOMPOSE BY (layer | segment | role)
        """
        self._expect(TokenType.INTROSPECT)
        glyph = self._parse_glyph_ref()
        self._expect(TokenType.DECOMPOSE)
        self._expect(TokenType.BY)
        
        # Expect layer, segment, or role
        if self._match(TokenType.LAYER):
            decompose_by = "layer"
            self._advance()
        elif self._match(TokenType.SEGMENT):
            decompose_by = "segment"
            self._advance()
        elif self._match(TokenType.ROLE):
            decompose_by = "role"
            self._advance()
        else:
            raise ParseError(
                message="Expected decomposition level",
                line=self.current.line,
                column=self.current.column,
                expected=["LAYER", "SEGMENT", "ROLE"],
                actual=self.current.type.name
            )
        
        return IntrospectNode(glyph=glyph, decompose_by=decompose_by)
    
    def _parse_trend(self) -> TrendNode:
        """
        Parse TREND query.
        
        Grammar:
            TREND glyph_ref OVER duration [AT hierarchy_level] [ALERT ON condition]
        """
        self._expect(TokenType.TREND)
        glyph = self._parse_glyph_ref()
        self._expect(TokenType.OVER)
        window = self._expect(TokenType.DURATION).value
        
        level = HierarchyLevel("cortex")
        alert_condition = None
        
        while not self._match(TokenType.EOF):
            if self._match_and_advance(TokenType.AT):
                level = self._parse_hierarchy_level()
            elif self._match_and_advance(TokenType.ALERT):
                self._expect(TokenType.ON)
                alert_condition = self._parse_alert_condition()
            else:
                break
        
        return TrendNode(
            glyph=glyph,
            window=window,
            level=level,
            alert_condition=alert_condition
        )
    
    def _parse_aggregate(self) -> AggregateNode:
        """
        Parse AGGREGATE query.
        
        Grammar:
            AGGREGATE function [field] [WHERE condition] [GROUP BY fields]
        """
        self._expect(TokenType.AGGREGATE)
        
        # Parse aggregation function
        if self._match(TokenType.COUNT, TokenType.AVG, TokenType.MIN, 
                       TokenType.MAX, TokenType.SUM):
            function = self._advance().type.name
        else:
            raise ParseError(
                message="Expected aggregation function",
                line=self.current.line,
                column=self.current.column,
                expected=["COUNT", "AVG", "MIN", "MAX", "SUM"],
                actual=self.current.type.name
            )
        
        # Parse optional field (for non-COUNT functions)
        field = None
        if function != "COUNT" and self._match(TokenType.IDENTIFIER):
            field = self._advance().value
        
        where = None
        group_by = None
        
        while not self._match(TokenType.EOF):
            if self._match_and_advance(TokenType.WHERE):
                where = self._parse_condition()
            elif self._match_and_advance(TokenType.GROUP):
                self._expect(TokenType.BY)
                group_by = self._parse_field_list()
            else:
                break
        
        return AggregateNode(
            function=function,
            field=field,
            group_by=group_by,
            where=where
        )
    
    def _parse_glyph_ref(self) -> GlyphRef:
        """
        Parse glyph reference: glyph("identifier")
        """
        self._expect(TokenType.GLYPH_FUNC)
        self._expect(TokenType.LPAREN)
        identifier = self._expect(TokenType.STRING).value
        self._expect(TokenType.RPAREN)
        return GlyphRef(identifier=identifier)
    
    def _parse_hierarchy_level(self) -> HierarchyLevel:
        """
        Parse hierarchy level: CORTEX | LAYER [path] | SEGMENT [path] | ROLE [path]
        """
        if self._match(TokenType.CORTEX):
            self._advance()
            return HierarchyLevel("cortex")
        elif self._match(TokenType.LAYER):
            self._advance()
            path = self._parse_path() if self._match(TokenType.IDENTIFIER) else None
            return HierarchyLevel("layer", path)
        elif self._match(TokenType.SEGMENT):
            self._advance()
            path = self._parse_path() if self._match(TokenType.IDENTIFIER) else None
            return HierarchyLevel("segment", path)
        elif self._match(TokenType.ROLE):
            self._advance()
            path = self._parse_path() if self._match(TokenType.IDENTIFIER) else None
            return HierarchyLevel("role", path)
        else:
            raise ParseError(
                message="Expected hierarchy level",
                line=self.current.line,
                column=self.current.column,
                expected=["CORTEX", "LAYER", "SEGMENT", "ROLE"],
                actual=self.current.type.name
            )
    
    def _parse_scope(self) -> str:
        """
        Parse scope: identifier[.identifier]*
        """
        return self._parse_path()
    
    def _parse_path(self) -> str:
        """
        Parse dotted path: identifier[.identifier]*
        """
        parts = [self._expect(TokenType.IDENTIFIER).value]
        while self._match_and_advance(TokenType.DOT):
            parts.append(self._expect(TokenType.IDENTIFIER).value)
        return ".".join(parts)
    
    def _parse_condition(self) -> Condition:
        """
        Parse WHERE clause condition with AND/OR/NOT support.
        
        Grammar:
            condition := or_expr
            or_expr := and_expr (OR and_expr)*
            and_expr := not_expr (AND not_expr)*
            not_expr := NOT not_expr | primary
            primary := comparison | '(' condition ')'
        """
        return self._parse_or_expr()
    
    def _parse_or_expr(self) -> Condition:
        """Parse OR expression."""
        left = self._parse_and_expr()
        
        while self._match_and_advance(TokenType.OR):
            right = self._parse_and_expr()
            if isinstance(left, LogicalCondition) and left.operator == "OR":
                left.operands.append(right)
            else:
                left = LogicalCondition("OR", [left, right])
        
        return left
    
    def _parse_and_expr(self) -> Condition:
        """Parse AND expression."""
        left = self._parse_not_expr()
        
        while self._match_and_advance(TokenType.AND):
            right = self._parse_not_expr()
            if isinstance(left, LogicalCondition) and left.operator == "AND":
                left.operands.append(right)
            else:
                left = LogicalCondition("AND", [left, right])
        
        return left
    
    def _parse_not_expr(self) -> Condition:
        """Parse NOT expression."""
        if self._match_and_advance(TokenType.NOT):
            operand = self._parse_not_expr()
            return LogicalCondition("NOT", [operand])
        return self._parse_primary_condition()
    
    def _parse_primary_condition(self) -> Condition:
        """Parse primary condition (comparison or parenthesized expression)."""
        if self._match_and_advance(TokenType.LPAREN):
            cond = self._parse_condition()
            self._expect(TokenType.RPAREN)
            return cond
        
        return self._parse_comparison()
    
    def _parse_comparison(self) -> ComparisonCondition:
        """
        Parse comparison: field operator value
        
        Field can be a dotted path for nested attribute access:
            field := identifier[.identifier]*
        
        Examples:
            color = "red"
            vehicle.identity.make = "Toyota"
        """
        # Parse field as dotted path (e.g., vehicle.identity.make)
        field = self._parse_path()
        
        # Parse operator
        if self._match(TokenType.EQ):
            operator = "="
        elif self._match(TokenType.LT):
            operator = "<"
        elif self._match(TokenType.GT):
            operator = ">"
        elif self._match(TokenType.LE):
            operator = "<="
        elif self._match(TokenType.GE):
            operator = ">="
        elif self._match(TokenType.NE):
            operator = "!="
        else:
            raise ParseError(
                message="Expected comparison operator",
                line=self.current.line,
                column=self.current.column,
                expected=["=", "<", ">", "<=", ">=", "!="],
                actual=self.current.type.name
            )
        self._advance()
        
        # Parse value
        if self._match(TokenType.STRING):
            value = self._advance().value
        elif self._match(TokenType.NUMBER):
            value = self._advance().value
        elif self._match(TokenType.IDENTIFIER):
            value = self._advance().value
        else:
            raise ParseError(
                message="Expected value",
                line=self.current.line,
                column=self.current.column,
                expected=["STRING", "NUMBER", "IDENTIFIER"],
                actual=self.current.type.name
            )
        
        return ComparisonCondition(field=field, operator=operator, value=value)
    
    def _parse_field_list(self) -> List[str]:
        """Parse comma-separated list of field names."""
        fields = [self._expect(TokenType.IDENTIFIER).value]
        while self._match_and_advance(TokenType.COMMA):
            fields.append(self._expect(TokenType.IDENTIFIER).value)
        return fields
    
    def _parse_alert_condition(self) -> str:
        """
        Parse alert condition: identifier operator number
        Returns as string for now (e.g., "drift > 0.2")
        Note: 'drift' is a keyword but can be used as field name in alert conditions
        """
        # Handle 'drift' keyword being used as field name
        if self._match(TokenType.DRIFT):
            field = "drift"
            self._advance()
        elif self._match(TokenType.IDENTIFIER):
            field = self._advance().value
        else:
            raise ParseError(
                message="Expected field name in alert condition",
                line=self.current.line,
                column=self.current.column,
                expected=["IDENTIFIER", "drift"],
                actual=self.current.type.name
            )
        
        if self._match(TokenType.GT):
            op = ">"
        elif self._match(TokenType.LT):
            op = "<"
        elif self._match(TokenType.GE):
            op = ">="
        elif self._match(TokenType.LE):
            op = "<="
        else:
            raise ParseError(
                message="Expected comparison operator in alert condition",
                line=self.current.line,
                column=self.current.column,
                expected=[">", "<", ">=", "<="],
                actual=self.current.type.name
            )
        self._advance()
        
        value = self._expect(TokenType.NUMBER).value
        return f"{field} {op} {value}"
    
    def _parse_positive_int(self, name: str) -> int:
        """Parse and validate a positive integer."""
        token = self._expect(TokenType.NUMBER)
        value = token.value
        
        if not isinstance(value, int) or value <= 0:
            raise ParseError(
                message=f"{name} must be a positive integer",
                line=token.line,
                column=token.column,
                expected=["positive integer"],
                actual=str(value)
            )
        
        return value
    
    def _parse_threshold(self) -> float:
        """Parse and validate a threshold value (0.0 to 1.0)."""
        token = self._expect(TokenType.NUMBER)
        value = float(token.value)
        
        if value < 0.0 or value > 1.0:
            raise ParseError(
                message="THRESHOLD must be between 0.0 and 1.0",
                line=token.line,
                column=token.column,
                expected=["number between 0.0 and 1.0"],
                actual=str(value)
            )
        
        return value


def parse(source: str) -> ASTNode:
    """
    Convenience function to parse GQL source string.
    
    Args:
        source: GQL query string
    
    Returns:
        Parsed AST node
    
    Example:
        >>> ast = parse('FIND SIMILAR TO "red car" LIMIT 10')
    """
    lexer = GQLLexer(source)
    parser = GQLParser(lexer.tokenize())
    return parser.parse()
