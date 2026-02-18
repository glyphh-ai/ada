"""
GQL Token Types and Token class.

Defines all token types recognized by the GQL lexer and the Token dataclass
that represents individual tokens with position information.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Union


class TokenType(Enum):
    """
    All token types recognized by the GQL lexer.
    
    Categories:
    - Keywords: GQL reserved words (FIND, SIMILAR, LIST, etc.)
    - Literals: String, number, duration, identifier
    - Operators: Comparison and punctuation
    - Special: Function calls and EOF
    """
    
    # Query operation keywords
    FIND = auto()
    SIMILAR = auto()
    TO = auto()
    IN = auto()
    LIMIT = auto()
    THRESHOLD = auto()
    LIST = auto()
    ALL = auto()
    WHERE = auto()
    COUNT = auto()
    COMPARE = auto()
    PREDICT = auto()
    DETECT = auto()
    DRIFT = auto()
    FROM = auto()
    INTROSPECT = auto()
    DECOMPOSE = auto()
    TREND = auto()
    OVER = auto()
    ALERT = auto()
    AGGREGATE = auto()
    GROUP = auto()
    
    # Logical operators
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Hierarchy keywords
    AT = auto()
    LAYER = auto()
    SEGMENT = auto()
    ROLE = auto()
    CORTEX = auto()
    BY = auto()
    ON = auto()
    WINDOW = auto()
    
    # Aggregation functions
    AVG = auto()
    MIN = auto()
    MAX = auto()
    SUM = auto()
    
    # Literals and identifiers
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    DURATION = auto()
    
    # Comparison operators
    EQ = auto()        # =
    LT = auto()        # <
    GT = auto()        # >
    LE = auto()        # <=
    GE = auto()        # >=
    NE = auto()        # !=
    
    # Punctuation
    LPAREN = auto()    # (
    RPAREN = auto()    # )
    COMMA = auto()     # ,
    DOT = auto()       # .
    
    # Special
    GLYPH_FUNC = auto()  # glyph keyword (before parenthesis)
    EOF = auto()


# Mapping from keyword strings to token types
KEYWORDS: dict[str, TokenType] = {
    "FIND": TokenType.FIND,
    "SIMILAR": TokenType.SIMILAR,
    "TO": TokenType.TO,
    "IN": TokenType.IN,
    "LIMIT": TokenType.LIMIT,
    "THRESHOLD": TokenType.THRESHOLD,
    "LIST": TokenType.LIST,
    "ALL": TokenType.ALL,
    "WHERE": TokenType.WHERE,
    "COUNT": TokenType.COUNT,
    "COMPARE": TokenType.COMPARE,
    "PREDICT": TokenType.PREDICT,
    "DETECT": TokenType.DETECT,
    "DRIFT": TokenType.DRIFT,
    "FROM": TokenType.FROM,
    "INTROSPECT": TokenType.INTROSPECT,
    "DECOMPOSE": TokenType.DECOMPOSE,
    "TREND": TokenType.TREND,
    "OVER": TokenType.OVER,
    "ALERT": TokenType.ALERT,
    "AGGREGATE": TokenType.AGGREGATE,
    "GROUP": TokenType.GROUP,
    "AND": TokenType.AND,
    "OR": TokenType.OR,
    "NOT": TokenType.NOT,
    "AT": TokenType.AT,
    "LAYER": TokenType.LAYER,
    "SEGMENT": TokenType.SEGMENT,
    "ROLE": TokenType.ROLE,
    "CORTEX": TokenType.CORTEX,
    "BY": TokenType.BY,
    "ON": TokenType.ON,
    "WINDOW": TokenType.WINDOW,
    "AVG": TokenType.AVG,
    "MIN": TokenType.MIN,
    "MAX": TokenType.MAX,
    "SUM": TokenType.SUM,
    "glyph": TokenType.GLYPH_FUNC,
}


@dataclass(frozen=True)
class Token:
    """
    A lexical token from GQL source.
    
    Attributes:
        type: The type of token (keyword, literal, operator, etc.)
        value: The token's value (string content, number, identifier name)
        line: Line number where token starts (1-indexed)
        column: Column number where token starts (1-indexed)
    
    Example:
        >>> token = Token(TokenType.STRING, "red car", 1, 16)
        >>> print(token)
        Token(STRING, 'red car', 1:16)
    """
    type: TokenType
    value: Any
    line: int
    column: int
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def is_keyword(self) -> bool:
        """Check if this token is a keyword."""
        return self.type in KEYWORDS.values()
    
    def is_operator(self) -> bool:
        """Check if this token is an operator."""
        return self.type in (
            TokenType.EQ, TokenType.LT, TokenType.GT,
            TokenType.LE, TokenType.GE, TokenType.NE
        )
    
    def is_literal(self) -> bool:
        """Check if this token is a literal value."""
        return self.type in (
            TokenType.STRING, TokenType.NUMBER,
            TokenType.DURATION, TokenType.IDENTIFIER
        )
