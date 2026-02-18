"""
GQL Lexer - Tokenizes GQL source into a stream of tokens.

The lexer handles:
- Keywords (case-insensitive for most, case-sensitive for 'glyph')
- String literals with escape sequences
- Numeric literals (integers and decimals)
- Duration literals (e.g., 30d, 7h, 15m, 45s)
- Identifiers
- Operators and punctuation
- Whitespace and newlines (skipped, but tracked for position)
"""

import re
from typing import Iterator, List, Optional
from glyphh.gql.tokens import Token, TokenType, KEYWORDS
from glyphh.gql.exceptions import LexerError


class GQLLexer:
    """
    Tokenizes GQL source into a stream of tokens.
    
    The lexer is case-insensitive for keywords (except 'glyph' function)
    and tracks line/column positions for error reporting.
    
    Example:
        >>> lexer = GQLLexer('FIND SIMILAR TO "red car" LIMIT 10')
        >>> tokens = lexer.tokenize()
        >>> for token in tokens:
        ...     print(token)
        Token(FIND, 'FIND', 1:1)
        Token(SIMILAR, 'SIMILAR', 1:6)
        Token(TO, 'TO', 1:14)
        Token(STRING, 'red car', 1:17)
        Token(LIMIT, 'LIMIT', 1:27)
        Token(NUMBER, 10, 1:33)
        Token(EOF, None, 1:35)
    """
    
    # Pattern for duration literals: number followed by d/h/m/s
    DURATION_PATTERN = re.compile(r'^(\d+)([dhms])$')
    
    # Pattern for identifiers: letter or underscore, followed by letters, digits, underscores, or hyphens
    IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_-]*$')
    
    def __init__(self, source: str):
        """
        Initialize the lexer with GQL source.
        
        Args:
            source: The GQL source string to tokenize
        """
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self._tokens: Optional[List[Token]] = None
    
    def tokenize(self) -> List[Token]:
        """
        Tokenize the entire source and return list of tokens.
        
        Returns:
            List of Token objects, ending with EOF token
        
        Raises:
            LexerError: If an invalid character or malformed literal is encountered
        """
        if self._tokens is not None:
            return self._tokens
        
        tokens = []
        for token in self:
            tokens.append(token)
        
        self._tokens = tokens
        return tokens
    
    def __iter__(self) -> Iterator[Token]:
        """Iterate over tokens lazily."""
        self.pos = 0
        self.line = 1
        self.column = 1
        
        while True:
            token = self._next_token()
            yield token
            if token.type == TokenType.EOF:
                break
    
    def _next_token(self) -> Token:
        """
        Get the next token from source.
        
        Returns:
            The next Token
        
        Raises:
            LexerError: If an invalid character is encountered
        """
        self._skip_whitespace()
        
        if self.pos >= len(self.source):
            return Token(TokenType.EOF, None, self.line, self.column)
        
        char = self.source[self.pos]
        start_line = self.line
        start_column = self.column
        
        # String literal
        if char == '"':
            value = self._read_string()
            return Token(TokenType.STRING, value, start_line, start_column)
        
        # Number or duration
        if char.isdigit():
            return self._read_number_or_duration(start_line, start_column)
        
        # Identifier or keyword
        if char.isalpha() or char == '_':
            value = self._read_identifier()
            # Check if it's a keyword (case-insensitive lookup, except 'glyph')
            upper_value = value.upper()
            if value == "glyph":
                return Token(TokenType.GLYPH_FUNC, value, start_line, start_column)
            elif upper_value in KEYWORDS:
                return Token(KEYWORDS[upper_value], value.upper(), start_line, start_column)
            else:
                return Token(TokenType.IDENTIFIER, value, start_line, start_column)
        
        # Operators
        if char == '=':
            self._advance()
            return Token(TokenType.EQ, '=', start_line, start_column)
        
        if char == '<':
            self._advance()
            if self.pos < len(self.source) and self.source[self.pos] == '=':
                self._advance()
                return Token(TokenType.LE, '<=', start_line, start_column)
            return Token(TokenType.LT, '<', start_line, start_column)
        
        if char == '>':
            self._advance()
            if self.pos < len(self.source) and self.source[self.pos] == '=':
                self._advance()
                return Token(TokenType.GE, '>=', start_line, start_column)
            return Token(TokenType.GT, '>', start_line, start_column)
        
        if char == '!':
            self._advance()
            if self.pos < len(self.source) and self.source[self.pos] == '=':
                self._advance()
                return Token(TokenType.NE, '!=', start_line, start_column)
            # Invalid: standalone '!'
            raise LexerError(
                message="Expected '=' after '!'",
                line=start_line,
                column=start_column,
                source_snippet=self._get_snippet(start_column),
                suggestion="Use '!=' for not-equal comparison"
            )
        
        # Punctuation
        if char == '(':
            self._advance()
            return Token(TokenType.LPAREN, '(', start_line, start_column)
        
        if char == ')':
            self._advance()
            return Token(TokenType.RPAREN, ')', start_line, start_column)
        
        if char == ',':
            self._advance()
            return Token(TokenType.COMMA, ',', start_line, start_column)
        
        if char == '.':
            self._advance()
            return Token(TokenType.DOT, '.', start_line, start_column)
        
        # Invalid character
        raise LexerError(
            message=f"Unexpected character '{char}'",
            line=start_line,
            column=start_column,
            source_snippet=self._get_snippet(start_column),
            invalid_char=char
        )
    
    def _advance(self) -> None:
        """Advance position by one character, updating line/column."""
        if self.pos < len(self.source):
            if self.source[self.pos] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1
    
    def _skip_whitespace(self) -> None:
        """Skip whitespace characters, updating line/column tracking."""
        while self.pos < len(self.source) and self.source[self.pos].isspace():
            self._advance()
    
    def _read_string(self) -> str:
        """
        Read a quoted string literal with escape sequence support.
        
        Supports escape sequences: \\, \", \n, \t, \r
        
        Returns:
            The string content (without quotes)
        
        Raises:
            LexerError: If string is unterminated or has invalid escape
        """
        start_line = self.line
        start_column = self.column
        
        # Skip opening quote
        self._advance()
        
        result = []
        while self.pos < len(self.source):
            char = self.source[self.pos]
            
            if char == '"':
                # End of string
                self._advance()
                return ''.join(result)
            
            if char == '\\':
                # Escape sequence
                self._advance()
                if self.pos >= len(self.source):
                    raise LexerError(
                        message="Unterminated escape sequence",
                        line=self.line,
                        column=self.column,
                        source_snippet=self._get_snippet(start_column)
                    )
                
                escape_char = self.source[self.pos]
                if escape_char == '\\':
                    result.append('\\')
                elif escape_char == '"':
                    result.append('"')
                elif escape_char == 'n':
                    result.append('\n')
                elif escape_char == 't':
                    result.append('\t')
                elif escape_char == 'r':
                    result.append('\r')
                else:
                    raise LexerError(
                        message=f"Invalid escape sequence '\\{escape_char}'",
                        line=self.line,
                        column=self.column - 1,
                        source_snippet=self._get_snippet(start_column),
                        suggestion="Valid escapes: \\\\, \\\", \\n, \\t, \\r"
                    )
                self._advance()
            elif char == '\n':
                # Newlines not allowed in strings
                raise LexerError(
                    message="Unterminated string literal (newline in string)",
                    line=start_line,
                    column=start_column,
                    source_snippet=self._get_snippet(start_column),
                    suggestion="Close string before newline or use \\n for newline character"
                )
            else:
                result.append(char)
                self._advance()
        
        # Reached end of source without closing quote
        raise LexerError(
            message="Unterminated string literal",
            line=start_line,
            column=start_column,
            source_snippet=self._get_snippet(start_column),
            suggestion="Add closing quote"
        )
    
    def _read_number_or_duration(self, start_line: int, start_column: int) -> Token:
        """
        Read a numeric literal or duration literal.
        
        Duration format: <number><unit> where unit is d, h, m, or s
        Number format: integer or decimal
        
        Returns:
            Token with NUMBER or DURATION type
        """
        result = []
        has_decimal = False
        
        while self.pos < len(self.source):
            char = self.source[self.pos]
            
            if char.isdigit():
                result.append(char)
                self._advance()
            elif char == '.' and not has_decimal:
                # Check if next char is digit (decimal point) or not (end of number)
                if self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit():
                    has_decimal = True
                    result.append(char)
                    self._advance()
                else:
                    break
            elif char in 'dhms' and not has_decimal:
                # Duration suffix
                result.append(char)
                self._advance()
                value = ''.join(result)
                return Token(TokenType.DURATION, value, start_line, start_column)
            else:
                break
        
        value_str = ''.join(result)
        if has_decimal:
            value = float(value_str)
        else:
            value = int(value_str)
        
        return Token(TokenType.NUMBER, value, start_line, start_column)
    
    def _read_identifier(self) -> str:
        """
        Read an identifier.
        
        Identifier format: [a-zA-Z_][a-zA-Z0-9_-]*
        
        Returns:
            The identifier string
        """
        result = []
        
        while self.pos < len(self.source):
            char = self.source[self.pos]
            if char.isalnum() or char in '_-':
                result.append(char)
                self._advance()
            else:
                break
        
        return ''.join(result)
    
    def _get_snippet(self, start_column: int, context_chars: int = 30) -> str:
        """
        Get a snippet of source around the current position for error messages.
        
        Args:
            start_column: Column where the problematic token started
            context_chars: Number of characters of context to include
        
        Returns:
            Source snippet string
        """
        # Find start of current line
        line_start = self.pos
        while line_start > 0 and self.source[line_start - 1] != '\n':
            line_start -= 1
        
        # Find end of current line
        line_end = self.pos
        while line_end < len(self.source) and self.source[line_end] != '\n':
            line_end += 1
        
        line_content = self.source[line_start:line_end]
        
        # Truncate if too long
        if len(line_content) > context_chars * 2:
            # Center around the error position
            error_pos = start_column - 1
            start = max(0, error_pos - context_chars)
            end = min(len(line_content), error_pos + context_chars)
            line_content = ('...' if start > 0 else '') + line_content[start:end] + ('...' if end < len(line_content) else '')
        
        return line_content
