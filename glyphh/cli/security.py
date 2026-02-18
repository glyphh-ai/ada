"""
Security utilities for the Glyphh CLI.

Provides input sanitization and validation to prevent:
- Command injection
- Path traversal
- Malicious input patterns
"""

import re
from typing import Optional


# Dangerous shell characters that could enable command injection
SHELL_METACHARACTERS = set(';&|`$(){}[]<>\\!#*?~')

# Patterns that look like injection attempts
INJECTION_PATTERNS = [
    r';\s*\w+',           # Command chaining with semicolon
    r'\|\s*\w+',          # Pipe to another command
    r'`[^`]+`',           # Backtick command substitution
    r'\$\([^)]+\)',       # $() command substitution
    r'\$\{[^}]+\}',       # ${} variable expansion
    r'>\s*[/\w]',         # Output redirection
    r'<\s*[/\w]',         # Input redirection
    r'&&\s*\w+',          # AND command chaining
    r'\|\|\s*\w+',        # OR command chaining
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    r'\.\.',              # Parent directory
    r'^/',                # Absolute path (when not expected)
    r'^~',                # Home directory expansion
]

# Maximum input lengths
MAX_QUERY_LENGTH = 1000
MAX_COMMAND_ARG_LENGTH = 256
MAX_PATH_LENGTH = 512


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


def sanitize_query(query: str) -> str:
    """
    Sanitize a natural language query.
    
    Removes or escapes potentially dangerous characters while
    preserving the semantic meaning for HDC matching.
    """
    if not query:
        return ""
    
    # Enforce length limit
    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]
    
    # Remove null bytes
    query = query.replace('\x00', '')
    
    # Remove ANSI escape sequences
    query = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', query)
    
    # Remove control characters (except newline, tab)
    query = ''.join(c for c in query if c == '\n' or c == '\t' or (ord(c) >= 32 and ord(c) < 127) or ord(c) > 127)
    
    return query.strip()


def validate_command_arg(arg: str, allow_paths: bool = False) -> str:
    """
    Validate and sanitize a command argument.
    
    Args:
        arg: The argument to validate
        allow_paths: Whether to allow file paths
        
    Returns:
        The sanitized argument
        
    Raises:
        SecurityError: If the argument contains dangerous patterns
    """
    if not arg:
        return ""
    
    # Enforce length limit
    if len(arg) > MAX_COMMAND_ARG_LENGTH:
        raise SecurityError(f"Argument too long (max {MAX_COMMAND_ARG_LENGTH} chars)")
    
    # Check for shell metacharacters
    dangerous_chars = SHELL_METACHARACTERS.intersection(set(arg))
    if dangerous_chars:
        raise SecurityError(f"Invalid characters in argument: {dangerous_chars}")
    
    # Check for injection patterns
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, arg):
            raise SecurityError("Invalid input pattern detected")
    
    # Check for path traversal if paths not allowed
    if not allow_paths:
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, arg):
                raise SecurityError("Path traversal not allowed")
    
    return arg


def validate_path(path: str, must_be_relative: bool = True) -> str:
    """
    Validate a file path argument.
    
    Args:
        path: The path to validate
        must_be_relative: Whether the path must be relative
        
    Returns:
        The validated path
        
    Raises:
        SecurityError: If the path is invalid or dangerous
    """
    if not path:
        return ""
    
    # Enforce length limit
    if len(path) > MAX_PATH_LENGTH:
        raise SecurityError(f"Path too long (max {MAX_PATH_LENGTH} chars)")
    
    # Remove null bytes
    path = path.replace('\x00', '')
    
    # Check for path traversal
    if '..' in path:
        raise SecurityError("Path traversal (..) not allowed")
    
    # Check for absolute paths if relative required
    if must_be_relative and (path.startswith('/') or path.startswith('~')):
        raise SecurityError("Absolute paths not allowed")
    
    # Check for shell metacharacters in path
    dangerous_chars = SHELL_METACHARACTERS.intersection(set(path))
    if dangerous_chars:
        raise SecurityError(f"Invalid characters in path: {dangerous_chars}")
    
    return path


def is_safe_identifier(name: str) -> bool:
    """
    Check if a string is a safe identifier (model name, org name, etc).
    
    Safe identifiers contain only alphanumeric characters, hyphens,
    and underscores, and don't start with a hyphen.
    """
    if not name:
        return False
    
    # Must match pattern: alphanumeric, hyphen, underscore
    # Cannot start with hyphen
    return bool(re.match(r'^[a-zA-Z0-9_][a-zA-Z0-9_-]*$', name))


def sanitize_for_display(text: str) -> str:
    """
    Sanitize text for terminal display.
    
    Removes ANSI escape sequences and control characters
    that could manipulate the terminal.
    """
    if not text:
        return ""
    
    # Remove ANSI escape sequences
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    
    # Remove other escape sequences
    text = re.sub(r'\x1b[^[].', '', text)
    
    # Remove control characters except newline, tab, carriage return
    text = ''.join(c for c in text if c in '\n\t\r' or (ord(c) >= 32 and ord(c) < 127) or ord(c) > 127)
    
    return text
