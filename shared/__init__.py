"""
Shared components for Glyphh Runtime.

Includes exceptions, middleware, retry utilities, and circuit breakers.
"""

from shared.exceptions import (
    AuthenticationException,
    AuthorizationException,
    EncodingException,
    GlyphNotFoundException,
    GlyphhRuntimeException,
    LicenseException,
    LicenseExpiredException,
    ModelIncompatibleException,
    ModelLoadException,
    ModelNotFoundException,
    NamespaceNotFoundException,
    NamespaceQuotaExceededException,
    NLQueryDisabledException,
    ValidationException,
)
from shared.retry import (
    CircuitBreaker,
    CircuitState,
    RetryConfig,
    circuit_breaker,
    retry_with_backoff,
)

__all__ = [
    # Exceptions
    "GlyphhRuntimeException",
    "ModelNotFoundException",
    "ModelIncompatibleException",
    "ModelLoadException",
    "NamespaceNotFoundException",
    "NamespaceQuotaExceededException",
    "AuthenticationException",
    "AuthorizationException",
    "LicenseException",
    "LicenseExpiredException",
    "ValidationException",
    "GlyphNotFoundException",
    "EncodingException",
    "NLQueryDisabledException",
    # Retry utilities
    "RetryConfig",
    "retry_with_backoff",
    "CircuitBreaker",
    "CircuitState",
    "circuit_breaker",
]
