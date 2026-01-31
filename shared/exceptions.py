"""
Custom exception classes for the Glyphh Runtime.

All exceptions follow a consistent structure with error codes,
messages, and optional details for debugging.
"""

from typing import Any, Dict, Optional


class GlyphhRuntimeException(Exception):
    """Base exception for all runtime errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "RUNTIME_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


# Model Management Exceptions
class ModelNotFoundException(GlyphhRuntimeException):
    """Raised when a model is not found"""
    
    def __init__(self, namespace: str):
        super().__init__(
            message=f"Model not found: {namespace}",
            error_code="MODEL_NOT_FOUND",
            status_code=404,
            details={"namespace": namespace}
        )


class ModelIncompatibleException(GlyphhRuntimeException):
    """Raised when a model is incompatible with the SDK version"""
    
    def __init__(self, model_version: str, sdk_version: str):
        super().__init__(
            message=f"Model version {model_version} is incompatible with SDK version {sdk_version}",
            error_code="MODEL_INCOMPATIBLE",
            status_code=400,
            details={"model_version": model_version, "sdk_version": sdk_version}
        )


class ModelLoadException(GlyphhRuntimeException):
    """Raised when a model fails to load"""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Failed to load model: {reason}",
            error_code="MODEL_LOAD_FAILED",
            status_code=400,
            details={"reason": reason}
        )


# Namespace Exceptions
class NamespaceNotFoundException(GlyphhRuntimeException):
    """Raised when a namespace is not found"""
    
    def __init__(self, namespace: str):
        super().__init__(
            message=f"Namespace not found: {namespace}",
            error_code="NAMESPACE_NOT_FOUND",
            status_code=404,
            details={"namespace": namespace}
        )


class NamespaceQuotaExceededException(GlyphhRuntimeException):
    """Raised when a namespace exceeds its resource quota"""
    
    def __init__(self, namespace: str, resource: str, limit: Any, current: Any):
        super().__init__(
            message=f"Namespace {namespace} exceeded {resource} quota",
            error_code="QUOTA_EXCEEDED",
            status_code=429,
            details={
                "namespace": namespace,
                "resource": resource,
                "limit": limit,
                "current": current
            }
        )


# Authentication Exceptions
class AuthenticationException(GlyphhRuntimeException):
    """Raised when authentication fails"""
    
    def __init__(self, reason: str = "Invalid or missing authentication token"):
        super().__init__(
            message=reason,
            error_code="AUTHENTICATION_FAILED",
            status_code=401,
            details={"reason": reason}
        )


class AuthorizationException(GlyphhRuntimeException):
    """Raised when authorization fails"""
    
    def __init__(self, operation: str, namespace: Optional[str] = None):
        super().__init__(
            message=f"Not authorized to perform {operation}",
            error_code="AUTHORIZATION_FAILED",
            status_code=403,
            details={"operation": operation, "namespace": namespace}
        )


# License Exceptions
class LicenseException(GlyphhRuntimeException):
    """Raised when license validation fails"""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"License validation failed: {reason}",
            error_code="LICENSE_INVALID",
            status_code=503,
            details={"reason": reason}
        )


class LicenseExpiredException(LicenseException):
    """Raised when license has expired"""
    
    def __init__(self, expired_at: str):
        super().__init__(reason=f"License expired at {expired_at}")
        self.error_code = "LICENSE_EXPIRED"
        self.details["expired_at"] = expired_at


# Validation Exceptions
class ValidationException(GlyphhRuntimeException):
    """Raised when request validation fails"""
    
    def __init__(self, field: str, reason: str):
        super().__init__(
            message=f"Validation failed for {field}: {reason}",
            error_code="VALIDATION_FAILED",
            status_code=400,
            details={"field": field, "reason": reason}
        )


# Glyph Exceptions
class GlyphNotFoundException(GlyphhRuntimeException):
    """Raised when a glyph is not found"""
    
    def __init__(self, glyph_id: str, namespace: str):
        super().__init__(
            message=f"Glyph not found: {glyph_id}",
            error_code="GLYPH_NOT_FOUND",
            status_code=404,
            details={"glyph_id": glyph_id, "namespace": namespace}
        )


class EncodingException(GlyphhRuntimeException):
    """Raised when concept encoding fails"""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Failed to encode concept: {reason}",
            error_code="ENCODING_FAILED",
            status_code=400,
            details={"reason": reason}
        )


# NL Query Exceptions
class NLQueryDisabledException(GlyphhRuntimeException):
    """Raised when NL query is disabled but requested"""
    
    def __init__(self):
        super().__init__(
            message="Natural language query interface is not enabled",
            error_code="NL_QUERY_DISABLED",
            status_code=501,
            details={"hint": "Set ENABLE_NL_QUERY=true to enable"}
        )
