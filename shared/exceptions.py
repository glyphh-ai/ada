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
    
    def __init__(self, org_id: str, model_id: str):
        super().__init__(
            message=f"Model not found: org={org_id}, model={model_id}",
            error_code="MODEL_NOT_FOUND",
            status_code=404,
            details={"org_id": org_id, "model_id": model_id}
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

    def __init__(self, namespace_id: str, org_id: str):
        super().__init__(
            message=f"Namespace not found: {namespace_id}",
            error_code="NAMESPACE_NOT_FOUND",
            status_code=404,
            details={"namespace_id": namespace_id, "org_id": org_id}
        )


class NamespaceQuotaExceededException(GlyphhRuntimeException):
    """Raised when a namespace exceeds its quota"""

    def __init__(self, namespace_id: str, org_id: str, limit: Any, current: Any):
        super().__init__(
            message=f"Namespace quota exceeded: {namespace_id}",
            error_code="NAMESPACE_QUOTA_EXCEEDED",
            status_code=429,
            details={
                "namespace_id": namespace_id,
                "org_id": org_id,
                "limit": limit,
                "current": current,
            }
        )


# Quota Exceptions
class QuotaExceededException(GlyphhRuntimeException):
    """Raised when a model exceeds its resource quota"""
    
    def __init__(
        self,
        org_id: str,
        model_id: str,
        resource: str,
        limit: Any,
        current: Any,
        message: Optional[str] = None
    ):
        default_message = f"Quota exceeded for org={org_id}, model={model_id}: {resource}"
        super().__init__(
            message=message or default_message,
            error_code="QUOTA_EXCEEDED",
            status_code=429,
            details={
                "org_id": org_id,
                "model_id": model_id,
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
    
    def __init__(
        self,
        operation: str = "access",
        org_id: Optional[str] = None,
        model_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        scope = ""
        if org_id:
            scope = f" on org={org_id}"
            if model_id:
                scope += f", model={model_id}"
        super().__init__(
            message=f"Not authorized to perform {operation}{scope}",
            error_code="AUTHORIZATION_FAILED",
            status_code=403,
            details={
                "operation": operation,
                "org_id": org_id,
                "model_id": model_id,
                "user_id": user_id,
            }
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
    
    def __init__(self, glyph_id: str, org_id: str, model_id: str):
        super().__init__(
            message=f"Glyph not found: {glyph_id}",
            error_code="GLYPH_NOT_FOUND",
            status_code=404,
            details={"glyph_id": glyph_id, "org_id": org_id, "model_id": model_id}
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


# Stored Procedure Exceptions
class ConflictException(GlyphhRuntimeException):
    """Raised when a resource conflict occurs (e.g., duplicate name)"""
    
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="CONFLICT",
            status_code=409,
            details={}
        )


class NotFoundException(GlyphhRuntimeException):
    """Raised when a resource is not found"""
    
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=404,
            details={}
        )
