"""
Configuration Validator for Glyphh Runtime.

Validates EncoderConfig structure, applies defaults, and checks SDK compatibility
at startup to catch configuration errors early.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from shared.sdk_adapter import get_sdk_adapter, SDKNotAvailableError

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(message)


class ConfigurationValidator:
    """
    Validates SDK configuration and applies defaults.
    
    Responsibilities:
    - Validate EncoderConfig has required layer structure
    - Apply default similarity_weight to Roles missing it
    - Check SDK compatibility at startup
    - Log configuration warnings
    """
    
    # Default similarity weight for roles that don't specify one
    DEFAULT_SIMILARITY_WEIGHT = 1.0
    
    def __init__(self):
        self._warnings: List[str] = []
    
    @property
    def warnings(self) -> List[str]:
        """Get accumulated warnings."""
        return self._warnings.copy()
    
    def clear_warnings(self) -> None:
        """Clear accumulated warnings."""
        self._warnings = []
    
    def validate_encoder_config(self, config: Any) -> Tuple[bool, List[str]]:
        """
        Validate EncoderConfig has required structure.
        
        Checks:
        - Config has layers attribute
        - At least one layer is defined
        - Each layer has at least one segment
        - Each segment has at least one role
        
        Args:
            config: EncoderConfig instance to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check layers attribute exists
        if not hasattr(config, 'layers'):
            errors.append(
                "EncoderConfig requires explicit layer definitions. "
                "Missing 'layers' attribute."
            )
            return False, errors
        
        # Check layers is not None or empty
        if config.layers is None or len(config.layers) == 0:
            errors.append(
                "EncoderConfig requires at least one LayerConfig. "
                "The 'layers' list is empty or None."
            )
            return False, errors
        
        # Validate each layer
        for i, layer in enumerate(config.layers):
            layer_name = getattr(layer, 'name', f'layer_{i}')
            
            # Check segments attribute
            if not hasattr(layer, 'segments'):
                errors.append(
                    f"LayerConfig '{layer_name}' missing 'segments' attribute."
                )
                continue
            
            # Check segments is not empty
            if layer.segments is None or len(layer.segments) == 0:
                errors.append(
                    f"LayerConfig '{layer_name}' requires at least one SegmentConfig."
                )
                continue
            
            # Validate each segment
            for j, segment in enumerate(layer.segments):
                segment_name = getattr(segment, 'name', f'segment_{j}')
                
                # Check roles attribute
                if not hasattr(segment, 'roles'):
                    errors.append(
                        f"SegmentConfig '{segment_name}' in layer '{layer_name}' "
                        f"missing 'roles' attribute."
                    )
                    continue
                
                # Check roles is not empty
                if segment.roles is None or len(segment.roles) == 0:
                    errors.append(
                        f"SegmentConfig '{segment_name}' in layer '{layer_name}' "
                        f"requires at least one Role."
                    )
        
        return len(errors) == 0, errors
    
    def validate_encoder_config_or_raise(self, config: Any) -> None:
        """
        Validate EncoderConfig and raise ConfigurationError if invalid.
        
        This method is used when validation failures should halt execution.
        The error message will contain "layer" and "required" keywords
        for configs missing layer definitions.
        
        Args:
            config: EncoderConfig instance to validate
            
        Raises:
            ConfigurationError: If config is invalid with descriptive message
        """
        is_valid, errors = self.validate_encoder_config(config)
        
        if not is_valid:
            # Join all errors into a single message
            error_message = "; ".join(errors)
            logger.error(f"EncoderConfig validation failed: {error_message}")
            raise ConfigurationError(error_message, field="encoder_config")
    
    def apply_defaults(self, config: Any) -> Any:
        """
        Apply default values to config where missing.
        
        Currently applies:
        - Default similarity_weight (1.0) to Roles missing it
        
        Args:
            config: EncoderConfig instance
            
        Returns:
            Config with defaults applied (modified in place)
        """
        if not hasattr(config, 'layers') or config.layers is None:
            return config
        
        for layer in config.layers:
            # Apply default layer similarity_weight
            if not hasattr(layer, 'similarity_weight') or layer.similarity_weight is None:
                if hasattr(layer, '__dict__'):
                    layer.similarity_weight = 1.0
                    self._warnings.append(
                        f"Layer '{layer.name}' missing similarity_weight, using default 1.0"
                    )
            
            if not hasattr(layer, 'segments') or layer.segments is None:
                continue
            
            for segment in layer.segments:
                if not hasattr(segment, 'roles') or segment.roles is None:
                    continue
                
                for role in segment.roles:
                    # Apply default similarity_weight to roles
                    if not hasattr(role, 'similarity_weight') or role.similarity_weight is None:
                        if hasattr(role, '__dict__'):
                            role.similarity_weight = self.DEFAULT_SIMILARITY_WEIGHT
                            role_name = getattr(role, 'name', 'unknown')
                            self._warnings.append(
                                f"Role '{role_name}' missing similarity_weight, "
                                f"using default {self.DEFAULT_SIMILARITY_WEIGHT}"
                            )
        
        # Log warnings
        for warning in self._warnings:
            logger.warning(warning)
        
        return config
    
    def validate_at_startup(self) -> Dict[str, Any]:
        """
        Perform SDK compatibility check at startup.
        
        Returns:
            Dict with:
            - sdk_available: bool
            - sdk_version: str
            - api_mode: "new" or "legacy"
            - compatible: bool
            - errors: list of error messages
            - warnings: list of warning messages
        """
        result = {
            "sdk_available": False,
            "sdk_version": "unknown",
            "api_mode": "unknown",
            "compatible": False,
            "errors": [],
            "warnings": [],
        }
        
        try:
            adapter = get_sdk_adapter()
            
            result["sdk_available"] = adapter.is_available
            result["sdk_version"] = adapter.sdk_version
            
            if not adapter.is_available:
                result["errors"].append(
                    "glyphh-sdk is not installed. "
                    "Install with: pip install glyphh-sdk>=0.1.0"
                )
                return result
            
            result["api_mode"] = "new" if adapter.is_new_api else "legacy"
            result["warnings"].extend(adapter.warnings)
            
            if adapter.is_new_api:
                result["compatible"] = True
                logger.info(
                    f"SDK compatibility check passed: "
                    f"version={result['sdk_version']}, api_mode={result['api_mode']}"
                )
            else:
                result["compatible"] = True  # Still compatible, just degraded
                result["warnings"].append(
                    f"SDK version {result['sdk_version']} uses legacy API. "
                    f"Some features may be unavailable. Upgrade to >=0.1.0 for full functionality."
                )
                logger.warning(
                    f"SDK using legacy API: version={result['sdk_version']}"
                )
            
        except SDKNotAvailableError as e:
            result["errors"].append(str(e))
            logger.error(f"SDK compatibility check failed: {e}")
        except Exception as e:
            result["errors"].append(f"Unexpected error during SDK check: {e}")
            logger.error(f"SDK compatibility check error: {e}")
        
        # Log all warnings
        for warning in result["warnings"]:
            logger.warning(f"SDK compatibility warning: {warning}")
        
        return result
    
    def validate_model_config(self, model: Any) -> Tuple[bool, List[str]]:
        """
        Validate a loaded model's configuration.
        
        Args:
            model: GlyphhModel instance
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check model has encoder_config
        if not hasattr(model, 'encoder_config'):
            errors.append("Model missing encoder_config attribute")
            return False, errors
        
        # Validate the encoder config
        is_valid, config_errors = self.validate_encoder_config(model.encoder_config)
        errors.extend(config_errors)
        
        # Check model has required attributes
        if not hasattr(model, 'name'):
            errors.append("Model missing name attribute")
        
        if not hasattr(model, 'version'):
            errors.append("Model missing version attribute")
        
        return len(errors) == 0, errors


# Global singleton instance
_validator: Optional[ConfigurationValidator] = None


def get_config_validator() -> ConfigurationValidator:
    """Get the global ConfigurationValidator singleton."""
    global _validator
    if _validator is None:
        _validator = ConfigurationValidator()
    return _validator
