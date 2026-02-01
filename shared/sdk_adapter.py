"""
SDK Adapter for Glyphh Runtime.

Provides version detection and compatibility layer for the glyphh-sdk,
abstracting differences between SDK versions and providing a consistent interface.
"""

import logging
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class SDKNotAvailableError(Exception):
    """Raised when the SDK is not installed or unavailable."""
    pass


class SDKAdapter:
    """
    Adapter for SDK version compatibility.
    
    Provides:
    - SDK version detection
    - API mode detection (new explicit vs legacy implicit)
    - Factory methods for SDK components with version-appropriate instantiation
    - Fallback handling for missing SDK features
    """
    
    def __init__(self):
        self._sdk_version: Optional[str] = None
        self._is_new_api: Optional[bool] = None
        self._config_classes: Optional[Dict[str, Type]] = None
        self._warnings: list = []
    
    @property
    def sdk_version(self) -> str:
        """Get the installed SDK version."""
        if self._sdk_version is None:
            try:
                import glyphh
                self._sdk_version = getattr(glyphh, '__version__', 'unknown')
            except ImportError:
                self._sdk_version = 'not_installed'
        return self._sdk_version
    
    @property
    def is_new_api(self) -> bool:
        """
        Check if the new explicit API is available.
        
        The new API requires:
        - EncoderConfig with explicit layers parameter
        - LayerConfig, SegmentConfig, Role classes
        - SimilarityCalculator class
        """
        if self._is_new_api is None:
            try:
                from glyphh import (
                    EncoderConfig, LayerConfig, SegmentConfig, Role,
                    SimilarityCalculator, Encoder
                )
                # Test that LayerConfig is required (new API)
                # In new API, EncoderConfig requires layers parameter
                self._is_new_api = True
                logger.info(f"SDK {self.sdk_version} detected with new explicit API")
            except ImportError as e:
                self._is_new_api = False
                self._warnings.append(f"New SDK API unavailable: {e}")
                logger.warning(f"SDK new API not available: {e}")
        return self._is_new_api
    
    @property
    def is_available(self) -> bool:
        """Check if SDK is installed and available."""
        return self.sdk_version != 'not_installed'
    
    @property
    def warnings(self) -> list:
        """Get any warnings from SDK detection."""
        return self._warnings.copy()
    
    def import_config_classes(self) -> Dict[str, Type]:
        """
        Import configuration classes from SDK.
        
        Returns dict with keys: EncoderConfig, LayerConfig, SegmentConfig, Role
        
        Raises:
            SDKNotAvailableError: If SDK is not installed
            ImportError: If required classes are not available
        """
        if self._config_classes is not None:
            return self._config_classes
        
        if not self.is_available:
            raise SDKNotAvailableError(
                "glyphh-sdk is required. Install with: pip install glyphh-sdk>=0.1.0"
            )
        
        try:
            from glyphh import EncoderConfig, LayerConfig, SegmentConfig, Role
            self._config_classes = {
                'EncoderConfig': EncoderConfig,
                'LayerConfig': LayerConfig,
                'SegmentConfig': SegmentConfig,
                'Role': Role,
            }
            return self._config_classes
        except ImportError as e:
            raise ImportError(
                f"Failed to import SDK configuration classes: {e}. "
                f"Ensure glyphh-sdk>=0.1.0 is installed."
            )
    
    def create_encoder(self, config: Any) -> Any:
        """
        Create an Encoder instance compatible with installed SDK.
        
        Args:
            config: EncoderConfig instance
            
        Returns:
            Encoder instance
            
        Raises:
            SDKNotAvailableError: If SDK is not installed
        """
        if not self.is_available:
            raise SDKNotAvailableError("SDK not available for encoder creation")
        
        try:
            from glyphh import Encoder
            return Encoder(config)
        except ImportError as e:
            raise SDKNotAvailableError(f"Failed to import Encoder: {e}")
        except Exception as e:
            raise ValueError(f"Failed to create Encoder: {e}")
    
    def create_similarity_calculator(self) -> Optional[Any]:
        """
        Create a SimilarityCalculator if available.
        
        Returns:
            SimilarityCalculator instance, or None if not available
        """
        if not self.is_available:
            return None
        
        try:
            from glyphh import SimilarityCalculator
            return SimilarityCalculator()
        except ImportError:
            self._warnings.append("SimilarityCalculator not available - using fallback")
            logger.warning("SimilarityCalculator not available in SDK")
            return None
        except Exception as e:
            self._warnings.append(f"Failed to create SimilarityCalculator: {e}")
            logger.warning(f"Failed to create SimilarityCalculator: {e}")
            return None
    
    def create_intent_encoder(self, config: Any) -> Optional[Any]:
        """
        Create an IntentEncoder if available.
        
        Args:
            config: EncoderConfig instance
            
        Returns:
            IntentEncoder instance, or None if not available
        """
        if not self.is_available:
            return None
        
        try:
            from glyphh import IntentEncoder
            encoder = IntentEncoder(config)
            # Add default patterns if available
            if hasattr(encoder, 'add_defaults'):
                encoder.add_defaults()
            return encoder
        except ImportError:
            logger.warning("IntentEncoder not available in SDK")
            return None
        except Exception as e:
            logger.warning(f"Failed to create IntentEncoder: {e}")
            return None
    
    def get_compatibility_status(self) -> Dict[str, Any]:
        """
        Get SDK compatibility status for health checks.
        
        Returns:
            Dict with version, api_mode, degraded status, and warnings
        """
        return {
            "version": self.sdk_version,
            "api_mode": "new" if self.is_new_api else "legacy",
            "available": self.is_available,
            "degraded": not self.is_new_api and self.is_available,
            "warnings": self.warnings,
        }


# Global singleton instance
_sdk_adapter: Optional[SDKAdapter] = None


def get_sdk_adapter() -> SDKAdapter:
    """Get the global SDKAdapter singleton."""
    global _sdk_adapter
    if _sdk_adapter is None:
        _sdk_adapter = SDKAdapter()
    return _sdk_adapter
