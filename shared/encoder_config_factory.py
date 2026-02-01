"""
Encoder Configuration Factory for Glyphh Runtime.

Creates properly structured EncoderConfig instances with explicit
LayerConfig, SegmentConfig, and Role definitions as required by the new SDK API.
"""

import logging
from typing import Any, Dict, List, Optional

from shared.sdk_adapter import get_sdk_adapter, SDKNotAvailableError

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class EncoderConfigFactory:
    """
    Factory for creating SDK-compatible EncoderConfig instances.
    
    All configs created by this factory use the new explicit structure
    with LayerConfig, SegmentConfig, and Role definitions.
    """
    
    @staticmethod
    def create_default_config(
        dimension: int = 10000,
        seed: int = 42,
    ) -> Any:
        """
        Create a default EncoderConfig with standard layer structure.
        
        The default structure includes:
        - semantic layer: concept and relation segments
        - temporal layer: sequence segment
        
        Args:
            dimension: Embedding dimension
            seed: Random seed for reproducibility
            
        Returns:
            EncoderConfig with explicit layer definitions
            
        Raises:
            SDKNotAvailableError: If SDK is not installed
            ConfigurationError: If config creation fails
        """
        adapter = get_sdk_adapter()
        classes = adapter.import_config_classes()
        
        EncoderConfig = classes['EncoderConfig']
        LayerConfig = classes['LayerConfig']
        SegmentConfig = classes['SegmentConfig']
        Role = classes['Role']
        
        try:
            config = EncoderConfig(
                dimension=dimension,
                seed=seed,
                layers=[
                    LayerConfig(
                        name="semantic",
                        similarity_weight=1.0,
                        segments=[
                            SegmentConfig(
                                name="concept",
                                roles=[
                                    Role(name="name", similarity_weight=1.0),
                                    Role(name="text", similarity_weight=0.9),
                                    Role(name="type", similarity_weight=0.8),
                                    Role(name="category", similarity_weight=0.7),
                                ]
                            ),
                            SegmentConfig(
                                name="attributes",
                                roles=[
                                    Role(name="primary", similarity_weight=1.0),
                                    Role(name="secondary", similarity_weight=0.7),
                                    Role(name="context", similarity_weight=0.5),
                                ]
                            ),
                        ]
                    ),
                    LayerConfig(
                        name="relational",
                        similarity_weight=0.8,
                        segments=[
                            SegmentConfig(
                                name="relations",
                                roles=[
                                    Role(name="subject", similarity_weight=0.8),
                                    Role(name="predicate", similarity_weight=1.0),
                                    Role(name="object", similarity_weight=0.8),
                                ]
                            ),
                        ]
                    ),
                ]
            )
            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to create default config: {e}")
    
    @staticmethod
    def create_from_model(model: Any) -> Any:
        """
        Extract and validate EncoderConfig from a loaded GlyphhModel.
        
        Args:
            model: Loaded GlyphhModel instance
            
        Returns:
            EncoderConfig extracted from model
            
        Raises:
            ConfigurationError: If model config is invalid or missing
        """
        if not hasattr(model, 'encoder_config'):
            raise ConfigurationError(
                "Model missing encoder_config. "
                "Ensure the model was created with a valid EncoderConfig."
            )
        
        config = model.encoder_config
        
        # Validate the config has required structure
        if not hasattr(config, 'layers') or not config.layers:
            raise ConfigurationError(
                "EncoderConfig requires explicit layer definitions. "
                "The model appears to use legacy config format."
            )
        
        # Validate each layer has segments
        for layer in config.layers:
            if not hasattr(layer, 'segments') or not layer.segments:
                raise ConfigurationError(
                    f"LayerConfig '{layer.name}' requires at least one SegmentConfig."
                )
            
            # Validate each segment has roles
            for segment in layer.segments:
                if not hasattr(segment, 'roles') or not segment.roles:
                    raise ConfigurationError(
                        f"SegmentConfig '{segment.name}' in layer '{layer.name}' "
                        f"requires at least one Role."
                    )
        
        return config
    
    @staticmethod
    def create_intent_config(
        dimension: int = 10000,
        seed: int = 42,
    ) -> Any:
        """
        Create EncoderConfig optimized for intent matching.
        
        This config is designed for natural language query matching
        with emphasis on semantic similarity.
        
        Args:
            dimension: Embedding dimension
            seed: Random seed
            
        Returns:
            EncoderConfig with intent-optimized layer structure
            
        Raises:
            SDKNotAvailableError: If SDK is not installed
            ConfigurationError: If config creation fails
        """
        adapter = get_sdk_adapter()
        classes = adapter.import_config_classes()
        
        EncoderConfig = classes['EncoderConfig']
        LayerConfig = classes['LayerConfig']
        SegmentConfig = classes['SegmentConfig']
        Role = classes['Role']
        
        try:
            config = EncoderConfig(
                dimension=dimension,
                seed=seed,
                layers=[
                    LayerConfig(
                        name="intent",
                        similarity_weight=1.0,
                        segments=[
                            SegmentConfig(
                                name="query",
                                roles=[
                                    Role(name="intent_type", similarity_weight=1.0),
                                    Role(name="phrase", similarity_weight=0.9),
                                    Role(name="keywords", similarity_weight=0.8),
                                    Role(name="context", similarity_weight=0.6),
                                ]
                            ),
                            SegmentConfig(
                                name="parameters",
                                roles=[
                                    Role(name="entity", similarity_weight=0.9),
                                    Role(name="value", similarity_weight=0.7),
                                    Role(name="modifier", similarity_weight=0.5),
                                ]
                            ),
                        ]
                    ),
                ]
            )
            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to create intent config: {e}")
    
    @staticmethod
    def create_minimal_config(
        dimension: int = 10000,
        seed: int = 42,
        layer_name: str = "default",
        segment_name: str = "main",
        role_names: Optional[List[str]] = None,
    ) -> Any:
        """
        Create a minimal EncoderConfig with a single layer/segment.
        
        Useful for simple use cases or testing.
        
        Args:
            dimension: Embedding dimension
            seed: Random seed
            layer_name: Name for the single layer
            segment_name: Name for the single segment
            role_names: List of role names (defaults to ["content"])
            
        Returns:
            Minimal EncoderConfig
        """
        adapter = get_sdk_adapter()
        classes = adapter.import_config_classes()
        
        EncoderConfig = classes['EncoderConfig']
        LayerConfig = classes['LayerConfig']
        SegmentConfig = classes['SegmentConfig']
        Role = classes['Role']
        
        if role_names is None:
            role_names = ["content"]
        
        roles = [Role(name=name, similarity_weight=1.0) for name in role_names]
        
        return EncoderConfig(
            dimension=dimension,
            seed=seed,
            layers=[
                LayerConfig(
                    name=layer_name,
                    similarity_weight=1.0,
                    segments=[
                        SegmentConfig(
                            name=segment_name,
                            roles=roles,
                        )
                    ]
                )
            ]
        )
    
    @staticmethod
    def extract_nl_encoder_config(model: Any) -> Optional[Dict[str, Any]]:
        """
        Extract NL encoder config from a loaded model.
        
        Extracts the NL encoder configuration (intent patterns) from a
        GlyphhModel's encoder_config if present.
        
        Args:
            model: Loaded GlyphhModel instance
            
        Returns:
            NL encoder config dict if present, None otherwise
            
        Example:
            >>> model = GlyphhModel.from_file("my_model.glyphh")
            >>> nl_config = EncoderConfigFactory.extract_nl_encoder_config(model)
            >>> if nl_config:
            ...     patterns = nl_config.get("patterns", [])
        """
        if not hasattr(model, 'encoder_config'):
            logger.debug("Model missing encoder_config")
            return None
        
        config = model.encoder_config
        
        # Check for nl_encoder_config in the encoder config
        if hasattr(config, 'nl_encoder_config') and config.nl_encoder_config is not None:
            nl_config = config.nl_encoder_config
            
            # If it has a to_dict method, use it
            if hasattr(nl_config, 'to_dict'):
                result = nl_config.to_dict()
                logger.info(
                    f"Extracted NL encoder config with {len(result.get('patterns', []))} patterns"
                )
                return result
            
            # If it's already a dict, return it
            if isinstance(nl_config, dict):
                logger.info(
                    f"Extracted NL encoder config dict with {len(nl_config.get('patterns', []))} patterns"
                )
                return nl_config
        
        # Also check for legacy intent_patterns field on the model itself
        if hasattr(model, 'intent_patterns') and model.intent_patterns is not None:
            logger.info(
                f"Extracted legacy intent_patterns with {len(model.intent_patterns.get('patterns', []))} patterns"
            )
            return model.intent_patterns
        
        logger.debug("No NL encoder config found in model")
        return None
