"""
Namespace Configuration Service.

Manages per-namespace configuration for similarity weights, beam search,
and fact tree parameters.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import ModelConfig
from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SimilarityWeights:
    """Weights for different edge types in similarity computation."""
    neural_cortex: float = 1.0
    neural_layer: float = 0.8
    neural_segment: float = 0.6
    neural_role: float = 0.4
    temporal_cortex: float = 0.7
    temporal_layer: float = 0.5
    temporal_segment: float = 0.3
    temporal_role: float = 0.2
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "SimilarityWeights":
        """Create from dictionary."""
        return cls(
            neural_cortex=data.get("neural_cortex", 1.0),
            neural_layer=data.get("neural_layer", 0.8),
            neural_segment=data.get("neural_segment", 0.6),
            neural_role=data.get("neural_role", 0.4),
            temporal_cortex=data.get("temporal_cortex", 0.7),
            temporal_layer=data.get("temporal_layer", 0.5),
            temporal_segment=data.get("temporal_segment", 0.3),
            temporal_role=data.get("temporal_role", 0.2),
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "neural_cortex": self.neural_cortex,
            "neural_layer": self.neural_layer,
            "neural_segment": self.neural_segment,
            "neural_role": self.neural_role,
            "temporal_cortex": self.temporal_cortex,
            "temporal_layer": self.temporal_layer,
            "temporal_segment": self.temporal_segment,
            "temporal_role": self.temporal_role,
        }


@dataclass
class NamespaceConfig:
    """Complete configuration for a namespace."""
    namespace: str
    similarity_weights: SimilarityWeights = field(default_factory=SimilarityWeights)
    beam_width: int = 5
    max_tree_depth: int = 3
    
    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "NamespaceConfig":
        """Create from ModelConfig database model."""
        weights = SimilarityWeights.from_dict(config.similarity_weights or {})
        return cls(
            namespace=config.namespace,
            similarity_weights=weights,
            beam_width=config.beam_width or settings.default_beam_width,
            max_tree_depth=config.max_tree_depth or settings.default_max_tree_depth,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "namespace": self.namespace,
            "similarity_weights": self.similarity_weights.to_dict(),
            "beam_width": self.beam_width,
            "max_tree_depth": self.max_tree_depth,
        }


class NamespaceConfigService:
    """
    Service for managing per-namespace configuration.
    
    Responsibilities:
    - Get/set similarity weights per namespace
    - Get/set beam search parameters per namespace
    - Get/set fact tree parameters per namespace
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize NamespaceConfigService.
        
        Args:
            session: Async SQLAlchemy session
        """
        self._session = session
    
    async def get_config(self, namespace: str) -> NamespaceConfig:
        """
        Get configuration for a namespace.
        
        Args:
            namespace: Model namespace
            
        Returns:
            NamespaceConfig with all settings
        """
        result = await self._session.execute(
            select(ModelConfig).where(ModelConfig.namespace == namespace)
        )
        config = result.scalar_one_or_none()
        
        if config:
            return NamespaceConfig.from_model_config(config)
        
        # Return defaults
        return NamespaceConfig(
            namespace=namespace,
            similarity_weights=SimilarityWeights(),
            beam_width=settings.default_beam_width,
            max_tree_depth=settings.default_max_tree_depth,
        )
    
    async def update_similarity_weights(
        self,
        namespace: str,
        weights: Dict[str, float],
    ) -> SimilarityWeights:
        """
        Update similarity weights for a namespace.
        
        Args:
            namespace: Model namespace
            weights: Dict of edge type -> weight
            
        Returns:
            Updated SimilarityWeights
        """
        # Get current weights
        current = await self.get_config(namespace)
        current_dict = current.similarity_weights.to_dict()
        
        # Merge with new weights
        current_dict.update(weights)
        new_weights = SimilarityWeights.from_dict(current_dict)
        
        # Update in database
        await self._session.execute(
            update(ModelConfig)
            .where(ModelConfig.namespace == namespace)
            .values(similarity_weights=new_weights.to_dict())
        )
        
        logger.info(f"Updated similarity weights for namespace '{namespace}'")
        
        return new_weights
    
    async def update_beam_width(
        self,
        namespace: str,
        beam_width: int,
    ) -> int:
        """
        Update beam width for a namespace.
        
        Args:
            namespace: Model namespace
            beam_width: New beam width (1-20)
            
        Returns:
            Updated beam width
        """
        # Validate
        beam_width = max(1, min(20, beam_width))
        
        # Update in database
        await self._session.execute(
            update(ModelConfig)
            .where(ModelConfig.namespace == namespace)
            .values(beam_width=beam_width)
        )
        
        logger.info(f"Updated beam width for namespace '{namespace}': {beam_width}")
        
        return beam_width
    
    async def update_max_tree_depth(
        self,
        namespace: str,
        max_depth: int,
    ) -> int:
        """
        Update max tree depth for a namespace.
        
        Args:
            namespace: Model namespace
            max_depth: New max depth (1-10)
            
        Returns:
            Updated max depth
        """
        # Validate
        max_depth = max(1, min(10, max_depth))
        
        # Update in database
        await self._session.execute(
            update(ModelConfig)
            .where(ModelConfig.namespace == namespace)
            .values(max_tree_depth=max_depth)
        )
        
        logger.info(f"Updated max tree depth for namespace '{namespace}': {max_depth}")
        
        return max_depth
    
    async def reset_to_defaults(self, namespace: str) -> NamespaceConfig:
        """
        Reset namespace configuration to defaults.
        
        Args:
            namespace: Model namespace
            
        Returns:
            Reset NamespaceConfig
        """
        defaults = NamespaceConfig(
            namespace=namespace,
            similarity_weights=SimilarityWeights(),
            beam_width=settings.default_beam_width,
            max_tree_depth=settings.default_max_tree_depth,
        )
        
        await self._session.execute(
            update(ModelConfig)
            .where(ModelConfig.namespace == namespace)
            .values(
                similarity_weights=defaults.similarity_weights.to_dict(),
                beam_width=defaults.beam_width,
                max_tree_depth=defaults.max_tree_depth,
            )
        )
        
        logger.info(f"Reset configuration for namespace '{namespace}' to defaults")
        
        return defaults
