"""
Model Configuration Service.

Manages per org/model configuration for similarity weights, beam search,
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
class ModelScopeConfig:
    """Complete configuration for an org/model."""
    org_id: str
    model_id: str
    similarity_weights: SimilarityWeights = field(default_factory=SimilarityWeights)
    beam_width: int = 5
    max_tree_depth: int = 3
    
    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "ModelScopeConfig":
        weights = SimilarityWeights.from_dict(config.similarity_weights or {})
        return cls(
            org_id=config.org_id,
            model_id=config.model_id,
            similarity_weights=weights,
            beam_width=config.beam_width or settings.default_beam_width,
            max_tree_depth=config.max_tree_depth or settings.default_max_tree_depth,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "org_id": self.org_id,
            "model_id": self.model_id,
            "similarity_weights": self.similarity_weights.to_dict(),
            "beam_width": self.beam_width,
            "max_tree_depth": self.max_tree_depth,
        }


class ModelConfigService:
    """Service for managing per org/model configuration."""
    
    def __init__(self, session: AsyncSession):
        self._session = session
    
    async def get_config(self, org_id: str, model_id: str) -> ModelScopeConfig:
        """Get configuration for an org/model."""
        result = await self._session.execute(
            select(ModelConfig).where(
                ModelConfig.org_id == org_id, ModelConfig.model_id == model_id
            )
        )
        config = result.scalar_one_or_none()
        
        if config:
            return ModelScopeConfig.from_model_config(config)
        
        return ModelScopeConfig(
            org_id=org_id,
            model_id=model_id,
            similarity_weights=SimilarityWeights(),
            beam_width=settings.default_beam_width,
            max_tree_depth=settings.default_max_tree_depth,
        )
    
    async def update_similarity_weights(
        self, org_id: str, model_id: str, weights: Dict[str, float],
    ) -> SimilarityWeights:
        """Update similarity weights for an org/model."""
        current = await self.get_config(org_id, model_id)
        current_dict = current.similarity_weights.to_dict()
        current_dict.update(weights)
        new_weights = SimilarityWeights.from_dict(current_dict)
        
        await self._session.execute(
            update(ModelConfig)
            .where(ModelConfig.org_id == org_id, ModelConfig.model_id == model_id)
            .values(similarity_weights=new_weights.to_dict())
        )
        
        logger.info(f"Updated similarity weights for org={org_id}, model={model_id}")
        return new_weights
    
    async def update_beam_width(self, org_id: str, model_id: str, beam_width: int) -> int:
        beam_width = max(1, min(20, beam_width))
        await self._session.execute(
            update(ModelConfig)
            .where(ModelConfig.org_id == org_id, ModelConfig.model_id == model_id)
            .values(beam_width=beam_width)
        )
        logger.info(f"Updated beam width for org={org_id}, model={model_id}: {beam_width}")
        return beam_width
    
    async def update_max_tree_depth(self, org_id: str, model_id: str, max_depth: int) -> int:
        max_depth = max(1, min(10, max_depth))
        await self._session.execute(
            update(ModelConfig)
            .where(ModelConfig.org_id == org_id, ModelConfig.model_id == model_id)
            .values(max_tree_depth=max_depth)
        )
        logger.info(f"Updated max tree depth for org={org_id}, model={model_id}: {max_depth}")
        return max_depth
    
    async def reset_to_defaults(self, org_id: str, model_id: str) -> ModelScopeConfig:
        defaults = ModelScopeConfig(
            org_id=org_id,
            model_id=model_id,
            similarity_weights=SimilarityWeights(),
            beam_width=settings.default_beam_width,
            max_tree_depth=settings.default_max_tree_depth,
        )
        
        await self._session.execute(
            update(ModelConfig)
            .where(ModelConfig.org_id == org_id, ModelConfig.model_id == model_id)
            .values(
                similarity_weights=defaults.similarity_weights.to_dict(),
                beam_width=defaults.beam_width,
                max_tree_depth=defaults.max_tree_depth,
            )
        )
        
        logger.info(f"Reset configuration for org={org_id}, model={model_id} to defaults")
        return defaults
