"""
Quota Service for Resource Management.

Tracks and enforces resource quotas per org/model for multi-tenancy.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Glyph, Edge, ModelConfig
from infrastructure.config import get_settings
from shared.exceptions import QuotaExceededException

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ResourceUsage:
    """Current resource usage for an org/model."""
    org_id: str
    model_id: str
    glyph_count: int
    edge_count: int
    storage_bytes: int
    memory_mb: float
    
    @property
    def storage_gb(self) -> float:
        return self.storage_bytes / (1024 ** 3)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "org_id": self.org_id,
            "model_id": self.model_id,
            "glyph_count": self.glyph_count,
            "edge_count": self.edge_count,
            "storage_bytes": self.storage_bytes,
            "storage_gb": round(self.storage_gb, 3),
            "memory_mb": round(self.memory_mb, 2),
        }


@dataclass
class ResourceQuotas:
    """Resource quotas for an org/model."""
    max_glyphs: int = 1000000
    max_storage_gb: float = 10.0
    max_memory_mb: float = 1024.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceQuotas":
        return cls(
            max_glyphs=data.get("max_glyphs", 1000000),
            max_storage_gb=data.get("storage_gb", 10.0),
            max_memory_mb=data.get("memory_mb", 1024.0),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_glyphs": self.max_glyphs,
            "storage_gb": self.max_storage_gb,
            "memory_mb": self.max_memory_mb,
        }


class QuotaService:
    """Service for tracking and enforcing resource quotas per org/model."""
    
    BYTES_PER_GLYPH = 768 * 4 + 500
    BYTES_PER_EDGE = 200
    
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_usage(self, org_id: str, model_id: str) -> ResourceUsage:
        """Get current resource usage for an org/model."""
        glyph_result = await self._session.execute(
            select(func.count(Glyph.id)).where(
                Glyph.org_id == org_id, Glyph.model_id == model_id
            )
        )
        glyph_count = glyph_result.scalar() or 0
        
        edge_result = await self._session.execute(
            select(func.count(Edge.id)).where(
                Edge.org_id == org_id, Edge.model_id == model_id
            )
        )
        edge_count = edge_result.scalar() or 0
        
        storage_bytes = glyph_count * self.BYTES_PER_GLYPH + edge_count * self.BYTES_PER_EDGE
        memory_mb = glyph_count * 0.004 + 50
        
        return ResourceUsage(
            org_id=org_id,
            model_id=model_id,
            glyph_count=glyph_count,
            edge_count=edge_count,
            storage_bytes=storage_bytes,
            memory_mb=memory_mb,
        )
    
    async def get_quotas(self, org_id: str, model_id: str) -> ResourceQuotas:
        """Get resource quotas for an org/model."""
        result = await self._session.execute(
            select(ModelConfig.resource_quotas).where(
                ModelConfig.org_id == org_id, ModelConfig.model_id == model_id
            )
        )
        quotas_dict = result.scalar_one_or_none()
        
        if quotas_dict:
            return ResourceQuotas.from_dict(quotas_dict)
        
        return ResourceQuotas(
            max_glyphs=settings.default_model_memory_mb * 250,
            max_storage_gb=settings.default_model_storage_gb,
            max_memory_mb=settings.default_model_memory_mb,
        )
    
    async def check_quota(
        self,
        org_id: str,
        model_id: str,
        operation: str,
        additional_glyphs: int = 0,
        additional_edges: int = 0,
    ) -> bool:
        """Check if an operation would exceed quotas. Raises if exceeded."""
        usage = await self.get_usage(org_id, model_id)
        quotas = await self.get_quotas(org_id, model_id)
        
        new_glyph_count = usage.glyph_count + additional_glyphs
        if new_glyph_count > quotas.max_glyphs:
            raise QuotaExceededException(
                org_id=org_id, model_id=model_id,
                resource="glyphs", limit=quotas.max_glyphs, current=usage.glyph_count,
            )
        
        new_storage_bytes = (
            usage.storage_bytes +
            additional_glyphs * self.BYTES_PER_GLYPH +
            additional_edges * self.BYTES_PER_EDGE
        )
        new_storage_gb = new_storage_bytes / (1024 ** 3)
        if new_storage_gb > quotas.max_storage_gb:
            raise QuotaExceededException(
                org_id=org_id, model_id=model_id,
                resource="storage",
                limit=f"{quotas.max_storage_gb} GB",
                current=f"{usage.storage_gb:.3f} GB",
            )
        
        new_memory_mb = usage.memory_mb + additional_glyphs * 0.004
        if new_memory_mb > quotas.max_memory_mb:
            raise QuotaExceededException(
                org_id=org_id, model_id=model_id,
                resource="memory",
                limit=f"{quotas.max_memory_mb} MB",
                current=f"{usage.memory_mb:.2f} MB",
            )
        
        return True
    
    async def update_quotas(
        self,
        org_id: str,
        model_id: str,
        max_glyphs: Optional[int] = None,
        max_storage_gb: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
    ) -> ResourceQuotas:
        """Update resource quotas for an org/model."""
        current = await self.get_quotas(org_id, model_id)
        
        new_quotas = ResourceQuotas(
            max_glyphs=max_glyphs if max_glyphs is not None else current.max_glyphs,
            max_storage_gb=max_storage_gb if max_storage_gb is not None else current.max_storage_gb,
            max_memory_mb=max_memory_mb if max_memory_mb is not None else current.max_memory_mb,
        )
        
        await self._session.execute(
            update(ModelConfig)
            .where(ModelConfig.org_id == org_id, ModelConfig.model_id == model_id)
            .values(resource_quotas=new_quotas.to_dict())
        )
        
        logger.info(f"Updated quotas for org={org_id}, model={model_id}: {new_quotas.to_dict()}")
        return new_quotas
    
    async def get_usage_report(self, org_id: str, model_id: str) -> Dict[str, Any]:
        """Get a detailed usage report for an org/model."""
        usage = await self.get_usage(org_id, model_id)
        quotas = await self.get_quotas(org_id, model_id)
        
        return {
            "org_id": org_id,
            "model_id": model_id,
            "usage": usage.to_dict(),
            "quotas": quotas.to_dict(),
            "utilization": {
                "glyphs_percent": round(usage.glyph_count / quotas.max_glyphs * 100, 2),
                "storage_percent": round(usage.storage_gb / quotas.max_storage_gb * 100, 2),
                "memory_percent": round(usage.memory_mb / quotas.max_memory_mb * 100, 2),
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
