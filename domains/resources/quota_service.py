"""
Quota Service for Resource Management.

Tracks and enforces resource quotas per namespace for multi-tenancy.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Glyph, Edge, ModelConfig
from infrastructure.config import get_settings
from shared.exceptions import NamespaceQuotaExceededException

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ResourceUsage:
    """Current resource usage for a namespace."""
    namespace: str
    glyph_count: int
    edge_count: int
    storage_bytes: int
    memory_mb: float
    
    @property
    def storage_gb(self) -> float:
        """Storage in GB."""
        return self.storage_bytes / (1024 ** 3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "namespace": self.namespace,
            "glyph_count": self.glyph_count,
            "edge_count": self.edge_count,
            "storage_bytes": self.storage_bytes,
            "storage_gb": round(self.storage_gb, 3),
            "memory_mb": round(self.memory_mb, 2),
        }


@dataclass
class ResourceQuotas:
    """Resource quotas for a namespace."""
    max_glyphs: int = 1000000
    max_storage_gb: float = 10.0
    max_memory_mb: float = 1024.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceQuotas":
        """Create from dictionary."""
        return cls(
            max_glyphs=data.get("max_glyphs", 1000000),
            max_storage_gb=data.get("storage_gb", 10.0),
            max_memory_mb=data.get("memory_mb", 1024.0),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_glyphs": self.max_glyphs,
            "storage_gb": self.max_storage_gb,
            "memory_mb": self.max_memory_mb,
        }


class QuotaService:
    """
    Service for tracking and enforcing resource quotas.
    
    Responsibilities:
    - Track resource usage per namespace (glyphs, storage, memory)
    - Enforce quotas before operations
    - Update usage after operations
    - Provide usage reports
    """
    
    # Estimated bytes per glyph (768 floats * 4 bytes + metadata overhead)
    BYTES_PER_GLYPH = 768 * 4 + 500  # ~3.5KB per glyph
    
    # Estimated bytes per edge
    BYTES_PER_EDGE = 200
    
    def __init__(self, session: AsyncSession):
        """
        Initialize QuotaService.
        
        Args:
            session: Async SQLAlchemy session
        """
        self._session = session

    async def get_usage(self, namespace: str) -> ResourceUsage:
        """
        Get current resource usage for a namespace.
        
        Args:
            namespace: Model namespace
            
        Returns:
            ResourceUsage with current counts and estimates
        """
        # Count glyphs
        glyph_result = await self._session.execute(
            select(func.count(Glyph.id)).where(Glyph.namespace == namespace)
        )
        glyph_count = glyph_result.scalar() or 0
        
        # Count edges
        edge_result = await self._session.execute(
            select(func.count(Edge.id)).where(Edge.namespace == namespace)
        )
        edge_count = edge_result.scalar() or 0
        
        # Estimate storage
        storage_bytes = (
            glyph_count * self.BYTES_PER_GLYPH +
            edge_count * self.BYTES_PER_EDGE
        )
        
        # Estimate memory (loaded model + cached data)
        # This is a rough estimate - actual memory depends on model size
        memory_mb = glyph_count * 0.004 + 50  # ~4KB per glyph in memory + base overhead
        
        return ResourceUsage(
            namespace=namespace,
            glyph_count=glyph_count,
            edge_count=edge_count,
            storage_bytes=storage_bytes,
            memory_mb=memory_mb,
        )
    
    async def get_quotas(self, namespace: str) -> ResourceQuotas:
        """
        Get resource quotas for a namespace.
        
        Args:
            namespace: Model namespace
            
        Returns:
            ResourceQuotas for the namespace
        """
        result = await self._session.execute(
            select(ModelConfig.resource_quotas).where(
                ModelConfig.namespace == namespace
            )
        )
        quotas_dict = result.scalar_one_or_none()
        
        if quotas_dict:
            return ResourceQuotas.from_dict(quotas_dict)
        
        # Return defaults
        return ResourceQuotas(
            max_glyphs=settings.default_namespace_memory_mb * 250,  # ~250 glyphs per MB
            max_storage_gb=settings.default_namespace_storage_gb,
            max_memory_mb=settings.default_namespace_memory_mb,
        )
    
    async def check_quota(
        self,
        namespace: str,
        operation: str,
        additional_glyphs: int = 0,
        additional_edges: int = 0,
    ) -> bool:
        """
        Check if an operation would exceed quotas.
        
        Args:
            namespace: Model namespace
            operation: Operation name (for error messages)
            additional_glyphs: Number of glyphs to add
            additional_edges: Number of edges to add
            
        Returns:
            True if within quota
            
        Raises:
            NamespaceQuotaExceededException: If quota would be exceeded
        """
        usage = await self.get_usage(namespace)
        quotas = await self.get_quotas(namespace)
        
        # Check glyph count
        new_glyph_count = usage.glyph_count + additional_glyphs
        if new_glyph_count > quotas.max_glyphs:
            raise NamespaceQuotaExceededException(
                namespace=namespace,
                resource="glyphs",
                limit=quotas.max_glyphs,
                current=usage.glyph_count,
            )
        
        # Check storage
        new_storage_bytes = (
            usage.storage_bytes +
            additional_glyphs * self.BYTES_PER_GLYPH +
            additional_edges * self.BYTES_PER_EDGE
        )
        new_storage_gb = new_storage_bytes / (1024 ** 3)
        if new_storage_gb > quotas.max_storage_gb:
            raise NamespaceQuotaExceededException(
                namespace=namespace,
                resource="storage",
                limit=f"{quotas.max_storage_gb} GB",
                current=f"{usage.storage_gb:.3f} GB",
            )
        
        # Check memory (estimate)
        new_memory_mb = usage.memory_mb + additional_glyphs * 0.004
        if new_memory_mb > quotas.max_memory_mb:
            raise NamespaceQuotaExceededException(
                namespace=namespace,
                resource="memory",
                limit=f"{quotas.max_memory_mb} MB",
                current=f"{usage.memory_mb:.2f} MB",
            )
        
        return True
    
    async def update_quotas(
        self,
        namespace: str,
        max_glyphs: Optional[int] = None,
        max_storage_gb: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
    ) -> ResourceQuotas:
        """
        Update resource quotas for a namespace.
        
        Args:
            namespace: Model namespace
            max_glyphs: New max glyphs (optional)
            max_storage_gb: New max storage in GB (optional)
            max_memory_mb: New max memory in MB (optional)
            
        Returns:
            Updated ResourceQuotas
        """
        # Get current quotas
        current = await self.get_quotas(namespace)
        
        # Build new quotas
        new_quotas = ResourceQuotas(
            max_glyphs=max_glyphs if max_glyphs is not None else current.max_glyphs,
            max_storage_gb=max_storage_gb if max_storage_gb is not None else current.max_storage_gb,
            max_memory_mb=max_memory_mb if max_memory_mb is not None else current.max_memory_mb,
        )
        
        # Update in database
        await self._session.execute(
            update(ModelConfig)
            .where(ModelConfig.namespace == namespace)
            .values(resource_quotas=new_quotas.to_dict())
        )
        
        logger.info(f"Updated quotas for namespace '{namespace}': {new_quotas.to_dict()}")
        
        return new_quotas
    
    async def get_usage_report(self, namespace: str) -> Dict[str, Any]:
        """
        Get a detailed usage report for a namespace.
        
        Args:
            namespace: Model namespace
            
        Returns:
            Dict with usage, quotas, and percentages
        """
        usage = await self.get_usage(namespace)
        quotas = await self.get_quotas(namespace)
        
        return {
            "namespace": namespace,
            "usage": usage.to_dict(),
            "quotas": quotas.to_dict(),
            "utilization": {
                "glyphs_percent": round(usage.glyph_count / quotas.max_glyphs * 100, 2),
                "storage_percent": round(usage.storage_gb / quotas.max_storage_gb * 100, 2),
                "memory_percent": round(usage.memory_mb / quotas.max_memory_mb * 100, 2),
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
