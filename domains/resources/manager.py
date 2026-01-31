"""
Resource Manager for Glyphh Runtime.

Tracks and enforces resource quotas per namespace for multi-tenancy.
Monitors memory, storage, and CPU usage.
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Edge, Glyph, ModelConfig
from infrastructure.config import get_settings
from shared.exceptions import NamespaceNotFoundException, NamespaceQuotaExceededException

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ResourceUsage:
    """Current resource usage for a namespace."""
    namespace: str
    memory_mb: float = 0.0
    storage_mb: float = 0.0
    glyph_count: int = 0
    edge_count: int = 0
    cpu_percent: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourceQuota:
    """Resource quotas for a namespace."""
    namespace: str
    max_memory_mb: float = 1024.0
    max_storage_gb: float = 10.0
    max_glyphs: int = 1000000
    max_edges: int = 5000000
    max_requests_per_minute: int = 60


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""
    allowed: bool
    resource: Optional[str] = None
    limit: Optional[float] = None
    current: Optional[float] = None
    message: Optional[str] = None


class ResourceManager:
    """
    Manages resource quotas and usage tracking per namespace.
    
    Responsibilities:
    - Track memory, storage, and CPU usage per namespace
    - Enforce quotas before operations
    - Provide usage statistics
    - Support per-namespace configuration
    """
    
    # Estimated bytes per glyph (embedding + metadata)
    BYTES_PER_GLYPH = 768 * 4 + 1024  # 768 floats + ~1KB metadata
    
    # Estimated bytes per edge
    BYTES_PER_EDGE = 256
    
    def __init__(self, db_session_factory):
        """
        Initialize ResourceManager.
        
        Args:
            db_session_factory: Async session factory for database access
        """
        self._db_session_factory = db_session_factory
        self._usage_cache: Dict[str, ResourceUsage] = {}
        self._quota_cache: Dict[str, ResourceQuota] = {}
        self._cache_ttl_seconds = 60
        self._last_cache_update: Dict[str, datetime] = {}
    
    async def get_usage(self, namespace: str, force_refresh: bool = False) -> ResourceUsage:
        """
        Get current resource usage for a namespace.
        
        Args:
            namespace: Namespace to get usage for
            force_refresh: Force refresh from database
            
        Returns:
            ResourceUsage for the namespace
        """
        # Check cache
        if not force_refresh and namespace in self._usage_cache:
            last_update = self._last_cache_update.get(namespace)
            if last_update:
                age = (datetime.utcnow() - last_update).total_seconds()
                if age < self._cache_ttl_seconds:
                    return self._usage_cache[namespace]
        
        # Fetch from database
        async with self._db_session_factory() as session:
            # Count glyphs
            glyph_result = await session.execute(
                select(func.count(Glyph.id)).where(Glyph.namespace == namespace)
            )
            glyph_count = glyph_result.scalar() or 0
            
            # Count edges
            edge_result = await session.execute(
                select(func.count(Edge.id)).where(Edge.namespace == namespace)
            )
            edge_count = edge_result.scalar() or 0
            
            # Estimate storage
            storage_bytes = (
                glyph_count * self.BYTES_PER_GLYPH +
                edge_count * self.BYTES_PER_EDGE
            )
            storage_mb = storage_bytes / (1024 * 1024)
            
            # Estimate memory (in-memory structures)
            # This is a rough estimate - actual memory depends on caching
            memory_mb = storage_mb * 0.5  # Assume 50% of storage is in memory
            
            usage = ResourceUsage(
                namespace=namespace,
                memory_mb=memory_mb,
                storage_mb=storage_mb,
                glyph_count=glyph_count,
                edge_count=edge_count,
                cpu_percent=0.0,  # CPU tracking requires process monitoring
                last_updated=datetime.utcnow(),
            )
            
            # Update cache
            self._usage_cache[namespace] = usage
            self._last_cache_update[namespace] = datetime.utcnow()
            
            # Update database
            await self._update_usage_in_db(session, namespace, usage)
            await session.commit()
            
            return usage
    
    async def _update_usage_in_db(
        self,
        session: AsyncSession,
        namespace: str,
        usage: ResourceUsage,
    ) -> None:
        """Update resource usage in database."""
        await session.execute(
            update(ModelConfig)
            .where(ModelConfig.namespace == namespace)
            .values(
                resource_usage={
                    "memory_mb": usage.memory_mb,
                    "storage_mb": usage.storage_mb,
                    "glyph_count": usage.glyph_count,
                    "edge_count": usage.edge_count,
                    "last_updated": usage.last_updated.isoformat(),
                },
                updated_at=datetime.utcnow(),
            )
        )
    
    async def get_quota(self, namespace: str) -> ResourceQuota:
        """
        Get resource quotas for a namespace.
        
        Args:
            namespace: Namespace to get quotas for
            
        Returns:
            ResourceQuota for the namespace
        """
        # Check cache
        if namespace in self._quota_cache:
            return self._quota_cache[namespace]
        
        # Fetch from database
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(ModelConfig.namespace == namespace)
            )
            config = result.scalar_one_or_none()
            
            if config is None:
                raise NamespaceNotFoundException(namespace)
            
            quotas = config.resource_quotas or {}
            
            quota = ResourceQuota(
                namespace=namespace,
                max_memory_mb=quotas.get("memory_mb", settings.default_namespace_memory_mb),
                max_storage_gb=quotas.get("storage_gb", settings.default_namespace_storage_gb),
                max_glyphs=quotas.get("max_glyphs", 1000000),
                max_edges=quotas.get("max_edges", 5000000),
                max_requests_per_minute=quotas.get("max_requests_per_minute", settings.rate_limit_per_minute),
            )
            
            # Update cache
            self._quota_cache[namespace] = quota
            
            return quota
    
    async def check_quota(
        self,
        namespace: str,
        additional_glyphs: int = 0,
        additional_edges: int = 0,
        additional_storage_mb: float = 0,
    ) -> QuotaCheckResult:
        """
        Check if an operation would exceed quotas.
        
        Args:
            namespace: Namespace to check
            additional_glyphs: Number of glyphs to add
            additional_edges: Number of edges to add
            additional_storage_mb: Additional storage in MB
            
        Returns:
            QuotaCheckResult indicating if operation is allowed
        """
        usage = await self.get_usage(namespace)
        quota = await self.get_quota(namespace)
        
        # Check glyph count
        new_glyph_count = usage.glyph_count + additional_glyphs
        if new_glyph_count > quota.max_glyphs:
            return QuotaCheckResult(
                allowed=False,
                resource="max_glyphs",
                limit=quota.max_glyphs,
                current=new_glyph_count,
                message=f"Glyph limit exceeded: {new_glyph_count} > {quota.max_glyphs}",
            )
        
        # Check edge count
        new_edge_count = usage.edge_count + additional_edges
        if new_edge_count > quota.max_edges:
            return QuotaCheckResult(
                allowed=False,
                resource="max_edges",
                limit=quota.max_edges,
                current=new_edge_count,
                message=f"Edge limit exceeded: {new_edge_count} > {quota.max_edges}",
            )
        
        # Check storage
        estimated_additional_storage = (
            additional_glyphs * self.BYTES_PER_GLYPH +
            additional_edges * self.BYTES_PER_EDGE
        ) / (1024 * 1024)
        new_storage_mb = usage.storage_mb + additional_storage_mb + estimated_additional_storage
        max_storage_mb = quota.max_storage_gb * 1024
        
        if new_storage_mb > max_storage_mb:
            return QuotaCheckResult(
                allowed=False,
                resource="storage_gb",
                limit=quota.max_storage_gb,
                current=new_storage_mb / 1024,
                message=f"Storage limit exceeded: {new_storage_mb/1024:.2f}GB > {quota.max_storage_gb}GB",
            )
        
        # Check memory (rough estimate)
        new_memory_mb = usage.memory_mb + (estimated_additional_storage * 0.5)
        if new_memory_mb > quota.max_memory_mb:
            return QuotaCheckResult(
                allowed=False,
                resource="memory_mb",
                limit=quota.max_memory_mb,
                current=new_memory_mb,
                message=f"Memory limit exceeded: {new_memory_mb:.2f}MB > {quota.max_memory_mb}MB",
            )
        
        return QuotaCheckResult(allowed=True)
    
    async def enforce_quota(
        self,
        namespace: str,
        additional_glyphs: int = 0,
        additional_edges: int = 0,
        additional_storage_mb: float = 0,
    ) -> None:
        """
        Enforce quotas - raises exception if exceeded.
        
        Args:
            namespace: Namespace to check
            additional_glyphs: Number of glyphs to add
            additional_edges: Number of edges to add
            additional_storage_mb: Additional storage in MB
            
        Raises:
            NamespaceQuotaExceededException: If quota would be exceeded
        """
        result = await self.check_quota(
            namespace,
            additional_glyphs=additional_glyphs,
            additional_edges=additional_edges,
            additional_storage_mb=additional_storage_mb,
        )
        
        if not result.allowed:
            raise NamespaceQuotaExceededException(
                namespace=namespace,
                resource=result.resource,
                limit=result.limit,
                current=result.current,
            )
    
    async def update_quota(
        self,
        namespace: str,
        max_memory_mb: Optional[float] = None,
        max_storage_gb: Optional[float] = None,
        max_glyphs: Optional[int] = None,
        max_edges: Optional[int] = None,
        max_requests_per_minute: Optional[int] = None,
    ) -> ResourceQuota:
        """
        Update resource quotas for a namespace.
        
        Args:
            namespace: Namespace to update
            max_memory_mb: New memory limit in MB
            max_storage_gb: New storage limit in GB
            max_glyphs: New glyph limit
            max_edges: New edge limit
            max_requests_per_minute: New rate limit
            
        Returns:
            Updated ResourceQuota
        """
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(ModelConfig.namespace == namespace)
            )
            config = result.scalar_one_or_none()
            
            if config is None:
                raise NamespaceNotFoundException(namespace)
            
            quotas = config.resource_quotas or {}
            
            if max_memory_mb is not None:
                quotas["memory_mb"] = max_memory_mb
            if max_storage_gb is not None:
                quotas["storage_gb"] = max_storage_gb
            if max_glyphs is not None:
                quotas["max_glyphs"] = max_glyphs
            if max_edges is not None:
                quotas["max_edges"] = max_edges
            if max_requests_per_minute is not None:
                quotas["max_requests_per_minute"] = max_requests_per_minute
            
            await session.execute(
                update(ModelConfig)
                .where(ModelConfig.namespace == namespace)
                .values(
                    resource_quotas=quotas,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()
            
            # Invalidate cache
            if namespace in self._quota_cache:
                del self._quota_cache[namespace]
            
            return await self.get_quota(namespace)
    
    async def get_all_usage(self) -> List[ResourceUsage]:
        """
        Get resource usage for all namespaces.
        
        Returns:
            List of ResourceUsage for all namespaces
        """
        async with self._db_session_factory() as session:
            result = await session.execute(select(ModelConfig.namespace))
            namespaces = [row[0] for row in result.fetchall()]
        
        usage_list = []
        for namespace in namespaces:
            try:
                usage = await self.get_usage(namespace)
                usage_list.append(usage)
            except Exception as e:
                logger.warning(f"Failed to get usage for {namespace}: {e}")
        
        return usage_list
    
    async def get_system_usage(self) -> Dict[str, Any]:
        """
        Get overall system resource usage.
        
        Returns:
            Dict with system-wide usage statistics
        """
        import psutil
        
        # Get process info
        process = psutil.Process(os.getpid())
        
        # Memory info
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        # CPU info
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # Disk info (for database directory)
        disk_usage = psutil.disk_usage("/")
        
        # Aggregate namespace usage
        all_usage = await self.get_all_usage()
        total_glyphs = sum(u.glyph_count for u in all_usage)
        total_edges = sum(u.edge_count for u in all_usage)
        total_storage_mb = sum(u.storage_mb for u in all_usage)
        
        return {
            "process": {
                "memory_mb": memory_info.rss / (1024 * 1024),
                "memory_percent": process.memory_percent(),
                "cpu_percent": cpu_percent,
                "threads": process.num_threads(),
            },
            "system": {
                "memory_total_gb": system_memory.total / (1024 ** 3),
                "memory_available_gb": system_memory.available / (1024 ** 3),
                "memory_percent": system_memory.percent,
                "disk_total_gb": disk_usage.total / (1024 ** 3),
                "disk_free_gb": disk_usage.free / (1024 ** 3),
                "disk_percent": disk_usage.percent,
            },
            "namespaces": {
                "count": len(all_usage),
                "total_glyphs": total_glyphs,
                "total_edges": total_edges,
                "total_storage_mb": total_storage_mb,
            },
        }
    
    def invalidate_cache(self, namespace: Optional[str] = None) -> None:
        """
        Invalidate cached usage/quota data.
        
        Args:
            namespace: Specific namespace to invalidate, or None for all
        """
        if namespace:
            self._usage_cache.pop(namespace, None)
            self._quota_cache.pop(namespace, None)
            self._last_cache_update.pop(namespace, None)
        else:
            self._usage_cache.clear()
            self._quota_cache.clear()
            self._last_cache_update.clear()
