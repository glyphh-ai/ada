"""
Resource Manager for Glyphh Runtime.

Tracks and enforces resource quotas per org/model for multi-tenancy.
Monitors memory, storage, and CPU usage.
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domains.models.db_models import Edge, Glyph, ModelConfig
from infrastructure.config import get_settings
from shared.exceptions import ModelNotFoundException, QuotaExceededException

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ResourceUsage:
    """Current resource usage for an org/model."""
    org_id: str
    model_id: str
    memory_mb: float = 0.0
    storage_mb: float = 0.0
    glyph_count: int = 0
    edge_count: int = 0
    cpu_percent: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourceQuota:
    """Resource quotas for an org/model."""
    org_id: str
    model_id: str
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
    Manages resource quotas and usage tracking per org/model.
    """
    
    BYTES_PER_GLYPH = 768 * 4 + 1024
    BYTES_PER_EDGE = 256
    
    def __init__(self, db_session_factory):
        self._db_session_factory = db_session_factory
        self._usage_cache: Dict[Tuple[str, str], ResourceUsage] = {}
        self._quota_cache: Dict[Tuple[str, str], ResourceQuota] = {}
        self._cache_ttl_seconds = 60
        self._last_cache_update: Dict[Tuple[str, str], datetime] = {}
    
    def _key(self, org_id: str, model_id: str) -> Tuple[str, str]:
        return (org_id, model_id)
    
    async def get_usage(self, org_id: str, model_id: str, force_refresh: bool = False) -> ResourceUsage:
        """Get current resource usage for an org/model."""
        key = self._key(org_id, model_id)
        
        if not force_refresh and key in self._usage_cache:
            last_update = self._last_cache_update.get(key)
            if last_update:
                age = (datetime.utcnow() - last_update).total_seconds()
                if age < self._cache_ttl_seconds:
                    return self._usage_cache[key]
        
        async with self._db_session_factory() as session:
            glyph_result = await session.execute(
                select(func.count(Glyph.id)).where(
                    Glyph.org_id == org_id, Glyph.model_id == model_id
                )
            )
            glyph_count = glyph_result.scalar() or 0
            
            edge_result = await session.execute(
                select(func.count(Edge.id)).where(
                    Edge.org_id == org_id, Edge.model_id == model_id
                )
            )
            edge_count = edge_result.scalar() or 0
            
            storage_bytes = glyph_count * self.BYTES_PER_GLYPH + edge_count * self.BYTES_PER_EDGE
            storage_mb = storage_bytes / (1024 * 1024)
            memory_mb = storage_mb * 0.5
            
            usage = ResourceUsage(
                org_id=org_id,
                model_id=model_id,
                memory_mb=memory_mb,
                storage_mb=storage_mb,
                glyph_count=glyph_count,
                edge_count=edge_count,
                cpu_percent=0.0,
                last_updated=datetime.utcnow(),
            )
            
            self._usage_cache[key] = usage
            self._last_cache_update[key] = datetime.utcnow()
            
            await self._update_usage_in_db(session, org_id, model_id, usage)
            await session.commit()
            
            return usage
    
    async def _update_usage_in_db(
        self, session: AsyncSession, org_id: str, model_id: str, usage: ResourceUsage,
    ) -> None:
        await session.execute(
            update(ModelConfig)
            .where(ModelConfig.org_id == org_id, ModelConfig.model_id == model_id)
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
    
    async def get_quota(self, org_id: str, model_id: str) -> ResourceQuota:
        """Get resource quotas for an org/model."""
        key = self._key(org_id, model_id)
        
        if key in self._quota_cache:
            return self._quota_cache[key]
        
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == org_id, ModelConfig.model_id == model_id
                )
            )
            config = result.scalar_one_or_none()
            
            if config is None:
                raise ModelNotFoundException(org_id, model_id)
            
            quotas = config.resource_quotas or {}
            
            quota = ResourceQuota(
                org_id=org_id,
                model_id=model_id,
                max_memory_mb=quotas.get("memory_mb", settings.default_model_memory_mb),
                max_storage_gb=quotas.get("storage_gb", settings.default_model_storage_gb),
                max_glyphs=quotas.get("max_glyphs", 1000000),
                max_edges=quotas.get("max_edges", 5000000),
                max_requests_per_minute=quotas.get("max_requests_per_minute", settings.rate_limit_per_minute),
            )
            
            self._quota_cache[key] = quota
            return quota
    
    async def check_quota(
        self,
        org_id: str,
        model_id: str,
        additional_glyphs: int = 0,
        additional_edges: int = 0,
        additional_storage_mb: float = 0,
    ) -> QuotaCheckResult:
        """Check if an operation would exceed quotas."""
        usage = await self.get_usage(org_id, model_id)
        quota = await self.get_quota(org_id, model_id)
        
        new_glyph_count = usage.glyph_count + additional_glyphs
        if new_glyph_count > quota.max_glyphs:
            return QuotaCheckResult(
                allowed=False, resource="max_glyphs",
                limit=quota.max_glyphs, current=new_glyph_count,
                message=f"Glyph limit exceeded: {new_glyph_count} > {quota.max_glyphs}",
            )
        
        new_edge_count = usage.edge_count + additional_edges
        if new_edge_count > quota.max_edges:
            return QuotaCheckResult(
                allowed=False, resource="max_edges",
                limit=quota.max_edges, current=new_edge_count,
                message=f"Edge limit exceeded: {new_edge_count} > {quota.max_edges}",
            )
        
        estimated_additional_storage = (
            additional_glyphs * self.BYTES_PER_GLYPH + additional_edges * self.BYTES_PER_EDGE
        ) / (1024 * 1024)
        new_storage_mb = usage.storage_mb + additional_storage_mb + estimated_additional_storage
        max_storage_mb = quota.max_storage_gb * 1024
        
        if new_storage_mb > max_storage_mb:
            return QuotaCheckResult(
                allowed=False, resource="storage_gb",
                limit=quota.max_storage_gb, current=new_storage_mb / 1024,
                message=f"Storage limit exceeded: {new_storage_mb/1024:.2f}GB > {quota.max_storage_gb}GB",
            )
        
        new_memory_mb = usage.memory_mb + (estimated_additional_storage * 0.5)
        if new_memory_mb > quota.max_memory_mb:
            return QuotaCheckResult(
                allowed=False, resource="memory_mb",
                limit=quota.max_memory_mb, current=new_memory_mb,
                message=f"Memory limit exceeded: {new_memory_mb:.2f}MB > {quota.max_memory_mb}MB",
            )
        
        return QuotaCheckResult(allowed=True)
    
    async def enforce_quota(
        self,
        org_id: str,
        model_id: str,
        additional_glyphs: int = 0,
        additional_edges: int = 0,
        additional_storage_mb: float = 0,
    ) -> None:
        """Enforce quotas — raises exception if exceeded."""
        result = await self.check_quota(
            org_id, model_id,
            additional_glyphs=additional_glyphs,
            additional_edges=additional_edges,
            additional_storage_mb=additional_storage_mb,
        )
        
        if not result.allowed:
            raise QuotaExceededException(
                org_id=org_id,
                model_id=model_id,
                resource=result.resource,
                limit=result.limit,
                current=result.current,
            )
    
    async def update_quota(
        self,
        org_id: str,
        model_id: str,
        max_memory_mb: Optional[float] = None,
        max_storage_gb: Optional[float] = None,
        max_glyphs: Optional[int] = None,
        max_edges: Optional[int] = None,
        max_requests_per_minute: Optional[int] = None,
    ) -> ResourceQuota:
        """Update resource quotas for an org/model."""
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig).where(
                    ModelConfig.org_id == org_id, ModelConfig.model_id == model_id
                )
            )
            config = result.scalar_one_or_none()
            
            if config is None:
                raise ModelNotFoundException(org_id, model_id)
            
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
                .where(ModelConfig.org_id == org_id, ModelConfig.model_id == model_id)
                .values(resource_quotas=quotas, updated_at=datetime.utcnow())
            )
            await session.commit()
            
            key = self._key(org_id, model_id)
            self._quota_cache.pop(key, None)
            
            return await self.get_quota(org_id, model_id)
    
    async def get_all_usage(self) -> List[ResourceUsage]:
        """Get resource usage for all org/models."""
        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelConfig.org_id, ModelConfig.model_id)
            )
            pairs = [(row[0], row[1]) for row in result.fetchall()]
        
        usage_list = []
        for org_id, model_id in pairs:
            try:
                usage = await self.get_usage(org_id, model_id)
                usage_list.append(usage)
            except Exception as e:
                logger.warning(f"Failed to get usage for org={org_id}, model={model_id}: {e}")
        
        return usage_list
    
    async def get_system_usage(self) -> Dict[str, Any]:
        """Get overall system resource usage."""
        import psutil
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        cpu_percent = process.cpu_percent(interval=0.1)
        disk_usage = psutil.disk_usage("/")
        
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
            "models": {
                "count": len(all_usage),
                "total_glyphs": total_glyphs,
                "total_edges": total_edges,
                "total_storage_mb": total_storage_mb,
            },
        }
    
    def invalidate_cache(self, org_id: Optional[str] = None, model_id: Optional[str] = None) -> None:
        """Invalidate cached usage/quota data."""
        if org_id and model_id:
            key = self._key(org_id, model_id)
            self._usage_cache.pop(key, None)
            self._quota_cache.pop(key, None)
            self._last_cache_update.pop(key, None)
        else:
            self._usage_cache.clear()
            self._quota_cache.clear()
            self._last_cache_update.clear()
