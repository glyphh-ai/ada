"""
Resource Management Domain.

Handles resource quota tracking and enforcement for multi-tenancy.
"""

from domains.resources.quota_service import QuotaService, ResourceQuotas, ResourceUsage
from domains.resources.namespace_config import (
    NamespaceConfig,
    NamespaceConfigService,
    SimilarityWeights,
)
from domains.resources.manager import (
    ResourceManager,
    ResourceUsage as ManagerResourceUsage,
    ResourceQuota,
    QuotaCheckResult,
)

__all__ = [
    "QuotaService",
    "ResourceUsage",
    "ResourceQuotas",
    "NamespaceConfigService",
    "NamespaceConfig",
    "SimilarityWeights",
    "ResourceManager",
    "ResourceQuota",
    "QuotaCheckResult",
]
