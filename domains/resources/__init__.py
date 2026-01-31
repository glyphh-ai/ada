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

__all__ = [
    "QuotaService",
    "ResourceUsage",
    "ResourceQuotas",
    "NamespaceConfigService",
    "NamespaceConfig",
    "SimilarityWeights",
]
