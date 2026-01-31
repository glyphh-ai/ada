"""
Unit tests for ResourceManager.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from domains.resources.manager import ResourceManager, ResourceUsage, ResourceQuota, QuotaCheckResult
from shared.exceptions import NamespaceNotFoundException, NamespaceQuotaExceededException


class TestResourceManager:
    """Tests for ResourceManager quota tracking and enforcement."""
    
    @pytest.fixture
    def mock_session_factory(self):
        """Create a mock session factory."""
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        
        factory = MagicMock(return_value=session)
        return factory
    
    @pytest.fixture
    def resource_manager(self, mock_session_factory):
        """Create ResourceManager instance for testing."""
        return ResourceManager(mock_session_factory)
    
    def test_quota_check_result_allowed(self):
        """Test QuotaCheckResult when allowed."""
        result = QuotaCheckResult(allowed=True)
        
        assert result.allowed is True
        assert result.resource is None
        assert result.message is None
    
    def test_quota_check_result_denied(self):
        """Test QuotaCheckResult when denied."""
        result = QuotaCheckResult(
            allowed=False,
            resource="max_glyphs",
            limit=1000,
            current=1500,
            message="Glyph limit exceeded",
        )
        
        assert result.allowed is False
        assert result.resource == "max_glyphs"
        assert result.limit == 1000
        assert result.current == 1500
    
    def test_resource_usage_defaults(self):
        """Test ResourceUsage default values."""
        usage = ResourceUsage(namespace="test")
        
        assert usage.namespace == "test"
        assert usage.memory_mb == 0.0
        assert usage.storage_mb == 0.0
        assert usage.glyph_count == 0
        assert usage.edge_count == 0
    
    def test_resource_quota_defaults(self):
        """Test ResourceQuota default values."""
        quota = ResourceQuota(namespace="test")
        
        assert quota.namespace == "test"
        assert quota.max_memory_mb == 1024.0
        assert quota.max_storage_gb == 10.0
        assert quota.max_glyphs == 1000000
    
    def test_invalidate_cache_specific_namespace(self, resource_manager):
        """Test cache invalidation for specific namespace."""
        # Populate cache
        resource_manager._usage_cache["ns1"] = ResourceUsage(namespace="ns1")
        resource_manager._usage_cache["ns2"] = ResourceUsage(namespace="ns2")
        resource_manager._quota_cache["ns1"] = ResourceQuota(namespace="ns1")
        
        # Invalidate ns1 only
        resource_manager.invalidate_cache("ns1")
        
        assert "ns1" not in resource_manager._usage_cache
        assert "ns2" in resource_manager._usage_cache
        assert "ns1" not in resource_manager._quota_cache
    
    def test_invalidate_cache_all(self, resource_manager):
        """Test cache invalidation for all namespaces."""
        # Populate cache
        resource_manager._usage_cache["ns1"] = ResourceUsage(namespace="ns1")
        resource_manager._usage_cache["ns2"] = ResourceUsage(namespace="ns2")
        
        # Invalidate all
        resource_manager.invalidate_cache()
        
        assert len(resource_manager._usage_cache) == 0
        assert len(resource_manager._quota_cache) == 0


class TestResourceManagerQuotaChecks:
    """Tests for quota checking logic."""
    
    @pytest.fixture
    def resource_manager_with_usage(self):
        """Create ResourceManager with pre-populated usage."""
        manager = ResourceManager(MagicMock())
        
        # Pre-populate cache
        manager._usage_cache["test_ns"] = ResourceUsage(
            namespace="test_ns",
            memory_mb=500.0,
            storage_mb=5000.0,
            glyph_count=50000,
            edge_count=100000,
        )
        manager._quota_cache["test_ns"] = ResourceQuota(
            namespace="test_ns",
            max_memory_mb=1024.0,
            max_storage_gb=10.0,
            max_glyphs=100000,
            max_edges=500000,
        )
        
        return manager
    
    @pytest.mark.asyncio
    async def test_check_quota_within_limits(self, resource_manager_with_usage):
        """Test quota check when within limits."""
        manager = resource_manager_with_usage
        
        # Mock get_usage and get_quota to return cached values
        manager.get_usage = AsyncMock(return_value=manager._usage_cache["test_ns"])
        manager.get_quota = AsyncMock(return_value=manager._quota_cache["test_ns"])
        
        result = await manager.check_quota(
            "test_ns",
            additional_glyphs=1000,
        )
        
        assert result.allowed is True
    
    @pytest.mark.asyncio
    async def test_check_quota_exceeds_glyphs(self, resource_manager_with_usage):
        """Test quota check when glyph limit exceeded."""
        manager = resource_manager_with_usage
        
        manager.get_usage = AsyncMock(return_value=manager._usage_cache["test_ns"])
        manager.get_quota = AsyncMock(return_value=manager._quota_cache["test_ns"])
        
        result = await manager.check_quota(
            "test_ns",
            additional_glyphs=60000,  # Would exceed 100000 limit
        )
        
        assert result.allowed is False
        assert result.resource == "max_glyphs"
    
    @pytest.mark.asyncio
    async def test_check_quota_exceeds_edges(self, resource_manager_with_usage):
        """Test quota check when edge limit exceeded."""
        manager = resource_manager_with_usage
        
        manager.get_usage = AsyncMock(return_value=manager._usage_cache["test_ns"])
        manager.get_quota = AsyncMock(return_value=manager._quota_cache["test_ns"])
        
        result = await manager.check_quota(
            "test_ns",
            additional_edges=500000,  # Would exceed 500000 limit
        )
        
        assert result.allowed is False
        assert result.resource == "max_edges"
    
    @pytest.mark.asyncio
    async def test_enforce_quota_raises_exception(self, resource_manager_with_usage):
        """Test that enforce_quota raises exception when exceeded."""
        manager = resource_manager_with_usage
        
        manager.get_usage = AsyncMock(return_value=manager._usage_cache["test_ns"])
        manager.get_quota = AsyncMock(return_value=manager._quota_cache["test_ns"])
        
        with pytest.raises(NamespaceQuotaExceededException) as exc_info:
            await manager.enforce_quota(
                "test_ns",
                additional_glyphs=60000,
            )
        
        assert exc_info.value.namespace == "test_ns"
        assert exc_info.value.resource == "max_glyphs"
