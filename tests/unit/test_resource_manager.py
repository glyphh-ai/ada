"""
Unit tests for ResourceManager.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from domains.resources.manager import ResourceManager, ResourceUsage, ResourceQuota, QuotaCheckResult
from shared.exceptions import ModelNotFoundException, QuotaExceededException


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
        usage = ResourceUsage(org_id="test_org", model_id="test_model")
        
        assert usage.org_id == "test_org"
        assert usage.model_id == "test_model"
        assert usage.memory_mb == 0.0
        assert usage.storage_mb == 0.0
        assert usage.glyph_count == 0
        assert usage.edge_count == 0
    
    def test_resource_quota_defaults(self):
        """Test ResourceQuota default values."""
        quota = ResourceQuota(org_id="test_org", model_id="test_model")
        
        assert quota.org_id == "test_org"
        assert quota.model_id == "test_model"
        assert quota.max_memory_mb == 1024.0
        assert quota.max_storage_gb == 10.0
        assert quota.max_glyphs == 1000000
    
    def test_invalidate_cache_specific_model(self, resource_manager):
        """Test cache invalidation for specific org/model."""
        key1 = ("org1", "model1")
        key2 = ("org2", "model2")
        resource_manager._usage_cache[key1] = ResourceUsage(org_id="org1", model_id="model1")
        resource_manager._usage_cache[key2] = ResourceUsage(org_id="org2", model_id="model2")
        resource_manager._quota_cache[key1] = ResourceQuota(org_id="org1", model_id="model1")
        
        resource_manager.invalidate_cache("org1", "model1")
        
        assert key1 not in resource_manager._usage_cache
        assert key2 in resource_manager._usage_cache
        assert key1 not in resource_manager._quota_cache
    
    def test_invalidate_cache_all(self, resource_manager):
        """Test cache invalidation for all models."""
        key1 = ("org1", "model1")
        key2 = ("org2", "model2")
        resource_manager._usage_cache[key1] = ResourceUsage(org_id="org1", model_id="model1")
        resource_manager._usage_cache[key2] = ResourceUsage(org_id="org2", model_id="model2")
        
        resource_manager.invalidate_cache()
        
        assert len(resource_manager._usage_cache) == 0
        assert len(resource_manager._quota_cache) == 0


class TestResourceManagerQuotaChecks:
    """Tests for quota checking logic."""
    
    @pytest.fixture
    def resource_manager_with_usage(self):
        """Create ResourceManager with pre-populated usage."""
        manager = ResourceManager(MagicMock())
        
        key = ("test_org", "test_model")
        manager._usage_cache[key] = ResourceUsage(
            org_id="test_org",
            model_id="test_model",
            memory_mb=500.0,
            storage_mb=5000.0,
            glyph_count=50000,
            edge_count=100000,
        )
        manager._quota_cache[key] = ResourceQuota(
            org_id="test_org",
            model_id="test_model",
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
        key = ("test_org", "test_model")
        
        manager.get_usage = AsyncMock(return_value=manager._usage_cache[key])
        manager.get_quota = AsyncMock(return_value=manager._quota_cache[key])
        
        result = await manager.check_quota(
            "test_org", "test_model",
            additional_glyphs=1000,
        )
        
        assert result.allowed is True
    
    @pytest.mark.asyncio
    async def test_check_quota_exceeds_glyphs(self, resource_manager_with_usage):
        """Test quota check when glyph limit exceeded."""
        manager = resource_manager_with_usage
        key = ("test_org", "test_model")
        
        manager.get_usage = AsyncMock(return_value=manager._usage_cache[key])
        manager.get_quota = AsyncMock(return_value=manager._quota_cache[key])
        
        result = await manager.check_quota(
            "test_org", "test_model",
            additional_glyphs=60000,
        )
        
        assert result.allowed is False
        assert result.resource == "max_glyphs"
    
    @pytest.mark.asyncio
    async def test_check_quota_exceeds_edges(self, resource_manager_with_usage):
        """Test quota check when edge limit exceeded."""
        manager = resource_manager_with_usage
        key = ("test_org", "test_model")
        
        manager.get_usage = AsyncMock(return_value=manager._usage_cache[key])
        manager.get_quota = AsyncMock(return_value=manager._quota_cache[key])
        
        result = await manager.check_quota(
            "test_org", "test_model",
            additional_edges=500000,
        )
        
        assert result.allowed is False
        assert result.resource == "max_edges"
    
    @pytest.mark.asyncio
    async def test_enforce_quota_raises_exception(self, resource_manager_with_usage):
        """Test that enforce_quota raises exception when exceeded."""
        manager = resource_manager_with_usage
        key = ("test_org", "test_model")
        
        manager.get_usage = AsyncMock(return_value=manager._usage_cache[key])
        manager.get_quota = AsyncMock(return_value=manager._quota_cache[key])
        
        with pytest.raises(QuotaExceededException):
            await manager.enforce_quota(
                "test_org", "test_model",
                additional_glyphs=60000,
            )
