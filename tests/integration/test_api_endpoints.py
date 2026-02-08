"""
Integration tests for API endpoints.

Tests complete request/response cycles through the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Integration tests for health check endpoints."""
    
    def test_health_check(self, client: TestClient):
        """Test liveness probe endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_readiness_check(self, client: TestClient):
        """Test readiness probe endpoint."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "timestamp" in data
    
    def test_metrics_endpoint(self, client: TestClient):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "runtime_uptime_seconds" in data
        assert "runtime_models_loaded" in data


class TestDeploymentEndpoints:
    """Integration tests for deployment API endpoints."""
    
    def test_get_status(self, client: TestClient):
        """Test runtime status endpoint."""
        response = client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "models_loaded" in data
        assert "deployment_mode" in data
    
    def test_list_models_empty(self, client: TestClient):
        """Test listing models when none are loaded."""
        response = client.get("/api/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    def test_list_tokens(self, client: TestClient):
        """Test listing tokens."""
        response = client.get("/api/tokens")
        
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data


class TestGlyphEndpoints:
    """Integration tests for glyph CRUD endpoints."""
    
    @pytest.mark.skip(reason="Requires model to be loaded")
    def test_create_glyph(self, client: TestClient):
        """Test glyph creation endpoint."""
        response = client.post(
            "/test_org/test_model/glyphs",
            json={
                "concept": "test concept",
                "metadata": {"test": True},
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "glyph_id" in data
    
    @pytest.mark.skip(reason="Requires model to be loaded")
    def test_list_glyphs(self, client: TestClient):
        """Test glyph listing endpoint."""
        response = client.get("/test_org/test_model/glyphs")
        
        assert response.status_code == 200
        data = response.json()
        assert "glyphs" in data
        assert "total" in data


class TestQueryEndpoints:
    """Integration tests for query endpoints."""
    
    @pytest.mark.skip(reason="Requires model to be loaded")
    def test_similarity_search(self, client: TestClient):
        """Test similarity search endpoint."""
        response = client.post(
            "/test_org/test_model/search",
            json={
                "query": "test query",
                "top_k": 10,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query_time_ms" in data
    
    @pytest.mark.skip(reason="Requires model to be loaded")
    def test_fact_tree(self, client: TestClient):
        """Test fact tree generation endpoint."""
        response = client.post(
            "/test_org/test_model/fact-tree",
            json={
                "claim": "test claim",
                "max_depth": 3,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "confidence" in data


class TestErrorHandling:
    """Integration tests for error handling."""
    
    def test_not_found_endpoint(self, client: TestClient):
        """Test 404 response for non-existent endpoint."""
        response = client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client: TestClient):
        """Test 405 response for wrong HTTP method."""
        response = client.delete("/health")
        
        assert response.status_code == 405
    
    def test_validation_error(self, client: TestClient):
        """Test validation error response."""
        # Missing required fields
        response = client.post(
            "/test_org/test_model/search",
            json={}
        )
        
        assert response.status_code == 422
