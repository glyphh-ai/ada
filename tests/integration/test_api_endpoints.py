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
        assert data["status"] in ("healthy", "degraded")
        assert "timestamp" in data


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
        """Test similarity search via MCP endpoint."""
        response = client.post(
            "/test_org/test_model/mcp",
            json={
                "tool": "nl_query",
                "arguments": {
                    "query": "test query",
                },
            }
        )
        
        assert response.status_code == 200
    
    @pytest.mark.skip(reason="Requires model to be loaded")
    def test_fact_tree(self, client: TestClient):
        """Test fact tree generation via MCP endpoint."""
        response = client.post(
            "/test_org/test_model/mcp",
            json={
                "tool": "gql_query",
                "arguments": {
                    "query": "LIST LIMIT 5",
                },
            }
        )
        
        assert response.status_code == 200


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
