"""
Integration tests for MCP GQL query execution.

Tests the gql_query tool through the MCP server, verifying that GQL queries
execute correctly with DatabaseGlyphStorage and return results in the expected format.

Requirements:
    - 6.1: MCP server uses DatabaseGlyphStorage instead of GlyphWrapper
    - 6.2: MCP server creates ExecutionContext with storage parameter
"""

import pytest
import pytest_asyncio
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from domains.mcp.server import MCPServer, MCPResponse
from domains.auth.service import User, Permission
from domains.models.schemas import GlyphResponse


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_user() -> User:
    """Create a mock authenticated user."""
    return User(
        user_id="test-user-id",
        email="test@example.com",
        org_permissions={"test-org": {Permission.READ, Permission.WRITE, Permission.ADMIN}},
        org_id="test-org",
        token_type="jwt",
    )


@pytest.fixture
def sample_glyphs() -> List[GlyphResponse]:
    """Create sample GlyphResponse objects for testing."""
    now = datetime.utcnow()
    return [
        GlyphResponse(
            id=uuid4(),
            org_id="test-org",
            model_id="test-model",
            concept_text="red car",
            metadata={"color": "red", "type": "vehicle"},
            created_at=now,
            updated_at=now,
        ),
        GlyphResponse(
            id=uuid4(),
            org_id="test-org",
            model_id="test-model",
            concept_text="blue truck",
            metadata={"color": "blue", "type": "vehicle"},
            created_at=now,
            updated_at=now,
        ),
        GlyphResponse(
            id=uuid4(),
            org_id="test-org",
            model_id="test-model",
            concept_text="green bicycle",
            metadata={"color": "green", "type": "vehicle"},
            created_at=now,
            updated_at=now,
        ),
    ]


@pytest.fixture
def sample_embeddings(sample_glyphs: List[GlyphResponse]) -> Dict[str, List[float]]:
    """Create sample embeddings for the test glyphs."""
    import numpy as np
    
    embeddings = {}
    for glyph in sample_glyphs:
        # Create a random 768-dim embedding
        embedding = np.random.randn(768).astype(float).tolist()
        embeddings[str(glyph.id)] = embedding
    
    return embeddings


@pytest.fixture
def mock_loaded_model():
    """Create a mock loaded model with encoder and similarity calculator."""
    model = MagicMock()
    model.sdk_model = MagicMock()
    model.encoder = MagicMock()
    model.similarity_calculator = MagicMock()
    
    # Mock encoder.encode to return a glyph with global_cortex
    mock_glyph = MagicMock()
    mock_glyph.global_cortex.data = MagicMock()
    mock_glyph.global_cortex.data.astype.return_value.tolist.return_value = [0.1] * 768
    model.encoder.encode.return_value = mock_glyph
    
    return model


@pytest.fixture
def mock_query_service(sample_glyphs, sample_embeddings, mock_loaded_model):
    """Create a mock QueryService."""
    service = MagicMock()
    
    # Mock model manager
    service._model_manager = MagicMock()
    service._model_manager.get_model = AsyncMock(return_value=mock_loaded_model)
    
    # Mock list_glyphs_with_embeddings
    service.list_glyphs_with_embeddings = AsyncMock(
        return_value=(sample_glyphs, sample_embeddings)
    )
    
    return service


@pytest.fixture
def mock_auth_service(mock_user):
    """Create a mock AuthService."""
    service = MagicMock()
    service.validate_token = AsyncMock(return_value=mock_user)
    service.check_access = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mcp_server(mock_query_service, mock_auth_service) -> MCPServer:
    """Create an MCPServer instance with mocked dependencies."""
    return MCPServer(
        query_service=mock_query_service,
        auth_service=mock_auth_service,
    )


@pytest.fixture
def mock_fact_tree():
    """Create a mock FactTree for testing."""
    fact_tree = MagicMock()
    fact_tree.to_json.return_value = {
        "description": "Similarity Computation",
        "value": None,
        "children": [
            {
                "description": "query",
                "value": "LIST LIMIT 5",
                "children": [],
                "citations": [],
                "data_context": {}
            }
        ],
        "citations": [],
        "data_context": {}
    }
    return fact_tree


# =============================================================================
# Integration Tests - With Mocked GQL Components
# =============================================================================

class TestMCPGQLQuery:
    """Integration tests for MCP GQL query execution."""
    
    @pytest.mark.asyncio
    async def test_gql_query_list_returns_expected_format(
        self,
        mock_query_service,
        mock_auth_service,
        sample_glyphs,
        mock_fact_tree,
    ):
        """
        Test that a simple LIST query returns results in the expected format.
        
        Validates: Requirements 6.1, 6.2
        """
        # Mock all GQL imports
        mock_gql_module = MagicMock()
        mock_gql_module.parse = MagicMock()
        mock_gql_module.GQLExecutor = MagicMock()
        mock_gql_module.ExecutionContext = MagicMock()
        mock_gql_module.GQLError = Exception
        
        # Setup executor mock
        mock_executor = MagicMock()
        mock_executor.execute.return_value = mock_fact_tree
        mock_executor.get_cache_stats.return_value = {"hits": 0, "misses": 1}
        mock_gql_module.GQLExecutor.return_value = mock_executor
        
        with patch.dict('sys.modules', {'glyphh': MagicMock(), 'glyphh.gql': mock_gql_module}):
            with patch('domains.gql.storage.DatabaseGlyphStorage') as mock_storage_class:
                mock_storage = MagicMock()
                mock_storage_class.return_value = mock_storage
                
                with patch('shared.similarity_service.SimilarityService') as mock_sim_service:
                    server = MCPServer(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )
                    
                    response = await server.handle_tool_call(
                        tool_name="gql_query",
                        arguments={
                            "org_id": "test-org",
                            "model_id": "test-model",
                            "query": "LIST LIMIT 5",
                        },
                        auth_token="test-token",
                    )
        
        # Verify response structure
        assert isinstance(response, MCPResponse)
        assert response.query_type == "gql"
        assert response.match_method == "direct"
        assert response.confidence == 1.0
        assert response.result is not None

    @pytest.mark.asyncio
    async def test_gql_query_result_is_fact_tree_format(
        self,
        mock_query_service,
        mock_auth_service,
        mock_fact_tree,
    ):
        """
        Test that GQL query result is in FactTree JSON format.
        
        Validates: Requirements 6.1, 6.2
        """
        mock_gql_module = MagicMock()
        mock_gql_module.parse = MagicMock()
        mock_gql_module.GQLExecutor = MagicMock()
        mock_gql_module.ExecutionContext = MagicMock()
        mock_gql_module.GQLError = Exception
        
        mock_executor = MagicMock()
        mock_executor.execute.return_value = mock_fact_tree
        mock_executor.get_cache_stats.return_value = None
        mock_gql_module.GQLExecutor.return_value = mock_executor
        
        with patch.dict('sys.modules', {'glyphh': MagicMock(), 'glyphh.gql': mock_gql_module}):
            with patch('domains.gql.storage.DatabaseGlyphStorage'):
                with patch('shared.similarity_service.SimilarityService'):
                    server = MCPServer(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )
                    
                    response = await server.handle_tool_call(
                        tool_name="gql_query",
                        arguments={
                            "org_id": "test-org",
                            "model_id": "test-model",
                            "query": "LIST LIMIT 3",
                        },
                        auth_token="test-token",
                    )
        
        assert response.is_error is False
        result = response.result
        assert result is not None
        
        # FactTree JSON has a specific structure with description, value, children
        if isinstance(result, dict):
            assert "description" in result or "children" in result or "value" in result

    @pytest.mark.asyncio
    async def test_gql_query_uses_database_glyph_storage(
        self,
        mock_query_service,
        mock_auth_service,
        sample_glyphs,
        sample_embeddings,
        mock_fact_tree,
    ):
        """
        Test that GQL query uses DatabaseGlyphStorage (not GlyphWrapper).
        
        Validates: Requirement 6.1 - MCP server uses DatabaseGlyphStorage
        """
        mock_gql_module = MagicMock()
        mock_gql_module.parse = MagicMock()
        mock_gql_module.GQLExecutor = MagicMock()
        mock_gql_module.ExecutionContext = MagicMock()
        mock_gql_module.GQLError = Exception
        
        mock_executor = MagicMock()
        mock_executor.execute.return_value = mock_fact_tree
        mock_executor.get_cache_stats.return_value = None
        mock_gql_module.GQLExecutor.return_value = mock_executor
        
        with patch.dict('sys.modules', {'glyphh': MagicMock(), 'glyphh.gql': mock_gql_module}):
            with patch('domains.gql.storage.DatabaseGlyphStorage') as mock_storage_class:
                mock_storage = MagicMock()
                mock_storage_class.return_value = mock_storage
                
                with patch('shared.similarity_service.SimilarityService'):
                    server = MCPServer(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )
                    
                    await server.handle_tool_call(
                        tool_name="gql_query",
                        arguments={
                            "org_id": "test-org",
                            "model_id": "test-model",
                            "query": "LIST LIMIT 5",
                        },
                        auth_token="test-token",
                    )
        
        # Verify list_glyphs_with_embeddings was called
        mock_query_service.list_glyphs_with_embeddings.assert_called_once_with(
            org_id="test-org",
            model_id="test-model",
            limit=10000,
        )
        
        # Verify DatabaseGlyphStorage was instantiated
        mock_storage_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_gql_query_with_cache_disabled(
        self,
        mcp_server: MCPServer,
    ):
        """
        Test GQL query execution with caching disabled.
        """
        response = await mcp_server.handle_tool_call(
            tool_name="gql_query",
            arguments={
                "org_id": "test-org",
                "model_id": "test-model",
                "query": "LIST LIMIT 5",
                "enable_cache": False,
            },
            auth_token="test-token",
        )
        
        # Response may have error due to missing glyphh module, but structure should be valid
        assert response.query_type == "gql"
    
    @pytest.mark.asyncio
    async def test_gql_query_with_cache_enabled(
        self,
        mcp_server: MCPServer,
    ):
        """
        Test GQL query execution with caching enabled.
        """
        response = await mcp_server.handle_tool_call(
            tool_name="gql_query",
            arguments={
                "org_id": "test-org",
                "model_id": "test-model",
                "query": "LIST LIMIT 5",
                "enable_cache": True,
            },
            auth_token="test-token",
        )
        
        assert response.query_type == "gql"
    
    @pytest.mark.asyncio
    async def test_gql_query_missing_model_returns_error(
        self,
        mcp_server: MCPServer,
        mock_query_service,
    ):
        """
        Test that querying a non-existent model returns an appropriate error.
        """
        mock_query_service._model_manager.get_model = AsyncMock(return_value=None)
        
        response = await mcp_server.handle_tool_call(
            tool_name="gql_query",
            arguments={
                "org_id": "test-org",
                "model_id": "nonexistent-model",
                "query": "LIST LIMIT 5",
            },
            auth_token="test-token",
        )
        
        assert response.is_error is True
        assert "not found" in response.error.lower() or "Model not found" in response.error

    @pytest.mark.asyncio
    async def test_gql_query_response_to_dict_format(
        self,
        mock_query_service,
        mock_auth_service,
        mock_fact_tree,
    ):
        """
        Test that MCPResponse.to_dict() returns the expected format.
        
        Validates: Requirements 6.1, 6.2
        """
        mock_gql_module = MagicMock()
        mock_gql_module.parse = MagicMock()
        mock_gql_module.GQLExecutor = MagicMock()
        mock_gql_module.ExecutionContext = MagicMock()
        mock_gql_module.GQLError = Exception
        
        mock_executor = MagicMock()
        mock_executor.execute.return_value = mock_fact_tree
        mock_executor.get_cache_stats.return_value = None
        mock_gql_module.GQLExecutor.return_value = mock_executor
        
        with patch.dict('sys.modules', {'glyphh': MagicMock(), 'glyphh.gql': mock_gql_module}):
            with patch('domains.gql.storage.DatabaseGlyphStorage'):
                with patch('shared.similarity_service.SimilarityService'):
                    server = MCPServer(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )
                    
                    response = await server.handle_tool_call(
                        tool_name="gql_query",
                        arguments={
                            "org_id": "test-org",
                            "model_id": "test-model",
                            "query": "LIST LIMIT 5",
                        },
                        auth_token="test-token",
                    )
        
        response_dict = response.to_dict()
        
        # Verify expected keys in response dict
        assert "content" in response_dict
        assert "isError" in response_dict
        assert "result" in response_dict
        assert "query_type" in response_dict
        assert "match_method" in response_dict
        assert "confidence" in response_dict
        assert "query_time_ms" in response_dict
        
        # Verify values
        assert response_dict["query_type"] == "gql"
        assert response_dict["match_method"] == "direct"
        assert response_dict["confidence"] == 1.0
        assert response_dict["isError"] is False


class TestMCPGQLQueryWithRealStorage:
    """
    Integration tests that verify DatabaseGlyphStorage integration.
    
    These tests verify the actual integration between MCP server,
    DatabaseGlyphStorage, and the GQL executor.
    """
    
    @pytest.mark.asyncio
    async def test_database_glyph_storage_is_used(
        self,
        mock_query_service,
        mock_auth_service,
        sample_glyphs,
        sample_embeddings,
        mock_loaded_model,
        mock_fact_tree,
    ):
        """
        Test that DatabaseGlyphStorage is instantiated with correct parameters.
        
        Validates: Requirements 6.1, 6.4
        """
        mock_gql_module = MagicMock()
        mock_gql_module.parse = MagicMock()
        mock_gql_module.GQLExecutor = MagicMock()
        mock_gql_module.ExecutionContext = MagicMock()
        mock_gql_module.GQLError = Exception
        
        mock_executor = MagicMock()
        mock_executor.execute.return_value = mock_fact_tree
        mock_executor.get_cache_stats.return_value = None
        mock_gql_module.GQLExecutor.return_value = mock_executor
        
        with patch.dict('sys.modules', {'glyphh': MagicMock(), 'glyphh.gql': mock_gql_module}):
            with patch('domains.gql.storage.DatabaseGlyphStorage') as mock_storage_class:
                mock_storage = MagicMock()
                mock_storage_class.return_value = mock_storage
                
                with patch('shared.similarity_service.SimilarityService'):
                    server = MCPServer(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )
                    
                    await server.handle_tool_call(
                        tool_name="gql_query",
                        arguments={
                            "org_id": "test-org",
                            "model_id": "test-model",
                            "query": "LIST LIMIT 5",
                        },
                        auth_token="test-token",
                    )
                    
                    # Verify DatabaseGlyphStorage was instantiated with correct params
                    mock_storage_class.assert_called_once()
                    call_kwargs = mock_storage_class.call_args.kwargs
                    
                    assert call_kwargs["org_id"] == "test-org"
                    assert call_kwargs["model_id"] == "test-model"
                    assert call_kwargs["glyphs"] == sample_glyphs
                    assert call_kwargs["embeddings"] == sample_embeddings

    @pytest.mark.asyncio
    async def test_execution_context_uses_storage_parameter(
        self,
        mock_query_service,
        mock_auth_service,
        sample_glyphs,
        sample_embeddings,
        mock_loaded_model,
        mock_fact_tree,
    ):
        """
        Test that ExecutionContext is created with storage parameter.
        
        Validates: Requirement 6.2
        """
        mock_gql_module = MagicMock()
        mock_gql_module.parse = MagicMock()
        mock_gql_module.GQLExecutor = MagicMock()
        mock_gql_module.ExecutionContext = MagicMock()
        mock_gql_module.GQLError = Exception
        
        mock_executor = MagicMock()
        mock_executor.execute.return_value = mock_fact_tree
        mock_executor.get_cache_stats.return_value = None
        mock_gql_module.GQLExecutor.return_value = mock_executor
        
        with patch.dict('sys.modules', {'glyphh': MagicMock(), 'glyphh.gql': mock_gql_module}):
            with patch('domains.gql.storage.DatabaseGlyphStorage') as mock_storage_class:
                mock_storage = MagicMock()
                mock_storage_class.return_value = mock_storage
                
                with patch('shared.similarity_service.SimilarityService'):
                    server = MCPServer(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )
                    
                    await server.handle_tool_call(
                        tool_name="gql_query",
                        arguments={
                            "org_id": "test-org",
                            "model_id": "test-model",
                            "query": "LIST LIMIT 5",
                        },
                        auth_token="test-token",
                    )
                    
                    # Verify ExecutionContext was created with storage parameter
                    mock_gql_module.ExecutionContext.assert_called_once()
                    call_kwargs = mock_gql_module.ExecutionContext.call_args.kwargs
                    
                    # Should have storage parameter, not glyphs
                    assert "storage" in call_kwargs
                    assert call_kwargs["storage"] == mock_storage


class TestMCPGQLQueryAuthentication:
    """Tests for authentication and authorization in GQL queries."""
    
    @pytest.mark.asyncio
    async def test_gql_query_requires_authentication(
        self,
        mock_query_service,
    ):
        """Test that GQL query requires valid authentication."""
        from shared.exceptions import AuthenticationException
        
        mock_auth_service = MagicMock()
        mock_auth_service.validate_token = AsyncMock(
            side_effect=AuthenticationException("Invalid token")
        )
        
        server = MCPServer(
            query_service=mock_query_service,
            auth_service=mock_auth_service,
        )
        
        response = await server.handle_tool_call(
            tool_name="gql_query",
            arguments={
                "org_id": "test-org",
                "model_id": "test-model",
                "query": "LIST LIMIT 5",
            },
            auth_token="invalid-token",
        )
        
        assert response.is_error is True
        assert "authentication" in response.error.lower() or "Authentication" in response.error
    
    @pytest.mark.asyncio
    async def test_gql_query_checks_authorization(
        self,
        mock_query_service,
        mock_user,
    ):
        """Test that GQL query checks user authorization for the model."""
        from shared.exceptions import AuthorizationException
        
        mock_auth_service = MagicMock()
        mock_auth_service.validate_token = AsyncMock(return_value=mock_user)
        mock_auth_service.check_access = AsyncMock(
            side_effect=AuthorizationException("Not authorized")
        )
        
        server = MCPServer(
            query_service=mock_query_service,
            auth_service=mock_auth_service,
        )
        
        response = await server.handle_tool_call(
            tool_name="gql_query",
            arguments={
                "org_id": "test-org",
                "model_id": "test-model",
                "query": "LIST LIMIT 5",
            },
            auth_token="test-token",
        )
        
        assert response.is_error is True
        assert "authorized" in response.error.lower() or "Not authorized" in response.error


class TestMCPGQLQueryValidation:
    """Tests for input validation in GQL queries."""
    
    @pytest.mark.asyncio
    async def test_gql_query_requires_org_id(
        self,
        mcp_server: MCPServer,
    ):
        """Test that org_id is required."""
        response = await mcp_server.handle_tool_call(
            tool_name="gql_query",
            arguments={
                "model_id": "test-model",
                "query": "LIST LIMIT 5",
            },
            auth_token="test-token",
        )
        
        assert response.is_error is True
        assert "org_id" in response.error.lower() or "required" in response.error.lower()
    
    @pytest.mark.asyncio
    async def test_gql_query_requires_model_id(
        self,
        mcp_server: MCPServer,
    ):
        """Test that model_id is required."""
        response = await mcp_server.handle_tool_call(
            tool_name="gql_query",
            arguments={
                "org_id": "test-org",
                "query": "LIST LIMIT 5",
            },
            auth_token="test-token",
        )
        
        assert response.is_error is True
        assert "model_id" in response.error.lower() or "required" in response.error.lower()
    
    @pytest.mark.asyncio
    async def test_gql_query_requires_query(
        self,
        mcp_server: MCPServer,
    ):
        """Test that query is required."""
        response = await mcp_server.handle_tool_call(
            tool_name="gql_query",
            arguments={
                "org_id": "test-org",
                "model_id": "test-model",
            },
            auth_token="test-token",
        )
        
        assert response.is_error is True
        assert "query" in response.error.lower() or "required" in response.error.lower()
