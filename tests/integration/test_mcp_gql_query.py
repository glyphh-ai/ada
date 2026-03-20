"""
Integration tests for MCP GQL query execution.

Tests the gql_query tool through the ToolHandler, verifying that GQL queries
execute correctly with DatabaseGlyphStorage and return results in the expected format.

Requirements:
    - 6.1: MCP server uses DatabaseGlyphStorage instead of GlyphWrapper
    - 6.2: MCP server creates ExecutionContext with storage parameter
"""

import json
import pytest
import pytest_asyncio
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from domains.mcp.server import ToolHandler
from domains.auth.service import User, Permission
from domains.models.schemas import GlyphResponse
from mcp.types import CallToolResult


# =============================================================================
# Helpers
# =============================================================================

def parse_tool_result(result: CallToolResult) -> Dict[str, Any]:
    """Parse the JSON payload from a CallToolResult."""
    if result.content and hasattr(result.content[0], "text"):
        return json.loads(result.content[0].text)
    return {}


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
    model.mcp_tools = None
    model.handle_mcp_tool_fn = None

    mock_glyph = MagicMock()
    mock_glyph.global_cortex.data = MagicMock()
    mock_glyph.global_cortex.data.astype.return_value.tolist.return_value = [0.1] * 768
    model.encoder.encode.return_value = mock_glyph

    return model


@pytest.fixture
def mock_query_service(sample_glyphs, sample_embeddings, mock_loaded_model):
    """Create a mock QueryService."""
    service = MagicMock()

    service._model_manager = MagicMock()
    service._model_manager.get_model = AsyncMock(return_value=mock_loaded_model)

    service.list_glyphs_with_embeddings = AsyncMock(
        return_value=(sample_glyphs, sample_embeddings)
    )

    mock_execute_result = MagicMock()
    mock_execute_result.scalars.return_value.all.return_value = []
    mock_execute_result.scalar.return_value = 0

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_execute_result)

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_session
    mock_cm.__aexit__.return_value = False
    service._session_factory = MagicMock(return_value=mock_cm)

    return service


@pytest.fixture
def mock_glyph_storage(sample_glyphs, sample_embeddings):
    """Patch GlyphStorage so the handler skips real DB access."""
    with patch('domains.models.storage.GlyphStorage') as mock_class:
        mock_instance = AsyncMock()
        mock_instance.list_glyphs_with_embeddings.return_value = (
            sample_glyphs, sample_embeddings,
        )
        mock_instance.get_hierarchical_embeddings.return_value = {}
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_auth_service(mock_user):
    """Create a mock AuthService."""
    service = MagicMock()
    service.validate_token = AsyncMock(return_value=mock_user)
    service.check_access = AsyncMock(return_value=True)
    return service


@pytest.fixture
def tool_handler(mock_query_service, mock_auth_service) -> ToolHandler:
    """Create a ToolHandler instance with mocked dependencies."""
    return ToolHandler(
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
        mock_glyph_storage,
    ):
        """
        Test that a simple LIST query returns results in the expected format.

        Validates: Requirements 6.1, 6.2
        """
        mock_gql_module = MagicMock()
        mock_gql_module.parse = MagicMock()
        mock_gql_module.GQLExecutor = MagicMock()
        mock_gql_module.ExecutionContext = MagicMock()
        mock_gql_module.GQLError = Exception

        mock_executor = MagicMock()
        mock_executor.execute.return_value = mock_fact_tree
        mock_executor.get_cache_stats.return_value = {"hits": 0, "misses": 1}
        mock_gql_module.GQLExecutor.return_value = mock_executor

        with patch.dict('sys.modules', {'glyphh': MagicMock(), 'glyphh.gql': mock_gql_module}):
            with patch('domains.gql.storage.DatabaseGlyphStorage') as mock_storage_class:
                mock_storage = MagicMock()
                mock_storage_class.return_value = mock_storage

                with patch('shared.similarity_service.SimilarityService') as mock_sim_service:
                    handler = ToolHandler(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )

                    result = await handler.call_tool(
                        tool_name="gql_query",
                        arguments={"query": "LIST LIMIT 5"},
                        org_id="test-org",
                        model_id="test-model",
                    )

        assert isinstance(result, CallToolResult)
        assert result.isError is False
        data = parse_tool_result(result)
        assert data["query_type"] == "gql"
        assert data["match_method"] == "direct"
        assert data["confidence"] == 1.0
        assert data["fact_tree"] is not None

    @pytest.mark.asyncio
    async def test_gql_query_result_is_fact_tree_format(
        self,
        mock_query_service,
        mock_auth_service,
        mock_fact_tree,
        mock_glyph_storage,
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
                    handler = ToolHandler(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )

                    result = await handler.call_tool(
                        tool_name="gql_query",
                        arguments={"query": "LIST LIMIT 3"},
                        org_id="test-org",
                        model_id="test-model",
                    )

        assert result.isError is False
        data = parse_tool_result(result)
        fact_tree = data.get("fact_tree")
        assert fact_tree is not None

        if isinstance(fact_tree, dict):
            assert "description" in fact_tree or "children" in fact_tree or "value" in fact_tree

    @pytest.mark.asyncio
    async def test_gql_query_uses_database_glyph_storage(
        self,
        mock_query_service,
        mock_auth_service,
        sample_glyphs,
        sample_embeddings,
        mock_fact_tree,
        mock_glyph_storage,
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
                    handler = ToolHandler(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )

                    await handler.call_tool(
                        tool_name="gql_query",
                        arguments={"query": "LIST LIMIT 5"},
                        org_id="test-org",
                        model_id="test-model",
                    )

        mock_glyph_storage.list_glyphs_with_embeddings.assert_called_once_with(
            org_id="test-org",
            model_id="test-model",
            limit=10000,
        )

    @pytest.mark.asyncio
    async def test_gql_query_with_cache_disabled(
        self,
        tool_handler: ToolHandler,
    ):
        """Test GQL query execution with caching disabled."""
        result = await tool_handler.call_tool(
            tool_name="gql_query",
            arguments={"query": "LIST LIMIT 5", "enable_cache": False},
            org_id="test-org",
            model_id="test-model",
        )

        data = parse_tool_result(result)
        assert data.get("query_type") == "gql"

    @pytest.mark.asyncio
    async def test_gql_query_with_cache_enabled(
        self,
        tool_handler: ToolHandler,
    ):
        """Test GQL query execution with caching enabled."""
        result = await tool_handler.call_tool(
            tool_name="gql_query",
            arguments={"query": "LIST LIMIT 5", "enable_cache": True},
            org_id="test-org",
            model_id="test-model",
        )

        data = parse_tool_result(result)
        assert data.get("query_type") == "gql"

    @pytest.mark.asyncio
    async def test_gql_query_missing_model_returns_error(
        self,
        tool_handler: ToolHandler,
        mock_query_service,
    ):
        """Test that querying a non-existent model returns an appropriate error."""
        mock_query_service._model_manager.get_model = AsyncMock(return_value=None)

        result = await tool_handler.call_tool(
            tool_name="gql_query",
            arguments={"query": "LIST LIMIT 5"},
            org_id="test-org",
            model_id="nonexistent-model",
        )

        assert result.isError is True
        error_text = result.content[0].text if result.content else ""
        assert "not found" in error_text.lower() or "Model not found" in error_text

    @pytest.mark.asyncio
    async def test_gql_query_response_dict_format(
        self,
        mock_query_service,
        mock_auth_service,
        mock_fact_tree,
        mock_glyph_storage,
    ):
        """
        Test that the tool result contains the expected response format.

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
                    handler = ToolHandler(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )

                    result = await handler.call_tool(
                        tool_name="gql_query",
                        arguments={"query": "LIST LIMIT 5"},
                        org_id="test-org",
                        model_id="test-model",
                    )

        assert result.isError is False
        data = parse_tool_result(result)

        assert "state" in data
        assert "fact_tree" in data
        assert "query_type" in data
        assert "match_method" in data
        assert "confidence" in data
        assert "query_time_ms" in data

        assert data["query_type"] == "gql"
        assert data["match_method"] == "direct"
        assert data["confidence"] == 1.0


class TestMCPGQLQueryWithRealStorage:
    """
    Integration tests that verify DatabaseGlyphStorage integration.

    These tests verify the actual integration between the tool handler,
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
        mock_glyph_storage,
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
                    handler = ToolHandler(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )

                    await handler.call_tool(
                        tool_name="gql_query",
                        arguments={"query": "LIST LIMIT 5"},
                        org_id="test-org",
                        model_id="test-model",
                    )

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
        mock_glyph_storage,
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
                    handler = ToolHandler(
                        query_service=mock_query_service,
                        auth_service=mock_auth_service,
                    )

                    await handler.call_tool(
                        tool_name="gql_query",
                        arguments={"query": "LIST LIMIT 5"},
                        org_id="test-org",
                        model_id="test-model",
                    )

                    mock_gql_module.ExecutionContext.assert_called_once()
                    call_kwargs = mock_gql_module.ExecutionContext.call_args.kwargs

                    assert "storage" in call_kwargs
                    assert call_kwargs["storage"] == mock_storage


class TestMCPGQLQueryValidation:
    """Tests for input validation in GQL queries.

    Note: org_id and model_id are now injected from the URL path by the
    ASGI middleware, not from tool arguments. These tests verify the
    handler still works when called with explicit org_id/model_id params.
    """

    @pytest.mark.asyncio
    async def test_gql_query_requires_query_argument(
        self,
        tool_handler: ToolHandler,
    ):
        """Test that query argument is required."""
        result = await tool_handler.call_tool(
            tool_name="gql_query",
            arguments={},
            org_id="test-org",
            model_id="test-model",
        )

        assert result.isError is True
        error_text = result.content[0].text if result.content else ""
        assert "query" in error_text.lower() or "error" in error_text.lower()
