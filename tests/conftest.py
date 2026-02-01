"""
Pytest configuration and fixtures for Glyphh Runtime tests.
"""

import asyncio
import os
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from hypothesis import settings, Verbosity
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from main import app
from infrastructure.database.connection import Base, get_db


# Test database URL (use SQLite for tests)
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "sqlite+aiosqlite:///:memory:"
)


# =============================================================================
# Hypothesis Configuration
# =============================================================================

# Configure Hypothesis settings for property-based tests
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,  # Disable deadline for CI
    verbosity=Verbosity.normal,
)

settings.register_profile(
    "dev",
    max_examples=20,
    deadline=None,
    verbosity=Verbosity.verbose,
)

settings.register_profile(
    "debug",
    max_examples=5,
    deadline=None,
    verbosity=Verbosity.verbose,
)

# Load profile from environment
profile = os.getenv("HYPOTHESIS_PROFILE", "dev")
settings.load_profile(profile)


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create test client"""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_namespace() -> str:
    """Generate a unique test namespace"""
    return f"test_namespace_{uuid4().hex[:8]}"


@pytest.fixture
def sample_embedding() -> list:
    """Generate a sample 768-dim embedding"""
    import numpy as np
    return np.random.randn(768).astype(float).tolist()


@pytest.fixture
def sample_glyph_data(sample_namespace, sample_embedding) -> dict:
    """Generate sample glyph data"""
    return {
        "namespace": sample_namespace,
        "concept_text": "test concept",
        "embedding": sample_embedding,
        "metadata": {"test": True},
    }


# =============================================================================
# SDK Mock Fixtures (for new explicit API)
# =============================================================================

@pytest.fixture
def mock_role():
    """Create a mock Role with explicit similarity_weight."""
    from unittest.mock import MagicMock
    
    role = MagicMock()
    role.name = "test_role"
    role.similarity_weight = 1.0
    return role


@pytest.fixture
def mock_segment_config(mock_role):
    """Create a mock SegmentConfig with roles."""
    from unittest.mock import MagicMock
    
    segment = MagicMock()
    segment.name = "test_segment"
    segment.roles = [mock_role]
    return segment


@pytest.fixture
def mock_layer_config(mock_segment_config):
    """Create a mock LayerConfig with segments."""
    from unittest.mock import MagicMock
    
    layer = MagicMock()
    layer.name = "test_layer"
    layer.similarity_weight = 1.0
    layer.segments = [mock_segment_config]
    return layer


@pytest.fixture
def mock_encoder_config(mock_layer_config):
    """
    Create a valid EncoderConfig with explicit structure.
    
    This fixture reflects the new SDK API requiring explicit
    LayerConfig, SegmentConfig, and Role definitions.
    """
    from unittest.mock import MagicMock
    
    config = MagicMock()
    config.dimension = 10000
    config.seed = 42
    config.layers = [mock_layer_config]
    return config


@pytest.fixture
def mock_glyphh_model(mock_encoder_config):
    """
    Create a mock GlyphhModel for packaging-only usage.
    
    This fixture reflects the correct SDK usage pattern where
    GlyphhModel is used only for loading/packaging, not for
    runtime encoding or similarity operations.
    """
    from unittest.mock import MagicMock
    
    model = MagicMock()
    model.encoder_config = mock_encoder_config
    model.version = "1.0.0"
    model.name = "test_model"
    # Metadata fields for marketplace display
    model.meta_name = "Test Model"
    model.short_description = "A test model for unit tests"
    model.long_description = "A detailed description of the test model for marketplace pages"
    return model


@pytest.fixture
def mock_encoder(mock_encoder_config):
    """
    Create a mock Encoder instance.
    
    The Encoder is used for encoding operations (not GlyphhModel).
    """
    from unittest.mock import MagicMock
    
    encoder = MagicMock()
    encoder.config = mock_encoder_config
    
    # Mock encode method to return a glyph with global_cortex
    mock_glyph = MagicMock()
    mock_glyph.global_cortex.data = [0.1] * 768
    encoder.encode.return_value = mock_glyph
    
    return encoder


@pytest.fixture
def mock_similarity_calculator(mock_encoder_config):
    """
    Create a mock SimilarityCalculator instance.
    
    The SimilarityCalculator is used for similarity operations.
    """
    from unittest.mock import MagicMock
    
    calculator = MagicMock()
    calculator.config = mock_encoder_config
    calculator.compute.return_value = 0.85
    calculator.compute_batch.return_value = [0.85, 0.72, 0.65]
    
    return calculator


@pytest.fixture
def mock_similarity_service(mock_similarity_calculator):
    """
    Create a mock SimilarityService with SDK calculator.
    """
    from shared.similarity_service import SimilarityService
    
    return SimilarityService(similarity_calculator=mock_similarity_calculator)


@pytest.fixture
def mock_sdk_adapter(mock_encoder, mock_similarity_calculator, mock_encoder_config):
    """
    Create a mock SDKAdapter for testing.
    """
    from unittest.mock import MagicMock, patch
    
    adapter = MagicMock()
    adapter.sdk_version = "0.1.0"
    adapter.is_new_api = True
    adapter.is_available = True
    adapter.warnings = []
    
    adapter.import_config_classes.return_value = {
        'EncoderConfig': MagicMock,
        'LayerConfig': MagicMock,
        'SegmentConfig': MagicMock,
        'Role': MagicMock,
    }
    
    adapter.create_encoder.return_value = mock_encoder
    adapter.create_similarity_calculator.return_value = mock_similarity_calculator
    adapter.create_intent_encoder.return_value = None  # Optional
    
    adapter.get_compatibility_status.return_value = {
        "version": "0.1.0",
        "api_mode": "new",
        "available": True,
        "degraded": False,
        "warnings": [],
    }
    
    return adapter
