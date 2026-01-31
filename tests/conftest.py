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
