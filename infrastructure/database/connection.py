"""
Database connection management with async SQLAlchemy and pgvector.
"""

import logging
import subprocess
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

from infrastructure.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Convert sync URL to async
database_url = settings.database_url
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Create async engine
engine = create_async_engine(
    database_url,
    echo=settings.log_level == "DEBUG",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for models
Base = declarative_base()


async def init_db() -> None:
    """Initialize database - enable pgvector and run migrations.
    
    Automatically applies any pending Alembic migrations on startup,
    making installation seamless for end users.
    """
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension enabled")
    
    # Run Alembic migrations automatically using subprocess
    # to avoid async context issues with Alembic's asyncio.run()
    import os
    
    # Find alembic.ini relative to the app root
    app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    alembic_ini = os.path.join(app_root, "alembic.ini")
    
    if os.path.exists(alembic_ini):
        try:
            # Run alembic as subprocess to avoid async context issues
            result = subprocess.run(
                [".venv/bin/alembic", "upgrade", "head"],
                cwd=app_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("Database migrations applied successfully")
            else:
                logger.error(f"Migration failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Failed to run migrations: {e}")
    else:
        logger.warning(f"alembic.ini not found at {alembic_ini}, skipping migrations")


async def close_db() -> None:
    """Close database connections"""
    await engine.dispose()
    logger.info("Database connections closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
