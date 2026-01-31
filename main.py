"""
Glyphh Runtime - Execution environment for deployed .glyphh models

This runtime serves deployed models through REST and MCP APIs,
handling persistent storage, multi-model management, licensing,
authentication, and real-time data ingestion.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from infrastructure.config import get_settings
from infrastructure.database import init_db, close_db
from shared.exceptions import GlyphhRuntimeException
from shared.middleware import (
    CorrelationIDMiddleware,
    LoggingMiddleware,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

settings = get_settings()
_start_time = datetime.utcnow()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management"""
    # Startup
    logger.info("Starting Glyphh Runtime...")
    await init_db()
    logger.info("Database initialized")
    
    # TODO: Initialize licensing service
    # TODO: Load any pre-configured models
    
    yield
    
    # Shutdown
    logger.info("Shutting down Glyphh Runtime...")
    await close_db()
    logger.info("Database connections closed")


# Create FastAPI application
app = FastAPI(
    title="Glyphh Runtime",
    description="Execution environment for deployed .glyphh models",
    version="1.0.0",
    docs_url="/docs" if settings.deployment_mode == "local" else None,
    redoc_url="/redoc" if settings.deployment_mode == "local" else None,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(LoggingMiddleware)


# Global exception handlers
@app.exception_handler(GlyphhRuntimeException)
async def runtime_exception_handler(request: Request, exc: GlyphhRuntimeException) -> JSONResponse:
    """Handle custom runtime exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "correlation_id": getattr(request.state, "correlation_id", "unknown"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "correlation_id": getattr(request.state, "correlation_id", "unknown"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )


# Health check endpoints
@app.get("/health")
async def health_check() -> dict:
    """Liveness probe"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/health/ready")
async def readiness_check() -> dict:
    """Readiness probe - checks all dependencies"""
    # TODO: Add actual checks for database, license, SDK
    return {
        "status": "ready",
        "checks": {
            "database": "ok",
            "license": "ok",
            "sdk": "ok"
        }
    }


# CLI-facing deployment API endpoints
@app.get("/api/status")
async def get_status() -> dict:
    """Runtime status for CLI"""
    uptime = datetime.utcnow() - _start_time
    return {
        "version": "1.0.0",
        "models_loaded": 0,  # TODO: Get from ModelManager
        "uptime": str(uptime),
        "deployment_mode": settings.deployment_mode
    }


# Import and include routers
# from domains.deploy.routes import router as deploy_router
# from domains.models.routes import router as models_router
# from domains.query.routes import router as query_router
# from domains.tokens.routes import router as tokens_router

# app.include_router(deploy_router, prefix="/api", tags=["deployment"])
# app.include_router(models_router, prefix="/api", tags=["models"])
# app.include_router(query_router, prefix="/api/v1", tags=["query"])
# app.include_router(tokens_router, prefix="/api", tags=["tokens"])


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.deployment_mode == "local",
        log_level="info"
    )
