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

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from infrastructure.config import get_settings, validate_settings
from infrastructure.database import init_db, close_db, get_db, async_session_maker
from shared.exceptions import GlyphhRuntimeException
from shared.middleware import (
    CorrelationIDMiddleware,
    LoggingMiddleware,
)
from domains.models.manager import ModelManager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

settings = get_settings()
_start_time = datetime.utcnow()

# Global model manager instance
model_manager: ModelManager = None


def get_model_manager() -> ModelManager:
    """Dependency for getting model manager"""
    return model_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management"""
    global model_manager
    
    # Startup
    logger.info("Starting Glyphh Runtime...")
    
    # Validate configuration
    try:
        validate_settings()
        logger.info(f"Configuration validated for {settings.deployment_mode} mode")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    
    await init_db()
    logger.info("Database initialized")
    
    # Initialize model manager
    model_manager = ModelManager(async_session_maker)
    logger.info("Model manager initialized")
    
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
async def readiness_check(db: AsyncSession = Depends(get_db)) -> dict:
    """Readiness probe - checks all dependencies"""
    checks = {}
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        checks["database"] = "error"
    
    # Check license (placeholder)
    checks["license"] = "ok"  # TODO: Implement actual license check
    
    # Check SDK (placeholder)
    checks["sdk"] = "ok"  # TODO: Implement actual SDK check
    
    # Determine overall status
    all_ok = all(v == "ok" for v in checks.values())
    
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks
    }


# CLI-facing deployment API endpoints
@app.get("/api/status")
async def get_status(manager: ModelManager = Depends(get_model_manager)) -> dict:
    """Runtime status for CLI"""
    uptime = datetime.utcnow() - _start_time
    models = await manager.list_models() if manager else []
    return {
        "version": "1.0.0",
        "models_loaded": len(models),
        "uptime": str(uptime),
        "deployment_mode": settings.deployment_mode
    }


@app.get("/api/models")
async def list_models(manager: ModelManager = Depends(get_model_manager)) -> dict:
    """List all deployed models"""
    models = await manager.list_models()
    return {"models": [m.model_dump() for m in models]}


@app.delete("/api/models/{model_id}")
async def delete_model(
    model_id: str,
    manager: ModelManager = Depends(get_model_manager)
) -> dict:
    """Remove a deployed model"""
    await manager.unload_model(model_id, delete_data=True)
    return {"status": "deleted", "model_id": model_id}


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
