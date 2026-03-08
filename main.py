"""
Glyphh Runtime - Execution environment for directory-based models

This runtime serves deployed models through REST and MCP APIs,
handling persistent storage, multi-model management, licensing,
authentication, and real-time data ingestion.

Models are deployed from .glyphh packages or custom_models/ directories containing
manifest.yaml, encoder.py, and data/*.jsonl files. On startup, the
runtime discovers and deploys eligible models to PostgreSQL.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from infrastructure.config import get_settings, validate_settings
from infrastructure.database import init_db, close_db, async_session_maker
from shared.exceptions import GlyphhRuntimeException
from glyphh.licensing import load_license, set_current_license
from shared.middleware import (
    CorrelationIDMiddleware,
    LoggingMiddleware,
)
from domains.models.manager import ModelManager
from domains.resources.manager import ResourceManager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

settings = get_settings()
_start_time = datetime.utcnow()

# Global model manager instance
model_manager: Optional[ModelManager] = None
resource_manager: Optional[ResourceManager] = None


def get_model_manager() -> ModelManager:
    """Dependency for getting model manager"""
    return model_manager


def get_resource_manager() -> ResourceManager:
    """Dependency for getting resource manager"""
    return resource_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management"""
    global model_manager, resource_manager
    
    # Startup
    logger.info("Starting Glyphh Runtime...")
    
    # Validate configuration
    try:
        validate_settings()
        logger.info(f"Configuration validated for {settings.deployment_mode} mode")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    
    # Load license (determines tier and limits)
    license_info = load_license()
    app.state.license = license_info
    set_current_license(license_info)
    logger.info(f"License: tier={license_info.tier}, org={license_info.org_id}")

    await init_db()
    logger.info("Database initialized")
    
    # Initialize model manager
    model_manager = ModelManager(async_session_maker)
    logger.info("Model manager initialized")

    # Initialize resource manager
    resource_manager = ResourceManager(async_session_maker)
    logger.info("Resource manager initialized")

    # Sync globals to glyphh.server so route handlers can find them
    # (routes import from glyphh.server, but uvicorn loads main.py)
    import glyphh.server as _srv
    _srv.model_manager = model_manager
    _srv.resource_manager = resource_manager
    
    # Auto-deploy models from directory (JSONL → DB)
    try:
        from scripts.deploy_models import deploy_all_models
        results = await deploy_all_models(
            model_manager=model_manager,
            session_factory=async_session_maker,
        )
        total_glyphs = sum(results.values())
        logger.info(f"Deployed {len(results)} models, {total_glyphs} total glyphs")
    except Exception as e:
        logger.debug(f"Model auto-deploy skipped: {e}")

    # Resume any incomplete staged exemplar encoding from a previous run
    try:
        await model_manager.resume_staged_encoding()
    except Exception as e:
        logger.warning(f"Staged encoding resume failed: {e}")

    yield
    
    # Graceful shutdown
    logger.info("Shutting down Glyphh Runtime...")
    
    # Allow in-flight requests to complete (30s timeout handled by uvicorn)
    logger.info("Draining connections...")
    
    # Close database connections
    await close_db()
    logger.info("Database connections closed")
    
    # Flush logs
    logging.shutdown()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Glyphh Runtime",
    description="Execution environment for directory-based models",
    version="0.2.11",
    docs_url="/docs" if settings.deployment_mode == "local" else None,
    redoc_url="/redoc" if settings.deployment_mode == "local" else None,
    lifespan=lifespan,
)

# CORS middleware — wildcard in local mode, explicit origins in production
if settings.deployment_mode == "local":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    origins = settings.cors_origins_production or settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
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
    """Handle custom runtime exceptions with CORS headers"""
    response = JSONResponse(
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
    # Add CORS headers to error responses for browser compatibility
    origin = request.headers.get("origin")
    if origin and _is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with CORS headers"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    response = JSONResponse(
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
    # Add CORS headers to error responses for browser compatibility
    origin = request.headers.get("origin")
    if origin and _is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


def _is_allowed_origin(origin: str) -> bool:
    """Check if origin is in allowed CORS origins list."""
    # Local mode allows everything
    if settings.deployment_mode == "local":
        return True
    import fnmatch
    origins = settings.cors_origins_production or settings.cors_origins
    for allowed in origins:
        if allowed == "*" or allowed == origin:
            return True
        if fnmatch.fnmatch(origin, allowed):
            return True
    return False


# Import and include routers
from api.routes import (
    health_router,
    org_scoped_router,
    org_level_router,
    listeners_router,
    tokens_router,
)

app.include_router(health_router)
app.include_router(tokens_router)
# listeners_router must come before org_scoped_router (more specific prefix)
app.include_router(listeners_router)
# org_level_router (/{org_id}/models) before org_scoped_router (/{org_id}/{model_id}/...)
app.include_router(org_level_router)
app.include_router(org_scoped_router)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.deployment_mode == "local",
        log_level="info"
    )
