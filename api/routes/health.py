"""
Health Check API Routes for Glyphh Runtime.

Endpoints for liveness and readiness probes.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Liveness probe.
    
    Returns healthy if the service is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.get("/health/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Readiness probe.
    
    Checks all dependencies (database, license, SDK) and returns
    ready if all are operational.
    """
    checks = {}
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        checks["database"] = "error"
    
    # Check model manager
    try:
        from main import model_manager
        if model_manager is not None:
            checks["model_manager"] = "ok"
        else:
            checks["model_manager"] = "not_initialized"
    except Exception as e:
        checks["model_manager"] = "error"
    
    # Check license (placeholder)
    checks["license"] = "ok"
    
    # Determine overall status
    all_ok = all(v == "ok" for v in checks.values())
    
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Prometheus-compatible metrics endpoint.
    
    Returns runtime metrics for monitoring.
    """
    from main import _start_time, model_manager
    
    uptime_seconds = (datetime.utcnow() - _start_time).total_seconds()
    models_loaded = 0
    
    if model_manager:
        try:
            models = await model_manager.list_models()
            models_loaded = len(models)
        except Exception:
            pass
    
    # Return metrics in a simple format
    # TODO: Implement proper Prometheus format
    return {
        "runtime_uptime_seconds": uptime_seconds,
        "runtime_models_loaded": models_loaded,
        "runtime_requests_total": 0,  # TODO: Track requests
        "runtime_errors_total": 0,    # TODO: Track errors
    }
