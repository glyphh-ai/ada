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
from shared.sdk_adapter import get_sdk_adapter
from shared.config_validator import get_config_validator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Liveness probe.
    
    Returns healthy if the service is running.
    Includes SDK compatibility status.
    """
    # Get SDK compatibility status
    adapter = get_sdk_adapter()
    sdk_status = adapter.get_compatibility_status()
    
    # Determine overall status based on SDK availability
    status = "healthy"
    if sdk_status["degraded"]:
        status = "degraded"
    
    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sdk_compatibility": sdk_status,
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
    
    # Check SDK compatibility
    adapter = get_sdk_adapter()
    sdk_status = adapter.get_compatibility_status()
    
    if sdk_status["available"]:
        if sdk_status["degraded"]:
            checks["sdk"] = "degraded"
        else:
            checks["sdk"] = "ok"
    else:
        checks["sdk"] = "unavailable"
    
    # Determine overall status
    # Consider degraded SDK as acceptable (not a failure)
    critical_checks = ["database", "model_manager", "license"]
    critical_ok = all(checks.get(k) == "ok" for k in critical_checks)
    sdk_acceptable = checks["sdk"] in ("ok", "degraded")
    
    if critical_ok and sdk_acceptable:
        status = "ready" if checks["sdk"] == "ok" else "degraded"
    else:
        status = "not_ready"
    
    return {
        "status": status,
        "checks": checks,
        "sdk_compatibility": sdk_status,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Prometheus-compatible metrics endpoint.
    
    Returns runtime metrics for monitoring.
    """
    from main import _start_time, model_manager, resource_manager
    
    uptime_seconds = (datetime.utcnow() - _start_time).total_seconds()
    models_loaded = 0
    
    if model_manager:
        try:
            models = await model_manager.list_models()
            models_loaded = len(models)
        except Exception:
            pass
    
    # Get resource usage
    resource_metrics = {}
    if resource_manager:
        try:
            system_usage = await resource_manager.get_system_usage()
            resource_metrics = {
                "runtime_memory_mb": system_usage["process"]["memory_mb"],
                "runtime_memory_percent": system_usage["process"]["memory_percent"],
                "runtime_cpu_percent": system_usage["process"]["cpu_percent"],
                "runtime_total_glyphs": system_usage["models"]["total_glyphs"],
                "runtime_total_edges": system_usage["models"]["total_edges"],
                "runtime_models_count": system_usage["models"]["count"],
            }
        except Exception as e:
            logger.warning(f"Failed to get resource metrics: {e}")
    
    # Return metrics in a simple format
    # TODO: Implement proper Prometheus format
    return {
        "runtime_uptime_seconds": uptime_seconds,
        "runtime_models_loaded": models_loaded,
        "runtime_requests_total": 0,  # TODO: Track requests
        "runtime_errors_total": 0,    # TODO: Track errors
        **resource_metrics,
    }


@router.get("/resources")
async def get_resource_usage() -> Dict[str, Any]:
    """
    Get detailed resource usage for all models.
    
    Returns memory, storage, and glyph counts per org/model.
    """
    from main import resource_manager
    
    if resource_manager is None:
        return {"error": "Resource manager not initialized"}
    
    try:
        system_usage = await resource_manager.get_system_usage()
        all_usage = await resource_manager.get_all_usage()
        
        return {
            "system": system_usage,
            "models": [
                {
                    "org_id": u.org_id,
                    "model_id": u.model_id,
                    "memory_mb": u.memory_mb,
                    "storage_mb": u.storage_mb,
                    "glyph_count": u.glyph_count,
                    "edge_count": u.edge_count,
                    "last_updated": u.last_updated.isoformat() + "Z",
                }
                for u in all_usage
            ],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(f"Failed to get resource usage: {e}")
        return {"error": str(e)}
