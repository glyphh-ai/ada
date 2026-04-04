"""
Health Check API Routes for Glyphh Runtime.

Single liveness probe endpoint.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter

from shared.sdk_adapter import get_sdk_adapter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Liveness probe.
    
    Returns healthy if the service is running.
    Includes SDK compatibility status.
    """
    adapter = get_sdk_adapter()
    sdk_status = adapter.get_compatibility_status()
    
    status = "healthy"
    if sdk_status["degraded"]:
        status = "degraded"
    
    # License and usage info
    from glyphh.licensing import get_current_license
    from glyphh.metering import get_meter

    license_info = get_current_license()
    meter = get_meter()
    usage = meter.get_usage(license_info.org_id)

    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sdk_compatibility": sdk_status,
        "license": {
            "tier": license_info.tier,
            "org_id": license_info.org_id,
            "max_encodings_per_month": license_info.max_encodings_per_month,
            "max_runtimes": license_info.max_runtimes,
        },
        "usage": {
            "operations_this_month": usage,
            "limit": license_info.max_encodings_per_month,
        },
    }
