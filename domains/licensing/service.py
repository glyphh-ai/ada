"""
Licensing Service for Glyphh Runtime.

Handles license validation for offline (self-hosted) and cloud deployments.
Supports grace periods and periodic revalidation.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

import httpx

from infrastructure.config import get_settings
from shared.exceptions import LicenseException, LicenseExpiredException

logger = logging.getLogger(__name__)


class LicenseStatus(str, Enum):
    """License validation status."""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    GRACE_PERIOD = "grace_period"
    UNKNOWN = "unknown"


@dataclass
class LicenseInfo:
    """License information."""
    status: LicenseStatus
    license_key: Optional[str] = None
    org_id: Optional[str] = None
    org_name: Optional[str] = None
    tier: str = "free"
    features: Dict[str, bool] = None
    expires_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None
    grace_period_ends_at: Optional[datetime] = None
    message: Optional[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = {}
    
    def is_valid(self) -> bool:
        """Check if license allows operation."""
        return self.status in (LicenseStatus.VALID, LicenseStatus.GRACE_PERIOD)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "status": self.status.value,
            "org_id": self.org_id,
            "org_name": self.org_name,
            "tier": self.tier,
            "features": self.features,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "grace_period_ends_at": self.grace_period_ends_at.isoformat() if self.grace_period_ends_at else None,
            "message": self.message,
        }


class LicensingService:
    """
    License validation service.
    
    Responsibilities:
    - Validate licenses on startup
    - Periodic revalidation (every 24 hours for self-hosted)
    - Grace period handling when validation fails
    - Feature flag checking based on license tier
    """
    
    # Revalidation interval (24 hours)
    REVALIDATION_INTERVAL = timedelta(hours=24)
    
    def __init__(self):
        """Initialize LicensingService."""
        self._settings = get_settings()
        self._license_info: Optional[LicenseInfo] = None
        self._revalidation_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def startup(self) -> LicenseInfo:
        """
        Validate license on startup.
        
        Returns:
            LicenseInfo with validation result
            
        Raises:
            LicenseException: If license is invalid and no grace period
        """
        logger.info(f"Validating license for {self._settings.deployment_mode} deployment")
        
        # Local mode: no license required
        if self._settings.deployment_mode == "local":
            self._license_info = LicenseInfo(
                status=LicenseStatus.VALID,
                tier="local",
                features={"all": True},
                validated_at=datetime.utcnow(),
                message="Local development mode - no license required",
            )
            return self._license_info
        
        # Cloud mode: internal validation
        if self._settings.deployment_mode == "cloud":
            self._license_info = await self._validate_cloud_license()
            return self._license_info
        
        # Self-hosted mode: call Platform API
        self._license_info = await self._validate_offline_license()
        
        # Start periodic revalidation
        if self._license_info.is_valid():
            self._start_revalidation_task()
        
        return self._license_info
    
    async def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        if self._revalidation_task:
            self._revalidation_task.cancel()
            try:
                await self._revalidation_task
            except asyncio.CancelledError:
                pass
        
        if self._http_client:
            await self._http_client.aclose()
    
    async def validate_license(self) -> LicenseInfo:
        """
        Validate license based on deployment mode.
        
        Returns:
            LicenseInfo with current status
        """
        if self._license_info is None:
            return await self.startup()
        return self._license_info
    
    async def _validate_offline_license(self) -> LicenseInfo:
        """
        Validate license for self-hosted deployment by calling Platform API.
        
        Returns:
            LicenseInfo with validation result
        """
        if not self._settings.license_key:
            return LicenseInfo(
                status=LicenseStatus.INVALID,
                message="No license key configured",
            )
        
        try:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(timeout=30.0)
            
            response = await self._http_client.post(
                f"{self._settings.platform_api_url}/v1/licenses/validate",
                json={
                    "license_key": self._settings.license_key,
                    "runtime_version": "1.0.0",  # TODO: Get from package
                    "deployment_mode": "self-hosted",
                },
                headers={"Content-Type": "application/json"},
            )
            
            if response.status_code == 200:
                data = response.json()
                return LicenseInfo(
                    status=LicenseStatus.VALID,
                    license_key=self._settings.license_key,
                    org_id=data.get("org_id"),
                    org_name=data.get("org_name"),
                    tier=data.get("tier", "standard"),
                    features=data.get("features", {}),
                    expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                    validated_at=datetime.utcnow(),
                    message="License validated successfully",
                )
            elif response.status_code == 401:
                return LicenseInfo(
                    status=LicenseStatus.INVALID,
                    message="Invalid license key",
                )
            elif response.status_code == 402:
                data = response.json()
                return LicenseInfo(
                    status=LicenseStatus.EXPIRED,
                    expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                    message="License has expired",
                )
            else:
                logger.warning(f"Unexpected response from Platform API: {response.status_code}")
                return await self._enter_grace_period("Platform API returned unexpected status")
                
        except httpx.RequestError as e:
            logger.warning(f"Failed to reach Platform API: {e}")
            return await self._enter_grace_period(f"Network error: {e}")
        except Exception as e:
            logger.error(f"License validation error: {e}")
            return await self._enter_grace_period(f"Validation error: {e}")
    
    async def _validate_cloud_license(self) -> LicenseInfo:
        """
        Validate license for cloud deployment (internal validation).
        
        Returns:
            LicenseInfo with validation result
        """
        # In cloud mode, the runtime is managed by Glyphh
        # License is validated internally without external API calls
        return LicenseInfo(
            status=LicenseStatus.VALID,
            tier="cloud",
            features={"all": True},
            validated_at=datetime.utcnow(),
            message="Cloud deployment - license managed by Glyphh",
        )
    
    async def _enter_grace_period(self, reason: str) -> LicenseInfo:
        """
        Enter grace period when license validation fails.
        
        Args:
            reason: Reason for entering grace period
            
        Returns:
            LicenseInfo with grace period status
        """
        grace_period_days = self._settings.license_grace_period_days
        grace_period_ends = datetime.utcnow() + timedelta(days=grace_period_days)
        
        # Check if we're already in grace period
        if self._license_info and self._license_info.status == LicenseStatus.GRACE_PERIOD:
            # Keep existing grace period end time
            if self._license_info.grace_period_ends_at:
                if datetime.utcnow() > self._license_info.grace_period_ends_at:
                    # Grace period expired
                    return LicenseInfo(
                        status=LicenseStatus.INVALID,
                        message=f"Grace period expired. {reason}",
                    )
                grace_period_ends = self._license_info.grace_period_ends_at
        
        logger.warning(
            f"Entering grace period until {grace_period_ends.isoformat()}. Reason: {reason}"
        )
        
        return LicenseInfo(
            status=LicenseStatus.GRACE_PERIOD,
            license_key=self._settings.license_key,
            validated_at=datetime.utcnow(),
            grace_period_ends_at=grace_period_ends,
            message=f"Operating in grace period. {reason}",
        )

    def _start_revalidation_task(self) -> None:
        """Start background task for periodic license revalidation."""
        if self._revalidation_task is not None:
            return
        
        self._revalidation_task = asyncio.create_task(
            self._periodic_revalidation(),
            name="license_revalidation"
        )
        logger.info("Started periodic license revalidation task")
    
    async def _periodic_revalidation(self) -> None:
        """
        Periodically revalidate license.
        
        Runs every 24 hours (configurable) to ensure license is still valid.
        """
        while True:
            try:
                await asyncio.sleep(self.REVALIDATION_INTERVAL.total_seconds())
                
                logger.info("Performing periodic license revalidation")
                
                if self._settings.deployment_mode == "self-hosted":
                    new_info = await self._validate_offline_license()
                else:
                    new_info = await self._validate_cloud_license()
                
                self._license_info = new_info
                
                if not new_info.is_valid():
                    logger.error(f"License revalidation failed: {new_info.message}")
                    # Don't raise - let grace period handle it
                else:
                    logger.info("License revalidation successful")
                    
            except asyncio.CancelledError:
                logger.info("License revalidation task cancelled")
                break
            except Exception as e:
                logger.error(f"Error during license revalidation: {e}")
                # Enter grace period on error
                self._license_info = await self._enter_grace_period(str(e))
    
    def get_license_info(self) -> Optional[LicenseInfo]:
        """Get current license information."""
        return self._license_info
    
    def check_feature(self, feature: str) -> bool:
        """
        Check if a feature is enabled by the license.
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature is enabled
        """
        if self._license_info is None:
            return False
        
        if not self._license_info.is_valid():
            return False
        
        # Check for "all" feature flag (local/cloud modes)
        if self._license_info.features.get("all", False):
            return True
        
        return self._license_info.features.get(feature, False)
    
    def require_valid_license(self) -> None:
        """
        Require a valid license to proceed.
        
        Raises:
            LicenseException: If license is invalid
            LicenseExpiredException: If license has expired
        """
        if self._license_info is None:
            raise LicenseException("License not validated")
        
        if self._license_info.status == LicenseStatus.INVALID:
            raise LicenseException(self._license_info.message or "Invalid license")
        
        if self._license_info.status == LicenseStatus.EXPIRED:
            raise LicenseExpiredException(
                self._license_info.expires_at.isoformat() if self._license_info.expires_at else "unknown"
            )
        
        if self._license_info.status == LicenseStatus.GRACE_PERIOD:
            # Check if grace period has expired
            if self._license_info.grace_period_ends_at:
                if datetime.utcnow() > self._license_info.grace_period_ends_at:
                    raise LicenseException("Grace period has expired")
            
            # Log warning but allow operation
            logger.warning("Operating in license grace period")
