"""
License file loader for Glyphh Runtime.

Resolution chain:
  1. GLYPHH_LICENSE env var (JSON string)
  2. ~/.glyphh/license.json file
  3. No license → free tier defaults
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# License file location
LICENSE_FILE = Path.home() / ".glyphh" / "license.json"


@dataclass
class LicenseInfo:
    """License information determining tier and limits."""

    org_id: str = "default"
    tier: str = "free"
    max_models: int = 3
    max_glyphs_per_model: int = 10_000
    rate_limit_per_minute: int = 60
    license_id: Optional[str] = None
    issued_at: Optional[str] = None
    expires_at: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.now(exp.tzinfo) > exp
        except (ValueError, TypeError):
            return False

    @property
    def is_free(self) -> bool:
        return self.tier == "free"

    def check_model_limit(self, current_count: int) -> bool:
        """True if another model can be created."""
        return self.max_models == -1 or current_count < self.max_models

    def check_glyph_limit(self, current_count: int) -> bool:
        """True if another glyph can be written."""
        return self.max_glyphs_per_model == -1 or current_count < self.max_glyphs_per_model


# Tier presets
FREE_TIER = LicenseInfo()

_TIER_DEFAULTS = {
    "free": {"max_models": 3, "max_glyphs_per_model": 10_000, "rate_limit_per_minute": 60},
    "advanced": {"max_models": 10, "max_glyphs_per_model": 250_000, "rate_limit_per_minute": 300},
    "pro": {"max_models": -1, "max_glyphs_per_model": -1, "rate_limit_per_minute": 1_000},
    "enterprise": {"max_models": -1, "max_glyphs_per_model": -1, "rate_limit_per_minute": -1},
}


def load_license() -> LicenseInfo:
    """Load license from env var or file. Returns free tier if not found."""

    # 1. GLYPHH_LICENSE env var (JSON string)
    env_val = os.environ.get("GLYPHH_LICENSE", "").strip()
    if env_val:
        try:
            data = json.loads(env_val)
            info = _parse_license(data)
            logger.info(f"License loaded from GLYPHH_LICENSE env var: tier={info.tier}, org={info.org_id}")
            return info
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Invalid GLYPHH_LICENSE env var: {e}")

    # 2. ~/.glyphh/license.json file
    if LICENSE_FILE.exists():
        try:
            data = json.loads(LICENSE_FILE.read_text())
            info = _parse_license(data)
            logger.info(f"License loaded from {LICENSE_FILE}: tier={info.tier}, org={info.org_id}")
            return info
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Invalid license file {LICENSE_FILE}: {e}")

    # 3. No license → free tier
    logger.info("No license found — using free tier defaults")
    return FREE_TIER


def _parse_license(data: dict) -> LicenseInfo:
    """Parse a license dict into LicenseInfo, applying tier defaults for missing limits."""
    tier = data.get("tier", "free")
    defaults = _TIER_DEFAULTS.get(tier, _TIER_DEFAULTS["free"])

    info = LicenseInfo(
        org_id=data.get("org_id", "default"),
        tier=tier,
        max_models=data.get("max_models", defaults["max_models"]),
        max_glyphs_per_model=data.get("max_glyphs_per_model", defaults["max_glyphs_per_model"]),
        rate_limit_per_minute=data.get("rate_limit_per_minute", defaults["rate_limit_per_minute"]),
        license_id=data.get("license_id"),
        issued_at=data.get("issued_at"),
        expires_at=data.get("expires_at"),
    )

    if info.is_expired:
        logger.warning(f"License {info.license_id} has expired ({info.expires_at}), falling back to free tier")
        return FREE_TIER

    return info


def save_license(data: dict) -> Path:
    """Save license data to ~/.glyphh/license.json."""
    LICENSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LICENSE_FILE.write_text(json.dumps(data, indent=2))
    return LICENSE_FILE


def remove_license() -> bool:
    """Remove the license file. Returns True if file existed."""
    if LICENSE_FILE.exists():
        LICENSE_FILE.unlink()
        return True
    return False
