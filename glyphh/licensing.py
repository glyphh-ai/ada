"""
License loader for Glyphh Runtime — Ed25519 signed JWT tokens.

Resolution chain:
  1. GLYPHH_LICENSE env var (JWT token string)
  2. ~/.glyphh/license.json file ({"token": "eyJ..."})
  3. Platform self-fetch (GLYPHH_RUNTIME_ID env var)
  4. No license → free tier defaults
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

_public_key = None


def _fetch_public_key_from_platform() -> Optional[str]:
    """Fetch the Ed25519 public key PEM from the Platform API."""
    platform_url = os.environ.get(
        "GLYPHH_PLATFORM_URL", "https://api.glyphh.ai/api/v1"
    )
    try:
        import httpx

        res = httpx.get(f"{platform_url}/runtimes/public-key", timeout=10)
        if res.status_code == 200:
            pem = res.text.strip()
            if pem.startswith("-----BEGIN PUBLIC KEY-----"):
                return pem
            logger.warning("Platform returned unexpected public key format")
        else:
            logger.warning(f"Platform returned {res.status_code} for public key fetch")
    except Exception as e:
        logger.warning(f"Could not fetch public key from Platform: {e}")
    return None


def _get_public_key():
    """Load Ed25519 public key from env var or fetch from Platform."""
    global _public_key
    if _public_key is not None:
        return _public_key

    raw = os.environ.get("GLYPHH_LICENSE_PUBLIC_KEY", "").strip()
    if not raw:
        raw = _fetch_public_key_from_platform() or ""

    if not raw:
        logger.warning(
            "No license public key available — set GLYPHH_LICENSE_PUBLIC_KEY "
            "or GLYPHH_PLATFORM_URL to enable license verification"
        )
        return None

    try:
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        _public_key = load_pem_public_key(raw.encode())
    except ImportError:
        logger.warning("cryptography package not installed — license verification disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to load license public key: {e}")
        return None

    return _public_key


def _verify_token(token_str: str) -> Optional[dict]:
    """Verify a JWT license token and return claims, or None on failure."""
    pub = _get_public_key()
    if pub is None:
        logger.warning("No public key available — cannot verify license token")
        return None

    try:
        import jwt
        claims = jwt.decode(token_str, pub, algorithms=["EdDSA"])
        return claims
    except ImportError:
        logger.warning("pyjwt package not installed — license verification disabled")
        return None
    except jwt.ExpiredSignatureError:
        logger.warning("License token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid license token: {e}")
        return None


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


def _claims_to_license_info(claims: dict) -> LicenseInfo:
    """Convert verified JWT claims into a LicenseInfo."""
    tier = claims.get("tier", "free")
    defaults = _TIER_DEFAULTS.get(tier, _TIER_DEFAULTS["free"])

    info = LicenseInfo(
        org_id=claims.get("org_id", "default"),
        tier=tier,
        max_models=claims.get("max_models", defaults["max_models"]),
        max_glyphs_per_model=claims.get("max_glyphs_per_model", defaults["max_glyphs_per_model"]),
        rate_limit_per_minute=claims.get("rate_limit_per_minute", defaults["rate_limit_per_minute"]),
        license_id=claims.get("license_id"),
        issued_at=claims.get("issued_at"),
        expires_at=claims.get("expires_at"),
    )

    if info.is_expired:
        logger.warning(f"License {info.license_id} has expired ({info.expires_at}), falling back to free tier")
        return FREE_TIER

    return info


def load_license() -> LicenseInfo:
    """Load license from env var, file, or Platform. Returns free tier if not found."""

    # 1. GLYPHH_LICENSE env var (JWT token string)
    env_val = os.environ.get("GLYPHH_LICENSE", "").strip()
    if env_val:
        claims = _verify_token(env_val)
        if claims:
            info = _claims_to_license_info(claims)
            logger.info(f"License loaded from GLYPHH_LICENSE env var: tier={info.tier}, org={info.org_id}")
            return info
        logger.warning("GLYPHH_LICENSE env var contains invalid or unverifiable token")

    # 2. ~/.glyphh/license.json file ({"token": "eyJ..."})
    if LICENSE_FILE.exists():
        try:
            data = json.loads(LICENSE_FILE.read_text())
            token_str = data.get("token", "")
            if token_str:
                claims = _verify_token(token_str)
                if claims:
                    info = _claims_to_license_info(claims)
                    logger.info(f"License loaded from {LICENSE_FILE}: tier={info.tier}, org={info.org_id}")
                    return info
                logger.warning(f"License file {LICENSE_FILE} contains invalid or unverifiable token")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Invalid license file {LICENSE_FILE}: {e}")

    # 3. Platform self-fetch (GLYPHH_RUNTIME_ID env var)
    runtime_id = os.environ.get("GLYPHH_RUNTIME_ID", "").strip()
    if runtime_id:
        token_str = _fetch_token_from_platform(runtime_id)
        if token_str:
            claims = _verify_token(token_str)
            if claims:
                save_license_token(token_str)
                info = _claims_to_license_info(claims)
                logger.info(f"License fetched from Platform: tier={info.tier}, org={info.org_id}")
                return info
            logger.warning("Platform returned invalid license token")

    # 4. No license → free tier
    logger.info("No license found — using free tier defaults")
    return FREE_TIER


def _fetch_token_from_platform(runtime_id: str) -> Optional[str]:
    """Fetch signed JWT license from Platform API. Returns token string or None."""
    platform_url = os.environ.get(
        "GLYPHH_PLATFORM_URL", "https://api.glyphh.ai/api/v1"
    )
    try:
        import httpx

        res = httpx.post(
            f"{platform_url}/runtimes/validate-license",
            json={"runtime_id": runtime_id},
            timeout=10,
        )
        if res.status_code == 200:
            data = res.json()
            if data.get("valid") and data.get("token"):
                return data["token"]
            logger.warning(f"Platform license validation failed: {data.get('error', 'unknown')}")
        else:
            logger.warning(f"Platform returned {res.status_code} for license fetch")
    except Exception as e:
        logger.warning(f"Could not reach Platform for license: {e}")
    return None


def save_license_token(token: str) -> Path:
    """Save a JWT license token to ~/.glyphh/license.json."""
    LICENSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LICENSE_FILE.write_text(json.dumps({"token": token}, indent=2))
    return LICENSE_FILE


def remove_license() -> bool:
    """Remove the license file. Returns True if file existed."""
    if LICENSE_FILE.exists():
        LICENSE_FILE.unlink()
        return True
    return False
