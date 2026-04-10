"""
Ada license validation — local file-based licensing.

Flow:
  1. User buys at glyphh.ai
  2. Platform generates a signed license file
  3. User downloads it (or `ada setup license` fetches it)
  4. Ada validates on boot — no phone-home needed

License file: ~/.ada/license.json
  {
    "id": "lic_abc123",
    "email": "user@example.com",
    "issued_at": "2026-04-09T00:00:00Z",
    "expires_at": null,           # null = perpetual
    "trial_expires_at": "2026-04-23T00:00:00Z",  # only for trial
    "plan": "pro",                # "trial", "pro", "enterprise"
    "signature": "base64..."      # HMAC-SHA256 of the payload
  }

Trial: 14 days from first run. No license file needed.
Paid: license file required. No expiry (perpetual).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ADA_HOME = Path.home() / ".ada"
LICENSE_FILE = ADA_HOME / "license.json"
TRIAL_FILE = ADA_HOME / ".trial"

TRIAL_DAYS = 14

# Public key for signature verification — shipped with Ada.
# The platform signs with the private key; Ada verifies with this.
# Using HMAC-SHA256 for simplicity. Upgrade to RSA/Ed25519 for production.
LICENSE_VERIFY_KEY = os.environ.get(
    "ADA_LICENSE_KEY",
    "ada-glyphh-2026-cognitive-brain-license-verification"
)


@dataclass
class LicenseStatus:
    valid: bool
    plan: str = "trial"          # trial, pro, enterprise
    days_remaining: Optional[int] = None  # None = perpetual
    email: Optional[str] = None
    message: str = ""


def check_license() -> LicenseStatus:
    """Check Ada's license status. Called on every boot.

    Priority:
      1. Valid license file → perpetual access
      2. Trial period → 14 days from first run
      3. Expired → prompt to purchase
    """
    ADA_HOME.mkdir(exist_ok=True)

    # 1. Check for license file
    if LICENSE_FILE.exists():
        status = _validate_license_file()
        if status.valid:
            return status

    # 2. Check trial
    return _check_trial()


def _validate_license_file() -> LicenseStatus:
    """Validate the license file signature and contents."""
    try:
        data = json.loads(LICENSE_FILE.read_text())

        # Verify signature
        payload = {k: v for k, v in data.items() if k != "signature"}
        payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected_sig = hmac.new(
            LICENSE_VERIFY_KEY.encode(),
            payload_str.encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(data.get("signature", ""), expected_sig):
            return LicenseStatus(valid=False, message="Invalid license signature")

        plan = data.get("plan", "pro")
        email = data.get("email")

        # Check expiry (if set)
        expires = data.get("expires_at")
        if expires:
            exp_dt = datetime.fromisoformat(expires.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if now > exp_dt:
                days = (now - exp_dt).days
                return LicenseStatus(
                    valid=False,
                    plan=plan,
                    email=email,
                    message=f"License expired {days} days ago",
                )
            days_remaining = (exp_dt - now).days
            return LicenseStatus(
                valid=True, plan=plan, email=email,
                days_remaining=days_remaining,
                message=f"{plan} license, {days_remaining} days remaining",
            )

        # Perpetual license
        return LicenseStatus(
            valid=True, plan=plan, email=email,
            days_remaining=None,
            message=f"{plan} license (perpetual)",
        )

    except Exception as e:
        logger.warning(f"License validation failed: {e}")
        return LicenseStatus(valid=False, message=f"License error: {e}")


def _check_trial() -> LicenseStatus:
    """Check or start the 14-day trial."""
    try:
        if TRIAL_FILE.exists():
            started = float(TRIAL_FILE.read_text().strip())
        else:
            # First run — start trial
            started = time.time()
            TRIAL_FILE.write_text(str(started))
            logger.info("Trial started")

        elapsed_days = (time.time() - started) / 86400
        remaining = int(TRIAL_DAYS - elapsed_days)

        if remaining > 0:
            return LicenseStatus(
                valid=True,
                plan="trial",
                days_remaining=remaining,
                message=f"Trial: {remaining} days remaining",
            )
        else:
            return LicenseStatus(
                valid=False,
                plan="trial",
                days_remaining=0,
                message="Trial expired. Purchase at https://glyphh.ai",
            )

    except Exception as e:
        logger.warning(f"Trial check failed: {e}")
        # Fail open on first run
        return LicenseStatus(valid=True, plan="trial", days_remaining=TRIAL_DAYS)


def install_license(license_path: str) -> LicenseStatus:
    """Install a license file from the given path."""
    import shutil

    src = Path(license_path).expanduser()
    if not src.exists():
        return LicenseStatus(valid=False, message=f"File not found: {license_path}")

    ADA_HOME.mkdir(exist_ok=True)
    shutil.copy2(src, LICENSE_FILE)
    os.chmod(LICENSE_FILE, 0o600)

    return _validate_license_file()
