from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _default_fingerprint() -> str:
    seed = f"{uuid.getnode()}|{platform.node()}|{platform.system()}|{platform.machine()}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise RuntimeError(f"Activation failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Activation failed: {exc.reason}") from exc


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except PermissionError:
        pass


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except PermissionError:
        pass


def _license_expired(license_file: dict, now: dt.datetime | None = None) -> bool:
    payload = license_file.get("payload") if isinstance(license_file, dict) else None
    if not isinstance(payload, dict):
        return True
    expires_at = payload.get("expires_at")
    if not isinstance(expires_at, int):
        return True
    grace_days = payload.get("offline_grace_days")
    if grace_days is None:
        grace_days = 30
    try:
        grace_days = int(grace_days)
    except (TypeError, ValueError):
        grace_days = 0
    now_ts = int((now or dt.datetime.utcnow()).timestamp())
    return now_ts > expires_at + (grace_days * 86400)


def load_license_file(settings) -> dict | None:
    return _read_json(Path(settings.runtime_license_path))


def license_status(settings, license_file: dict | None = None) -> tuple[bool, str]:
    license_file = license_file or load_license_file(settings)
    if not license_file:
        return False, "missing"
    payload = license_file.get("payload") if isinstance(license_file, dict) else None
    if not isinstance(payload, dict):
        return False, "invalid"
    if _license_expired(license_file):
        return False, "expired"
    return True, "active"


def license_is_valid(settings, license_file: dict | None = None) -> bool:
    return license_status(settings, license_file)[0]


def _license_expires_at(license_file: dict) -> int | None:
    payload = license_file.get("payload") if isinstance(license_file, dict) else None
    if not isinstance(payload, dict):
        return None
    expires_at = payload.get("expires_at")
    return expires_at if isinstance(expires_at, int) else None


def _license_payload_value(license_file: dict, key: str) -> str | None:
    payload = license_file.get("payload") if isinstance(license_file, dict) else None
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _load_runtime_secret(path: Path) -> str | None:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _should_renew(license_file: dict, renew_days: int, now: dt.datetime | None = None) -> bool:
    expires_at = _license_expires_at(license_file)
    if expires_at is None:
        return True
    now_ts = int((now or dt.datetime.utcnow()).timestamp())
    return expires_at - now_ts <= max(renew_days, 0) * 86400


def _renew_license(settings, license_file: dict) -> dict | None:
    if not settings.platform_api_base:
        return None
    runtime_secret = _load_runtime_secret(Path(settings.runtime_secret_path))
    if not runtime_secret:
        return None
    org_id = _license_payload_value(license_file, "org_id")
    runtime_id = _license_payload_value(license_file, "runtime_id")
    if not org_id or not runtime_id:
        return None
    device_fingerprint = _license_payload_value(license_file, "device_fingerprint") or _default_fingerprint()
    renew_url = settings.platform_api_base.rstrip("/") + "/runtime/renew"
    payload = {
        "org_id": org_id,
        "runtime_id": runtime_id,
        "device_fingerprint": device_fingerprint,
        "runtime_secret": runtime_secret,
    }
    if settings.runtime_endpoint_url:
        payload["endpoint_url"] = settings.runtime_endpoint_url
    response = _post_json(renew_url, payload)
    return response if isinstance(response, dict) else None


def ensure_runtime_license(settings) -> None:
    license_path = Path(settings.runtime_license_path)
    secret_path = Path(settings.runtime_secret_path)
    license_file = _read_json(license_path)

    if license_file and not _license_expired(license_file):
        if _should_renew(license_file, settings.runtime_license_renew_days):
            renewed = _renew_license(settings, license_file)
            if renewed:
                _write_json(license_path, renewed)
        return

    if not sys.stdin.isatty():
        raise RuntimeError(
            "Runtime license missing or expired. Run the activation CLI to generate a license file."
        )

    if license_file:
        print("License expired or revoked. Enter a new activation key to continue.")
    else:
        print("License not found. Enter an activation key to continue.")

    platform_url = settings.platform_api_base or input("Platform API base URL: ").strip()
    if not platform_url:
        raise RuntimeError("Platform API base URL is required for activation.")
    org_id = input("Org ID: ").strip()
    if not org_id:
        raise RuntimeError("Org ID is required for activation.")
    runtime_id = input("Runtime ID: ").strip()
    if not runtime_id:
        raise RuntimeError("Runtime ID is required for activation.")
    activation_key = input("Activation key: ").strip()
    if not activation_key:
        raise RuntimeError("Activation key is required for activation.")
    fingerprint = _default_fingerprint()
    activation_url = platform_url.rstrip("/") + "/runtime/activate"

    payload = {
        "org_id": org_id,
        "runtime_id": runtime_id,
        "device_fingerprint": fingerprint,
        "activation_key": activation_key,
    }
    if settings.runtime_endpoint_url:
        payload["endpoint_url"] = settings.runtime_endpoint_url

    response = _post_json(activation_url, payload)
    license_file = response.get("license")
    runtime_secret = response.get("runtime_secret")
    if not license_file or not runtime_secret:
        raise RuntimeError("Activation response missing license or runtime_secret.")

    _write_json(license_path, license_file)
    _write_text(secret_path, runtime_secret)
    print(f"License saved to {license_path}")
