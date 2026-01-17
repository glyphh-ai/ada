#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Activate a Glyphh runtime via activation key.")
    parser.add_argument("--platform-url", default=os.getenv("GLYPH_PLATFORM_API_BASE"))
    parser.add_argument("--org-id", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--activation-key", required=True)
    parser.add_argument("--endpoint-url", default=None)
    parser.add_argument("--device-fingerprint", default=None)
    parser.add_argument(
        "--license-path",
        default=str(Path(".glyphh") / "license.json"),
    )
    parser.add_argument(
        "--secret-path",
        default=str(Path(".glyphh") / "runtime_secret"),
    )
    args = parser.parse_args()

    if not args.platform_url:
        print("Error: --platform-url (or GLYPH_PLATFORM_API_BASE) is required.", file=sys.stderr)
        return 2

    fingerprint = args.device_fingerprint or _default_fingerprint()
    activation_url = args.platform_url.rstrip("/") + "/runtime/activate"
    payload = {
        "org_id": args.org_id,
        "runtime_id": args.runtime_id,
        "device_fingerprint": fingerprint,
        "activation_key": args.activation_key,
    }
    if args.endpoint_url:
        payload["endpoint_url"] = args.endpoint_url

    try:
        response = _post_json(activation_url, payload)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    license_file = response.get("license")
    runtime_secret = response.get("runtime_secret")
    if not license_file or not runtime_secret:
        print("Error: Activation response missing license or runtime_secret.", file=sys.stderr)
        return 1

    license_path = Path(args.license_path)
    secret_path = Path(args.secret_path)
    _write_json(license_path, license_file)
    _write_text(secret_path, runtime_secret)

    print("Activation complete.")
    print(f"License saved to: {license_path}")
    print(f"Runtime secret saved to: {secret_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
