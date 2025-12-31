from __future__ import annotations

import base64
import json
import logging
import threading
import time
import datetime as dt
import hashlib
import hmac
import urllib.request
from dataclasses import dataclass
from typing import Dict


logger = logging.getLogger("glyphh.runtime.usage")


@dataclass
class RuntimeIdentity:
    org_id: str
    runtime_id: str
    model_id: str


def _decode_jwt(token: str, secret: str, algorithm: str) -> dict:
    if algorithm != "HS256":
        raise ValueError("Unsupported token algorithm")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid token")
    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode()
    sig = base64.urlsafe_b64decode(signature_b64 + "==")
    expected = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise ValueError("bad signature")
    payload_json = base64.urlsafe_b64decode(payload_b64 + "==")
    return json.loads(payload_json)


class UsageTracker:
    def __init__(
        self,
        *,
        platform_api_base: str,
        runtime_token: str,
        runtime_version: str,
        jwt_secret: str,
        jwt_algorithm: str,
        flush_seconds: int = 60,
    ) -> None:
        self._platform_api_base = platform_api_base.rstrip("/")
        self._runtime_token = runtime_token
        self._runtime_version = runtime_version
        self._jwt_secret = jwt_secret
        self._jwt_algorithm = jwt_algorithm
        self._flush_seconds = max(10, flush_seconds)
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {}
        self._window_start = dt.datetime.utcnow()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._identity = self._parse_identity()

    def _parse_identity(self) -> RuntimeIdentity:
        payload = _decode_jwt(self._runtime_token, self._jwt_secret, self._jwt_algorithm)
        if payload.get("token_type") != "runtime":
            raise ValueError("runtime token required for usage tracking")
        now = int(time.time())
        exp = payload.get("exp")
        if isinstance(exp, int) and exp < now:
            raise ValueError("runtime token expired")
        runtime_id = payload.get("runtime_id")
        model_id = payload.get("model_id")
        org_id = payload.get("org_id") or payload.get("org")
        if not runtime_id or not model_id or not org_id:
            raise ValueError("runtime token missing identity claims")
        return RuntimeIdentity(
            org_id=str(org_id),
            runtime_id=str(runtime_id),
            model_id=str(model_id),
        )

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def record(self, event_type: str, count: int = 1) -> None:
        if count <= 0:
            return
        with self._lock:
            self._counts[event_type] = self._counts.get(event_type, 0) + count

    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(self._flush_seconds)
            try:
                self.flush()
            except Exception:
                logger.exception("usage metrics flush failed")

    def flush(self) -> None:
        with self._lock:
            if not self._counts:
                self._window_start = dt.datetime.utcnow()
                return
            snapshot = dict(self._counts)
            self._counts.clear()
            window_start = self._window_start
            window_end = dt.datetime.utcnow()
            self._window_start = window_end

        metrics = [
            {
                "event_type": event_type,
                "count": count,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
            }
            for event_type, count in snapshot.items()
        ]
        payload = json.dumps(
            {
                "runtime_version": self._runtime_version,
                "metrics": metrics,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._platform_api_base}/runtime/usage",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._runtime_token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
