from __future__ import annotations

import json
import logging
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Request

from ..core.config import get_settings


logger = logging.getLogger("glyphh.runtime.audit")
_lock = threading.Lock()


def _rotate_if_needed(path: Path, max_bytes: int) -> None:
    if not path.exists() or max_bytes <= 0:
        return
    if path.stat().st_size < max_bytes:
        return
    rotated = path.with_suffix(path.suffix + ".1")
    if rotated.exists():
        rotated.unlink()
    path.rename(rotated)


def _write_jsonl(path: Path, payload: str, max_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed(path, max_bytes)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")


def _drain_event(url: str, payload: str) -> None:
    req = urllib.request.Request(
        url,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()


def log_audit_event(event: dict[str, Any]) -> None:
    settings = get_settings()
    payload = dict(event)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    payload.setdefault("runtime_version", settings.runtime_version)
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    with _lock:
        _write_jsonl(Path(settings.audit_log_path), payload_json, settings.audit_log_max_bytes)
    if settings.audit_log_drain_url:
        try:
            _drain_event(settings.audit_log_drain_url, payload_json)
        except Exception:
            logger.exception("audit log drain failed")


def log_tool_access_decision(
    *,
    request: Request,
    tool: str,
    decision: str,
    reason: str | None = None,
    model_id: str | None = None,
) -> None:
    claims = getattr(request.state, "runtime_claims", {}) or {}
    event = {
        "event": "tool_access",
        "decision": decision,
        "tool": tool,
        "reason": reason,
        "model_id": model_id,
        "subject": claims.get("sub") or claims.get("user_id"),
        "runtime_id": claims.get("runtime_id"),
        "org_id": claims.get("org_id") or claims.get("org"),
        "scopes": claims.get("scopes") or claims.get("scope"),
        "request_id": request.headers.get("x-request-id"),
        "client_ip": request.client.host if request.client else None,
        "path": request.url.path,
        "method": request.method,
    }
    log_audit_event(event)
