from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..core.config import get_settings

_LOCK = threading.Lock()


def _default_payload() -> dict[str, Any]:
    return {"allowlist": [], "disabled": [], "rate_limits": {}}


def load_overrides() -> dict[str, Any]:
    settings = get_settings()
    path = Path(settings.listener_overrides_path)
    if not path.exists():
        return _default_payload()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return _default_payload()
    for key, default in _default_payload().items():
        data.setdefault(key, default)
    return data


def save_overrides(payload: dict[str, Any]) -> None:
    settings = get_settings()
    path = Path(settings.listener_overrides_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def update_listener_override(
    *,
    listener_id: str,
    enabled: bool | None = None,
    allowlisted: bool | None = None,
    throttle: int | None = None,
) -> dict[str, Any]:
    with _LOCK:
        payload = load_overrides()
        disabled = set(payload.get("disabled") or [])
        allowlist = set(payload.get("allowlist") or [])
        rate_limits = payload.get("rate_limits") or {}

        if enabled is not None:
            if enabled:
                disabled.discard(listener_id)
            else:
                disabled.add(listener_id)
        if allowlisted is not None:
            if allowlisted:
                allowlist.add(listener_id)
            else:
                allowlist.discard(listener_id)
        if throttle is None:
            rate_limits.pop(listener_id, None)
        else:
            rate_limits[listener_id] = {"throttle": max(1, int(throttle))}

        payload["disabled"] = sorted(disabled)
        payload["allowlist"] = sorted(allowlist)
        payload["rate_limits"] = rate_limits
        save_overrides(payload)
        return payload
