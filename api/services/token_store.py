from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from ..core.config import get_settings

_LOCK = threading.Lock()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"tokens": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "tokens" not in data:
        return {"tokens": []}
    return data


def _save_store(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def create_token(name: str | None, scopes: list[str]) -> dict[str, Any]:
    settings = get_settings()
    store_path = Path(settings.api_token_store_path)
    token_value = secrets.token_urlsafe(32)
    token_id = secrets.token_hex(8)
    token_hash = _hash_token(token_value)
    record = {
        "id": token_id,
        "name": name or "token",
        "hash": token_hash,
        "scopes": scopes,
        "created_at": _now_iso(),
        "revoked": False,
    }
    with _LOCK:
        store = _load_store(store_path)
        store["tokens"].append(record)
        _save_store(store_path, store)
    return {"id": token_id, "token": token_value, "scopes": scopes, "created_at": record["created_at"]}


def list_tokens() -> list[dict[str, Any]]:
    settings = get_settings()
    store_path = Path(settings.api_token_store_path)
    with _LOCK:
        store = _load_store(store_path)
    items = []
    for token in store.get("tokens", []):
        items.append(
            {
                "id": token.get("id"),
                "name": token.get("name"),
                "scopes": token.get("scopes") or [],
                "created_at": token.get("created_at"),
                "revoked": bool(token.get("revoked")),
            }
        )
    return items


def revoke_token(token_id: str) -> bool:
    settings = get_settings()
    store_path = Path(settings.api_token_store_path)
    with _LOCK:
        store = _load_store(store_path)
        updated = False
        for token in store.get("tokens", []):
            if token.get("id") == token_id:
                token["revoked"] = True
                updated = True
                break
        if updated:
            _save_store(store_path, store)
        return updated


def verify_token(token_value: str) -> dict[str, Any] | None:
    settings = get_settings()
    store_path = Path(settings.api_token_store_path)
    token_hash = _hash_token(token_value)
    with _LOCK:
        store = _load_store(store_path)
    for token in store.get("tokens", []):
        if token.get("revoked"):
            continue
        if token.get("hash") == token_hash:
            return token
    return None
