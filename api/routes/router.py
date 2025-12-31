from __future__ import annotations

from fastapi import APIRouter


def api_router(*, prefix: str = "", tags: list[str] | None = None) -> APIRouter:
    return APIRouter(prefix=prefix, tags=tags or [])
