from __future__ import annotations

from .router import api_router


router = api_router(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
