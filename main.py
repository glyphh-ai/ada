"""Thin shim — the real app lives in glyphh.server."""

from glyphh.server import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    from infrastructure.config import get_settings

    settings = get_settings()
    uvicorn.run("glyphh.server:app", host=settings.host, port=settings.port, log_level="info")
