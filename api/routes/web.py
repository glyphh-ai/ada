"""
Web UI routes — login page, dashboard, static assets.

Serves the Glyphh dashboard as a vanilla JS SPA.
Local mode skips login; non-local requires Platform authentication.
"""

import base64
import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader

from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["web"])
settings = get_settings()

# Jinja2 setup
_WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"
_TEMPLATES_DIR = _WEB_DIR / "templates"
_PUBLIC_DIR = Path(__file__).resolve().parent.parent.parent / "public"

_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=True,
)

# Pre-load assets as base64 (once at import time)
def _load_b64(filename: str) -> str | None:
    p = _PUBLIC_DIR / filename
    if p.exists():
        try:
            return base64.b64encode(p.read_bytes()).decode("ascii")
        except Exception:
            pass
    return None

_LOGO_B64 = _load_b64("glyphh-logo.png")
_FAVICON_B64 = _load_b64("favicon.png")

PLATFORM_URL = "https://api.glyphh.ai/api/v1"


def _get_platform_url() -> str:
    """Platform API URL (production unless internal dev override)."""
    import os
    override = os.environ.get("_GLYPHH_INTERNAL_DEV_OVERRIDE")
    return override if override else PLATFORM_URL


def _get_first_model() -> tuple[str, str]:
    """Get the first available org_id/model_id from the model manager.

    Falls back to 'local-dev-org'/'demo' if nothing is loaded yet.
    """
    try:
        from glyphh.server import model_manager
        if model_manager and model_manager._models:
            key = next(iter(model_manager._models))
            return key  # (org_id, model_id) tuple
    except Exception:
        pass
    return ("local-dev-org", "demo")


@router.get("/", include_in_schema=False)
async def root_redirect(request: Request):
    """Redirect / to the dashboard for the first available model."""
    org_id, model_id = _get_first_model()
    return RedirectResponse(url=f"/{org_id}/{model_id}/chat", status_code=302)


@router.get("/login", include_in_schema=False)
async def login_page(request: Request):
    """Render login page (non-local deployments)."""
    if settings.deployment_mode == "local":
        return RedirectResponse(url="/", status_code=302)

    template = _jinja_env.get_template("login.html")
    html = template.render(
        logo_b64=_LOGO_B64,
        favicon_b64=_FAVICON_B64,
        platform_url=_get_platform_url(),
    )
    return HTMLResponse(content=html)
