"""
API Routes for Glyphh Runtime.
"""

from api.routes.deployment import router as deployment_router
from api.routes.glyphs import router as glyphs_router
from api.routes.query import router as query_router
from api.routes.health import router as health_router
from api.routes.org_scoped import router as org_scoped_router

__all__ = [
    "deployment_router",
    "glyphs_router",
    "query_router",
    "health_router",
    "org_scoped_router",
]
