"""
API Routes for Glyphh Runtime.
"""

from api.routes.deployment import router as deployment_router
from api.routes.glyphs import router as glyphs_router
from api.routes.query import router as query_router
from api.routes.health import router as health_router
from api.routes.org_scoped import router as org_scoped_router
from api.routes.listeners import router as listeners_router
from api.routes.nl_query import router as nl_query_router
from api.routes.jobs import router as jobs_router
from api.routes.chat import router as chat_router
from api.routes.procedures import router as procedures_router
from api.routes.charts import router as charts_router
from api.routes.viewer import router as viewer_router

__all__ = [
    "deployment_router",
    "glyphs_router",
    "query_router",
    "health_router",
    "org_scoped_router",
    "listeners_router",
    "nl_query_router",
    "jobs_router",
    "chat_router",
    "procedures_router",
    "charts_router",
    "viewer_router",
]
