from __future__ import annotations

import logging
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
GLYPHH_SDK_ROOT = ROOT / "glyphh-sdk"
if GLYPHH_SDK_ROOT.exists() and str(GLYPHH_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(GLYPHH_SDK_ROOT))

from .core import models
from .core.config import get_settings
from .core.db import Base, engine
from .services.listener_runtime import ListenerManager
from .services.usage_metrics import UsageTracker
from .routes import bundles as bundle_routes
from .routes import charts as charts_routes
from .routes import health as health_routes
from .routes import ingest as ingest_routes
from .routes import listeners as listener_routes
from .routes import model_tools as model_tools_routes
from .routes import nl as nl_routes
from .routes import query as query_routes
from .routes import trends as trends_routes
from .routes import viewer as viewer_routes
from .services.auth_runtime import build_runtime_auth_middleware


logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title="Glyphh Runtime")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(charts_routes.router, prefix="/api/v1")
app.include_router(bundle_routes.router, prefix="/api/v1")
app.include_router(health_routes.router, prefix="/api/v1")
app.include_router(ingest_routes.router, prefix="/api/v1")
app.include_router(listener_routes.router, prefix="/api/v1")
app.include_router(model_tools_routes.router, prefix="/api/v1")
app.include_router(nl_routes.router, prefix="/api/v1")
app.include_router(query_routes.router, prefix="/api/v1")
app.include_router(trends_routes.router, prefix="/api/v1")
app.include_router(viewer_routes.router, prefix="/api/v1")

app.middleware("http")(build_runtime_auth_middleware(settings))

listener_manager = ListenerManager()
app.state.listener_manager = listener_manager
app.state.usage_tracker = None


@app.on_event("startup")
async def startup_listener_manager() -> None:
    Base.metadata.create_all(bind=engine)
    await listener_manager.start()
    if settings.usage_metrics_enabled and settings.platform_api_base and settings.runtime_token:
        try:
            tracker = UsageTracker(
                platform_api_base=settings.platform_api_base,
                runtime_token=settings.runtime_token,
                runtime_version=settings.runtime_version,
                jwt_secret=settings.jwt_secret,
                jwt_algorithm=settings.jwt_algorithm,
                flush_seconds=settings.usage_metrics_flush_seconds,
            )
            tracker.start()
            app.state.usage_tracker = tracker
            listener_manager.set_usage_tracker(tracker)
            logger.info("runtime usage metrics enabled")
        except Exception:
            logger.exception("failed to start runtime usage metrics")


@app.on_event("shutdown")
async def shutdown_listener_manager() -> None:
    await listener_manager.stop()
    tracker = getattr(app.state, "usage_tracker", None)
    if tracker:
        tracker.stop()
