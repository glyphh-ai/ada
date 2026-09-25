"""
Glyphh Runtime Server — Ada's brain with production infrastructure.

Single FastAPI application combining Ada's cognitive pipeline with
Glyphh's auth, licensing, metering, and deployment infrastructure.

Importable as ``glyphh.server:app`` for both pip-installed CLI usage and
Docker/production deployments.  The repo-root ``main.py`` is a thin shim
that re-exports this app.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from infrastructure.config import get_settings, validate_settings
from infrastructure.database import init_db, close_db, async_session_maker
from shared.exceptions import GlyphhRuntimeException
from glyphh.licensing import load_license, set_current_license
from shared.middleware import (
    CorrelationIDMiddleware,
    LoggingMiddleware,
)
from domains.models.manager import ModelManager
from domains.resources.manager import ResourceManager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

settings = get_settings()
_start_time = datetime.utcnow()

# ── Global state ────────────────────────────────────────────────────────────

model_manager: Optional[ModelManager] = None
resource_manager: Optional[ResourceManager] = None
brain: Optional[object] = None  # domains.brain.think.Brain — set in lifespan


def get_model_manager() -> ModelManager:
    """Dependency for getting model manager"""
    return model_manager


def get_resource_manager() -> ResourceManager:
    """Dependency for getting resource manager"""
    return resource_manager


def get_brain():
    """Dependency for getting the brain"""
    return brain


# ── Capabilities directory ──────────────────────────────────────────────────

CAPABILITIES_DIR = Path(__file__).resolve().parent.parent / "capabilities"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Boot Ada's brain with Glyphh production infrastructure."""
    global model_manager, resource_manager, brain

    logger.info("Waking up...")

    # ── Configuration ───────────────────────────────────────────────────
    try:
        validate_settings()
        logger.info("Configuration validated")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

    # ── Licensing & metering ────────────────────────────────────────────
    license_info = load_license()
    app.state.license = license_info
    set_current_license(license_info)
    logger.info(
        f"License: tier={license_info.tier}, org={license_info.org_id}, "
        f"ops={license_info.format_limit()}/mo, runtimes={license_info.max_runtimes}"
    )

    from glyphh.metering import get_meter
    meter = get_meter()
    usage = meter.get_usage(license_info.org_id)
    if usage > 0:
        logger.info(f"Usage this month: {usage:,} ops")
        if not license_info.is_unlimited:
            if usage >= license_info.max_encodings_per_month:
                logger.warning(
                    f"Monthly operation limit exceeded ({usage:,}/{license_info.max_encodings_per_month:,})"
                )
            elif license_info.encoding_warning_threshold() and usage >= license_info.encoding_warning_threshold():
                logger.warning(
                    f"Approaching monthly limit ({usage:,}/{license_info.max_encodings_per_month:,})"
                )

    # ── Database ────────────────────────────────────────────────────────
    await init_db()
    logger.info("Memory banks online")

    # ── Model & resource managers ───────────────────────────────────────
    model_manager = ModelManager(async_session_maker)
    resource_manager = ResourceManager(async_session_maker)

    # ── Load innate capabilities ────────────────────────────────────────
    from domains.brain.loader import CapabilityLoader, register_capabilities_in_db

    brain_state = CapabilityLoader.boot(CAPABILITIES_DIR)
    app.state.brain_state = brain_state

    # Register capabilities in DB (loads exemplars, creates encoders)
    await register_capabilities_in_db(brain_state, model_manager)

    # ── Initialize Ada's LLM ───────────────────────────────────────────
    from domains.brain.llm import AdaLLM

    llm = AdaLLM()
    app.state.llm = llm

    # ── Initialize the think pipeline ──────────────────────────────────
    from domains.brain.think import Brain

    global brain
    brain = Brain(
        brain_state=brain_state,
        model_manager=model_manager,
        llm=llm,
        session_factory=async_session_maker,
    )
    app.state.brain = brain

    # ── Load persistent memories ───────────────────────────────────────
    from glyphh.memory.thought_persistence import load_thoughts
    loaded = await load_thoughts(async_session_maker, brain.cognitive.thought_space)
    if loaded:
        logger.info(f"Restored {loaded} memories from database")
    else:
        logger.info(f"No persisted memories — using {brain.cognitive.thought_space.count} seed memories")

    # ── Load persistent threads ───────────────────────────────────────
    from domains.brain.thread_persistence import load_threads
    thread_count = await load_threads(async_session_maker, brain.thread_store)
    if thread_count:
        logger.info(f"Restored {thread_count} context threads from database")

    # ── Seed user identity from auth ──────────────────────────────────
    try:
        from glyphh.cli.auth import get_user
        user = get_user()
        if user:
            name = user.get("first_name", user.get("email", ""))
            email = user.get("email")
            if name:
                brain.seed_user_identity(name, email)
                logger.info(f"Seeded user identity: {name}")
    except Exception:
        pass  # server may run without CLI auth context

    logger.info("Think pipeline online")

    # ── Initialize MCP ─────────────────────────────────────────────────
    from domains.auth.service import AuthService
    from domains.mcp.app import create_mcp_session_managers

    auth_service = AuthService()
    json_manager, sse_manager = create_mcp_session_managers(brain, auth_service)
    app.state.mcp_session_managers = (json_manager, sse_manager)
    logger.info("MCP endpoint online at /mcp")

    # ── Resume any incomplete encoding from previous boot ──────────────
    try:
        await model_manager.resume_staged_encoding()
    except Exception as e:
        logger.warning(f"Staged encoding resume: {e}")

    # ── Wait for ALL encoding to finish before accepting requests ──────
    if model_manager._encoding_in_progress:
        logger.info(
            f"Waiting for exemplar encoding to finish: "
            f"{[k[1] for k in model_manager._encoding_in_progress]}"
        )
        while model_manager._encoding_in_progress:
            await asyncio.sleep(1.0)
        logger.info("All exemplars encoded")

    # ── Start dream loop ───────────────────────────────────────────────
    brain.start_dreaming()
    logger.info("Dream loop active")

    # ── Start background persistence worker ────────────────────────────
    async def _persist_worker():
        """Flush thought + thread queues to SQLite every 2 seconds."""
        from domains.brain.thread_persistence import save_thread
        while True:
            try:
                saved = await brain.flush_persist_queue()
                if saved:
                    logger.debug(f"Persisted {saved} thoughts")
            except Exception:
                pass
            # Persist any dirty threads
            try:
                for thread in brain.thread_store.all_threads():
                    await save_thread(async_session_maker, thread)
            except Exception:
                pass
            await asyncio.sleep(2.0)

    persist_task = asyncio.create_task(_persist_worker())

    logger.info("Ada is awake.")

    # Run MCP session managers
    async with json_manager.run():
        async with sse_manager.run():
            yield

    # ── Shutdown ────────────────────────────────────────────────────────
    logger.info("Going to sleep...")
    persist_task.cancel()
    await brain.flush_persist_queue()
    brain.stop_dreaming()

    # Flush memory strengths to DB
    from glyphh.memory.thought_persistence import flush_all_strengths
    await flush_all_strengths(async_session_maker, brain.cognitive.thought_space)

    # Flush all threads to DB
    from domains.brain.thread_persistence import save_thread
    for thread in brain.thread_store.all_threads():
        try:
            await save_thread(async_session_maker, thread)
        except Exception:
            pass

    meter.flush()
    await close_db()
    logger.info("Ada is asleep.")
    logging.shutdown()


# Create FastAPI application
app = FastAPI(
    title="Glyphh Runtime",
    description="Ada's cognitive brain with production infrastructure",
    version="2.6.7",
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    lifespan=lifespan,
)

# CORS middleware
if settings.cors_allow_all:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    origins = settings.cors_origins_production or settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )

# Custom middleware
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(LoggingMiddleware)


# Global exception handlers
@app.exception_handler(GlyphhRuntimeException)
async def runtime_exception_handler(request: Request, exc: GlyphhRuntimeException) -> JSONResponse:
    """Handle custom runtime exceptions with CORS headers"""
    response = JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "correlation_id": getattr(request.state, "correlation_id", "unknown"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )
    origin = request.headers.get("origin")
    if origin and _is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with CORS headers"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    response = JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "correlation_id": getattr(request.state, "correlation_id", "unknown"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )
    origin = request.headers.get("origin")
    if origin and _is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


def _is_allowed_origin(origin: str) -> bool:
    """Check if origin is in allowed CORS origins list."""
    if settings.cors_allow_all:
        return True
    import fnmatch
    origins = settings.cors_origins_production or settings.cors_origins
    for allowed in origins:
        if allowed == "*" or allowed == origin:
            return True
        if fnmatch.fnmatch(origin, allowed):
            return True
    return False


# Import and include routers
from api.routes.health import router as health_router
from api.routes.strand import router as strand_router
from api.routes.tokens import router as tokens_router

app.include_router(health_router)
app.include_router(strand_router)
app.include_router(tokens_router)


# MCP routing middleware — intercepts /mcp requests and forwards to the
# MCP SDK's Streamable HTTP handler. Single endpoint, no org/model.
from domains.mcp.app import MCPRoutingMiddleware

app.add_middleware(MCPRoutingMiddleware, mcp_app_getter=lambda: getattr(app.state, "mcp_session_managers", None))
