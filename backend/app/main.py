"""FastAPI application factory for DroneRecon.

Usage::

    # As a factory (preferred for uvicorn)
    uvicorn app.main:create_app --factory

    # Direct run
    python -m app.main
"""

from __future__ import annotations

import os

# Prevent OpenMP runtime library conflict crashes (libiomp5 vs libomp) on macOS/CPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Thread budget before torch/OpenMP load: the BLAS runtimes read these once, at
# import time, so they must be set from a stdlib-only module here. The budget is
# the PHYSICAL core count — see app/services/cpu_budget.py for the measurement.
from app.services.cpu_budget import apply_thread_environment  # noqa: E402

apply_thread_environment()

# OpenMP load order is load-bearing: this host SIGSEGVs (exit 139 — a native
# crash no `except` can catch) when torch's runtime starts after pycolmap's
# libomp is loaded. The route imports below pull pycolmap in at import time
# (app/services/camera_pose_estimator.py), so torch has to be imported first.
# See app/services/depth_prefetch.py for the measurement and the guard that
# assumes this order.
try:  # noqa: E402
    import torch  # noqa: F401,E402
except ImportError:  # pragma: no cover - torch is installed in every deployment
    pass  # the depth stage reports the missing dependency in its own terms

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.db.engine import close_db, init_db
from app.error_handlers import register_error_handlers
from app.logging_config import get_logger, setup_logging
from app.middleware.auth import AuthMiddleware
from app.middleware.request_logging import RequestLoggingMiddleware
from app.middleware.timing import TimingMiddleware
from app.routes.auth import router as auth_router
from app.routes.data_videos import router as data_videos_router
from app.routes.demo import router as demo_router
from app.routes.dense import router as dense_router
from app.routes.digital_twin import router as digital_twin_router
from app.routes.frame_extraction import router as frame_extraction_router
from app.routes.health import router as health_router
from app.routes.intel import router as intel_router
from app.routes.mission import router as mission_router
from app.routes.missions import router as missions_router
from app.routes.pipeline import router as pipeline_router
from app.routes.reconstruction import router as reconstruction_router
from app.routes.runs import router as runs_router
from app.routes.system import router as system_router
from app.routes.upload import router as upload_router

log = get_logger("drone_recon.app")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


_queue_stop: asyncio.Event | None = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Async lifespan — runs on startup and shutdown."""
    # --- Startup ---
    setup_logging(
        log_level=settings.server.log_level,
        json_output=not settings.debug,
    )
    log.info(
        "app_starting",
        version=settings.version,
        debug=settings.debug,
        device=settings.ai.device,
    )

    await init_db()

    # Bootstrap admin (env-configured, idempotent) + start the queue worker.
    from app.db.engine import get_session
    from app.services.auth_service import ensure_bootstrap_admin

    async for db in get_session():
        await ensure_bootstrap_admin(db)

    global _queue_stop
    from app.services import job_queue

    job_queue.register_default_executors()
    _queue_stop = asyncio.Event()
    _queue_task = asyncio.create_task(job_queue.worker_loop(_queue_stop))

    log.info(
        "app_ready",
        host=settings.server.host,
        port=settings.server.port,
    )

    yield

    # --- Shutdown ---
    log.info("app_shutting_down")
    if _queue_stop is not None:
        _queue_stop.set()
        try:
            await asyncio.wait_for(_queue_task, timeout=5)
        except (asyncio.TimeoutError, Exception):
            _queue_task.cancel()
    await close_db()
    log.info("app_stopped")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="AI-enabled single-pass drone video to 3D model generation",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # --- Custom middleware (order matters: outermost runs first) ---
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TimingMiddleware)

    # --- Error handlers ---
    register_error_handlers(app)

    # --- Routes ---
    app.include_router(health_router)
    app.include_router(system_router)
    app.include_router(auth_router)
    app.include_router(upload_router)
    app.include_router(frame_extraction_router)
    app.include_router(reconstruction_router)
    app.include_router(dense_router)
    app.include_router(pipeline_router)
    app.include_router(digital_twin_router)
    app.include_router(intel_router)
    app.include_router(mission_router)
    app.include_router(missions_router)
    app.include_router(runs_router)
    app.include_router(demo_router)
    app.include_router(data_videos_router)

    return app


# ---------------------------------------------------------------------------
# Direct invocation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        log_level=settings.server.log_level,
    )
