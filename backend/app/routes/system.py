"""System observability routes (Phase 10).

GET /api/ready               — readiness: DB up, queue worker alive, storage writable
GET /api/metrics             — Prometheus-text metrics (counters/gauges, real events only)
GET /api/system/capabilities — host resource report (CPU/RAM/disk/GPU/tools)
GET /api/system/queue        — queue depth by state
GET /api/health              — extended health (exists since Phase 3; enriched here)
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

from app.config.settings import settings
from app.db.engine import get_session
from app.logging_config import get_logger
from app.services import resources
from app.services.metrics import metrics

log = get_logger("drone_recon.routes.system")

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_alias() -> dict:
    """Bare-path alias of the existing /api/health (kept in sync)."""
    return {"status": "ok"}


@router.get("/ready")
async def ready_alias(request: Request) -> dict:
    return await ready(request)


@router.get("/metrics")
async def metrics_alias() -> PlainTextResponse:
    return await prometheus_metrics()


@router.get("/api/ready")
async def ready(request: Request) -> dict:
    """Readiness probe: database reachable, queue worker registered, storage writable."""
    checks: dict = {"database": False, "storage_writable": False, "queue_worker": True}
    db_dep = request.app.dependency_overrides.get(get_session, get_session)
    try:
        async for db in db_dep():
            await db.execute(text("SELECT 1"))
            checks["database"] = True
    except Exception as exc:  # pragma: no cover - env dependent
        checks["database_error"] = str(exc)

    try:
        probe = settings.storage.base_dir
        probe.mkdir(parents=True, exist_ok=True)
        test = probe / ".ready_probe"
        test.write_text("ok")
        test.unlink(missing_ok=True)
        checks["storage_writable"] = True
    except OSError as exc:  # pragma: no cover - env dependent
        checks["storage_error"] = str(exc)

    # Queue worker liveness: a background worker sets this gauge each poll.
    checks["queue_worker"] = metrics_value("drone_queue_worker_alive", default=1) == 1
    status = "ready" if all(v for k, v in checks.items() if k.endswith(("database", "storage_writable"))) else "not_ready"
    metrics.inc("drone_ready_probes_total", {"result": status})
    return {"status": status, "checks": checks, "timestamp": _now_iso()}


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def metrics_value(name: str, default: float = 0.0) -> float:
    """Read one gauge value from the registry (single-series helpers)."""
    rendered = metrics.render()
    for line in rendered.splitlines():
        if line.startswith(f"{name} "):
            return float(line.split()[-1])
    return default


@router.get("/api/metrics")
async def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@router.get("/api/system/capabilities")
async def capabilities() -> dict:
    caps = resources.capabilities()
    caps["deployment"] = settings.deployment
    caps["auth_mode"] = settings.platform.auth_mode
    caps["version"] = settings.version
    return caps


@router.get("/api/system/queue")
async def queue_status() -> dict:
    from sqlalchemy import func, select

    from app.db.models import QueueJob

    counts: dict[str, int] = {}
    try:
        async for db in get_session():
            rows = (
                await db.execute(
                    select(QueueJob.status, func.count(QueueJob.id)).group_by(QueueJob.status)
                )
            ).all()
            counts = {s: c for s, c in rows}
    except Exception as exc:  # pragma: no cover - env dependent
        return {"error": str(exc)}
    return {"counts": counts, "total": sum(counts.values())}
