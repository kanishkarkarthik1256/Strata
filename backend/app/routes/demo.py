"""Demo API routes for Canonical Demo Live Processing and Fast Demo Precomputed Mode."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.exceptions import NotFoundError
from app.logging_config import get_logger
from app.services import canonical_demo_service, run_service

log = get_logger("drone_recon.routes.demo")

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.post("/process-canonical")
async def process_canonical_demo_endpoint(
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Execute live end-to-end processing of data/base.mp4.

    Runs the real pipeline from scratch, creates output/<run_id>/, and registers
    the new run.
    """
    run_id, report = await canonical_demo_service.process_canonical_demo(db)

    return {
        "job_id": run_id,
        "run_id": run_id,
        "status": report.get("status", "completed"),
        "message": f"Canonical demo processed successfully: {run_id}",
        "output_directory": f"output/{run_id}",
        "pipeline_summary": report,
    }


@router.get("/fast-demo")
async def fast_demo_endpoint() -> dict[str, Any]:
    """Open the precomputed fast demo reconstruction.

    Selects the newest run discovered under the legacy precomputed-demo root
    (``backend/outputs``). Those runs are tagged ``is_demo`` at discovery —
    selection is by origin, never by run-id substring, and normal missions
    are never silently substituted when no demo run exists.
    """
    demo_runs = [r for r in run_service.list_runs() if r.get("is_demo")]
    if not demo_runs:
        raise NotFoundError("No precomputed demo run is available")
    fast_run = demo_runs[0]  # list_runs() is newest-first

    return {
        "run_id": fast_run["run_id"],
        "is_precomputed_fast_demo": True,
        "mode": "FAST_DEMO",
        "badge": "FAST DEMO — PRECOMPUTED RESULT",
        "message": "Precomputed demo reconstruction loaded.",
        "details": fast_run,
    }
