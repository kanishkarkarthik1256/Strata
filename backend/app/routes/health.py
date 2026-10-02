"""Health-check endpoint.

Returns application status, version, GPU availability, disk space,
and database connectivity in a single JSON payload.
"""

from __future__ import annotations

import datetime as _dt
import shutil
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.engine import get_session

router = APIRouter(tags=["health"])


def _gpu_info() -> dict[str, Any]:
    """Return GPU status if CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            free_mem = torch.cuda.mem_get_info(0)[0]
            return {
                "available": True,
                "device": device_name,
                "total_memory_mb": round(total_mem / (1024**2)),
                "free_memory_mb": round(free_mem / (1024**2)),
                "cuda_version": torch.version.cuda,
            }
        return {"available": False, "reason": "CUDA not available on this device"}
    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}


def _disk_info() -> dict[str, Any]:
    """Return disk space for the storage volume."""
    usage = shutil.disk_usage(settings.storage.base_path)
    return {
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
        "path": str(settings.storage.base_path),
    }


def _colmap_info() -> dict[str, Any]:
    """Check whether the COLMAP binary is on PATH."""
    found = shutil.which(settings.colmap.binary_path)
    return {
        "binary": settings.colmap.binary_path,
        "available": found is not None,
        "path": found,
    }


@router.get("/api/health")
async def health_check(
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Comprehensive health-check endpoint.

    Returns 200 with full status when healthy; includes warnings for
    degraded subsystems.
    """
    # Database check
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    status = "healthy"
    warnings: list[str] = []

    if not db_ok:
        warnings.append("database_unreachable")
        status = "degraded"

    gpu = _gpu_info()
    if not gpu["available"]:
        warnings.append("gpu_unavailable")

    colmap = _colmap_info()
    if not colmap["available"]:
        warnings.append("colmap_not_found")

    if warnings:
        status = "degraded"

    return {
        "status": status,
        "version": settings.version,
        "app_name": settings.app_name,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "subsystems": {
            "database": {"ok": db_ok},
            "gpu": gpu,
            "disk": _disk_info(),
            "colmap": colmap,
        },
        "warnings": warnings,
    }
