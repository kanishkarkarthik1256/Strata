"""Canonical Demo Processing Service.

Executes live end-to-end processing of `data/base.mp4` through the real STRATA
reconstruction pipeline. Dynamically creates a unique run output directory in
`output/<run_id>/`, computes SHA-256 traceability, and registers the run.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.models import Project
from app.logging_config import get_logger
from app.schemas.pipeline import PipelineStartRequest
from app.services.metadata_extraction import extract_metadata
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.video_validation import validate_all

log = get_logger("drone_recon.services.canonical_demo")

CANONICAL_VIDEO_REL = "data/base.mp4"


def get_canonical_video_path() -> Path:
    """Resolve data/base.mp4 relative to repository root."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    path = repo_root / CANONICAL_VIDEO_REL
    if not path.exists():
        # Fallback relative to cwd
        path = Path.cwd() / CANONICAL_VIDEO_REL
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical demo video '{CANONICAL_VIDEO_REL}' not found at {path}. "
            "Please ensure data/base.mp4 is present in the repository."
        )
    return path


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


async def process_canonical_demo(
    db: AsyncSession,
    *,
    force_stages: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Execute live end-to-end processing of data/base.mp4.

    Creates a new project record, sets up output/<run_id>/ workspace, runs the
    real pipeline, writes manifest.json with SHA-256 traceability, and registers
    the run.
    """
    video_path = get_canonical_video_path()
    sha256_hash = compute_sha256(video_path)

    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"canonical_run_{timestamp_str}_{uuid.uuid4().hex[:6]}"

    log.info("canonical_demo_started", run_id=run_id, video=str(video_path), sha256=sha256_hash)

    # 1. Setup workspace directories: output/<run_id>/ and data/storage/<run_id>/
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    output_base = repo_root / "output"
    output_dir = output_base / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    storage_dir = settings.storage.project_dir(run_id)
    storage_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy data/base.mp4 into storage & output workspaces
    copied_video = storage_dir / "base.mp4"
    shutil.copy2(video_path, copied_video)
    shutil.copy2(video_path, output_dir / "base.mp4")

    # 3. Validate video & extract metadata
    validate_all(copied_video)
    metadata = extract_metadata(copied_video)

    # 4. Create Project database record
    project = Project(
        id=run_id,
        name=f"Canonical Demo ({run_id})",
        video_filename="base.mp4",
        video_path=str(copied_video),
        video_duration_sec=metadata.duration_sec,
        video_fps=metadata.fps,
        video_width=metadata.width,
        video_height=metadata.height,
        status="uploaded",
        current_stage="ingestion",
    )
    db.add(project)
    await db.commit()

    # 5. Run autonomous pipeline synchronously/in-thread
    pipeline_req = PipelineRequest(
        force=force_stages or ["frames", "sparse", "depth", "dense", "georef"]
    )
    report = run_autonomous_pipeline(run_id, pipeline_req)

    pipeline_status = report.get("status", "unknown")

    # 6. Update Project status
    res = await db.execute(select(Project).where(Project.id == run_id))
    p_rec = res.scalar_one_or_none()
    if p_rec:
        p_rec.status = "completed" if pipeline_status == "completed" else "failed"
        if pipeline_status != "completed":
            p_rec.error_message = report.get("error", "Canonical pipeline failed")
        await db.commit()

    # 7. Write input-traceable manifest.json into output/<run_id>/ and workspace
    manifest = {
        "run_id": run_id,
        "is_canonical_demo": True,
        "is_precomputed_fast_demo": False,
        "status": pipeline_status,
        "input": {
            "filename": "base.mp4",
            "relative_path": CANONICAL_VIDEO_REL,
            "size_bytes": video_path.stat().st_size,
            "sha256": sha256_hash,
            "duration_seconds": metadata.duration_sec,
            "resolution": f"{metadata.width}x{metadata.height}",
            "fps": metadata.fps,
        },
        "outputs": {
            "output_directory": f"output/{run_id}",
            "storage_directory": str(storage_dir),
        },
        "pipeline_summary": {
            "total_time_ms": report.get("run_time_ms", 0.0),
            "stages": report.get("stages", {}),
            "profile": report.get("profile", {}),
        },
    }

    with open(output_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    with open(storage_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    # 8. Copy generated artifacts into output/<run_id>/
    _sync_artifacts_to_output(storage_dir, output_dir)

    log.info(
        "canonical_demo_finished",
        run_id=run_id,
        status=pipeline_status,
        output_dir=str(output_dir),
    )

    return run_id, report


def _sync_artifacts_to_output(src_dir: Path, dst_dir: Path) -> None:
    """Sync all generated artifacts from storage workspace to output/<run_id>/."""
    for sub in ["reconstruction", "cameras", "geospatial", "analysis", "reports", "depth", "selected", "validation"]:
        s_sub = src_dir / sub
        if s_sub.exists() and s_sub.is_dir():
            d_sub = dst_dir / sub
            d_sub.mkdir(parents=True, exist_ok=True)
            for item in s_sub.glob("*"):
                if item.is_file():
                    shutil.copy2(item, d_sub / item.name)

    # Root files
    # The retired metric_accuracy_report.json is gone: accuracy now lives in
    # validation/validation_report.json (copied with the validation/ subtree).
    for fname in ["sparse_model.ply", "poses.json", "reconstruction_report.json", "performance.json", "pipeline_report.json"]:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)
