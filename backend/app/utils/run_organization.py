"""Run organization utilities for Phase 9.5.

Manages the standardized output and documentation structure:
- outputs/<RUN_ID>/ for all pipeline artifacts
- docs/<RUN_ID>/ for all reports
- Matching run IDs between outputs and docs
- Manifest generation and management
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_run_id(dataset: str, mission: str, timestamp: datetime | None = None) -> str:
    """Generate a deterministic run ID.
    
    Format: <dataset>_<mission>_<YYYYMMDD_HHMMSS>
    Example: shitan_ms1_20260907_183000
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    # Sanitize dataset and mission names
    dataset_clean = dataset.lower().replace(" ", "_").replace("-", "_")
    mission_clean = mission.lower().replace(" ", "_").replace("-", "_")
    
    return f"{dataset_clean}_{mission_clean}_{ts_str}"


def create_run_structure(
    run_id: str,
    base_dir: Path | str = ".",
    stages: list[str] | None = None,
) -> dict[str, Path]:
    """Create the run directory structure.
    
    Only creates directories for stages that will actually run.
    If *stages* is ``None``, creates the base directories plus frames
    and sparse (the minimum for any run).
    
    Returns a dict mapping logical names to Path objects.
    """
    base = Path(base_dir)
    
    # Create output directories
    output_base = base / "outputs" / run_id
    docs_base = base / "docs" / run_id
    
    directories: dict[str, Path] = {
        "output_base": output_base,
        "docs_base": docs_base,
        "frames": output_base / "frames",
        "sparse": output_base / "sparse",
        "metrics": output_base / "metrics",
    }
    
    # Only create stage directories that will actually be used
    _STAGE_DIRS = ["depth", "depth_learned", "video", "dense", "pointcloud", "mesh", "texture", "geospatial", "metric_scale", "semantics", "config"]
    if stages is None:
        stages = ["frames", "sparse", "metrics"]
    
    for stage in stages:
        if stage in _STAGE_DIRS and stage not in directories:
            directories[stage] = output_base / stage
    
    # Create all directories
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return directories


def get_git_commit() -> str | None:
    """Get the current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def create_manifest(
    run_id: str,
    dataset: str,
    mission: str,
    started_at: datetime,
    status: str = "PARTIAL",
    input_info: dict[str, Any] | None = None,
    stages: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Create a run manifest."""
    completed_at = datetime.now(timezone.utc)
    
    manifest = {
        "run_id": run_id,
        "dataset": dataset,
        "mission": mission,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "status": status,
        "pipeline_version": "9.5.0",
        "git_commit": get_git_commit(),
        "input": input_info or {},
        "stages": stages or {},
        "artifacts": artifacts or {},
        "metrics": metrics or {},
        "limitations": limitations or [],
    }
    
    return manifest


def save_manifest(manifest: dict[str, Any], output_dir: Path, docs_dir: Path) -> None:
    """Save manifest to both output and docs directories."""
    manifest_path_output = output_dir / "manifest.json"
    manifest_path_docs = docs_dir / "manifest.json"
    
    # Custom JSON encoder to handle non-serializable types like IFDRational
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'numerator') and hasattr(obj, 'denominator'):
                # Handle IFDRational and similar types
                return float(obj)
            elif hasattr(obj, 'isoformat'):
                # Handle datetime objects
                return obj.isoformat()
            elif isinstance(obj, bytes):
                return obj.decode('utf-8', errors='replace')
            return super().default(obj)
    
    with open(manifest_path_output, "w") as f:
        json.dump(manifest, f, indent=2, cls=CustomEncoder)
    
    with open(manifest_path_docs, "w") as f:
        json.dump(manifest, f, indent=2, cls=CustomEncoder)


def load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    """Load a manifest from disk."""
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def list_runs(base_dir: Path | str = ".") -> list[dict[str, Any]]:
    """List all available runs with their status."""
    base = Path(base_dir)
    outputs_dir = base / "outputs"
    
    runs = []
    if outputs_dir.exists():
        for run_dir in sorted(outputs_dir.iterdir()):
            if run_dir.is_dir():
                manifest = load_manifest(run_dir / "manifest.json")
                if manifest:
                    runs.append(manifest)
    
    return runs


def verify_run_consistency(run_id: str, base_dir: Path | str = ".") -> dict[str, bool]:
    """Verify that outputs and docs are consistent for a run."""
    base = Path(base_dir)
    
    output_manifest = load_manifest(base / "outputs" / run_id / "manifest.json")
    docs_manifest = load_manifest(base / "docs" / run_id / "manifest.json")
    
    return {
        "output_exists": output_manifest is not None,
        "docs_exists": docs_manifest is not None,
        "manifests_match": output_manifest == docs_manifest if (output_manifest and docs_manifest) else False,
    }
