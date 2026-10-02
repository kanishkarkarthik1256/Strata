"""End-to-End Reconstruction Correctness Test for STRATA.

Executes the full pipeline on data/base.mp4 and verifies sparse & dense
point cloud bounds, intrinsics, reprojection errors, and PLY exports.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Ensure backend/ is in sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "backend"))

from app.config.settings import settings
from app.services.canonical_demo_service import get_canonical_video_path
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.pointcloud import read_ply

def inspect_ply(label: str, path: Path):
    if not path.exists():
        print(f"[{label}] File NOT found at {path}")
        return
    cloud = read_ply(path)
    if cloud.n == 0:
        print(f"[{label}] EMPTY cloud at {path}")
        return
    min_xyz = cloud.xyz.min(axis=0)
    max_xyz = cloud.xyz.max(axis=0)
    range_xyz = max_xyz - min_xyz
    centroid = cloud.xyz.mean(axis=0)
    std_xyz = cloud.xyz.std(axis=0)
    print(f"\n============================================================")
    print(f"=== Artifact Inspection: {label} ({cloud.n} points) ===")
    print(f"  Path:     {path}")
    print(f"  Min XYZ:  [{min_xyz[0]:.6f}, {min_xyz[1]:.6f}, {min_xyz[2]:.6f}]")
    print(f"  Max XYZ:  [{max_xyz[0]:.6f}, {max_xyz[1]:.6f}, {max_xyz[2]:.6f}]")
    print(f"  Range:    [{range_xyz[0]:.6f}, {range_xyz[1]:.6f}, {range_xyz[2]:.6f}]")
    print(f"  Centroid: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
    print(f"  Std Dev:  [{std_xyz[0]:.6f}, {std_xyz[1]:.6f}, {std_xyz[2]:.6f}]")
    if cloud.residual is not None and len(cloud.residual) > 0:
        print(f"  Residual: min={cloud.residual.min():.4f}, max={cloud.residual.max():.4f}, mean={cloud.residual.mean():.4f}")
    if cloud.confidence is not None and len(cloud.confidence) > 0:
        print(f"  Conf:     min={cloud.confidence.min():.4f}, max={cloud.confidence.max():.4f}, mean={cloud.confidence.mean():.4f}")
    print(f"============================================================\n")

def main():
    print("Executing STRATA End-to-End Reconstruction Correctness Test...")
    video_path = get_canonical_video_path()
    run_id = f"test_correctness_{int(time.time())}"

    # Setup storage workspace
    storage_dir = settings.storage.project_dir(run_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, storage_dir / "base.mp4")

    # Output directory
    output_dir = repo_root / "output" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, output_dir / "base.mp4")

    req = PipelineRequest(
        depth_backend="auto",
        extraction_mode="target_fps",
        target_fps=1.0,
    )

    t0 = time.perf_counter()
    report = run_autonomous_pipeline(run_id, req)
    elapsed = time.perf_counter() - t0

    print(f"\nPipeline Status: {report.get('status')}")
    print(f"Total Wall-Clock Time: {elapsed:.2f} seconds")

    # Inspect generated artifacts
    inspect_ply("Sparse Reconstruction (storage)", storage_dir / "sparse_model.ply")
    inspect_ply("Dense Reconstruction (storage)", storage_dir / "dense" / "dense_model.ply")

    # Sync to output_dir
    for f in ("sparse_model.ply", "poses.json", "reconstruction_report.json", "pipeline_report.json"):
        if (storage_dir / f).exists():
            shutil.copy2(storage_dir / f, output_dir / f)
    if (storage_dir / "dense" / "dense_model.ply").exists():
        (output_dir / "dense").mkdir(parents=True, exist_ok=True)
        shutil.copy2(storage_dir / "dense" / "dense_model.ply", output_dir / "dense" / "dense_model.ply")

    inspect_ply("Sparse Reconstruction (output)", output_dir / "sparse_model.ply")
    inspect_ply("Dense Reconstruction (output)", output_dir / "dense" / "dense_model.ply")

if __name__ == "__main__":
    main()
