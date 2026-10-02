"""Reconstruction Correctness Diagnostic Script for STRATA.

Traces coordinate bounds, intrinsics, PyCOLMAP SfM outputs, georeferencing,
and PLY export stages step by step for data/base.mp4.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure backend/ is in sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "backend"))

from app.config.settings import settings
from app.services.camera_pose_estimator import estimate_poses
from app.services.frame_extractor import extract_frames
from app.services.metadata_extraction import extract_metadata
from app.services.pointcloud import read_ply, save_ply
from app.services.sparse_reconstruction import _write_poses_json, _write_sparse_ply, run_sparse_reconstruction
from app.services.video_validation import validate_all

def print_bounds(name: str, coords: np.ndarray, units: str = "meters"):
    """Print coordinate statistics for a 3D point cloud."""
    if len(coords) == 0:
        print(f"[{name}] EMPTY point cloud")
        return
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    range_xyz = max_xyz - min_xyz
    centroid = coords.mean(axis=0)
    std_xyz = coords.std(axis=0)
    print(f"=== Bounds: {name} ({len(coords)} points, units: {units}) ===")
    print(f"  Min XYZ:  [{min_xyz[0]:.6f}, {min_xyz[1]:.6f}, {min_xyz[2]:.6f}]")
    print(f"  Max XYZ:  [{max_xyz[0]:.6f}, {max_xyz[1]:.6f}, {max_xyz[2]:.6f}]")
    print(f"  Range:    [{range_xyz[0]:.6f}, {range_xyz[1]:.6f}, {range_xyz[2]:.6f}]")
    print(f"  Centroid: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
    print(f"  Std Dev:  [{std_xyz[0]:.6f}, {std_xyz[1]:.6f}, {std_xyz[2]:.6f}]")
    print("=" * 60)

def main():
    print("Starting STRATA Reconstruction Correctness Diagnostic...")
    video_path = repo_root / "data" / "base.mp4"
    if not video_path.exists():
        print(f"Error: {video_path} does not exist.")
        sys.exit(1)

    work_dir = repo_root / "output" / "debug_reconstruction_run"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Video validation & Metadata
    print("\n--- 1. Video Validation & Metadata Extraction ---")
    validate_all(video_path)
    meta = extract_metadata(video_path)
    print(f"Video File: {video_path.name}")
    print(f"Resolution: {meta.width}x{meta.height}")
    print(f"FPS: {meta.fps}, Duration: {meta.duration_sec}s, Frames: {meta.frame_count}")

    # 2. Keyframe Selection
    print("\n--- 2. Keyframe Selection ---")
    frames_res = extract_frames(
        video_path,
        work_dir,
        extraction_mode="target_fps",
        target_fps=1.0,
    )
    selected_dir = work_dir / "selected"
    selected_files = sorted(selected_dir.glob("*.jpg"))
    print(f"Selected Keyframes: {len(selected_files)} / {frames_res['candidates_extracted']}")
    for f in selected_files:
        img = cv2.imread(str(f))
        h, w, c = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"  {f.name}: {w}x{h}, Blur Score={blur:.2f}")

    # 3. Direct PyCOLMAP Execution & Raw Geometry Inspection
    print("\n--- 3. Direct PyCOLMAP Inspection ---")
    import pycolmap
    colmap_images_dir = work_dir / "colmap_images"
    colmap_images_dir.mkdir(parents=True, exist_ok=True)
    for f in selected_files:
        img = cv2.imread(str(f))
        cv2.imwrite(str(colmap_images_dir / f"{f.stem}.png"), img)

    db_path = work_dir / "debug_colmap.db"
    if db_path.exists():
        db_path.unlink()

    sift_opts = pycolmap.SiftExtractionOptions()
    sift_opts.max_num_features = settings.colmap.max_features
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(colmap_images_dir),
        camera_model="PINHOLE",
        sift_options=sift_opts,
    )
    pycolmap.match_sequential(database_path=str(db_path))

    mapping = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(colmap_images_dir),
        output_path=str(work_dir / "debug_colmap_output"),
    )
    reconstructions = getattr(mapping, "reconstructions", None)
    if reconstructions is None:
        reconstructions = [r for r in mapping.values() if r is not None]

    best = max(
        (r for r in reconstructions if r is not None and len(r.images)),
        key=lambda r: len(r.points3D),
        default=None,
    )

    if best is not None:
        print(f"PyCOLMAP Registered Cameras: {len(best.images)}")
        print(f"PyCOLMAP 3D Points: {len(best.points3D)}")

        raw_xyz = np.array([pt.xyz for pt_id, pt in best.points3D.items()], dtype=np.float64)
        raw_errors = [float(pt.error) for pt in best.points3D.values() if pt.error is not None]
        print(f"PyCOLMAP Mean Reprojection Error: {np.mean(raw_errors):.4f} px (min: {np.min(raw_errors):.4f}, max: {np.max(raw_errors):.4f})")
        print_bounds("RAW PyCOLMAP points3D", raw_xyz, "arbitrary SfM units")

        print("\nPyCOLMAP Camera Poses & Intrinsics:")
        cam_positions_c2w = []
        cam_centers_w = []
        for img_id, img in best.images.items():
            cam = best.cameras[img.camera_id]
            cfw = img.cam_from_world()
            R = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
            tvec = np.asarray(cfw.translation, dtype=np.float64)
            # Camera center in world space: C = -R^T @ tvec
            center = -R.T @ tvec
            cam_positions_c2w.append(tvec)
            cam_centers_w.append(center)
            fx, fy, cx, cy = cam.params
            print(f"  Camera {img.name} (ID {img_id}): {cam.model} {cam.width}x{cam.height}, fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            print(f"    tvec (c2w translation): [{tvec[0]:.4f}, {tvec[1]:.4f}, {tvec[2]:.4f}]")
            print(f"    Center (world coords): [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")

        print_bounds("PyCOLMAP tvec (c2w raw translations)", np.array(cam_positions_c2w), "arbitrary SfM units")
        print_bounds("PyCOLMAP True World Camera Centers", np.array(cam_centers_w), "arbitrary SfM units")

    # 4. STRATA estimate_poses & _write_sparse_ply Test
    print("\n--- 4. STRATA estimate_poses Execution ---")
    recon = estimate_poses(selected_dir, "debug_job")
    print(f"STRATA Recon backend: {recon.backend}, registered: {recon.num_registered}, points: {recon.num_points}")
    if recon.points3d:
        strata_xyz = np.vstack([p.position for p in recon.points3d])
        strata_reproj = [p.mean_reproj_error for p in recon.points3d]
        print(f"STRATA SparsePoint3D mean_reproj_error array min={min(strata_reproj)}, max={max(strata_reproj)}")
        print_bounds("STRATA SparsePoint3D positions", strata_xyz, "reconstruction units")

    # 5. Export PLY and Read Back
    print("\n--- 5. Sparse PLY Export & Readback Test ---")
    _write_sparse_ply(recon, work_dir)
    ply_path = work_dir / "sparse_model.ply"
    if ply_path.exists():
        cloud = read_ply(ply_path)
        print_bounds("Exported sparse_model.ply readback", cloud.xyz, "PLY units")
        print(f"Exported residual field min: {cloud.residual.min():.4f}, max: {cloud.residual.max():.4f}, mean: {cloud.residual.mean():.4f}")

if __name__ == "__main__":
    main()
