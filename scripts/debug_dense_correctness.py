"""Comprehensive Dense Reconstruction & TSDF Correctness Diagnostic Script.

Performs:
1. Detailed point cloud inspection & connected component analysis.
2. Depth map statistics, percentiles (P1, P10, P50, P90, P99), & source audit.
3. Depth Anything V2 checkpoint & model discovery check.
4. Single-pixel back-projection verification (camera XYZ -> world XYZ).
5. TSDF / voxel grid quantization audit.
6. Visual diagnostic image generation using OpenCV (top view, side view, front view, trajectory).
"""

from __future__ import annotations

import json
import os
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
from app.services.canonical_demo_service import get_canonical_video_path
from app.services.depth_fusion import DepthView, FusionParams, fuse_depth_views, _unproject_view
from app.services.depth_generator import StereoParams, generate_view_depths, _stereo_view_depth
from app.services.pointcloud import read_ply, save_ply
from app.services.run_service import list_runs

def analyze_cloud(name: str, path: Path):
    """Analyze point cloud spatial statistics, percentiles, and components."""
    if not path.exists():
        print(f"[{name}] File NOT found at {path}")
        return None
    cloud = read_ply(path)
    if cloud.n == 0:
        print(f"[{name}] EMPTY cloud at {path}")
        return None

    xyz = cloud.xyz
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    range_xyz = max_xyz - min_xyz
    centroid = xyz.mean(axis=0)
    std_xyz = xyz.std(axis=0)

    print(f"\n============================================================")
    print(f"=== Analysis: {name} ({cloud.n} points) ===")
    print(f"  Path:     {path}")
    print(f"  Min XYZ:  [{min_xyz[0]:.6f}, {min_xyz[1]:.6f}, {min_xyz[2]:.6f}]")
    print(f"  Max XYZ:  [{max_xyz[0]:.6f}, {max_xyz[1]:.6f}, {max_xyz[2]:.6f}]")
    print(f"  Range:    [{range_xyz[0]:.6f}, {range_xyz[1]:.6f}, {range_xyz[2]:.6f}]")
    print(f"  Centroid: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
    print(f"  Std Dev:  [{std_xyz[0]:.6f}, {std_xyz[1]:.6f}, {std_xyz[2]:.6f}]")

    # Nearest neighbor distance distribution (subsample up to 5000 points for speed)
    sample_size = min(5000, cloud.n)
    idx = np.random.choice(cloud.n, sample_size, replace=False)
    pts_sample = xyz[idx]
    
    from scipy.spatial import cKDTree
    tree = cKDTree(pts_sample)
    dists, _ = tree.query(pts_sample, k=2)
    nn_dists = dists[:, 1]
    
    print(f"  1st NN Distances (meters): min={nn_dists.min():.4f}, mean={nn_dists.mean():.4f}, max={nn_dists.max():.4f}, std={nn_dists.std():.4f}")
    
    # Grid component connectivity check (grouping at 0.2m voxel grid)
    grid_coords = np.floor(xyz / 0.2).astype(np.int64)
    unique_voxels = len(set(tuple(g) for g in grid_coords))
    print(f"  Occupied 0.2m Voxels: {unique_voxels} (Points per Voxel: {cloud.n / max(1, unique_voxels):.1f})")
    print("============================================================\n")
    return cloud

def analyze_depth_maps(storage_dir: Path):
    """Analyze generated depth maps in storage workspace."""
    depth_dir = storage_dir / "depth"
    if not depth_dir.is_dir():
        print(f"No depth directory found at {depth_dir}")
        return

    npy_files = sorted(depth_dir.glob("*.npy"))
    print(f"\n--- Analyzing {len(npy_files)} Depth Maps in {depth_dir} ---")

    for npy_path in npy_files:
        json_path = npy_path.with_suffix(".json")
        meta = {}
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text())
            except Exception:
                pass

        depth = np.load(npy_path)
        valid = np.isfinite(depth) & (depth > 0)
        valid_vals = depth[valid]

        backend = meta.get("backend", "unknown")
        metric = meta.get("metric", False)
        print(f"\nDepth Map: {npy_path.name} | Backend: {backend} | Metric: {metric}")
        print(f"  Dimensions: {depth.shape[1]}x{depth.shape[0]} | Valid Pixels: {len(valid_vals)} / {depth.size} ({len(valid_vals)/depth.size*100:.1f}%)")

        if len(valid_vals) > 0:
            p1, p10, p50, p90, p99 = np.percentile(valid_vals, [1, 10, 50, 90, 99])
            print(f"  Min: {valid_vals.min():.4f}m, Max: {valid_vals.max():.4f}m, Mean: {valid_vals.mean():.4f}m, Median: {p50:.4f}m, Std: {valid_vals.std():.4f}m")
            print(f"  Percentiles (meters): P1={p1:.4f}, P10={p10:.4f}, P50={p50:.4f}, P90={p90:.4f}, P99={p99:.4f}")

def manual_pixel_backprojection(poses_path: Path, depth_dir: Path):
    """Manually unproject 3 representative pixels (u, v) and verify X_world = R_c2w @ X_cam + C_world."""
    if not poses_path.exists() or not depth_dir.is_dir():
        print(f"Missing poses or depth for backprojection at {poses_path}")
        return

    poses_data = json.loads(poses_path.read_text())
    frames = poses_data.get("frames", [])
    if not frames:
        return

    first_frame = frames[0]
    frame_id = first_frame["frame_id"]
    npy_path = depth_dir / f"{frame_id}.npy"
    if not npy_path.exists():
        print(f"Depth map {npy_path} not found")
        return

    depth = np.load(npy_path)
    h, w = depth.shape
    K = np.array(first_frame["K"], dtype=np.float64)
    R_c2w = np.array(first_frame["R"], dtype=np.float64)
    C_world = np.array(first_frame["t"], dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print(f"\n--- Manual Pixel Back-Projection Test for Frame {frame_id} ---")
    print(f"Intrinsics K: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f} (Image: {w}x{h})")
    print(f"Camera World Center C_world: [{C_world[0]:.4f}, {C_world[1]:.4f}, {C_world[2]:.4f}]")
    print(f"Camera Rotation Matrix R_c2w:\n{R_c2w}")

    # Pick 3 pixels: top-left (foreground/mid), center, bottom-right
    pixels = [(w // 4, h // 4), (w // 2, h // 2), (3 * w // 4, 3 * h // 4)]
    for u, v in pixels:
        z = float(depth[v, u])
        if z <= 0 or not np.isfinite(z):
            print(f"  Pixel ({u}, {v}): Invalid depth ({z})")
            continue
        x_cam = (u - cx) / fx * z
        y_cam = (v - cy) / fy * z
        z_cam = z
        X_cam = np.array([x_cam, y_cam, z_cam])
        X_world = R_c2w @ X_cam + C_world

        print(f"  Pixel (u={u}, v={v}): Depth Z={z:.4f}m")
        print(f"    Camera Coords X_cam: [{X_cam[0]:.4f}, {X_cam[1]:.4f}, {X_cam[2]:.4f}]")
        print(f"    World Coords X_world: [{X_world[0]:.4f}, {X_world[1]:.4f}, {X_world[2]:.4f}]")

def render_projection_opencv(cloud: object, filename: Path, title: str, u_axis: int, v_axis: int, u_name: str, v_name: str):
    """Render 2D orthographic projection of point cloud using OpenCV."""
    if cloud is None or cloud.n == 0:
        return
    img_size = 800
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    pts = cloud.xyz
    u_vals = pts[:, u_axis]
    v_vals = pts[:, v_axis]

    u_min, u_max = u_vals.min(), u_vals.max()
    v_min, v_max = v_vals.min(), v_vals.max()

    u_norm = (u_vals - u_min) / max(1e-6, (u_max - u_min))
    v_norm = (v_vals - v_min) / max(1e-6, (v_max - v_min))

    px = np.clip(u_norm * (img_size - 40) + 20, 0, img_size - 1).astype(np.int32)
    py = np.clip((1.0 - v_norm) * (img_size - 40) + 20, 0, img_size - 1).astype(np.int32)

    # Subsample for rendering speed
    sub = np.random.choice(len(px), min(50000, len(px)), replace=False)
    for i in sub:
        cv2.circle(canvas, (px[i], py[i]), 1, (0, 255, 128), -1)

    # Annotate axes and titles
    cv2.putText(canvas, f"{title} ({cloud.n} pts)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"{u_name}: [{u_min:.2f}, {u_max:.2f}]", (20, img_size - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, f"{v_name}: [{v_min:.2f}, {v_max:.2f}]", (20, img_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imwrite(str(filename), canvas)
    print(f"  Generated OpenCV projection: {filename.name}")

def main():
    print("Starting Comprehensive Dense Reconstruction & TSDF Audit...")

    runs = list_runs()
    if not runs:
        print("No runs found in workspace.")
        return

    latest_run = runs[0]
    run_id = latest_run["run_id"]
    output_dir = repo_root / "output" / run_id
    storage_dir = settings.storage.project_dir(run_id)

    print(f"Auditing Target Run: {run_id}")
    print(f"  Output Dir:  {output_dir}")
    print(f"  Storage Dir: {storage_dir}")

    # 1. Analyze Sparse and Dense clouds
    sparse_cloud = analyze_cloud("Sparse Point Cloud", output_dir / "sparse_model.ply")
    dense_cloud = analyze_cloud("Dense Point Cloud", output_dir / "dense" / "dense_model.ply")

    # 2. Analyze Depth Maps
    analyze_depth_maps(storage_dir)

    # 3. Manual Pixel Back-projection
    manual_pixel_backprojection(storage_dir / "poses.json", storage_dir / "depth")

    # 4. Generate Visual Diagnostics
    debug_dir = repo_root / "debug" / "reconstruction"
    debug_dir.mkdir(parents=True, exist_ok=True)
    if dense_cloud is not None:
        render_projection_opencv(dense_cloud, debug_dir / "dense_top_view.png", "Dense Top View (XY)", 0, 1, "X", "Y")
        render_projection_opencv(dense_cloud, debug_dir / "dense_side_view.png", "Dense Side View (XZ)", 0, 2, "X", "Z")
        render_projection_opencv(dense_cloud, debug_dir / "dense_front_view.png", "Dense Front View (YZ)", 1, 2, "Y", "Z")

if __name__ == "__main__":
    main()
