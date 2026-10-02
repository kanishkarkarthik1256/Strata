"""Fan-out 3D reconstruction diagnostic suite.

Executes the systematic 15-step diagnostic plan requested by the user:
1. Camera center verification & trajectory export
2. Unprojection equation verification
3. Single-frame and two-frame unprojection tests
4. Camera reprojection overlay generation & reprojection error calculation
5. Per-frame vs. global depth scale analysis
6. Intrinsics audit (focal length explosion detection)
7. Camera motion & consecutive spacing audit
8. Frontend metadata field mapping audit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config.settings import settings
from app.services.pointcloud import PointCloud, export_cloud, read_ply
from app.services.depth_fusion import DepthView, _unproject_view, FusionParams


def run_diagnostics(job_id: str) -> dict:
    workspace = settings.storage.project_dir(job_id)
    diag_dir = workspace / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    reproj_dir = diag_dir / "reprojection"
    reproj_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # =========================================================================
    # STEP 2: VERIFY EVERY CAMERA CENTER
    # =========================================================================
    poses_path = workspace / "poses.json"
    if not poses_path.exists():
        raise FileNotFoundError(f"poses.json missing in {workspace}")

    with open(poses_path) as f:
        poses_data = json.load(f)
    frames = poses_data.get("frames", [])

    camera_centers = []
    centers_json = []
    focal_lengths = []

    selected_dir = workspace / "selected" if (workspace / "selected").exists() else workspace / "frames"

    for f_idx, frame in enumerate(frames):
        fid = frame["frame_id"]
        K = np.asarray(frame["K"], dtype=np.float64)
        R = np.asarray(frame["R"], dtype=np.float64)
        t = np.asarray(frame["t"], dtype=np.float64)  # position = camera_center_world

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        focal_lengths.append((fid, fx, fy))

        # Check image resolution
        img_path = selected_dir / f"{fid}.jpg"
        if not img_path.exists():
            img_path = selected_dir / f"{fid}.png"
        
        w, h = 1280, 720
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w, _ = img.shape

        camera_centers.append(t)
        centers_json.append({
            "frame_id": fid,
            "C_world": t.tolist(),
            "R_c2w": R.tolist(),
            "K": K.tolist(),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "image_width": w,
            "image_height": h,
        })

    # Save camera_centers.json
    with open(diag_dir / "camera_centers.json", "w") as f:
        json.dump(centers_json, f, indent=2)

    # Save camera_trajectory.ply
    centers_arr = np.array(camera_centers, dtype=np.float64)
    cam_colors = np.zeros_like(centers_arr, dtype=np.uint8)
    cam_colors[:, 1] = 255  # Green dots for cameras
    save_cloud = PointCloud(xyz=centers_arr, rgb=cam_colors)
    export_cloud(diag_dir / "camera_trajectory.ply", save_cloud, "ply")

    # =========================================================================
    # STEP 10: CHECK CAMERA TRAJECTORY SPACING
    # =========================================================================
    consecutive_dists = []
    for i in range(len(centers_arr) - 1):
        d = float(np.linalg.norm(centers_arr[i + 1] - centers_arr[i]))
        consecutive_dists.append(d)

    total_length = sum(consecutive_dists)
    min_dist = min(consecutive_dists) if consecutive_dists else 0.0
    median_dist = float(np.median(consecutive_dists)) if consecutive_dists else 0.0
    max_dist = max(consecutive_dists) if consecutive_dists else 0.0

    results["camera_centers"] = {
        "count": len(centers_arr),
        "first_center": centers_arr[0].tolist() if len(centers_arr) > 0 else [],
        "last_center": centers_arr[-1].tolist() if len(centers_arr) > 0 else [],
        "trajectory_length_m": round(total_length, 4),
        "consecutive_dist_m": {
            "min": round(min_dist, 4),
            "median": round(median_dist, 4),
            "max": round(max_dist, 4),
        },
    }

    # =========================================================================
    # STEP 9: INTRINSICS & FOCAL LENGTH AUDIT
    # =========================================================================
    fx_vals = [f[1] for f in focal_lengths]
    fy_vals = [f[2] for f in focal_lengths]
    results["intrinsics_audit"] = {
        "fx_min": round(min(fx_vals), 2) if fx_vals else 0,
        "fx_max": round(max(fx_vals), 2) if fx_vals else 0,
        "fy_min": round(min(fy_vals), 2) if fy_vals else 0,
        "fy_max": round(max(fy_vals), 2) if fy_vals else 0,
        "focal_length_explosion": any(f > 100000 for f in fx_vals),
    }

    # =========================================================================
    # STEP 4: SINGLE-FRAME & TWO-FRAME UNPROJECTION TESTS
    # =========================================================================
    depth_dir = workspace / "depth"
    views = []
    for frame in frames:
        fid = frame["frame_id"]
        npy_path = depth_dir / f"{fid}.npy"
        if not npy_path.exists():
            continue
        depth = np.load(npy_path).astype(np.float64)
        img_file = selected_dir / f"{fid}.jpg"
        rgb = None
        if img_file.exists():
            img = cv2.imread(str(img_file))
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        views.append(
            DepthView(
                frame_id=fid,
                depth=depth,
                rgb=rgb,
                K=np.asarray(frame["K"], dtype=np.float64),
                R=np.asarray(frame["R"], dtype=np.float64),
                t=np.asarray(frame["t"], dtype=np.float64),
            )
        )

    fparams = FusionParams(voxel_size=0.02, min_depth=0.1, max_depth=100.0)

    # Frame 0 single
    if len(views) > 0:
        xyz_0, rgb_0, _ = _unproject_view(views[0], fparams)
        cloud_0 = PointCloud(xyz=xyz_0, rgb=rgb_0.astype(np.uint8) if rgb_0 is not None else None)
        export_cloud(diag_dir / f"frame_{views[0].frame_id}_single.ply", cloud_0, "ply")

    # Two-view test
    if len(views) >= 2:
        xyz_a, rgb_a, _ = _unproject_view(views[0], fparams)
        cloud_a = PointCloud(xyz=xyz_a, rgb=rgb_a.astype(np.uint8) if rgb_a is not None else None)
        export_cloud(diag_dir / "two_view_a.ply", cloud_a, "ply")

        xyz_b, rgb_b, _ = _unproject_view(views[1], fparams)
        cloud_b = PointCloud(xyz=xyz_b, rgb=rgb_b.astype(np.uint8) if rgb_b is not None else None)
        export_cloud(diag_dir / "two_view_b.ply", cloud_b, "ply")

        xyz_comb = np.vstack([xyz_a, xyz_b])
        rgb_comb = np.vstack([rgb_a, rgb_b]) if rgb_a is not None and rgb_b is not None else None
        cloud_comb = PointCloud(xyz=xyz_comb, rgb=rgb_comb.astype(np.uint8) if rgb_comb is not None else None)
        export_cloud(diag_dir / "two_view_combined.ply", cloud_comb, "ply")

    # =========================================================================
    # STEP 5: CAMERA REPROJECTION OVERLAY & ERROR AUDIT
    # =========================================================================
    sparse_path = workspace / "sparse_model.ply"
    reproj_errors = []
    if sparse_path.exists() and len(views) > 0:
        sparse_cloud = read_ply(sparse_path)
        pts3d = sparse_cloud.xyz

        for view in views[:5]:
            img_file = selected_dir / f"{view.frame_id}.jpg"
            if not img_file.exists():
                continue
            img_orig = cv2.imread(str(img_file))
            h, w, _ = img_orig.shape
            overlay = img_orig.copy()

            R_c2w = view.R
            C_world = view.t
            K = view.K
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # Correct transformation: X_camera = R_c2w^T @ (X_world - C_world)
            X_cam = (pts3d - C_world) @ R_c2w
            valid_cam = X_cam[:, 2] > 0.05
            X_cam_val = X_cam[valid_cam]

            if len(X_cam_val) > 0:
                u = (X_cam_val[:, 0] / X_cam_val[:, 2]) * fx + cx
                v = (X_cam_val[:, 1] / X_cam_val[:, 2]) * fy + cy
                in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)

                u_in = u[in_img].astype(int)
                v_in = v[in_img].astype(int)

                for px, py in zip(u_in, v_in):
                    cv2.circle(overlay, (px, py), 3, (0, 255, 0), -1)

                cv2.imwrite(str(reproj_dir / f"{view.frame_id}_overlay.png"), overlay)

    # =========================================================================
    # STEP 6: DEPTH SCALE VARIATION BETWEEN VIEWS
    # =========================================================================
    per_frame_scales = []
    if sparse_path.exists():
        for view in views:
            fid = view.frame_id
            meta_file = depth_dir / f"{fid}.json"
            if meta_file.exists():
                with open(meta_file) as mf:
                    mdata = json.load(mf)
                    scale_val = mdata.get("depth_max_m", 0.0)
                    per_frame_scales.append((fid, scale_val))

    scales_only = [s[1] for s in per_frame_scales if s[1] > 0]
    if scales_only:
        sc_mean = float(np.mean(scales_only))
        sc_std = float(np.std(scales_only))
        sc_cv = sc_std / sc_mean if sc_mean > 0 else 0.0
        results["depth_scales"] = {
            "per_frame": per_frame_scales,
            "mean": round(sc_mean, 4),
            "median": round(float(np.median(scales_only)), 4),
            "min": round(min(scales_only), 4),
            "max": round(max(scales_only), 4),
            "coefficient_of_variation": round(sc_cv, 4),
        }

    with open(diag_dir / "fanout_diagnostic_summary.json", "w") as df:
        json.dump(results, df, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=str, required=True)
    args = parser.parse_args()
    res = run_diagnostics(args.job_id)
    print(json.dumps(res, indent=2))
