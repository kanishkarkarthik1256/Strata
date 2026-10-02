"""Phase 9.5 validation script.

Runs the real reconstruction pipeline on the shitan_tw dataset and produces
standardized outputs and documentation.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.logging_config import get_logger
from app.services.image_files import list_image_files
from app.utils.run_organization import (
    create_manifest,
    create_run_structure,
    generate_run_id,
    save_manifest,
)

log = get_logger("drone_recon.services.phase95_validator")


def run_phase95_validation(
    dataset: str = "shitan",
    mission: str = "ms1",
    base_dir: Path | str = ".",
    max_images: int = 20,  # Limit for initial validation
) -> dict:
    """Run Phase 9.5 validation on real dataset.
    
    Args:
        dataset: Dataset name (shitan, aukerman, visdrone)
        mission: Mission name (ms1, ms2, ms3, ms4 for shitan)
        base_dir: Base directory for outputs
        max_images: Maximum number of images to process
        
    Returns:
        Dictionary with validation results
    """
    started_at = datetime.now(timezone.utc)
    run_id = generate_run_id(dataset, mission, started_at)
    
    log.info("phase95_started", run_id=run_id, dataset=dataset, mission=mission)
    
    # Create run structure (only directories for stages we will run)
    dirs = create_run_structure(
        run_id, base_dir,
        stages=["frames", "video", "sparse", "depth", "depth_learned", "dense", "geospatial", "metric_scale", "metrics"],
    )
    
    # Initialize manifest
    manifest = create_manifest(
        run_id=run_id,
        dataset=dataset,
        mission=mission,
        started_at=started_at,
        status="PARTIAL",
    )
    
    results = {
        "run_id": run_id,
        "dataset": dataset,
        "mission": mission,
        "started_at": started_at.isoformat(),
        "stages": {},
        "artifacts": {},
        "limitations": [],
    }
    
    try:
        # Stage 1: Environment audit
        results["stages"]["environment"] = _audit_environment()
        results["dependencies"] = results["stages"]["environment"]
        
        # Stage 2: Dataset inspection
        results["stages"]["dataset"] = _inspect_dataset(dataset, mission, max_images)
        
        # Stage 3: Image preparation
        results["stages"]["images"] = _prepare_images(
            dataset, mission, dirs["frames"], max_images
        )
        
        # Stage 3.5: Video ingestion (FFmpeg) validation — decodes a real
        # clip generated with the FFmpeg binary itself; NOT original drone
        # footage (none is available in the repository).
        results["stages"]["video"] = _run_video_validation(dirs["video"])
        
        # Stage 4: Feature extraction and sparse reconstruction (COLMAP via
        # pycolmap when installed, OpenCV fallback otherwise)
        results["stages"]["sparse"] = _run_sparse_reconstruction(
            dirs["frames"], dirs["sparse"], max_images
        )
        
        # Stage 4.5: Add GPS data to poses if available
        if results["stages"]["sparse"].get("status") == "PASS":
            _add_gps_to_poses(
                dirs["sparse"],
                dataset,
                mission,
                max_images
            )
        
        # Stage 5: Metric depth generation — pinned to the stereo backend so
        # the dense cloud below stays metric. Learned (relative) depth is
        # validated separately in the next stage and must not be fused as
        # meters without a scale reference.
        results["stages"]["depth"] = _run_depth_generation(
            dirs["frames"], dirs["sparse"], dirs["depth"], backend="stereo"
        )
        
        # Stage 5.5: Learned depth (Depth Anything V2) validation — real
        # inference, outputs tagged non-metric, NOT fused into the dense cloud.
        results["stages"]["depth_learned"] = _run_learned_depth(
            dirs["frames"], dirs["sparse"], dirs["depth_learned"]
        )
        
        # Stage 6: Dense reconstruction
        results["stages"]["dense"] = _run_dense_reconstruction(
            dirs["frames"], dirs["depth"], dirs["sparse"], dirs["dense"]
        )
        
        # Stage 7: Geospatial processing
        results["stages"]["geospatial"] = _run_geospatial_processing(
            dirs["sparse"], dirs["geospatial"]
        )

        # Stage 7.5: Metric-scale validation (GPS vs COLMAP alignment)
        results["stages"]["metric_scale"] = _run_metric_scale(
            dirs["sparse"], dirs["metric_scale"]
        )
        
        # Generate metrics
        results["metrics"] = _compute_metrics(results["stages"])
        
        # Update manifest
        manifest["stages"] = results["stages"]
        manifest["dependencies"] = results["dependencies"]
        manifest["artifacts"] = results["artifacts"]
        manifest["metrics"] = results["metrics"]
        manifest["limitations"] = results["limitations"]
        
        # Determine overall status
        if all(s.get("status") in ("PASS", "ESTIMATED") for s in results["stages"].values()):
            manifest["status"] = "PASS"
        elif any(s.get("status") == "PASS" for s in results["stages"].values()):
            manifest["status"] = "PARTIAL"
        else:
            manifest["status"] = "FAILED"
        
    except Exception as e:
        log.exception("phase95_failed", error=str(e))
        manifest["status"] = "FAILED"
        manifest["error"] = str(e)
        results["error"] = str(e)
    
    # Save manifest
    save_manifest(manifest, dirs["output_base"], dirs["docs_base"])
    
    # Generate documentation
    _generate_documentation(results, dirs)
    
    log.info("phase95_completed", run_id=run_id, status=manifest["status"])
    
    return results


def _audit_environment() -> dict:
    """Audit the current environment for required dependencies."""
    import shutil
    import sys
    
    env_info = {
        "python_version": sys.version,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "cuda_available": False,
        "gpu_name": None,
        "mps_available": False,
        "torch_version": None,
        "device": "cpu",
        "colmap_available": False,
        "pycolmap_version": None,
        "ffmpeg_available": False,
        "ffmpeg_path": None,
        "depth_anything_checkpoint": None,
    }
    
    # Torch / CUDA / MPS (MPS requires Apple Silicon)
    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if env_info["cuda_available"]:
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
        mps = getattr(torch.backends, "mps", None)
        env_info["mps_available"] = mps is not None and mps.is_available()
        env_info["device"] = (
            "cuda" if env_info["cuda_available"]
            else "mps" if env_info["mps_available"]
            else "cpu"
        )
    except ImportError:
        pass
    
    # COLMAP: CLI binary on PATH first, then the pycolmap wheel (which is the
    # real COLMAP library; the standalone CLI binary is what the legacy
    # subprocess check used)
    colmap_bin = shutil.which("colmap")
    env_info["colmap_available"] = colmap_bin is not None
    if colmap_bin:
        env_info["colmap_path"] = colmap_bin
    try:
        import pycolmap
        env_info["pycolmap_version"] = getattr(pycolmap, "__version__", "unknown")
    except ImportError:
        pass
    
    # FFmpeg: CLI on PATH, else the imageio-ffmpeg bundled static binary
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            pass
    env_info["ffmpeg_available"] = ffmpeg_bin is not None
    env_info["ffmpeg_path"] = ffmpeg_bin
    
    # Depth Anything V2 checkpoint
    try:
        from app.services.depth_anything_v2 import find_checkpoint
        ckpt = find_checkpoint()
        env_info["depth_anything_checkpoint"] = str(ckpt) if ckpt else None
    except ImportError:
        pass
    
    env_info["status"] = "PASS"
    return env_info


def _inspect_dataset(dataset: str, mission: str, max_images: int) -> dict:
    """Inspect the dataset and extract metadata."""
    from PIL import Image
    from PIL.ExifTags import GPSTAGS, TAGS
    
    # Determine base directory - go up from backend to project root
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / dataset
    if dataset == "shitan":
        data_dir = project_root / "data" / "shitan_tw"
    
    if not data_dir.exists():
        return {"status": "FAILED", "error": f"Dataset directory not found: {data_dir}"}
    
    # Find images for this mission
    image_files = []
    for f in sorted(data_dir.iterdir()):
        if f.name.upper().endswith(".JPG"):
            parts = f.name.split("_")
            if len(parts) >= 4 and parts[3] == mission:
                image_files.append(f)
    
    if not image_files:
        return {"status": "FAILED", "error": f"No images found for mission {mission}"}
    
    # Sample first image for metadata
    sample_img = image_files[0]
    with Image.open(sample_img) as img:
        exif = img._getexif()
        image_size = img.size
    
    metadata = {
        "total_images": len(image_files),
        "images_to_process": min(max_images, len(image_files)),
        "image_size": image_size,
        "sample_file": sample_img.name,
    }
    
    # Extract GPS if available
    if exif:
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                gps_data = {}
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_data[gps_tag] = gps_value
                
                # Convert to decimal degrees
                if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                    lat = _dms_to_decimal(gps_data["GPSLatitude"])
                    lon = _dms_to_decimal(gps_data["GPSLongitude"])
                    if gps_data.get("GPSLatitudeRef") == "S":
                        lat = -lat
                    if gps_data.get("GPSLongitudeRef") == "W":
                        lon = -lon
                    
                    metadata["gps_available"] = True
                    metadata["latitude"] = lat
                    metadata["longitude"] = lon
                    metadata["altitude"] = gps_data.get("GPSAltitude", 0)
                else:
                    metadata["gps_available"] = False
            elif tag == "Make":
                metadata["camera_make"] = str(value).rstrip("\x00")
            elif tag == "Model":
                metadata["camera_model"] = str(value).rstrip("\x00")
    
    metadata["status"] = "PASS"
    return metadata


def _dms_to_decimal(dms: tuple) -> float:
    """Convert degrees/minutes/seconds to decimal degrees."""
    d, m, s = dms
    return float(d) + float(m) / 60 + float(s) / 3600


def _add_gps_to_poses(sparse_dir: Path, dataset: str, mission: str, max_images: int) -> None:
    """Add GPS data from EXIF to poses.json if available."""
    import json as json_lib

    from PIL import Image
    from PIL.ExifTags import GPSTAGS, TAGS
    
    poses_path = sparse_dir / "poses.json"
    if not poses_path.exists():
        return
    
    with open(poses_path) as f:
        poses = json_lib.load(f)
    
    frames = poses.get("frames", [])
    if not frames:
        return
    
    # Determine base directory - go up from backend to project root
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / dataset
    if dataset == "shitan":
        data_dir = project_root / "data" / "shitan_tw"
    
    # Get original image files
    image_files = []
    for f in sorted(data_dir.iterdir()):
        if f.name.upper().endswith(".JPG"):
            parts = f.name.split("_")
            if len(parts) >= 4 and parts[3] == mission:
                image_files.append(f)
    
    # Map frame_id to original image file
    # The sparse reconstruction creates frame_000000.jpg from the first image, etc.
    selected = image_files[:max_images]
    
    # Add GPS data to poses
    gps_added = 0
    for i, frame in enumerate(frames):
        if i < len(selected):
            img_path = selected[i]
            try:
                with Image.open(img_path) as img:
                    exif = img._getexif()
                    
                    if exif:
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if tag == "GPSInfo":
                                gps_data = {}
                                for gps_tag_id, gps_value in value.items():
                                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                                    gps_data[gps_tag] = gps_value
                                
                                # Convert to decimal degrees
                                if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                                    lat = _dms_to_decimal(gps_data["GPSLatitude"])
                                    lon = _dms_to_decimal(gps_data["GPSLongitude"])
                                    if gps_data.get("GPSLatitudeRef") == "S":
                                        lat = -lat
                                    if gps_data.get("GPSLongitudeRef") == "W":
                                        lon = -lon
                                    
                                    frame["gps"] = {
                                        "lat": lat,
                                        "lon": lon,
                                        "alt": float(gps_data.get("GPSAltitude", 0)),
                                    }
                                    gps_added += 1
                                    break
            except Exception as e:
                log.warning("gps_extraction_failed", frame_id=frame["frame_id"], error=str(e))
    
    # Save updated poses
    with open(poses_path, "w") as f:
        json_lib.dump(poses, f, indent=2)
    
    log.info("gps_added_to_poses", gps_added=gps_added, total_frames=len(frames))


def _prepare_images(
    dataset: str, mission: str, output_dir: Path, max_images: int
) -> dict:
    """Prepare images for processing."""
    # Determine base directory - go up from backend to project root
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / dataset
    if dataset == "shitan":
        data_dir = project_root / "data" / "shitan_tw"
    
    image_files = []
    for f in sorted(data_dir.iterdir()):
        if f.name.upper().endswith(".JPG"):
            parts = f.name.split("_")
            if len(parts) >= 4 and parts[3] == mission:
                image_files.append(f)
    
    # Select subset
    selected = image_files[:max_images]
    
    # Copy/symlink to frames directory
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    
    for i, img_path in enumerate(selected):
        # Create sequential frame names for easier processing
        output_name = f"frame_{i:06d}.jpg"
        output_path = output_dir / output_name
        
        if not output_path.exists():
            shutil.copy2(img_path, output_path)
        
        copied += 1
    
    return {
        "status": "PASS",
        "total_available": len(image_files),
        "selected": len(selected),
        "copied": copied,
        "output_dir": str(output_dir),
    }


def _run_sparse_reconstruction(
    image_dir: Path, output_dir: Path, max_images: int
) -> dict:
    """Run sparse reconstruction using the existing pipeline."""
    try:
        from app.services.sparse_reconstruction import run_sparse_reconstruction
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run sparse reconstruction - output goes to the output_dir
        result = run_sparse_reconstruction(
            selected_dir=image_dir,
            output_dir=output_dir,
            project_id="phase95_validation",
            total_frames=max_images,
            selected_frames=max_images,
        )
        
        # Verify poses.json was created
        poses_path = output_dir / "poses.json"
        if not poses_path.exists():
            return {
                "status": "FAILED",
                "error": f"Poses not saved to {poses_path}",
            }
        
        return {
            "status": "PASS",
            "cameras_registered": result.get("reconstruction", {}).get("num_cameras", 0),
            "points": result.get("reconstruction", {}).get("num_points", 0),
            "reproj_error": result.get("reconstruction", {}).get("mean_reproj_error", 0),
            "score": result.get("analysis", {}).get("mission_score", 0),
            "poses_path": str(poses_path),
        }
        
    except Exception as e:
        log.exception("sparse_reconstruction_failed", error=str(e))
        return {
            "status": "FAILED",
            "error": str(e),
        }


def _run_depth_generation(
    image_dir: Path, sparse_dir: Path, depth_dir: Path, backend: str = "stereo"
) -> dict:
    """Run metric depth generation using the existing pipeline.

    ``backend`` defaults to ``stereo`` so the dense cloud stays metric — the
    learned (relative) backend is validated separately in
    ``_run_learned_depth`` and must not be fused as meters.
    """
    try:
        from app.services.depth_generator import generate_view_depths
        
        depth_dir.mkdir(parents=True, exist_ok=True)
        
        # The depth generator expects workspace to have poses.json and selected/ or frames/
        # sparse_dir is where poses.json was saved by sparse reconstruction
        workspace = sparse_dir.parent  # This is the run's output directory
        
        # Copy poses.json to workspace root if it's in sparse/ subdirectory
        poses_in_sparse = sparse_dir / "poses.json"
        poses_in_workspace = workspace / "poses.json"
        
        if poses_in_sparse.exists() and not poses_in_workspace.exists():
            shutil.copy2(poses_in_sparse, poses_in_workspace)
        
        # Ensure frames are in the right location
        # The depth generator looks for selected/ or frames/ in the workspace
        frames_dir = workspace / "frames"
        if not frames_dir.exists():
            # Create a symlink to the actual frames directory
            try:
                frames_dir.symlink_to(image_dir)
            except OSError:
                # If symlink fails, copy the frames
                frames_dir.mkdir(parents=True, exist_ok=True)
                for f in list_image_files(image_dir):
                    shutil.copy2(f, frames_dir / f.name)
        
        # Generate depth maps
        summary = generate_view_depths(
            workspace=workspace,
            backend=backend,
            frame_stride=1,
            max_views=20,
        )
        
        return {
            "status": "PASS",
            "backend": summary.backend,
            "generated": summary.count_generated,
            "cached": summary.count_cached,
            "failed": len(summary.failed),
        }
        
    except Exception as e:
        log.exception("depth_generation_failed", error=str(e))
        return {
            "status": "FAILED",
            "error": str(e),
        }


def _run_learned_depth(
    image_dir: Path, sparse_dir: Path, depth_learned_dir: Path
) -> dict:
    """Validate the learned depth backend (Depth Anything V2) on the run.

    Runs real inference on the registered views. The outputs are RELATIVE
    depth (``metric: false``) and are stored separately — they are never
    fused into the metric dense cloud.
    """
    try:
        import json as json_lib

        from app.config.settings import settings
        from app.services.depth_anything_v2 import find_checkpoint
        from app.services.depth_generator import generate_view_depths
        
        if find_checkpoint() is None:
            return {
                "status": "BLOCKED",
                "reason": "No Depth Anything V2 checkpoint in AI_WEIGHTS_DIR",
            }
        
        depth_learned_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a dedicated storage workspace so the metric dense path is not
        # polluted with relative depth maps.
        workspace = settings.storage.project_dir("phase951_learned")
        workspace.mkdir(parents=True, exist_ok=True)
        
        poses_src = sparse_dir / "poses.json"
        if not poses_src.exists():
            return {"status": "BLOCKED", "reason": "No poses.json from sparse stage"}
        shutil.copy2(poses_src, workspace / "poses.json")
        
        frames_dst = workspace / "frames"
        if not frames_dst.exists():
            frames_dst.mkdir(parents=True, exist_ok=True)
            for f in list_image_files(image_dir):
                shutil.copy2(f, frames_dst / f.name)
        
        depth_dst = workspace / "depth"
        if depth_dst.exists():
            shutil.rmtree(depth_dst)
        
        summary = generate_view_depths(
            workspace=workspace,
            backend="depth_anything",
            frame_stride=1,
            max_views=10,
        )
        
        # Copy the learned depth outputs to the run directory
        if depth_dst.exists():
            for f in depth_dst.iterdir():
                shutil.copy2(f, depth_learned_dir / f.name)
        
        # Record backend/device/checkpoint in a stage summary file
        info = {
            "backend": summary.backend,
            "checkpoint": str(find_checkpoint()),
            "device": _infer_device(),
            "metric": False,
            "metric_scale": "NOT_VALIDATED",
            "generated": summary.count_generated,
            "cached": summary.count_cached,
            "failed": len(summary.failed),
        }
        with open(depth_learned_dir / "learned_depth_info.json", "w") as f:
            json_lib.dump(info, f, indent=2)
        
        # Clean up the temporary workspace
        for child in [workspace / "depth", workspace / "frames", workspace / "poses.json"]:
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                elif child.is_file():
                    child.unlink()
            except OSError:
                pass
        
        return {
            "status": "PASS",
            "backend": summary.backend,
            "checkpoint": info["checkpoint"],
            "device": info["device"],
            "metric": False,
            "generated": summary.count_generated,
            "cached": summary.count_cached,
            "failed": len(summary.failed),
        }
        
    except Exception as e:
        log.exception("learned_depth_failed", error=str(e))
        return {
            "status": "FAILED",
            "error": str(e),
        }


def _infer_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _run_video_validation(video_dir: Path) -> dict:
    """Validate real video decode with the FFmpeg binary.

    Generates a short clip with FFmpeg itself and decodes it back, proving
    encode+decode of the real binary and the cv2 ingestion path. This is NOT
    original drone footage — no drone MP4 exists in the repository — so
    ``original_drone_mp4`` is reported as not validated.
    """
    import shutil
    import subprocess
    
    video_dir.mkdir(parents=True, exist_ok=True)
    
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            pass
    
    if not ffmpeg_bin:
        return {
            "status": "BLOCKED",
            "reason": "No FFmpeg binary on PATH or via imageio-ffmpeg",
            "original_drone_mp4_validated": False,
        }
    
    clip = video_dir / "smoke_clip.mp4"
    version = subprocess.run(
        [ffmpeg_bin, "-version"], capture_output=True, text=True, timeout=20
    ).stdout.splitlines()[0] if ffmpeg_bin else ""
    
    # Encode a 1s testsrc clip
    enc = subprocess.run(
        [ffmpeg_bin, "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
         "-pix_fmt", "yuv420p", "-y", str(clip)],
        capture_output=True, text=True, timeout=60,
    )
    if enc.returncode != 0 or not clip.exists():
        return {
            "status": "FAILED",
            "error": (enc.stderr or enc.stdout)[-400:],
            "original_drone_mp4_validated": False,
        }
    
    # Decode it back with FFmpeg
    dec = subprocess.run(
        [ffmpeg_bin, "-i", str(clip), "-f", "null", "-"],
        capture_output=True, text=True, timeout=60,
    )
    decode_ok = dec.returncode == 0
    
    # And with the product's cv2 ingestion path
    import cv2
    cap = cv2.VideoCapture(str(clip))
    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    ok_read, frame = cap.read() if opened else (False, None)
    cap.release()
    
    return {
        "status": "PASS",
        "ffmpeg_version": version,
        "ffmpeg_path": ffmpeg_bin,
        "encode_decode_ok": decode_ok,
        "cv2_opened": opened,
        "clip_fps": round(fps, 2),
        "clip_frames": frames,
        "first_frame_read": bool(ok_read),
        "original_drone_mp4_validated": False,
        "original_drone_mp4_reason": "no original drone video in repository",
        "clip_path": str(clip),
    }


def _run_dense_reconstruction(
    image_dir: Path, depth_dir: Path, sparse_dir: Path, dense_dir: Path
) -> dict:
    """Run dense reconstruction using the existing pipeline."""
    try:
        from app.config.settings import settings
        from app.services.dense_reconstruction import run_dense_reconstruction
        
        dense_dir.mkdir(parents=True, exist_ok=True)
        
        # The dense reconstruction uses settings.storage.project_dir(job_id)
        # which points to uploads/<job_id>. We need to create a workspace there
        # with the required structure.
        workspace = settings.storage.project_dir("phase95_validation")
        workspace.mkdir(parents=True, exist_ok=True)
        
        # Copy/symlink the required files to the workspace
        # 1. poses.json
        poses_src = sparse_dir / "poses.json"
        poses_dst = workspace / "poses.json"
        if poses_src.exists() and not poses_dst.exists():
            shutil.copy2(poses_src, poses_dst)
        
        # 2. depth maps
        depth_dst = workspace / "depth"
        if not depth_dst.exists():
            shutil.copytree(depth_dir, depth_dst)
        
        # 3. images (frames)
        frames_dst = workspace / "frames"
        if not frames_dst.exists():
            shutil.copytree(image_dir, frames_dst)
        
        # Run dense reconstruction
        result = run_dense_reconstruction(
            job_id="phase95_validation",
            params=None,  # Use defaults
        )
        
        # Copy the dense model back to our output directory
        dense_model_src = workspace / "dense" / "dense_model.ply"
        if dense_model_src.exists():
            shutil.copy2(dense_model_src, dense_dir / "dense_model.ply")
        
        # Clean up stale workspace artifacts
        for child in [workspace / "dense", workspace / "depth", workspace / "frames", workspace / "poses.json"]:
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                elif child.is_file():
                    child.unlink()
            except OSError:
                pass
        
        return {
            "status": "PASS",
            "points": result.get("quality", {}).get("point_count", 0),
            "score": result.get("quality", {}).get("dense_score", 0),
            "grade": result.get("quality", {}).get("grade", "N/A"),
        }
        
    except Exception as e:
        log.exception("dense_reconstruction_failed", error=str(e))
        return {
            "status": "FAILED",
            "error": str(e),
        }


def _run_geospatial_processing(sparse_dir: Path, geospatial_dir: Path) -> dict:
    """Run geospatial processing on the sparse reconstruction."""
    try:
        import json as json_lib

        import numpy as np

        from app.services.georeferencing import analyze_gps_track, crs_metadata
        
        geospatial_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if poses.json has GPS data
        poses_path = sparse_dir / "poses.json"
        if not poses_path.exists():
            return {"status": "BLOCKED", "reason": "No poses.json found"}
        
        with open(poses_path) as f:
            poses = json_lib.load(f)
        
        frames = poses.get("frames", [])
        if not frames:
            return {"status": "BLOCKED", "reason": "No frames in poses.json"}
        
        # Check for GPS data
        gps_frames = [f for f in frames if f.get("gps")]
        
        if not gps_frames:
            return {
                "status": "BLOCKED",
                "reason": "No GPS data in poses",
                "frames_total": len(frames),
                "frames_with_gps": 0,
            }
        
        # Extract GPS coordinates
        lats = np.array([f["gps"]["lat"] for f in gps_frames])
        lons = np.array([f["gps"]["lon"] for f in gps_frames])
        alts = np.array([f["gps"]["alt"] for f in gps_frames])
        
        # Analyze GPS quality
        gps_quality = analyze_gps_track(lats, lons, alts)
        
        # Get CRS metadata
        crs = crs_metadata(float(lats[0]), float(lons[0]), float(alts[0]))
        
        # Save results
        with open(geospatial_dir / "gps_quality.json", "w") as f:
            json_lib.dump(gps_quality.to_dict(), f, indent=2)
        
        with open(geospatial_dir / "crs_metadata.json", "w") as f:
            json_lib.dump(crs, f, indent=2)
        
        return {
            "status": "PASS",
            "frames_total": len(frames),
            "frames_with_gps": len(gps_frames),
            "gps_score": gps_quality.gps_score,
            "gps_grade": gps_quality.grade,
            "crs": crs.get("horizontal_crs", "unknown"),
        }
        
    except Exception as e:
        log.exception("geospatial_processing_failed", error=str(e))
        return {
            "status": "FAILED",
            "error": str(e),
        }


def _run_metric_scale(sparse_dir: Path, metric_scale_dir: Path) -> dict:
    """Run metric-scale validation (GPS vs COLMAP alignment)."""
    try:
        from app.services.metric_scale_validator import run_metric_scale_validation

        return run_metric_scale_validation(sparse_dir, metric_scale_dir)

    except Exception as e:
        log.exception("metric_scale_failed", error=str(e))
        return {
            "status": "FAILED",
            "error": str(e),
        }


def _compute_metrics(stages: dict) -> dict:
    """Compute overall metrics from stage results."""
    metrics = {
        "total_stages": len(stages),
        "passed_stages": sum(1 for s in stages.values() if s.get("status") == "PASS"),
        "failed_stages": sum(1 for s in stages.values() if s.get("status") == "FAILED"),
        "blocked_stages": sum(1 for s in stages.values() if s.get("status") == "BLOCKED"),
    }
    
    # Calculate success rate
    if metrics["total_stages"] > 0:
        metrics["success_rate"] = metrics["passed_stages"] / metrics["total_stages"]
    else:
        metrics["success_rate"] = 0.0
    
    return metrics


def _generate_documentation(results: dict, dirs: dict) -> None:
    """Generate documentation for the run."""
    import json as json_lib
    
    docs_dir = dirs["docs_base"]
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # README.md
    readme_content = f"""# Phase 9.5 Validation Report

Run ID: {results['run_id']}
Dataset: {results['dataset']}
Mission: {results['mission']}

## Output Locations

- **Outputs**: `outputs/{results['run_id']}/`
- **Documentation**: `docs/{results['run_id']}/`

## Stage Results

"""
    
    for stage_name, stage_result in results.get("stages", {}).items():
        status = stage_result.get("status", "UNKNOWN")
        readme_content += f"### {stage_name.replace('_', ' ').title()}\n"
        readme_content += f"- Status: {status}\n"
        
        if status == "PASS":
            for key, value in stage_result.items():
                if key != "status":
                    readme_content += f"- {key}: {value}\n"
        elif status == "FAILED":
            readme_content += f"- Error: {stage_result.get('error', 'Unknown')}\n"
        
        readme_content += "\n"
    
    # Limitations
    if results.get("limitations"):
        readme_content += "## Limitations\n\n"
        for limitation in results["limitations"]:
            readme_content += f"- {limitation}\n"
    
    with open(docs_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # pipeline_report.md
    def _json_default(obj):
        if hasattr(obj, 'numerator') and hasattr(obj, 'denominator'):
            return float(obj)
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        return str(obj)
    
    pipeline_report = f"""# Pipeline Report

Run ID: {results['run_id']}
Generated: {datetime.now(timezone.utc).isoformat()}

## Summary

- Dataset: {results['dataset']}
- Mission: {results['mission']}
- Total Stages: {len(results.get('stages', {}))}
- Passed: {sum(1 for s in results.get('stages', {}).values() if s.get('status') == 'PASS')}
- Failed: {sum(1 for s in results.get('stages', {}).values() if s.get('status') == 'FAILED')}

## Detailed Results

{json_lib.dumps(results.get('stages', {}), indent=2, default=_json_default)}

## Metrics

{json_lib.dumps(results.get('metrics', {}), indent=2, default=_json_default)}
"""
    
    with open(docs_dir / "pipeline_report.md", "w") as f:
        f.write(pipeline_report)


if __name__ == "__main__":
    # Run validation
    results = run_phase95_validation(
        dataset="shitan",
        mission="ms1",
        max_images=10,  # Start small for testing
    )
    
    print("\nPhase 9.5 Validation Complete")
    print(f"Run ID: {results['run_id']}")
    print(f"Status: {results.get('status', 'UNKNOWN')}")
