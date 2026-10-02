"""Object detection over the registered views (guarded YOLO backend).

Detections are 2D boxes with a class + confidence. Each detection is linked
to a 3D world point by unprojecting its box centre through the frame's pose
and depth map — the same math the twin engine uses to geospatialise objects.

The YOLO backend (``ultralytics`` or an ONNX export) is optional and
guarded: without the runtime/weights the stage records zero detections with
an explicit reason instead of fabricating boxes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.image_files import list_image_files
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.object_detector")


def unproject_pixel(px: np.ndarray, depth: float, K: np.ndarray, R: np.ndarray,
                    t: np.ndarray) -> np.ndarray:
    """Back-project a pixel (x, y) at camera-space depth *depth* to world."""
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (px[0] - cx) * depth / fx
    y_cam = (px[1] - cy) * depth / fy
    return R.T @ (np.array([x_cam, y_cam, depth]) - t)


def link_detections_to_world(detections: list[dict], frame_meta: dict) -> list[dict]:
    """Attach world coordinates + camera info to each 2D detection.

    ``frame_meta`` must carry ``K``/``R``/``t``/``frame_id`` and either a
    ``depth_map`` (np.ndarray) or a ``depth_at_centre`` float. Detections
    without a valid depth are kept with ``world=None``.
    """
    K = np.asarray(frame_meta["K"], dtype=np.float64)
    R = np.asarray(frame_meta["R"], dtype=np.float64)
    t = np.asarray(frame_meta["t"], dtype=np.float64)
    out = []
    for det in detections:
        bx = det.get("bbox", [0, 0, 0, 0])
        centre = np.array([(bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0])
        depth = None
        if "depth_map" in frame_meta and frame_meta["depth_map"] is not None:
            dm = frame_meta["depth_map"]
            # ``centre`` is in FRAME pixels (the detector ran on the frame),
            # while the depth map may be stored in a coarser grid. Scale the
            # lookup into the map's grid; the unprojection below still uses
            # the frame K with the frame-pixel centre.
            sx, sy = frame_meta.get("depth_scale", (1.0, 1.0))
            u = int(np.clip(round(centre[0] * sx), 0, dm.shape[1] - 1))
            v = int(np.clip(round(centre[1] * sy), 0, dm.shape[0] - 1))
            depth = float(dm[v, u]) if dm[v, u] > 0 else None
        elif "depth_at_centre" in frame_meta:
            depth = float(frame_meta["depth_at_centre"])
        entry = {
            **det,
            "frame_id": frame_meta["frame_id"],
            "world": None,
        }
        if depth and depth > 0:
            entry["world"] = unproject_pixel(centre, depth, K, R, t).tolist()
        out.append(entry)
    return out


def _run_yolo(frames_dir: Path) -> tuple[list[dict], str]:
    """Run YOLO detection over the frames (guarded ultralytics/onnx)."""
    try:
        import ultralytics  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        return [], f"yolo backend unavailable: install 'ultralytics' or set SEMANTIC_DETECTION_BACKEND=none ({exc})"
    model_path = Path(settings.ai.yolo_model)
    if not model_path.exists():
        return [], f"yolo weights not found at {model_path}"
    model = ultralytics.YOLO(str(model_path))
    detections: list[dict] = []
    for img_path in list_image_files(frames_dir)[: settings.ai.max_batch_size * 8]:
        results = model(str(img_path), verbose=False)
        for r in results:
            fid = img_path.stem
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append({
                    "frame_id": fid,
                    "class": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [x1, y1, x2, y2],
                })
    return detections, "yolo"


@register
class ObjectDetectionStage(PipelineStage):
    name = "object_detection"
    description = "2D object detection with depth-linked world positions"
    artifact_rel = "objects/detections.json"

    def validate_inputs(self) -> None:
        # Detection needs registered frames + poses for the depth linking.
        if not (self.workspace / "poses.json").exists():
            raise StageNotApplicable("no poses.json — detections cannot be geolinked")

    def execute(self) -> None:
        backend = settings.semantic.detection_backend
        frames_dir = self.workspace / "selected"
        if not frames_dir.is_dir():
            frames_dir = self.workspace / "frames"
        note = ""
        detections: list[dict] = []
        if backend != "none" and frames_dir.is_dir():
            detections, note = _run_yolo(frames_dir)

        # Depth-link detections against per-frame poses + depth maps.
        poses = json.loads((self.workspace / "poses.json").read_text()).get("frames", [])
        pose_by_frame = {p["frame_id"]: p for p in poses}
        depth_dir = self.workspace / "depth"
        linked = []
        for det in detections:
            meta = pose_by_frame.get(det.get("frame_id"))
            if meta is None:
                linked.append({**det, "world": None})
                continue
            dm = None
            scale = (1.0, 1.0)
            for name in (f"{meta['frame_id']}.npy", f"{meta['frame_id']}_depth.npy"):
                p = depth_dir / name
                if p.exists():
                    try:
                        dm = np.load(p)
                        from app.services.depth_generator import read_depth_geometry

                        _K, sx, sy = read_depth_geometry(p, meta)
                        scale = (sx, sy)
                    except (OSError, ValueError):
                        dm = None
                    break
            linked += link_detections_to_world(
                [det], {**meta, "depth_map": dm, "depth_scale": scale}
            )

        obj_dir = self.workspace / "objects"
        obj_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "detections": linked,
            "count": len(linked),
            "backend": "yolo" if detections else "none",
            "note": note or ("no detections — object classes come from the semantic stage"
                             if backend != "none" else "detection disabled (SEMANTIC_DETECTION_BACKEND=none)"),
            "world_linked": sum(1 for d in linked if d.get("world")),
        }
        (obj_dir / "detections.json").write_text(json.dumps(report, indent=2))
        self._count = len(linked)
        self._detail = report
        self._outputs = [{"kind": "data", "name": "detections",
                          "path": str(obj_dir / "detections.json")}]
