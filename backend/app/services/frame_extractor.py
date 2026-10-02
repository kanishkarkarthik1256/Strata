"""Frame extraction orchestrator.

Reads a video once, scores every frame, applies two-stage rejection:
1. Hard reject: blurry, duplicate, extreme exposure
2. Rank survivors by composite score, keep the top N

Memory-efficient: processes one frame at a time, never loads all into RAM.

Output layout under uploads/<job_id>/:
  frames/       — all extracted frames (numbered)
  selected/     — frames that passed quality gates
  rejected/     — frames that were rejected, with reasons
  contact_sheet.jpg — grid thumbnail visualization
  quality_report.json — full quality data for every frame
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config.settings import settings
from app.db.models import Frame, Project
from app.logging_config import get_logger
from app.services.duplicate_detector import compute_phash, is_duplicate_of_kept
from app.services.quality_analyzer import FrameQuality, analyze_frame
from app.exceptions import InvalidVideoError
from app.services.video_validation import ensure_cv2_readable

log = get_logger("drone_recon.services.frame_extractor")


def extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    extraction_mode: str = "target_fps",
    target_fps: float | None = None,
    every_n: int | None = None,
    interval_sec: float | None = None,
    quality_threshold: float | None = None,
    top_percent: float | None = None,
    calibration: dict | None = None,
    frame_budget: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> dict:
    """Extract and score frames from a video.

    Parameters
    ----------
    video_path : Path to the video file.
    output_dir : Base output directory (frames/, selected/, rejected/ are created inside).
    extraction_mode : One of "every_frame", "every_n", "target_fps", "interval".
    target_fps : Target extraction rate (used when mode is "target_fps").
    every_n : Extract every Nth frame (used when mode is "every_n").
    interval_sec : Extract one frame per interval in seconds (used when mode is "interval").
    quality_threshold : Hard minimum composite score to keep a frame.
    top_percent : If set, keep only this percentage of best frames (0.0–1.0).
    frame_budget : If set and more frames pass the quality gates, reduce the
        kept set to at most this many frames via the coverage-aware
        trajectory-bucket selector (``_select_temporal_buckets`` — telemetry
        arclength buckets when flight_poses.csv exists, temporal buckets
        otherwise; best-quality frame within each bucket). This is the
        duration-aware keyframe budget: candidate extraction stays dense
        enough to judge quality, but SfM/depth/dense only process
        geometrically diverse keyframes.
    calibration : Optional dataset intrinsics.json payload. When it carries
        distortion coefficients (k1/k2), frames are undistorted once here so
        every downstream stage (SfM, depth, fusion) sees pinhole geometry —
        the calibration's distortion is then already applied, and K is used
        as-is downstream.

    Returns
    -------
    Dict with extraction results: counts, frame metadata, paths.
    """
    frames_dir = output_dir / "frames"
    selected_dir = output_dir / "selected"
    rejected_dir = output_dir / "rejected"
    for d in (frames_dir, selected_dir, rejected_dir):
        d.mkdir(parents=True, exist_ok=True)

    # cv2 may be unable to decode the upload (FFMPEG-less OpenCV build +
    # H.264/HEVC); ensure_cv2_readable transcodes once via the bundled ffmpeg
    # binary in that case and every decode below uses the returned path.
    video_path = ensure_cv2_readable(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # One-time distortion map: significant radial distortion (e.g. JB3D
    # k1=0.117, k2=-0.220 -> ~10% edge displacement) must be removed before
    # any geometry assumes a pinhole camera.
    map1 = map2 = None
    if calibration and calibration.get("k1") and (
        abs(float(calibration["k1"])) > 1e-6 or abs(float(calibration.get("k2") or 0.0)) > 1e-6
    ):
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if int(calibration.get("width", w)) == w and int(calibration.get("height", h)) == h:
            K = np.array([
                [float(calibration["fx"]), 0, float(calibration["cx"])],
                [0, float(calibration["fy"]), float(calibration["cy"])],
                [0, 0, 1],
            ])
            dist = np.array([
                float(calibration["k1"]), float(calibration.get("k2") or 0.0), 0.0, 0.0,
            ])
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
            log.info("extraction_undistort_enabled", k1=calibration["k1"],
                     k2=calibration.get("k2"), resolution=[w, h])
        else:
            log.warning("extraction_undistort_skipped", reason="calibration resolution mismatch",
                        calib=[calibration.get("width"), calibration.get("height")], video=[w, h])
    # Determine which frame indices to extract
    extract_indices = _compute_extract_indices(total_frames, video_fps, extraction_mode, target_fps, every_n, interval_sec)

    log.info(
        "extraction_started",
        video=str(video_path),
        total_frames=total_frames,
        video_fps=round(video_fps, 2),
        extract_count=len(extract_indices),
    )

    # Phase 1: Extract and score all candidate frames (streaming, one at a time)
    candidates: list[dict] = []
    prev_frame: Optional[np.ndarray] = None
    kept_hashes: list[np.ndarray] = []  # kept in order; dedup uses the tail
    frame_num = 0
    extracted_idx = 0

    # Skip-over without decoding: ``grab()`` advances the stream, ``retrieve()``
    # decodes the current frame. ``read() == grab() + retrieve()``, so the
    # frames we keep are bit-identical — but the frames we discard (the vast
    # majority: a 7-minute 1080p flight is ~12 600 frames for 200 candidates)
    # are no longer decoded at all. This is the stage whose cost scales with
    # VIDEO LENGTH rather than with the candidate count, so it is the one that
    # made long footage slow (measured 199.6 s for 200 views on airport_1).
    while extracted_idx < len(extract_indices):
        target = extract_indices[extracted_idx]
        while frame_num < target:
            if not cap.grab():
                break
            frame_num += 1
        if frame_num < target:
            break  # stream ended before the next candidate index
        if not cap.grab():
            break
        if progress is not None and (extracted_idx % 8 == 0 or extracted_idx + 1 == len(extract_indices)):
            try:
                progress(min(1.0, (extracted_idx + 1) / max(1, len(extract_indices))))
            except Exception:  # progress must never break extraction
                log.exception("frames_progress_publish_failed")
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            break
        frame_num += 1

        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        timestamp = (frame_num - 1) / video_fps
        quality = analyze_frame(frame, prev_frame)

        # Dedup against the RECENT kept window only. A global pHash check
        # (vs every frame ever kept) deletes genuine re-observations on
        # translating aerial footage — the whole scene drifts slowly so
        # later viewpoints hash-close-match early ones (measured: 85 of
        # 208 candidates culled, selection collapsed to one flight
        # segment). Static-scene redundancy is a local property: recent
        # frames suffice.
        recent_hashes = kept_hashes[-RECENT_DEDUP_WINDOW:]
        if quality.rejection_reason == "" and is_duplicate_of_kept(
            frame, recent_hashes, motion=quality.motion
        ):
            quality.rejection_reason = "duplicate_of_kept"

        # Save the frame
        frame_filename = f"frame_{extracted_idx:06d}.jpg"
        frame_path = frames_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

        candidate = {
            "index": extracted_idx,
            "frame_num": frame_num - 1,
            "timestamp_sec": round(timestamp, 3),
            "filename": frame_filename,
            "file_path": str(frame_path),
            "quality": quality,
            "kept": quality.rejection_reason == "",
        }
        candidates.append(candidate)

        if candidate["kept"]:
            kept_hashes.append(compute_phash(frame))
            # Copy to selected
            import shutil
            shutil.copy2(frame_path, selected_dir / frame_filename)

        prev_frame = frame
        extracted_idx += 1

    cap.release()

    # Phase 2: Rank and select top N if top_percent is specified.
    # Coverage-aware: pure quality ranking collapses the selection onto the
    # single highest-quality segment of the timeline (measured: 9 frames in
    # one 26 s window of a 415 s flight), which starves SfM of scene
    # coverage. Splitting the timeline into equal temporal buckets and
    # picking the best frame *within* each bucket preserves coverage while
    # still preferring sharp, well-exposed frames.
    if top_percent is not None and top_percent < 1.0:
        kept = [c for c in candidates if c["kept"]]
        rejected = [c for c in candidates if not c["kept"]]
        keep_count = max(1, int(len(kept) * top_percent))
        chosen = _select_temporal_buckets(kept, keep_count, video_path)

        chosen_set = {id(c) for c in chosen}
        for c in kept:
            if id(c) not in chosen_set:
                c["kept"] = False
                c["quality"].rejection_reason = "below_top_threshold"
                # Move from selected to rejected
                src = selected_dir / c["filename"]
                if src.exists():
                    src.unlink()
                rejected.append(c)

        candidates = chosen + rejected

    # Phase 3: Apply quality threshold
    if quality_threshold is not None:
        for c in candidates:
            if c["kept"] and c["quality"].composite < quality_threshold:
                c["kept"] = False
                c["quality"].rejection_reason = "below_quality_threshold"
                src = selected_dir / c["filename"]
                if src.exists():
                    src.unlink()

    # Phase 3b: Duration-aware keyframe budget. Apply AFTER quality gates so
    # only geometrically diverse, sharp, well-exposed frames survive, and
    # BEFORE the report so the budget decision is recorded. Coverage decides
    # across buckets (trajectory arclength when telemetry exists), quality
    # decides within a bucket — the same selector Phase-2 already trusts.
    budget_applied = False
    if frame_budget is not None:
        kept_now = [c for c in candidates if c["kept"]]
        if len(kept_now) > frame_budget:
            chosen = _select_temporal_buckets(kept_now, frame_budget, video_path)
            chosen_set = {id(c) for c in chosen}
            for c in kept_now:
                if id(c) not in chosen_set:
                    c["kept"] = False
                    c["quality"].rejection_reason = "above_frame_budget"
                    src = selected_dir / c["filename"]
                    if src.exists():
                        src.unlink()
            budget_applied = True

    # Phase 4: Move rejected frames
    for c in candidates:
        if not c["kept"]:
            src = frames_dir / c["filename"]
            if src.exists():
                import shutil
                shutil.move(str(src), str(rejected_dir / c["filename"]))

    # Count final results
    selected_count = sum(1 for c in candidates if c["kept"])
    rejected_count = len(candidates) - selected_count

    # Generate contact sheet
    _generate_contact_sheet(candidates, selected_dir, rejected_dir, output_dir)

    # Generate quality report
    report = {
        "video_path": str(video_path),
        "extraction_mode": extraction_mode,
        "total_video_frames": total_frames,
        "video_fps": round(video_fps, 2),
        "candidates_extracted": len(candidates),
        "selected_count": selected_count,
        "rejected_count": rejected_count,
        "frame_budget": frame_budget,
        "frame_budget_applied": budget_applied,
        "frames": [
            {
                "index": c["index"],
                "frame_num": c["frame_num"],
                "timestamp_sec": c["timestamp_sec"],
                "filename": c["filename"],
                "kept": c["kept"],
                "rejection_reason": c["quality"].rejection_reason or None,
                "scores": {
                    "blur": round(float(c["quality"].blur), 4),
                    "sharpness": round(float(c["quality"].sharpness), 4),
                    "exposure": round(float(c["quality"].exposure), 4),
                    "motion": round(float(c["quality"].motion), 4),
                    "composite": round(float(c["quality"].composite), 4),
                },
            }
            for c in candidates
        ],
    }

    report_path = output_dir / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(
        "extraction_complete",
        candidates=len(candidates),
        selected=selected_count,
        rejected=rejected_count,
    )

    return report


#: Dedup compares each candidate against the last N kept frames. At the
#: default 0.5–2 fps extraction rates this is a 5–20 s window: long enough
#: to catch hover/drift redundancy, short enough that legitimate
#: re-observations of the same scene area survive as loop-closure views.
RECENT_DEDUP_WINDOW = 10


#: seek_to_index() verifies position by decoding up to this many frames
#: past the requested index (OpenCV seeks are keyframe-aligned, so small
#: overshoots are normal). Keeps decode cost bounded per regenerated frame.
_SEEK_VERIFY_LIMIT = 30


def seek_to_index(video_path: Path, index: int, out_path: Path) -> bool:
    """Regenerate ONE candidate frame from *video_path* by decode index.

    Writes ``out_path`` (JPEG) only when the file does not already exist —
    an existing frame is never overwritten, so a regenerated file can never
    diverge from the artifact the rest of the pipeline already consumed.
    ``index`` is the video decode index (the quality report's ``frame_num``
    column), not the candidate sequence number.

    Returns True when out_path exists afterwards (pre-existing or written).
    Returns False when the video cannot be opened, the index is out of
    range, or the seek cannot land on the frame — callers keep their
    existing honest per-view failure in that case.
    """
    if out_path.exists():
        return True
    try:
        video_path = ensure_cv2_readable(video_path)
    except InvalidVideoError:
        return False
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if index < 0 or (total > 0 and index >= total):
            return False
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
            return False
        got = None
        decoded = 0
        while got is None and decoded < _SEEK_VERIFY_LIMIT:
            ok, frame = cap.read()
            if not ok:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) - 1 >= index:
                got = frame
            decoded += 1
        if got is None:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(out_path), got))
    finally:
        cap.release()


def _select_temporal_buckets(kept: list[dict], keep_count: int, video_path: Path) -> list[dict]:
    """Pick *keep_count* frames spread across the whole trajectory.

    Coverage buckets by **trajectory arclength** when the adapter's
    flight_poses.csv sits next to the video (equal GROUND coverage — a
    slow frontal approach gets as many frames as a fast cross pass;
    time-uniform buckets under-sample the flight's final approach, which
    is exactly the frontal imagery a tower scene needs). Falls back to
    equal temporal buckets when no telemetry exists. Quality decides
    *within* a bucket; coverage decides *across* buckets. The input is
    ordered by construction (frames are scored in video order).
    """
    if len(kept) <= keep_count:
        return list(kept)
    ordered = sorted(kept, key=lambda c: c["frame_num"])
    positions = _candidate_positions(ordered, video_path)
    if positions is not None:
        # Cumulative 2-D arclength per candidate; stationary stretches do not
        # advance the arclength, so their frames compete inside one bucket.
        d = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        total = float(cum[-1])
    else:
        cum = np.arange(len(ordered), dtype=float)
        total = float(cum[-1]) if len(cum) else 0.0
    chosen: list[dict] = []
    targets = np.linspace(0.0, total, keep_count + 1)
    for b in range(keep_count):
        lo, hi = targets[b], targets[b + 1]
        idx = np.where((cum >= lo) & (cum <= hi))[0]
        if idx.size == 0:
            idx = np.array([min(int(round(lo / max(total, 1e-9) * (len(ordered) - 1))), len(ordered) - 1)])
        bucket = [ordered[i] for i in idx]
        chosen.append(max(bucket, key=lambda c: c["quality"].composite))
    return chosen


def _candidate_positions(ordered: list[dict], video_path: Path) -> np.ndarray | None:
    """ENU (x, y) per candidate from the dataset telemetry, or None.

    frame_id in flight_poses.csv is the 0-based video frame index (verified
    SRT contract), matching ``frame_num`` on candidates. For data-run
    missions the workspace copy (written at ingest, before extraction) sits
    next to the copied video that extract_frames reads.
    """
    csv_path = video_path.parent / "flight_poses.csv"
    if not csv_path.is_file():
        return None
    want = {c["frame_num"] for c in ordered}
    pos: dict[int, tuple[float, float]] = {}
    try:
        import csv as _csv

        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            for row in _csv.DictReader(fh):
                try:
                    fid = int(float(row["frame_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if fid in want:
                    pos[fid] = (float(row["x"]), float(row["y"]))
    except OSError:
        return None
    if len(pos) < max(2, len(ordered) // 2):
        return None
    return np.array([pos.get(c["frame_num"], (np.nan, np.nan)) for c in ordered])


def _compute_extract_indices(
    total_frames: int,
    video_fps: float,
    mode: str,
    target_fps: float | None,
    every_n: int | None,
    interval_sec: float | None,
) -> list[int]:
    """Compute which frame indices to extract based on the extraction mode."""
    if mode == "every_frame":
        return list(range(total_frames))

    if mode == "every_n":
        n = every_n or 10
        return list(range(0, total_frames, n))

    if mode == "target_fps":
        fps = target_fps or settings.processing.target_fps
        if fps >= video_fps:
            return list(range(total_frames))
        step = max(1, int(video_fps / fps))
        return list(range(0, total_frames, step))

    if mode == "interval":
        interval = interval_sec or 1.0
        step = max(1, int(interval * video_fps))
        return list(range(0, total_frames, step))

    # Fallback: every frame
    return list(range(total_frames))


def _generate_contact_sheet(
    candidates: list[dict],
    selected_dir: Path,
    rejected_dir: Path,
    output_dir: Path,
    thumb_size: tuple[int, int] = (160, 120),
    cols: int = 8,
) -> None:
    """Create a contact sheet JPEG showing selected vs rejected frames."""
    if not candidates:
        return

    thumbs = []
    for c in candidates:
        if c["kept"]:
            img_path = selected_dir / c["filename"]
            border_color = (0, 200, 0)  # green
        else:
            img_path = rejected_dir / c["filename"]
            border_color = (0, 0, 200)  # red

        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, thumb_size)
                # Draw border
                cv2.rectangle(img, (0, 0), (thumb_size[0] - 1, thumb_size[1] - 1), border_color, 2)
                # Overlay score
                score_text = f"{c['quality'].composite:.2f}"
                cv2.putText(img, score_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if c["quality"].rejection_reason:
                    cv2.putText(img, c["quality"].rejection_reason[:12], (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                thumbs.append(img)

    if not thumbs:
        return

    # Pad to fill grid
    while len(thumbs) % cols != 0:
        thumbs.append(np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8))

    rows = []
    for i in range(0, len(thumbs), cols):
        rows.append(np.hstack(thumbs[i:i + cols]))

    sheet = np.vstack(rows)
    cv2.imwrite(str(output_dir / "contact_sheet.jpg"), sheet)
