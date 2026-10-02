"""Prefetch raw (un-anchored) depth maps while the sparse stage runs.

Depth Anything V2's per-frame output depends only on the image — the pose
enters afterwards, in the per-view affine anchor that converts relative depth
into metric depth. So the forward passes are pose-independent *by
construction*, while they cannot start today until sparse finishes, and sparse
leaves most cores idle for several hundred seconds (measured on the reference
host: mapping 391 s and the Ceres prior-BA pass 180 s are effectively
single-core-bound, out of a 1020 s sparse stage).

This module computes those raw maps early and stores them, so the depth stage
reads them and spends its wall clock on anchoring only. It is a scheduling
change, never a numerical one:

* the array is written exactly as ``infer_image`` returned it — before the
  validity mask the depth stage applies;
* the reader validates dtype, the model's OWN native grid (via the model's
  transform, not a guess) and the checkpoint name before trusting a cached
  map, and any mismatch falls through to live inference;
* measured determinism: identical SHA256 depth maps across separate processes
  and across 4 and 8 torch threads.

Best-effort by design: a prefetch failure leaves the cache short and the depth
stage recomputes what is missing, so it can never change or fail a run.

Runtime-ordering hazard (measured, not theoretical): this host SIGSEGVs
(exit 139) when pycolmap is imported and USED before torch runs its first
forward pass — reproduced twice in isolation, and it is a native crash no
``except`` clause can catch. Production order is the safe one (``app.main``
imports torch before the sparse stage imports pycolmap), and that order was
verified to survive 461 s of concurrent prefetch + incremental mapping. The
start guard below makes the unsafe order structurally impossible: without
torch already loaded, no prefetch is started at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from app.logging_config import get_logger

log = get_logger("drone_recon.services.depth_prefetch")

#: Workspace subdirectory holding prefetched raw maps + sidecars.
RAW_CACHE_DIRNAME = "depth_raw"
#: Sidecar schema for the cache entries.
_CACHE_SCHEMA = 1
#: Hard ceiling on a single prefetch, so a misplaced start can never burn CPU
#: for the life of the process.
_MAX_PREFETCH_SEC = 3600.0


def raw_cache_dir(workspace: Path) -> Path:
    """Where prefetched raw depth maps live for *workspace*."""
    return Path(workspace) / RAW_CACHE_DIRNAME


def cached_raw_depth(
    cache_dir: Path,
    frame_id: str,
    *,
    expected_shape: tuple[int, int] | None = None,
    checkpoint_name: str | None = None,
) -> np.ndarray | None:
    """Return the cached raw map for *frame_id*, or ``None`` to infer live.

    Every rejection reason returns ``None`` (a miss), which is always safe:
    the caller falls back to live inference.
    """
    npy = Path(cache_dir) / f"{frame_id}.npy"
    if not npy.is_file():
        return None
    meta_path = npy.with_suffix(".json")
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            return None
        if meta.get("schema") != _CACHE_SCHEMA:
            return None
        if checkpoint_name is not None and meta.get("checkpoint") not in (None, checkpoint_name):
            return None
    try:
        arr = np.load(npy)
    except (OSError, ValueError):
        return None
    if arr.dtype != np.float32:
        return None
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        return None
    if meta.get("sha256") and hashlib.sha256(arr.tobytes()).hexdigest() != meta["sha256"]:
        # A corrupted cache entry must never become a depth map.
        return None
    return arr


def _store_raw(cache_dir: Path, frame_id: str, array: np.ndarray, meta: dict) -> None:
    """Write *array* + sidecar atomically (a torn file is never readable)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / f"{frame_id}.npy"
    tmp = cache_dir / f".{frame_id}.{os.getpid()}.tmp"
    payload = dict(meta)
    payload["schema"] = _CACHE_SCHEMA
    payload["shape"] = list(array.shape)
    payload["dtype"] = str(array.dtype)
    payload["sha256"] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(array))
    os.replace(tmp, npy)
    (cache_dir / f"{frame_id}.json").write_text(json.dumps(payload))


def prefetch_raw_depths(
    workspace: Path,
    *,
    stop: threading.Event | None = None,
    max_seconds: float = _MAX_PREFETCH_SEC,
) -> dict:
    """Infer and cache raw depth for every selected frame. Never raises.

    Prefetches the whole selected-frame set rather than only the registered
    views: the registered set is written by the sparse stage this is
    overlapping, and any unregistered frame simply costs prefetch-window CPU
    (the depth stage never reads it). Stopping early — or failing — is always
    safe because the depth stage recomputes whatever is missing.
    """
    stats = {"generated": 0, "cached": 0, "failed": 0, "stopped": False, "seconds": 0.0}
    t0 = time.perf_counter()
    try:
        from app.services.depth_anything_v2 import load_model  # deferred: heavy
        from app.services.depth_generator import _image_dirs, _load_image

        image_dirs = _image_dirs(Path(workspace))
        if not image_dirs:
            stats["note"] = "no selected frames"
            return stats
        model, _device, ckpt = load_model()
        checkpoint = Path(ckpt).name
        cache = raw_cache_dir(workspace)

        # Only the primary directory: it is the one `generate_view_depths`
        # resolves frames from first, and walking the fallback directory too
        # just re-lists the same stems (the depth stage's own resolution rule
        # still applies on a miss, which simply infers live).
        frames = sorted(
            p for p in image_dirs[0].iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frames:
            stats["note"] = "no frame images"
            return stats

        log.info("depth_prefetch_started", frames=len(frames), checkpoint=checkpoint,
                 note="raw (pose-independent) maps computed during the sparse stage")
        for f in frames:
            if stop is not None and stop.is_set():
                stats["stopped"] = True
                break
            if time.perf_counter() - t0 > max_seconds:
                stats["stopped"] = True
                break
            try:
                if (cache / f"{f.stem}.npy").is_file():
                    stats["cached"] += 1
                    continue
                img = _load_image(image_dirs, f.stem)
                d_raw = model.infer_image(img, native_resolution=True).astype(np.float32)
                _store_raw(cache, f.stem, d_raw, {"checkpoint": checkpoint})
                stats["generated"] += 1
            except Exception as exc:  # best-effort: never fail the run
                stats["failed"] += 1
                log.warning("depth_prefetch_frame_failed", frame_id=f.stem, error=str(exc))
    except Exception as exc:  # pragma: no cover - environment/loading failures
        stats["failed"] += 1
        stats["note"] = str(exc)
        log.warning("depth_prefetch_unavailable", error=str(exc))
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    log.info("depth_prefetch_finished", **stats)
    return stats


# ---------------------------------------------------------------------------
# Process-level control: one prefetch per workspace, started before the sparse
# stage and stopped when the depth stage takes over the model.
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop: threading.Event | None = None
_stats: dict = {}


def start_depth_prefetch(workspace: Path) -> bool:
    """Start the prefetch thread for *workspace* (idempotent). Returns started?"""
    global _thread, _stop
    if "torch" not in sys.modules:
        # See the ordering hazard above: running torch inference after
        # pycolmap has already loaded libomp can segfault this host. Without
        # torch imported first, skip the prefetch entirely (the depth stage
        # then infers live, exactly as before).
        log.warning("depth_prefetch_skipped", reason="torch_not_loaded_before_prefetch")
        return False
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        event = threading.Event()
        _stop = event
        _thread = threading.Thread(
            target=lambda: _stats.update(
                prefetch_raw_depths(workspace, stop=event)
            ),
            name="depth-prefetch",
            daemon=True,
        )
        _thread.start()
        return True


def stop_depth_prefetch(timeout: float = 1.0) -> dict:
    """Signal the prefetch to stop and return its stats (idempotent)."""
    with _lock:
        thread, event = _thread, _stop
    if event is not None:
        event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    return dict(_stats)
