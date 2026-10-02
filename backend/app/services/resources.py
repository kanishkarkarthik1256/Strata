"""Resource detection — CPU / RAM / disk / GPU capability reporting.

Reports *measured* values from the host (via os/shutil/torch). Values that
cannot be measured are reported as ``None`` rather than invented, so callers
never mistake an absent probe for a real number.
"""

from __future__ import annotations

import os
import platform as _platform
import shutil
import subprocess
from functools import lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import Any
from pathlib import Path

from app.config.settings import settings


def ffmpeg_bin() -> str | None:
    """Resolve a usable ffmpeg binary: PATH first, then imageio-ffmpeg's
    bundled static build. Returns the path or None — callers decide.
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except (ImportError, RuntimeError):
        pass
    return None


def ffprobe_bin() -> str | None:
    """Resolve ffprobe: PATH, then next to the resolved ffmpeg binary.

    The imageio-ffmpeg wheel ships only the ffmpeg executable, so a PATH
    ffprobe may not exist even when ffmpeg does.
    """
    path = shutil.which("ffprobe")
    if path:
        return path
    exe = ffmpeg_bin()
    if exe:
        cand = os.path.join(os.path.dirname(exe), "ffprobe")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _module_version(module_name: str) -> str | None:
    """Actually installed version of a Python module, or None.

    Measured at runtime — never inferred from a plan or requirements file.
    """
    try:
        mod = import_module(module_name)
    except Exception:
        return None
    version = getattr(mod, "__version__", None)
    if isinstance(version, str) and version:
        return version
    return None


def _package_version(dist_name: str) -> str | None:
    try:
        return _dist_version(dist_name)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=None)
def _executable_version(executable: str) -> str | None:
    """Version line of an executable on PATH/known dirs, or None.

    Runs the binary's ``-version``/``--version`` probe once and caches it —
    capabilities are host facts that do not change during a process lifetime.
    Returns the raw first version-looking line; an unavailable executable is
    None, never a guessed string.
    """
    path = tool_available(executable)
    if path is None:
        return None
    for flag in ("-version", "--version", "version"):
        try:
            out = subprocess.run(
                [path, flag], capture_output=True, text=True, timeout=8
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        text_out = (out.stdout or out.stderr or "").strip()
        if not text_out:
            continue
        for line in text_out.splitlines():
            if "version" in line.lower():
                return line.strip()[:160]
        return text_out.splitlines()[0].strip()[:160]
    return None


def dependency_versions() -> dict[str, Any]:
    """Measured runtime dependency facts for the capabilities payload.

    Every value is detected from the actual environment (import, package
    metadata, or executable probe). Unavailable → None → the UI shows
    ``unknown``; nothing is inferred from architecture or plans.
    """
    torch_version = _module_version("torch")
    pycolmap_version = _module_version("pycolmap") or _package_version("pycolmap")

    @lru_cache(maxsize=1)
    def _binary_version_line(path: str | None) -> str | None:
        """Version line for a resolved binary path (PATH-independent)."""
        if path is None:
            return None
        for flag in ("-version", "--version", "version"):
            try:
                out = subprocess.run([path, flag], capture_output=True, text=True, timeout=8)
            except (OSError, subprocess.TimeoutExpired):
                continue
            text_out = (out.stdout or out.stderr or "").strip()
            for line in text_out.splitlines():
                if "version" in line.lower():
                    return line.strip()[:160]
            if text_out:
                return text_out.splitlines()[0].strip()[:160]
        return None
    return {
        "python": _platform.python_version(),
        "torch": torch_version,
        "pycolmap": pycolmap_version,
        "opencv": _module_version("cv2"),
        "ffmpeg": _binary_version_line(ffmpeg_bin()),
        "colmap": _executable_version("colmap"),
    }


def cpu_info() -> dict[str, Any]:
    count = os.cpu_count()
    load = None
    try:  # loadavg exists on POSIX; float average over 1 minute
        load = round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        load = None
    # ``cores``/``count`` stay the logical processor count (what the OS reports).
    # ``thread_budget`` is what the compute kernels actually use — the physical
    # count, which is measurably faster than oversubscribing logical cores.
    from app.services.cpu_budget import inference_threads, physical_cpu_count

    return {
        "cores": count,
        "load_1m": load,
        "count": count,
        "physical_cores": physical_cpu_count(),
        "thread_budget": inference_threads(),
    }


def memory_info() -> dict[str, Any]:
    """Peak process RSS in bytes where the OS exposes it.

    ``ru_maxrss`` units are platform-dependent: bytes on macOS, KiB on Linux —
    normalise so the reported value is always bytes.
    """
    try:
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss = rss * 1024  # Linux reports KiB; macOS already reports bytes
    except (ImportError, AttributeError):
        rss = None
    return {"process_rss_bytes": rss}


def disk_info() -> dict[str, Any]:
    usage = shutil.disk_usage(settings.storage.base_path)
    return {
        "path": str(settings.storage.base_path),
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }


def gpu_info() -> dict[str, Any]:
    """Report CUDA / MPS GPU details when PyTorch is installed."""
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        base = {
            "cuda_available": cuda_ok,
            "mps_available": mps_ok,
            "torch_version": _module_version("torch"),
        }
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            free_mem, _total = torch.cuda.mem_get_info(0)
            return {
                **base,
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(props.total_memory),
                "free_memory_bytes": int(free_mem),
                "cuda_version": torch.version.cuda,
                "device_index": 0,
            }
        elif mps_ok:
            return {
                **base,
                "available": True,
                "device": "Apple Silicon MPS (Metal Acceleration)",
                "reason": "mps_available",
            }
        return {
            **base,
            "available": False,
            "device": "CPU",
            "reason": "cpu_fallback",
            "detail": "No CUDA/MPS device — reconstruction runs on CPU (slower, fully supported)",
        }
    except ImportError:
        return {
            "available": False,
            "cuda_available": False,
            "mps_available": False,
            "torch_version": None,
            "reason": "torch_not_installed",
        }


def _smoke_workspace() -> Path | None:
    """Writable smoke directory under the base path, cleaned of prior state."""
    base = Path(settings.storage.base_path)
    ws = base / "smoke"
    try:
        ws.mkdir(parents=True, exist_ok=True)
        return ws if os.access(ws, os.W_OK) else None
    except OSError:
        return None


def _first_data_video() -> Path | None:
    """A real video to smoke-test with — prefers data/airport1/video.mp4.

    Falls back to any data-folder MP4 so the smoke still exercises a real
    decode if the preferred dataset is absent. Never invents a file.
    """
    try:
        from app.services.data_video_service import data_dir
    except Exception:
        return None
    try:
        base = Path(data_dir())
    except TypeError:
        return None
    if not base.is_dir():
        return None
    preferred = base / "airport1" / "video.mp4"
    if preferred.is_file() and preferred.stat().st_size > 0:
        return preferred
    for p in sorted(base.glob("*.mp4")):
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None



KNOWN_SEARCH_DIRS = [
    "/usr/local/bin",
    "/opt/homebrew/bin",
]


def tool_available(name: str) -> str | None:
    """Return the resolved path of *name* on PATH or known search paths, or None."""
    path = shutil.which(name)
    if path:
        return path
    for sdir in KNOWN_SEARCH_DIRS:
        cand = os.path.join(sdir, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


@lru_cache(maxsize=1)
def depth_smoke_status() -> dict[str, Any] | None:
    """Real Depth Anything V2 CPU inference on a real data-video frame.

    Runs the actual vendored stack end-to-end (checkpoint load + forward
    pass) and asserts a sane output. Cached for the process lifetime.
    Returns a result dict, or None if any step fails — never a guess.
    """
    ws = _smoke_workspace()
    if ws is None:
        return None
    try:
        import time

        import cv2
        import numpy as np

        from app.services.depth_anything_v2 import detect_device, find_checkpoint, load_model

        ckpt = find_checkpoint()
        if ckpt is None or not ckpt.is_file():
            return None
        video = _first_data_video()
        if video is None:
            return None
        cap = cv2.VideoCapture(str(video))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            return None

        t0 = time.time()
        model, device, path = load_model()
        load_s = time.time() - t0
        t0 = time.time()
        depth = model.infer_image(frame)
        infer_s = time.time() - t0
        if not isinstance(depth, np.ndarray) or depth.shape != frame.shape[:2]:
            return None
        if not np.isfinite(depth).all() or float(depth.max()) <= float(depth.min()):
            return None
        return {
            "ok": True,
            "checkpoint": path,
            "device": device,
            "video": str(video),
            "load_seconds": round(load_s, 2),
            "infer_seconds": round(infer_s, 2),
            "output": "relative inverse depth (not metric)",
        }
    except Exception:
        return None


@lru_cache(maxsize=1)
def colmap_smoke_status() -> dict[str, Any] | None:
    """Real pycolmap SfM on 6 real frames from the data folder.

    Feature extraction → matching → incremental mapping through the same
    API the pipeline uses, run in a child process so a native abort cannot
    take the backend down. PNGs are used because the pycolmap macOS x86_64
    wheel cannot decode JPEG. Cached for the process lifetime.
    """
    ws = _smoke_workspace()
    if ws is None:
        return None
    video = _first_data_video()
    if video is None:
        return None
    code = "\n".join(
        [
            "import sys, numpy as np, cv2, pycolmap, shutil",
            "from pathlib import Path",
            "tmp = Path(sys.argv[1]); img = tmp / 'images'; img.mkdir()",
            "cap = cv2.VideoCapture(sys.argv[2])",
            "for i, idx in enumerate([0, 2400, 4800, 7200, 9600, 12000]):",
            "    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)",
            "    ok, f = cap.read()",
            "    if not ok: print('FAIL decode'); sys.exit(1)",
            "    cv2.imwrite(str(img / f'img{i:02d}.png'), f)",
            "cap.release()",
            "db = str(tmp / 'db.db')",
            "pycolmap.extract_features(database_path=db, image_path=str(img))",
            "pycolmap.match_exhaustive(database_path=db)",
            "recs = pycolmap.incremental_mapping(database_path=db, image_path=str(img), output_path=str(tmp / 'out'))",
            "if not recs: print('FAIL no_model'); sys.exit(1)",
            "rec = list(recs.values())[0]",
            "errs = [rec.points3D[pid].error for pid in rec.points3D] if len(rec.points3D) else []",
            "print(f'OK {len(rec.images)} {len(rec.points3D)} '",
            "      f'{float(np.mean(errs)) if errs else -1:.3f}')",
            "shutil.rmtree(tmp, ignore_errors=True)",
        ]
    )
    try:
        import time

        import sys as _sys

        tmp = ws / "colmap_smoke"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        res = subprocess.run(
            [_sys.executable, "-c", code, str(tmp), str(video)],
            capture_output=True,
            text=True,
            timeout=240,
        )
        elapsed = round(time.time() - t0, 1)
        out = (res.stdout or "").strip().splitlines()
        last = out[-1] if out else ""
        if res.returncode == 0 and last.startswith("OK "):
            _, n_imgs, n_pts, err = last.split()
            return {
                "ok": True,
                "images_registered": int(n_imgs),
                "points3d": int(n_pts),
                "mean_reproj_px": float(err),
                "seconds": elapsed,
            }
        return None
    except Exception:
        return None


@lru_cache(maxsize=1)
def video_decode_smoke_status() -> dict[str, Any] | None:
    """Real decode of first/mid/last frames of a data-folder video."""
    try:
        import cv2

        video = _first_data_video()
        if video is None:
            return None
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            frames_ok = 0
            for target in (0, total // 2, max(total - 1, 0)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    frames_ok += 1
        finally:
            cap.release()
        if frames_ok < 3 or w <= 0 or h <= 0:
            return None
        return {
            "ok": True,
            "video": str(video),
            "frames_total": total,
            "fps": round(fps, 3),
            "resolution": f"{w}x{h}",
            "probe_frames": frames_ok,
        }
    except Exception:
        return None


def tool_available(name: str) -> str | None:
    """Return the resolved path of *name* on PATH or known search paths, or None."""
    path = shutil.which(name)
    if path:
        return path
    for sdir in KNOWN_SEARCH_DIRS:
        cand = os.path.join(sdir, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def depth_model_status(workspace: Any = None) -> dict[str, Any]:
    """Report explicit lifecycle status for DEPTH_MODEL.

    ``smoke_tested`` is only True after a real inference on a real frame in
    this process (memory of a past run is not proof this runtime can run).
    """
    status = {
        "installed": False,
        "detected": False,
        "smoke_tested": False,
        "used": False,
        "validated": False,
        "checkpoint": None,
        "status_level": "UNAVAILABLE",
    }
    try:
        import torch  # noqa: F401

        from app.services.depth_anything_v2 import find_checkpoint

        status["installed"] = True
        ckpt = find_checkpoint()
        if ckpt is not None and ckpt.is_file():
            status["detected"] = True
            status["checkpoint"] = str(ckpt)
            if depth_smoke_status() is not None:
                status["smoke_tested"] = True
                status["status_level"] = "READY"
    except Exception:
        pass

    return status


def capabilities() -> dict[str, Any]:
    """Full host capability report — the single source for /health etc."""
    gpu = gpu_info()
    return {
        "compute": {
            "cpu": cpu_info(),
            "gpu": gpu,
            "device": "cuda" if gpu.get("available") else "cpu",
        },
        "memory": memory_info(),
        "disk": disk_info(),
        "tools": {
            "colmap": tool_available("colmap") is not None,
            "ffprobe": ffprobe_bin() is not None,
            "ffmpeg": ffmpeg_bin() is not None,
        },
        "depth_model": depth_model_status(),
        "smoke_tests": {
            "depth": depth_smoke_status(),
            "colmap": colmap_smoke_status(),
            "video_decode": video_decode_smoke_status(),
        },
        "dependencies": dependency_versions(),
    }


def stage_feasibility(stage_gpu_required: bool = False) -> dict[str, Any]:
    """Whether a stage can run here — fails *clearly*, never silently fakes.

    Stages that need the GPU get an explicit ``feasible: false`` reason when
    CUDA is absent; the caller must use a legitimate CPU fallback or fail.
    """
    gpu = gpu_info()
    if not stage_gpu_required:
        return {"feasible": True, "device": "cpu", "gpu": gpu}
    return {
        "feasible": bool(gpu.get("available")),
        "device": "cuda" if gpu.get("available") else None,
        "reason": None if gpu.get("available") else "gpu_required_but_unavailable",
        "gpu": gpu,
    }
