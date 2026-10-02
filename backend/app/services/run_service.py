"""Run discovery and artifact serving (Phase 11.1).

Phase 9.5 runs are filesystem artifacts: ``outputs/<RUN_ID>/`` containing a
``manifest.json`` plus stage outputs (dense/sparse PLY, poses.json, reports).
This service is the single read path for those artifacts:

* :func:`list_runs` / :func:`get_run` — discovery + metadata enrichment
  (point counts come from the actual PLY headers / reconstruction reports).
* :func:`resolve_artifact` — validated, traversal-safe artifact paths.

Artifact serving is deliberately constrained: run ids match a strict pattern
and artifact paths are resolved with containment checks, so the endpoint can
never escape the runs root.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from app.config.settings import settings
from app.logging_config import get_logger

log = get_logger("drone_recon.services.run_service")

#: Strict run-id pattern (no slashes, no dots-only segments, no traversal).
_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

#: Extensions this service is allowed to serve. Only what the UI needs today.
_ARTIFACT_EXTENSIONS = frozenset({
    ".ply", ".obj", ".glb", ".gltf", ".json", ".npy", ".npz",
    ".png", ".jpg", ".jpeg", ".txt", ".md", ".html",
})

_MAX_ARTIFACT_FILES = 300
_MAX_ARTIFACT_NAME_LEN = 512


class RunNotFoundError(Exception):
    """Raised when a run id does not resolve to a run directory."""


class InvalidArtifactError(Exception):
    """Raised when an artifact name fails validation."""


# Anchor for relative settings paths: this file lives at
# <repo>/backend/app/services/run_service.py, so the backend root (the process
# cwd the pipeline/validator convention assumes) is three parents up. Anchoring
# to __file__ instead of cwd keeps run discovery stable no matter where the
# server is launched from (e.g. uvicorn started at the repo root).
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_runs_dir() -> Path:
    """Resolve the configured runs dir relative to the backend root, not cwd."""
    path = Path(settings.storage.runs_dir)
    if not path.is_absolute():
        anchored = _BACKEND_ROOT / path
        if anchored.is_dir() or not (Path.cwd() / path).is_dir():
            return anchored
        # Legacy layout (cwd-relative) still wins while it exists.
        return Path.cwd() / path
    return path


def runs_root() -> Path:
    """The runs directory (absolute; created on demand)."""
    path = _resolve_runs_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(run_id: str) -> Path:
    """Resolve *run_id* to a run directory, or raise :class:`RunNotFoundError`."""
    if not run_id or not _RUN_ID_RE.match(run_id) or ".." in run_id:
        raise InvalidArtifactError("invalid run id")

    # Check outputs/
    root_outputs = runs_root().resolve()
    cand1 = (root_outputs / run_id).resolve()
    if cand1.is_relative_to(root_outputs) and cand1.is_dir():
        return cand1

    # Check output/ (later-phase runs; anchored like runs_root above).
    output_root = (_BACKEND_ROOT.parent / "output").resolve()
    if output_root.is_dir():
        cand_out = (output_root / run_id).resolve()
        if cand_out.is_relative_to(output_root) and cand_out.is_dir():
            return cand_out

    # Check data/storage/
    root_storage = settings.storage.base_dir.resolve()
    cand2 = (root_storage / run_id).resolve()
    if cand2.is_relative_to(root_storage) and cand2.is_dir():
        return cand2

    raise RunNotFoundError(run_id)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _ply_vertex_count(path: Path) -> int | None:
    """Parse the vertex count from a PLY header without loading the file."""
    try:
        header = path.open("rb").read(4096).split(b"end_header", 1)[0]
    except OSError:
        return None
    for line in header.splitlines():
        if line.startswith(b"element vertex "):
            try:
                return int(line.split()[-1])
            except ValueError:
                return None
    return None


def _ply_face_count(path: Path) -> int | None:
    """Parse the face count from a PLY header without loading the file."""
    try:
        header = path.open("rb").read(4096).split(b"end_header", 1)[0]
    except OSError:
        return None
    for line in header.splitlines():
        if line.startswith(b"element face "):
            try:
                return int(line.split()[-1])
            except ValueError:
                return None
    return None


def _report_points(run: Path, name: str, key: str) -> int | None:
    report = _read_json(run / name)
    if not report:
        return None
    value = report.get("reconstruction", {}).get(key)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _counts(run: Path) -> dict[str, Any]:
    """Point / camera / GPS / mesh counts derived from the actual artifacts."""
    out: dict[str, Any] = {}
    dense_ply = run / "dense" / "dense_model.ply"
    if dense_ply.exists():
        out["dense_points"] = _ply_vertex_count(dense_ply)
    sparse_ply = run / "sparse" / "sparse_model.ply"
    if not sparse_ply.exists():
        # Data-video runs write the sparse model at the workspace root.
        sparse_ply = run / "sparse_model.ply"
    if sparse_ply.exists():
        out["sparse_points"] = _ply_vertex_count(sparse_ply)
    out["sparse_points"] = out.get("sparse_points") or _report_points(
        run, "sparse/reconstruction_report.json", "num_points")

    mesh_ply = run / "mesh" / "mesh.ply"
    if not mesh_ply.exists():
        mesh_ply = run / "mesh" / "base_mesh.ply"
    if not mesh_ply.exists():
        mesh_ply = run / "diagnostics" / "mesh.ply"
    if mesh_ply.exists():
        out["mesh_vertices"] = _ply_vertex_count(mesh_ply)
        out["mesh_faces"] = _ply_face_count(mesh_ply)

    report = _read_json(run / "reconstruction_report.json") or _read_json(run / "sparse" / "reconstruction_report.json")
    if report:
        recon = report.get("reconstruction", {})
        if recon.get("num_cameras") is not None:
            out["cameras"] = recon["num_cameras"]
        # Data runs record ``mean_reproj_error``; Phase-9.5 runs record
        # ``mean_reprojection_error_px`` — accept both spellings.
        reproj = recon.get("mean_reprojection_error_px", recon.get("mean_reproj_error"))
        if reproj is not None:
            out["mean_reprojection_error_px"] = reproj
        ba = report.get("stages", {}).get("bundle_adjustment", {})
        if ba:
            out["bundle_adjustment"] = ba

    # Sparse↔dense consistency (dense stage / backfilled report): the
    # geometric-agreement numbers a reviewer needs next to the model.
    dense_report = _read_json(run / "dense_report.json")
    if dense_report:
        cons = dense_report.get("stages", {}).get("sparse_dense_consistency")
        if isinstance(cons, dict) and cons.get("median_m") is not None:
            out["sparse_dense"] = {
                "correspondences": cons.get("correspondences"),
                "median_m": cons.get("median_m"),
                "p95_m": cons.get("p95_m"),
                "rmse_m": cons.get("rmse_m"),
                "within_3m_pct": cons.get("within_3m_pct"),
            }
            screened = cons.get("screened_median_m")
            if screened is not None:
                out["sparse_dense"]["screened"] = {
                    "correspondences": cons.get("screened_correspondences"),
                    "dropped": cons.get("screened_dropped"),
                    "median_m": screened,
                    "p95_m": cons.get("screened_p95_m"),
                    "rmse_m": cons.get("screened_rmse_m"),
                    "note": "points beyond their depth-uncertainty budget excluded",
                }
        mq = dense_report.get("stages", {}).get("mesh_quality") or _read_json(run / "mesh_quality_report.json")
        if isinstance(mq, dict) and mq.get("faces"):
            out["mesh_quality"] = {
                "giant_faces": mq.get("giant_faces_gt_10voxels"),
                "components": mq.get("components"),
                "largest_component_pct": mq.get("largest_component_pct"),
            }
    poses = _read_json(run / "poses.json") or _read_json(run / "sparse" / "poses.json")
    if poses and isinstance(poses.get("frames"), list):
        frames = poses["frames"]
        out["cameras"] = out.get("cameras") or len(frames)
        out["gps_points"] = sum(1 for f in frames if isinstance(f, dict) and f.get("gps"))
        if not out["gps_points"]:
            # Not every run attaches per-frame GPS to poses.json; a run
            # georeferenced against external telemetry keeps its fixes in
            # georef/gps_report.json, matched to frames by time. Serving 0
            # there reads as "no GPS" for a run whose georef matched every
            # frame (measured on flight_to_tower_7511dc: 60/60 matched while
            # the run card said 0). Prefer the poses record; fall back to the
            # georef sync.
            report = _read_json(run / "georef" / "gps_report.json") or {}
            sync = report.get("sync") if isinstance(report, dict) else None
            matched = (sync or {}).get("matched_frames")
            if isinstance(matched, int) and matched > 0:
                out["gps_points"] = matched
    return out


# Artifacts the UI consumes directly. These must never be crowded out of
# the capped inventory by bulk per-frame files: airport1 carries 1,700+
# matching files and the depth/NPY mass pushed mesh_viewer.glb past the cap,
# silently downgrading the viewer to the untextured PLY fallback.
_PRIMARY_ARTIFACTS = (
    "mesh/mesh_viewer.glb",
    "combined_model.ply",
    "dense/dense_model.ply",
    "sparse_model.ply",
    "sparse/sparse_model.ply",
    "mesh/mesh.ply",
    "mesh/mesh_full.ply",
    "mesh/base_mesh.ply",
)


def _artifact_inventory(run: Path) -> list[dict[str, Any]]:
    """Relative artifact listing (allowed extensions only, capped).

    Primary UI models are listed first, then other model files (.ply), then
    bulk per-frame artifacts — a run with hundreds of depth PNGs/NPYs can
    never crowd ``mesh_viewer.glb`` or the root models out of the capped
    listing.

    Traversal is ``os.scandir`` rather than ``Path.rglob``: every entry it
    yields already carries its type (so no per-file ``is_file`` syscall) and
    only the files that pass the extension filter are stat-ed. Measured on the
    76-run store: rglob cost 11.8 s of a 12.5 s ``list_runs`` across 109k
    stats; this walk produces the identical listing for ~0.2 s. The final sort
    key is (priority, path) and therefore traversal-order independent.
    """
    items: list[dict[str, Any]] = []
    stack: list[tuple[str, str]] = [("", str(run))]
    while stack:
        rel_dir, abs_dir = stack.pop()
        try:
            with os.scandir(abs_dir) as entries:
                batch = list(entries)
        except OSError:
            continue
        for entry in batch:
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append((rel, entry.path))
                    continue
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    continue
                if Path(entry.name).suffix.lower() not in _ARTIFACT_EXTENSIONS:
                    continue
                items.append({"path": rel, "size_bytes": entry.stat().st_size})
            except OSError:  # raced removal / unreadable entry
                continue
    items.sort(key=lambda it: (
        0 if it["path"] in _PRIMARY_ARTIFACTS else 1 if it["path"].lower().endswith(".ply") else 2,
        it["path"],
    ))
    return items[:_MAX_ARTIFACT_FILES]


# Legacy root holding the precomputed demo runs that shipped with early
# phases. Runs discovered here are tagged ``is_demo`` — they are not user
# missions and must never be presented as normal results.
_LEGACY_DEMO_ROOT = (_BACKEND_ROOT / "outputs").resolve()


def _run_summary(
    run_id: str, run: Path, manifest: dict[str, Any], *, is_demo: bool = False
) -> dict[str, Any]:
    stages = {name: (st.get("status") if isinstance(st, dict) else None)
              for name, st in manifest.get("stages", {}).items()}
    return {
        "run_id": run_id,
        "dataset": manifest.get("dataset"),
        "mission": manifest.get("mission"),
        "status": manifest.get("status"),
        "pipeline_version": manifest.get("pipeline_version"),
        "started_at": manifest.get("started_at"),
        "completed_at": manifest.get("completed_at"),
        "stages": stages,
        "metrics": manifest.get("metrics", {}),
        "dependencies": manifest.get("dependencies", {}),
        "limitations": manifest.get("limitations", []),
        "artifacts": _artifact_inventory(run),
        "is_demo": is_demo,
        **_counts(run),
    }


_RUNS_CACHE: tuple[float, list[dict[str, Any]]] | None = None
_CACHE_TTL_SEC = 30.0


def invalidate_runs_cache() -> None:
    global _RUNS_CACHE
    _RUNS_CACHE = None


def list_runs() -> list[dict[str, Any]]:
    """Discover all runs (dirs containing a manifest.json or pipeline_report.json), newest first."""
    global _RUNS_CACHE
    now = time.monotonic()
    if _RUNS_CACHE is not None:
        cached_time, cached_data = _RUNS_CACHE
        if now - cached_time < _CACHE_TTL_SEC:
            return cached_data

    output_dir = _BACKEND_ROOT.parent / "output"
    roots = [runs_root(), settings.storage.base_dir]
    if output_dir.is_dir() and not any("pytest" in str(r) for r in roots):
        roots.insert(0, output_dir)
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()

    try:
        demo_root = Path(_LEGACY_DEMO_ROOT).resolve()
    except (OSError, ValueError):
        demo_root = None

    for root in roots:
        if not root.is_dir():
            continue
        is_demo_root = demo_root is not None and root.resolve() == demo_root
        for entry in sorted(root.iterdir(), reverse=True):
            if not entry.is_dir() or entry.name in seen:
                continue
            manifest = _read_json(entry / "manifest.json")
            if manifest is None:
                report = _read_json(entry / "pipeline_report.json")
                if report:
                    manifest = {
                        "run_id": entry.name,
                        "dataset": entry.name,
                        "mission": f"Mission {entry.name}",
                        "status": report.get("status", "completed"),
                        "stages": report.get("stages", {}),
                        "metrics": report.get("profile", {}),
                    }
            if not manifest:
                continue
            try:
                seen.add(entry.name)
                runs.append(_run_summary(entry.name, entry, manifest, is_demo=is_demo_root))
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("run_summary_failed", run_id=entry.name, error=str(exc))

    _RUNS_CACHE = (now, runs)
    return runs


def get_run(run_id: str) -> dict[str, Any]:
    """Full detail for one run (summary + raw manifest)."""
    run = run_dir(run_id)
    manifest = _read_json(run / "manifest.json")
    if manifest is None:
        report = _read_json(run / "pipeline_report.json")
        if report:
            manifest = {
                "run_id": run_id,
                "dataset": run_id,
                "mission": f"Mission {run_id}",
                "status": report.get("status", "completed"),
                "stages": report.get("stages", {}),
                "metrics": report.get("profile", {}),
            }
        else:
            raise RunNotFoundError(run_id)
    summary = _run_summary(run_id, run, manifest)
    summary["manifest"] = manifest
    return summary


def resolve_artifact(run_id: str, name: str) -> Path:
    """Resolve an artifact path safely, or raise :class:`InvalidArtifactError`.

    *name* may contain forward slashes for subdirectories (e.g.
    ``dense/dense_model.ply``). Rejects absolute paths, traversal segments,
    unknown extensions and any path that escapes the run directory.
    """
    if not name or len(name) > _MAX_ARTIFACT_NAME_LEN:
        raise InvalidArtifactError("empty or oversized artifact name")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise InvalidArtifactError("path traversal is not allowed")
    if path.suffix.lower() not in _ARTIFACT_EXTENSIONS:
        raise InvalidArtifactError(f"extension not allowed: {path.suffix or 'none'}")
    run = run_dir(run_id)
    candidate = (run / path).resolve()
    if not candidate.is_relative_to(run):
        raise InvalidArtifactError("artifact escapes the run directory")
    if not candidate.is_file():
        raise InvalidArtifactError("artifact does not exist")
    return candidate


def get_viewer_alignment(run_id: str) -> dict[str, Any]:
    """Return the run's viewer-alignment transform (computing it once).

    The cached ``viewer_alignment.json`` is reused unless any upstream
    geometry/pose artifact is newer than the cache.
    """
    from app.services import viewer_alignment

    run = run_dir(run_id)
    cached = run / "viewer_alignment.json"
    if cached.is_file():
        cache_mtime = cached.stat().st_mtime
        stale = any(
            (run / rel).is_file() and (run / rel).stat().st_mtime > cache_mtime
            for rel in viewer_alignment._GEOMETRY_CANDIDATES
        ) or (
            (run / "poses.json").is_file()
            and (run / "poses.json").stat().st_mtime > cache_mtime
        )
        if not stale:
            try:
                return json.loads(cached.read_text())
            except Exception:
                pass
    return viewer_alignment.compute_alignment(run)


