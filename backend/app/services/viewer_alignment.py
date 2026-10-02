"""Viewer alignment — a PRESENTATION transform, never a reconstruction change.

Estimates the dominant ground plane of a finished run from its own artifacts
(dense/mesh/sparse geometry + the camera-up prior from poses.json) and derives
a rigid transform

    X_view = R_view @ X_world + t_view        (R orthonormal, det=+1, scale=1)

such that in the viewer frame:

* the ground normal is +Y  (three.js up; ground sits on the y=0 grid),
* the flight-corridor direction is +X where reliably estimable,
* the ground-plane centroid maps to the origin (t_view is framing only).

The authoritative mesh/point/camera artifacts are never modified; the frontend
applies this same transform to every 3D layer so nothing detaches. Results are
cached in ``viewer_alignment.json`` inside the run directory.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.logging_config import get_logger

log = get_logger("drone_recon.services.viewer_alignment")

#: Geometry sources tried in order (same world frame; first existing wins).
_GEOMETRY_CANDIDATES = (
    "dense/dense_model.ply",
    "mesh/base_mesh.ply",
    "mesh/mesh.ply",
    "combined_model.ply",
    "sparse/sparse_model.ply",
    "sparse_model.ply",
)

_ALIGN_JSON = "viewer_alignment.json"
_MAX_POINTS = 150_000  # after voxel downsampling — plenty for a plane fit
_BAND_FRACTION = 0.35  # lowest band along the up prior = candidate ground
_MIN_BAND_INLIER_FRAC = 0.25  # below this the point evidence is not trusted
_MIN_CORRIDOR_RATIO = 1.8  # PCA variance ratio for corridor-axis confidence
#: how well the fleet's optical axis must look AT the reconstructed shell before
#: the shell normal (not the image-up heuristic) is trusted as the vertical
_MIN_SHELL_AXIS_ALIGNMENT = 0.8
#: payload/cache schema — bumped so stale (wrong-up) alignments recompute
_ALIGNMENT_VERSION = 2


# ---------------------------------------------------------------------------
# loading helpers
# ---------------------------------------------------------------------------


def _load_points(run_dir: Path) -> np.ndarray | None:
    """Load the first available geometry artifact as an (N,3) point array."""
    import open3d as o3d

    for rel in _GEOMETRY_CANDIDATES:
        path = run_dir / rel
        if not path.is_file():
            continue
        try:
            if path.suffix == ".ply":
                pc = o3d.io.read_point_cloud(str(path))
                pts = np.asarray(pc.points)
                if len(pts) < 500:  # maybe it's a mesh — read as mesh instead
                    mesh = o3d.io.read_triangle_mesh(str(path))
                    pts = np.asarray(mesh.vertices)
            else:
                continue
            if len(pts) >= 500:
                return np.asarray(pts, dtype=np.float64)
        except Exception as exc:  # unreadable artifact — try the next source
            log.warning("viewer_alignment_geometry_read_failed", source=rel, error=str(exc))
    return None


def _downsample(pts: np.ndarray) -> np.ndarray:
    if len(pts) <= _MAX_POINTS:
        return pts
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    voxel = max(extent / 256.0, 1e-6)
    step = max(1, int(len(pts) * voxel**3 / max(extent**3, 1e-12) / _MAX_POINTS) + 1)
    # deterministic stride decimation (open3d voxel grid can merge differently
    # across versions; a fixed stride keeps the result reproducible)
    return pts[::step][:_MAX_POINTS] if step > 1 else pts[::_MAX_POINTS and 1]


def _camera_frames(run_dir: Path) -> np.ndarray:
    """Rotation matrices from poses.json as an (n,3,3) array (world = R @ cam)."""
    path = run_dir / "poses.json"
    if not path.is_file():
        return np.zeros((0, 3, 3))
    try:
        frames = json.loads(path.read_text()).get("frames", [])
    except Exception:
        return np.zeros((0, 3, 3))
    rots = [np.asarray(f["R"], dtype=np.float64) for f in frames if f.get("R")]
    rots = [r for r in rots if r.shape == (3, 3) and np.isfinite(r).all()]
    return np.asarray(rots) if rots else np.zeros((0, 3, 3))


def _camera_centers(run_dir: Path) -> np.ndarray:
    path = run_dir / "poses.json"
    if not path.is_file():
        return np.zeros((0, 3))
    try:
        frames = json.loads(path.read_text()).get("frames", [])
    except Exception:
        return np.zeros((0, 3))
    pts = [np.asarray(f["t"], dtype=np.float64) for f in frames if f.get("t")]
    pts = [p for p in pts if p.shape == (3,) and np.isfinite(p).all()]
    return np.asarray(pts) if pts else np.zeros((0, 3))


def _camera_fleet_priors(rots: np.ndarray) -> dict:
    """World directions of the camera fleet's own axes (from poses.json).

    Two candidate verticals fall out of a monocular aerial reconstruction:

    * ``axis`` — the mean optical axis (+Z_cam). A downward-looking camera has
      the ground ALONG this axis, so world up ≈ -axis. This is the right
      candidate for nadir survey imagery.
    * ``image_up`` — mean of -Y_cam, the world direction of the top of the
      frame. This is world up only for a LEVEL camera; for nadir imagery it is
      a HORIZONTAL ground direction, which is why it must not be used blindly.
    """
    if len(rots) == 0:
        return {"cameras": 0, "axis": None, "image_up": None, "axis_spread": None}
    axis = rots[:, :, 2]
    image_up = -rots[:, :, 1]
    a = axis.mean(0)
    na = float(np.linalg.norm(a))
    u = image_up.mean(0)
    nu = float(np.linalg.norm(u))
    return {
        "cameras": int(len(rots)),
        "axis": a / na if na > 1e-9 else None,
        "image_up": u / nu if nu > 1e-9 else None,
        # how tightly the optical axes cluster: near-nadir survey imagery is tight
        "axis_spread": round(float(np.median(axis @ (a / na))) if na > 1e-9 else 0.0, 4),
    }


def _enu_vertical(run_dir: Path) -> np.ndarray | None:
    """The true vertical in the ARTIFACT frame, taken from the georef alignment.

    When telemetry placed the run, ``georef/alignment.json`` maps the artifact
    frame to the local ENU frame (X east, Y north, Z up). ENU up expressed in
    artifact coordinates is therefore ``R^T @ [0,0,1]`` — a metric, signed,
    exactly-known ground normal that needs no camera or geometry assumption.
    Returns None unless the stored transform is a proper rigid rotation.
    """
    path = run_dir / "georef" / "alignment.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
        m = np.asarray(doc.get("matrix"), dtype=np.float64)
    except Exception:
        return None
    if m.shape != (4, 4) or not np.isfinite(m).all():
        return None
    R = m[:3, :3]
    det = float(np.linalg.det(R))
    if det <= 0:
        return None  # singular or a mirror — refusing to guess an up axis
    # The stored transform may be a SCALED rotation (a run placed with a
    # measured visual->telemetry scale carries 0.9888 here on
    # flight_to_tower_7511dc): normalise by the uniform scale before demanding
    # orthonormality, or a valid ENU frame gets thrown away.
    Rn = R / (det ** (1.0 / 3.0))
    if float(np.max(np.abs(Rn @ Rn.T - np.eye(3)))) > 1e-3:
        return None
    up = Rn.T @ np.array([0.0, 0.0, 1.0])
    n = float(np.linalg.norm(up))
    if n < 1e-9:
        return None
    return up / n


def _sheet_normal(pts: np.ndarray, cameras: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Normal of the dominant surface, oriented toward the cameras.

    A single-pass reconstruction is a shell, and the cameras fly on the side
    the ground normal points to — that is the one sign we always know. Returns
    (normal or None, planarity: fraction of variance carried by the shell).
    """
    if len(pts) < 100:
        return None, 0.0
    centered = pts - pts.mean(0)
    cov = centered.T @ centered / len(centered)
    w, v = np.linalg.eigh(cov)
    total = float(w.sum())
    if total <= 0:
        return None, 0.0
    n = v[:, 0]
    planarity = float(1.0 - w[0] / max(w[1], 1e-12))
    if len(cameras) >= 3:
        # up points from the surface toward the fleet
        if float((cameras.mean(0) - pts.mean(0)) @ n) < 0:
            n = -n
    return n / np.linalg.norm(n), planarity


def _vertical_prior(run_dir: Path, pts: np.ndarray | None, cameras: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """The signed up prior, in descending order of authority.

    1. ``enu_vertical`` — metric, exact (telemetry-placed runs).
    2. ``camera_facing_surface`` — the shell normal, when the fleet's optical
       axes point at that shell (nadir imagery: up ≈ -axis and the surface is
       perpendicular to it). Correct for the common nadir survey case where
       the old image-up assumption was ~90 deg wrong.
    3. ``camera_image_up`` — level-flight assumption, the last camera-based
       resort (unchanged legacy behaviour).
    4. ``smallest_extent_axis`` — no camera data at all.
    """
    evidence: dict = {}
    enu = _enu_vertical(run_dir)
    if enu is not None:
        evidence["up_prior"] = "enu_vertical"
        evidence["enu_vertical_world"] = [round(float(x), 6) for x in enu]
        return enu, evidence

    rots = _camera_frames(run_dir)
    fleet = _camera_fleet_priors(rots)
    evidence["cameras"] = fleet["cameras"]
    if fleet["axis"] is not None:
        evidence["optical_axis_world"] = [round(float(x), 6) for x in fleet["axis"]]
        evidence["optical_axis_cluster"] = fleet["axis_spread"]

    if pts is not None:
        sheet, planarity = _sheet_normal(pts, cameras)
        evidence["shell_planarity"] = round(planarity, 4)
        if sheet is not None:
            evidence["shell_normal_world"] = [round(float(x), 6) for x in sheet]
            if fleet["axis"] is not None:
                facing = float(abs(sheet @ fleet["axis"]))
                evidence["shell_axis_alignment"] = round(facing, 4)
                # the fleet looks AT the shell (nadir): the shell normal is the
                # vertical, and the sign already points toward the cameras
                if facing >= _MIN_SHELL_AXIS_ALIGNMENT:
                    evidence["up_prior"] = "camera_facing_surface"
                    return sheet, evidence
            else:
                evidence["up_prior"] = "camera_facing_surface"
                return sheet, evidence

    if fleet["image_up"] is not None:
        evidence["up_prior"] = "camera_image_up"
        evidence["camera_image_up_world"] = [round(float(x), 6) for x in fleet["image_up"]]
        return fleet["image_up"], evidence

    if pts is not None:
        extent = pts.max(0) - pts.min(0)
        u = np.zeros(3)
        u[int(np.argmin(extent))] = 1.0
        evidence["up_prior"] = "smallest_extent_axis"
        return u, evidence

    evidence["up_prior"] = "unavailable"
    return None, evidence


# ---------------------------------------------------------------------------
# ground-plane estimation
# ---------------------------------------------------------------------------


def _plane_from_3pts(p: np.ndarray) -> np.ndarray | None:
    n = np.cross(p[1] - p[0], p[2] - p[0])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None
    n = n / norm
    d = -n @ p[0]
    return np.array([*n, d])


def _ransac_ground_plane(band: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """Deterministic RANSAC plane fit; returns (normal(3,), inlier mask)."""
    if len(band) < 50:
        return None
    extent = float(np.linalg.norm(band.max(0) - band.min(0)))
    thr = max(extent / 300.0, 1e-4)
    best_mask = None
    for _ in range(200):
        idx = rng.choice(len(band), size=3, replace=False)
        plane = _plane_from_3pts(band[idx])
        if plane is None:
            continue
        dist = np.abs(band @ plane[:3] + plane[3])
        mask = dist < thr
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask = mask
    if best_mask is None or best_mask.sum() < 50:
        return None
    # refine: PCA of inliers, smallest-eigenvalue direction = normal
    inliers = band[best_mask]
    centered = inliers - inliers.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n = vt[2]
    dist = np.abs(band @ n - float(band[best_mask].mean(0) @ n))
    best_mask = dist < thr
    return n / np.linalg.norm(n), best_mask


def _estimate_ground(
    pts: np.ndarray, up_prior: np.ndarray, evidence: dict | None = None
) -> tuple[np.ndarray | None, dict]:
    """Estimate the dominant ground normal from the band the prior selects.

    ``up_prior`` is a SIGNED vertical (see ``_vertical_prior``); the ground is
    the low band along it. The fitted plane's normal has an arbitrary sign from
    the SVD, so it is canonicalised against the prior BEFORE agreement is
    judged -- rejecting a level plane over a sign flip was its own bug.
    """
    evidence: dict = dict(evidence or {})

    lo = pts @ up_prior
    cutoff = np.quantile(lo, _BAND_FRACTION)
    band = pts[lo <= cutoff]
    evidence["band_points"] = int(len(band))

    rng = np.random.default_rng(0)
    fit = _ransac_ground_plane(band, rng)
    if fit is None:
        evidence["method"] = "camera_up_prior"
        return None, evidence
    n, mask = fit
    inlier_frac = float(mask.mean())
    if float(n @ up_prior) < 0:  # SVD sign is arbitrary
        n = -n
    inliers = band[mask]
    rms = float(np.sqrt(np.mean((inliers @ n - float(inliers.mean(0) @ n)) ** 2)))
    agreement = float(n @ up_prior)
    evidence.update(
        band_inlier_fraction=round(inlier_frac, 4),
        plane_rms_m=round(rms, 5),
        camera_up_agreement=round(agreement, 4),
        plane_to_prior_deg=round(float(np.degrees(np.arccos(np.clip(abs(agreement), -1, 1)))), 3),
    )
    if inlier_frac < _MIN_BAND_INLIER_FRAC or agreement < 0.5:
        evidence["method"] = f"up_prior:{evidence.get('up_prior', 'unknown')}"
        return None, evidence
    evidence["method"] = "geometry_plane_fit"
    return n, evidence


# ---------------------------------------------------------------------------
# corridor axis
# ---------------------------------------------------------------------------


def _corridor_axis(cameras: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Dominant horizontal direction of the flight, for +X alignment."""
    evidence: dict = {}
    if len(cameras) >= 5:
        proj = cameras - (cameras @ normal)[:, None] * normal
        centered = proj - proj.mean(0)
        cov = centered.T @ centered / len(centered)
        w, v = np.linalg.eigh(cov)
        ratio = float(w[2] / max(w[1], 1e-12))
        evidence["corridor_pca_ratio"] = round(ratio, 3)
        if ratio >= _MIN_CORRIDOR_RATIO:
            h = v[:, 2]
            # stable sign: from first-half centroid to second-half centroid
            half = len(cameras) // 2
            s = proj[half:].mean(0) - proj[:half].mean(0)
            if float(h @ s) < 0:
                h = -h
            return h, evidence
    evidence["corridor_axis"] = "not_reliable"
    return None, evidence


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _identity_payload(run_dir: Path, reason: str) -> dict:
    return {
        "version": _ALIGNMENT_VERSION,
        "run_id": run_dir.name,
        "source_coordinate_convention": (
            "original reconstruction coordinates (poses.json: X_world = R @ X_cam + t, "
            "t = camera centre; metric)"
        ),
        "viewer_coordinate_convention": (
            "rigidly rotated presentation frame: X_view = R_view @ X_world + t_view; "
            "ground normal → +Y, corridor → +X where reliable; scale exactly 1.0; "
            "authoritative artifacts are never modified"
        ),
        "R_view": np.eye(3).tolist(),
        "t_view": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "ground_normal_world": None,
        "ground_reference_point_world": None,
        "horizontal_direction_world": None,
        "method": "identity_fallback",
        "confidence": "low",
        "leveling": {
            "up_prior_source": None,
            "confidence": "low",
            "plane_measured": False,
        },
        "fallback_reason": reason,
        "inputs": _alignment_inputs(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def compute_alignment(run_dir: Path) -> dict:
    """Compute (or reuse) the viewer alignment for one run directory."""
    cached = _load_cached(run_dir)
    if cached is not None:
        return cached

    pts = _load_points(run_dir)
    cameras = _camera_centers(run_dir)
    if pts is not None:
        pts = _downsample(pts)
    up_prior, prior_evidence = _vertical_prior(run_dir, pts, cameras)
    if pts is None and up_prior is None:
        payload = _identity_payload(run_dir, "no geometry artifacts and no camera poses")
        _save(run_dir, payload)
        return payload

    evidence: dict = {
        "cameras": int(len(cameras)),
        "points_sampled": int(len(pts)) if pts is not None else 0,
        "up_prior_source": prior_evidence.get("up_prior"),
        "up_prior": prior_evidence,
    }

    ground = None
    if pts is not None and up_prior is not None:
        ground, geo_evidence = _estimate_ground(pts, up_prior, prior_evidence)
        evidence["ground_estimation"] = geo_evidence
    if ground is None and up_prior is not None:
        # geometry inconclusive: the prior IS the ground normal, and the report
        # says which prior that was rather than implying a plane was measured
        ground = up_prior
        evidence.setdefault("ground_estimation", {})[
            "method"
        ] = f"up_prior:{prior_evidence.get('up_prior', 'unknown')}"
    if ground is None:
        payload = _identity_payload(run_dir, "ground plane could not be estimated")
        _save(run_dir, payload)
        return payload

    # orient ground normal upward (viewer +Y)
    if up_prior is not None and float(ground @ up_prior) < 0:
        ground = -ground

    # corridor → +X
    corridor, corr_evidence = (
        _corridor_axis(cameras, ground) if len(cameras) >= 5 else (None, {})
    )
    evidence["corridor"] = corr_evidence
    if corridor is None:
        # stable fallback: project the largest mesh/point AABB horizontal axis
        if pts is not None:
            extent = pts.max(0) - pts.min(0)
            a = np.zeros(3)
            a[int(np.argmax(extent))] = 1.0
            corridor = a - (a @ ground) * ground
            if np.linalg.norm(corridor) < 1e-6:
                corridor = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ ground) * ground
            corridor /= np.linalg.norm(corridor)
        else:
            corridor = np.cross(ground, [0.0, 0.0, 1.0])
            corridor /= np.linalg.norm(corridor)

    # orthonormal viewer basis: e_x = corridor, e_y = ground normal, e_z = e_x × e_y
    e_x = corridor - (corridor @ ground) * ground
    e_x /= np.linalg.norm(e_x)
    e_y = ground
    e_z = np.cross(e_x, e_y)
    # rounded to 12 dp so the SERVED matrix is still orthonormal to ~1e-12
    # (the payload claims a rigid rotation; 9 dp broke that claim at ~1e-9)
    R = np.round(np.vstack([e_x, e_y, e_z]), 12)  # rows: X_view = R @ X_world

    # t: ground-plane centroid → origin (ground sits on the viewer y=0 grid)
    ground_ref = None
    if pts is not None:
        d = pts @ e_y
        band = pts[d <= np.quantile(d, _BAND_FRACTION)]
        # refined inliers around the plane through the band centroid
        c0 = band.mean(0)
        resid = np.abs((band - c0) @ e_y)
        ground_ref = band[resid <= max(np.quantile(resid, 0.5), 1e-6)].mean(0)
    t = -(R @ ground_ref) if ground_ref is not None else np.zeros(3)

    # rigidity verification on a sample of real points
    verification: dict = {
        "det_R": round(float(np.linalg.det(R)), 9),
        "orthonormality_error": round(float(np.max(np.abs(R @ R.T - np.eye(3)))), 12),
    }
    if pts is not None:
        sample = pts[np.random.default_rng(0).choice(len(pts), size=min(10_000, len(pts)), replace=False)]
        transformed = sample @ R.T + t
        d0 = np.linalg.norm(sample[1:] - sample[:-1], axis=1)
        d1 = np.linalg.norm(transformed[1:] - transformed[0:-1], axis=1)
        verification["distance_preservation_max_rel_err"] = float(
            np.max(np.abs(d1 - d0) / np.maximum(d0, 1e-12))
        )
        bb = transformed.max(0) - transformed.min(0)
        verification["aligned_bbox_extent_m"] = [round(float(x), 3) for x in bb]
        verification["ground_height_range_m"] = round(
            float(transformed[:, 1].max() - np.quantile(transformed[:, 1], 0.05)), 3
        )

    # Levelling confidence is about WHERE the vertical came from: a metric ENU
    # vertical needs no assumption, a measured plane is good evidence, and a
    # camera-only prior is an assumption that must be visible to the user.
    prior_source = prior_evidence.get("up_prior")
    method = evidence.get("ground_estimation", {}).get("method", "unknown")
    if method == "geometry_plane_fit":
        leveling_confidence = "high"
    elif prior_source == "enu_vertical":
        leveling_confidence = "high"
    elif prior_source == "camera_facing_surface":
        leveling_confidence = "medium"
    else:
        leveling_confidence = "low"
    confidence = "high" if (
        leveling_confidence == "high"
        and evidence.get("corridor", {}).get("corridor_pca_ratio", 0) >= _MIN_CORRIDOR_RATIO
    ) else ("medium" if leveling_confidence != "low" else "low")

    payload = {
        "version": _ALIGNMENT_VERSION,
        "run_id": run_dir.name,
        "source_coordinate_convention": (
            "original reconstruction coordinates (poses.json: X_world = R @ X_cam + t, "
            "t = camera centre; metric)"
        ),
        "viewer_coordinate_convention": (
            "rigidly rotated presentation frame: X_view = R_view @ X_world + t_view; "
            "ground normal → +Y, corridor → +X where reliable; scale exactly 1.0; "
            "authoritative artifacts are never modified"
        ),
        "R_view": [[float(x) for x in row] for row in R],
        "t_view": [round(float(x), 6) for x in t],
        "scale": 1.0,
        "ground_normal_world": [round(float(x), 6) for x in ground],
        "ground_reference_point_world": (
            [round(float(x), 4) for x in ground_ref] if ground_ref is not None else None
        ),
        "horizontal_direction_world": (
            [round(float(x), 6) for x in corridor] if corridor is not None else None
        ),
        "method": method if method != "unknown" else f"up_prior:{prior_source}",
        "confidence": confidence,
        "inputs": _alignment_inputs(run_dir),
        "leveling": {
            "up_prior_source": prior_source,
            "confidence": leveling_confidence,
            "plane_measured": method == "geometry_plane_fit",
            "note": (
                "which evidence fixed the viewer's up axis; 'high' means a metric "
                "ENU vertical or a measured plane, 'medium' a fleet-geometry "
                "inference, 'low' an unverified camera assumption"
            ),
        },
        "evidence": evidence,
        "verification": verification,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(run_dir, payload)
    return payload


def _input_fingerprint(run_dir: Path, rel: str) -> list[int] | None:
    """(size, mtime_ns) of one geometry input, or None when absent."""
    p = run_dir / rel
    if not p.is_file():
        return None
    st = p.stat()
    return [int(st.st_size), int(st.st_mtime_ns)]


def _alignment_inputs(run_dir: Path) -> dict[str, list[int] | None]:
    """The files the alignment was derived from — the cache's validity key.

    The alignment is fitted to the CURRENT geometry, so a re-run that changes
    poses or the dense/sparse cloud must not be served the transform computed
    from the previous model (that silently leaves the viewer unlevelled for the
    rebuilt run).
    """
    inputs: dict[str, list[int] | None] = {}
    for rel in _GEOMETRY_CANDIDATES:
        inputs[rel] = _input_fingerprint(run_dir, rel)
    inputs["poses.json"] = _input_fingerprint(run_dir, "poses.json")
    inputs["georef/alignment.json"] = _input_fingerprint(run_dir, "georef/alignment.json")
    return inputs


def _load_cached(run_dir: Path) -> dict | None:
    path = run_dir / _ALIGN_JSON
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return None
    if doc.get("version") != _ALIGNMENT_VERSION or "R_view" not in doc:
        return None
    stored = doc.get("inputs")
    if not isinstance(stored, dict):
        return None  # pre-fingerprint payload: unknown provenance, recompute
    if stored != _alignment_inputs(run_dir):
        log.info("viewer_alignment_stale_inputs", run=run_dir.name)
        return None
    return doc


def _save(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / _ALIGN_JSON).write_text(json.dumps(payload, indent=2))
    log.info("viewer_alignment_written", run=run_dir.name, method=payload.get("method"))
