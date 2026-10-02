"""Formal metric-validation engine (validation ONLY — never modifies artifacts).

Separates the three accuracy types the SIH evaluation cares about:

A. RELATIVE GEOMETRIC CONSISTENCY — reprojection, cross-view, dense→mesh.
   Consumed from the run's existing reports; never proof of absolute accuracy.

B. METRIC SCALE ACCURACY — known real-world distances vs reconstructed
   distances (scale_error = |d_recon − d_gt| / d_gt).

C. ABSOLUTE SPATIAL ACCURACY — reconstructed coordinates vs INDEPENDENT
   reference (LiDAR / checkpoints). The reference is held out: reconstruction
   never reads it. Registration (recon→reference frame) uses only the camera
   centers' correspondence with the flight log — never the reference itself.

≤1 m certification — STRATA ENGINEERING CRITERION (the SIH specification does
not define a statistic; this is documented and applied consistently):
    certified ≤1 m  ⟺  3D RMSE ≤ 1.0 m  AND  3D P95 ≤ 1.0 m
    measured STRATA→reference (accuracy direction), independent reference,
    registration residual reported alongside. Coverage is reported separately
    and never converted into accuracy. A median ≤1 m is NEVER promoted to a
    whole-reconstruction ≤1 m claim.

Dataset split: airport8/airport3 = development (any tuning), berliner_dom1/
berliner_dom3/bundestag3 = held-out evaluation. Recorded in every report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.logging_config import get_logger
from app.services import lidar

log = get_logger("drone_recon.services.metric_validation")

#: STRATA engineering certification thresholds (see module docstring).
CERT_RMSE_M = 1.0
CERT_P95_M = 1.0

#: Coverage percent a reference must have of the reconstruction's own surface
#: for a certification claim (below this the accuracy sample is too sparse to
#: speak for the reconstruction).
CERT_MIN_COVERAGE_PCT = 80.0

THRESHOLDS = (0.25, 0.5, 1.0, 2.0, 5.0)

#: A held-out reference is frequently delivered in its OWN local frame (a
#: dataset render frame, a survey grid) while the reconstruction lives in the
#: run's GPS-anchored ENU frame. Comparing raw coordinates then measures a
#: frame offset, not accuracy. The registration is therefore MEASURED, and the
#: measurement is gated so a bad fit can never masquerade as accuracy:
#:   * the aligned median must beat a rolled null by this factor, and
#:   * the rigid fit may not need more than this rotation (frames unrelated by
#:     a near-aligned transform are not silently rotated into agreement).
ALIGN_NULL_MAX_FRACTION = 0.5
ALIGN_MAX_ROTATION_DEG = 10.0
#: Points used for the fit / for the honest evaluation are DISJOINT halves of
#: the query sample, so the reported accuracy is not fitted on the same points
#: that produced the transform.
ALIGN_FIT_FRACTION = 0.5
#: A candidate offset must retain this fraction of the best achievable overlap
#: to compete at all — sliver overlaps can fit anything.
COVERAGE_KEEP_FRACTION = 0.6

DEV_SCENES = ("airport8", "airport3")
HELD_OUT_SCENES = ("berliner_dom1", "berliner_dom3", "bundestag3")


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity mapping src→dst. Returns (R, t, scale).

    Standard formulation (Umeyama 1991): scale = Σσᵢ / Σ‖src−μ‖².
    """
    s = np.asarray(src, float)
    d = np.asarray(dst, float)
    mu_s, mu_d = s.mean(0), d.mean(0)
    sc, dc = s - mu_s, d - mu_d
    cov = (dc.T @ sc) / len(s)
    u, w, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt))
    diag = np.array([1.0, 1.0, sign])
    R = u @ np.diag(diag) @ vt
    if with_scale:
        var = float((sc**2).sum() / len(s))  # mean square deviation of src
        scale = float((np.diag(diag) @ np.diag(w)).sum()) / var if var > 1e-12 else 1.0
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    return R, t, scale


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2))) if len(x) else float("nan")


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(x, float), q)) if len(x) else float("nan")


def threshold_percentages(d: np.ndarray) -> dict:
    return {f"within_{t}m_pct": round(float((d <= t).mean() * 100), 2) for t in THRESHOLDS}


# ---------------------------------------------------------------------------
# A. relative consistency (consumed from existing run reports)
# ---------------------------------------------------------------------------


def relative_consistency(ws: Path) -> dict:
    """Collect the run's own relative-quality evidence. No reference data.

    Reads the CURRENT report layouts (a run's evidence lives under
    ``reconstruction_report.json → reconstruction/tracks/stages.bundle_adjustment``
    and ``dense_report.json → stages.cross_view_stats / stages.sparse_dense_consistency``)
    with fallbacks to the older top-level keys. Reading only the retired keys
    made every geometry check report "not measured" for runs that had in fact
    measured them — the same honesty defect as claiming a measurement that was
    never taken, just in the other direction.

    Every block names the file it came from, so a served figure can always be
    traced back to the artifact that produced it.
    """
    out: dict = {"available": False, "sources": {}}
    try:
        rr = json.loads((ws / "reconstruction_report.json").read_text())
        recon = rr.get("reconstruction") or {}
        tracks = rr.get("tracks") or rr.get("track_stats") or {}
        ba = (rr.get("stages") or {}).get("bundle_adjustment") or rr.get("bundle_adjustment") or {}
        median_px = (tracks.get("reprojection_median_px")
                     if tracks.get("reprojection_median_px") is not None
                     else recon.get("mean_reproj_error", rr.get("mean_reprojection_error_px")))
        out["reprojection"] = {
            "median_px": median_px,
            "p95_px": tracks.get("reprojection_p95_px"),
            "mean_px": recon.get("mean_reproj_error", rr.get("mean_reprojection_error_px")),
            "num_cameras": recon.get("num_cameras"),
            "num_points": recon.get("num_points"),
            "ba_backend": ba.get("backend"),
            "ba_initial_px": ba.get("initial_error_px", ba.get("initial")),
            "ba_final_px": ba.get("final_error_px", ba.get("final")),
            "ba_converged": ba.get("converged"),
            "ba_gps_prior_cameras": ba.get("gps_prior_cameras"),
            "ba_gps_prior_rms_m": ba.get("gps_prior_rms_m"),
        }
        out["source_reprojection"] = "reconstruction_report.json"
        if tracks:
            out["tracks"] = {
                k: tracks.get(k)
                for k in ("points", "total", "positive_depth_pct", "median_track_length",
                          "observations_median", "observations_p95", "min_triangulation_angle_median_deg",
                          "gate_flags")
                if tracks.get(k) is not None
            }
        out["available"] = median_px is not None or bool(tracks)
    except Exception as exc:
        out["error"] = str(exc)
    try:
        dr = json.loads((ws / "dense_report.json").read_text())
        stages = dr.get("stages") or {}
        xv = stages.get("cross_view_stats") or {}
        sd = stages.get("sparse_dense_consistency") or {}
        quality = dr.get("quality") or {}
        cross_median = xv.get("median_disagreement_m", dr.get("cross_view_disagreement_median_m"))
        out["cross_view"] = {
            "median_m": cross_median,
            "p95_m": xv.get("p95_disagreement_m"),
            "measure": "depth disagreement between overlapping views (geometry only)",
            "source": "dense_report.json → stages.cross_view_stats",
        }
        out["sparse_dense"] = {
            "correspondences": sd.get("correspondences"),
            "median_m": sd.get("median_m"),
            "p95_m": sd.get("p95_m"),
            "within_3m_pct": sd.get("within_3m_pct"),
            "screened_median_m": sd.get("screened_median_m"),
            "source": "dense_report.json → stages.sparse_dense_consistency",
        }
        # Compatibility block for callers reading the older shape.
        out["dense"] = {
            "points": quality.get("point_count"),
            "cross_view_disagreement_median_m": cross_median,
            "mean_spacing_m": quality.get("mean_spacing"),
            "coverage_percent": quality.get("coverage_percent"),
            "grade": quality.get("grade"),
        }
        out["source_cross_view"] = "dense_report.json → stages.cross_view_stats"
        out["source_sparse_dense"] = "dense_report.json → stages.sparse_dense_consistency"
    except Exception as exc:
        out.setdefault("errors", []).append(f"dense_report: {exc}")
    try:
        mr = json.loads((ws / "mesh_quality_report.json").read_text())
        dm = mr.get("dense_support") or mr.get("dense_to_mesh") or {}
        if dm:
            out["dense_to_mesh"] = {
                "support_pct": dm.get("support_percent", dm.get("support_pct")),
                "median_m": dm.get("median_support_distance_m", dm.get("median_m")),
                "p95_m": dm.get("p95_support_distance_m", dm.get("p95_m")),
                "components": mr.get("components"),
                "largest_component_pct": mr.get("largest_component_pct"),
                "edge_median_m": mr.get("edge_median_m"),
                "source": "mesh_quality_report.json → dense_support",
            }
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# B. scale validation
# ---------------------------------------------------------------------------


def scale_validation_from_trajectory(cam_centers: np.ndarray, gt_centers: np.ndarray) -> dict:
    """Scale error from the flight-path length (a known real-world distance)."""
    def path_len(P):
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    d_recon = path_len(cam_centers)
    d_gt = path_len(gt_centers)
    scale_error = abs(d_recon - d_gt) / d_gt if d_gt > 0 else float("nan")
    return {
        "known_distance": "flight_path_length",
        "reconstructed_m": round(d_recon, 3),
        "ground_truth_m": round(d_gt, 3),
        "scale_error": round(scale_error, 5),
        "scale_factor": round(d_recon / d_gt, 5) if d_gt > 0 else None,
        "note": "trajectory length from telemetry correspondence; pairwise camera "
                "distances are identical by construction for telemetry-fixed runs, "
                "so path length is the honest independent distance here",
    }


# ---------------------------------------------------------------------------
# C. registration + absolute validation
# ---------------------------------------------------------------------------


def registration_from_telemetry(
    recon_centers: np.ndarray, telemetry_centers: np.ndarray
) -> dict:
    """Rigid (scale-free) registration recon→telemetry/GT frame + residual.

    Uses ONLY camera-center correspondence (frame_id ↔ telemetry row). The
    reference LiDAR is never consulted; the telemetry frame is the dataset's
    metric frame (verified: telemetry XY over LiDAR ground ⇒ 0.29 m).
    """
    n = min(len(recon_centers), len(telemetry_centers))
    if n < 3:
        return {"matched_frames": n, "sufficient": False,
                "reason": "fewer than 3 camera↔telemetry correspondences"}
    R, t, s = umeyama(recon_centers[:n], telemetry_centers[:n], with_scale=True)
    resid = np.linalg.norm(((R @ (s * recon_centers[:n]).T).T + t) - telemetry_centers[:n], axis=1)
    return {
        "matched_frames": int(n),
        "sufficient": bool(n >= 10),
        "method": "umeyama similarity on camera centers (recon→telemetry frame)",
        "scale": round(float(s), 6),
        "scale_is_identity": bool(abs(float(s) - 1.0) < 0.01),
        "residual_median_m": round(float(np.median(resid)), 4),
        "residual_p95_m": round(float(np.percentile(resid, 95)), 4),
        "residual_rmse_m": round(_rmse(resid), 4),
        "transform_matrix": [[round(float(x), 9) for x in row] for row in R],
        "translation": [round(float(x), 6) for x in t],
    }


def checkpoint_metrics(recon_pts: np.ndarray, gt_pts: np.ndarray) -> dict:
    """Checkpoint-style error statistics on correspondence points (§3)."""
    if len(recon_pts) == 0 or len(recon_pts) != len(gt_pts):
        return {"N": 0}
    e = np.linalg.norm(recon_pts - gt_pts, axis=1)
    horiz = np.linalg.norm((recon_pts - gt_pts)[:, :2], axis=1)
    vert = np.abs((recon_pts - gt_pts)[:, 2])
    return {
        "N": int(len(e)),
        "mean": round(float(e.mean()), 4),
        "median": round(float(np.median(e)), 4),
        "rmse": round(_rmse(e), 4),
        "p90": round(_pct(e, 90), 4),
        "p95": round(_pct(e, 95), 4),
        "max": round(float(e.max()), 4),
        "horizontal_rmse": round(_rmse(horiz), 4),
        "vertical_rmse": round(_rmse(vert), 4),
        "horizontal_p95": round(_pct(horiz, 95), 4),
        "vertical_p95": round(_pct(vert, 95), 4),
        **threshold_percentages(e),
    }


def cloud_to_reference(
    recon: np.ndarray,
    ref: np.ndarray,
    ref_normals: np.ndarray | None = None,
    max_query: int = 40_000,
    max_ref: int = 3_000_000,
) -> dict:
    """Bidirectional cloud-to-reference distances (accuracy vs coverage split).

    STRATA→reference measures ACCURACY of reconstructed surface points.
    reference→STRATA measures COVERAGE (a missing surface is never counted as
    an accurate point — it shows up here, not in the accuracy numbers).
    """
    from scipy.spatial import cKDTree

    if len(recon) == 0 or len(ref) == 0:
        return {"available": False, "reason": "empty cloud"}
    ref_q = ref[:: max(1, int(np.ceil(len(ref) / max_ref)))]
    rq = recon[np.linspace(0, len(recon) - 1, min(max_query, len(recon))).astype(int)]

    tree_ref = cKDTree(ref_q)
    d_sr, idx = tree_ref.query(rq, k=1)

    out: dict = {
        "reference_points_used": int(len(ref_q)),
        "query_points": int(len(rq)),
        "strata_to_reference_m": {
            "median": round(float(np.median(d_sr)), 4),
            "mean": round(float(d_sr.mean()), 4),
            "rmse": round(_rmse(d_sr), 4),
            "p90": round(_pct(d_sr, 90), 4),
            "p95": round(_pct(d_sr, 95), 4),
            "p99": round(_pct(d_sr, 99), 4),
            "max": round(float(d_sr.max()), 4),
            **threshold_percentages(d_sr),
        },
    }
    # point-to-plane where reference normals exist (subset with normals)
    if ref_normals is not None:
        nq = ref_normals[:: max(1, int(np.ceil(len(ref_normals) / max_ref)))]
        n_at = nq[idx]
        vec = rq - ref_q[idx]
        plane_d = np.abs(np.einsum("ij,ij->i", vec, n_at))
        ok = np.isfinite(plane_d)
        out["strata_to_reference_point_to_plane_m"] = {
            "median": round(float(np.median(plane_d[ok])), 4),
            "rmse": round(_rmse(plane_d[ok]), 4),
            "p95": round(_pct(plane_d[ok], 95), 4),
            "note": "uses reference normals; |(p−q)·n_ref| — signed surface distance",
        }
    # coverage direction
    tree_rec = cKDTree(rq)
    d_rs, _ = tree_rec.query(ref_q, k=1)
    out["reference_to_strata_coverage_m"] = {
        "median": round(float(np.median(d_rs)), 4),
        "p95": round(_pct(d_rs, 95), 4),
        "coverage_within_1m_pct": round(float((d_rs <= 1.0).mean() * 100), 2),
        "note": "COVERAGE, not accuracy: fraction of the reference surface the "
                "reconstruction actually represents",
    }
    # coverage of the reconstruction's own footprint by the reference
    recon_extent_box = (rq.min(0), rq.max(0))
    in_box = np.all((ref_q >= recon_extent_box[0] - 5) & (ref_q <= recon_extent_box[1] + 5), axis=1)
    cov = float(in_box.mean() * 100)
    out["reference_coverage_of_recon_bbox_pct"] = round(cov, 2)
    return out


# ---------------------------------------------------------------------------
# §13 error decomposition — why are cloud-to-reference numbers what they are?
# ---------------------------------------------------------------------------


def error_decomposition(
    recon: np.ndarray,
    ref_points: np.ndarray,
    ref_class: np.ndarray | None = None,
    sample: int = 40_000,
    max_ref: int = 3_000_000,
    seed: int = 0,
) -> dict:
    """Decompose STRATA→reference error into interpretable causes.

    Measures: horizontal vs vertical components, signed height bias,
    error vs reconstruction height, error vs reference class (where LiDAR
    classification exists), and the reconstruction's own sampling spacing
    (a lower bound on any NN-based distance). Registration quality is
    evidenced separately via `registration`; coverage via `cloud_to_reference`.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    n = min(sample, len(recon))
    rq = recon[rng.choice(len(recon), size=n, replace=False)]
    stride = max(1, int(np.ceil(len(ref_points) / max_ref)))
    Pq = ref_points[::stride]
    tree = cKDTree(Pq)
    d, idx = tree.query(rq, k=1)
    delta = rq - Pq[idx]
    horiz = np.linalg.norm(delta[:, :2], axis=1)
    out: dict = {
        "query_points": int(n),
        "reference_points_used": int(len(Pq)),
        "horizontal_median_m": round(float(np.median(horiz)), 4),
        "vertical_abs_median_m": round(float(np.median(np.abs(delta[:, 2]))), 4),
        "vertical_signed_median_m": round(float(np.median(delta[:, 2])), 4),
        "note": "signed vertical median > 0 means the reconstruction sits ABOVE "
                "the reference surface on average (systematic, not random)",
    }
    # self-spacing of the reconstruction cloud itself (NOT the query sample):
    # the sampling-gap floor any NN-based distance inherits from cloud density
    if len(recon) >= 2000:
        sub_n = min(400_000, len(recon))
        sub = recon[rng.choice(len(recon), size=sub_n, replace=False)]
        self_d, _ = cKDTree(sub).query(sub[:: max(1, len(sub) // 20_000)], k=2)
        out["recon_self_nn_median_m"] = round(float(np.median(self_d[:, 1])), 4)
        out["recon_points_total"] = int(len(recon))
        out["sampling_gap_note"] = (
            "NN distances cannot fall below ~half the reconstruction's own point "
            "spacing; if recon_self_nn_median is comparable to the median error, "
            "sampling sparsity — not surface error — dominates"
        )
    # error vs reconstruction height
    zs = rq[:, 2]
    edges = np.percentile(zs, [0, 25, 50, 75, 100])
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue  # degenerate zero-height band (constant z) — no info
        m = (zs >= lo) & (zs <= hi if hi == edges[-1] else zs < hi)
        if m.sum() >= 50:
            bands.append({
                "z_range_m": [round(float(lo), 2), round(float(hi), 2)],
                "n": int(m.sum()),
                "median_m": round(float(np.median(d[m])), 4),
                "p95_m": round(_pct(d[m], 95), 4),
            })
    out["error_by_height_band"] = bands
    # error vs reference class
    if ref_class is not None:
        cq = ref_class[::stride]
        c_at = cq[idx]
        by_class = {}
        for c in np.unique(c_at):
            m = c_at == c
            by_class[str(int(c))] = {
                "n": int(m.sum()),
                "median_m": round(float(np.median(d[m])), 4),
                "p95_m": round(_pct(d[m], 95), 4),
            }
        out["error_by_reference_class"] = by_class
    return out


# ---------------------------------------------------------------------------
# certification
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# canonical artifact — the ONE accuracy artifact and its ONE writer
# ---------------------------------------------------------------------------

#: Canonical accuracy artifact, relative to the run workspace. Every consumer
#: (API, UI, reports) reads this name; nothing else may invent one.
ARTIFACT_RELATIVE_PATH = Path("validation") / "validation_report.json"


#: Certification status for a run where nothing has been measured yet.
STATUS_NOT_MEASURED = "NOT MEASURED — NO ACCURACY REPORT"

#: The one sentence describing a run with no measurements.
REASON_NOT_MEASURED = "no accuracy measurement has been performed for this run"


@dataclass
class AccuracyFacts:
    """The measured facts the capability ladder is derived from.

    Deliberately a plain input record: the ladder is a pure function of
    measurements, never of which files happen to exist. ``has_relative_*``
    is what separates a run that was measured (relative reconstruction
    exists) from one that simply has no accuracy report.
    """

    scale_factor: float = 1.0
    has_reference: bool = False
    reference_validated: bool = False
    has_georeferencing: bool = False
    has_relative_measurement: bool = False


def build_report(
    run_id: str,
    *,
    scene: str | None = None,
    dataset_split: str | None = None,
) -> dict:
    """Construct the canonical report shape — and assert NOTHING.

    SINGLE OWNER of the report shape: ``validate_run`` fills this skeleton
    with measurements and ``not_certified_report`` serves it unchanged. Both
    therefore produce identical keys, so the served envelope can never drift
    from a real report.

    Every measured section starts as ``None`` and relative quality reads
    "not measured". A field only carries a value once something measured it —
    the previous envelope hardcoded relative quality to "available" (and so
    claimed reprojection/cross-view/dense→mesh diagnostics for runs that
    never executed), which is the fabrication class this module exists to
    prevent.
    """
    return {
        "scene": scene,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_split": dataset_split,
        "certification_criterion": {
            "rule": "3D RMSE ≤ 1.0 m AND 3D P95 ≤ 1.0 m (STRATA→reference)",
            "label": "STRATA engineering criterion (SIH spec does not define a statistic)",
            "min_reference_coverage_pct": CERT_MIN_COVERAGE_PCT,
        },
        # measured sections — absent until measured
        "relative_consistency": None,
        "registration": None,
        "checkpoint_validation": None,
        "scale_validation": None,
        "reference_comparison": None,
        "reference_alignment": None,
        "reference_terrain_split": None,
        "reference_type": None,
        "reference_source": None,
        "error_decomposition": None,
        "figures": None,
        "accuracy_certified": False,
        "certification_status": STATUS_NOT_MEASURED,
        "certification_reason": REASON_NOT_MEASURED,
        "accuracy_summary": {
            "relative_reconstruction": "not measured",
            "metric_scale": "not measured",
            "absolute": STATUS_NOT_MEASURED,
        },
        "capability_state": "NOT_MEASURED",
        "capability_reason": REASON_NOT_MEASURED,
    }


def capability_state(facts: AccuracyFacts) -> tuple[str, str]:
    """Derive the capability claim and its justification — single owner.

    Ladder (a claim is only made when its evidence exists):
        METRIC_VALIDATED   independent reference (checkpoints/LiDAR) validated
        GPS_GEOREFERENCED  telemetry correspondence georeferenced the run
        SCALED             metric scale calibrated, nothing independent checked
        RELATIVE_ONLY      uncalibrated relative reconstruction

    This replaced a duplicate derivation in the retired
    ``build_metric_accuracy_report``, whose defaults claimed
    GPS_GEOREFERENCED for runs with no GPS at all.
    """
    if facts.reference_validated and facts.has_reference:
        return "METRIC_VALIDATED", (
            "Metric validated against an independent reference "
            "(checkpoints/LiDAR, held out from reconstruction)"
        )
    if facts.has_georeferencing:
        return "GPS_GEOREFERENCED", (
            "GPS georeferenced (not validated against independent ground truth)"
        )
    if abs(facts.scale_factor - 1.0) > 1e-4:
        return "SCALED", "Scaled relative reconstruction (not validated)"
    if facts.has_relative_measurement:
        return "RELATIVE_ONLY", "Relative uncalibrated reconstruction (not validated)"
    return "NOT_MEASURED", REASON_NOT_MEASURED


def not_certified_report(run_id: str, reason: str) -> dict:
    """The honest envelope for a run with no measurement artifact.

    Built through :func:`build_report`, so it shares the report's exact shape
    (and cannot drift from it) while asserting nothing: no accuracy figure, no
    provenance, no uncertainty, and relative quality reported as not measured.
    """
    report = build_report(run_id)
    report["certification_reason"] = reason
    report["capability_reason"] = reason
    return report


def persist_report(workspace: Path, report: dict) -> Path:
    """Write the canonical accuracy artifact. THE ONLY writer in the codebase.

    Every caller (the CLI, the API) persists through this function so the
    artifact can never be produced with a second schema or a second name.
    """
    ws = Path(workspace)
    path = ws / ARTIFACT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    log.info("metric_validation_report_written", path=str(path),
             certification_status=report.get("certification_status"))
    return path


def load_report(workspace: Path) -> dict | None:
    """Read the canonical artifact, or None when this run has none."""
    path = Path(workspace) / ARTIFACT_RELATIVE_PATH
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("metric_validation_report_unreadable", path=str(path), error=str(exc))
        return None


def certify(stats: dict, reg: dict, ref_available: bool) -> tuple[str, str]:
    """Apply the STRATA engineering criterion. Returns (status, reason)."""
    if not ref_available:
        return "NOT CERTIFIED — NO INDEPENDENT REFERENCE", \
            "no independent LiDAR/checkpoint reference exists for this scene"
    if not reg.get("sufficient"):
        return "NOT CERTIFIED — INSUFFICIENT GEOREFERENCING", \
            reg.get("reason", "camera↔reference correspondence below minimum")
    a = stats.get("strata_to_reference_m")
    if not a:
        return "NOT CERTIFIED — NO INDEPENDENT REFERENCE", "reference comparison unavailable"
    rmse, p95 = a["rmse"], a["p95"]
    cov = stats.get("reference_coverage_of_recon_bbox_pct", 0.0)
    if rmse <= CERT_RMSE_M and p95 <= CERT_P95_M:
        if cov < CERT_MIN_COVERAGE_PCT:
            return "VALIDATION AVAILABLE — COVERAGE LIMITED", (
                f"accuracy passes (RMSE {rmse:.2f} m, P95 {p95:.2f} m) but reference covers "
                f"only {cov:.0f}% of the reconstruction footprint"
            )
        return "CERTIFIED ≤1m (STRATA engineering criterion)", (
            f"3D RMSE {rmse:.3f} m ≤ 1.0 m and 3D P95 {p95:.3f} m ≤ 1.0 m on "
            f"{a['query_points']} sampled reconstruction points vs independent reference"
        )
    return "VALIDATION AVAILABLE — ABOVE 1m", (
        f"3D RMSE {rmse:.3f} m / P95 {p95:.3f} m against the STRATA engineering "
        f"criterion (RMSE ≤ {CERT_RMSE_M} m and P95 ≤ {CERT_P95_M} m)"
    )


# ---------------------------------------------------------------------------
# per-run entry point
# ---------------------------------------------------------------------------


def load_ply_xyz(path: Path) -> np.ndarray:
    from app.services.pointcloud import read_ply

    return read_ply(path).xyz


#: Where the fused reconstruction lives, in order of preference.
DENSE_CANDIDATES = ("dense/dense_model.ply", "combined_model.ply")


def dense_points(ws: Path) -> np.ndarray | None:
    """The run's fused cloud, or None when it has not produced one."""
    for rel in DENSE_CANDIDATES:
        path = Path(ws) / rel
        if path.is_file():
            try:
                return load_ply_xyz(path)
            except Exception:
                return None
    return None


def run_id_for_scene(scene: str, runs: dict) -> str | None:
    return runs.get(scene)


# ---------------------------------------------------------------------------
# internal validation — what is measurable without an independent reference
# ---------------------------------------------------------------------------

#: Declared STRATA engineering sanity bounds for INTERNAL consistency. These
#: are not accuracy claims and are deliberately loose: they exist to flag a run
#: whose self-consistency broke, not to grade it.
INTERNAL_REPROJ_MEDIAN_PX = 2.0
INTERNAL_SPARSE_DENSE_MEDIAN_M = 2.0
INTERNAL_TRAJECTORY_FIT_MEDIAN_M = 5.0
#: Row-order pairing between camera centres and the georef track is only
#: trusted while the residual stays in this band; above it the pairing itself
#: is suspect (a sync bug would otherwise be reported as a geometry fault).
_ROW_PAIRING_TRUST_M = 25.0


def _camera_centres(ws: Path) -> np.ndarray | None:
    try:
        frames = json.loads((ws / "poses.json").read_text())["frames"]
        centres = np.array([f["t"] for f in frames if isinstance(f, dict) and "t" in f], float)
        return centres if len(centres) >= 3 else None
    except Exception:
        return None


def _telemetry_track(ws: Path) -> np.ndarray | None:
    """Per-frame telemetry ENU from ``georef/gps_track.csv``.

    The georef stage writes one row per matched frame, in frame order, so row
    *i* corresponds to camera *i*. That correspondence is an assumption, so
    :func:`validate_run_internal` verifies it against the fit and refuses it
    when the residual says the rows are not really paired.
    """
    import csv

    path = ws / "georef" / "gps_track.csv"
    if not path.is_file():
        return None
    try:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        return np.array([[float(r["east_m"]), float(r["north_m"]), float(r["up_m"])]
                         for r in rows], float)
    except (OSError, ValueError, KeyError):
        return None


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map a cloud through a 4×4 rigid transform.

    Measurement only: validation never rewrites the run's artifacts, so the
    transform exists to put two clouds in ONE frame for the duration of the
    comparison, nothing more.
    """
    P = np.asarray(points, dtype=np.float64)
    M = np.asarray(matrix, dtype=np.float64)
    return P @ M[:3, :3].T + M[:3, 3]


def _max_height_grid(P, lo, gsd, nx, ny):
    """Maximum-return height per cell (NaN where empty) — the surface form."""
    ix = np.clip(((P[:, 0] - lo[0]) / gsd).astype(np.int64), 0, nx - 1)
    iy = np.clip(((P[:, 1] - lo[1]) / gsd).astype(np.int64), 0, ny - 1)
    z = np.full(nx * ny, -np.inf, dtype=np.float64)
    np.maximum.at(z, iy * nx + ix, P[:, 2])
    z[~np.isfinite(z)] = np.nan
    return z.reshape(ny, nx)


def _grid_shift_search(ref_grid, rec_grid, cells, min_cells):
    """Best integer (di, dj) moving *rec* onto *ref*, by MAD of the height residual.

    The score is the median absolute deviation ABOUT the median residual, not
    the median |Δz|: the two clouds differ by a large **vertical datum offset**
    as well as the horizontal one (226 m on the reference here), so a score
    that includes that constant offset is flat across the whole search and
    picks noise. Removing the median per candidate makes the score depend only
    on how well the two SURFACES line up; the vertical offset is then read off
    the winner. This is the same discipline ``dsm_accuracy.coregister`` uses.
    """
    h, w = ref_grid.shape
    pad = cells
    big = np.full((h + 2 * pad, w + 2 * pad), np.nan, dtype=np.float64)
    big[pad:pad + h, pad:pad + w] = rec_grid
    cand: list[tuple[float, int, int, int]] = []
    max_cov = 0
    for dj in range(-cells, cells + 1):
        for di in range(-cells, cells + 1):
            # Slicing, not np.roll: a circular roll wraps cells across the grid
            # and the search happily reports a wrap-around "offset" the data
            # cannot support. Padding with NaN makes every shifted view drop
            # out of the comparison instead of wrapping into a fake overlap.
            view = big[pad - dj:pad - dj + h, pad - di:pad - di + w]
            m = np.isfinite(view) & np.isfinite(ref_grid)
            n = int(m.sum())
            if n < min_cells:
                continue
            dz = view[m] - ref_grid[m]
            score = float(np.median(np.abs(dz - np.median(dz))))
            cand.append((score, di, dj, n))
            max_cov = max(max_cov, n)
    if not cand:
        return None
    # A shift that slides most of the reconstruction off the tile can fit
    # anything from its remaining sliver, so only offsets retaining most of the
    # BEST achievable coverage compete — the same rule the DSM co-registration
    # uses, which is what stops a "registered 2.5 km away" sliver fit.
    floor = max(min_cells, COVERAGE_KEEP_FRACTION * max_cov)
    eligible = [c for c in cand if c[3] >= floor]
    return min(eligible or cand, key=lambda c: c[0])


def coregister_reference(
    recon: np.ndarray,
    ref: np.ndarray,
    *,
    max_ref_points: int = 400_000,
    seed: int = 0,
) -> dict | None:
    """Measure the rigid transform placing the reconstruction in the reference frame.

    A held-out reference is commonly delivered in its OWN local frame (a
    dataset render frame, a survey grid) while the reconstruction lives in the
    run's GPS-anchored ENU frame. Distances between the two then measure the
    frame difference, not the reconstruction — a systematic offset that no
    amount of reconstruction quality can remove, which is why the raw
    comparison reported ~227 m on every LiDAR run instead of a surface error.

    The registration is therefore MEASURED, and measured honestly:

    * the transform is **rigid with unit scale** and near-aligned (a rotation
      beyond :data:`ALIGN_MAX_ROTATION_DEG` means the clouds are not the same
      scene under a frame change, and the fit is refused rather than used);
    * it is fitted on ONE half of the query sample and the accuracy is
      reported on the OTHER half, so the number is not fitted on its own
      evidence;
    * it is **gated against a rolled null** (random offsets over the scene):
      an alignment that does not beat chance by :data:`ALIGN_NULL_MAX_FRACTION`
      is reported as failed instead of being converted into an accuracy figure.

    Scale is never fitted. Fitting scale to a reference is how a wrong
    reconstruction is made to look right, and the STRATA criterion is about
    metric accuracy at the reconstruction's own scale.

    Returns a serialisable record (with a 4×4 ``transform`` mapping
    ``recon→ref``), or ``None`` when the clouds are too small to register.
    """
    from scipy.spatial import cKDTree

    P = np.asarray(recon, dtype=np.float64)
    R = np.asarray(ref, dtype=np.float64)
    stride = max(1, int(np.ceil(len(R) / max_ref_points)))
    R = R[::stride]
    if len(P) < 200 or len(R) < 200:
        return None

    rng = np.random.default_rng(seed)
    q = P[rng.choice(len(P), size=min(len(P), 40_000), replace=False)]
    k = max(1, int(len(q) * ALIGN_FIT_FRACTION))
    fit, ev = q[:k], q[k:]
    if len(ev) < 100:
        return None
    tree_ref = cKDTree(R)

    # ---- coarse-to-fine horizontal registration on maximum-height fields ---
    span = float(max(np.ptp(R[:, 0]), np.ptp(R[:, 1]), 50.0))
    lo2 = np.minimum(P[:, :2].min(0), R[:, :2].min(0)) - 0.1 * span
    hi2 = np.maximum(P[:, :2].max(0), R[:, :2].max(0)) + 0.1 * span
    # One cell per ~32 across the UNION bbox, so the sweep (which spans the grid
    # dimension in cells) covers every offset the two clouds can possibly have —
    # a cap on the sweep would silently miss a reference delivered far away.
    extent = float(max(hi2[0] - lo2[0], hi2[1] - lo2[1], 50.0))
    gsd = max(extent / 32.0, 2.0)
    nx = max(int((hi2[0] - lo2[0]) / gsd) + 1, 2)
    ny = max(int((hi2[1] - lo2[1]) / gsd) + 1, 2)
    ref_g = _max_height_grid(R, lo2, gsd, nx, ny)
    rec_g = _max_height_grid(P, lo2, gsd, nx, ny)
    # The overlap floor is a fraction of the RECONSTRUCTION's own occupancy: a
    # reconstruction covers a small part of a survey tile, so a floor keyed to
    # the tile would reject every shift and register nothing.
    # A low absolute floor: the coarse cell can be large enough that a compact
    # reconstruction occupies only a handful of them, and the coverage rule
    # inside the search (not this floor) is what rejects sliver fits.
    min_cells = max(6, int(0.25 * np.isfinite(rec_g).sum()))
    coarse = _grid_shift_search(ref_g, rec_g, max(nx, ny), min_cells)
    if coarse is None:
        return None
    # A cell shift of di/dj moves the reconstruction by exactly di·gsd, dj·gsd.
    dx = coarse[1] * gsd
    dy = coarse[2] * gsd
    dx, dy = float(coarse[1]) * gsd, float(coarse[2]) * gsd

    # ---- vertical offset: the median height residual at the winning shift --
    shifted = fit + np.array([dx, dy, 0.0])
    d_fit, idx_fit = tree_ref.query(shifted, k=1)
    dz = float(np.median(R[idx_fit][:, 2] - shifted[:, 2]))
    trans = np.array([dx, dy, dz], dtype=np.float64)

    # ---- refine on the metric that is actually REPORTED -------------------
    # The height-field search only places the right basin: a max-height cell is
    # whatever is tallest in it (roof, canopy), so its optimum is coarse and can
    # prefer a neighbouring cell. The final placement is therefore refined on
    # the point-to-point nearest-neighbour MEDIAN — the very quantity the
    # accuracy report uses — by a shrinking pattern search wide enough to
    # recover a one-cell disagreement with the grid stage.
    S = fit[:: max(1, len(fit) // 3000)]

    def _refine(start, init_step, iters=18):
        cur = np.asarray(start, dtype=np.float64)
        step = float(init_step)
        best_med = float(np.median(tree_ref.query(S + cur, k=1)[0]))
        for _ in range(iters):
            cand_best = None
            for ox in (-step, 0.0, step):
                for oy in (-step, 0.0, step):
                    for oz in (-step, 0.0, step):
                        cand = cur + np.array([ox, oy, oz])
                        m = float(np.median(tree_ref.query(S + cand, k=1)[0]))
                        if cand_best is None or m < cand_best[0]:
                            cand_best = (m, cand)
            if cand_best[0] < best_med - 1e-9:
                best_med, cur = cand_best
            step /= 2.5
        return best_med, cur

    # Two seeds, because each is weak in a different way: the grid winner is
    # precise in the basin it found but can be a cell off, while the
    # centroid difference is within a few metres of the truth without any
    # surface reasoning. Whichever lands lower wins; nothing is fitted twice.
    centroid = np.median(R, axis=0) - np.median(fit, axis=0)
    walk = 2.0 * gsd
    _m1, trans = _refine(trans, walk)
    _m2, trans2 = _refine(centroid, walk)
    if _m2 < _m1:
        trans = trans2

    # ---- rigid refine (translation-dominated; rotation must stay tiny) -----
    X = fit + trans
    Rm = np.eye(3)
    for _ in range(25):
        d, idx = tree_ref.query(X, k=1)
        keep = d <= np.percentile(d, 80.0)
        if keep.sum() < 50:
            break
        Rm, tvec, _s = umeyama(X[keep], R[idx][keep], with_scale=False)
        Xn = X @ Rm.T + tvec
        if np.linalg.norm(Xn - X, axis=1).mean() < 1e-6:
            break
        X = Xn
    rotation_deg = float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))))

    # The ICP ran on points ALREADY moved by the coarse translation, so the
    # recon→ref map is  y = Rm·(x + trans) + tvec  — trans must be composed in,
    # not dropped (dropping it left a ~0 shift and silently compared the raw
    # frames again).
    t_full = Rm @ trans + tvec
    fitted = np.eye(4)
    fitted[:3, :3] = Rm
    fitted[:3, 3] = t_full

    def eval_median(pts, matrix=None, shift=None):
        if matrix is not None:
            A = apply_transform(pts, matrix)
        else:
            A = pts if shift is None else pts + shift
        return float(np.median(tree_ref.query(A, k=1)[0]))

    median_before = eval_median(ev)
    median_fitted = eval_median(ev, matrix=fitted)
    # A reference that is ALREADY in the reconstruction's frame must not be
    # perturbed — a fit that does not improve the comparison is discarded in
    # favour of the identity, so an in-frame tile is compared exactly as given.
    if median_fitted < median_before:
        matrix, median_after, applied = fitted, median_fitted, True
    else:
        matrix, median_after, applied = np.eye(4), median_before, False
    # rolled null: the same comparison at random offsets over the scene
    nulls = []
    for _ in range(5):
        off = rng.uniform(-span, span, size=3)
        nulls.append(eval_median(ev, shift=off))
    null_median = float(np.median(nulls))
    null_best = float(np.min(nulls))

    gate_reasons = []
    if applied and rotation_deg > ALIGN_MAX_ROTATION_DEG:
        gate_reasons.append(f"rotation {rotation_deg:.2f}° > {ALIGN_MAX_ROTATION_DEG}°")
    # The gate is the same whether or not a transform was needed: the compared
    # clouds must be far closer than a chance placement of the same clouds.
    # An unregistered 227 m offset fails this; a fitted 3 m alignment passes it.
    if median_after >= ALIGN_NULL_MAX_FRACTION * null_best:
        gate_reasons.append(
            f"median {median_after:.3f} m not below "
            f"{ALIGN_NULL_MAX_FRACTION:.0%} of the rolled null {null_best:.3f} m"
        )

    return {
        "available": True,
        "method": "height-field translation search + rigid refine, gated",
        "applied_to": "reconstruction registered into the reference frame (measurement only)",
        "scale": 1.0,
        "scale_note": "scale is never fitted to a reference — a fitted scale can hide a wrong reconstruction",
        "rotation_deg": round(rotation_deg, 4),
        "translation_m": [round(float(v), 4) for v in t_full],
        "transform": [[round(float(v), 9) for v in row] for row in matrix],
        "fit_points": int(len(fit)),
        "eval_points": int(len(ev)),
        "applied": bool(applied),
        "median_before_m": round(median_before, 4),
        "median_after_m": round(median_after, 4),
        "null_median_m": round(null_median, 4),
        "null_best_m": round(null_best, 4),
        "overlap_cells_coarse": int(coarse[3]),
        "gate_passed": not gate_reasons,
        "gate_reason": "; ".join(gate_reasons) or "aligned and gated against a rolled null",
    }


def accuracy_by_terrain(
    recon: np.ndarray,
    ref: np.ndarray,
    ref_class: np.ndarray | None,
    *,
    max_query: int = 40_000,
    max_ref: int = 3_000_000,
) -> dict | None:
    """Accuracy split into GROUND returns vs everything above them.

    A LiDAR comparison that reports one number over a mixed tile is dominated
    by vegetation and rooftops the reconstruction was never asked to
    reproduce exactly. The split keeps the survey-relevant figure (accuracy on
    the terrain surface) separable from the canopy, and the ASPRS class codes
    come from the reference itself.
    """
    if ref_class is None:
        return None
    from scipy.spatial import cKDTree

    R = np.asarray(ref, dtype=np.float64)
    C = np.asarray(ref_class)
    if len(C) != len(R) or len(R) == 0:
        return None
    stride = max(1, int(np.ceil(len(R) / max_ref)))
    Rq, Cq = R[::stride], C[::stride]
    tree = cKDTree(Rq)
    q = np.asarray(recon, dtype=np.float64)
    q = q[np.linspace(0, len(q) - 1, min(max_query, len(q))).astype(int)]
    d, idx = tree.query(q, k=1)
    cat = Cq[idx]
    out = {}
    for label, mask in (
        ("ground", cat == 2),
        ("off_terrain", cat != 2),
    ):
        if int(mask.sum()) < 50:
            out[label] = {"n": int(mask.sum())}
            continue
        dm = d[mask]
        out[label] = {
            "n": int(mask.sum()),
            "median_m": round(float(np.median(dm)), 4),
            "rmse_m": round(_rmse(dm), 4),
            "p95_m": round(_pct(dm, 95), 4),
            "within_1m_pct": round(float((dm <= 1.0).mean() * 100), 2),
        }
    out["note"] = (
        "ground = reference class 2 (ASPRS); off_terrain = everything else "
        "in the tile (vegetation, roofs, noise)"
    )
    return out


def validate_run_internal(ws: Path, run_id: str | None = None, *, persist: bool = True) -> dict:
    """Build the accuracy report for a run, against its held-out reference if it has one.

    The shape comes from :func:`build_report` (the single owner), so this can
    never drift from the envelope served for an unmeasured run. What it adds
    is the measurement that IS possible without a reference, with the
    circularity made explicit instead of hidden:

    * geometry consistency — reprojection, cross-view, sparse↔dense — never
      saw telemetry, so it is independent evidence about the reconstruction;
    * trajectory agreement — the visual path against the telemetry prior. The
      placement already fitted a similarity to that same prior, so a small
      residual is expected BY CONSTRUCTION and is recorded as such. A large
      one still means something real: the visual trajectory's shape could not
      be reconciled with the flight track.

    Absolute accuracy is measured against the run's own **held-out LiDAR
    input** when one is present (LAS/LAZ, rasterised and compared by
    :mod:`app.services.lidar`) — that reference is never read by
    reconstruction, so the comparison is genuinely independent. With no LiDAR
    it stays NOT MEASURED and nothing here is certified.
    """
    ws = Path(ws)
    report = build_report(run_id or ws.name)
    report["relative_consistency"] = relative_consistency(ws)

    recon = _camera_centres(ws)
    tele = _telemetry_track(ws)
    correspondence: dict = {"available": False}
    if recon is not None and tele is not None:
        if len(recon) == len(tele):
            correspondence = {
                "available": True,
                "frames": int(len(recon)),
                "basis": "row order of georef/gps_track.csv ↔ poses.json frames",
            }
            reg = registration_from_telemetry(recon, tele)
            report["registration"] = count_aware_registration(reg, correspondence)
            report["checkpoint_validation"] = checkpoint_metrics(recon, tele)
            report["scale_validation"] = scale_validation_from_trajectory(recon, tele)
        else:
            report["registration"] = {
                "matched_frames": int(min(len(recon), len(tele))), "sufficient": False,
                "reason": f"camera count ({len(recon)}) != telemetry rows ({len(tele)}) — "
                          f"row-order correspondence not assumed",
            }
    else:
        report["registration"] = {
            "matched_frames": 0, "sufficient": False,
            "reason": "no camera centres and/or no georef telemetry track for this run",
        }
    report["correspondence"] = correspondence

    # Absolute accuracy against the run's own held-out LiDAR. This is the only
    # place in this module that reads a reference, and it reads it after the
    # model exists — reconstruction never sees it.
    held_out = lidar.reference_points(ws)
    dense = dense_points(ws) if held_out is not None else None
    reference_stats: dict | None = None
    alignment_failed = False
    if held_out is not None and dense is not None:
        ref_points, ref_path = held_out
        # The reference usually arrives in the DATASET's own local frame while
        # the reconstruction is in the run's GPS-anchored ENU frame, so the two
        # must be registered before any distance between them means anything.
        # The registration is measured, gated and reported; the run's own
        # artifacts are never modified by it.
        alignment = coregister_reference(dense, ref_points.xyz)
        report["reference_alignment"] = alignment
        if alignment is None:
            report["reference_alignment"] = {
                "available": False,
                "reason": "reference and reconstruction too small to register",
            }
            alignment_failed = True
        elif not alignment.get("gate_passed"):
            # Converting an ungated fit into an accuracy number is exactly the
            # fabrication this module exists to prevent, so a failed gate
            # reports itself instead of a distance.
            log.warning("reference_registration_gate_failed", run=ws.name,
                        reason=alignment.get("gate_reason"))
            alignment_failed = True
        else:
            recon_in_ref = apply_transform(dense, alignment["transform"])
            reference_stats = cloud_to_reference(recon_in_ref, ref_points.xyz)
            if reference_stats.get("available") is False:
                reference_stats = None
            else:
                report["reference_comparison"] = reference_stats
                report["reference_type"] = (
                    "independent LiDAR (held-out; never read by reconstruction; "
                    "registered into the reconstruction frame by a gated rigid fit)"
                )
                report["reference_source"] = ref_path.name
                report["error_decomposition"] = error_decomposition(
                    recon_in_ref, ref_points.xyz, ref_points.classification
                )
                report["reference_terrain_split"] = accuracy_by_terrain(
                    recon_in_ref, ref_points.xyz, ref_points.classification
                )
    if reference_stats is None:
        report["reference_comparison"] = None
        report["reference_type"] = None
        report["reference_source"] = None
    ref_available = reference_stats is not None

    status, reason = certify(
        reference_stats or {}, report["registration"], ref_available=ref_available
    )
    if alignment_failed:
        # A reference that could not be registered is NOT an accuracy verdict.
        status = "NOT CERTIFIED — REFERENCE REGISTRATION FAILED"
        reason = (
            (report.get("reference_alignment") or {}).get("gate_reason")
            or (report.get("reference_alignment") or {}).get("reason")
            or "the held-out reference could not be registered to the reconstruction"
        )
    report["accuracy_certified"] = status.startswith("CERTIFIED")
    report["certification_status"] = status
    report["certification_reason"] = reason
    report["validation_kind"] = (
        "held_out_reference"
        if ref_available
        else ("reference_registration_failed" if alignment_failed else "internal_consistency")
    )
    report["internal_validation"] = _internal_validation(report)

    scale = report.get("scale_validation") or {}
    rel = report.get("relative_consistency") or {}
    report["accuracy_summary"] = {
        "relative_reconstruction": "measured" if rel.get("available") else "not measured",
        "metric_scale": (
            f"telemetry-derived (path-length agreement "
            f"{100 * (1 - abs(1 - (scale.get('scale_factor') or 1.0))):.2f}%)"
            if scale.get("scale_factor")
            else "not measured"
        ),
        "absolute": (
            report["certification_status"]
            if (ref_available or alignment_failed)
            else STATUS_NOT_MEASURED
        ),
    }
    facts = AccuracyFacts(
        scale_factor=float(scale.get("scale_factor") or 1.0),
        has_reference=ref_available,
        reference_validated=bool(report["accuracy_certified"]),
        has_georeferencing=bool((report.get("registration") or {}).get("sufficient")),
        has_relative_measurement=bool(rel.get("available")),
    )
    report["capability_state"], report["capability_reason"] = capability_state(facts)
    if persist:
        persist_report(ws, report)
    return report


def count_aware_registration(reg: dict, correspondence: dict) -> dict:
    """Attach the correspondence provenance and the row-pairing guard."""
    out = dict(reg)
    out["correspondence"] = correspondence.get("basis")
    out["frames_paired"] = correspondence.get("frames")
    median = out.get("residual_median_m")
    if median is not None and median > _ROW_PAIRING_TRUST_M:
        out["sufficient"] = False
        out["row_pairing_suspect"] = True
        out["reason"] = (
            f"fit residual {median:.1f} m exceeds {_ROW_PAIRING_TRUST_M:.0f} m — the camera↔telemetry "
            f"row pairing is suspect, so trajectory agreement is not reported"
        )
    return out


def _internal_validation(report: dict) -> dict:
    """Plain-language verdicts over the measurements, with provenance."""
    rel = report.get("relative_consistency") or {}
    reg = report.get("registration") or {}
    scale = report.get("scale_validation") or {}
    dense = rel.get("dense") or {}
    reproj = (rel.get("reprojection") or {}).get("median_px")
    checks: list[dict] = []

    def check(name: str, value, unit: str, criterion, independent: bool, note: str) -> None:
        measured = value is not None
        checks.append({
            "name": name,
            "measured": measured,
            "value": round(float(value), 4) if measured else None,
            "unit": unit,
            "criterion": f"≤ {criterion} {unit}" if criterion is not None else None,
            "pass": (bool(value <= criterion) if measured and criterion is not None else None),
            "independent_of_telemetry": independent,
            "note": note,
        })

    check("Sparse reprojection error (median)", reproj, "px", INTERNAL_REPROJ_MEDIAN_PX, True,
          "bundle-adjusted reprojection — geometry only, telemetry never entered it")
    cross = (rel.get("cross_view") or {}).get("median_m")
    if cross is None:
        cross = dense.get("cross_view_disagreement_median_m")
    check("Cross-view depth agreement (median)", cross, "m",
          INTERNAL_SPARSE_DENSE_MEDIAN_M, True,
          "depth disagreement between overlapping views on the fused surface — geometry only")
    check("Sparse↔dense agreement (median)",
          (rel.get("sparse_dense") or {}).get("median_m"), "m",
          INTERNAL_SPARSE_DENSE_MEDIAN_M, True,
          "sparse cloud vs fused dense surface nearest-neighbour agreement — geometry only")
    check("Visual vs telemetry trajectory (fit residual median)",
          reg.get("residual_median_m") if reg.get("sufficient") else None, "m",
          INTERNAL_TRAJECTORY_FIT_MEDIAN_M, False,
          "the placement fitted a similarity to this same telemetry track, so a small "
          "value is expected by construction; a large one means the visual trajectory "
          "could not be reconciled with the flight path")
    check("Metric scale vs telemetry (path-length error)",
          (100 * scale["scale_error"]) if scale.get("scale_error") is not None else None,
          "%", 1.0, False,
          "scale was DERIVED from telemetry during placement, so this confirms the "
          "placement, not the reconstruction's independent metric truth")

    measured = [c for c in checks if c["measured"]]
    failed = [c for c in measured if c["pass"] is False]
    independent = [c for c in measured if c["independent_of_telemetry"]]
    if not measured:
        statement = "Nothing was measured for this run."
    elif failed:
        statement = (
            f"{len(failed)} of {len(measured)} internal checks are outside their sanity "
            "bounds: " + "; ".join(f"{c['name']} = {c['value']} {c['unit']}" for c in failed)
        )
    elif independent:
        statement = (
            f"All {len(measured)} measurable internal checks are within their sanity bounds. "
            f"{len(independent)} of them never used telemetry — "
            + ", ".join(c["name"] for c in independent)
            + " — so they are independent evidence about the geometry."
        )
    else:
        statement = (
            f"{len(measured)} checks were measurable and are within their sanity bounds, but "
            "every one of them depends on the telemetry that placed this run, so none of "
            "them is independent evidence about the geometry."
        )
    return {
        "label": "Internal consistency — no independent reference available",
        "verified": bool(measured) and not failed,
        "checks": checks,
        "statement": statement,
        "not_measured": [
            "absolute spatial accuracy (needs independent ground truth / checkpoints)",
        ],
    }


def validate_run(
    run_dir: Path,
    scene: str,
    reference_npz: Path | None,
    telemetry_csv: Path | None,
    dev_or_heldout: str,
    persist: bool = False,
) -> dict:
    """Full protocol for one run.

    Read-only by default. With ``persist=True`` the report is written through
    :func:`persist_report`, the artifact's single writer — callers never write
    the file themselves.
    """
    ws = Path(run_dir)
    # Same skeleton the no-measurement envelope uses, then filled with what is
    # actually measured — one shape, one owner (build_report).
    report: dict = build_report(ws.name, scene=scene, dataset_split=dev_or_heldout)

    # A. relative
    report["relative_consistency"] = relative_consistency(ws)

    # C. absolute — registration
    recon_centers = None
    telemetry_centers = None
    try:
        frames = json.loads((ws / "poses.json").read_text())["frames"]
        recon_centers = np.array([f["t"] for f in frames])
    except Exception:
        pass
    if telemetry_csv is not None and Path(telemetry_csv).is_file() and recon_centers is not None:
        import csv

        with open(telemetry_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
        by_id = {int(float(r["frame_id"])): r for r in rows}
        # frame_i ↔ telemetry row (via extractor stride recorded in quality_report)
        mapping = []
        qr = ws / "quality_report.json"
        stride = 1
        if qr.is_file():
            try:
                q = json.loads(qr.read_text())
                kept = [f for f in q.get("frames", []) if f.get("kept")]
                if kept:
                    stride = int(np.median(np.diff([f.get("frame_num", f["index"]) for f in kept])) or 1)
            except Exception:
                stride = 1
        for f in frames:
            try:
                i = int(str(f["frame_id"]).split("_")[-1].split(".")[0])
            except ValueError:
                continue
            row = by_id.get(i * stride) or by_id.get(i)
            if row is not None:
                mapping.append((np.array(f["t"], float),
                                np.array([float(row[k]) for k in ("x", "y", "z")])))
        if mapping:
            recon_centers_m = np.array([m[0] for m in mapping])
            telemetry_centers = np.array([m[1] for m in mapping])
            reg = registration_from_telemetry(recon_centers_m, telemetry_centers)
            reg["frame_correspondence"] = f"frame_i ↔ telemetry row i×{stride} (extractor stride)"
            report["registration"] = reg
            report["checkpoint_validation"] = checkpoint_metrics(recon_centers_m, telemetry_centers)
            report["scale_validation"] = scale_validation_from_trajectory(
                recon_centers_m, telemetry_centers
            )
    if "registration" not in report:
        report["registration"] = {
            "matched_frames": 0, "sufficient": False,
            "reason": "no telemetry/GPS correspondence available for this run",
        }

    # C. absolute — reference comparison
    ref_class = None
    if reference_npz is not None and Path(reference_npz).is_file() and recon_centers is not None:
        z = np.load(reference_npz)
        P = np.asarray(z["points"], float)
        normals = np.asarray(z["normals"], float) if "normals" in z.files else None
        ref_class = np.asarray(z["classification"]) if "classification" in z.files else None
        dense = dense_points(ws)
        if dense is not None:
            stats = cloud_to_reference(dense, P, normals)
            report["reference_comparison"] = stats
            report["error_decomposition"] = error_decomposition(dense, P, ref_class)
            report["reference_type"] = "independent LiDAR (held-out; never read by reconstruction)"
            report["reference_source"] = str(reference_npz)
        else:
            report["reference_comparison"] = {"available": False, "reason": "no dense artifact"}
            report["reference_type"] = None
    else:
        report["reference_comparison"] = None
        report["reference_type"] = None

    ref_available = bool(report.get("reference_comparison"))
    status, reason = certify(
        report.get("reference_comparison") or {},
        report.get("registration") or {},
        ref_available,
    )
    report["accuracy_certified"] = status.startswith("CERTIFIED")
    report["certification_status"] = status
    report["certification_reason"] = reason
    # The measured sections now always EXIST (None until measured), so read
    # them defensively — presence is no longer a signal.
    scale = report.get("scale_validation") or {}
    rel = report.get("relative_consistency") or {}
    report["accuracy_summary"] = {
        # Same vocabulary as the internal-consistency builder above: the
        # Reports page matches this token, and "available"/"unavailable" here
        # made a measured relative reconstruction render as "Unavailable".
        "relative_reconstruction": "measured" if rel.get("available") else "not measured",
        "metric_scale": (
            "validated (trajectory scale error %.2f%%)" % (100 * scale["scale_error"])
            if scale.get("scale_error") is not None
            else "not validated"
        ),
        "absolute": report["certification_status"],
    }
    facts = AccuracyFacts(
        scale_factor=float((report.get("scale_validation") or {}).get("scale_factor") or 1.0),
        has_reference=ref_available,
        reference_validated=bool(report.get("accuracy_certified")),
        has_georeferencing=bool((report.get("registration") or {}).get("sufficient")),
        has_relative_measurement=bool((report.get("relative_consistency") or {}).get("available")),
    )
    report["capability_state"], report["capability_reason"] = capability_state(facts)
    if persist:
        persist_report(ws, report)
    return report
