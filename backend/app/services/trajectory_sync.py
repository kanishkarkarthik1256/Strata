"""Single-pass trajectory bridge: telemetry CSV -> metric position prior.

Root-cause fix for the fragmented single-pass reconstruction: an uploaded
GPS telemetry CSV reached only the georef stage, so the sparse stage
reconstructed video-only and monocular drift collapsed the scale
(flight_to_tower_7511dc: visual trajectory 8.4 x 4.7 m vs true 142 x 41 m
— a ~17x monocular scale collapse).

Design rules (mandated):

* GPS = authoritative POSITION prior.  Course-over-ground = soft heading
  prior only when reliable.  Roll/pitch = UNKNOWN — synthetic zeros are
  never injected as pose observations; visual SfM estimates attitude.
* Timestamps flow through the real chain: selected frame -> source-frame
  index -> video PTS/timestamp (preserved by the extractor) -> nearest
  telemetry sample.  Timestamps are never derived from selected-frame
  indices alone.
* The measured visual->telemetry similarity SCALE is always reported
  first.  The historical [0.25, 4] band is a configurable unit/frame
  sanity WARNING, never a hard failure — legitimate monocular scale
  recovery (here ~17x) must not be rejected.
* Trajectory comparison is diagnostic-only: its similarity transform is
  used to report/placement-test, never to fabricate geometry.
* PASS is a composite: temporal sync AND shape AND continuity/pose-jumps
  AND path-length plausibility must agree — no single metric decides.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.logging_config import get_logger
from app.services.sparse_conditioning import (
    MIN_DEPTH_BASELINE_RATIO,
    point_min_baseline_m,
)

log = get_logger("drone_recon.services.trajectory_sync")

#: Robust jump rejection: fixes implying > (median + sigma * spread) speed
#: above a 5 m/s floor are rejected (GPS spikes), iterated to stability.
DEFAULT_JUMP_SIGMA = 3.0
#: Temporal-sync default: a frame may sit at most this far (s) from its
#: nearest telemetry sample to count as matched.
DEFAULT_MAX_DT_S = 0.75
#: Temporal-sync PASS: at least this fraction of frames matched.
DEFAULT_MATCH_FRACTION = 0.8
#: Common grid (s) for trajectory SHAPE comparison and continuity checks:
#: coarse enough to sit above both the visual frame cadence (~0.033 s)
#: and the GNSS fix cadence (~0.15 s), fine enough to resolve flight turns.
SHAPE_GRID_INTERVAL_S = 0.5
#: Direction comparison applies only to segments where BOTH trajectories
#: actually move at least this fast (m/s).  At hover/takeoff/landing the
#: telemetry step (~0.03-0.07 m per 0.5 s) is smaller than GNSS noise
#: (~1 m), so direction there measures jitter, not flight (measured on
#: 7511dc: all negative cosines sat at takeoff/landing; on moving
#: segments p10 cosine was 0.985).
SHAPE_MIN_SPEED_M_S = 1.0
#: Minimum fraction of grid segments that must be direction-eligible for
#: the direction criterion to be judgeable at all.
SHAPE_MIN_ELIGIBLE_FRAC = 0.4


# ---------------------------------------------------------------------------
# telemetry CSV -> metric ENU position prior
# ---------------------------------------------------------------------------

@dataclass
class PositionPrior:
    """Metric ENU trajectory from telemetry — a POSITION-only prior.

    ``positions`` is (N, 3) ENU (x=east, y=north, z=up, anchor at the first
    valid fix).  ``timestamps`` are the telemetry's own seconds (relative or
    absolute as the source provided — preserved, never rewritten).  No
    attitude is represented: unknown stays unknown.
    """

    positions: np.ndarray            # (N, 3) ENU metres
    timestamps: np.ndarray           # (N,) seconds, source units preserved
    anchor: tuple[float, float, float]  # lat0, lon0, alt0 (WGS84)
    rejected: list[dict] = field(default_factory=list)  # every rejected sample + reason

    @property
    def n(self) -> int:
        return int(len(self.positions))


def _load_gps_samples(
    csv_path: Path,
    *,
    video_fps: float | None = None,
    video_duration_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Schema-detected load -> (t, lat, lon, alt, provenance) for GPS rows.

    Time base, in priority order:
    1. the telemetry's own timestamp column (absolute or relative — used as
       recorded);
    2. ``frame_number`` at the known video fps (rows logged one-per-video-
       frame; the video's presentation timeline, NOT a fabricated clock —
       validated against the video duration before use);
    3. NOTHING.  When neither exists a row-index counter is NOT invented:
       a positional index silently maps telemetry row i to video moment i,
       pairing every frame with the wrong flight segment (measured on
       airport1: a 416 s flight rendered as "duration_s: 12472").  Without
       a real time base the trajectory is unusable for temporal sync and
       the caller must say so.
    """
    from app.services.telemetry import (
        TelemetryError,
        derive_timestamps_from_frame_numbers,
        load_telemetry_with_schema,
    )

    samples, _schema = load_telemetry_with_schema(csv_path)  # also writes schema artifacts
    provenance: dict = {"time_base": "telemetry_timestamp_column"}
    if all(s.timestamp_sec is None for s in samples):
        samples, note = derive_timestamps_from_frame_numbers(
            samples,
            video_fps if video_fps and video_fps > 0 else 0.0,
            video_duration_sec=video_duration_sec,
        )
        if not all(s.timestamp_sec is not None for s in samples):
            reason = note or (
                "telemetry has no timestamp column and no usable frame_number "
                "column at the known video fps"
            )
            raise TelemetryError(
                f"telemetry carries no usable time base: {reason}. "
                "Temporal sync and telemetry-assisted localization require the "
                "telemetry rows to be mappable to video moments — add a "
                "timestamp column or a per-video-frame frame_number column."
            )
        provenance = {"time_base": "frame_number_at_video_fps",
                      "rows_derived": len(samples), "video_fps": video_fps,
                      "note": note}
    rows = [s for s in samples if s.latitude is not None and s.longitude is not None]
    if len(rows) < 3:
        raise TelemetryError(
            f"telemetry CSV has {len(rows)} usable GPS rows (<3) — cannot build a trajectory"
        )
    t = np.array([float(s.timestamp_sec) for s in rows])
    lat = np.array([float(s.latitude) for s in rows])
    lon = np.array([float(s.longitude) for s in rows])
    alt = np.array([float(s.altitude_m if s.altitude_m is not None else 0.0) for s in rows])
    return t, lat, lon, alt, provenance


def _collapse_stale_fixes(
    t: np.ndarray, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Collapse stale repeated GPS rows to unique fixes.

    Consumer drone logs are logged at video rate (~30 Hz) but the GNSS
    receiver updates at a slower fix rate (~1-10 Hz); rows between fixes
    repeat the previous position verbatim (flight_to_tower_7511dc: 933
    rows, 203 unique positions, 78% co-located).  Speed statistics must
    be computed at FIX cadence, not row cadence, or every real motion
    step looks like a spike while 78% of samples read as perfectly
    stationary.

    A row is "stale" when its (lat, lon) equals the previous kept fix's
    (lat, lon).  Each co-located run collapses to its first row (the
    instant the fix was recorded); altitude moves with the position it
    belongs to.  Never drops genuine motion.
    """
    if len(lat) == 0:
        return t, lat, lon, alt, {"rows_in": 0, "fixes_out": 0, "stale_rows": 0}
    keep = np.ones(len(lat), dtype=bool)
    keep[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1])
    info = {
        "rows_in": int(len(lat)),
        "fixes_out": int(keep.sum()),
        "stale_rows": int((~keep).sum()),
        "fix_rate_hz": round(float(keep.sum() / max(t[-1] - t[0], 1e-9)), 2),
        "log_rate_hz": round(float(len(t) / max(t[-1] - t[0], 1e-9)), 2),
    }
    return t[keep], lat[keep], lon[keep], alt[keep], info


def _reject_jumps(
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, t: np.ndarray,
    sigma: float = DEFAULT_JUMP_SIGMA,
) -> tuple[np.ndarray, list[dict]]:
    """Boolean keep-mask + per-fix rejection reasons (call on unique fixes).

    A fix is rejected when the step it implies is faster than a robust
    speed band (median + sigma x MAD, floored at 5 m/s above median) —
    GPS single-sample spikes die, real accelerations survive.  Iterated
    until stable (<=5 passes).  Every rejection is reported with reason.
    """
    from app.services.telemetry_schema import _haversine_m

    keep = np.ones(len(lat), dtype=bool)
    reasons: dict[int, str] = {}
    for _ in range(5):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        d_m = np.array([
            _haversine_m(lat[a], lon[a], lat[b], lon[b])
            for a, b in zip(idx[:-1], idx[1:])
        ])
        dt = np.diff(t[idx])
        dt = np.where(dt > 1e-6, dt, np.nan)
        speed = d_m / dt
        med = float(np.nanmedian(speed))
        mad = float(np.nanmedian(np.abs(speed - med))) or 1e-6
        limit = med + sigma * max(5.0 * mad, 5.0)
        bad = speed > limit
        if not bad.any():
            break
        new_keep = keep.copy()
        drop = idx[1:][bad]  # later fix of a jump pair is the spike
        for j, sp in zip(drop, speed[bad]):
            if j not in reasons:
                reasons[int(j)] = (
                    f"implied speed {sp:.1f} m/s > robust limit {limit:.1f} m/s (GPS spike)"
                )
        new_keep[drop] = False
        if int(new_keep.sum()) == int(keep.sum()):
            break
        keep = new_keep
    ordered = [{"index": i, "reason": r} for i, r in sorted(reasons.items())]
    return keep, ordered


def telemetry_position_prior(
    telemetry_csv: Path,
    *,
    jump_sigma: float = DEFAULT_JUMP_SIGMA,
    video_fps: float | None = None,
    video_duration_sec: float | None = None,
) -> tuple[PositionPrior, dict]:
    """Build the metric ENU position prior from a telemetry CSV.

    Returns (PositionPrior, report_dict).  Raises TelemetryError when the
    file cannot yield a trajectory — the caller keeps video-only behaviour
    and reports the reason honestly.  ``video_fps``/``video_duration_sec``
    enable the frame_number→time derivation when the file has no timestamp
    column; the provenance of the time base is always reported.
    """
    from app.services.georeferencing import wgs84_to_enu
    from app.services.telemetry import TelemetryError

    t, lat, lon, alt, time_prov = _load_gps_samples(
        telemetry_csv,
        video_fps=video_fps,
        video_duration_sec=video_duration_sec,
    )
    t, lat, lon, alt, fix_info = _collapse_stale_fixes(t, lat, lon, alt)
    if fix_info["fixes_out"] < 3:
        raise TelemetryError(
            f"telemetry yields only {fix_info['fixes_out']} unique GPS fixes (<3) — "
            "trajectory unusable"
        )
    keep, rejected = _reject_jumps(lat, lon, alt, t, sigma=jump_sigma)
    if int(keep.sum()) < 3:
        raise TelemetryError(
            f"only {int(keep.sum())} GPS fixes survive jump rejection — trajectory unusable"
        )
    t, lat, lon, alt = t[keep], lat[keep], lon[keep], alt[keep]

    lat0, lon0, alt0 = float(lat[0]), float(lon[0]), float(alt[0])
    enu = wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0)
    prior = PositionPrior(
        positions=np.asarray(enu, dtype=np.float64),
        timestamps=t,
        anchor=(lat0, lon0, alt0),
        rejected=rejected,
    )
    step = np.linalg.norm(np.diff(enu, axis=0), axis=1)
    dt = np.diff(t)
    speeds = step / np.where(dt > 1e-6, dt, np.nan)
    report = {
        "rows_in": fix_info["rows_in"],
        "unique_gps_fixes": fix_info["fixes_out"],
        "stale_rows_collapsed": fix_info["stale_rows"],
        **time_prov,
        "gps_fix_rate_hz": fix_info["fix_rate_hz"],
        "log_rate_hz": fix_info["log_rate_hz"],
        "samples_in": int(len(keep) + len(rejected)),
        "samples_used": int(keep.sum()),
        "samples_rejected": len(rejected),
        "enu_anchor": {"lat": lat0, "lon": lon0, "alt": alt0},
        "span_m": [round(float(v), 2) for v in (enu.max(0) - enu.min(0))],
        "duration_s": round(float(t[-1] - t[0]), 3),
        "path_length_m": round(float(step.sum()), 2),
        "max_speed_m_s": round(float(np.nanmax(speeds)), 2),
        "rejected_samples": rejected,
        "attitude": "position-only prior (roll/pitch unknown; heading soft, unused here)",
    }
    log.info(
        "telemetry_position_prior_built",
        samples_used=report["samples_used"], rejected=len(rejected),
        span_m=report["span_m"], path_length_m=report["path_length_m"],
    )
    return prior, report


# ---------------------------------------------------------------------------
# temporal synchronization gate (PTS-based)
# ---------------------------------------------------------------------------

def temporal_sync_report(
    quality_report: Path,
    telemetry_csv: Path,
    video_fps: float,
    *,
    video_duration_sec: float | None = None,
    max_dt_s: float = DEFAULT_MAX_DT_S,
    match_fraction: float = DEFAULT_MATCH_FRACTION,
) -> dict:
    """Prove selected frames pair with the telemetry of their own moment.

    Chain: kept frame -> source frame index -> video PTS/timestamp (from
    the extractor's quality report — constant-FPS was verified from the
    timestamp diffs themselves, never assumed) -> nearest telemetry
    sample -> its position.  Telemetry rows logged one-per-video-frame
    without a timestamp column get their time base from frame_number at
    the video fps (validated against the video duration) — the same
    derivation ``_load_gps_samples`` uses, so both paths share one clock.
    """
    qr = json.loads(quality_report.read_text())
    kept = [f for f in qr.get("frames", []) if f.get("kept")]
    if not kept:
        return {"pass": False, "error": "no kept frames in quality_report"}

    from app.services.telemetry import (
        derive_timestamps_from_frame_numbers,
        load_telemetry_with_schema,
    )

    samples, _ = load_telemetry_with_schema(telemetry_csv)
    derivation_note = None
    if all(s.timestamp_sec is None for s in samples):
        samples, derivation_note = derive_timestamps_from_frame_numbers(
            samples, video_fps, video_duration_sec=video_duration_sec
        )
        if all(s.timestamp_sec is None for s in samples):
            return {
                "pass": False,
                "error": "telemetry has no usable timestamps and frame numbers "
                f"cannot provide one ({derivation_note or 'no frame_number column'})",
            }
    tel_t = np.array([s.timestamp_sec for s in samples if s.timestamp_sec is not None])
    if len(tel_t) == 0:
        return {"pass": False, "error": "telemetry has no usable timestamps"}

    v_first = float(kept[0].get("timestamp_sec", 0.0))
    v_last = float(kept[-1].get("timestamp_sec", 0.0))
    j_first = int(np.clip(np.searchsorted(tel_t, v_first), 0, len(tel_t) - 1))
    j_last = int(np.clip(np.searchsorted(tel_t, v_last), 0, len(tel_t) - 1))

    matched = 0
    dts: list[float] = []
    unmatched: list[int] = []
    for f in kept:
        vt = f.get("timestamp_sec")
        if vt is None:
            unmatched.append(int(f.get("index", -1)))
            continue
        j = int(np.clip(np.searchsorted(tel_t, float(vt)), 0, len(tel_t) - 1))
        dt = abs(float(tel_t[j]) - float(vt))
        if dt <= max_dt_s:
            matched += 1
            dts.append(dt)
        else:
            unmatched.append(int(f.get("index", -1)))
    dts_arr = np.array(dts) if dts else np.array([np.nan])
    match_pct = round(100.0 * matched / max(1, len(kept)), 1)
    passed = bool(matched >= max(2, int(match_fraction * len(kept))))
    return {
        "video_first_ts": round(v_first, 3),
        "video_last_ts": round(v_last, 3),
        "telemetry_first_ts": round(float(tel_t[j_first]), 3),
        "telemetry_last_ts": round(float(tel_t[j_last]), 3),
        "estimated_global_offset_s": round(float(tel_t[j_first]) - v_first, 3),
        "video_fps": round(float(video_fps), 3),
        "median_frame_to_telemetry_dt_s": round(float(np.nanmedian(dts_arr)), 4),
        "p95_frame_to_telemetry_dt_s": round(float(np.nanpercentile(dts_arr, 95)), 4),
        "max_frame_to_telemetry_dt_s": round(float(np.nanmax(dts_arr)), 4),
        "matched_frames": matched,
        "unmatched_frames": len(kept) - matched,
        "unmatched_indices": unmatched[:20],
        "match_percent": match_pct,
        "max_dt_threshold_s": max_dt_s,
        "match_fraction_required": match_fraction,
        **({"timestamp_provenance": derivation_note} if derivation_note else {}),
        "pass": passed,
    }


# ---------------------------------------------------------------------------
# cadence normalization — compare like-for-like signals
# ---------------------------------------------------------------------------

def resample_path(
    positions: np.ndarray,
    timestamps: np.ndarray,
    interval_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-resample a trajectory onto a uniform time grid.

    Purpose: the visual trajectory steps at frame cadence (~0.033 s, with
    cm-scale SfM jitter) while telemetry fixes update at GNSS cadence
    (~0.15 s, quantized).  Per-frame velocity direction and jump detection
    on RAW steps measure jitter-vs-quantization, not trajectory shape.
    Resampling BOTH signals onto one common grid (default 0.5 s — well
    above both cadences, far below flight duration) makes the comparison
    like-for-like.  Linear interpolation only: never smoothing that moves
    geometry, never fabrication beyond the sampled span.
    """
    P = np.asarray(positions, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    if len(P) < 2 or interval_s <= 0:
        return P, t
    t0, t1 = float(t[0]), float(t[-1])
    if not np.isfinite(t1 - t0) or t1 - t0 < interval_s:
        return P, t
    grid = np.arange(t0, t1 + 1e-9, interval_s)
    out = np.column_stack([np.interp(grid, t, P[:, k]) for k in range(P.shape[1])])
    return out, grid


# ---------------------------------------------------------------------------
# diagnostic similarity alignment + shape comparison
# ---------------------------------------------------------------------------

def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (scale, R, t) with src @ (s R).T + t ~= dst.

    Diagnostic use only — never applied to reconstruct geometry.
    """
    return _umeyama_fit(np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64))


def _umeyama_fit(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    # UNNORMALIZED cross-covariance: normalizing by n would divide the
    # fitted scale by n as well (verified bug: s_fit = s_true/n).
    H = Sc.T @ Dc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    denom = float((Sc ** 2).sum())
    scale = float(S.sum() / denom) if denom > 1e-12 else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def compare_trajectories(
    visual_centers: np.ndarray,
    telemetry: PositionPrior,
    *,
    visual_timestamps: np.ndarray | None = None,
    scale_warn_min: float | None = None,
    scale_warn_max: float | None = None,
) -> dict:
    """Shape-and-motion comparison, normalized by the telemetry trajectory.

    The similarity fit is diagnostic-only; its scale is REPORTED first and
    checked against a configurable sanity band that can only ever WARN
    (unit/frame-convention mixups) — never reject legitimate monocular
    scale recovery. The band defaults come from settings.telemetry
    (TELEMETRY_SCALE_WARN_MIN/MAX) — a dataset/config sanity gate, not a
    hard-coded truth.
    """
    if scale_warn_min is None or scale_warn_max is None:
        from app.config.settings import settings as _settings

        scale_warn_min = (
            scale_warn_min
            if scale_warn_min is not None
            else _settings.telemetry.scale_warn_min
        )
        scale_warn_max = (
            scale_warn_max
            if scale_warn_max is not None
            else _settings.telemetry.scale_warn_max
        )
    V = np.asarray(visual_centers, dtype=np.float64)
    T = np.asarray(telemetry.positions, dtype=np.float64)
    if visual_timestamps is not None:
        # Correspondence mode: telemetry resampled AT the cameras' own
        # timestamps (temporal-sync proof carried into the comparison).
        vt = np.asarray(visual_timestamps, dtype=np.float64)
        Tq = np.column_stack([
            np.interp(vt, telemetry.timestamps, T[:, k]) for k in range(3)
        ])
        Vn = V
        n = len(Vn)
    else:
        n = min(len(V), len(T))
        if n < 4:
            return {"pass": False, "error": f"fewer than 4 correspondences ({n})"}
        Vn = V[:n]
        # Resample telemetry onto the visual index axis (uniform in time).
        tv = telemetry.timestamps
        tt = np.linspace(tv[0], tv[-1], n)
        Tq = np.column_stack([
            np.interp(tt, tv, T[:, k]) for k in range(3)
        ])
    if len(Vn) < 4:
        return {"pass": False, "error": f"fewer than 4 correspondences ({len(Vn)})"}
    n = len(Vn)

    scale, R, tr = umeyama_similarity(Vn, Tq)
    aligned = scale * (Vn @ R.T) + tr
    resid = np.linalg.norm(aligned - Tq, axis=1)

    # Cadence normalization: velocity direction / turns / path length are
    # computed on BOTH paths resampled onto one common grid (0.5 s), because
    # per-frame visual steps (cm-scale jitter at ~30 Hz) against quantized
    # GNSS fixes (~0.15 s cadence) measure sampling noise, not shape.
    # NOTE: the path-length ratio is a GATE criterion, so it is measured
    # over the SAME temporal coverage on both sides — the intersection of
    # the video and telemetry spans.  Interpolating either side beyond its
    # own coverage would compare measured geometry against extrapolation.
    if visual_timestamps is not None:
        vt = np.asarray(visual_timestamps, dtype=np.float64)
        grid = np.arange(
            max(vt[0], telemetry.timestamps[0]),
            min(vt[-1], telemetry.timestamps[-1]) + 1e-9,
            SHAPE_GRID_INTERVAL_S,
        )
        Vs = np.column_stack([np.interp(grid, vt, Vn[:, k]) for k in range(3)])
        Ts = np.column_stack([
            np.interp(grid, telemetry.timestamps, T[:, k]) for k in range(3)
        ])
    else:
        Vs, grid = resample_path(Vn, np.arange(len(Vn), dtype=float), SHAPE_GRID_INTERVAL_S)
        Ts, _ = resample_path(Tq, np.arange(len(Tq), dtype=float), SHAPE_GRID_INTERVAL_S)
    if len(Vs) < 4:
        return {"pass": False, "error": "common grid too short for shape comparison"}

    tel_len = float(np.linalg.norm(np.diff(Ts, axis=0), axis=1).sum())
    vis_len = float(np.linalg.norm(np.diff(Vs, axis=0), axis=1).sum())
    tel_span = float(np.linalg.norm(Tq.max(0) - Tq.min(0)))

    dv, dt_ = np.diff(Vs, axis=0), np.diff(Ts, axis=0)
    nv, nt = np.linalg.norm(dv, axis=1), np.linalg.norm(dt_, axis=1)
    ok = (nv > 1e-9) & (nt > 1e-9)
    cos = np.full(len(Vs) - 1, np.nan)
    cos[ok] = (dv[ok] * dt_[ok]).sum(1) / (nv[ok] * nt[ok])
    # Direction eligibility: compare travel direction only where BOTH
    # signals actually move (>= SHAPE_MIN_SPEED_M_S).  Near hover the
    # telemetry step is below GNSS noise and direction is meaningless
    # jitter — excluded from the GATE, reported for transparency.
    interval = float(np.median(np.diff(grid))) if len(grid) > 1 else SHAPE_GRID_INTERVAL_S
    speed_v, speed_t = nv / max(interval, 1e-6), nt / max(interval, 1e-6)
    eligible = ok & (speed_v > SHAPE_MIN_SPEED_M_S) & (speed_t > SHAPE_MIN_SPEED_M_S)
    n_elig = int(eligible.sum())
    eligible_frac = n_elig / max(1, len(cos))
    cos_e = cos[eligible]

    vz, tz = Vs[:, 2], Ts[:, 2]
    # Level flight is the common case, and a constant altitude series has zero
    # variance, so the correlation is undefined (0/0 -> nan). Report that
    # honestly as "not measurable" instead of shipping nan into the shape
    # score: two flat profiles agree perfectly in shape, but ONE flat profile
    # is a real disagreement and must score as such.
    vz_c, tz_c = vz - vz.mean(), tz - tz.mean()
    vz_flat, tz_flat = float(np.ptp(vz_c)) < 1e-9, float(np.ptp(tz_c)) < 1e-9
    if vz_flat and tz_flat:
        alt_corr = 1.0
        alt_corr_measurable = True
    elif vz_flat or tz_flat:
        alt_corr = 0.0
        alt_corr_measurable = True
    else:
        alt_corr = float(np.corrcoef(vz_c, tz_c)[0, 1])
        alt_corr_measurable = bool(np.isfinite(alt_corr))

    def _turns(P: np.ndarray, deg: float = 30.0) -> int:
        if len(P) < 3:
            return 0
        d1, d2 = P[1:-1] - P[:-2], P[2:] - P[1:-1]
        a, b = np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1)
        good = (a > 1e-9) & (b > 1e-9)
        ang = np.zeros(len(d1))
        ang[good] = np.degrees(np.arccos(np.clip(
            (d1[good] * d2[good]).sum(1) / (a[good] * b[good]), -1, 1)))
        return int((ang > deg).sum())

    end_v = float(np.linalg.norm(Vn[-1] - Vn[0]))
    end_t = float(np.linalg.norm(Tq[-1] - Tq[0]))
    len_ratio = vis_len / max(tel_len, 1e-9)

    scale_note = (
        "OK" if scale_warn_min <= scale <= scale_warn_max
        else "UNIT/FRAME-CONVENTION WARNING (diagnostic only — legitimate "
             "monocular scale recovery may legitimately exceed this band)"
    )
    result = {
        "n_correspondences": n,
        "shape_grid_interval_s": SHAPE_GRID_INTERVAL_S,
        "shape_grid_points": int(len(Vs)),
        "measured_similarity_scale_visual_per_telemetry": round(scale, 4),
        "scale_sanity": {
            "warn_band": [scale_warn_min, scale_warn_max],
            "verdict": scale_note,
            "hard_failure": False,  # scale can never fail the gate by itself
        },
        "position_residual_normalized_by_telemetry_length": {
            "median": round(float(np.median(resid)) / max(tel_len, 1e-9), 5),
            "p95": round(float(np.percentile(resid, 95)) / max(tel_len, 1e-9), 5),
            "max": round(float(resid.max()) / max(tel_len, 1e-9), 5),
        },
        "position_residual_m_raw": {
            "median": round(float(np.median(resid)), 3),
            "p95": round(float(np.percentile(resid, 95)), 3),
            "max": round(float(resid.max()), 3),
        },
        "velocity_direction": {
            "median_cosine": round(float(np.nanmedian(cos_e)) if n_elig else float("nan"), 4),
            "p10_cosine": round(float(np.nanpercentile(cos_e, 10)) if n_elig else float("nan"), 4),
            "eligible_segments": n_elig,
            "eligible_fraction": round(eligible_frac, 3),
            "min_speed_threshold_m_s": SHAPE_MIN_SPEED_M_S,
            "median_cosine_all_segments": round(float(np.nanmedian(cos)), 4),
            "p10_cosine_all_segments": round(float(np.nanpercentile(cos, 10)), 4),
        },
        "altitude_trend_correlation": round(alt_corr, 4),
        "altitude_trend_measurable": alt_corr_measurable,
        "cumulative_path_length": {
            "visual_m": round(vis_len, 2),
            "telemetry_m": round(tel_len, 2),
            "ratio": round(len_ratio, 4),
        },
        "start_end_displacement": {
            "visual_m": round(end_v, 2),
            "telemetry_m": round(end_t, 2),
            "ratio": round(end_v / max(end_t, 1e-9), 4),
        },
        "turn_events_gt_30deg": {"visual": _turns(Vn), "telemetry": _turns(Tq)},
        "telemetry_scene_span_m": round(tel_span, 2),
        "similarity_transform": {
            "scale": round(scale, 6),
            "rotation_3x3": np.round(R, 6).tolist(),
            "translation_3x1": np.round(tr, 4).tolist(),
        },
    }

    # Composite shape verdict: same structure + plausible transform.
    # Direction is judged on ELIGIBLE (moving) segments only; if the flight
    # barely moves (hover-dominated), direction is reported NOT JUDGEABLE
    # and the gate fails honestly rather than passing on jitter.
    direction_ok = (
        eligible_frac >= SHAPE_MIN_ELIGIBLE_FRAC
        and n_elig >= 4
        and float(np.nanmedian(cos_e)) > 0.8
        and float(np.nanpercentile(cos_e, 10)) > 0.0
    )
    result["direction_judgeable"] = bool(eligible_frac >= SHAPE_MIN_ELIGIBLE_FRAC and n_elig >= 4)
    result["pass"] = bool(
        direction_ok                       # travelling the same way (when moving)
        and 0.5 <= len_ratio <= 2.0        # path-length plausibility
    )
    return result


# ---------------------------------------------------------------------------
# composite gate — no single metric decides
# ---------------------------------------------------------------------------

def composite_trajectory_gate(
    sync_report: dict,
    shape_report: dict,
    continuity_report: dict,
    *,
    match_fraction: float = DEFAULT_MATCH_FRACTION,
) -> dict:
    """Composite PASS: temporal sync AND shape AND continuity agree.

    Each input is required (a missing report fails the gate); the verdict
    is the conjunction of the three sub-verdicts.  Scale NEVER decides:
    out-of-band scale is a diagnostic warning only (a legitimate ~17x
    monocular scale recovery must not fail the run).
    """
    subs = {
        "temporal_sync": bool(sync_report.get("pass", False)),
        "trajectory_shape": bool(shape_report.get("pass", False)),
        "pose_continuity": bool(continuity_report.get("pass", False)),
    }
    return {
        "pass": all(subs.values()),
        "components": subs,
        "scale_verdict": shape_report.get("scale_sanity", {}).get("verdict", "unavailable"),
        "rule": (
            "PASS requires temporal sync AND trajectory shape AND pose "
            "continuity; similarity scale is reported and can only warn"
        ),
    }


# ---------------------------------------------------------------------------
# piecewise local correction (drift bends) -> soft metric priors for BA
# ---------------------------------------------------------------------------

@dataclass
class _PiecewiseCorrection:
    """Per-camera rigid correction (R_i, s_i, t_i) mapping the visual model
    into the telemetry ENU frame piecewise along the trajectory."""

    camera_transform: dict[str, tuple[float, np.ndarray, np.ndarray]]
    segments: list[dict] = field(default_factory=list)
    global_transform: tuple[float, np.ndarray, np.ndarray] | None = None


def _piecewise_rigid_correction(
    centers: dict[str, np.ndarray],
    pairs: list[tuple[str, int, float]],
    telemetry: "PositionPrior",
    camera_ts: dict[str, float],
    *,
    window: int = 8,
    overlap: int = 4,
    min_inliers: int = 4,
) -> tuple[_PiecewiseCorrection | None, dict]:
    """Fit overlapping-window rigid+scale corrections along the flight.

    Motivation (measured, run 7511dc rerun5): the video-only SfM trajectory
    carries DRIFT BENDS — contiguous bands (frames 32–54, 68–90) displaced
    up to 85 m from the telemetry line, 27 visual turns vs 14 real.  A
    single global similarity cannot represent that warping (the robust fit
    rejects 35/55 cameras), and pulling such a state into place with soft
    priors alone stretches the model (post-BA path length 5.3x telemetry).

    The correction is piecewise: cameras in each sliding time window get
    their own rigid+uniform-scale similarity into telemetry ENU, fitted on
    that window's time-matched pairs only.  Each camera takes the transform
    of the LAST window (in time) that covers it — no pose interpolation, no
    deformation within a window.  Windows with too few inliers after robust
    fitting are reported unresolved and left to BA's soft priors rather
    than given a fabricated transform.

    Scale safety: window size >= 8 pairs spans >= ~0.27 s of flight (~4 m
    of real motion at the measured 15 m/s) — enough geometry to pin scale
    and rotation.  Window scale ratios are reported; a spread far beyond
    the global scale indicates the window is too small to pin scale, so
    window fits whose scale deviates >30% from the global robust scale are
    discarded (their cameras fall back to the global transform).
    """
    report: dict = {"window": window, "overlap": overlap, "segments": []}
    if len(pairs) < max(4, window // 2):
        report["error"] = f"only {len(pairs)} time-matched pairs — piecewise correction skipped"
        return None, report

    # time-order the matched pairs by the camera's own video timestamp
    ordered = sorted(pairs, key=lambda p: camera_ts.get(Path(p[0]).stem, 0.0))
    n = len(ordered)
    step = max(1, window - overlap)

    # Global robust scale as the sanity anchor for window fits.
    V_all = np.array([centers[name] for name, _j, _dt in ordered])
    T_all = np.array([telemetry.positions[j] for _name, j, _dt in ordered])
    try:
        s_glob, R_glob, t_glob, _info = umeyama_similarity_robust(V_all, T_all)
    except ValueError:
        s_glob, R_glob, t_glob = umeyama_similarity(V_all, T_all)

    camera_transform: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    segments: list[dict] = []
    # PASS 1: fit every window robustly and record its scale.
    fits: list[tuple[dict, float, np.ndarray, np.ndarray, dict]] = []
    for start in range(0, n, step):
        chunk = ordered[start:start + window]
        if len(chunk) < min_inliers:
            continue
        V = np.array([centers[name] for name, _j, _dt in chunk])
        T = np.array([telemetry.positions[j] for _name, j, _dt in chunk])
        if float(np.linalg.norm(V.max(0) - V.min(0))) < 1e-6:
            continue
        try:
            s, R, t, info = umeyama_similarity_robust(V, T)
        except ValueError:
            continue
        fits.append((chunk, s, R, t, info))
    # PASS 2: accept windows whose scale agrees with the MEDIAN window
    # scale (NOT the global fit — the global estimate is itself corrupted
    # by the drift bends; measured: global 6.4 vs consistent local ~10,
    # which made the old global anchor reject every correct window).
    scales = np.array([f[1] for f in fits]) if fits else np.array([])
    s_med = float(np.median(scales)) if len(scales) else s_glob
    for chunk, s, R, t, info in fits:
        scale_ok = 0.7 * s_med <= s <= 1.3 * s_med
        seg = {
            "cameras": [Path(name).stem for name, _j, _dt in chunk],
            "inliers": info["inliers"],
            "outliers": info["outliers"],
            "scale": round(s, 4),
            "scale_vs_median": round(s / max(s_med, 1e-9), 4),
            "inlier_residual_m": info["inlier_residual_m"],
            "accepted": bool(scale_ok and info["inliers"] >= min_inliers),
        }
        if seg["accepted"]:
            for name, _j, _dt in chunk:
                # LAST window covering a camera wins (iterate forward).
                camera_transform[name] = (s, R, t)
        segments.append(seg)

    report["segments"] = segments
    report["n_windows"] = len(segments)
    report["accepted_windows"] = sum(1 for s in segments if s["accepted"])
    report["median_window_scale"] = round(s_med, 4)
    report["global_scale"] = round(s_glob, 4)
    report["cameras_with_window_transform"] = len(camera_transform)
    report["cameras_falling_back_to_global"] = len(ordered) - len(camera_transform)
    report["usage"] = (
        "piecewise rigid+uniform-scale corrections into telemetry ENU; "
        "per-camera transform from the last covering window; no pose "
        "interpolation; unresolved cameras keep the global transform"
    )
    if not camera_transform:
        report["error"] = "no window produced an accepted fit"
        return None, report
    return _PiecewiseCorrection(
        camera_transform=camera_transform,
        segments=segments,
        global_transform=(s_glob, R_glob, t_glob),
    ), report

def camera_timestamps_from_quality_report(
    quality_report: Path,
) -> dict[str, float]:
    """Camera STEM -> video timestamp (s), keyed by filename stem.

    Built from the extractor's quality report: kept frames carry both the
    filename and the preserved PTS ``timestamp_sec``.  Keys are stems
    (``frame_000012``), NOT full filenames: the COLMAP camera names come
    from ``colmap_images/`` copies whose extension may differ from the
    selected-frame files (frame_000012.jpg vs frame_000012.png — the exact
    mismatch that produced 0 correspondences on flight_to_tower_7511dc).
    This is the ONLY sanctioned camera<->time mapping — never positional
    frame indices.
    """
    qr = json.loads(quality_report.read_text())
    out: dict[str, float] = {}
    for f in qr.get("frames", []):
        if not f.get("kept"):
            continue
        ts = f.get("timestamp_sec")
        name = f.get("filename")
        if ts is None or not name:
            continue
        out[Path(name).stem] = float(ts)
    return out


def umeyama_similarity_robust(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    iters: int = 10,
    keep_frac: float = 0.6,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Outlier-rejecting similarity fit (trimmed LS + MAD classification).

    A drifted/locally-misregistered camera segment must not contaminate the
    global placement transform.  Two phases:

    1. Trimmed-LS initialization: repeatedly keep the ``keep_frac`` best
       correspondences by residual, refit — converges to the majority
       geometry even when ~40% of pairs disagree.
    2. Final classification on the CONVERGED fit: 3xMAD residual band
       names the outliers; the reported fit is the inlier refit.

    NOT an arbitrary rejection of bad news: rejected correspondences are
    REPORTED per camera, and inlier residuals stay in the report.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 4:
        raise ValueError(f"robust fit needs >= 4 correspondences, got {n}")

    # Phase 1: multi-start trimmed refits.  Trimmed LS has local minima
    # (a contaminated start can keep drifting points while shedding true
    # inliers), so several deterministic random starts race; the fit with
    # the lowest kept-median residual wins.
    keep_n = max(4, int(keep_frac * n))
    rng = np.random.default_rng(12345)  # deterministic: reproducible runs
    starts = [np.arange(n)]
    base = max(4, keep_n // 2)
    for _ in range(8):
        starts.append(np.sort(rng.choice(n, size=base, replace=False)))

    best: tuple[float, np.ndarray, float, np.ndarray, np.ndarray] | None = None
    for start in starts:
        idx = np.sort(np.asarray(start))
        scale, R, t = _umeyama_fit(src[idx], dst[idx])
        for _ in range(iters):
            resid = np.linalg.norm(scale * (src @ R.T) + t - dst, axis=1)
            order = np.argsort(resid)
            sel = np.sort(order[:keep_n])
            if np.array_equal(sel, idx):
                break
            idx = sel
            scale, R, t = _umeyama_fit(src[idx], dst[idx])
        resid = np.linalg.norm(scale * (src @ R.T) + t - dst, axis=1)
        score = float(np.median(resid[idx]))
        if best is None or score < best[0]:
            best = (score, idx, scale, R, t)
    assert best is not None
    _score, idx, scale, R, t = best

    # Phase 2: 3xMAD classification on the converged fit.
    resid = np.linalg.norm(scale * (src @ R.T) + t - dst, axis=1)
    for _ in range(4):
        med = float(np.median(resid[idx]))
        mad = float(np.median(np.abs(resid[idx] - med))) or 1e-6
        limit = med + 3.0 * max(1.4826 * mad, 1e-3)
        new_idx = np.flatnonzero(resid <= limit)
        if len(new_idx) < 4:
            break
        if np.array_equal(new_idx, idx):
            break
        idx = new_idx
        scale, R, t = _umeyama_fit(src[idx], dst[idx])
        resid = np.linalg.norm(scale * (src @ R.T) + t - dst, axis=1)

    out_idx = np.flatnonzero(~np.isin(np.arange(n), idx))
    info = {
        "inliers": int(len(idx)),
        "outliers": int(len(out_idx)),
        "outlier_indices": [int(i) for i in out_idx],
        "rejected_correspondences": [
            {"index": int(i), "residual_m": round(float(resid[i]), 3)}
            for i in out_idx
        ],
        "inlier_residual_m": {
            "median": round(float(np.median(resid[idx])), 3),
            "p95": round(float(np.percentile(resid[idx], 95)), 3),
            "max": round(float(resid[idx].max()), 3),
        },
    }
    return scale, R, t, info


def correspondences_by_time(
    visual_centers: dict[str, np.ndarray],
    visual_ts: dict[str, float],
    telemetry: PositionPrior,
    *,
    max_dt_s: float = DEFAULT_MAX_DT_S,
) -> tuple[list[tuple[str, int, float]], dict]:
    """(camera_name, telemetry_fix_idx, |dt|) triples whose video timestamp
    sits within ``max_dt_s`` of a unique GPS fix — the temporal-sync proof
    made concrete: frame N pairs with the telemetry of ITS OWN moment.
    """
    pairs: list[tuple[str, int, float]] = []
    dts: list[float] = []
    centers_by_stem = {Path(name).stem: name for name in visual_centers}
    for name, ts in visual_ts.items():
        # visual_ts keys are stems (quality report); visual_centers keys are
        # COLMAP camera names (possibly different extension) — match by stem.
        if name not in centers_by_stem:
            continue
        j = int(np.clip(np.searchsorted(telemetry.timestamps, ts), 0, telemetry.n - 1))
        dt = abs(float(telemetry.timestamps[j]) - ts)
        if dt <= max_dt_s:
            pairs.append((centers_by_stem[name], j, dt))
            dts.append(dt)
    dts_arr = np.asarray(dts, dtype=np.float64) if dts else np.array([np.nan])
    report = {
        "cameras_total": len(visual_ts),
        "cameras_matched": len(pairs),
        "match_percent": round(100.0 * len(pairs) / max(1, len(visual_ts)), 1),
        "median_dt_s": round(float(np.nanmedian(dts_arr)), 4),
        "max_dt_s": round(float(np.nanmax(dts_arr)), 4),
    }
    return pairs, report


def apply_similarity_to_reconstruction(result, scale: float, R: np.ndarray, t: np.ndarray) -> None:
    """Rigid+uniform-scale similarity placement of a reconstruction, in place.

    X' = s R X + t for camera centres AND 3D points; rotations compose with
    R (world-to-camera: R'_w2c = R_w2c R^T, t'_w2c = s t_w2c - R_w2c R^T t),
    which scales camera-space depth by exactly s and leaves every image
    projection invariant — so reprojection errors are untouched by the
    placement.  Uniform scale only: never per-axis, never a deformation.
    """
    R = np.asarray(R, dtype=np.float64)
    for cam in result.cameras.values():
        C = np.asarray(cam.position, dtype=np.float64)
        R_c2w = np.asarray(cam.rotation, dtype=np.float64)
        # STRATA canonical convention (depth_generator/geometry): R is
        # CAMERA-TO-WORLD, t IS the camera centre:  X_w = R_c2w X_cam + C.
        # Under placement X' = s R X + t:
        #   C' = s R C + t
        #   R'_c2w = R R_c2w   (so X_cam' = R'_c2w^T (X' - C') = s X_cam —
        #   depths scale by s, every projection is invariant)
        C_new = scale * (R @ C) + t
        R_c2w_new = R @ R_c2w
        cam.position = C_new
        cam.rotation = R_c2w_new
        if hasattr(cam, "quaternion"):
            from scipy.spatial.transform import Rotation as _Rot
            cam.quaternion = _Rot.from_matrix(R_c2w_new).as_quat()[[3, 0, 1, 2]]  # wxyz
        if hasattr(cam, "translation"):
            cam.translation = C_new
    for pt in result.points3d:
        X = np.asarray(pt.position, dtype=np.float64)
        pt.position = scale * (R @ X) + t


def apply_piecewise_correction(
    result,
    correction: _PiecewiseCorrection,
    camera_ts: dict[str, float] | None = None,
) -> dict:
    """Apply per-camera rigid+scale corrections to the reconstruction, in place.

    Cameras: X' = s_i R_i X + t_i from the camera's OWN window transform.
    Cameras NOT covered by an accepted window take the transform of the
    NEAREST accepted window in time — the global robust fit is meaningless
    for a scale-incoherent model, and mixing gauges between neighbouring
    cameras tears the chain (measured: 80-180 m/s boundary jumps when the
    fallback used the global transform).  Nearest-window fallback keeps
    every adjacent pair consistent; BA with soft priors then reconciles
    the extrapolated stretches.

    Points: a point's transform is derived from its FIRST observing
    camera's window transform (same rigid rule) — points move with the
    cameras that saw them, so observations stay consistent and projections
    are invariant within each camera's frame of reference up to the
    between-window discontinuity BA then smooths via soft metric priors.
    Rotation composition follows the codebase camera-to-world convention
    (R_c2w' = R_i R_c2w, depths scale by s_i, projections invariant).
    """
    cams_by_stem = {Path(name).stem: name for name in result.cameras}
    applied = 0
    # Nearest-in-time fallback: camera_ts keys are stems.
    anchor_stems = [
        (camera_ts.get(Path(n).stem), n) for n in correction.camera_transform
    ]
    anchor_stems = [(t, n) for t, n in anchor_stems if t is not None]
    anchor_stems.sort()
    fallback_transforms = 0
    #: Fallback trust radius (s): a camera further than this from ANY accepted
    #: window is OUTSIDE the piecewise solution's measured territory.  Its
    #: true transform is unknown — recording it as UNRESOLVED (identity kept,
    #: named in the report) is the honest act; extrapolating a transform from
    #: a distant window fabricates placement across an unmeasured gap and
    #: tears the chain at the resolved boundary (measured: 78-185 m/s jumps
    #: at i=44-48 when the global fit contaminated the fallback pool).
    fallback_time_band_s = 3.5
    unresolved: list[str] = []
    for name, cam in result.cameras.items():
        tr = correction.camera_transform.get(name)
        if tr is None and anchor_stems:
            stem = Path(name).stem
            ts_i = camera_ts.get(stem)
            if ts_i is not None:
                t_nearest, n_nearest = min(
                    anchor_stems, key=lambda a: abs(a[0] - ts_i)
                )
                if abs(t_nearest - ts_i) <= fallback_time_band_s:
                    tr = correction.camera_transform.get(n_nearest)
                    if tr is not None:
                        fallback_transforms += 1
                else:
                    unresolved.append(Path(name).stem)
            else:
                unresolved.append(Path(name).stem)
        elif tr is None:
            unresolved.append(Path(name).stem)
        if tr is None and correction.global_transform is not None:
            tr = correction.global_transform
        if tr is None:
            continue
        s, R, t = tr
        C = np.asarray(cam.position, dtype=np.float64)
        R_c2w = np.asarray(cam.rotation, dtype=np.float64)
        cam.position = s * (R @ C) + t
        R_new = R @ R_c2w
        cam.rotation = R_new
        if hasattr(cam, "quaternion"):
            from scipy.spatial.transform import Rotation as _Rot
            cam.quaternion = _Rot.from_matrix(R_new).as_quat()[[3, 0, 1, 2]]
        if hasattr(cam, "translation"):
            cam.translation = cam.position
        applied += 1
    if fallback_transforms or unresolved:
        correction.segments.append({
            "note": "nearest-window fallback applied within trust band",
            "cameras_fallback": fallback_transforms,
            "cameras_unresolved": unresolved,
            "fallback_time_band_s": fallback_time_band_s,
        })
    # Points: transform by their first observer's transform.  The same
    # nearest-window rule applies to points whose first observer fell back
    # (it was resolved above and recorded on the camera itself via the
    # applied transform map we rebuild here for the point pass).
    per_cam = dict(correction.camera_transform)
    # Extend per_cam with the fallback assignment each camera actually got
    # (nearest accepted window WITHIN the trust band) so points follow their
    # observer exactly.  Unresolved cameras resolve to the global transform
    # (documented, honest, smooth) for the point pass only — points need a
    # transform to stay projection-consistent with their cameras, and the
    # retriangulation immediately after discards torn geometry anyway.
    for name, cam in result.cameras.items():
        if name in per_cam:
            continue
        stem = Path(name).stem
        ts_i = camera_ts.get(stem) if camera_ts is not None else None
        if ts_i is not None and anchor_stems:
            t_nearest, n_nearest = min(
                anchor_stems, key=lambda a: abs(a[0] - ts_i)
            )
            if abs(t_nearest - ts_i) <= 3.5:
                tr = correction.camera_transform.get(n_nearest)
                if tr is not None:
                    per_cam[name] = tr
                    continue
        if correction.global_transform is not None:
            per_cam[name] = correction.global_transform
    moved = 0
    for pt in result.points3d:
        # resolve the point's first observing camera directly
        obs_name = None
        for cam_name, _uv in pt.observations:
            if cam_name in result.cameras:
                obs_name = cam_name
                break
        tr = per_cam.get(obs_name) if obs_name else None
        if tr is None:
            continue
        s, R, t = tr
        X = np.asarray(pt.position, dtype=np.float64)
        pt.position = s * (R @ X) + t
        moved += 1
    return {
        "cameras_transformed": applied,
        "points_transformed": moved,
        "cameras_unresolved": unresolved,
    }


def _dlt_triangulate(
    R_w2c: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    uvs: np.ndarray,
) -> np.ndarray | None:
    """Least-squares DLT triangulation from one camera set.

    R_w2c: (n,3,3) world-to-camera rotations; C: (n,3) camera centres;
    K: (n,3,3) intrinsics; uvs: (n,2) pixel observations.  Returns the
    world point or None when degenerate (fewer than 2 views, ill-conditioned
    system, behind-camera solution).

    The 2n DLT rows are assembled with numpy rather than a per-view Python
    loop: this runs once per point (~60k points per run) and each row is a
    pure broadcast of the same projection matrix, so the loop only cost
    interpreter time. A is built identically, so the SVD sees the same matrix.
    """
    n = len(uvs)
    if n < 2:
        return None
    R_w2c = np.asarray(R_w2c, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)
    # P = K [R | -R C] per view, then the two rows (u P2 - P0, v P2 - P1).
    # Plain matmul keeps the same BLAS kernel as the per-view K[i] @ M[i] call
    # it replaced, so the matrix the SVD sees is not merely equivalent but
    # identical — the last singular vector is only well-determined when the
    # ray system is, and an ill-conditioned point would otherwise drift.
    P = np.concatenate([R_w2c, -np.matmul(R_w2c, C[:, :, None])], axis=2)
    P = np.matmul(K, P)
    A = np.empty((2 * n, 4), dtype=np.float64)
    A[0::2] = uvs[:, 0, None] * P[:, 2, :] - P[:, 0, :]
    A[1::2] = uvs[:, 1, None] * P[:, 2, :] - P[:, 1, :]
    try:
        _u, s_vt, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    if s_vt.size < 4 or s_vt[2] < 1e-9 * max(s_vt[0], 1e-12):
        return None  # ill-conditioned (near-parallel rays)
    Xh = vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    X = Xh[:3] / Xh[3]
    # reject behind-camera solutions
    if np.any(np.matmul(R_w2c, (X - C)[:, :, None])[:, 2, 0] <= 0):
        return None
    return X


#: Below this many points a run is already too thin for the integrity
#: fraction to be meaningful (a 3-point cloud can only retain 0% or 100%).
_MIN_POINTS_FOR_INTEGRITY = 50


def _min_point_retention() -> float:
    """Retention floor for the placement integrity gate (settings-owned).

    Imported lazily: this module and sparse_conditioning import each other, so
    the settings module is deliberately not pulled in at import time.
    """
    from app.config.settings import settings as _settings

    return float(_settings.telemetry.placement_min_point_retention)


def _snapshot_placement_state(result) -> dict:
    """Capture the geometry a placement may overwrite (so it can be undone)."""
    return {
        "points": len(result.points3d),
        "cameras": {
            name: (
                np.asarray(cam.position, dtype=np.float64).copy(),
                np.asarray(cam.rotation, dtype=np.float64).copy(),
            )
            for name, cam in result.cameras.items()
        },
        "points3d": [
            (
                pt,
                np.asarray(pt.position, dtype=np.float64).copy(),
                list(pt.observations),
                pt.track_length,
            )
            for pt in result.points3d
        ],
    }


def _restore_placement_state(result, snapshot: dict) -> None:
    """Undo a placement: geometry and observation sets return to the snapshot."""
    for name, (pos, rot) in snapshot["cameras"].items():
        cam = result.cameras.get(name)
        if cam is not None:
            cam.position = pos
            cam.rotation = rot
    for pt, pos, obs, tl in snapshot["points3d"]:
        pt.position = pos
        pt.observations = obs
        pt.track_length = tl
    result.points3d = [entry[0] for entry in snapshot["points3d"]]
    result.num_points = len(result.points3d)


def retriangulate_points(result, max_obs_err_px: float = 10.0) -> dict:
    """Re-fit every sparse point from its observations through the CURRENT
    (placed) cameras, dropping inconsistent observations.

    Why: a piecewise (per-window) placement moves different cameras by
    different rigid transforms, so transforming point positions by any
    single per-point rule TEARS the point cloud relative to the cameras at
    window boundaries — and BA starting from a torn state fights its own
    priors (measured: 830 m/s camera jumps post-BA, initial reproj 1098 px).
    Re-triangulating from the placed cameras restores point<->camera
    consistency BEFORE BA, which then polishes both from a coherent basin.

    After the DLT refit each observation is re-projected; observations whose
    pixel error exceeds ``max_obs_err_px`` are DROPPED (a torn observation
    is poison for BA — keeping it forces a compromise that bows the whole
    model), a second refit runs on the surviving observations, and points
    left with <2 observations are removed entirely (reported, never kept as
    torn geometry).
    """
    cams = {
        name: (
            np.asarray(cam.rotation, dtype=np.float64).T,   # c2w -> w2c
            np.asarray(cam.position, dtype=np.float64),
            np.asarray(cam.intrinsics, dtype=np.float64),
        )
        for name, cam in result.cameras.items()
    }
    moved = dropped_points = 0
    dropped_obs = 0
    nullspace_points = 0
    survivors: list = []
    for pt in result.points3d:
        obs = [(name, uv) for name, uv in pt.observations if name in cams]
        if len(obs) < 2:
            dropped_points += 1
            continue
        X = _dlt_triangulate(
            np.stack([cams[n][0] for n, _uv in obs]),
            np.stack([cams[n][1] for n, _uv in obs]),
            np.stack([cams[n][2] for n, _uv in obs]),
            np.array([uv for _n, uv in obs], dtype=np.float64),
        )
        if X is None:
            dropped_points += 1
            continue
        # Per-observation pixel error of the refit point; drop outliers.
        # Batched over the point's own track: the previous form called the
        # projection helper once per observation (812k calls on the run under
        # measurement) to produce one scalar each. Same convention, same
        # z <= 1e-6 drop, same threshold — including the count of dropped
        # observations, so the refusal statistics are unchanged.
        names = [name for name, _uv in obs]
        uvs = np.asarray([uv for _name, uv in obs], dtype=np.float64)
        R_c2w = np.asarray([cams[name][0].T for name in names], dtype=np.float64)
        Cv = np.asarray([cams[name][1] for name in names], dtype=np.float64)
        Kv = np.abs(np.asarray([cams[name][2] for name in names], dtype=np.float64))
        # (X − C) @ R, batched through the same matmul kernel the single
        # per-observation call used.
        Xc = np.matmul((X - Cv)[:, None, :], R_c2w)[:, 0, :]
        z = Xc[:, 2]
        ok = z > 1e-6
        dropped_obs += int((~ok).sum())
        err = np.full(len(obs), np.inf, dtype=np.float64)
        if ok.any():
            denom = np.where(np.abs(z) > 1e-12, z, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                u = Xc[:, 0] / denom * Kv[:, 0, 0] + Kv[:, 0, 2]
                v = Xc[:, 1] / denom * Kv[:, 1, 1] + Kv[:, 1, 2]
            err[ok] = np.hypot(u[ok] - uvs[ok, 0], v[ok] - uvs[ok, 1])
        keep = err <= max_obs_err_px
        dropped_obs += int((~keep).sum())
        kept_obs: list[tuple[str, np.ndarray]] = [
            obs[i] for i in np.nonzero(keep)[0]
        ]
        if len(kept_obs) < 2:
            dropped_points += 1
            continue
        # Conditioning screen (single owner: app.services.sparse_conditioning):
        # drop points whose observing cameras have too little baseline
        # relative to their depth. Hover-segment points reproject fine at ANY
        # depth (tiny baselines forgive everything), so reprojection alone
        # cannot catch them — but their along-ray position is unconstrained
        # and measured ~2x wrong. Keeping them poisons the depth-alignment
        # reference (the dense-stage Criteria A–G audit then correctly
        # refuses the run).
        min_base = point_min_baseline_m([n for n, _uv in kept_obs], cams)
        depth = float(np.median(np.matmul(
            np.asarray([cams[n][0] for n, _uv in kept_obs], dtype=np.float64),
            (X - np.asarray([cams[n][1] for n, _uv in kept_obs], dtype=np.float64))[:, :, None],
        )[:, 2, 0]))
        if depth > 0 and min_base < MIN_DEPTH_BASELINE_RATIO * depth:
            nullspace_points += 1
            dropped_points += 1
            continue
        if len(kept_obs) < len(obs):
            # second refit on the surviving observations only
            X2 = _dlt_triangulate(
                np.stack([cams[n][0] for n, _uv in kept_obs]),
                np.stack([cams[n][1] for n, _uv in kept_obs]),
                np.stack([cams[n][2] for n, _uv in kept_obs]),
                np.array([uv for _n, uv in kept_obs], dtype=np.float64),
            )
            if X2 is not None:
                X = X2
        pt.position = X
        pt.observations = kept_obs
        pt.track_length = len(kept_obs)
        survivors.append(pt)
        moved += 1
    result.points3d = survivors
    result.num_points = len(survivors)
    return {
        "points_refit": moved,
        "points_dropped": dropped_points,
        "nullspace_points_dropped": nullspace_points,
        "observations_dropped": dropped_obs,
        "min_depth_baseline_ratio": MIN_DEPTH_BASELINE_RATIO,
    }


def _global_placement_report(
    match_report: dict,
    *,
    scale: float,
    R: np.ndarray,
    tr: np.ndarray,
    robust_info: dict,
    outlier_frac: float,
    result,
    pairs,
    telemetry: PositionPrior,
    refusal: dict | None = None,
    piecewise_report: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Apply the global robust similarity and build THE placement report.

    Single owner of the global-similarity placement: both the plain path and
    the piecewise-refused fallback come through here, so the scale, rigid
    transform and post-alignment residuals are reported identically in either
    case (a refusal that omitted them silently removed the input a downstream
    scale sanity check reads).
    """
    apply_similarity_to_reconstruction(result, scale, R, tr)
    T = np.array([telemetry.positions[j] for _name, j, _dt in pairs])
    aligned_resid = np.linalg.norm(
        np.array([result.cameras[name].position for name, _j, _dt in pairs]) - T, axis=1
    )
    pair_names = [name for name, _j, _dt in pairs]
    outlier_by_camera = {
        pair_names[o["index"]]: o["residual_m"] for o in robust_info["rejected_correspondences"]
    }
    report = {
        **match_report,
        "placement_mode": "global_similarity" if refusal else "global_robust",
        "similarity_scale_visual_per_telemetry": round(scale, 6),
        "rotation_3x3": np.round(R, 6).tolist(),
        "translation_3x1": np.round(tr, 4).tolist(),
        "robust_fit": robust_info,
        "global_robust_fit": {
            "scale": round(scale, 6),
            "inliers": robust_info["inliers"],
            "outliers": robust_info["outliers"],
            "outlier_fraction": round(outlier_frac, 3),
            "inlier_residual_m": robust_info["inlier_residual_m"],
        },
        "outlier_cameras": outlier_by_camera,
        "residual_after_alignment_m": {
            "median": round(float(np.median(aligned_resid)), 3),
            "p95": round(float(np.percentile(aligned_resid, 95)), 3),
            "max": round(float(aligned_resid.max()), 3),
            "inlier_median": robust_info["inlier_residual_m"]["median"],
        },
        "engagement_reason": (
            "outlier_fraction" if outlier_frac > 0.25 else "inlier_residual"
        ),
        "metric_targets_for_ba": len(
            [1 for name, _j, _dt in pairs if name in result.cameras]
        ),
        "usage": (
            "soft metric_targets prior for joint BA; model placed by the "
            "measured ROBUST similarity (rigid + uniform scale, outliers "
            "named) — scale is a measurement, reported, never silent"
        ),
    }
    if refusal is not None:
        report["piecewise"] = {**(piecewise_report or {}), "refused": refusal}
        report["usage"] = (
            "global robust similarity (piecewise refused: it retained "
            f"{refusal['points_retained_fraction']} of the points, below the "
            f"{refusal['retained_floor']} floor); "
            "scale is a measurement, reported, never silent"
        )
    targets = {
        name: telemetry.positions[j]
        for name, j, _dt in pairs
        if name in result.cameras
    }
    return targets, report


def align_reconstruction_to_telemetry(
    result,
    telemetry: PositionPrior,
    camera_ts: dict[str, float],
    *,
    max_dt_s: float = DEFAULT_MAX_DT_S,
) -> tuple[dict[str, np.ndarray], dict]:
    """Place a video-only reconstruction into the telemetry ENU frame.

    Correspondences are TIME-matched (camera PTS -> nearest unique GPS fix
    within ``max_dt_s``); the similarity (s, R, t) fitted on them is the
    MEASURED visual->telemetry placement — applied rigidly to the whole
    model.  Returns (metric_targets for BA keyed by camera name, report).
    """
    centers = {name: np.asarray(cam.position, dtype=np.float64) for name, cam in result.cameras.items()}
    pairs, match_report = correspondences_by_time(centers, camera_ts, telemetry, max_dt_s=max_dt_s)
    if len(pairs) < 4:
        return {}, {**match_report, "error": f"fewer than 4 time-matched correspondences ({len(pairs)})"}
    V = np.array([centers[name] for name, _j, _dt in pairs])
    T = np.array([telemetry.positions[j] for _name, j, _dt in pairs])
    # ROBUST global fit first: a drifted/locally-misregistered camera
    # segment must not contaminate the placement.  When the robust fit
    # leaves >25% outliers, the trajectory carries DRIFT BENDS that no
    # single similarity can represent — switch to the piecewise correction
    # (per-window rigid+scale fits along the flight).  All matched cameras
    # keep metric_targets so BA refines from a good initialization.
    scale, R, tr, robust_info = umeyama_similarity_robust(V, T)
    outlier_frac = robust_info["outliers"] / max(1, len(pairs))
    inlier_med = float(robust_info["inlier_residual_m"]["median"])
    piecewise_report: dict | None = None
    # Engage piecewise when the global fit is insufficient in EITHER sense:
    # too many cameras disagree (drift bends), OR even the agreeing cameras
    # sit too far from telemetry (diffuse warp — a bent trajectory can keep
    # outlier fraction low while sitting metres off everywhere; measured
    # 7511dc rerun9: 21.8% outliers but 8.7 m inlier median).  2.0 m is the
    # GNSS noise scale of this log (quantized ~1 m fixes); beyond it the
    # global similarity is not a faithful placement.
    if outlier_frac > 0.25 or inlier_med > 2.0:
        correction, piecewise_report = _piecewise_rigid_correction(
            centers, pairs, telemetry, camera_ts
        )
        if correction is not None:
            # Snapshot the pre-placement state: if the piecewise correction
            # turns out not to be a faithful placement (see the integrity
            # gate below) the run must fall back to the global similarity
            # from the ORIGINAL geometry, not from a half-applied one.
            snapshot = _snapshot_placement_state(result)
            apply_info = apply_piecewise_correction(
                result, correction, camera_ts=camera_ts
            )
            # Restore point<->camera consistency before BA: per-window
            # transforms tear the cloud at window boundaries, so re-fit
            # every point from its own observations through the PLACED
            # cameras (projections invariant, geometry coherent again).
            retri_info = retriangulate_points(result)
            apply_info["retriangulated"] = retri_info
            piecewise_report["applied"] = apply_info
            pre_points = int(snapshot["points"])
            retained = len(result.points3d) / max(1, pre_points)
            apply_info["points_retained_fraction"] = round(retained, 5)
            # ---- placement integrity gate ---------------------------------
            # A placement that cannot re-explain the observations is not a
            # placement, it is a tear. Measured on furnerhem_6_34be62: the
            # piecewise correction left 569,903 of 666,675 observations
            # >10 px off their refit points, so retriangulation retained
            # 355 of 52,882 points (0.7%) — a one-point-per-60 m scaffold
            # that starved per-view depth anchoring (29 m alignment error),
            # smeared the fused cloud into 4.8 m-separated duplicate layers
            # and produced a 9,748-component mesh out of a spotless SfM
            # model (89 cameras, 0.33 px). The global similarity is a single
            # rigid transform: it cannot tear the cloud, so it preserves the
            # observations and is the honest fallback. Refuse by name.
            if pre_points >= _MIN_POINTS_FOR_INTEGRITY and retained < _min_point_retention():
                _restore_placement_state(result, snapshot)
                refusal = {
                    "reason": "low_point_retention",
                    "points_before": pre_points,
                    "points_after_torn": len(result.points3d),
                    "points_retained_fraction": round(retained, 5),
                    "retained_floor": _min_point_retention(),
                    "observations_dropped": retri_info.get("observations_dropped"),
                    "note": (
                        "per-window transforms left the observations "
                        "unexplainable; the rigid global similarity preserves "
                        "the point cloud, so it is used instead. The "
                        "trajectory's residual warp is reported, not hidden."
                    ),
                }
                log.warning(
                    "piecewise_placement_refused_low_retention",
                    points_before=pre_points,
                    points_after=len(result.points3d),
                    retained_fraction=round(retained, 5),
                    observations_dropped=retri_info.get("observations_dropped"),
                    note="piecewise correction could not re-explain the observations "
                         "— falling back to the global robust similarity",
                )
                targets, report = _global_placement_report(
                    match_report,
                    scale=scale,
                    R=R,
                    tr=tr,
                    robust_info=robust_info,
                    outlier_frac=outlier_frac,
                    result=result,
                    pairs=pairs,
                    telemetry=telemetry,
                    refusal=refusal,
                    piecewise_report=piecewise_report,
                )
                return targets, report
            report = {
                **match_report,
                "placement_mode": "piecewise_rigid",
                "global_robust_fit": {
                    "scale": round(scale, 6),
                    "inliers": robust_info["inliers"],
                    "outliers": robust_info["outliers"],
                    "outlier_fraction": round(outlier_frac, 3),
                    "inlier_residual_m": robust_info["inlier_residual_m"],
                },
                "piecewise": piecewise_report,
                "engagement_reason": (
                    "outlier_fraction" if outlier_frac > 0.25 else "inlier_residual"
                ),
                "metric_targets_for_ba": len(
                    [1 for name, _j, _dt in pairs if name in result.cameras]
                ),
                "usage": (
                    "piecewise rigid+uniform-scale corrections (drift bends "
                    "detected: robust global fit left "
                    f"{robust_info['outliers']}/{len(pairs)} outliers); BA "
                    "refines with soft metric priors — scale is a measurement, "
                    "reported, never silent"
                ),
            }
            targets = {
                name: telemetry.positions[j]
                for name, j, _dt in pairs
                if name in result.cameras
            }
            return targets, report
    targets = {
        name: telemetry.positions[j]
        for name, j, _dt in pairs
        if name in result.cameras
    }
    return _global_placement_report(
        match_report,
        scale=scale,
        R=R,
        tr=tr,
        robust_info=robust_info,
        outlier_frac=outlier_frac,
        result=result,
        pairs=pairs,
        telemetry=telemetry,
    )


def pose_jump_report(
    centers: np.ndarray,
    timestamps: np.ndarray,
    *,
    sigma: float = DEFAULT_JUMP_SIGMA,
) -> dict:
    """Continuity diagnostic on the PLACED camera trajectory: steps whose
    implied speed exceeds a robust band (median + sigma x MAD, floored at
    5 m/s above median) are impossible-jump candidates — reported, never
    silently smoothed.
    """
    C = np.asarray(centers, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    if len(C) < 3:
        return {"pass": False, "error": "fewer than 3 cameras"}
    order = np.argsort(t)
    C, t = C[order], t[order]
    # Cadence normalization: raw per-frame steps are dominated by SfM
    # jitter at frame cadence (~0.033 s); jumps that matter are visible at
    # the common shape grid (0.5 s).  Raw-jump stats are still reported.
    Cg, tg = resample_path(C, t, SHAPE_GRID_INTERVAL_S)
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    dt = np.diff(t)
    speed = step / np.where(dt > 1e-6, dt, np.nan)
    med = float(np.nanmedian(speed))
    mad = float(np.nanmedian(np.abs(speed - med))) or 1e-6
    limit = med + sigma * max(5.0 * mad, 5.0)
    bad_idx = np.flatnonzero(speed > limit)
    step_g = np.linalg.norm(np.diff(Cg, axis=0), axis=1)
    dtg = np.diff(tg)
    speed_g = step_g / np.where(dtg > 1e-6, dtg, np.nan)
    med_g = float(np.nanmedian(speed_g))
    mad_g = float(np.nanmedian(np.abs(speed_g - med_g))) or 1e-6
    limit_g = med_g + sigma * max(5.0 * mad_g, 5.0)
    bad_g = np.flatnonzero(speed_g > limit_g)
    return {
        "pass": bool(len(bad_g) == 0),
        "cameras": int(len(C)),
        "grid_interval_s": SHAPE_GRID_INTERVAL_S,
        "grid_points": int(len(Cg)),
        "median_speed_m_s": round(med_g, 3),
        "max_speed_m_s": round(float(np.nanmax(speed_g)), 3),
        "jump_limit_m_s": round(limit_g, 3),
        "jump_count": int(len(bad_g)),
        "jump_examples": [
            {"i": int(i), "speed_m_s": round(float(speed_g[i]), 2)}
            for i in bad_g[:10]
        ],
        "raw_frame_jump_count": int(len(bad_idx)),
        "raw_max_speed_m_s": round(float(np.nanmax(speed)), 3),
    }


def trajectory_alignment_report(
    visual_centers: np.ndarray,
    telemetry: PositionPrior,
) -> dict:
    """Shared-ENU trajectory diagnostic: raw and aligned states together.

    Prevents the visualization-fakes-agreement trap: both trajectories are
    emitted in ONE coordinate frame (telemetry ENU) before and after the
    diagnostic similarity alignment, with the transform parameters and
    residuals saved numerically.  Plots must be drawn from this file.
    """
    V = np.asarray(visual_centers, dtype=np.float64)
    T = np.asarray(telemetry.positions, dtype=np.float64)
    n = min(len(V), len(T))
    scale, R, tr = umeyama_similarity(V[:n], T[:n])
    aligned = scale * (V @ R.T) + tr
    resid = np.linalg.norm(aligned[:n] - T[:n], axis=1)
    tv = telemetry.timestamps
    tt = np.linspace(tv[0], tv[-1], n)
    return {
        "frame": "telemetry ENU (anchor at first valid fix)",
        "anchor": telemetry.anchor,
        "video_timestamps": [round(float(v), 3) for v in tt],
        "telemetry_timestamps": [round(float(v), 3) for v in tv[:]],
        "visual_before_alignment_enu": np.round(V, 3).tolist(),
        "telemetry_enu": np.round(T, 3).tolist(),
        "visual_after_alignment_enu": np.round(aligned, 3).tolist(),
        "transform": {
            "scale": round(scale, 6),
            "rotation_3x3": np.round(R, 6).tolist(),
            "translation_3x1": np.round(tr, 4).tolist(),
        },
        "residuals_after_alignment_m": {
            "median": round(float(np.median(resid)), 3),
            "p95": round(float(np.percentile(resid, 95)), 3),
            "max": round(float(resid.max()), 3),
        },
        "usage_note": (
            "Plot 'visual_after_alignment_enu' and 'telemetry_enu' together; "
            "'visual_before_alignment_enu' is the pre-placement state in the "
            "same axes for honest contrast. Never align plots independently."
        ),
    }
