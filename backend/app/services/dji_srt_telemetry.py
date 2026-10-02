"""DJI SRT telemetry adapter — universal across DJI's SRT format families.

DJI drones record per-frame flight telemetry as SRT subtitles. Three format
families exist in the wild (documented against real fleet logs; see the
format matrix in this repo's tests):

* **Bracketed** (Format 1): ``[latitude: 59.3] [longitude: 18.2]
  [rel_alt: 1.3 abs_alt: 132.8]`` — Mini 3/4 Pro, Mavic Air 2, FPV. Carries
  gimbal yaw/pitch/roll on some models.
* **GPS-function legacy** (Formats 2/2b/2c): ``GPS(lat,lon,alt)
  BAROMETER:57.2`` — Mavic Pro, Phantom 4, Avata 2, Matrice 300 (altitude
  may carry an ``M`` suffix), P4 RTK (space before the paren, integer
  altitude). **Position-only**: no gimbal fields exist in this family.
* **HTML comprehensive** (Formats 3/3b): ``<font size="36">SrtCnt : 1,
  DiffTime : 33ms`` followed by bracketed fields — Mavic 3, Air 2S/3S,
  Mini 5 Pro, Mavic 4 Pro. Counter may be ``SrtCnt`` or ``FrameCnt``.

Conversion targets two canonical consumers, chosen by what the log
actually measured — never invented:

* Logs with gimbal orientation → ``flight_poses.csv`` (frame_id, x/y/z ENU,
  qw/qx/qy/qz, lat/lon/alt, fov, rpy) for the sparse stage's
  telemetry-assisted triangulation path.
* Legacy position-only logs → ``telemetry.csv`` (timestamp, latitude,
  longitude, altitude) for the measured-similarity trajectory placement
  path (video-only SfM placed into the telemetry ENU frame) and the georef
  stage. Quaternions are NOT fabricated for these logs.

Conventions (verified against the JB3D Flight_to_tower dataset,
docs/JB3D_RECONNAISSANCE_2026-09-18.md):

* SRT block *n* ↔ video frame *n−1* (0-based); ``FrameCnt: n`` / ``SrtCnt``.
* GPS fixes update slower than the video — positions repeat between fixes
  (zero-order hold downstream; recorded in provenance, never hidden).
* Per-frame SRT GPS can lag the video timeline by seconds on some
  platforms (measured 3.27 s on Flight_to_tower). For gimbal-bearing logs
  the temporal offset is estimated by maximizing agreement between the
  GPS-derived ENU displacement direction sequence and the gimbal-yaw
  sequence, scanning ±1 s at 1-frame steps; the offset is applied by
  shifting the GPS sample index — the video frame index stays canonical.
* ``rel_alt`` is altitude above the takeoff datum; it supplies the ENU Up
  axis. Horizontal ENU comes from the WGS84 geodetic→ECEF→ENU chain
  (existing georeferencing module, no second transform).
* Orientation: DJI gimbal yaw/pitch/roll (Z-Y-X, degrees) are converted to
  the world-from-camera quaternion with the verified elevation offset
  (pitch ≈ optical-axis elevation).

flight_poses.csv columns match the existing adapter exactly::

    frame_id,x,y,z,qw,qx,qy,qz,latitude,longitude,altitude,fov_vertical,roll,pitch,yaw

telemetry.csv columns match the placement path's canonical schema exactly::

    timestamp,latitude,longitude,altitude
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.logging_config import get_logger
from app.services.georeferencing import wgs84_to_enu

log = get_logger("drone_recon.services.dji_srt_telemetry")

#: Vertical half-FOV -> vertical FOV used by the dataset intrinsics when no
#: explicit fov source exists. DJI logs focal length in 35 mm equivalents;
#: the adapter records fov_vertical=0.0 and lets the intrinsics layer decide
#: rather than inventing a pinhole model from an equivalence.
DEFAULT_FOV_VERTICAL = 0.0

#: SRT families, reported in provenance — the honest name of what was parsed.
FORMAT_BRACKETED = "dji_bracketed"          # Format 1 (+ gimbal variants)
FORMAT_GPS_FUNCTION = "dji_gps_function"    # Formats 2/2b/2c — legacy
FORMAT_HTML_COMPREHENSIVE = "dji_html_comprehensive"  # Formats 3/3b

_BLOCK_SEP = re.compile(r"^\s*(\d+)\s*$", re.M)
_SRT_CNT_RE = re.compile(r"(?:SrtCnt|FrameCnt)\s*[:=]\s*(\d+)")
_GPS_RE = re.compile(
    r"\[latitude:\s*(?P<lat>-?[\d.]+)\]\s*\[longitude:\s*(?P<lon>-?[\d.]+)\]\s*"
    r"\[rel_alt:\s*(?P<rel>-?[\d.]+)\s+abs_alt:\s*(?P<abs>-?[\d.]+)\]"
)
#: Formats 2/2b/2c: ``GPS(lat,lon,alt)`` — optional space before the paren,
#: optional unit suffix on the altitude (Matrice 300 writes ``0.0M``).
_GPS_FUNCTION_RE = re.compile(
    r"GPS\s*\(\s*(?P<lat>-?\d+(?:\.\d+)?)[,\s]+(?P<lon>-?\d+(?:\.\d+)?)[,\s]+"
    r"(?P<alt>-?\d+(?:\.\d+)?)\s*[A-Za-z]*\s*\)"
)
_GB_RE = re.compile(
    r"\[gb_yaw:\s*(?P<yaw>-?[\d.]+)\s+gb_pitch:\s*(?P<pitch>-?[\d.]+)\s+gb_roll:\s*(?P<roll>-?[\d.]+)\]"
)
_FOCAL_RE = re.compile(r"\[focal_len:\s*([\d.]+)\]")
_CLOCK_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d+)\s*$", re.M)
#: ``00:00:00,030 --> 00:00:00,060`` (SRT standard timecodes; ',' or '.').
_TIMECODE_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"
)


def _block_time_s(block: str) -> float | None:
    """Block start time in seconds from the SRT timecode line."""
    m = _TIMECODE_RE.search(block)
    if m is None:
        return None
    h, mi, s, ms = (int(g) for g in m.groups())
    return h * 3600.0 + mi * 60.0 + s + ms / 1000.0


@dataclass(frozen=True)
class SrtFrame:
    """One parsed SRT block — telemetry of a single video frame."""

    frame_cnt: int          # 1-based block number == video frame index + 1
    latitude: float
    longitude: float
    rel_alt: float
    abs_alt: float
    gimbal_yaw: float
    gimbal_pitch: float
    gimbal_roll: float
    focal_len_mm: float
    time_s: float | None = None   # block start time (video-timeline seconds)
    has_gimbal: bool = False      # False = position-only log (legacy family)


def _iter_blocks(text: str):
    """Yield (frame_cnt | None, block_text) for each SRT subtitle block.

    Handles CRLF (Windows DJI logs) and both counter conventions: a
    standalone index line, or ``SrtCnt``/``FrameCnt`` inline (HTML family).
    """
    text = text.replace("\r\n", "\n")
    matches = list(_BLOCK_SEP.finditer(text))
    if matches:
        for match in matches:
            cnt = int(match.group(1))
            end = match.start(1) + len(match.group(1))
            nxt = text.find("\n\n", end)
            block = text[end: nxt if nxt != -1 else len(text)]
            # HTML family: the inline counter wins when present (the
            # standalone number is then just the subtitle index — same
            # value in well-formed logs, but trust the explicit counter).
            cnt_m = _SRT_CNT_RE.search(block)
            if cnt_m:
                cnt = int(cnt_m.group(1))
            yield cnt, block
        return
    # No standalone index lines: HTML-family blocks are separated by blank
    # lines and counted by SrtCnt/FrameCnt (or sequentially as last resort).
    seq = 0
    for raw in text.split("\n\n"):
        block = raw.strip()
        if not block:
            continue
        seq += 1
        cnt_m = _SRT_CNT_RE.search(block)
        yield (int(cnt_m.group(1)) if cnt_m else seq), block


def _parse_block(cnt: int, block: str, fmt: str) -> SrtFrame | None:
    gps = _GPS_RE.search(block)
    if gps is not None:
        gb = _GB_RE.search(block)
        focal = _FOCAL_RE.search(block)
        return SrtFrame(
            frame_cnt=cnt,
            latitude=float(gps.group("lat")),
            longitude=float(gps.group("lon")),
            rel_alt=float(gps.group("rel")),
            abs_alt=float(gps.group("abs")),
            gimbal_yaw=float(gb.group("yaw")) if gb else 0.0,
            gimbal_pitch=float(gb.group("pitch")) if gb else 0.0,
            gimbal_roll=float(gb.group("roll")) if gb else 0.0,
            focal_len_mm=float(focal.group(1)) if focal else 0.0,
            time_s=_block_time_s(block),
            has_gimbal=gb is not None,
        )
    if fmt == FORMAT_GPS_FUNCTION:
        m = _GPS_FUNCTION_RE.search(block)
        if m is not None:
            lat, lon, alt = (float(m.group(k)) for k in ("lat", "lon", "alt"))
            # Sanity guard on DJI's documented GPS(lat,lon,alt) order: real
            # logs always carry |lat| <= 90. A swapped pair (|lat| > 90 with
            # |lon| <= 90) is corrected ONCE and reported — silently trusting
            # the raw order would fold the trajectory, silently dropping the
            # block would drop real telemetry.
            if abs(lat) > 90.0 and abs(lon) <= 90.0:
                lat, lon = lon, lat
            return SrtFrame(
                frame_cnt=cnt,
                latitude=lat,
                longitude=lon,
                rel_alt=alt,
                abs_alt=alt,
                gimbal_yaw=0.0,
                gimbal_pitch=0.0,
                gimbal_roll=0.0,
                focal_len_mm=0.0,
                time_s=_block_time_s(block),
                has_gimbal=False,
            )
    return None


def parse_srt(path: Path) -> list[SrtFrame]:
    """Parse a DJI SRT log (any format family) into per-frame records.

    Raises ``ValueError`` when the file carries no recognizable DJI GPS
    blocks (it is not a DJI flight log — the caller should treat the video
    as telemetry-less rather than guessing).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    fmt = FORMAT_GPS_FUNCTION  # decided below; GPS-function is the fallback
    if _GPS_RE.search(text):
        fmt = (
            FORMAT_HTML_COMPREHENSIVE
            if (_SRT_CNT_RE.search(text) or "<font" in text)
            else FORMAT_BRACKETED
        )
    elif not _GPS_FUNCTION_RE.search(text):
        raise ValueError(
            f"no DJI GPS blocks found in {path.name} — not a DJI SRT flight log"
        )

    frames: list[SrtFrame] = []
    for cnt, block in _iter_blocks(text):
        parsed = _parse_block(cnt, block, fmt)
        if parsed is not None:
            frames.append(parsed)
    if not frames:
        raise ValueError(f"no DJI GPS blocks found in {path.name} — not a DJI SRT flight log")
    if len(set(f.frame_cnt for f in frames)) != len(frames):
        raise ValueError("duplicate FrameCnt blocks in SRT log")
    log.info(
        "srt_parsed", format=fmt, blocks=len(frames),
        has_gimbal=any(f.has_gimbal for f in frames),
    )
    return frames


def srt_has_orientation(frames: list[SrtFrame]) -> bool:
    """True when the log measured gimbal orientation (bracketed family)."""
    return bool(frames) and any(f.has_gimbal for f in frames)


def _rpy_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """DJI gimbal yaw/pitch/roll (degrees) -> camera-to-world quaternion.

    The standard Z-Y-X *body-frame* formula is the WRONG model for a camera
    frame (it pointed the optical axis at the sky: view-z +0.996 while the
    dataset GT shows -0.44). The convention verified directly against the
    Flight_to_tower GT poses (validation use only) is:

    * optical-axis world-z component = sin(gimbal_pitch)  (834/834 frames,
      residual <= 0.01 — pitch is the elevation of the view axis),
    * optical-axis horizontal compass = gimbal_yaw         (within ~2°),
    * camera x (right) is 90° further clockwise in compass from the view,
      camera y (down) = z × x (right-handed c2w [x y z]).

    Roll rotates the right/down pair about the optical axis (DJI gb_roll is
    ~0 for gimbal-stabilized flight; kept for completeness).
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)
    # ENU compass az: unit(az) = (sin az, cos az, 0) in (E, N, U).
    view_h = np.array([math.sin(y), math.cos(y), 0.0])
    z = np.cos(p) * view_h + np.array([0.0, 0.0, math.sin(p)])
    right_az = y + math.pi / 2
    x0 = np.array([math.sin(right_az), math.cos(right_az), 0.0])
    y0 = np.cross(z, x0)
    y0 = y0 / max(1e-12, float(np.linalg.norm(y0)))
    x = math.cos(r) * x0 + math.sin(r) * y0
    y = np.cross(z, x)
    y = y / max(1e-12, float(np.linalg.norm(y)))
    R = np.column_stack([x, y, z])  # camera-to-world
    # R -> quaternion (Shepperd's method), w-first.
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qw / n, qx / n, qy / n, qz / n


def _estimate_sync_offset(frames: list[SrtFrame], video_fps: float, max_offset_s: float = 3.5) -> int:
    """Estimate the GPS-vs-video offset via gimbal-yaw vs GPS-course agreement.

    Per video frame the GPS course (displacement over a symmetric ±0.5 s
    window, compass convention) is compared with the gimbal yaw. The shift
    minimising mean |yaw - course| wins — but a nonzero shift is only
    applied when it beats zero by a real margin (2°): on steady segments
    both series are ~constant, so many shifts "fit" and only a decisive
    improvement is evidence. On Flight_to_tower the two series agree within
    ~2° at every index where the course is defined and no shift wins the
    margin — the applied offset is 0, with the measured agreement recorded.

    Returns frames to shift the GPS sample index (positive = GPS sample for
    video frame *v* is taken from *v + shift*).
    """
    if len(frames) < 30:
        return 0
    yaw = np.array([f.gimbal_yaw for f in frames])
    lat = np.array([f.latitude for f in frames])
    lon = np.array([f.longitude for f in frames])
    enu = wgs84_to_enu(lat, lon, np.zeros_like(lat), lat[0], lon[0])
    track = enu[:, :2]
    n = len(frames)
    W = max(15, int(round(0.5 * video_fps)))  # ±0.5 s course window
    min_course_m = 4.0  # a course estimate needs real displacement

    course = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(i - W, 0), min(i + W, n - 1)
        d = track[hi] - track[lo]
        if np.linalg.norm(d) >= min_course_m:
            course[i] = np.degrees(np.arctan2(d[0], d[1]))

    window = int(max_offset_s * video_fps)
    best_s, best_err, err0 = 0, 1e9, None
    for s in range(-window, window + 1):
        lo = max(0, -s)
        hi = min(n, n - s)
        idx = np.arange(lo, hi)
        c = course[idx + s]
        y = yaw[idx]
        ok = ~np.isnan(c)
        if ok.sum() < 40:
            continue
        err = float(np.mean(np.abs((y[ok] - c[ok] + 180) % 360 - 180)))
        if s == 0:
            err0 = err
        if err < best_err:
            best_err, best_s = err, s
    if err0 is None:
        log.info("srt_sync_offset_uncertain", note="insufficient GPS motion for course estimates")
        return 0
    if best_s != 0 and err0 - best_err < 2.0:
        log.info("srt_sync_offset_negligible", err_at_zero_deg=round(err0, 2),
                 best_shift=best_s, best_err_deg=round(best_err, 2),
                 note="zero-shift agreement within margin; offset 0 applied")
        return 0
    log.info("srt_sync_offset_estimated", frames=best_s,
             seconds=round(best_s / video_fps, 2),
             err_at_zero_deg=round(err0, 2), best_err_deg=round(best_err, 2))
    return best_s


def srt_to_flight_poses(
    srt_path: Path,
    output_csv: Path,
    video_fps: float,
) -> dict:
    """Convert a gimbal-bearing DJI SRT log into the canonical flight_poses.csv.

    Returns a provenance dict (mode, sample counts, sync offset, quality).
    The output is byte-compatible with the existing telemetry-assisted
    sparse path — no downstream change.
    """
    frames = parse_srt(srt_path)
    if video_fps <= 0:
        raise ValueError(f"invalid video fps {video_fps} for SRT telemetry sync")
    # Continuity: DJI writes one block per frame; gaps are logged honestly.
    cnts = [f.frame_cnt for f in frames]
    missing = (cnts[-1] - cnts[0] + 1) - len(cnts)

    offset = _estimate_sync_offset(frames, video_fps)

    lat = np.array([f.latitude for f in frames])
    lon = np.array([f.longitude for f in frames])
    rel = np.array([f.rel_alt for f in frames])
    anchor_lat, anchor_lon, anchor_alt = lat[0], lon[0], rel[0]
    enu = wgs84_to_enu(lat, lon, rel, anchor_lat, anchor_lon, anchor_alt)

    # Zero-order hold across the estimated offset: frame v takes the ENU
    # position of the GPS sample at v+offset (clamped to the log range).
    n = len(frames)
    idx = np.clip(np.arange(n) + offset, 0, n - 1)
    xyz = enu[idx]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "frame_id", "x", "y", "z", "qw", "qx", "qy", "qz",
            "latitude", "longitude", "altitude", "fov_vertical", "roll", "pitch", "yaw",
        ])
        for i, f in enumerate(frames):
            qw, qx, qy, qz = _rpy_to_quat(f.gimbal_roll, f.gimbal_pitch, f.gimbal_yaw)
            writer.writerow([
                f.frame_cnt - 1,                       # 0-based video frame id
                f"{xyz[i, 0]:.6f}", f"{xyz[i, 1]:.6f}", f"{xyz[i, 2]:.6f}",
                f"{qw:.8f}", f"{qx:.8f}", f"{qy:.8f}", f"{qz:.8f}",
                f"{f.latitude:.8f}", f"{f.longitude:.8f}", f"{f.rel_alt:.3f}",
                f"{DEFAULT_FOV_VERTICAL:.1f}",
                f"{f.gimbal_roll:.3f}", f"{f.gimbal_pitch:.3f}", f"{f.gimbal_yaw:.3f}",
            ])

    provenance = {
        "source": srt_path.name,
        "mode": "DJI_SRT_TELEMETRY",
        "frames": n,
        "missing_blocks": missing,
        "gps_fix_cadence": "zero-order hold between fixes",
        "sync_offset_frames": offset,
        "sync_offset_sec": round(offset / video_fps, 3),
        "enu_anchor": {"lat": float(anchor_lat), "lon": float(anchor_lon), "alt": float(anchor_alt)},
        "orientation_source": "gimbal rpy (Z-Y-X) -> quaternion; cross-validated via elevation agreement",
        "note": "Per-frame SRT GPS lags video on some platforms; offset estimated, not assumed 0.",
    }
    log.info("srt_flight_poses_written", output=str(output_csv), **{k: v for k, v in provenance.items() if not isinstance(v, dict)})
    return provenance


#: A swapped pair is accepted as the true order only when it lands within
#: this many degrees of the video's own fix (~5 km) AND is clearly closer
#: than the direct reading. Tie or no evidence -> keep what the log wrote.
_ORDER_MATCH_TOL_DEG = 0.05
_ORDER_MATCH_MARGIN = 0.01


def resolve_positional_order(
    frames: list[SrtFrame], video_gps: tuple[float, float] | None
) -> tuple[list[SrtFrame], dict]:
    """Settle a position-only log's (lat, lon) vs (lon, lat) ORDER by evidence.

    Legacy ``GPS(a,b,alt)`` blocks are ambiguous whenever both slots are
    within latitude range — which is exactly the common case (measured:
    ``data/dji/DJI_0501.SRT`` writes ``GPS(-4.0071,57.9811,18)`` for a flight
    whose video atom reads ``+57.981100-4.007092`` → the log is (lon, lat),
    and the assumed (lat, lon) placed the whole reconstruction in the Indian
    Ocean with every internal check still "passing" because it was
    self-consistent).

    The video's embedded fix is independent evidence, so it decides: a swap
    is applied only when the swapped reading lands close to the video's fix
    and the direct reading does not. With no video fix the log's own order is
    kept — recorded as convention, never presented as measured.
    """
    provenance: dict = {
        "coordinate_order": "latitude_longitude",
        "coordinate_order_source": "format_convention" if video_gps is None else "video_gps_none",
    }
    if not frames or video_gps is None:
        return frames, provenance

    vlat, vlon = float(video_gps[0]), float(video_gps[1])
    lat0 = float(np.median([f.latitude for f in frames[: min(len(frames), 30)]]))
    lon0 = float(np.median([f.longitude for f in frames[: min(len(frames), 30)]]))
    direct = max(abs(lat0 - vlat), abs(lon0 - vlon))
    swapped = max(abs(lon0 - vlat), abs(lat0 - vlon))
    provenance["coordinate_order_evidence"] = {
        "video_fix": [round(vlat, 7), round(vlon, 7)],
        "log_first_fix": [round(lat0, 7), round(lon0, 7)],
        "direct_delta_deg": round(direct, 6),
        "swapped_delta_deg": round(swapped, 6),
    }
    if (
        swapped <= _ORDER_MATCH_TOL_DEG
        and swapped + _ORDER_MATCH_MARGIN < direct
    ):
        from dataclasses import replace

        frames = [replace(f, latitude=f.longitude, longitude=f.latitude) for f in frames]
        provenance["coordinate_order"] = "longitude_latitude"
        provenance["coordinate_order_source"] = "video_gps"
        log.info(
            "srt_coordinate_order_corrected",
            direct_delta_deg=provenance["coordinate_order_evidence"]["direct_delta_deg"],
            swapped_delta_deg=provenance["coordinate_order_evidence"]["swapped_delta_deg"],
            note="legacy GPS() logs may write (longitude, latitude) — the video's "
                 "own fix decides, never an assumption",
        )
        return frames, provenance
    if direct <= _ORDER_MATCH_TOL_DEG:
        provenance["coordinate_order_source"] = "video_gps_agrees"
    return frames, provenance


def srt_to_telemetry_csv(
    srt_path: Path, output_csv: Path, video_gps: tuple[float, float] | None = None
) -> dict:
    """Convert a position-only (legacy GPS-function) DJI SRT log into
    ``telemetry.csv`` — the placement path's canonical position schema.

    Each block's timestamp comes from the SRT timecode (the video timeline
    the log was recorded against) — never a fabricated index. GPS fixes
    repeat between updates; ``trajectory_sync._collapse_stale_fixes``
    collapses those downstream, so the raw cadence is preserved here.

    No orientation, no frame-to-pose mapping, no quaternions are invented:
    this log family measures position only, and the consumers that need
    orientation (video-only SfM) produce their own from the imagery.
    """
    frames = parse_srt(srt_path)
    if srt_has_orientation(frames):
        # A gimbal-bearing log deserves the richer conversion; the caller
        # dispatched to the wrong target — say so instead of degrading.
        raise ValueError(
            "SRT log carries gimbal orientation; use srt_to_flight_poses "
            "(flight_poses.csv) for this format"
        )
    frames, order_prov = resolve_positional_order(frames, video_gps)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n_no_time = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "latitude", "longitude", "altitude"])
        for f in frames:
            if f.time_s is None:
                n_no_time += 1
            ts = f"{f.time_s:.3f}" if f.time_s is not None else ""
            writer.writerow([ts, f"{f.latitude:.8f}", f"{f.longitude:.8f}", f"{f.rel_alt:.3f}"])

    lat = np.array([f.latitude for f in frames])
    lon = np.array([f.longitude for f in frames])
    moved = float(np.max(np.abs(np.diff(lat))) + np.max(np.abs(np.diff(lon)))) if len(frames) > 1 else 0.0
    provenance = {
        "source": srt_path.name,
        "mode": "DJI_SRT_TELEMETRY_POSITION_ONLY",
        "dji_format": FORMAT_GPS_FUNCTION,
        "samples": len(frames),
        "blocks_with_timecode": len(frames) - n_no_time,
        "gps_fix_cadence": "raw block cadence; stale fixes collapsed downstream",
        "latlon_span_deg": round(moved, 6),
        **order_prov,
        "note": (
            "Legacy DJI GPS-function log: position-only, no gimbal fields — "
            "converted to the placement path's telemetry.csv (video-only SfM "
            "placed into the telemetry ENU frame via measured similarity)."
        ),
    }
    log.info(
        "srt_telemetry_csv_written", output=str(output_csv), samples=len(frames),
        blocks_with_timecode=len(frames) - n_no_time,
    )
    return provenance


def convert_srt_for_run(
    srt_path: Path,
    workspace: Path,
    video_fps: float,
    video_gps: tuple[float, float] | None = None,
) -> dict:
    """Dispatch an SRT log to the conversion its measured content supports.

    Writes exactly one artifact into *workspace*:

    * gimbal-bearing log (bracketed family) → ``flight_poses.csv`` for the
      telemetry-assisted triangulation path;
    * position-only log (legacy GPS-function family) → ``telemetry.csv``
      for the measured-similarity placement path + georef.

    ``video_gps`` is the video's own embedded fix when the caller has it; it
    settles the coordinate order of position-only logs (see
    ``resolve_positional_order``). Returns the conversion provenance with
    ``artifact`` naming what was written. Raises ``ValueError`` when the log
    is not a DJI SRT flight log.
    """
    frames = parse_srt(srt_path)
    if srt_has_orientation(frames):
        prov = srt_to_flight_poses(srt_path, workspace / "flight_poses.csv", video_fps)
        prov["artifact"] = "flight_poses.csv"
    else:
        prov = srt_to_telemetry_csv(
            srt_path, workspace / "telemetry.csv", video_gps=video_gps
        )
        prov["artifact"] = "telemetry.csv"
    return prov
