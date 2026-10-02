"""Geospatial alignment and GPS quality analysis.

Pure-numpy WGS84 math (no external deps for the core path):

* geodetic (lat, lon, alt) → ECEF (exact WGS84 ellipsoid formulas)
* ECEF → local ENU offsets around a WGS84 anchor
* least-squares similarity (Umeyama) aligning a reconstruction (world
  meters) to ENU tie points — the standard "GPS-aligned" registration
* GPS quality report: drift, discontinuities, altitude consistency,
  smoothness, and a 0-100 GPS quality score

UTM / arbitrary EPSG output is available through ``pyproj`` when installed
(optional); without it the module still produces local ENU coordinates plus
full WGS84/ECEF values, which are stored in the CRS metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.logging_config import get_logger

log = get_logger("drone_recon.services.georeferencing")

# WGS84 ellipsoid constants
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)

_GRADES = [(85, "Excellent"), (70, "Good"), (50, "Fair"), (0, "Poor")]


def validate_gps_point(lat: float, lon: float, alt: float = 0.0) -> bool:
    """Strict validation for WGS84 coordinates. Rejects (0, 0), NaN, Inf, and out-of-bounds."""
    if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt)):
        return False
    if abs(lat) < 1e-6 and abs(lon) < 1e-6:
        return False  # Reject (0, 0) default fallback
    if lat < -90.0 or lat > 90.0:
        return False
    if lon < -180.0 or lon > 180.0:
        return False
    return True


def filter_valid_gps_points(points: list[dict]) -> list[dict]:
    """Filter list of dicts with 'lat', 'lon', 'alt' to valid WGS84 coordinates."""
    valid = []
    for p in points:
        lat = float(p.get("lat", 0.0))
        lon = float(p.get("lon", 0.0))
        alt = float(p.get("alt", 0.0))
        if validate_gps_point(lat, lon, alt):
            valid.append(p)
    return valid



# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


def geodetic_to_ecef(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:
    """WGS84 geodetic → ECEF (returns (N, 3) meters)."""
    lat_r = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon_r = np.deg2rad(np.asarray(lon, dtype=np.float64))
    h = np.asarray(alt, dtype=np.float64)
    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    x = (n + h) * cos_lat * np.cos(lon_r)
    y = (n + h) * cos_lat * np.sin(lon_r)
    z = (n * (1.0 - _WGS84_E2) + h) * sin_lat
    return np.column_stack([x, y, z])


def enu_rotation(lat0: float, lon0: float) -> np.ndarray:
    """Rotation matrix mapping ECEF delta vectors to local ENU at the anchor.

    ENU frame: X east, Y north, Z up. Rows are the basis vectors of ENU
    expressed in ECEF coordinates.
    """
    lat_r = np.deg2rad(lat0)
    lon_r = np.deg2rad(lon0)
    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    sin_lon, cos_lon = np.sin(lon_r), np.cos(lon_r)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return np.vstack([east, north, up])


def ecef_to_enu(xyz_ecef: np.ndarray, anchor_ecef: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Convert ECEF points to ENU offsets (meters) around the anchor."""
    delta = np.asarray(xyz_ecef, dtype=np.float64) - np.asarray(anchor_ecef, dtype=np.float64)
    return delta @ enu_rotation(lat0, lon0).T


def wgs84_to_enu(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float = 0.0,
) -> np.ndarray:
    """Convert WGS84 track points to local ENU around a (lat0, lon0, alt0) anchor."""
    ecef = geodetic_to_ecef(lat, lon, alt)
    anchor = geodetic_to_ecef(np.array([lat0]), np.array([lon0]), np.array([alt0]))[0]
    return ecef_to_enu(ecef, anchor, lat0, lon0)


def ecef_to_geodetic(xyz_ecef: np.ndarray) -> np.ndarray:
    """ECEF → WGS84 geodetic (Bowring 1985, one refinement step).

    Returns (N, 3) as ``[lat_deg, lon_deg, alt_m]``. Millimetre-accurate for
    terrestrial points, which is far below any survey uncertainty here.
    """
    xyz = np.asarray(xyz_ecef, dtype=np.float64).reshape(-1, 3)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a = _WGS84_A
    e2 = _WGS84_E2
    b = a * np.sqrt(1.0 - e2)
    ep2 = (a * a - b * b) / (b * b)
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    # Bowring's parametric latitude seed, then one standard refinement.
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3)
    for _ in range(2):
        sin_lat = np.sin(lat)
        n = a / np.sqrt(1.0 - e2 * sin_lat**2)
        alt = p / np.maximum(np.cos(lat), 1e-12) - n
        lat = np.arctan2(z + e2 * n * sin_lat, p)
    sin_lat = np.sin(lat)
    n = a / np.sqrt(1.0 - e2 * sin_lat**2)
    alt = np.where(p > 1e-9, p / np.maximum(np.cos(lat), 1e-12) - n, np.abs(z) - b)
    return np.column_stack([np.rad2deg(lat), np.rad2deg(lon), alt])


def enu_to_wgs84(
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float = 0.0,
) -> np.ndarray:
    """Local ENU offsets → WGS84 (exact inverse of :func:`wgs84_to_enu`).

    Returns (N, 3) as ``[lat_deg, lon_deg, alt_m]``. Used to place a
    reconstruction's footprint back onto a map: the ENU→ECEF rotation is the
    transpose of the forward one, so the round trip is lossless to float
    precision rather than an approximation.
    """
    enu = np.column_stack([np.asarray(east, dtype=np.float64).ravel(),
                           np.asarray(north, dtype=np.float64).ravel(),
                           np.asarray(up, dtype=np.float64).ravel()])
    anchor = geodetic_to_ecef(np.array([lat0]), np.array([lon0]), np.array([alt0]))[0]
    ecef = enu @ enu_rotation(lat0, lon0) + anchor
    return ecef_to_geodetic(ecef)


def project_to_utm(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, epsg: int | None = None):
    """Project WGS84 to a projected CRS via pyproj (optional dependency).

    With ``epsg=None`` an UTM zone is chosen per point set by its longitude.
    Raises a clear error when pyproj is not installed.
    """
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError(
            "Projected coordinates (UTM/EPSG) require the optional 'pyproj' package — "
            "install it with 'pip install pyproj' and retry. Local ENU/WGS84 output "
            "is available without it."
        ) from exc
    if epsg is None:
        zone = int(np.floor((float(np.mean(lon)) + 180.0) / 6.0) % 60) + 1
        epsg = 32600 + zone if float(np.mean(lat)) >= 0 else 32700 + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = transformer.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(xs), np.asarray(ys), np.asarray(alt), epsg


# ---------------------------------------------------------------------------
# Similarity alignment (Umeyama) — world meters → ENU
# ---------------------------------------------------------------------------


def align_to_enu(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares similarity (Umeyama) mapping *source* onto *target*.

    Returns (transform 4x4, scale). With fewer than 2 correspondences the
    transform is a translation by the first pair (rigid scale 1).
    """
    s = np.asarray(source, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if len(s) == 0 or len(t) != len(s):
        raise ValueError("align_to_enu needs equal, non-empty correspondence sets")

    if len(s) == 1:
        m = np.eye(4)
        m[:3, 3] = t[0] - s[0]
        return m, 1.0

    mu_s = s.mean(axis=0)
    mu_t = t.mean(axis=0)
    s_c = s - mu_s
    t_c = t - mu_t
    var_s = float(np.mean(np.einsum("ij,ij->i", s_c, s_c)))
    if var_s == 0:
        m = np.eye(4)
        m[:3, 3] = mu_t - mu_s
        return m, 1.0

    cov = (t_c.T @ s_c) / len(s)
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    diag = np.array([1.0, 1.0, d])
    r = u @ np.diag(diag) @ vt
    scale = float(np.trace(np.diag(diag) @ vt @ cov.T @ u) / var_s)
    trans = mu_t - scale * r @ mu_s

    m = np.eye(4)
    m[:3, :3] = scale * r
    m[:3, 3] = trans
    return m, scale


# ---------------------------------------------------------------------------
# GPS quality analysis
# ---------------------------------------------------------------------------


@dataclass
class GpsQuality:
    """GPS track quality metrics and report."""

    points: int = 0
    path_length_m: float = 0.0
    mean_speed_m_s: float = 0.0
    max_speed_m_s: float = 0.0
    altitude_std_m: float = 0.0
    discontinuities: int = 0
    drift_m: float = 0.0
    smoothness: float = 1.0
    gps_score: float = 100.0
    grade: str = "Excellent"
    suggestions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "points": self.points,
            "path_length_m": round(self.path_length_m, 2),
            "mean_speed_m_s": round(self.mean_speed_m_s, 2),
            "max_speed_m_s": round(self.max_speed_m_s, 2),
            "altitude_std_m": round(self.altitude_std_m, 2),
            "discontinuities": self.discontinuities,
            "drift_m": round(self.drift_m, 2),
            "smoothness": round(self.smoothness, 4),
            "gps_score": round(self.gps_score, 2),
            "grade": self.grade,
            "suggestions": self.suggestions,
        }


def analyze_gps_track(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    timestamps: np.ndarray | None = None,
    jump_threshold_m: float = 30.0,
) -> GpsQuality:
    """Analyze a WGS84 GPS track for drift, jumps, and altitude problems."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)
    n = len(lat)
    if n < 2:
        return GpsQuality(points=n, suggestions=["Not enough GPS points for analysis (need >= 2)"])

    enu = wgs84_to_enu(lat, lon, alt, float(lat[0]), float(lon[0]), float(alt[0]))
    steps = np.linalg.norm(np.diff(enu, axis=0), axis=1)
    path = float(steps.sum())
    ts = timestamps if timestamps is not None else np.arange(n, dtype=np.float64)
    dt = np.diff(ts)
    dt_safe = np.where(dt > 0, dt, 1.0)
    speeds = steps / dt_safe

    # Discontinuities: isolated segments far longer than the local median step.
    median_step = float(np.median(steps)) if n > 2 else 0.0
    threshold = max(jump_threshold_m, 5.0 * median_step)
    jumps = steps > threshold
    discontinuities = int(jumps.sum())

    # Altitude consistency: residual std after removing a linear trend.
    if n >= 3:
        coeffs = np.polyfit(np.arange(n), alt, 1)
        altitude_resid = alt - np.polyval(coeffs, np.arange(n))
    else:
        altitude_resid = alt - alt.mean()
    alt_std = float(np.sqrt(np.mean(altitude_resid**2)))

    # Drift: deviation of the track from the straight line between endpoints
    # (loops/zig-zags inflate this), measured in meters.
    if n >= 3 and path > 0:
        seg = enu[-1] - enu[0]
        seg_len = np.linalg.norm(seg)
        if seg_len > 1e-6:
            proj = ((enu - enu[0]) @ seg) / (seg_len**2)
            proj = np.clip(proj, 0.0, 1.0)
            line = enu[0] + proj[:, None] * seg
            drift = float(np.mean(np.linalg.norm(enu - line, axis=1)))
        else:
            drift = float(np.sqrt(np.mean(np.einsum("ij,ij->i", enu - enu[0], enu - enu[0]))))
    else:
        drift = 0.0

    # Smoothness: 1 - normalized mean of the (speed) second differences.
    if n >= 4:
        accel = np.abs(np.diff(speeds, n=2))
        smoothness = float(np.clip(1.0 - accel.mean() / max(speeds.mean() + 1e-6, 1e-6), 0.0, 1.0))
    else:
        smoothness = 1.0

    drift_score = float(np.clip(1.0 - drift / max(path * 0.2, 1.0), 0.0, 1.0))
    jump_score = float(np.clip(1.0 - discontinuities / max(n / 5.0, 1.0), 0.0, 1.0))
    alt_score = float(np.clip(1.0 - alt_std / 20.0, 0.0, 1.0))
    score = 100.0 * (0.4 * drift_score + 0.3 * jump_score + 0.3 * alt_score)
    grade = next(g for cutoff, g in _GRADES if score >= cutoff)

    suggestions = []
    if discontinuities:
        suggestions.append(f"{discontinuities} GPS jump(s) detected — check for signal loss or RTK gaps")
    if drift > path * 0.1:
        suggestions.append("Track drifts from a straight line — consider RTK/PPK corrections")
    if alt_std > 10:
        suggestions.append("Altitude is inconsistent — check barometer/altimeter calibration")
    if n < 8:
        suggestions.append("Few GPS samples — increase telemetry logging rate")

    log.info("gps_quality_analyzed", points=n, score=round(score, 2), grade=grade)
    return GpsQuality(
        points=n,
        path_length_m=path,
        mean_speed_m_s=float(np.mean(speeds)) if len(speeds) else 0.0,
        max_speed_m_s=float(np.max(speeds)) if len(speeds) else 0.0,
        altitude_std_m=alt_std,
        discontinuities=discontinuities,
        drift_m=drift,
        smoothness=smoothness,
        gps_score=float(score),
        grade=grade,
        suggestions=suggestions,
    )


def crs_metadata(lat0: float, lon0: float, alt0: float = 0.0, epsg: int | None = None) -> dict:
    """CRS description for reconstruction outputs."""
    return {
        "horizontal_crs": f"EPSG:{epsg}" if epsg else "local_enu",
        "anchor_wgs84": {"lat": round(lat0, 7), "lon": round(lon0, 7), "alt": round(alt0, 3)},
        "units": "meters",
        "note": "ENU is anchored at the first GPS fix; projected CRS available via pyproj",
    }
