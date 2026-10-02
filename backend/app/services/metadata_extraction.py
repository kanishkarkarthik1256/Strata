"""Extract rich metadata from a video file.

Uses OpenCV for basic properties (resolution, fps, codec, frame count) and
ffprobe for detailed metadata (bitrate, duration, GPS, camera info).
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from app.logging_config import get_logger
from app.schemas.upload import VideoMetadata
from app.services.video_validation import ensure_cv2_readable

log = get_logger("drone_recon.services.metadata")


def extract_metadata(filepath: Path) -> VideoMetadata:
    """Extract full metadata from a video file.

    Tries ffprobe first for rich metadata, falls back to OpenCV-only
    extraction if ffprobe is unavailable.
    """
    log.info("extracting_metadata", filename=filepath.name)

    # Try ffprobe for rich metadata (raw file — codec/GPS tags live there)
    ffprobe_data = _run_ffprobe(filepath)

    # Basic props from OpenCV; ensure_cv2_readable transcodes once if this
    # OpenCV build cannot decode the container (FFMPEG-less build).
    filepath = ensure_cv2_readable(filepath)
    cap = cv2.VideoCapture(str(filepath))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = _decode_fourcc(fourcc)
    finally:
        cap.release()

    file_size = filepath.stat().st_size

    # Merge ffprobe data if available
    if ffprobe_data:
        duration = ffprobe_data.get("duration_sec", 0.0)
        bitrate = ffprobe_data.get("bitrate_kbps", 0.0)
        creation_time = ffprobe_data.get("creation_time")
        gps_lat = ffprobe_data.get("gps_lat")
        gps_lon = ffprobe_data.get("gps_lon")
        gps_alt = ffprobe_data.get("gps_alt")
        camera_make = ffprobe_data.get("camera_make")
        camera_model = ffprobe_data.get("camera_model")
        # ffprobe may give a more accurate codec name
        if ffprobe_data.get("codec"):
            codec = ffprobe_data["codec"]
    else:
        # Fallback: compute duration from frame count / fps
        duration = frame_count / fps if fps > 0 else 0.0
        bitrate = _estimate_bitrate(filepath, duration)
        creation_time = None
        gps_lat = gps_lon = gps_alt = None
        camera_make = camera_model = None

    # Last resort for the video's own fix: read the container's ISO 6709 atom
    # directly. ffprobe is not always installed (absent in this environment),
    # and the fix is real evidence the pipeline uses to check SRT coordinate
    # order — a guess would be worse than a null here.
    if gps_lat is None or gps_lon is None:
        container_gps = read_container_gps(filepath)
        if container_gps is not None:
            gps_lat = container_gps["lat"]
            gps_lon = container_gps["lon"]
            if gps_alt is None:
                gps_alt = container_gps["alt"]

    metadata = VideoMetadata(
        filename=filepath.name,
        duration_sec=round(duration, 3),
        fps=round(fps, 2),
        width=width,
        height=height,
        codec=codec,
        frame_count=frame_count,
        bitrate_kbps=round(bitrate, 1),
        file_size_bytes=file_size,
        creation_time=creation_time,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        gps_alt=gps_alt,
        camera_make=camera_make,
        camera_model=camera_model,
    )

    log.info(
        "metadata_extracted",
        filename=filepath.name,
        duration=duration,
        resolution=f"{width}x{height}",
        fps=round(fps, 2),
        codec=codec,
        has_gps=gps_lat is not None,
    )

    return metadata


def _run_ffprobe(filepath: Path) -> Optional[dict]:
    """Run ffprobe and parse structured output. Returns None if unavailable."""
    from app.services.resources import ffprobe_bin

    probe_bin = ffprobe_bin()
    if probe_bin is None:
        # The imageio-ffmpeg wheel ships ffmpeg without ffprobe; the ffmpeg
        # banner still carries duration/bitrate/creation metadata.
        bridge = _probe_via_ffmpeg(filepath)
        if bridge is not None:
            return bridge
        log.info("ffprobe_not_found", note="falling back to OpenCV-only metadata")
        return None
    try:
        result = subprocess.run(
            [
                probe_bin,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning("ffprobe_failed", returncode=result.returncode)
            return None

        data = json.loads(result.stdout)
        return _parse_ffprobe(data)
    except FileNotFoundError:
        log.info("ffprobe_not_found", note="falling back to OpenCV-only metadata")
        return None
    except (json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        log.warning("ffprobe_parse_error", error=str(exc))
        return None


def _probe_via_ffmpeg(filepath: Path) -> Optional[dict]:
    """Extract basic metadata via ``ffmpeg -i`` stderr when ffprobe is absent.

    Produces the same flat keys as :func:`_parse_ffprobe` for the fields the
    banner exposes (duration, bitrate, creation_time); returns None when no
    ffmpeg binary exists at all.
    """
    from app.services.resources import ffmpeg_bin

    exe = ffmpeg_bin()
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, "-hide_banner", "-i", str(filepath)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text_out = result.stderr or ""
    info: dict = {}

    import re as _re

    m = _re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text_out)
    if m:
        h, mn, s = m.groups()
        info["duration_sec"] = int(h) * 3600 + int(mn) * 60 + float(s)
    m = _re.search(r"bitrate: (\d+) kb/s", text_out)
    if m:
        info["bitrate_kbps"] = float(m.group(1))
    m = _re.search(r"creation_time\s*:\s*([^\n,]+)", text_out)
    if m:
        try:
            info["creation_time"] = datetime.fromisoformat(
                m.group(1).strip().replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            pass
    m = _re.search(r"Video: (\w+)[,\s]", text_out)
    if m:
        info["codec"] = m.group(1).lower()
    return info or None


def _parse_ffprobe(data: dict) -> dict:
    """Parse ffprobe JSON output into a flat metadata dict."""
    result: dict = {}

    fmt = data.get("format", {})
    result["duration_sec"] = float(fmt.get("duration", 0))
    result["bitrate_kbps"] = float(fmt.get("bit_rate", 0)) / 1000

    # Creation time from format tags
    tags = fmt.get("tags", {})
    ct = tags.get("creation_time")
    if ct:
        try:
            result["creation_time"] = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

    # GPS from format tags (common drone formats: DJI, GoPro)
    _extract_gps_from_tags(tags, result)

    # Camera info from format tags
    result["camera_make"] = tags.get("com.apple.quicktime.make") or tags.get("make")
    result["camera_model"] = tags.get("com.apple.quicktime.model") or tags.get("model")

    # Video stream codec
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            result["codec"] = stream.get("codec_name", "")
            # Some drone videos put GPS in stream tags
            stream_tags = stream.get("tags", {})
            _extract_gps_from_tags(stream_tags, result)
            if not result.get("camera_make"):
                result["camera_make"] = stream_tags.get("make")
            if not result.get("camera_model"):
                result["camera_model"] = stream_tags.get("model")
            break

    return result


#: ISO 6709 location string, e.g. ``+57.981100-4.007092+57.200/`` — the
#: form DJI and iPhones write into the QuickTime ``©xyz`` atom. Latitude is
#: signed and 2-3 digits, longitude signed and 3 digits, altitude optional.
_ISO6709_RE = re.compile(
    r"(?P<lat>[+-]\d{1,2}(?:\.\d+)?)(?P<lon>[+-]\d{1,3}(?:\.\d+)?)"
    r"(?P<alt>[+-]\d+(?:\.\d+)?)?"
)


def parse_iso6709(value: str) -> Optional[dict]:
    """Parse an ISO 6709 location string into lat/lon/alt.

    ``+57.981100-4.007092+57.200/``, ``+57.9811-004.0071/`` and
    ``+57.9811-4.00709/`` are all valid renderings of the same fix. Returns
    ``None`` when the string carries no coordinate pair — never a guessed one.
    """
    if not value:
        return None
    text = value.strip().strip("/").replace(" ", "")
    m = _ISO6709_RE.match(text)
    if m is None:
        return None
    lat = _safe_float(m.group("lat"))
    lon = _safe_float(m.group("lon"))
    if lat is None or lon is None or abs(lat) > 90.0 or abs(lon) > 180.0:
        return None
    return {"lat": lat, "lon": lon, "alt": _safe_float(m.group("alt"))}


def read_container_gps(filepath: Path) -> Optional[dict]:
    """Read the container's embedded ISO 6709 fix (QuickTime/DJI ``©xyz``).

    The atom lives in ``moov/udta`` — at the END of a DJI MP4 (verified:
    ``data/dji/DJI_0501.MP4``, last 60 KB of 62 MB) — so the head and tail of
    the file are scanned without decoding the video. This exists because
    ``ffprobe`` is not always installed (it is absent in this environment),
    and the SRT adapter needs the video's own fix as independent evidence for
    the coordinate ORDER of position-only logs (see
    ``dji_srt_telemetry.resolve_positional_order``).
    """
    try:
        size = filepath.stat().st_size
        with open(filepath, "rb") as fh:
            head = fh.read(min(size, 256 * 1024))
            if size > 256 * 1024:
                fh.seek(max(0, size - 2 * 1024 * 1024))
                tail = fh.read()
            else:
                tail = b""
    except OSError:
        return None
    for chunk in (head, tail):
        for marker in (b"\xa9xyz", b"xyz"):
            idx = chunk.find(marker)
            if idx < 0:
                continue
            # The location string follows the atom's 12-byte header
            # (size+type+locale); scan a short window for the ISO 6709 text.
            window = chunk[idx: idx + 96].decode("latin-1", errors="ignore")
            m = re.search(r"[+-]\d{1,2}(?:\.\d+)?[+-]\d{1,3}(?:\.\d+)?[+-]?\d*(?:\.\d+)?", window)
            if m is None:
                continue
            parsed = parse_iso6709(m.group(0))
            if parsed is not None:
                return parsed
    return None


def _extract_gps_from_tags(tags: dict, result: dict) -> None:
    """Extract GPS coordinates from various metadata tag formats."""
    # DJI format: "GPSLatitude=28.6139;GPSLongitude=77.2090;GPSAltitude=100.0"
    location = tags.get("location")
    if location and isinstance(location, str):
        coords = {}
        for part in location.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                coords[k.strip()] = v.strip()
        result.setdefault("gps_lat", _safe_float(coords.get("GPSLatitude")))
        result.setdefault("gps_lon", _safe_float(coords.get("GPSLongitude")))
        result.setdefault("gps_alt", _safe_float(coords.get("GPSAltitude")))

    # QuickTime ISO 6709 (tag key varies by muxer; the value form does not).
    for key in (
        "com.apple.quicktime.location.ISO6709",
        "com.apple.quicktime.location.iso6709",
        "location-eng",
        "location",
    ):
        raw = tags.get(key)
        if isinstance(raw, str):
            iso = parse_iso6709(raw)
            if iso is not None:
                result.setdefault("gps_lat", iso["lat"])
                result.setdefault("gps_lon", iso["lon"])
                if iso["alt"] is not None:
                    result.setdefault("gps_alt", iso["alt"])

    # QuickTime format: separate tags
    result.setdefault("gps_lat", _safe_float(tags.get("com.apple.quicktime.latitude")))
    result.setdefault("gps_lon", _safe_float(tags.get("com.apple.quicktime.longitude")))
    result.setdefault("gps_alt", _safe_float(tags.get("com.apple.quicktime.altitude")))

    # Numeric string GPS (some FFmpeg-tagged files)
    result.setdefault("gps_lat", _safe_float(tags.get("GPSLatitude")))
    result.setdefault("gps_lon", _safe_float(tags.get("GPSLongitude")))
    result.setdefault("gps_alt", _safe_float(tags.get("GPSAltitude")))


def _estimate_bitrate(filepath: Path, duration: float) -> float:
    """Estimate bitrate in kbps from file size and duration."""
    if duration <= 0:
        return 0.0
    size_bytes = filepath.stat().st_size
    return (size_bytes * 8) / (duration * 1000)


def _safe_float(value: Optional[str]) -> Optional[float]:
    """Convert a string to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _decode_fourcc(fourcc_int: int) -> str:
    """Convert an OpenCV fourcc integer to a readable string."""
    chars = [
        chr((fourcc_int >> 0) & 0xFF),
        chr((fourcc_int >> 8) & 0xFF),
        chr((fourcc_int >> 16) & 0xFF),
        chr((fourcc_int >> 24) & 0xFF),
    ]
    return "".join(c if c.isprintable() else "?" for c in chars)
