"""Video validation service.

Checks an uploaded video file for:
- Correct file extension
- Readable OpenCV container (corruption check)
- Non-zero duration
- Valid resolution
- Supported video codec

The server's OpenCV build may lack FFMPEG (macOS wheels ship without it), so
H.264/HEVC uploads can be unreadable to cv2 while perfectly decodable by the
ffmpeg binary bundled with ``imageio_ffmpeg``. ``ensure_cv2_readable`` owns
that decision once for every consumer (validation, metadata, frame
extraction, per-frame healing): cv2 opens it → use as-is; otherwise transcode
once to a cv2-readable sibling and return that path. Only a file ffmpeg also
cannot decode is declared corrupt — so a decodable upload is never deleted.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2

from app.config.settings import settings
from app.exceptions import InvalidVideoError
from app.logging_config import get_logger

log = get_logger("drone_recon.services.validation")

# Map of OpenCV codec fourcc integers → human-readable names
_KNOWN_CODECS: dict[int, str] = {
    cv2.VideoWriter_fourcc(*"mp4v"): "mp4v",
    cv2.VideoWriter_fourcc(*"avc1"): "avc1 (H.264)",
    cv2.VideoWriter_fourcc(*"h264"): "h264",
    cv2.VideoWriter_fourcc(*"hevc"): "hevc (H.265)",
    cv2.VideoWriter_fourcc(*"XVID"): "XVID",
    cv2.VideoWriter_fourcc(*"MJPG"): "MJPG",
    cv2.VideoWriter_fourcc(*"VP80"): "VP8",
    cv2.VideoWriter_fourcc(*"VP90"): "VP9",
    cv2.VideoWriter_fourcc(*"X264"): "x264",
}


def validate_extension(filepath: Path) -> None:
    """Reject files whose extension is not in the supported list."""
    ext = filepath.suffix.lower().lstrip(".")
    supported = settings.processing.supported_video_formats
    if ext not in supported:
        raise InvalidVideoError(
            f"Unsupported video format '.{ext}'. "
            f"Supported: {', '.join(supported)}"
        )


def cv2_opens(filepath: Path) -> bool:
    """True when this OpenCV build can open and would decode the file."""
    cap = cv2.VideoCapture(str(filepath))
    ok = cap.isOpened()
    cap.release()
    return ok


def _ffmpeg_binary() -> str | None:
    """Bundled imageio-ffmpeg binary, then PATH ffmpeg, else None."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - import/lookup environment-specific
        return shutil.which("ffmpeg")


def ensure_cv2_readable(video_path: Path) -> Path:
    """Return a path this OpenCV build can decode, transcoding once if needed.

    Fast path: cv2 opens the file → the same path comes back untouched.
    Fallback: transcode the video stream with the bundled ffmpeg binary to a
    sibling ``<stem>_cv2dec.mp4`` (frame-exact: ``-fps_mode passthrough``, so
    frame indices and timestamps stay valid for downstream consumers) and
    return that. The transcode is cached — an existing readable sibling is
    reused. Raises ``InvalidVideoError`` only when even ffmpeg cannot produce
    a decodable stream, i.e. the upload is genuinely corrupt.
    """
    if cv2_opens(video_path):
        return video_path
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    ok = cap.isOpened()
    cap.release()
    if ok:
        return video_path

    out = video_path.with_name(f"{video_path.stem}_cv2dec.mp4")
    if out.exists() and cv2_opens(out):
        log.info("video_transcode_reused", transcoded=out.name)
        return out

    ff = _ffmpeg_binary()
    if ff is None:
        raise InvalidVideoError(
            f"Cannot open video file with OpenCV and no ffmpeg fallback is "
            f"available: {video_path.name}"
        )

    tmp = out.with_name(out.stem + ".tmp.mp4")
    cmd = [
        ff, "-y", "-i", str(video_path),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
        str(tmp),
    ]
    log.info("video_cv2_unreadable_transcoding", source=video_path.name)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        tmp.unlink(missing_ok=True)
        raise InvalidVideoError(
            f"Video transcode timed out — file may be pathological: {video_path.name}"
        ) from exc
    if proc.returncode != 0 or not tmp.exists() or not cv2_opens(tmp):
        tmp.unlink(missing_ok=True)
        tail = (proc.stderr or "")[-400:]
        log.warning("video_transcode_failed", source=video_path.name, stderr_tail=tail)
        raise InvalidVideoError(
            f"Cannot open video file — the file may be corrupted or empty: "
            f"{video_path.name} (decode failed even via ffmpeg)"
        )
    tmp.replace(out)
    log.info("video_transcoded_for_cv2", transcoded=out.name)
    return out


def validate_readable(filepath: Path) -> tuple[int, int, float, int, str]:
    """Open the video with OpenCV and return basic properties.

    Returns (width, height, fps, frame_count, codec_name).

    Raises ``InvalidVideoError`` if the file is corrupt, zero-length,
    or has an unreadable container even after the ffmpeg fallback.
    """
    filepath = ensure_cv2_readable(filepath)
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        raise InvalidVideoError(
            f"Cannot open video file — the file may be corrupted or empty: {filepath.name}"
        )

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_name = _decode_fourcc(fourcc)

        # Basic sanity checks
        if width <= 0 or height <= 0:
            raise InvalidVideoError(
                f"Invalid video resolution: {width}x{height}"
            )

        if fps <= 0:
            raise InvalidVideoError(
                f"Invalid video FPS: {fps}"
            )

        if frame_count <= 0:
            raise InvalidVideoError(
                "Video has zero frames — the file may be empty or corrupt"
            )

        # Attempt to read the first frame to confirm the container is intact
        ret, frame = cap.read()
        if not ret or frame is None:
            raise InvalidVideoError(
                "Cannot read any frames from the video — the file may be corrupt"
            )

        return width, height, fps, frame_count, codec_name

    finally:
        cap.release()


def validate_resolution(width: int, height: int) -> None:
    """Warn if the resolution is outside commonly supported ranges.

    Very small videos (< 320px) are unlikely to produce useful 3D reconstructions.
    """
    if width < 320 or height < 240:
        raise InvalidVideoError(
            f"Resolution {width}x{height} is too small. "
            "Minimum supported is 320x240."
        )


def validate_all(filepath: Path) -> tuple[int, int, float, int, str]:
    """Run every validation step and return basic properties.

    This is the single entry-point the upload service should call.

    Returns (width, height, fps, frame_count, codec_name).
    """
    log.info("validating_video", filename=filepath.name, size_bytes=filepath.stat().st_size)

    validate_extension(filepath)
    width, height, fps, frame_count, codec_name = validate_readable(filepath)
    validate_resolution(width, height)

    log.info(
        "video_valid",
        filename=filepath.name,
        width=width,
        height=height,
        fps=round(fps, 2),
        frame_count=frame_count,
        codec=codec_name,
    )
    return width, height, fps, frame_count, codec_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_fourcc(fourcc_int: int) -> str:
    """Convert an OpenCV fourcc integer to a readable string."""
    chars = [
        chr((fourcc_int >> 0) & 0xFF),
        chr((fourcc_int >> 8) & 0xFF),
        chr((fourcc_int >> 16) & 0xFF),
        chr((fourcc_int >> 24) & 0xFF),
    ]
    return "".join(c if c.isprintable() else "?" for c in chars)
