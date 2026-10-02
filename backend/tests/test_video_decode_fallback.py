"""Video decode fallback — uploads must never die to a codec cv2 can't open.

The server's OpenCV build has no FFMPEG (macOS wheel) and its AVFoundation
backend refuses some uploaded encodings; before the fallback, validation
rejected such files and the upload handler deleted them (empty workspaces:
estrel_cd4476, Video_Mission_d887a0). These tests pin the contract:

1. A cv2-readable file passes through untouched (no transcode sidecar).
2. A cv2-rejected but ffmpeg-decodable file is transcoded once to a cached
   ``*_cv2dec.mp4`` sibling that cv2 CAN decode, frame count preserved, and
   the original upload still exists afterwards.
3. The sibling is reused (no second transcode) on the next call.
4. A genuinely corrupt file (not decodable even via ffmpeg) raises
   ``InvalidVideoError`` — and the file still exists so the user can
   re-download/inspect it rather than it vanishing silently.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.exceptions import InvalidVideoError
from app.services import video_validation as vv


def _write_stripe_pngs(directory: Path, n: int = 24) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((240, 320, 3), 40 + i, np.uint8)
        x = (i * 13) % 320
        img[:, x:x + 20] = 255
        cv2.imwrite(str(directory / f"f{i:03d}.png"), img)


@pytest.fixture(scope="module")
def ffmpeg_h264_video(tmp_path_factory) -> Path:
    """A valid H.264 mp4 built with the bundled ffmpeg binary."""
    import imageio_ffmpeg

    tmp = tmp_path_factory.mktemp("vidfix")
    _write_stripe_pngs(tmp / "pngs")
    out = tmp / "upload.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", "24",
         "-i", str(tmp / "pngs" / "f%03d.png"), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(out)],
        capture_output=True, timeout=180, check=True,
    )
    assert out.exists() and vv.cv2_opens(out)
    return out


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def test_validate_all_passes_cv2_readable_file_untouched(ffmpeg_h264_video: Path, tmp_path: Path):
    """Fast path: no transcode sibling may appear for a file cv2 opens."""
    work = tmp_path / "run.mp4"
    work.write_bytes(ffmpeg_h264_video.read_bytes())
    width, height, fps, frame_count, codec = vv.validate_all(work)
    assert (width, height) == (320, 240)
    assert fps == pytest.approx(24.0)
    assert frame_count == 24
    siblings = list(tmp_path.glob("*_cv2dec*"))
    assert siblings == [], "validate_all must not transcode a cv2-readable file"


def test_ensure_cv2_readable_transcodes_cv2_rejected_upload(
    ffmpeg_h264_video: Path, tmp_path: Path, monkeypatch
):
    """cv2-rejected + ffmpeg-decodable -> cached _cv2dec.mp4, source kept."""
    work = tmp_path / "upload.mp4"
    work.write_bytes(ffmpeg_h264_video.read_bytes())

    real_opens = vv.cv2_opens

    def fake_opens(path: Path) -> bool:
        # Simulate exactly the failing build: refuses the upload itself,
        # accepts anything produced downstream.
        return Path(path).resolve() != work.resolve() and real_opens(path)

    monkeypatch.setattr(vv, "cv2_opens", fake_opens)

    out = vv.ensure_cv2_readable(work)
    assert out != work
    assert out.name == "upload_cv2dec.mp4"
    assert out.exists() and real_opens(out)
    assert _count_frames(out) == 24
    # The user's upload survives — it is never the fallback that deletes it.
    assert work.exists() and work.stat().st_size > 0
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_ensure_cv2_readable_reuses_cached_sibling(
    ffmpeg_h264_video: Path, tmp_path: Path, monkeypatch
):
    work = tmp_path / "upload.mp4"
    work.write_bytes(ffmpeg_h264_video.read_bytes())
    real_opens = vv.cv2_opens

    def fake_opens(path: Path) -> bool:
        return Path(path).resolve() != work.resolve() and real_opens(path)

    monkeypatch.setattr(vv, "cv2_opens", fake_opens)
    first = vv.ensure_cv2_readable(work)
    stamp = first.stat().st_mtime_ns
    second = vv.ensure_cv2_readable(work)
    assert second == first
    assert second.stat().st_mtime_ns == stamp, "second call must reuse, not re-transcode"


def test_validate_all_raises_for_corrupt_file_and_preserves_it(tmp_path: Path):
    """Genuinely broken bytes must raise — but the file must survive for the user."""
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"\x00" * 4096)
    with pytest.raises(InvalidVideoError):
        vv.validate_all(corrupt)
    assert corrupt.exists(), "validation must not delete the user's upload"


def test_seek_to_index_fails_cleanly_for_corrupt_video(tmp_path: Path):
    from app.services.frame_extractor import seek_to_index

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"\x01" * 2048)
    out = tmp_path / "regen.jpg"
    assert seek_to_index(corrupt, 0, out) is False
    assert not out.exists()
