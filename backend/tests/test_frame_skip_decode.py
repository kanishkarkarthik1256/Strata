"""Skip-over contract for frame extraction.

Stage ``frames`` is the one whose cost tracks VIDEO LENGTH (every frame of
the file used to be decoded via ``cap.read()`` before being discarded), so
it now advances with ``cap.grab()`` — which does not decode — and calls
``cap.retrieve()`` only for the candidates that are actually kept. Since
``read() == grab() + retrieve()``, the kept frames are bit-identical.

That equivalence is the whole point of the change, so it is pinned here:
these tests fail if the skip drifts by even one frame, which is the only
way this optimisation can silently corrupt a reconstruction. The video is
per-frame random noise specifically so that an off-by-one is unmissable —
neighbouring frames of a real flight are similar, noise neighbours are not.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services.frame_extractor import _compute_extract_indices, extract_frames

FPS = 30.0
FRAME_COUNT = 120
WIDTH, HEIGHT = 160, 120


def _write_noise_video(path, frame_count: int = FRAME_COUNT) -> list[np.ndarray]:
    """Write a video whose every frame is a distinct, seed-derived pattern."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
    )
    assert writer.isOpened(), "could not open synthetic video writer"
    frames = []
    for i in range(frame_count):
        rng = np.random.default_rng(i)
        frame = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        frames.append(frame)
        writer.write(frame)
    writer.release()
    return frames


def _reference_frame(video_path, index: int) -> np.ndarray:
    """Independently decode frame ``index`` the slow, unambiguous way."""
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    frame = None
    for _ in range(index + 1):
        ok, frame = cap.read()
        assert ok, f"video ended before reference frame {index}"
    cap.release()
    return frame


def _saved_path(output_dir, entry):
    """Kept frames live in frames/ (copied to selected/); rejected move out."""
    for sub in ("frames", "selected", "rejected"):
        candidate = output_dir / sub / entry["filename"]
        if candidate.exists():
            return candidate
    raise AssertionError(f"no saved file for {entry['filename']}")


@pytest.fixture()
def noise_video(tmp_path):
    path = tmp_path / "noise.mp4"
    _write_noise_video(path)
    return path


class TestSkipOverExtraction:
    def test_interval_extraction_decodes_exactly_the_requested_frames(
        self, noise_video, tmp_path
    ):
        out = tmp_path / "out"
        out.mkdir()
        report = extract_frames(
            noise_video,
            out,
            extraction_mode="interval",
            interval_sec=1.0,
            quality_threshold=0.0,
        )

        expected = _compute_extract_indices(
            FRAME_COUNT, FPS, "interval", None, None, 1.0
        )
        got = [f["frame_num"] for f in report["frames"]]
        assert got == expected, (
            f"skip-over changed which frames are decoded: expected {expected}, "
            f"got {got}"
        )

        # Content must be the frame at that index — not its neighbour.
        for entry in report["frames"]:
            saved = cv2.imread(str(_saved_path(out, entry)))
            assert saved is not None
            truth = _reference_frame(noise_video, entry["frame_num"])
            diff = np.abs(saved.astype(np.int16) - truth.astype(np.int16)).mean()
            assert diff < 20, (
                f"{entry['filename']} (frame_num={entry['frame_num']}) is not the "
                f"frame at that index: mean abs diff {diff:.1f} vs its own "
                "neighbour (JPEG re-encode accounts for a few levels)"
            )

    def test_timestamp_tracks_the_frame_actually_decoded(self, noise_video, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        report = extract_frames(
            noise_video,
            out,
            extraction_mode="interval",
            interval_sec=1.0,
            quality_threshold=0.0,
        )
        for entry in report["frames"]:
            expected_ts = round(entry["frame_num"] / FPS, 3)
            assert entry["timestamp_sec"] == expected_ts

    def test_dense_extraction_is_unaffected(self, noise_video, tmp_path):
        """every_frame mode must still decode every frame, in order."""
        out = tmp_path / "out"
        out.mkdir()
        report = extract_frames(
            noise_video,
            out,
            extraction_mode="every_n",
            every_n=40,
            quality_threshold=0.0,
        )
        got = [f["frame_num"] for f in report["frames"]]
        assert got == [0, 40, 80]
        for entry in report["frames"]:
            saved = cv2.imread(str(_saved_path(out, entry)))
            truth = _reference_frame(noise_video, entry["frame_num"])
            diff = np.abs(saved.astype(np.int16) - truth.astype(np.int16)).mean()
            assert diff < 20
