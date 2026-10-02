"""Tests for the frame extraction pipeline.

Unit tests for each detector, plus integration test with synthetic video.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from httpx import AsyncClient

from app.services.blur_detector import blur_score, is_blurry
from app.services.frame_extractor import seek_to_index
from app.services.duplicate_detector import (
    compute_phash,
    hamming_distance,
    is_consecutive_duplicate,
    is_duplicate_of_kept,
    ssim_score,
)
from app.services.quality_analyzer import analyze_frame
from app.services.frame_extractor import extract_frames


# ---------------------------------------------------------------------------
# Blur detector tests
# ---------------------------------------------------------------------------


class TestBlurDetector:
    def test_sharp_frame_has_high_score(self):
        """A sharp, detailed frame should have a high Laplacian variance."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert blur_score(frame) > 0

    def test_blurry_frame_has_low_score(self):
        """A heavily blurred frame should have a low score."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        blurry = cv2.GaussianBlur(frame, (51, 51), 0)
        assert blur_score(blurry) < blur_score(frame)

    def test_is_blurry_respects_threshold(self):
        """is_blurry should use the threshold correctly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # flat = blurry
        assert is_blurry(frame, threshold=100) is True
        assert is_blurry(frame, threshold=0.0) is False


# ---------------------------------------------------------------------------
# Duplicate detector tests
# ---------------------------------------------------------------------------


class TestDuplicateDetector:
    def test_moving_frame_is_never_a_consecutive_duplicate(self):
        """Field failure (London_Mission_ca3c5b): at 59.94 fps candidate cadence,
        SSIM between consecutive candidates reaches ~0.86 >= the 0.85 overlap
        threshold while mean flow is ~1.4 px (motion 0.14) — real parallax.
        Rejecting those frames starved SfM to 1 frame → zero registered
        cameras. A frame with measured change must veto the SSIM verdict."""
        # Smooth structured scene (gradients + soft blobs) like real imagery:
        # random noise decorrelates under any shift, real textures do not.
        xx, yy = np.meshgrid(np.arange(640), np.arange(480))
        base = (40 + 90 * xx / 640 + 60 * yy / 480).astype(np.float64)
        for cx, cy, r in ((160, 120, 90), (480, 300, 120), (300, 400, 70)):
            base += 60 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
        base8 = np.clip(base, 0, 255).astype(np.uint8)
        # Sub-pixel translation: SSIM stays high, flow is real (~1.4 px).
        moved = cv2.warpAffine(base8, np.float32([[1, 0, 1.4], [0, 1, 0]]), (640, 480))
        frame = cv2.cvtColor(moved, cv2.COLOR_GRAY2BGR)
        prev = cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR)
        assert ssim_score(frame, prev) >= 0.85, "fixture lost the high-SSIM regime"
        # Without the veto (motion=None): still a duplicate by SSIM.
        assert is_consecutive_duplicate(frame, prev, motion=None) is True
        # With the measured motion (0.14 >= floor): NOT a duplicate.
        assert is_consecutive_duplicate(frame, prev, motion=0.14) is False

    def test_static_frame_still_detected_as_consecutive_duplicate(self):
        """The veto must not open the door to real static redundancy: an
        unchanged camera on a static scene (motion ~0.0) stays a duplicate."""
        rng = np.random.default_rng(4)
        base = rng.integers(0, 255, (480, 640), dtype=np.uint8)
        frame = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        assert is_consecutive_duplicate(frame, frame.copy(), motion=0.0) is True
        assert is_consecutive_duplicate(frame, frame.copy(), motion=None) is True

    def test_moving_frame_is_never_a_kept_hash_duplicate(self):
        """Field failure (sunset_06cfea): pHash hamming 2–8 vs kept frames
        while flow showed real change — the hash match was an artifact of
        pHash's coarse layout bits under slow drift, and 22/32 candidates
        died. A moving frame must veto the pHash verdict too."""
        from app.services.duplicate_detector import compute_phash

        xx, yy = np.meshgrid(np.arange(640), np.arange(480))
        base = (40 + 90 * xx / 640 + 60 * yy / 480).astype(np.float64)
        for cx, cy, r in ((160, 120, 90), (480, 300, 120), (300, 400, 70)):
            base += 60 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
        base8 = np.clip(base, 0, 255).astype(np.uint8)
        moved = cv2.warpAffine(base8, np.float32([[1, 0, 3.0], [0, 1, 0]]), (640, 480))
        frame = cv2.cvtColor(moved, cv2.COLOR_GRAY2BGR)
        kept = [compute_phash(cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR))]
        # hamming distance ≤ threshold for this pair (assert the premise):
        assert hamming_distance(compute_phash(frame), kept[0]) <= 8
        assert is_duplicate_of_kept(frame, kept, motion=0.3) is False
        # Without the veto the hash still matches (behaviour preserved for
        # static redundancy).
        assert is_duplicate_of_kept(frame, kept, motion=None) is True

    def test_identical_frames_have_high_ssim(self):
        """Two copies of the same frame should have SSIM close to 1.0."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert ssim_score(frame, frame) > 0.95

    def test_different_frames_have_low_ssim(self):
        """Two random frames should have lower SSIM."""
        a = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        b = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert ssim_score(a, b) < 0.5

    def test_consecutive_duplicate_detection(self):
        """Same frame twice should be detected as duplicate."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert is_consecutive_duplicate(frame, frame, threshold=0.9) is True
        assert is_consecutive_duplicate(frame, None) is False

    def test_phash_distance_identical(self):
        """Identical frames should have zero Hamming distance."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        h = compute_phash(frame)
        assert hamming_distance(h, h) == 0

    def test_phash_distance_different(self):
        """Different textured frames should have higher Hamming distance than identical ones."""
        a = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        b = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        same_dist = hamming_distance(compute_phash(a), compute_phash(a))
        diff_dist = hamming_distance(compute_phash(a), compute_phash(b))
        assert same_dist == 0
        assert diff_dist >= same_dist

    def test_duplicate_of_kept(self):
        """A frame identical to one in kept_hashes should be flagged."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        h = compute_phash(frame)
        assert is_duplicate_of_kept(frame, [h], threshold=8) is True
        assert is_duplicate_of_kept(frame, [], threshold=8) is False


# ---------------------------------------------------------------------------
# Quality analyzer tests
# ---------------------------------------------------------------------------


class TestQualityAnalyzer:
    def test_random_frame_has_valid_scores(self):
        """A random frame should produce valid 0-1 scores."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        q = analyze_frame(frame)
        assert 0.0 <= q.blur <= 1.0
        assert 0.0 <= q.sharpness <= 1.0
        assert 0.0 <= q.exposure <= 1.0
        assert 0.0 <= q.motion <= 1.0
        assert 0.0 <= q.composite <= 1.0

    def test_flat_frame_is_rejected(self):
        """A completely flat (black) frame should be rejected as poor exposure."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        q = analyze_frame(frame)
        assert q.rejection_reason != ""

    def test_motion_score_zero_for_first_frame(self):
        """First frame (no prev) should have motion score 0."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        q = analyze_frame(frame, prev_frame=None)
        assert q.motion == 0.0


# ---------------------------------------------------------------------------
# Frame extractor integration test
# ---------------------------------------------------------------------------


class TestFrameExtractor:
    def test_top_percent_selection_spans_timeline(self, tmp_path: Path, monkeypatch):
        """Regression: top-N ranking must not collapse selection onto one
        temporal cluster of the video.

        Field failure (airport1 data run 20260918_132723): 9 selected frames
        all inside one 26 s window of a 415 s flight — quality-ranked top-N
        with zero temporal diversity. Selection must spread across buckets.
        """
        import app.services.frame_extractor as fe

        synthetic = tmp_path / "flight.avi"
        self._make_panning_video(synthetic, n_frames=120, drift_px=40)
        report = extract_frames(
            synthetic,
            tmp_path / "out",
            extraction_mode="every_n",
            every_n=2,
            top_percent=0.15,
        )
        kept = [f for f in report["frames"] if f["kept"]]
        assert len(kept) >= 4, f"selection collapsed to {len(kept)} frames"
        nums = [f["frame_num"] for f in kept]
        span = (max(nums) - min(nums)) / 119
        assert span >= 0.5, f"selected frames span only {span:.0%} of the video"

    def test_recent_window_dedup_keeps_late_frames(self, tmp_path: Path, monkeypatch):
        """Regression: dedup must not cull an out-and-back survey's return leg.

        Out-and-back survey legs are the standard aerial pattern. A global
        pHash window used to cull the entire return leg (same terrain as the
        outbound hashes -> "duplicate"), destroying loop closure and half
        the scene coverage. Field evidence: 85/208 candidates culled on the
        airport1 run, selection collapsed to one 26 s window.

        The motion veto is now the primary protection — moving frames are
        never hash-duplicates — so even an unbounded window must keep the
        return leg; the bounded recent window remains for genuinely static
        redundancy.
        """
        import app.services.frame_extractor as fe

        synthetic = tmp_path / "flight.avi"
        self._make_panning_video(synthetic, n_frames=120, drift_px=40, return_leg=True)

        for window in (10, 10**9):
            monkeypatch.setattr(fe, "RECENT_DEDUP_WINDOW", window)
            rep = extract_frames(synthetic, tmp_path / f"out_{window}", extraction_mode="every_n", every_n=2)
            kept_return = sum(1 for f in rep["frames"] if f["kept"] and f["frame_num"] >= 60)
            assert kept_return >= 20, (
                f"return leg culled at window={window}: kept_return={kept_return}"
            )

    def test_high_fps_slow_drift_keeps_multi_view_coverage(self, tmp_path: Path):
        """Field failure (London_Mission_ca3c5b, 59.94 fps): consecutive
        every_n=1 candidates sit ~1 px apart — SSIM ~0.86 cleared the 0.85
        overlap threshold while every other gate was healthy — so 149/150
        candidates were rejected as ``duplicate`` and SfM got 1 frame
        (zero registered cameras). With the motion veto, the same regime
        keeps real multi-view coverage."""
        synthetic = tmp_path / "drift.avi"
        self._make_panning_video(synthetic, n_frames=90, drift_px=30)
        report = extract_frames(synthetic, tmp_path / "out", extraction_mode="every_n", every_n=1)
        kept = [f for f in report["frames"] if f["kept"]]
        assert len(kept) >= 40, (
            f"slow-drift candidates were eaten as duplicates: kept={len(kept)}"
        )
        span = (kept[-1]["frame_num"] - kept[0]["frame_num"]) / 89
        assert span >= 0.5, f"selection collapsed onto one segment: span={span:.0%}"

    @staticmethod
    def _make_panning_video(
        path: Path, n_frames: int, drift_px: int, return_leg: bool = False
    ) -> None:
        """Synthetic translating-camera video: a wide textured terrain viewed
        through a sliding window (whole frame moves, no wrap-around).

        ``return_leg=True`` pans right for half the frames, then pans back —
        the standard out-and-back survey line.
        """
        rng = np.random.default_rng(7)
        wide_w = 1920 + n_frames * drift_px
        terrain = rng.integers(40, 215, (1080, wide_w), dtype=np.uint8)
        terrain = cv2.cvtColor(cv2.GaussianBlur(terrain, (0, 0), 1), cv2.COLOR_GRAY2BGR)
        rng2 = np.random.default_rng(11)
        # Sparse world objects: a heavy-laplacian texture would break pHash
        # locality (bit flips make far-apart views hash-close); sparse dark
        # rectangles on smooth terrain reproduce the field failure regime —
        # slow global drift where distant viewpoints hash-close-match.
        for _ in range(8):
            x0 = int(rng2.integers(0, wide_w - 400))
            y0 = int(rng2.integers(100, 800))
            w0, h0 = int(rng2.integers(150, 350)), int(rng2.integers(100, 220))
            shade = int(rng2.integers(30, 120))
            terrain[y0:y0 + h0, x0:x0 + w0] = (shade, shade, shade)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920, 1080))
        max_x = wide_w - 1920
        for i in range(n_frames):
            frac = i / max(1, n_frames - 1)
            x = int(max_x * (2 * frac if frac <= 0.5 else 2 * (1 - frac))) if return_leg else min(i * drift_px, max_x)
            writer.write(terrain[:, x:x + 1920])
        writer.release()

    def test_extract_from_synthetic_video(self, synthetic_video: Path, tmp_path: Path):
        """Full extraction pipeline on a synthetic video."""
        output_dir = tmp_path / "extraction_output"
        report = extract_frames(
            synthetic_video,
            output_dir,
            extraction_mode="every_frame",
        )

        # Should have extracted frames
        assert report["candidates_extracted"] > 0
        assert report["selected_count"] > 0

        # Output directories should exist
        assert (output_dir / "frames").is_dir()
        assert (output_dir / "selected").is_dir()
        assert (output_dir / "rejected").is_dir()

        # Quality report should exist
        assert (output_dir / "quality_report.json").is_file()

        # Contact sheet should exist
        assert (output_dir / "contact_sheet.jpg").is_file()

        # All frame files should exist on disk
        for frame in report["frames"]:
            frame_path = output_dir / "frames" / frame["filename"]
            assert frame_path.exists(), f"Frame {frame['filename']} missing"

    def test_extraction_with_quality_threshold(self, synthetic_video: Path, tmp_path: Path):
        """High quality threshold should reject more frames."""
        output_dir = tmp_path / "extraction_threshold"
        report = extract_frames(
            synthetic_video,
            output_dir,
            extraction_mode="every_frame",
            quality_threshold=0.9,  # Very high threshold
        )
        # With a high threshold, fewer frames should be kept
        assert report["selected_count"] <= report["candidates_extracted"]
        assert report["rejected_count"] >= 0

    def test_extraction_every_n(self, synthetic_video: Path, tmp_path: Path):
        """Extracting every Nth frame should reduce candidate count."""
        output_dir = tmp_path / "extraction_every_n"
        report_all = extract_frames(synthetic_video, tmp_path / "all", extraction_mode="every_frame")
        report_n = extract_frames(synthetic_video, output_dir, extraction_mode="every_n", every_n=5)

        assert report_n["candidates_extracted"] < report_all["candidates_extracted"]


# ---------------------------------------------------------------------------
# Integration test via API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_frame_extraction_api(client: AsyncClient, synthetic_video: Path):
    """Upload a video, then run frame extraction via the API."""
    # Upload
    with open(synthetic_video, "rb") as f:
        upload_resp = await client.post(
            "/api/upload",
            files={"file": ("test_drone.avi", f, "video/avi")},
        )
    assert upload_resp.status_code == 201
    job_id = upload_resp.json()["job_id"]

    # Start extraction
    extract_resp = await client.post(
        f"/api/frame-extraction/{job_id}",
        json={"extraction_mode": "every_frame"},
    )
    assert extract_resp.status_code == 200
    data = extract_resp.json()
    assert data["selected_count"] > 0
    assert data["total_candidates"] > 0

    # Get status
    status_resp = await client.get(f"/api/frame-extraction/{job_id}")
    assert status_resp.status_code == 200
    status = status_resp.json()
    assert len(status["frames"]) > 0
    assert status["selected_count"] > 0


# ---------------------------------------------------------------------------
# seek_to_index — single-frame regeneration (missing source-image healing)
# ---------------------------------------------------------------------------

def _write_gray_video(path, n=4, size=32):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (size, size))
    assert w.isOpened()
    for i in range(n):
        w.write(np.full((size, size, 3), i * 60, np.uint8))
    w.release()


class TestSeekToIndex:
    @pytest.fixture()
    def video(self, tmp_path):
        p = tmp_path / "video.avi"
        _write_gray_video(p)
        return p

    def test_writes_the_requested_frame(self, video, tmp_path):
        out = tmp_path / "frame_000001.jpg"
        assert seek_to_index(video, 1, out) is True
        img = cv2.imread(str(out))
        assert img is not None
        # Frame 1 is the second gray level (60), not 0 or 180.
        assert abs(img.mean() - 60.0) < 20.0

    def test_never_overwrites_an_existing_frame(self, video, tmp_path):
        out = tmp_path / "frame_000001.jpg"
        out.write_bytes(b"sentinel")
        assert seek_to_index(video, 1, out) is True
        assert out.read_bytes() == b"sentinel"

    def test_false_when_video_missing(self, tmp_path):
        assert seek_to_index(tmp_path / "nope.avi", 0, tmp_path / "f.jpg") is False

    def test_false_when_index_out_of_range(self, video, tmp_path):
        assert seek_to_index(video, 99, tmp_path / "f.jpg") is False
        assert not (tmp_path / "f.jpg").exists()
