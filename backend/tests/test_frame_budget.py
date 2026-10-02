"""Duration-aware keyframe budget — regression tests.

The budget must: (1) never raise the kept count above the budget,
(2) keep temporal/trajectory coverage (never collapse onto one segment),
(3) prefer the best-quality frame within each coverage bucket,
(4) be a no-op when the kept count is already below the budget, and
(5) integrate through extract_frames() with the report recording it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.services.frame_extractor import extract_frames, _select_temporal_buckets


@dataclass
class _Q:
    composite: float
    rejection_reason: str = ""


def _mk(n: int, best_every: int = 7) -> list[dict]:
    """n candidate dicts with deterministic quality (every `best_every`-th is best)."""
    out = []
    for i in range(n):
        out.append({
            "index": i,
            "frame_num": i,
            "timestamp_sec": float(i),
            "filename": f"frame_{i:06d}.jpg",
            "file_path": f"/tmp/frame_{i:06d}.jpg",
            "quality": _Q(1.0 if i % best_every == 0 else 0.5),
            "kept": True,
        })
    return out


def test_budget_never_exceeds_cap(tmp_path: Path):
    for budget in (5, 17, 40):
        chosen = _select_temporal_buckets(_mk(60), budget, tmp_path)
        assert len(chosen) <= budget


def test_budget_preserves_full_coverage(tmp_path: Path):
    """Chosen frames must span the whole timeline, not one segment."""
    cands = _mk(60)
    chosen = _select_temporal_buckets(cands, 6, tmp_path)
    nums = sorted(c["frame_num"] for c in chosen)
    assert nums[0] <= 9  # something from the start
    assert nums[-1] >= 50  # something from the end
    # and spread: max gap between consecutive picks bounded by ~2 buckets
    gaps = [b - a for a, b in zip(nums, nums[1:])]
    assert max(gaps) <= 20


def test_budget_prefers_best_quality_in_bucket(tmp_path: Path):
    cands = _mk(30, best_every=5)  # best frames at 0,5,10,15,20,25
    chosen = _select_temporal_buckets(cands, 6, tmp_path)
    # With 6 buckets over 30 frames (5-wide), each bucket's best (quality 1.0)
    # should win its bucket — all picks must be the high-quality frames.
    assert all(c["quality"].composite == 1.0 for c in chosen)


def test_budget_noop_when_under_budget(tmp_path: Path):
    cands = _mk(10)
    chosen = _select_temporal_buckets(cands, 20, tmp_path)
    assert len(chosen) == 10
    assert chosen is not cands  # returns the full list, unmodified


def test_extract_frames_budget_wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """extract_frames applies the budget end-to-end and records it."""
    import cv2
    import numpy as np

    video_path = tmp_path / "in.avi"
    fps, n = 10.0, 90  # 9 s clip
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    rng = np.random.default_rng(0)
    for i in range(n):
        # Textured mid-exposure frames (random noise per frame): pass the
        # blur/exposure gates and differ enough for the phash dedup.
        frame = rng.integers(40, 215, size=(48, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    out = extract_frames(video_path, tmp_path / "ws", extraction_mode="every_frame", frame_budget=12)
    assert out["selected_count"] <= 12
    assert out["frame_budget"] == 12
    assert out["frame_budget_applied"] is True


def test_extract_frames_budget_disabled(tmp_path: Path):
    """frame_budget=None keeps the pre-budget behaviour (all kept frames)."""
    import cv2
    import numpy as np

    video_path = tmp_path / "in.avi"
    fps, n = 10.0, 40
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    rng = np.random.default_rng(1)
    for i in range(n):
        frame = rng.integers(40, 215, size=(48, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    out = extract_frames(video_path, tmp_path / "ws", extraction_mode="every_frame", frame_budget=None)
    assert out["frame_budget"] is None
    assert out["frame_budget_applied"] is False
