"""The batched conditioning statistic must return the loop's exact float.

`_min_observation_angle_deg` is the smallest pairwise ray angle subtended at a
3D point by its observing cameras. It runs once per reconstructed point
(59,922 times on the run under measurement) and its pairwise Python double
loop was the largest single block inside the pose backend — 132.5 s of the
813.9 s `pose_estimation` span. It was rewritten as one batched computation.

The value it feeds (`min_triangulation_angle_deg`, and through it
`rel_depth_uncertainty`) is reported to four decimals and compared against the
pipeline's own gates, so "close" is not good enough here: the rewrite is
written term by term to reproduce the same floats, and this test pins that by
comparing against a verbatim copy of the loop on the geometries that occur in
practice — short tracks, long tracks, pairs sharing a centre, and the
degenerate cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.camera_pose_estimator import _min_observation_angle_deg


def _reference_min_observation_angle_deg(xyz, centers) -> float | None:
    """Verbatim copy of the double loop that was replaced."""
    if len(centers) < 2:
        return None
    min_angle: float | None = None
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            va = xyz - centers[i]
            vb = xyz - centers[j]
            cosang = float(np.dot(va, vb) /
                           (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
            ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
            min_angle = ang if min_angle is None else min(min_angle, ang)
    return min_angle


def _cases(seed: int, n_cases: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_cases):
        n = int(rng.integers(2, 25))
        # A flown trajectory shape: centres spread over tens of metres.
        centers = [rng.normal(0, 30, 3) for _ in range(n)]
        xyz = rng.normal(0, 30, 3)
        yield xyz, centers


def test_batched_angle_is_bit_identical_to_the_loop():
    checked = 0
    for xyz, centers in _cases(seed=17, n_cases=4000):
        reference = _reference_min_observation_angle_deg(xyz, centers)
        batched = _min_observation_angle_deg(xyz, centers)
        assert reference is not None and batched is not None
        # Bit-for-bit: `min` over the same per-pair floats is exact.
        assert batched == reference, (batched, reference, len(centers))
        checked += 1
    assert checked == 4000


def test_batched_angle_matches_on_collinear_and_clustered_cameras():
    """Ray systems that sit on the arccos clamp boundary."""
    rng = np.random.default_rng(23)
    checked = 0
    for _ in range(600):
        n = int(rng.integers(2, 12))
        # Centres on a line through the point: many exactly-0 and exactly-180
        # angles, where the clip at ±1 decides the value.
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)
        centers = [axis * float(rng.normal(0, 20)) for _ in range(n)]
        xyz = axis * float(rng.normal(0, 5))
        reference = _reference_min_observation_angle_deg(xyz, centers)
        batched = _min_observation_angle_deg(xyz, centers)
        assert batched == reference, (batched, reference)
        checked += 1
    assert checked == 600


def test_batched_angle_handles_duplicate_centres():
    """A repeated centre is a real input, and the loop does NOT return exactly
    0 for it: the ``+ 1e-12`` in the denominator keeps the cosine just below
    one, so the angle is ~7e-06 deg rather than 0. The batched form must
    reproduce that same non-zero value."""
    xyz = np.array([10.0, -4.0, 6.0])
    centers = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]),
               np.array([1.0, 2.0, 3.0])]
    reference = _reference_min_observation_angle_deg(xyz, centers)
    assert reference is not None and 0.0 < reference < 1e-4
    assert _min_observation_angle_deg(xyz, centers) == reference


def test_batched_angle_refuses_fewer_than_two_centres():
    xyz = np.array([10.0, -4.0, 6.0])
    assert _min_observation_angle_deg(xyz, []) is None
    assert _min_observation_angle_deg(xyz, [np.array([1.0, 2.0, 3.0])]) is None
    # A single centre is still one that cannot constrain depth.
    assert _reference_min_observation_angle_deg(xyz, [np.array([1.0, 2.0, 3.0])]) is None


def test_batched_angle_on_real_track_lengths():
    """Track lengths from the run under measurement (median 10, p95 35)."""
    rng = np.random.default_rng(31)
    for n in (2, 3, 4, 5, 10, 35, 87):
        for _ in range(40):
            centers = [rng.normal(0, 40, 3) for _ in range(n)]
            xyz = rng.normal(0, 40, 3)
            assert _min_observation_angle_deg(xyz, centers) == \
                _reference_min_observation_angle_deg(xyz, centers)


def test_batched_angle_accepts_lists_not_just_arrays():
    """Callers pass a list of numpy centres; xyz may be a list too."""
    xyz = [10.0, -4.0, 6.0]
    centers = [[1.0, 2.0, 3.0], [-4.0, 0.0, 8.0], [3.0, 3.0, -2.0]]
    np_centers = [np.asarray(c) for c in centers]
    assert _min_observation_angle_deg(np.asarray(xyz), np_centers) == \
        _reference_min_observation_angle_deg(np.asarray(xyz), np_centers)
