"""Regression tests for the .JPG (uppercase-extension) dataset bug.

DJI drones write ``.JPG``; POSIX globs are case-sensitive, so lowercase-only
``glob("*.jpg")`` calls returned empty lists. The worst instance produced an
empty COLMAP database, and pycolmap's matcher crashed the whole process with
SIGABRT (exit 134) instead of raising a Python exception. These tests pin the
shared case-insensitive discovery helpers at every call site class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services import camera_pose_estimator
from app.services.image_files import (
    count_image_files,
    find_image_file,
    list_image_files,
)


def _write_image(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(path.suffix.lower(), img)
    assert ok
    path.write_bytes(buf.tobytes())


cv2 = pytest.importorskip("cv2")


class TestListImageFiles:
    def test_accepts_uppercase_jpg(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "DJI_0001.JPG")
        _write_image(tmp_path / "DJI_0002.JPG")
        files = list_image_files(tmp_path)
        assert [p.name for p in files] == ["DJI_0001.JPG", "DJI_0002.JPG"]
        assert count_image_files(tmp_path) == 2

    def test_accepts_mixed_case_and_jpeg(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "a.jpg")
        _write_image(tmp_path / "b.JPG")
        _write_image(tmp_path / "c.jpeg")
        _write_image(tmp_path / "d.JPG")
        assert count_image_files(tmp_path) == 4

    def test_ignores_non_images_and_dirs(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "a.jpg")
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "sub").mkdir()
        _write_image(tmp_path / "sub" / "nested.jpg")
        assert count_image_files(tmp_path) == 1

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert list_image_files(tmp_path / "nope") == []
        assert count_image_files(tmp_path / "nope") == 0


class TestFindImageFile:
    def test_resolves_uppercase(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "frame_001.JPG")
        found = find_image_file(tmp_path, "frame_001")
        assert found is not None and found.name == "frame_001.JPG"

    def test_resolves_lowercase_too(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "frame_001.jpg")
        found = find_image_file(tmp_path, "frame_001")
        assert found is not None and found.name == "frame_001.jpg"

    def test_returns_real_case_on_ci_filesystem(self, tmp_path: Path) -> None:
        # On case-insensitive filesystems (macOS default), is_file() is True
        # for a wrong-case path; the helper must still return the real name.
        _write_image(tmp_path / "Frame_001.JPG")
        found = find_image_file(tmp_path, "Frame_001")
        assert found is not None and found.name == "Frame_001.JPG"

    def test_missing_stem_returns_none(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "frame_001.JPG")
        assert find_image_file(tmp_path, "frame_999") is None


class TestPrepareColmapImages:
    """The crash site: COLMAP preparation must copy .JPG frames, never
    silently produce an empty image dir."""

    def test_uppercase_jpgs_are_reencoded_as_png(self, tmp_path: Path) -> None:
        selected = tmp_path / "selected"
        selected.mkdir()
        for i in range(3):
            _write_image(selected / f"20181221_ms1_{i:03d}.JPG", size=(64, 48))

        image_dir = camera_pose_estimator._prepare_colmap_images(selected)
        pngs = sorted(image_dir.glob("*.png"))
        assert len(pngs) == 3

    def test_empty_selection_is_detected_by_frames_check(self, tmp_path: Path) -> None:
        # Guard rail: an empty/missing selection is reported as zero images
        # by the shared counter (used by the orchestrator artifact check),
        # so the stage fails honestly instead of reaching COLMAP with an
        # empty database.
        selected = tmp_path / "selected"
        selected.mkdir()
        assert count_image_files(selected) == 0
        _write_image(selected / "frame.JPG")
        assert count_image_files(selected) == 1
