"""Image-file discovery for pipeline stages.

Globs are case-sensitive on POSIX, and real drone cameras (e.g. DJI) write
uppercase extensions (``.JPG``). Every stage that counts or reads selected
frames must go through these helpers so a lowercase-only glob can never
silently produce an empty list — the failure that previously crashed the
COLMAP matching stage.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def list_image_files(directory: Path) -> list[Path]:
    """All image files in *directory* (non-recursive), extension-case-insensitive."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def count_image_files(directory: Path) -> int:
    return len(list_image_files(directory))


def find_image_file(directory: Path, stem: str) -> Path | None:
    """Resolve ``directory/<stem>.<jpg-ish>`` regardless of extension case.

    Resolves through the actual directory listing rather than probing suffix
    spellings: on case-insensitive filesystems (macOS default) a probed
    ``.jpg`` path ``is_file()`` True even when the real file is ``.JPG``, and
    returning that wrong-case path breaks on case-sensitive Linux mounts.
    """
    if not directory.is_dir():
        return None
    for path in list_image_files(directory):
        if path.stem == stem:
            return path
    return None
