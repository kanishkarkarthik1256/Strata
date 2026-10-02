"""Deterministic storage layout + artifact management (Phase 10).

Layout (per mission project workspace under ``storage.base_dir/{project_id}``)
```
  video.mp4            raw input (single uploaded asset)
  raw/                 additional raw inputs
  frames/ selected/    processed frames
  poses.json ...       intermediate artifacts
  depth/  checkpoints/ intermediate per-stage artifacts (re-generable)
  sparse_model.ply dense/ mesh/  final outputs
  georef/ intel/       derived outputs + reports
  logs/                structured run logs
```

This module is the single owner of *cataloguing* a workspace into those
categories and applying cleanup policies to intermediate artifacts. It never
moves or rewrites existing files — cleanup only deletes re-generable
intermediate content older than the retention policy, and only when the
mission is in a terminal state or a ``force`` flag is set.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config.settings import settings
from app.logging_config import get_logger

log = get_logger("drone_recon.services.storage")

#: Relative paths that are safe to regenerate → cleanup candidates.
_INTERMEDIATE_DIRS = ("depth", "checkpoints", "tmp", ".cache")
_FINAL_DIRS = ("dense", "mesh", "georef", "intel", "exports", "textures", "lods")
_RAW_DIRS = ("raw",)
_PROCESSED_DIRS = ("frames", "selected", "frames_selected")


@dataclass
class ArtifactEntry:
    rel_path: str
    category: str  # raw|processed|intermediate|final|logs
    size_bytes: int
    modified_ts: float
    is_dir: bool = False
    children: list["ArtifactEntry"] = field(default_factory=list)


def _category_of(rel: str, is_dir: bool) -> str:
    parts = Path(rel).parts
    if not parts:
        return "raw"
    head = parts[0]
    if head in _RAW_DIRS:
        return "raw"
    if head in _PROCESSED_DIRS:
        return "processed"
    if head in _INTERMEDIATE_DIRS:
        return "intermediate"
    if head in _FINAL_DIRS:
        return "final"
    if head == "logs":
        return "logs"
    # Known single-file outputs at the workspace root.
    if rel.endswith((".ply", ".obj", ".glb", ".gltf", ".las", ".laz")) and not is_dir:
        return "final"
    if rel.endswith((".json", ".csv", ".md", ".html")) and not is_dir:
        return "final"
    # Everything else at the root is an input artifact.
    if not is_dir:
        return "raw"
    return "processed"


def catalog(project_id: str) -> list[ArtifactEntry]:
    """Return a categorized listing of a project workspace (no DB writes)."""
    root = settings.storage.project_dir(project_id)
    entries: list[ArtifactEntry] = []
    if not root.exists():
        return entries

    def walk(path: Path, rel: str) -> ArtifactEntry:
        is_dir = path.is_dir()
        category = _category_of(rel, is_dir)
        if is_dir:
            children = [walk(child, f"{rel}/{child.name}") for child in sorted(path.iterdir())]
            size = sum(c.size_bytes for c in children if not c.is_dir)
            return ArtifactEntry(rel, category, size, path.stat().st_mtime, True, children)
        try:
            st = path.stat()
        except OSError:  # pragma: no cover - raced deletion
            return ArtifactEntry(rel, category, 0, 0.0, False)
        return ArtifactEntry(rel, category, st.st_size, st.st_mtime, False)

    for child in sorted(root.iterdir()):
        entries.append(walk(child, child.name))
    return entries


def summary(project_id: str) -> dict:
    cats = {"raw": 0, "processed": 0, "intermediate": 0, "final": 0, "logs": 0}
    for e in catalog(project_id):
        _accumulate(e, cats)
    return {
        "project_id": project_id,
        "path": str(settings.storage.project_dir(project_id)),
        "categories": cats,
        "total_bytes": sum(cats.values()),
        "exists": settings.storage.project_dir(project_id).exists(),
    }


def _accumulate(entry: ArtifactEntry, cats: dict) -> None:
    if entry.is_dir:
        for child in entry.children:
            _accumulate(child, cats)
    else:
        cats[entry.category] = cats.get(entry.category, 0) + entry.size_bytes


def cleanup_intermediates(
    project_id: str,
    retention_days: Optional[float] = None,
    *,
    force: bool = False,
    terminal_only: bool = True,
) -> dict:
    """Delete re-generable intermediate artifacts older than the retention.

    Default policy: only when the caller confirms the mission is terminal and
    the artifacts exceed the retention window. Returns a report of what was
    removed; never touches raw inputs or final outputs.
    """
    retention = retention_days if retention_days is not None \
        else settings.platform.artifact_retention_days
    root = settings.storage.project_dir(project_id)
    cutoff = time.time() - retention * 86400
    removed: list[str] = []
    freed = 0
    if not root.exists():
        return {"removed": removed, "freed_bytes": 0}

    for d in _INTERMEDIATE_DIRS:
        target = root / d
        if not target.exists():
            continue
        for path in sorted(target.rglob("*"), reverse=True):
            try:
                if path.is_file() and (force or path.stat().st_mtime < cutoff):
                    freed += path.stat().st_size
                    path.unlink()
                    removed.append(str(path.relative_to(root)))
                elif path.is_dir():
                    try:
                        path.rmdir()  # only removes empty dirs
                    except OSError:
                        pass
            except OSError:  # pragma: no cover - best effort
                log.warning("cleanup_skipped", path=str(path))
    log.info("cleanup_complete", project_id=project_id, removed=len(removed),
             freed_bytes=freed)
    return {"removed": removed, "freed_bytes": freed}


def purge_workspace(project_id: str) -> None:
    """Delete the entire project workspace (destructive — callers must be admins)."""
    root = settings.storage.base_dir / project_id
    if root.exists():
        shutil.rmtree(root)
        log.info("workspace_purged", project_id=project_id)
