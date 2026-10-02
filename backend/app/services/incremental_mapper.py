"""Incremental mapper — merges new dense chunks into the running model.

Supports the "incremental digital twin" requirement: a mission streams
frames in, each chunk is fused and the mapper folds it into the current
model on the shared voxel grid. Points landing in an already-covered voxel
are merged (weighted average, observation counts add), points in new voxels
extend the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.logging_config import get_logger
from app.services.depth_fusion import voxel_merge
from app.services.pointcloud import PointCloud

log = get_logger("drone_recon.services.incremental_mapper")


@dataclass
class MergeStats:
    """Effect of one incremental merge."""

    existing_points: int = 0
    new_points: int = 0
    merged_points: int = 0  # chunk points absorbed into existing voxels
    total_points: int = 0


class IncrementalMapper:
    """Accumulates a model by merging point cloud chunks on a voxel grid."""

    def __init__(self, voxel_size: float = 0.05, job_id: str = "") -> None:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")
        self.voxel_size = float(voxel_size)
        self.job_id = job_id
        self.model: PointCloud | None = None
        self.chunks_processed = 0
        self.history: list[dict] = []

    def add(self, chunk: PointCloud, progress: Callable[[str, float, dict], None] | None = None) -> MergeStats:
        """Merge *chunk* into the current model."""
        if chunk.n == 0:
            return MergeStats(existing_points=self.model.n if self.model else 0)

        if self.model is None:
            self.model = chunk
            stats = MergeStats(new_points=chunk.n, total_points=chunk.n)
        else:
            existing_n = self.model.n
            self.model = voxel_merge(
                np.concatenate([self.model.xyz, chunk.xyz]),
                np.concatenate([self.model.ensure_colors(), chunk.ensure_colors()])
                if self.model.rgb is not None or chunk.rgb is not None
                else None,
                np.concatenate(
                    [
                        self.model.confidence if self.model.confidence is not None else np.ones(existing_n),
                        chunk.confidence if chunk.confidence is not None else np.ones(chunk.n),
                    ]
                ),
                self.voxel_size,
            )
            stats = MergeStats(
                existing_points=existing_n,
                merged_points=int(np.clip(existing_n + chunk.n - self.model.n, 0, chunk.n)),
                new_points=self.model.n - existing_n,
                total_points=self.model.n,
            )

        self.chunks_processed += 1
        self.history.append(
            {
                "chunk": self.chunks_processed,
                "points": chunk.n,
                "total_points": self.model.n,
                "merged_points": stats.merged_points,
            }
        )
        if progress:
            progress("incremental_merge", 1.0, stats.__dict__)
        log.info(
            "incremental_merge",
            job_id=self.job_id,
            chunk=self.chunks_processed,
            chunk_points=chunk.n,
            total_points=self.model.n,
        )
        return stats

    @property
    def current(self) -> PointCloud | None:
        return self.model

    def finish(self) -> PointCloud:
        if self.model is None:
            raise ValueError("No chunks were added to the incremental mapper")
        return self.model
