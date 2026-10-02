"""Digital twin engine — the evolving dense model for a job.

The twin owns:
* the current dense ``PointCloud`` (updated incrementally via the mapper),
* job metadata (source frames, units, GPS origin when provided),
* a versioned update log,
* metric measurements (bounding box, footprint, 2.5D coverage),
* mission intelligence: weak / under-covered regions and suggested
  improvements (reduced overlap, lighting, side passes).

Mesh integration and semantic attributes are deliberately deferred to later
phases; the representation is a point cloud plus metadata only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.logging_config import get_logger
from app.services.confidence_estimator import CONFIDENCE_CLASSES, dense_level_from_conf
from app.services.incremental_mapper import IncrementalMapper, MergeStats
from app.services.point_statistics import footprint_cells, weak_regions
from app.services.pointcloud import PointCloud

log = get_logger("drone_recon.services.digital_twin")

_WEAK_SUGGESTIONS = {
    "high": "Increase forward/side overlap on the next pass",
    "medium": "Fly slower and add a cross-hatch side pass for weak regions",
    "low": "Consider RTK or GCPs to tighten geometry in sparse areas",
}


@dataclass
class TwinMeasurements:
    """Geometric measurements of the twin model."""

    bbox_min: list[float] = field(default_factory=list)
    bbox_max: list[float] = field(default_factory=list)
    dimensions_m: list[float] = field(default_factory=list)
    footprint_area_m2: float = 0.0
    mean_confidence: float = 0.0
    point_count: int = 0

    def to_dict(self) -> dict:
        return {
            "bbox_min": [round(v, 4) for v in self.bbox_min],
            "bbox_max": [round(v, 4) for v in self.bbox_max],
            "dimensions_m": [round(v, 4) for v in self.dimensions_m],
            "footprint_area_m2": round(self.footprint_area_m2, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "point_count": self.point_count,
        }


class DigitalTwin:
    """Incrementally built point-based digital twin for one job."""

    def __init__(self, job_id: str, voxel_size: float = 0.05, metadata: dict | None = None) -> None:
        self.job_id = job_id
        self.voxel_size = float(voxel_size)
        self.metadata: dict = {
            "job_id": job_id,
            "representation": "point_cloud",
            "units": "meters",
            "created_at": time.time(),
            **({"source": metadata} if metadata else {}),
        }
        self._mapper = IncrementalMapper(voxel_size=self.voxel_size, job_id=job_id)
        self.version = 0
        self.updates: list[dict] = []

    # -- incremental updates -------------------------------------------------

    def update(self, chunk: PointCloud, label: str = "chunk") -> MergeStats:
        """Fold a new dense chunk into the twin and version the result."""
        stats = self._mapper.add(chunk)
        self.version += 1
        self.updates.append(
            {
                "version": self.version,
                "label": label,
                "points_added": stats.new_points + stats.merged_points,
                "total_points": self._mapper.model.n if self._mapper.model else 0,
            }
        )
        return stats

    @property
    def cloud(self) -> PointCloud | None:
        return self._mapper.current

    # -- analysis -------------------------------------------------------------

    def measurements(self) -> TwinMeasurements:
        cloud = self.cloud
        if cloud is None or cloud.n == 0:
            return TwinMeasurements()
        bmin, bmax = cloud.bounds()
        cells_area = len(footprint_cells(cloud, self.voxel_size)[1]) * self.voxel_size**2
        conf = cloud.confidence.mean() if cloud.confidence is not None else 0.0
        return TwinMeasurements(
            bbox_min=[float(v) for v in bmin],
            bbox_max=[float(v) for v in bmax],
            dimensions_m=[float(v) for v in bmax - bmin],
            footprint_area_m2=float(cells_area),
            mean_confidence=float(conf),
            point_count=cloud.n,
        )

    def confidence_summary(self) -> dict:
        """Per-class point counts for the four confidence classes."""
        cloud = self.cloud
        if cloud is None or cloud.confidence is None or cloud.n == 0:
            return {c: 0 for c in CONFIDENCE_CLASSES}
        levels = dense_level_from_conf(cloud.confidence)
        return {c: int(np.sum(levels == c)) for c in CONFIDENCE_CLASSES}

    def region_intelligence(self, max_regions: int = 10) -> dict:
        """Weak/occluded region centroids plus actionable suggestions."""
        cloud = self.cloud
        if cloud is None or cloud.n == 0:
            return {"weak_regions": [], "suggestions": []}
        regions = weak_regions(cloud, self.voxel_size, max_regions=max_regions)
        suggestions: list[str] = []
        conf = cloud.confidence.mean() if cloud.confidence is not None else 1.0
        if regions:
            if conf >= 0.7:
                suggestions.append(_WEAK_SUGGESTIONS["high"])
            elif conf >= 0.45:
                suggestions.append(_WEAK_SUGGESTIONS["medium"])
            else:
                suggestions.append(_WEAK_SUGGESTIONS["low"])
        if conf < 0.45:
            suggestions.append("Overall confidence is low — improve lighting and reduce motion blur")
        if regions and len(regions) >= 5:
            suggestions.append("Large sparse areas detected — add a cross-hatch side pass")
        return {"weak_regions": regions[:max_regions], "suggestions": suggestions}

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "metadata": self.metadata,
            "version": self.version,
            "updates": self.updates[-20:],
            "measurements": self.measurements().to_dict(),
            "confidence": self.confidence_summary(),
            "intelligence": self.region_intelligence(),
        }
