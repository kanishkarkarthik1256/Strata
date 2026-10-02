"""Ground truth and metric accuracy evaluation schemas for STRATA."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class GCPRole(str, Enum):
    """Role of a ground control point."""

    CONTROL = "control"
    CHECK = "check"


class GCPPoint(BaseModel):
    """Ground Control Point or Check Point."""

    id: str = Field(..., description="Unique point identifier, e.g. GCP_001")
    latitude: float = Field(..., description="WGS84 latitude in degrees")
    longitude: float = Field(..., description="WGS84 longitude in degrees")
    altitude: float = Field(0.0, description="Ellipsoidal/orthometric altitude in meters")
    role: GCPRole = Field(GCPRole.CHECK, description="control = influences alignment, check = independent evaluation only")
    pixel_coords: Optional[dict[str, list[float]]] = Field(
        default=None, description="Optional image pixel coordinates {frame_id: [x, y]}"
    )
    accuracy_m: float = Field(0.02, description="Survey uncertainty in meters")


class KnownDistance(BaseModel):
    """A known physical distance between two points or features."""

    id: str = Field(..., description="Identifier for distance measurement")
    point_a: list[float] = Field(..., description="[x, y, z] position in world/ENU frame of point A")
    point_b: list[float] = Field(..., description="[x, y, z] position in world/ENU frame of point B")
    distance_m: float = Field(..., description="True surveyed distance in meters")
    description: Optional[str] = Field(None, description="e.g. 'Scale bar 1' or 'Building width'")


class ReferenceGeometry(BaseModel):
    """Paths to high-accuracy reference scan files."""

    mesh_path: Optional[str] = Field(None, description="Path to ground-truth PLY/OBJ reference mesh")
    pointcloud_path: Optional[str] = Field(None, description="Path to ground-truth PLY reference point cloud")
    coordinate_system: str = Field("ENU", description="Coordinate reference system of reference geometry")


class GroundTruthData(BaseModel):
    """Machine-readable ground-truth specification (ground_truth.json)."""

    coordinate_system: str = Field("EPSG:4326", description="CRS of point coordinates")
    units: str = Field("meters", description="Linear units")
    control_points: list[GCPPoint] = Field(default_factory=list, description="Control points used for alignment")
    check_points: list[GCPPoint] = Field(default_factory=list, description="Independent evaluation points")
    known_distances: list[KnownDistance] = Field(default_factory=list, description="Surveyed physical dimensions")
    reference_geometry: Optional[ReferenceGeometry] = Field(None, description="Ground-truth 3D geometry")
    is_synthetic: bool = Field(False, description="True if generated from controlled synthetic benchmark")


class CameraPositionMetrics(BaseModel):
    """Accuracy metrics for estimated camera positions against ground truth."""

    registered_cameras: int = 0
    total_cameras: int = 0
    registration_rate_percent: float = 0.0
    horizontal_rmse_m: float = 0.0
    vertical_rmse_m: float = 0.0
    rmse_3d_m: float = 0.0
    mae_3d_m: float = 0.0
    median_error_m: float = 0.0
    max_error_m: float = 0.0


class DistanceAccuracyMetrics(BaseModel):
    """Relative distance measurement accuracy."""

    num_distances: int = 0
    mean_absolute_error_m: float = 0.0
    max_absolute_error_m: float = 0.0
    mean_relative_error: float = 0.0
    mean_percentage_error: float = 0.0
    details: list[dict[str, Any]] = Field(default_factory=list)


class SurfaceDistanceMetrics(BaseModel):
    """Point cloud / mesh surface comparison metrics against 3D ground truth."""

    num_samples: int = 0
    mean_error_m: float = 0.0
    median_error_m: float = 0.0
    rmse_m: float = 0.0
    p50_m: float = 0.0
    p90_m: float = 0.0
    p95_m: float = 0.0
    p99_m: float = 0.0
    max_error_m: float = 0.0


class DepthAccuracyMetrics(BaseModel):
    """Depth estimation accuracy against ground truth depth maps."""

    model_type: str = "Relative depth model"  # Or "Calibrated metric depth"
    is_metric: bool = False
    mae_m: Optional[float] = None
    rmse_m: Optional[float] = None
    relative_error: Optional[float] = None
    delta_1_25: Optional[float] = None  # % of pixels with max(d/d*, d*/d) < 1.25
    delta_1_25_sq: Optional[float] = None
    delta_1_25_cu: Optional[float] = None


class ReprojectionMetrics(BaseModel):
    """SfM feature reprojection error metrics."""

    mean_reproj_px: float = 0.0
    median_reproj_px: float = 0.0
    p95_reproj_px: float = 0.0
    max_reproj_px: float = 0.0
    num_observations: int = 0
    num_points: int = 0


class CheckPointMetrics(BaseModel):
    """Accuracy metrics computed strictly on independent check points."""

    num_check_points: int = 0
    horizontal_rmse_m: float = 0.0
    vertical_rmse_m: float = 0.0
    rmse_3d_m: float = 0.0
    mae_3d_m: float = 0.0
    max_error_m: float = 0.0
    per_point_errors: list[dict[str, Any]] = Field(default_factory=list)
