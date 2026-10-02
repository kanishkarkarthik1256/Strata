"""Centralized configuration for DroneRecon.

All values are loaded from environment variables (or a .env file) and validated
by Pydantic. Access via the module-level ``settings`` singleton.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_ENV_FILE = os.getenv("DRONE_RECON_ENV_FILE", ".env")


def _default_storage_path() -> str:
    return str(Path.cwd() / "data" / "storage")


def _default_database_url() -> str:
    return f"sqlite+aiosqlite:///{Path.cwd() / 'data' / 'drone_recon.db'}"


# ---------------------------------------------------------------------------
# Nested setting groups
# ---------------------------------------------------------------------------


class ServerSettings(BaseSettings):
    """HTTP server configuration."""

    model_config = SettingsConfigDict(env_prefix="SERVER_")

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


class StorageSettings(BaseSettings):
    """Filesystem storage paths."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    base_path: str = Field(default_factory=_default_storage_path)
    #: Root directory of Phase 9.5 run outputs (outputs/<RUN_ID>/manifest.json).
    #: Relative paths resolve against the process cwd (the validator's convention).
    runs_dir: str = "outputs"
    max_upload_size_mb: int = 10240  # 10 GB
    temp_dir: str = "/tmp/drone-recon"

    @property
    def base_dir(self) -> Path:
        path = Path(self.base_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def temp_path(self) -> Path:
        path = Path(self.temp_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def project_dir(self, project_id: str) -> Path:
        """Return (and create) the directory for a specific project."""
        d = self.base_dir / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d


class ProcessingSettings(BaseSettings):
    """Frame extraction and pipeline tuning parameters."""

    model_config = SettingsConfigDict(env_prefix="PROCESSING_")

    max_frames: int = 500
    target_fps: float = 2.0
    min_frame_score: float = 0.3
    blur_threshold: float = 100.0
    motion_threshold: float = 0.4
    frame_overlap_threshold: float = 0.85
    dedup_hash_threshold: int = 8
    max_upload_size_mb: int = 10240
    supported_video_formats: list[str] = ["mp4", "mov", "avi", "mkv"]
    supported_resolutions: list[str] = ["1080p", "4k", "2k", "720p"]


class AISettings(BaseSettings):
    """AI model configuration and paths."""

    model_config = SettingsConfigDict(env_prefix="AI_")

    device: Literal["cuda", "cpu", "mps"] = "cuda"
    depth_model_name: str = "depth-anything-v2-base"
    # Depth Anything V2 encoder variant: 'vitb' (default, ~2.5x depth-stage
    # CPU cost of 'vits' but halves cross-view disagreement → finer
    # auto-adapted fusion voxel and denser clouds; measured on airport1:
    # NN spacing 2.10→1.07 m, mesh edge 3.02→1.86 m) or 'vits' (fast).
    # Falls back to the only checkpoint present when the chosen variant's
    # weights are missing, so fresh checkouts without vitb weights keep
    # working on vits.
    depth_encoder: str = "vitb"
    yolo_model: str = "yolo11x.pt"
    sam2_model: str = "sam2_hiera_large.pt"
    weights_dir: str = "models/weights"
    max_batch_size: int = 8
    inference_precision: Literal["fp16", "fp32"] = "fp16"

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        # Defer torch import to use-time: importing torch at settings-load
        # time adds ~2.5 s to every cold request path because pydantic
        # validates the Settings() instance once when the module is imported.
        # Actual CUDA availability is checked at use sites (e.g.
        # feature_matcher._init_backend, depth model loaders) where torch is
        # already being imported for the real work.
        if v not in {"cuda", "cpu", "mps"}:
            return "cpu"
        return v

    @property
    def weights_path(self) -> Path:
        p = Path(self.weights_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


class COLMAPSettings(BaseSettings):
    """COLMAP binary and reconstruction parameters."""

    model_config = SettingsConfigDict(env_prefix="COLMAP_")

    binary_path: str = "colmap"
    max_features: int = 10240
    # 10240 is the lowest cap passing ALL unchanged quality gates from the
    # controlled 4-arm benchmark (cap_benchmark/benchmark_summary.json):
    # tracks 103,184 = 86.0% of the uncapped 119,919 baseline (gate ≥80%),
    # reprojection median 0.243 px / p95 0.595 px (both better than
    # baseline), 0 negative-depth tracks, 191/191 registered cameras.
    # 8192 FAILED the track gate (85,891 = 71.6%).
    # The bundled pycolmap CPU-SIFT wheel ignores max_num_features (measured:
    # cap 8192 stored up to 20,352 rows), so the cap is enforced on the
    # feature database before matching. Disable only to reproduce the
    # un-enforced legacy behavior.
    enforce_feature_cap: bool = True
    dense_point_count: int = 1_000_000
    matcher_type: Literal["exhaustive", "sequential"] = "sequential"
    camera_model: Literal["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL"] = "PINHOLE"
    ba_max_num_iterations: int = 30
    ba_max_num_images: int = 300


class DenseSettings(BaseSettings):
    """Dense reconstruction (depth fusion + point cloud) parameters."""

    model_config = SettingsConfigDict(env_prefix="DENSE_")

    voxel_size: float = 0.05  # meters — fusion grid / merge resolution
    min_depth_m: float = 0.2
    max_depth_m: float = 200.0
    pixel_noise_px: float = 0.5  # assumed per-pixel depth noise (px)
    max_points_per_view: int = 1_000_000
    max_fusion_points: int = 8_000_000
    sor_k: int = 20  # statistical outlier removal neighbors
    sor_std_ratio: float = 2.0
    ror_radius_m: float = 0.4  # radius outlier removal radius
    ror_min_neighbors: int = 5
    normal_k: int = 20  # PCA normal estimation neighbors
    min_confidence: float = 0.05

    # Per-view depth generation (stereo backend + optional refinement)
    # Sparse-anchoring gate for learned-depth views: a per-view affine fit
    # whose median landmark error exceeds this factor × the sparse depth-
    # uncertainty budget (MAX_RELATIVE_DEPTH_ERR × center range) is refused
    # and never fused. Factor 1.5 sits inside a measured separation gap on
    # flight_to_tower_7511dc: collapsed-gradient views miss at 1.99-2.43×
    # budget while every healthy view lands at ≤1.27×. Dataset sanity bound,
    # not a fundamental constant — override via DENSE_ANCHOR_GATE_FACTOR.
    sparse_anchor_gate_factor: float = 1.5
    depth_backend: Literal["auto", "depth_anything", "colmap", "stereo"] = "auto"
    stereo_min_disparity: int = 0
    stereo_num_disparities: int = 96
    stereo_block_size: int = 5
    stereo_uniqueness_ratio: int = 10
    stereo_max_views: int = 200
    refine_median: bool = False
    refine_median_k: int = 5
    refine_bilateral: bool = False
    refine_edge_preserving: bool = False
    refine_hole_fill: bool = False


class TelemetrySettings(BaseSettings):
    """Telemetry-prior configuration.

    The visual->telemetry similarity scale warning band is a DATASET sanity
    bound, not a fundamental property of single-pass reconstruction: a
    legitimate monocular scale recovery can exceed the default band, and
    different footage/unit conventions may need different bounds. Override
    via TELEMETRY_SCALE_WARN_MIN / TELEMETRY_SCALE_WARN_MAX.
    """

    model_config = SettingsConfigDict(env_prefix="TELEMETRY_")

    scale_warn_min: float = 0.25
    scale_warn_max: float = 4.0

    # Minimum fraction of the mapper's sparse points that metric telemetry
    # placement must retain. Telemetry placement re-anchors observations in
    # the metric frame; if that re-anchoring destroys most of the mapper's
    # points, the placed reconstruction is starved (tiny point count, floats
    # below the surface, empty meshes) even though the trajectories agree.
    # The gate refuses such a placement so the caller keeps the visual model
    # with honestly reported, degraded georeferencing instead.
    # Set to 0 to disable. Override via TELEMETRY_PLACEMENT_MIN_POINT_RETENTION.
    placement_min_point_retention: float = 0.30


class PipelineSettings(BaseSettings):
    """Autonomous pipeline orchestration parameters."""

    model_config = SettingsConfigDict(env_prefix="PIPE_")

    stage_retries: int = 1
    default_extraction_mode: str = "every_n"
    default_every_n: int = 10
    default_target_fps: float = 2.0
    # Duration-aware keyframe budget: clamp(duration_s * fps_budget, 60, 200).
    # 1.3 is benchmark-calibrated on airport_53_8f0e90 (127 s): gates pass at
    # 160 frames (tracks 80.2% of uncapped, all other gates green) and fail at
    # 150; 1.3 fps_budget selects ~165 frames there — verified-pass region
    # with ~6x run-to-run noise margin.
    keyframe_fps_budget: float = 1.3


class MeshSettings(BaseSettings):
    """Mesh generation / texturing / LOD parameters (Phase 7)."""

    model_config = SettingsConfigDict(env_prefix="MESH_")

    method: str = "auto"  # auto|surface|alpha|poisson|ball_pivot
    # drops triangles whose 3D edge exceeds factor x the local NN spacing of
    # their endpoints. 10.0 measured on test_run_29_bedd74 (1.26 m local
    # spacing, 119 m of relief): 4 shattered the surface into 108,659 patches
    # (largest 44.7% of faces), 10 keeps 93.3% in one surface at the same
    # cost, and the cap stays ten sampling scales long so curtains are still
    # refused. See MeshParams.surface_max_edge_factor.
    surface_max_edge_factor: float = 10.0
    surface_min_points: int = 50
    alpha_multiplier: float = 1.5  # alpha radius = multiplier * median edge of Delaunay
    # Production mesher depth. Cell size = cloud extent / 2**depth, so 10 on a
    # 500 m scene is 0.49 m — the fused cloud's own sampling scale. The shell
    # that Poisson closes around the sampled volume is cut away before the mesh
    # is used (see mesh_generator._crop_poisson_shell), so this is no longer
    # "watertight closure invents geometry".
    poisson_depth: int = 10
    # × measured P90 NN spacing (production). The smallest ball must span the
    # neighbour gaps that actually occur: rooting at the MEDIAN spacing left
    # the ball under P90 (0.665 vs 0.942 m on Video_Mission_9ab1aa) and
    # shattered the mesh into 108k edge-components (largest 1.8% of faces).
    # P90-rooted [1.0, 1.5, 2.33] raised the largest component to 74.3% on
    # that run and 75.0% on estrel_c0cf6d, with fewer oversized bridge
    # triangles than a [1, 2, 4] ladder at the same basis.
    ball_pivot_radii_factors: list[float] = [1.0, 1.5, 2.33]
    decimate_preset: str = "high"  # ultra|high|medium|low (quality -> triangle budget)
    lod_ratios: list[float] = [1.0, 0.5, 0.25, 0.12, 0.06]
    repair_min_component_faces: int = 12
    repair_max_hole_edges: int = 64
    weld_precision: float = 1e-7
    atlas_width: int = 8192
    atlas_max_faces_per_image: int = 1_000_000
    texture_max_camera_dist: float = 500.0
    # Viewer GLB triangle budget. 500k was sized for the vits-era output
    # (~1.0M faces); the vitb mesh carries ~2x the faces, and a fixed 500k
    # LOD discarded roughly half the gained detail exactly where users zoom
    # (measured: LOD edges 1.47x coarser than the authoritative mesh).
    # 1.1M keeps the full airport1-class mesh intact and stays a real LOD
    # for larger scenes; embedded JPEG adds ~25-30 MB to the GLB either way.
    # Trade-off named: heavier first load (76-105 MB), same browser budget
    # per frame (three.js frustum-culls; a scene is drawn or not).
    viewer_triangle_budget: int = 1_100_000


class SemanticSettings(BaseSettings):
    """Semantic scene understanding + object detection parameters (Phase 7)."""

    model_config = SettingsConfigDict(env_prefix="SEMANTIC_")

    backend: str = "auto"  # auto|classical|neural
    model_path: str = "models/weights/semantic.onnx"
    detection_backend: str = "auto"  # auto|none|yolo
    ground_plane_tol_m: float = 0.15
    min_face_confidence: float = 0.25
    classes: list[str] = [
        "ground", "structure", "roof", "wall", "vegetation", "water", "vehicle", "unclassified",
    ]


class DatabaseSettings(BaseSettings):
    """Database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    url: str = Field(default_factory=_default_database_url)
    echo: bool = False


class IntelSettings(BaseSettings):
    """Geospatial-intelligence engine parameters (Phase 8).

    Thresholds feed the environmental quality scoring, damage-detection
    rules, infrastructure measurements and the risk engine. All values are
    physical/statistical cut-offs with clear units, not magic flags.
    """

    model_config = SettingsConfigDict(env_prefix="INTEL_")

    # --- environmental analysis (frame metrics) ---
    env_sample_frames: int = 24
    lowlight_luma_threshold: float = 60.0  # mean 0-255 luminance below → low light
    blur_min_focus: float = 90.0  # Laplacian variance below → likely blur
    fog_dark_channel_max: float = 0.65  # dark-channel mean above → haze/smoke
    visibility_max_m: float = 15000.0  # horizon bound for the visibility estimate
    shadow_low_pct: float = 0.15  # dark-region share above → strong shadows
    glare_high_pct: float = 0.05  # saturated share above → sun glare

    # --- scene intelligence ---
    open_vocab_backend: str = "auto"  # auto|classical|dino_sam2
    dino_model_path: str = "models/weights/grounding_dino.onnx"
    sam2_model_path: str = "models/weights/sam2.onnx"
    min_concept_confidence: float = 0.3

    # --- damage assessment (mesh geometry rules) ---
    damage_min_area_m2: float = 4.0
    collapse_min_fragment_faces: int = 40
    flood_min_area_m2: float = 8.0
    debris_max_height_m: float = 3.0
    damage_class_weights: dict[str, float] = {
        "ground": 0.0, "structure": 0.6, "roof": 0.7, "wall": 0.7,
        "vegetation": 0.3, "water": 1.0, "vehicle": 0.5, "unclassified": 0.35,
    }

    # --- risk engine ---
    risk_slope_deg: float = 25.0
    risk_grid_cells: int = 96
    flood_buffer_m: float = 25.0
    debris_buffer_m: float = 15.0
    access_slope_deg: float = 15.0

    # --- infrastructure ---
    min_road_width_cloud_m: float = 1.5
    tree_min_height_m: float = 1.0

    # --- rag ---
    rag_backend: str = "auto"  # auto|lexical|transformers
    rag_model_name: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 5
    rag_chunk_words: int = 96

    # --- reporting ---
    report_formats: list[str] = ["json", "md", "html"]  # pdf added when available
    org_name: str = "DroneRecon"
    mission_operator: str = "mission_control"


class PlatformSettings(BaseSettings):
    """Enterprise platform parameters (Phase 10).

    ``auth_mode`` controls how strictly the API authenticates callers:
    ``disabled`` keeps the pre-Phase-10 endpoints open (backward
    compatibility) while the mission-management surface is always protected;
    ``required`` additionally blocks unauthenticated access to every /api/*
    route except an explicit public allowlist.

    Queue settings bound the in-process worker that executes enqueued
    missions against the existing pipeline orchestrator.
    """

    model_config = SettingsConfigDict(env_prefix="PLATFORM_")

    auth_mode: Literal["disabled", "required"] = "disabled"
    session_ttl_hours: float = 12.0
    token_bytes: int = 32

    # Bootstrap admin — created at startup when credentials are provided.
    bootstrap_admin_email: str = ""
    bootstrap_admin_password: str = ""

    # Queue worker
    queue_max_workers: int = 1
    queue_poll_seconds: float = 1.0
    queue_default_priority: int = 5
    queue_max_attempts: int = 2

    # Intermediate artifact cleanup (days kept before purge)
    artifact_retention_days: int = 7

    #: Paths reachable without a token even when auth_mode=required.
    public_paths: list[str] = [
        "/api/health",
        "/api/ready",
        "/api/metrics",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
        "/api/auth/login",
        "/api/auth/register",
    ]


class PlanningSettings(BaseSettings):
    """Mission planning + learning parameters (Phase 9).

    Camera/drone constants feed the physical simulation model; every output
    that depends on them is an estimate (see the ``method`` labels in module
    output). Learning thresholds gate what the history engine may claim:
    below ``min_history_for_ml`` no prediction is labelled as learned.
    """

    model_config = SettingsConfigDict(env_prefix="PLAN_")

    # --- camera model (nadir photogrammetry) ---
    camera_width_px: int = 5472
    camera_height_px: int = 3648
    sensor_width_mm: float = 13.2
    sensor_height_mm: float = 8.8
    focal_mm: float = 8.8

    # --- drone / battery physics proxy ---
    drone_mass_kg: float = 2.0
    battery_capacity_wh: float = 60.0
    battery_reserve_pct: float = 20.0
    hover_power_w_per_kg: float = 70.0
    drag_coeff: float = 0.06  # W per (m/s)^3 parasitic drag term
    avionics_w: float = 12.0

    # --- baseline scenario defaults (user-editable per request) ---
    default_altitude_m: float = 60.0
    default_speed_m_s: float = 8.0
    default_forward_overlap: float = 0.7
    default_side_overlap: float = 0.6
    default_pattern: str = "lawnmower"

    # --- optimization search grid ---
    optimize_alts_m: list[float] = [40.0, 50.0, 60.0, 80.0]
    optimize_speeds_m_s: list[float] = [5.0, 8.0, 12.0]
    optimize_overlaps: list[float] = [0.7, 0.8]
    max_candidate_plans: int = 300

    # --- coverage analysis ---
    coverage_grid_cells: int = 48
    blind_conf_threshold: float = 0.4
    obstacle_clearance_m: float = 15.0

    # --- mission history / learning gates ---
    min_history_for_stats: int = 1
    min_history_for_similar: int = 1
    min_history_for_ml: int = 5
    similar_top_k: int = 3
    quality_ref_altitude_m: float = 40.0  # GSD anchor for the quality index


# ---------------------------------------------------------------------------
# Root settings (composed from all groups)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root application settings — aggregates all sub-groups."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Sub-groups (loaded from their own env prefixes)
    server: ServerSettings = Field(default_factory=ServerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    ai: AISettings = Field(default_factory=AISettings)
    colmap: COLMAPSettings = Field(default_factory=COLMAPSettings)
    dense: DenseSettings = Field(default_factory=DenseSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    mesh: MeshSettings = Field(default_factory=MeshSettings)
    semantic: SemanticSettings = Field(default_factory=SemanticSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    intel: IntelSettings = Field(default_factory=IntelSettings)
    planning: PlanningSettings = Field(default_factory=PlanningSettings)
    platform: PlatformSettings = Field(default_factory=PlatformSettings)

    # Top-level
    app_name: str = "STRATA"
    debug: bool = False
    version: str = "0.1.0"
    deployment: str = "development"  # development|testing|production

    @model_validator(mode="after")
    def _ensure_directories(self) -> Settings:
        """Create critical directories on startup."""
        self.storage.base_dir.mkdir(parents=True, exist_ok=True)
        self.storage.temp_path.mkdir(parents=True, exist_ok=True)
        self.ai.weights_path.mkdir(parents=True, exist_ok=True)
        return self


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()


settings = get_settings()
