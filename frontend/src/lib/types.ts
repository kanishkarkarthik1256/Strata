/**
 * Backend contract types — shaped from the ACTUAL FastAPI responses
 * (see docs/phase-11.1/ACTUAL_API_CONTRACT.md). These are the single source
 * of truth for every API call; do not invent fields the backend does not send.
 */

/* ---- GET /api/system/capabilities (host resource report) ---- */
export interface HostCapabilities {
  compute: {
    cpu: { cores: number | null; load_1m: number | null; count: number | null };
    gpu: {
      available: boolean;
      reason?: string | null;
      device?: string | null;
      total_memory_bytes?: number | null;
      free_memory_bytes?: number | null;
      cuda_version?: string | null;
      device_index?: number | null;
      cuda_available?: boolean;
      mps_available?: boolean;
      torch_version?: string | null;
    };
    device: string;
  };
  memory: { process_rss_bytes: number | null };
  disk: { path: string; total_gb: number; free_gb: number; used_gb: number };
  tools: { colmap: boolean; ffprobe: boolean; ffmpeg: boolean };
  /** Measured runtime dependency versions (backend probes import/which). */
  dependencies?: {
    python: string | null;
    torch: string | null;
    pycolmap: string | null;
    opencv: string | null;
    ffmpeg: string | null;
    colmap: string | null;
  };
  /** Real smoke-test evidence from the backend (null = failed/not run). */
  smoke_tests?: {
    depth?: { ok: boolean; device?: string; infer_seconds?: number } | null;
    colmap?: { ok: boolean; images_registered?: number; points3d?: number; mean_reproj_px?: number; seconds?: number } | null;
    video_decode?: { ok: boolean; video?: string; frames_total?: number; resolution?: string } | null;
  };
  depth_model?: {
    installed: boolean;
    detected: boolean;
    /** True only after a real inference smoke test in this process. */
    smoke_tested?: boolean;
    checkpoint: string | null;
    status_level: string;
  };
  deployment?: string;
  auth_mode?: string;
  version?: string;
}

/* ---- GET /api/system/queue ---- */
/* ---- Phase 9.5.1 run manifest (manifest.dependencies) ---- */
export interface RunDependencies {
  python_version?: string;
  opencv_version?: string;
  numpy_version?: string;
  cuda_available?: boolean;
  gpu_name?: string | null;
  mps_available?: boolean;
  torch_version?: string;
  device?: string;
  colmap_available?: boolean;
  pycolmap_version?: string | null;
  ffmpeg_available?: boolean;
  depth_anything_checkpoint?: string | null;
  status?: string;
}

export interface RunArtifact {
  path: string;
  size_bytes: number;
}

/* ---- GET /api/runs ---- */
/** Ground-truth DSM accuracy for runs whose dataset ships a reference grid.
 * Served by /api/reconstruction/dsm-accuracy/{run_id}; a 404 means the run
 * has no registered reference and the UI renders no accuracy claim.
 * Horizontal placement is measured (co-registration), never assumed —
 * ``alignment_gate_failed`` means the geometry could not be registered
 * better than decorrelated terrain, so no score is served. */
export interface DsmAccuracy {
  status: string;
  mae_m?: number;
  p95_m?: number;
  height_datum_offset_m?: number;
  coverage?: number;
  reference?: string;
  gate_mae_m?: number;
  measured_offset_m?: [number, number];
  registration_coverage?: number;
}

export interface RunSummary {
  run_id: string;
  dataset: string | null;
  mission: string | null;
  status: string | null;
  /** Discovered under the legacy precomputed-demo root (backend/outputs). */
  is_demo?: boolean;
  pipeline_version?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  stages: Record<string, string | null>;
  metrics: Record<string, unknown>;
  dependencies: RunDependencies;
  limitations: string[];
  artifacts: RunArtifact[];
  dense_points?: number | null;
  sparse_points?: number | null;
  mesh_vertices?: number | null;
  mesh_faces?: number | null;
  cameras?: number | null;
  gps_points?: number | null;
  mean_reprojection_error_px?: number | null;
  bundle_adjustment?: Record<string, unknown>;
  /** Sparse↔dense NN consistency from the dense stage (mandate §22). */
  sparse_dense?: {
    correspondences: number | null;
    median_m: number | null;
    p95_m: number | null;
    rmse_m: number | null;
    within_3m_pct?: number | null;
    screened?: {
      correspondences: number | null;
      dropped: number | null;
      median_m: number | null;
      p95_m: number | null;
      rmse_m: number | null;
      note?: string;
    } | null;
  } | null;
  /** Mesh quality audit (giant triangles / components / orientation). */
  mesh_quality?: {
    giant_faces: number | null;
    components: number | null;
    largest_component_pct: number | null;
  } | null;
}

export interface RunDetail extends RunSummary {
  manifest: Record<string, unknown>;
}

export interface RunListResponse {
  runs: RunSummary[];
}

/* ---- GET /api/runs/{run_id}/artifact/poses.json ---- */
export interface PoseFrame {
  frame_id: string;
  K: number[][];
  R: number[][];
  t: number[];
  gps?: { lat: number; lon: number; alt: number };
}

export interface PosesJson {
  frames: PoseFrame[];
}

/** Rigid presentation transform served by GET /api/runs/{id}/viewer-alignment.
 *  X_view = R_view @ X_world + t_view — ground normal → +Y, corridor → +X.
 *  Presentation-only: authoritative reconstruction artifacts are never modified. */
export interface ViewerAlignment {
  version: number;
  run_id: string;
  source_coordinate_convention: string;
  viewer_coordinate_convention: string;
  /** Row-major 3×3: view = R_view · world (rows are the viewer basis in world coords). */
  R_view: number[][];
  t_view: number[];
  scale: number;
  ground_normal_world: number[] | null;
  ground_reference_point_world: number[] | null;
  horizontal_direction_world: number[] | null;
  method: string;
  confidence: string;
  /** Which evidence fixed the viewer's up axis, and how far it can be trusted. */
  leveling?: {
    up_prior_source: string | null;
    confidence: string;
    plane_measured: boolean;
    note?: string;
  } | null;
  fallback_reason?: string;
  created_at: string;
}

/* ---- GET /api/missions (auth required) ---- */
export interface Mission {
  id: string;
  project_id: string | null;
  name: string;
  status: string;
  priority: number;
  current_stage: string | null;
  stage_progress: number;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
  completed_at: string | null;
  owner?: { id: string; email: string; role: string } | null;
}

export interface MissionListResponse {
  missions: Mission[];
  count: number;
}

/* ---- /api/auth ---- */
export interface AuthUser {
  id: string;
  email: string;
  role: string;
}

export interface LoginResponse {
  token: string;
  user: AuthUser;
  token_type: string;
}

export interface RegisterResponse {
  id: string;
  email: string;
  role: string;
}

/* ---- Upload & Video Ingestion Types (Phase 11.3) ---- */
export interface VideoMetadata {
  filename: string;
  duration_sec: number;
  fps: number;
  width: number;
  height: number;
  codec: string;
  frame_count: number;
  bitrate_kbps: number;
  file_size_bytes: number;
  creation_time?: string | null;
  gps_lat?: number | null;
  gps_lon?: number | null;
  gps_alt?: number | null;
  camera_make?: string | null;
  camera_model?: string | null;
}

export interface UploadResponse {
  job_id: string;
  status: string;
  message: string;
  /** Server-measured video properties (post-validation); absent on older backends. */
  metadata?: VideoMetadata | null;
}

export interface DataVideoInfo {
  name: string;
  size_bytes: number;
  duration_sec?: number;
  width?: number;
  height?: number;
  fps?: number;
  /** GPS/pose logs sitting beside the video on the server (poses.csv,
   *  movingdrone_telemetry.csv, video.SRT). Empty = video-only dataset, which
   *  the mission form refuses to start. */
  gps_sources?: string[];
  /** Held-out LiDAR references beside the video (LAS/LAZ/NPZ). Optional: a
   *  dataset without one reconstructs normally and simply reports no absolute
   *  accuracy, because there is nothing independent to measure against. */
  lidar_sources?: string[];
}

export interface DataVideoListResponse {
  videos: DataVideoInfo[];
  count: number;
}

export interface DataVideoStartResponse {
  run_id: string;
  video: string;
  status: string;
  message: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  filename: string;
  metadata?: VideoMetadata | null;
  workspace_path: string;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface PipelineStageDetail {
  status: string;
  duration_ms: number;
  count: number;
  error: string;
  detail: Record<string, unknown>;
  /** Live fraction (0..1) from stage events; absent in finished-run reports. */
  progress?: number | null;
}

/* ---- Depth / dense stage diagnostics (honest surfacing) ---- */

/** Why one depth view failed before or during inference. */
export interface DepthViewFailure {
  frame: string;
  reason: string;
}

/** A depth view that was generated but excluded from fusion, with the named
 *  conditioning reason the backend refused it. */
export interface DepthViewExclusion {
  frame: string;
  reason: string;
}

/** Depth-stage detail payload (subset; older backends omit newer keys). */
export interface DepthStageDetail {
  generated?: string[];
  cached?: string[];
  failed?: string[];
  excluded?: DepthViewExclusion[];
  failure_reasons?: Record<string, string>;
  count_generated?: number;
  count_excluded?: number;
  backend?: string;
}

/** Sparse-stage telemetry placement provenance (a degraded placement still
 *  completes the stage, so this is the only signal the user gets). */
export interface PlacementStageDetail {
  mode?: string | null;
  matched_cameras?: number | null;
  match_percent?: number | null;
  points_retained_fraction?: number | null;
  refused?: {
    reason?: string;
    points_before?: number;
    points_after?: number;
    points_retained_fraction?: number;
    retained_floor?: number;
    observations_dropped?: number;
    note?: string;
  } | null;
}

/** Dense-stage refusal detail: the exact A–G criteria that failed. */
export interface DenseStageDetail {
  failing_criteria?: string[];
  audited_count?: number;
  dominant_failure_reason?: string;
  reason?: string;
}

export interface PipelineStartResponse {
  job_id: string;
  /** "queued" — execution belongs to the durable queue worker, not the HTTP request. */
  status: string;
  run_time_ms?: number;
  error?: string;
  stages?: Record<string, PipelineStageDetail>;
  message: string;
}

export interface PipelineStatusResponse {
  job_id: string;
  status: string;
  run_time_ms: number;
  error: string;
  stages: Record<string, PipelineStageDetail>;
  profile: Record<string, unknown>;
  resume: Record<string, boolean>;
}
/* ---- GET /api/runs/{id}/analysis (area/elevation/flight/error engine) ---- */

/** Area computed by the analysis engine. Two honest methods are always
 *  present: density-supported occupied cells (the ground actually modelled)
 *  and the convex hull (the site covered, gaps included). */
export interface AreaSite {
  units: string;
  cell_size_m: number;
  area_m2: number;
  area_ha: number;
  area_km2: number;
  area_method: string;
  outline_area_m2: number;
  outline_area_km2: number;
  outline_method: string;
  occupied_cells: number;
  bbox: { min: [number, number]; max: [number, number]; size: [number, number] };
  centroid: { enu: [number, number]; wgs84: { lat: number; lon: number } | null };
  outline_wgs84: [number, number][] | null;
  footprint_cells: { cell_size_m: number; origin_enu: [number, number]; cells: [number, number, number][] } | null;
}

export interface AreaElevationPoint {
  enu: [number, number, number];
  wgs84: { lat: number; lon: number } | null;
  alt_m: number | null;
}

export interface AreaElevation {
  units: string;
  surface: string;
  lowest: AreaElevationPoint | null;
  highest: AreaElevationPoint | null;
  min: number;
  max: number;
  relief: number;
  mean: number;
  median: number;
  robust_min: number;
  robust_max: number;
  robust_relief: number;
  histogram: { bin_start_m: number; count: number }[] | null;
  datum: string;
}

export interface AreaFlight {
  available: boolean;
  points: number;
  path_length_m: number | null;
  mean_speed_m_s: number | null;
  max_speed_m_s: number | null;
  altitude_std_m: number | null;
  discontinuities: number;
  drift_m: number | null;
  gps_score: number | null;
  grade: string | null;
  track_wgs84: [number, number][] | null;
  agl_min_m: number | null;
  agl_max_m: number | null;
  agl_mean_m: number | null;
  gsd_m_per_px: number | null;
}

/** Per-cell sparse↔dense NN error grid (m). Cell = [ix, iy, median_m, n, p95_m]. */
export type AreaErrorCell = [number, number, number, number, number];

export interface AreaErrorMap {
  available: boolean;
  cell_size_m: number;
  min_points_per_cell: number;
  origin_enu: [number, number];
  cells: AreaErrorCell[];
}

/** Metric-validation block embedded in the analysis payload (may be null when
 *  the run has no validation artifact — the page must handle both). */
export interface AreaMetricValidation {
  validation_kind: string;
  certification_status: string;
  certification_reason: string;
  internal_validation: {
    label: string;
    verified: boolean;
    checks: {
      name: string;
      measured: boolean;
      value: number | null;
      unit: string;
      criterion: string;
      pass: boolean | null;
      note?: string;
    }[];
  } | null;
}

export interface RunAnalysis {
  run_id: string;
  generated_at: string;
  frame: {
    kind: string;
    metric: boolean;
    units: string;
    anchor_wgs84: { lat: number; lon: number; alt: number } | null;
    alignment_scale: number | null;
    source: string;
  };
  sources: { surface: string; surface_points: number; accuracy_map: string | null };
  site: AreaSite | null;
  elevation: AreaElevation | null;
  density: { points: number; points_per_m2: number; mean_spacing_m: number; coverage_percent: number; occlusion_percent: number } | null;
  flight: AreaFlight | null;
  consistency: { correspondences: number; median_m: number; p95_m: number; within_3m_pct: number; measure: string; source: string } | null;
  accuracy_map: AreaErrorMap;
  available: boolean;
  notes: string[];
  metric_validation: AreaMetricValidation | null;
}

/* ---- GET /api/runs/{id}/manifest (canonical, written by the pipeline) ---- */

/** Provenance of the exact input that produced a run. Historical runs that
 *  predate provenance report `{ kind: "unknown" }` — never guessed. */
export interface RunSource {
  kind: "uploaded" | "data_video" | "dataset" | "synthetic_test" | "unknown";
  original_filename?: string;
  sha256?: string;
  file_size_bytes?: number;
  created_at?: string;
  duration_sec?: number;
  width?: number;
  height?: number;
  fps?: number;
  dataset_name?: string;
  sequence_id?: string;
  dataset_relative_path?: string;
}

export interface RunManifest {
  run_id: string;
  dataset: string;
  mission: string;
  status: string;
  pipeline_version: string;
  source: RunSource;
  stages: Record<string, { status?: string; detail?: Record<string, unknown>; [k: string]: unknown }>;
  metrics: Record<string, unknown>;
  telemetry?: { mode?: string; [k: string]: unknown } | null;
  started_at?: string | null;
  completed_at?: string | null;
  git_commit?: string | null;
  limitations?: string[];
  input?: Record<string, unknown>;
}
