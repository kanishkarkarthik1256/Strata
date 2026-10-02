# STRATA — SIH Requirements Compliance Matrix

Status legend: GREEN production-ready / YELLOW present, needs hardening / RED gap.
Every non-green row has a concrete resolution path at the bottom.

| # | SIH requirement | What STRATA has today | Status | Evidence / location |
|---|---|---|---|---|
| 1 | Single drone pass | Reconstructs exactly one continuous input sequence; run report carries `single_pass_enforced: true, input_sequence_count: 1` on every run — auditable, not a convention. | GREEN | `pipeline_orchestrator._finish` marker |
| 2 | Drone video input | Full frame extraction (OpenCV decode, quality gates, phash dedup, duration-aware keyframe budget). mp4/mov/avi/mkv/m4v/webm. | GREEN | `frame_extractor.py` |
| 3 | GPS / flight metadata | Three ingestion paths: embedded GPS; external CSV with schema auto-detection; **DJI `.srt` flight logs** (now accepted at upload, converted to per-frame metric `flight_poses.csv` with sync-offset estimation). | GREEN | `telemetry_schema.py`, `dji_srt_telemetry.py`, `upload_service.py` |
| 4 | SfM | pycolmap incremental SfM, SIFT (GPU-auto) + sequential/exhaustive matching, STRATA explicit joint BA with real observations, telemetry-assisted pose prior. | GREEN | `camera_pose_estimator.py`, `bundle_adjustment.py` |
| 5 | Dense reconstruction | Depth Anything V2 (CUDA/MPS/CPU auto) -> scale-aligned depth (per-view + global affine) -> cross-view-consistent fusion -> filtered metric cloud, gated by an honest A-G depth audit. | GREEN | `depth_generator.py`, `depth_fusion.py`, `depth_diagnostics.py` |
| 6 | Textured 3D model | Multi-view texturing built and validated as an experiment (airport3): xatlas UV unwrap, frustum-gated view selection, exposure normalization, occlusion-tested bake, 11,640^2 atlas at 9.2 cm/texel; texture-content registration verified against imagery. | YELLOW — not yet the production default | `scripts/phase3_texture_experiment.py` / `_rebake.py`; `mesh/phase3_texture/` |
| 7 | Georeferencing | Telemetry-based ENU placement with sufficiency gating, honest VIDEO_ONLY mode, GPS in poses.json, viewer alignment metadata. | GREEN | `telemetry.py`, `georeferencing.py`, `_stage_georef` |
| 8 | Metric accuracy | Formal 3-tier validation (relative / scale / absolute) with independent LiDAR, bidirectional NN + point-to-plane, Umeyama registration, per-scene validation_report.json, STRATA <=1 m engineering criterion (3D RMSE AND P95), UI Accuracy card with four honest statuses. Honest state: airport3 horizontal 0.24 m median, vertical 1.05 m; no scene certified <=1 m yet (vertical bias is the dominant error). | GREEN (layer) / YELLOW (result) | `metric_validation.py`, Reports Accuracy card |
| 9 | Limited viewing angles | Not yet compensated: occluded/steeply-viewed surfaces stay unobserved (airport3: ~30% of faces had no in-frustum candidate). | RED — R1 | resolution plan below |
| 10 | Motion blur / compression | Blur/exposure gates at extraction; depth model is tolerant; compression artifacts still flow into matching. | YELLOW — R2 | resolution plan below |
| 11 | Variable illumination | Texturing does exposure normalization + per-view gain; extraction gates extremes; SfM stages unaddressed. | YELLOW — R2 | resolution plan below |
| 12 | Dynamic objects | No masking/detection; moving objects ghost into cloud/mesh. | RED — R3 | resolution plan below |
| 13 | GPS noise | Jump detection (>200 m AND >200 m/s -> SUSPECT) + per-frame quality report; SRT zero-order-hold + sync-offset estimation; no trajectory smoothing model yet. | YELLOW — R4 | resolution plan below |
| 14 | Occluded surfaces | Fundamental single-pass limitation; handled honestly — surfaces absent, never hallucinated, no forced watertightness. | RED (inherent) — R5 | resolution plan below |
| 15 | Near-real-time | Optimization campaign: 2.5x end-to-end on CPU (82.8 -> ~33 min for the calibration clip), thread saturation, feature cap 10240, keyframe budget, BA consolidation (-36% sparse), CUDA auto-enablement. | YELLOW — R6 | resolution plan below |
| 16 | Learned depth / neural | Depth Anything V2 IS the learned-depth core (monocular priors scale-aligned to SfM). Missing: a neural degraded-mode fallback when SfM is weak. | YELLOW — R7 | resolution plan below |

## Resolution plan (ordered by impact against red/yellow rows)

- **R1 Limited viewing angles.** Surface the per-face view_support count the texturing stage already computes into quality reports; camera-frustum overlay in the viewer colored by support so unobserved regions are visible pre-report; flight-planning hint generator (advisory additional-pass angles from coverage gaps — the single-pass reconstruction contract is unchanged).
- **R2 Blur/compression/illumination.** Thread the already-measured per-view quality score into matching (drop lowest-quality frames from pair generation first) and texture view selection; deterministic exposure normalization before descriptor computation.
- **R3 Dynamic objects.** Multi-view outlier rejection in depth fusion: dynamic objects violate static-scene cross-view consistency; points whose disagreement exceeds the static band get a `dynamic_suspect` provenance flag and are excluded from meshing (evidence preserved in the cloud, never deleted).
- **R4 GPS noise.** Robust trajectory filter before telemetry-assisted localization: RANSAC-style polynomial fit over ENU samples (reject >3-sigma residuals), with a filter report (samples removed, residual before/after) persisted in telemetry artifacts. SRT sync-offset already handles the temporal part.
- **R5 Occluded surfaces.** Cannot be solved within the single-pass contract; R1 makes the limitation visible and measurable. AI completion is prohibited from presenting unobserved geometry as reconstruction (project rule).
- **R6 Near-real-time.** Remaining bottleneck: COLMAP mapping (~60% of sparse). Next: matching-graph reduction (sequential window + telemetry-guided candidate pairs — the pair prior already exists in the benchmark harness). GPU stages auto-enable via CUDA.
- **R7 Neural fallback.** When SfM registers <70% of frames, offer Depth-Anything-only reconstruction (telemetry- or median-aligned scale) as a labeled degraded mode — flagged in the report, never presented as full photogrammetric quality.

## SRT compatibility — implementation summary

- `safe_telemetry_filename` accepts `.srt` (backend allowlist was CSV-only).
- Upload path: SRT saved as `telemetry_upload.srt`, converted at upload to `flight_poses.csv` (per-frame ENU + gimbal quaternion + sync-offset estimate) plus `srt_telemetry_provenance.json`; the upload response surfaces frame count and sync offset.
- Data-folder route `start-with-telemetry` accepts `.srt` via the new `flight_poses_path` parameter into the mission starter.
- The sparse stage consumes `flight_poses.csv` automatically (telemetry-assisted SfM); GPS lands in `poses.json` -> georef -> metric validation — no further wiring.
- Frontend telemetry picker accepts `.srt`, labels updated, and the start payload only sends `telemetry_csv` for CSV files (SRT needs no key).
- 6 new tests (extension gate, parse, convert, monotonic trajectory, non-DJI rejection, fps guard).
