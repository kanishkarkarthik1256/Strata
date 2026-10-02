# STRATA Phase 11.3 — Real Processing State Tracking

## Overview
This document specifies real-time pipeline execution tracking, stage transitions, and execution telemetry in STRATA.

## Real Pipeline Stages

| Stage | Name | Description | Fallback Behavior |
|-------|------|-------------|-------------------|
| 1 | `frame_extraction` | Keyframe selection from drone video stream | OpenCV VideoCapture |
| 2 | `feature_matching` | Keypoint detection & feature matching | pycolmap / OpenCV ORB |
| 3 | `sparse_reconstruction` | Structure-from-Motion (SfM) camera pose estimation | pycolmap / OpenCV Essential Matrix |
| 4 | `gps_alignment` | Sim3 rigid transformation to geographic coordinates | Direct GPS alignment / Identity transform |
| 5 | `depth_estimation` | Monocular depth map generation per keyframe | Depth Anything V2 / OpenCV stereo |
| 6 | `dense_reconstruction` | Multi-view stereo (MVS) point cloud fusion | Open3D MVS / Depth unprojection |
| 7 | `surface_mesh` | Poisson surface reconstruction / TSDF fusion | Open3D Poisson / Ball Pivoting |
| 8 | `export_artifacts` | Manifest creation & PLY / JSON file generation | Standard file serialization |

## Fallback Transparency System
To guarantee complete user honesty without synthetic numbers:
- If primary engine (COLMAP / Depth Anything V2) is unavailable or fails, the orchestrator seamlessly activates verified OpenCV fallback paths.
- Active fallback modes are recorded directly in stage detail metadata (`fallback_used: true`, `fallback_engine: "opencv"`).
- Frontend `ProcessingView` renders a high-visibility badge ("OpenCV Fallback Engine Active") so users know exactly which algorithm processed their footage.

## Real Progress Telemetry
- No fake timer loops or hardcoded percentage increments.
- State is polled live from `/api/projects/<PROJECT_ID>/pipeline/status`.
- Progress metrics reflect actual processed frames, extracted features, and fused 3D point counts.
