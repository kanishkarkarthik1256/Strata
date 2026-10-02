# STRATA Phase 11.3 — Mission Lifecycle Specification

## Overview
This document defines the lifecycle states and transitions for real missions in STRATA.

## Mission State Machine

```
   [ Draft / Form ]
          |
          | User uploads video
          v
   [ Uploaded ]  (Project record created in DB, workspace initialized)
          |
          | User triggers "START RECONSTRUCTION"
          v
   [ Processing ] (Orchestrator running stages 1-8 asynchronously)
       /      \
      /        \
     v          v
[ Completed ]  [ Failed ]
```

## Lifecycle States

1. **Uploaded (`uploaded`)**:
   - Video file successfully saved to `data/storage/<PROJECT_ID>/<filename>`.
   - Metadata extracted and validated.
   - Project entry present in database.

2. **Processing (`processing`)**:
   - Backend pipeline orchestrator active in background thread.
   - Stage progression monitored via `/api/projects/<PROJECT_ID>/pipeline/status`.
   - Execution detail logged per stage (frame_extraction, feature_matching, sparse, alignment, depth, dense, mesh, export).

3. **Completed (`completed`)**:
   - All reconstruction stages finished successfully.
   - Workspace contains generated `manifest.json`, `dense/dense_model.ply`, `sparse/sparse_model.ply`, `poses.json`, and reports.
   - Mission accessible in 3D Viewer, Analysis, Reports, and Copilot.

4. **Failed (`failed`)**:
   - Pipeline terminated due to error (e.g. insufficient feature matches, missing dependencies, or unrecoverable processing error).
   - Detailed failure reason logged in stage status detail and `manifest.json`.
   - User provided with clear retry CTA or diagnostic details.

## Dual-Root Discovery Lifecycle
- **Demo Mission (`shitan_ms1_20260909_131251`)**: Resides in `outputs/`. Labeled clearly as `PRECOMPUTED DEMO`.
- **User Missions**: Reside in `data/storage/<PROJECT_ID>/`.
- `run_service.py` transparently scans both roots so all features (3D Viewer, Copilot, Reports) operate identically for demo and live real missions.
