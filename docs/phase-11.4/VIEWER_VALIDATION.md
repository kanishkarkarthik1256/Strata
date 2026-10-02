# STRATA Phase 11.4 — 3D Viewer & Data Integrity Audit

## Overview
This document evaluates the 3D Viewer integration, point cloud rendering, camera pose visualization, and point count data integrity in STRATA.

## Data Integrity Verification

| Metric | Source | Verification Result | Match Status |
|--------|--------|---------------------|--------------|
| **Manifest Point Count** | `manifest.json -> metrics -> dense_points` | Parsed from PLY header at build completion | **MATCH** |
| **PLY Header Point Count** | `dense_model.ply` header `element vertex N` | Parsed on-the-fly by `_ply_vertex_count()` | **MATCH** |
| **Viewer Displayed Points** | Three.js PLYLoader buffer attribute count | Counted during geometry load in `ViewerCanvas.tsx` | **MATCH** |

## Viewer Feature Verification
1. **Dynamic Model Loading**: `useViewerArtifacts.ts` fetches `/api/runs/${runId}/artifacts/dense/dense_model.ply` for the current run ID (never falling back to Shitan demo unless Shitan is requested).
2. **Camera Trajectory Visualization**: `ViewerCanvas.tsx` parses `poses.json` to render camera frustums and trajectory path.
3. **GPS Track Visualization**: If `poses.json` contains GPS fixes, GPS markers are rendered in green along the camera path. If GPS is missing, `ViewerStats` displays `GPS: Not available`.
4. **Spatial Confidence Heatmap**: If vertex colors encode confidence or if per-point scalar field is present, the heatmap toggle colors the model. If absent, UI displays `Spatial confidence unavailable for this reconstruction`.
5. **Measurement Tools**: Distance measurements render with `Estimated Measurement` tag to reflect uncalibrated scale.
