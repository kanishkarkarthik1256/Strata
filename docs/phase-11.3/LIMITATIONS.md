# STRATA Phase 11.3 — System Limitations & Bounds

## Overview
This document specifies technical limitations, scale bounds, and operational constraints of STRATA's real footage ingestion pipeline.

## Ingestion & Pipeline Limitations

1. **File Size & Format Bounds**:
   - Maximum upload file size: 2 GB (default settings).
   - Supported extensions: `.mp4`, `.mov`, `.avi`, `.mkv`.
   - Audio tracks present in video files are ignored (only video stream is processed).

2. **GPS & Telemetry Extraction**:
   - GPS metadata extraction relies on standard EXIF / MP4 container metadata tags.
   - Videos without embedded GPS metadata default to local spatial coordinate system `(0, 0, 0)`.
   - Metric scale validation requires GCP ground control points or dual-frequency RTK logs (uncalibrated single-camera video produces scale estimates from COLMAP / GPS alignment).

3. **Fallback Engine Characteristics**:
   - COLMAP / Depth Anything V2 are the primary reconstruction engines.
   - When running under fallback mode (pure OpenCV), point cloud density is limited compared to deep neural depth models.
   - The UI explicitly communicates fallback execution to ensure zero false representation.

4. **Resource Constraints**:
   - Pipeline processing runs asynchronously on the backend server. High resolution (4K) videos with > 1000 frames require sufficient GPU/CPU RAM during dense MVS fusion.
