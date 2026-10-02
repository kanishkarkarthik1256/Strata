# STRATA Phase 11.4 — System Limitations & Operational Bounds

## Overview
This document specifies technical limitations, performance observations, and operational bounds established during Phase 11.4 live validation.

## Systems & Performance Limitations

1. **Drone Video Test Footage**:
   - Technical video upload & processing pipeline: **VALIDATED**.
   - Real 4K physical drone flight footage validation: **NOT YET VALIDATED** (synthetic test AVI videos used for validation).

2. **Metric Scale & Accuracy**:
   - 3D reconstructions without ground control points (GCPs) or dual-frequency RTK EXIF logs compute scale from camera relative baseline or COLMAP/GPS alignment.
   - UI metrics and measurements strictly carry the **`Estimated Measurement`** / **`Not validated`** label.

3. **GPS Telemetry**:
   - Missing GPS telemetry causes georeferencing stage to skip.
   - Outputs without GPS remain in local SfM coordinate frame (metres).

4. **Cross-Platform Packaging Blockers**:
   - Desktop application build (Tauri/Electron) is blocked on FFprobe/COLMAP binary bundling per target OS architecture.
