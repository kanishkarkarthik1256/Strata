# STRATA Phase 11.4 — GPS Telemetry Policy & Validation Audit

## Overview
This document specifies the GPS telemetry policy, missing GPS handling, and georeferencing enforcement in STRATA.

## Missing GPS Policy

> [!IMPORTANT]
> **Zero Coordinate Fallback Prohibition**:
> STRATA strictly prohibits substituting `(0, 0, 0)` (0 latitude, 0 longitude, 0 altitude) or any fake coordinates for missing telemetry.
> When GPS telemetry is absent from uploaded footage or EXIF tags, the system MUST report:
> **`GPS: NOT AVAILABLE`**

## Implementation & Audit Details

1. **Metadata Extraction (`metadata_extraction.py`)**:
   - `gps_lat`, `gps_lon`, and `gps_alt` return `None` when container/EXIF tags do not contain location metadata.
   - No default coordinates or geographic sentinels are injected.

2. **Pipeline Georeferencing Stage (`pipeline_orchestrator._stage_georef`)**:
   - Checks `if not gps_points`.
   - If no GPS points exist, the georeferencing stage records:
     `note: "no GPS telemetry available — georeferencing skipped (outputs stay in local coordinates)"`
   - **No georeferenced output files (`gps_track.csv`, `dense_model_enu.ply`) are produced from missing GPS**.

3. **UI Representation**:
   - `NewMissionModal.tsx`: Displays `GPS Telemetry: Not Available` badge when `metadata?.gps_lat == null`.
   - `ViewerStats.tsx`: Renders `GPS: Not available` when GPS count is null/zero.
   - `Dashboard.tsx`: Displays `—` or `Not available` in the GPS column for runs without telemetry.
   - `Analysis.tsx`: Renders `GPS fixes: Not available`.

4. **Copilot Grounded Q&A (`run_service.py`)**:
   - Answers `GPS unavailable — no GPS telemetry is recorded for this run` when queried about coordinates on a non-GPS run.

## Regression Verification
- Automated regression suite added in `backend/tests/test_gps_validation.py` verifying `None` returns, stage skipping without fake origin files, and honest Copilot answers.
