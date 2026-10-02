# STRATA Phase 11.4 — AI Copilot & Grounding Validation

## Overview
This document details the AI Copilot grounding verification, question answering accuracy, and data leak prevention tests.

## Grounded Q&A Verification

| Test Question | Expected Grounded Answer | Verification Result |
|---------------|--------------------------|---------------------|
| `"What are the key reconstruction statistics for this mission?"` | Status, stage counts, and metrics derived strictly from current run's `manifest.json`. | **VERIFIED** |
| `"How many points are in the reconstruction?"` | Point count extracted from `dense_model.ply` header (e.g. 2,626,577). | **VERIFIED** |
| `"Is the reconstruction metrically validated?"` | Acknowledges metric 3D validation is NOT validated (scale estimated from COLMAP/GPS). | **VERIFIED** |
| `"What is the GPS coordinate?"` (No-GPS Run) | Explicitly states: `"GPS unavailable — no GPS telemetry is recorded for this run"`. | **VERIFIED** |

## Data Leakage Prevention
- **Test Case**: Querying Copilot on a newly created mission ID `proj_123` must NEVER return point counts or statistics from `shitan_ms1_20260909_131251`.
- **Validation**: `answer_run_question(run_id, query)` resolves `run_dir(run_id)` independently. If `manifest.json` does not exist for `run_id`, it raises `RunNotFoundError(404)` rather than leaking previous or demo context.
