# STRATA Phase 11.4 — Phase 11.3 Claim Verification Audit

## Overview
This document cross-checks every major claim from Phase 11.3 documentation against actual source code, API responses, generated files, and execution state.

## Verification Matrix

| Claim | Evidence | Status | Action |
|-------|----------|--------|--------|
| **1. STRATA Rebranding** | `backend/app/config/settings.py` has `app_name = "STRATA"`; Frontend displays STRATA header logo. | **VERIFIED** | Maintain branding across all modules. |
| **2. Preserved Backend Architecture** | COLMAP, pycolmap, Depth Anything V2, OpenCV fallbacks, SQLite DB, and orchestrator intact. | **VERIFIED** | Preserved without modifications to core pipelines. |
| **3. Streamed Video Upload & Sanitization** | `upload_service.py` streams chunks with `safe_filename()` enforcing safe stems and extensions. | **VERIFIED** | Retained with cross-platform backslash normalization. |
| **4. Dual-Root Run Discovery** | `run_service.py` scans both `outputs/` (demo runs) and `data/storage/` (project workspaces). | **VERIFIED** | Verified with test suite (`test_runs.py`). |
| **5. Pre-Flight Input Quality Card** | `NewMissionModal.tsx` renders resolution, duration, FPS, and GPS fix status prior to trigger. | **VERIFIED** | Verified in UI modal. |
| **6. Real Progress Telemetry** | `ProcessingView.tsx` polls `/api/projects/<ID>/pipeline/status` for real stage states without fake timers. | **VERIFIED** | Verified live backend polling. |
| **7. Fallback Transparency Badge** | High-visibility badge rendered in `ProcessingView.tsx` when OpenCV fallback engine executes. | **VERIFIED** | Verified in UI component. |
| **8. Grounded AI Copilot Q&A** | `answer_run_question()` parses actual PLY headers and `manifest.json` for selected run ID. | **VERIFIED** | Verified un-hallucinated stats. |
| **9. Missing GPS Handling** | `metadata_extraction.py` substituted `(0.0, 0.0, 0.0)` when GPS EXIF was absent. | **FAILED** | **REPAIR REQUIRED**: Change fallback so missing GPS is `None` / `GPS: NOT AVAILABLE` rather than `(0,0,0)`. |
| **10. Video Metadata Nomenclature** | Ordinary MP4 container metadata was referred to as "EXIF". | **PARTIALLY VERIFIED** | **REPAIR REQUIRED**: Distinguish container metadata (FFprobe/OpenCV) from true EXIF/XMP/DJI metadata. |
| **11. Real Drone Video Validation** | Repository contains standard synthetic/test video files, but no high-res real drone flight footage. | **NOT VERIFIED** | Label technical pipeline as VALIDATED, but mark REAL DRONE VIDEO VALIDATION as NOT YET VALIDATED. |
