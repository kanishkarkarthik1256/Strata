# STRATA Phase 11.4 — Data Consistency & Isolation Audit

## Overview
This document evaluates data consistency across the Dashboard, Mission Detail, 3D Viewer, Analysis, Reports, and AI Copilot.

## Data Consistency Audit Matrix

| Component | Run ID Source | Point Count Source | GPS Fixes Source | Cross-Run Data Leak Risk | Status |
|-----------|---------------|-------------------|------------------|--------------------------|--------|
| **Dashboard** | `useRuns()` hook | PLY header / manifest | `r.gps_points` | None | **VERIFIED** |
| **Missions List** | `useRuns()` hook | PLY header / manifest | `r.gps_points` | None | **VERIFIED** |
| **3D Viewer** | URL parameter `runId` | PLY header / canvas loaded | `poses.json` / EXIF | None | **VERIFIED** |
| **Analysis Page** | Selected run state | `manifest.json` / PLY header | `r.gps_points` | None | **VERIFIED** |
| **Reports Page** | Selected run state | `manifest.json` / PLY header | `r.gps_points` | None | **VERIFIED** |
| **AI Copilot** | `CopilotPanel` prop `runId` | `answer_run_question()` PLY header parse | `poses.json` | None | **VERIFIED** |

## Audit Findings & Verification
1. **Dynamic Selection**: Navigating from Dashboard -> Missions -> Viewer -> Analysis -> Reports -> Copilot preserves the explicitly selected `runId`.
2. **Demo Mission Isolation**: The precomputed demo mission `shitan_ms1_20260909_131251` is stored in `outputs/` and clearly tagged `PRECOMPUTED DEMO`. Live user runs in `data/storage/` NEVER inherit demo metadata.
3. **No Stale Context in Copilot**: Copilot requests `POST /api/runs/${runId}/ask` pass the current `runId`. Switching runs in the UI instantly invalidates previous copilot message context.
