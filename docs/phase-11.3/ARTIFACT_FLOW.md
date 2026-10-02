# STRATA Phase 11.3 — Artifact Generation & Serving Flow

## Overview
This document specifies how reconstruction artifacts are stored, indexed, and served to the 3D Viewer, Copilot, and Reports modules.

## Artifact Directory Layout

```
data/storage/<PROJECT_ID>/
├── <video_filename>.mp4
├── manifest.json
├── poses.json
├── frames/
│   ├── frame_000000.jpg
│   └── ...
├── sparse/
│   ├── sparse_model.ply
│   └── reconstruction_report.json
├── dense/
│   └── dense_model.ply
├── mesh/
│   └── surface_mesh.ply
└── reports/
    └── summary_report.json
```

## Security & Path Traversal Guarantees
- `run_service.resolve_artifact(run_id, artifact_path)` strictly validates artifact requests:
  - Validates `run_id` pattern (`^[a-zA-Z0-9][a-zA-Z0-9_.-]*$`).
  - Verifies file extensions against allowlist (`.ply`, `.obj`, `.glb`, `.gltf`, `.json`, `.npy`, `.png`, `.jpg`, `.jpeg`, `.txt`, `.md`, `.html`).
  - Confirms candidate path is contained within the run directory (`is_relative_to(run_dir)`).
- Serves HTTP 404 for missing files and HTTP 400 for forbidden extension/traversal attempts.

## Downstream Integration
1. **3D Viewer**: Requests `dense/dense_model.ply` or `sparse/sparse_model.ply` directly via `/api/runs/<RUN_ID>/artifacts/dense/dense_model.ply`.
2. **AI Copilot**: Reads `manifest.json`, `poses.json`, and header-parsed PLY point counts to answer user questions using verified, un-hallucinated numbers.
3. **Reports & Exports**: Uses `manifest.json` metrics and stage timings to render PDF/JSON site survey reports.
