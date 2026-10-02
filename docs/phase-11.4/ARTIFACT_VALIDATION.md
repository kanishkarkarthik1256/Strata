# STRATA Phase 11.4 — Artifact Generation & Inventory Audit

## Overview
This document specifies automated artifact inventory generation and validation procedures in STRATA.

## Artifact Inventory Protocol

When a reconstruction pipeline completes, the orchestrator writes the following files to `data/storage/<PROJECT_ID>/`:

| Artifact Path | Required File | Format | Validation Criteria |
|---------------|---------------|--------|---------------------|
| `manifest.json` | Yes | JSON | Valid JSON containing pipeline version, run ID, status, and metrics |
| `poses.json` | Yes | JSON | Frame poses array with rotation/translation matrices |
| `sparse/sparse_model.ply` | Yes | Binary PLY | Header contains `element vertex <N>` |
| `dense/dense_model.ply` | Yes | Binary PLY | Header contains `element vertex <N>` |
| `mesh/surface_mesh.ply` | Optional | Binary PLY | Generated when surface mesh stage succeeds |
| `reports/summary_report.json` | Yes | JSON | Detailed stage timing and execution summary |

## Automatic Inventory Generation
`run_service._artifact_inventory(run)` inspects the workspace directory using `rglob("*")` filtered against `_ARTIFACT_EXTENSIONS` (`.ply`, `.obj`, `.glb`, `.gltf`, `.json`, `.npy`, `.png`, `.jpg`, `.jpeg`, `.txt`, `.md`, `.html`), returning an inventory array with relative path and file size bytes.
