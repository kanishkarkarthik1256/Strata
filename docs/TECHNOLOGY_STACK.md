# STRATA — Technology Stack & Tooling Inventory

**Project:** STRATA (a.k.a. Aeromap-3D) — AI-enabled drone video → 3D model reconstruction platform
**Repo:** `drone-recon/` (FastAPI backend + React/Three.js frontend + Tauri desktop shell)
**Problem context:** SIH26158 — Robotics and Drones

Everything below is read from the actual dependency manifests, import scans, and installed packages in this checkout (September 2026).

---

## 1. Architecture at a Glance

```
drone-recon/
├── backend/          FastAPI service — full reconstruction pipeline (Python)
│   ├── app/routes/       16 HTTP route modules (auth, upload, pipeline, runs, demo, …)
│   ├── app/services/     96 service modules — the actual geometry/ML/IO engine
│   ├── app/db/           SQLAlchemy async models, engine, migrations (SQLite)
│   ├── app/schemas/      Pydantic request/response models
│   ├── models/weights/   depth_anything_v2_vits.pth (Depth Anything V2 ViT-S)
│   ├── scripts/          validation gates, profiling, batch drivers
│   └── tests/            pytest suite (unit + integration)
├── frontend/         React SPA — upload, missions dashboard, 3D viewer
│   ├── src/pages/        NewMission, Processing, Dashboard, Viewer, …
│   ├── src/hooks/        useRuns (run selection/context, localStorage persistence)
│   ├── src/lib/          api.ts — typed API client
│   └── src-tauri/        Tauri desktop shell (strata-desktop)
└── docs/             Phase reports & validation documentation
```

**Pipeline (stages, in order):** video decode → frame extraction/quality scoring/selection → SIFT feature extraction → brute-force matching → geometric verification → track construction → triangulation (DLT) → camera registration → joint pycolmap bundle adjustment → Depth Anything V2 inference → per-view depth alignment → multi-view dense fusion → dense filtering → normal estimation → BPA meshing (Phase 2 adaptive) → GLB LOD export → georeferencing/reports.

---

## 2. Languages & Runtimes

| Technology | Version | Where |
|---|---|---|
| **Python** | ≥3.11 declared (`pyproject.toml`); local venv reports 3.9.6 | backend |
| **TypeScript** | ~5.6.3 (strict, `tsc -b`) | frontend |
| **JavaScript/ESM** | `"type": "module"` | frontend |
| **Node.js** | v24.11.1 | frontend tooling |
| **Rust** | Edition 2021 | frontend/src-tauri (desktop shell only) |

---

## 3. Backend — Python Dependencies

### Web / API layer
| Package | Version | Purpose |
|---|---|---|
| FastAPI | ≥0.115 | HTTP API, dependency injection |
| Starlette | ≥0.41 | ASGI foundation (under FastAPI) |
| Uvicorn[standard] | ≥0.32 | ASGI server (production runs use it directly) |
| python-multipart | ≥0.0.18 | multipart/form-data video uploads |
| Pydantic | ≥2.10 | schemas, validation, settings |
| pydantic-settings | ≥2.7 | config via env |
| python-dotenv | ≥1.1 | `.env` loading |
| aiofiles | ≥24.1 | async file writes (uploads) |
| httpx | ≥0.28 | async HTTP client (tests + service calls) |
| structlog | ≥24.4 | structured logging |

### Database
| Package | Version | Purpose |
|---|---|---|
| SQLAlchemy | ≥2.0.36 | ORM, async engine |
| aiosqlite | ≥0.21 | async SQLite driver — the actual store (`backend/data/*.db`) |

*(SQLite ships with Python — no separate DB server.)*

### Computer vision / image I/O
| Package | Version | Purpose |
|---|---|---|
| OpenCV (headless) | ≥4.10 (installed 4.x) | video decode, SIFT, BFMatcher, geometric verification, contact sheets |
| Pillow | ≥11.0 | image decoding, EXIF, thumbnails |
| imagehash | ≥4.3 | perceptual hashing (duplicate/near-duplicate frame detection) |
| piexif | ≥1.1.3 | EXIF handling for image-based ingestion |

### Numerical / scientific
| Package | Version | Purpose |
|---|---|---|
| NumPy | ≥1.26 | all array math (unprojection, DLT, fusion, lexsort/searchsorted provenance) |
| SciPy | ≥1.14 | KD-trees (cKDTree), spatial queries, optimization helpers |
| Matplotlib | 3.9.4 (installed) | plots inside diagnostics/reports |

### AI / ML
| Package | Version | Purpose |
|---|---|---|
| PyTorch | ≥2.5 | **Depth Anything V2 (ViT-S)** monocular depth inference, CPU (MPS detected-but-unavailable in current env) |
| torchvision | ≥0.20 | transforms, ViT backbone utilities |

**Model weights:** `backend/models/weights/depth_anything_v2_vits.pth`
**Model implementation vendored at:** `backend/app/services/depth_anything_v2/` (`dinov2.py`, `dpt.py`, `blocks.py`, `transform.py` — DINOv2 backbone + DPT head, run locally, no external API).

### 3D / geometry / photogrammetry
| Package | Version | Purpose |
|---|---|---|
| **pycolmap** | 3.12.5 (installed) | **real joint bundle adjustment**; COLMAP text model I/O; incremental mapping helpers |
| Open3D | 0.19.0 (installed) | point clouds, voxel downsample, normals, **BPA ball-pivoting meshing**, Poisson (diagnostic), KD-tree search |
| trimesh | ≥4.5 | mesh inspection/processing |
| pygltflib | ≥1.16 | GLB/glTF export (viewer LODs) |

### Geospatial
| Package | Version | Purpose |
|---|---|---|
| pyproj | ≥3.7 | coordinate transforms (WGS84 ↔ local ENU), georeferencing |
| laspy | ≥2.5 | LiDAR `.las/.laz` reading (**validation/reference use only** — never fed into reconstruction) |

### Dev tooling (backend)
pytest ≥8.3, pytest-asyncio ≥0.24, pytest-cov ≥6.0, ruff ≥0.8 (lint, py311 target, line 100), mypy ≥1.13 (strict + pydantic plugin), hatchling (build backend).

---

## 4. Frontend Dependencies

### Runtime
| Package | Version | Purpose |
|---|---|---|
| **React** | ^18.3.1 | UI framework (hooks, no state lib — context in `useRuns`) |
| **react-router-dom** | ^6.28.0 | routing (`/`, missions, processing, viewer, reports) |
| **three** | ^0.170.0 | **3D rendering** — the mesh/dense-cloud viewer |
| three examples (bundled) | — | `GLTFLoader`, `PLYLoader`, `OrbitControls` |

### Build & test
| Package | Version | Purpose |
|---|---|---|
| Vite | ^6.0.0 | dev server + bundler (port 5173, `/api`+`/health` proxied to `127.0.0.1:8000`, 600 s proxy timeout for long pipeline calls) |
| @vitejs/plugin-react | ^4.3.4 | React fast refresh |
| TypeScript | ~5.6.3 | typechecking (`tsc -b` in build) |
| Vitest | ^5.0.0 | unit tests (64 frontend tests) |
| @testing-library/react + jest-dom + user-event | ^16/^7/^14 | component testing |
| jsdom | ^29.1.1 | DOM test environment |

**Styling:** plain CSS (`App.css` + per-page styles) — **no Tailwind/CSS-in-JS framework**.

---

## 5. Desktop Shell (optional packaging)

| Technology | Version | Purpose |
|---|---|---|
| **Tauri** | 1.5 | wraps the web app as a native desktop app (`strata-desktop`); plugins: shell-open, shell-sidecar, dialog, fs, path |
| serde / serde_json | 1.0 | Rust-side config serialization |
| tauri-build | 1.5 | build hook |

---

## 6. Algorithms & Techniques (the "invisible" stack)

| Area | Technique | Implementation |
|---|---|---|
| Sparse mapping | SIFT features, k=2 ratio-test BF matching, RANSAC/essential-matrix verification, persistent multi-view tracks, DLT triangulation, **joint bundle adjustment (Levenberg-style via pycolmap)** | `feature_extractor.py`, `feature_matcher.py`, `geometric_verifier.py`, `camera_pose_estimator.py`, `bundle_adjustment.py` |
| Depth | Depth Anything V2 (DINOv2 ViT-S + DPT head), global affine alignment between inferred depth and sparse geometry, per-view alignment with persistent audit reports | `depth_anything_v2/`, `depth_generator.py`, `depth_fusion.py` |
| Dense | Multi-view unprojection + voxel fusion, conservative filtering, cross-view disagreement stats, **vectorized per-point provenance (lexsort/searchsorted)** | `dense_reconstruction.py`, `depth_fusion.py`, `dense_diagnostics.py` |
| Mesh | BPA ball-pivoting with measured-spacing-driven radii (Phase 2 adaptive), sheet-gap/double-sheet protection, dense-to-mesh support metrics, matched-scale layer classification | `mesh.py`, `mesh_phase2.py`, `mesh_generator.py` |
| Georef | GPS/telemetry-driven ENU alignment, pyproj WGS84↔ENU, universal UAV telemetry schema detection (aliases, units, timestamp families, DMS, decimal commas) | `georeferencing.py`, `telemetry_schema.py`, `dji_srt_telemetry.py` |
| Auth | PBKDF2-SHA256 password hashing, session cookies | `auth_service.py`, `routes/auth.py` |
| Integrity | SHA-256 input-identity gates (upload→pipeline provenance), input freeze verification before mesh stages | `upload_service.py`, `pipeline_orchestrator.py`, `mesh_phase2.py` |
| Perf | KD-tree spatial indexing everywhere, vectorized PLY serialization (58× vs naive), detached process spawner for long jobs | `pointcloud.py`, `mesh.py`, `scripts/_spawn_detached.py` |

---

## 7. Data Formats Handled

**Inputs:** MP4/H.264 video, CSV telemetry (many UAV dialects), JPG/EXIF image sets, SRT telemetry (DJI), `.las/.laz` LiDAR (validation only), COLMAP text models.
**Outputs:** PLY (sparse/dense/mesh, binary), GLB/glTF (viewer LODs), JSON reports (`telemetry_schema.json`, `telemetry_quality.json`, `mesh_quality_report.json`, `layer_source_classification.json`, `runtime_profile.json`, …), PNG contact sheets/diagnostics.

---

## 8. Run & Test Commands

```bash
# Backend (from drone-recon/backend)
.venv/bin/uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
.venv/bin/pytest tests -x -q                     # fast suites
.venv/bin/pytest tests/test_pipeline_upgrade.py  # full-pipeline integration (~3 min)

# Frontend (from drone-recon/frontend)
npm run dev        # Vite on :5173, proxies /api → :8000
npm run test       # vitest
npm run build      # tsc -b && vite build

# Desktop (optional)
npm run tauri      # Tauri dev shell
```

**External service requirements: none** — everything (ML model, DB, file storage) runs locally and offline.
