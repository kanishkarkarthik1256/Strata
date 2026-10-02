# Phase 9.5.1 — Real Reconstruction Stack Activation

Status: **PASS** (functional validation, not a performance benchmark)

This phase activated the real reconstruction software stack on top of the
existing Phase 1–9 architecture:

- **COLMAP SfM** via the `pycolmap` 3.12.5 wheel (the Homebrew `colmap`
  CLI build never completed; the wheel is the real COLMAP 3.12 engine)
- **PyTorch 2.2.2** (CPU only — Intel Mac, no CUDA, no MPS)
- **Depth Anything V2 (ViT-Small)** learned depth with the official
  checkpoint, vendored faithful architecture
- **FFmpeg 7.1** via `imageio-ffmpeg` 0.6.0 (encode + decode smoke-validated)
- Existing fallbacks preserved: SIFT/OpenCV pose estimation, Stereo SGBM depth

## What was actually executed and verified

| Capability | Status | Evidence |
|---|---|---|
| COLMAP feature extraction | VALIDATED | 10×4000×3000 Shitan ms1 images, ~10–15k SIFT features each |
| COLMAP matching | VALIDATED | sequential matcher, 45 pairs |
| COLMAP incremental mapping + BA | VALIDATED | 10/10 images registered, 2796–2799 points, mean reproj ~1.09–1.14 px |
| Learned depth (Depth Anything V2) | VALIDATED | official checkpoint loaded (state-dict exact match), 10/10 views, full-res depth maps, device=cpu |
| Metric depth | NOT VALIDATED | no scale reference / ground truth; outputs tagged `metric: false` |
| FFmpeg encode+decode | VALIDATED | testsrc encode → ffmpeg decode → cv2 ingestion path, 10 frames @ 10 fps |
| Original drone MP4 ingestion | NOT VALIDATED | no original drone video exists in `data/` |
| DJI EXIF GPS extraction | VALIDATED | 10/10 frames, GPS quality 91.89/100 (Excellent) |
| Metric 3D accuracy | NOT VALIDATED | no ground truth / surveyed control points |
| Fallback reconstruction | PASS | SIFT/OpenCV + Stereo SGBM remain operational (regression suite + prior Phase 9.5 run) |

## Functional run (the real pipeline)

- Run ID: `shitan_ms1_20260909_131251`
- Input: 10 real DJI FC330 images from `data/shitan_tw` (mission ms1), 4000×3000
- Stages: environment → dataset → images → video → sparse (COLMAP) →
  GPS attach → depth (stereo) → depth_learned (Depth Anything) →
  dense → geospatial → **all PASS**
- Outputs: `backend/outputs/shitan_ms1_20260909_131251/`
- Docs: `backend/docs/shitan_ms1_20260909_131251/`

## Regression

- Tests: **184/184 PASS**
- Lint: `ruff check --select F,I` clean on all changed files
- No tests disabled or weakened

## Bugs found and fixed this phase

1. **pycolmap 3.12 API breaks** in `camera_pose_estimator._run_colmap`
   (written against pycolmap 0.6): `incremental_mapping` return type,
   `points3D` vs `points3d`, `cam_from_world()` method vs property,
   `Rotation3d` vs Quaternion, `CameraMap` vs dict, `match_images` → removed.
   All fixed; verified by running real SfM.
2. **JPEG decode failure** in the pycolmap macOS x86_64 wheel: any `.jpg`
   raises BITMAP_ERROR while PNG decodes correctly. Fixed by losslessly
   re-encoding frames to PNG for the COLMAP stage only (pixels unchanged).
3. **`Track.length` is a method in pycolmap 3.12**: `np.array([...],
   dtype=np.int32)` on the bound method raised `TypeError: int() argument
   ... not 'method'` in `_write_sparse_ply`, failing the full sparse
   pipeline. Fixed with a `_track_length()` helper (callable-safe).
4. **`depth_generator` backend selection**: `auto` is now metric-safe
   (stereo only); learned depth is explicit-only so relative depth can
   never silently corrupt metric fusion.

## Files changed

- `backend/app/services/camera_pose_estimator.py` — pycolmap 3.12 integration
- `backend/app/services/depth_generator.py` — `depth_anything` backend
- `backend/app/services/phase95_validator.py` — video / learned-depth /
  GPS-attach stages, richer environment audit
- `backend/app/utils/run_organization.py` — `video` + `depth_learned` stage dirs
- `backend/app/services/depth_anything_v2/` — vendored official DPT/DINOv2
  architecture + checkpoint loader
- `backend/tests/test_pipeline_upgrade.py` — environment-conditional test
- New venv packages: `pycolmap==3.12.5`, `torch==2.2.2` (numpy pinned to
  1.26.4), `imageio-ffmpeg==0.6.0`, `huggingface_hub`

## Status block

```text
PHASE 9.5.1 STATUS: PASS

SYSTEM:
OS: macOS 26.3.1 (x86_64)
Architecture: x86_64
CPU: Intel Core i7-1068NG7 @ 2.30GHz
RAM: 32 GB
GPU: Intel Iris Plus (1.5 GB) — no CUDA, no MPS

COLMAP:
Installed: YES (pycolmap 3.12.5 wheel; no CLI binary — brew build did not complete)
Detected: YES
Smoke-tested: YES (10/10 images, 2799 points, 1.09 px, ~179 s)
Used: YES (functional run: 10/10, 2796 points, 1.14 px)
Validated: YES

PyCOLMAP:
Installed: YES (3.12.5)
Smoke-tested: YES
Validated: YES

PyTorch:
Installed: YES (2.2.2)
CPU: VALIDATED
MPS: NOT AVAILABLE (Intel i7 — no Apple Silicon GPU)
CUDA: NOT AVAILABLE (no NVIDIA GPU)
Smoke-tested: YES (real tensor ops + depth inference on CPU)

Learned Depth:
Model: Depth Anything V2 (ViT-Small, DPT)
Checkpoint: models/weights/depth_anything_v2_vits.pth (official, 98 MB)
Loaded: YES (state-dict exact match)
Inference: YES (10/10 views, 4000×3000)
Device: cpu
Validated: YES (real inference; RELATIVE depth, metric=false)

FFmpeg:
Installed: YES (imageio-ffmpeg 0.6.0 → FFmpeg 7.1 static binary)
Detected: YES
Decode smoke test: PASS (encode → decode → cv2 ingestion, 10 frames @ 10 fps)

Original Drone MP4:
Available: NO (no .mp4/.mov in data/)
Tested: NO

GPS:
Available: YES (DJI FC330 EXIF)
Source: EXIF (lat 24.5309, lon 120.9307, alt 359.5)
Validated: YES (10/10 frames, quality 91.89/100 Excellent)

Metric 3D: NOT VALIDATED

Ground Truth: NOT AVAILABLE

Real Reconstruction: PASS
Fallback Reconstruction: PASS

Regression:
Tests: 184/184 PASS
Lint: ruff F,I clean on changed files

Output: outputs/shitan_ms1_20260909_131251/
Documentation: docs/shitan_ms1_20260909_131251/

Remaining blockers:
- Original drone MP4 ingestion NOT VALIDATED (no drone video in repository)
- Metric 3D accuracy NOT VALIDATED (no ground truth / scale reference)
- CPU-only inference (no MPS/CUDA on this machine)
- CLI colmap + OpenMVS binaries not installed
```