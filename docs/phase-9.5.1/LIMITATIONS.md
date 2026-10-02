# Limitations (Phase 9.5.1)

## NOT validated (do not claim otherwise)

1. **Original drone MP4 ingestion** — no drone video exists in the
   repository. VisDrone-VID is an image-frame sequence and was not
   repackaged as fake "original video".
2. **Metric 3D accuracy / scale** — no surveyed ground truth, RTK/PPK,
   known distances, or calibrated camera geometry. Learned depth is
   explicitly relative (`metric=false`). 2D GPS georeferencing works;
   3D metric georeferencing does not.
3. **Ground-truth reconstruction accuracy** — no reference model exists;
   no RMSE/alignment numbers are reported.
4. **GPU acceleration** — this Intel Mac has neither CUDA nor MPS;
   everything ran CPU-only. MPS availability is checked independently of
   CUDA and reported honestly.
5. **Performance targets** — the <15-minute goal was explicitly NOT
   benchmarked or claimed. This phase was functional validation only.

## Known quality artifacts

- **COLMAP intrinsics drift**: without an intrinsic prior, COLMAP converges
  to a wide PINHOLE (fx≈1330 px on 4000 px images) even though reprojection
  error is ~1.1 px. Internally consistent, but not metric and not the DJI
  lens model. (Phase 9.5's OpenCV fallback used focal=3200 px.)
- **Stereo depth gaps**: 2 of 9 views produced no valid disparity (frames 4
  and 7 — weak texture/baseline at the estimated scale) and were skipped
  gracefully.
- **Dense grade "Poor"**: expected for a 10-frame single strip; not a
  regression, and not a claim about larger missions.
- **pycolmap macOS x86_64 wheel cannot decode JPEG** — the COLMAP stage
  re-encodes frames losslessly to PNG. This is a documented wheel defect.

## Environment / toolchain gaps

- CLI `colmap` and OpenMVS binaries: not installed (Homebrew build timed out).
- `pyproject.toml` declares `torch>=2.5.0` / `open3d>=0.18.0`, but the
  working Python 3.9 venv runs torch 2.2.2 with no open3d — pre-existing
  mismatch, documented not fixed.
- The depth model checkpoint (98 MB) lives under `backend/models/weights/`,
  which is not gitignored — flag for .gitignore in a later pass if the repo
  should not carry model weights.

## Fixed this phase (regression-free)

- pycolmap 3.12 API differences (return types, `points3D`, `cam_from_world()`,
  `Rotation3d`, `CameraMap`, `Track.length()` method) — see COLMAP_VALIDATION.md
- `TypeError: int() ... not 'method'` in `_write_sparse_ply` — Track.length
- `depth_generator` `auto` now metric-safe (stereo only)
- Tests: 184/184 PASS; ruff `F,I` clean on changed files