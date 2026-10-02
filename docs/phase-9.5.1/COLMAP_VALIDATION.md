# COLMAP Validation (Phase 9.5.1)

## What was tested

**Smoke test** (via `app/services/camera_pose_estimator.estimate_poses`):

- Input: 10 real Shitan TW ms1 images (DJI FC330, 4000×3000), copied as `frame_*.jpg`
- Feature extraction: COLMAP SIFT (pycolmap), ~10–15k features/image
- Matching: sequential matcher
- Incremental mapping: `pycolmap.incremental_mapping` (registers images,
  triangulates, runs global bundle adjustment)
- Result: **10/10 images registered, 2799 sparse points,
  mean reprojection error 1.09 px, ~179 s wall time**

**Functional run** (`run_phase95_validation`, run `shitan_ms1_20260909_131251`):

- Same inputs, through the full pipeline stage
- Result: **10/10 registered, 2796 sparse points, mean reproj 1.14 px**,
  `sparse_model.ply` + `poses.json` persisted, mission score 99.0/Excellent

## Integration issues found and fixed (pycolmap 0.6 → 3.12)

| Issue | Symptom | Fix |
|---|---|---|
| `incremental_mapping` return type | 0.6 returned dict; 3.12 returns `MappingResult` | read `.reconstructions` |
| `points3d` vs `points3D` | empty sparse parse | `best.points3D` |
| `Image.cam_from_world` is a method | `TypeError` / method object in dict | `img.cam_from_world()` |
| `Rotation3d` not a quaternion | wrong rotations | `cfw.rotation.matrix()` |
| `CameraMap` has no `.get()` | parse crash | `best.cameras[img.camera_id]` |
| `Track.length` is a method | `int() ... not 'method'` in `_write_sparse_ply` | `_track_length()` helper (callable-safe) |
| JPEG decode broken in wheel | `BITMAP_ERROR` for any `.jpg` on macOS x86_64 wheel | lossless PNG re-encode for COLMAP stage only |
| `match_images` removed | no such function in 3.12 | `match_sequential` / `match_exhaustive` |

## Honest caveats

- The pycolmap macOS x86_64 wheel **cannot decode JPEGs**; frames are
  re-encoded losslessly to PNG for the COLMAP stage. Other stages keep
  consuming the original JPEGs.
- Without an intrinsic prior, COLMAP converged to a wide-angle PINHOLE
  (fx≈1330 px for a 4000 px image). This is a **quality** artifact of
  unconstrained SfM, not a crash — the reconstruction is internally
  consistent (mean reproj ~1.1 px) but must not be interpreted as metric.
- `colmap` CLI and OpenMVS are not installed; only the pycolmap wheel path
  is validated.