# STRATA — Fragmented 3D Reconstruction: Root-Cause Diagnosis & Fix

**Date:** 2026-09-13
**Input:** `data/base.mp4` — SHA256 `8f746307bef443b3b4e5a3b2a96ace746d1c4b058be15bf7b9edf65385afb6da`
**Method:** every claim below was proven numerically against live runs; no cosmetic cleanup was applied to any cloud or mesh.

---

## 1. Runs used (all fresh, via the real API)

| Run | Purpose | Depth backend executed |
|---|---|---|
| `data_run_20260913_153644_3e9fb2` | Reproduce the failure (pre-fix `auto`) | `stereo` (SGBM) — reproduces the fragmented output users saw |
| `data_run_20260913_155203_21e65d` | Fix 1 verification (`auto` → DA-V2) | `depth_anything` |
| `data_run_20260913_160634_6d0033` | Fix 2 verification (inverse-depth alignment) | `depth_anything` |

Recorded facts for the reproduction run: 10 frames selected, 10 cameras registered (COLMAP backend, mean sparse reprojection error 0.43 px, 1,422 sparse points), 9 depth maps, fused 578,905 → dense cloud, mesh Poisson-completed. Everything "succeeded" while the geometry was fragmented — confirming the brief's warning that stage completion is not geometric correctness.

## 2. Pipeline data flow (as actually implemented)

```
base.mp4 → frames (entropy selection) → sparse (pycolmap 3.12.5 SfM)
   → poses.json  {K, R, t} with X_world = R @ X_cam + t, t = camera centre (T_wc)
   → sparse_model.ply (SfM points, arbitrary global scale)
   → depth stage:
        auto → stereo SGBM (pre-fix) | Depth Anything V2 (post-fix)
        DA-V2 raw D_raw → global alignment fit from projected SfM points
   → dense: per-pixel unprojection X_c = Z·K⁻¹[u,v,1]ᵀ, world via T_wc,
        confidence-weighted voxel merge (this is the TSDF-equivalent fusion)
   → filters/normals → dense_model.ply → Poisson mesh
```

## 3. Verified conventions (proven, not assumed)

| Item | Verified value | Evidence |
|---|---|---|
| Pose convention | **T_wc** (camera→world); `X_w = R·X_c + t`, `t` = camera centre | Projection test on real sparse points: 97.9 % in-bounds under `X_c = Rᵀ(X_w−C)` vs 52.6 % under the w2c reading (`geometry_report.json: pose_direction`) |
| Fusion transform | Consistent with T_wc: `world = cam @ R.T + t` | Code + the same projection test |
| Depth parameterisation | **Z = camera-axis depth** (pinhole), used consistently in `_unproject_view` | Code inspection + single-view cloud coherence |
| Image/depth indexing | `depth[v, u]`, H×W = 1280×720 matches RGB | Arrays and `find_image_file` shapes checked |
| pycolmap `calibration_matrix()` | Round-trips params faithfully (verified on a synthetic PINHOLE camera) | Live check, pycolmap 3.12.5 |
| Intrinsics **quality** | COLMAP estimated **degenerate K** for this run: fx = 2943 vs fy = 909 (3.2× anisotropy is physically impossible for a real camera) | `poses.json` of all three runs |
| DA-V2 output semantics | **Inverse depth (disparity-like):** fitting `1/Z = α·D_raw + β` on 13,341 real SfM correspondences gives median error **1.04 m** vs **5.81 m** for affine-in-Z (r² 0.85 vs 0.64) | Empirical test on `data_run_20260913_155203_21e65d` (§E method of the brief) |

## 4. Raw DA-V2 output (pre-alignment), frame_000000

Stats from `diagnostics/depth/frame_000000_depth_raw.npy`: shape (1280, 720) float32, min 0.0, max 6.67, mean 1.52, median 0.50, p99 5.76 — finite everywhere, no sentinels. The vendor head (`dpt.py`) ends in ReLU → non-negative output; `infer_image` resizes back to source resolution with `bilinear/align_corners=True`. Semantics: relative **inverse depth** (see §3).

## 5. ROOT CAUSES (two, both proven)

### RC1 — `auto` selected SGBM stereo for monocular forward-motion video (PRIMARY)

`_choose_backend("auto")` returned `stereo`, treating consecutive video frames as a calibrated stereo pair. Forward motion yields near-zero *horizontal* disparity, so SGBM returns subpixel noise that the `Q`-matrix reprojection converts to enormous depths. Measured on the reproduction run:

- depth map frame_000000: median **9,987.6 m**, p95 9,987.8, max 9,986.9–9,987.8 across frames (urban drone footage!),
- valid ratios 0.9986 — the junk passed every sanity filter,
- fused single-view bbox 236 m; two-view median NN disagreement **2.19 m**; cross-view relative depth error median **99 %**.

This is the fragmentation: each frame paints a different, 10-km-deep shell; voxel merge cannot reconcile them.

### RC2 — alignment equation mismatched the model's output semantics

Even after forcing DA-V2, `_compute_global_affine_alignment` fitted `Z = a·D_raw + b` (affine in Z). The model's output is inverse-depth-like: the Z-affine fit compresses far surfaces into a narrow band (the fitted map saturates at its 99th-percentile bound ≈ 31 m) and degrades cross-view consistency (multi-view p95 error 12.8 %). The global fit **was** sequence-level (one (a,b) for all frames — architecture correct), but the equation family was wrong for the quantity being modeled.

## 6. Hypothesis table

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Wrong depth semantics | **CONFIRMED (RC2)** | §3 semantics test: 1/Z fit beats Z-fit 5.6× in median error |
| H2 | Inconsistent per-frame scale | RULED OUT | Alignment is global (single (a,b) across all views, `depth_global_affine_fitted` log); per-frame fits only used as fallback when global is unavailable |
| H3 | Pose inversion | RULED OUT | Projection test §3 — stored convention is T_wc and code uses it consistently |
| H4 | Wrong camera/world transform | RULED OUT | Same test; fusion math matches poses.json semantics |
| H5 | Portrait W/H swap | RULED OUT | depth (1280, 720) matches image (1280, 720, 3); cx=360, cy=640 correct |
| H6 | Wrong intrinsics after resize | RULED OUT (for the pipeline); **degenerate K flagged** | DA-V2 resizes internally and maps output back to source resolution; K degeneracy is a COLMAP estimate-quality issue on this clip (2-view NN still 0.05 m post-fix), reported, not tuned |
| H7 | depth/RGB indexing mismatch | RULED OUT | `depth[v, u]` everywhere; single-view cloud coherent |
| H8 | Invalid depth entering fusion | RULED OUT post-fix | Fusion audit: 0 non-finite, 0 negative, invalid = 0.0 excluded; pre-fix the junk was *finite but absurd* (median 9,987 m) — no per-map validator catches that; stage-level range bound from SfM now guards it |
| H9 | Excessive depth noise | PARTIAL | Pre-fix: catastrophic (99 % cross-view error). Post-fix: p95 6 % — acceptable for fusion |
| H10 | Insufficient temporal overlap | RULED OUT | 504,370 overlapping point pairs compared across frames 0↔2 |
| H11 | Dynamic objects | NOT EVALUATED | No detector in pipeline; outliers rejected only via 3σ on correspondences |
| H12 | Voxel/truncation settings | RULED OUT as cause | Fusion params (0.05 m voxel, depth 0.2–200 m) fine once inputs are sane |
| H13 | Depth range clipping | RULED OUT post-fix | z_min/z_max derived from SfM percentile bounds per view |
| H14 | Confidence/masking bug | RULED OUT | Confidence model (`σ ~ z²/f·px_noise`) monotone and bounded; observed confs sane |
| H15 | Frame-to-frame scale inconsistency | RULED OUT | Single global fit; two-view centroid gap 0.014 m (fix-1 run) |

## 7. Fixes applied (minimal, in `depth_generator.py` only)

1. **`_choose_backend("auto")`**: monocular video contract — auto now selects `depth_anything` when the checkpoint is available, falling back to `stereo` only when it is not. Stereo remains available explicitly. (The `depth_anything` path already tags every artifact `metric: false` — no honesty regression; stereo was *never* honest either, since the SfM scale is arbitrary, see §9.)
2. **Alignment equation**: replaced `_fit_affine_robust` (Z-space) with `_fit_inverse_depth_robust` fitting `1/Z = a·D_raw + b`, `Z = 1/(a·D_raw + b)`, slope guard `a > 0`. Applied in both the global fit and the per-view fallback. Fit report formula/`slope_sign` fields updated; `metric_scale` artifact annotation now states the inverse-depth relation and Metric Scale: ESTIMATED / Validation: NOT VALIDATED.
3. Skipped the wasted global-affine computation when the stereo backend is selected.

No smoothing, no hole filling, no decimation, no Poisson tricks, no confidence fabrication. PipelineStage architecture untouched; orchestrator untouched.

## 8. Before / after (same frames, same poses, same tests)

| Metric | Pre-fix (stereo) | Fix 1 (DA-V2, Z-affine) | Fix 2 (DA-V2, 1/Z) |
|---|---|---|---|
| Depth backend | stereo | depth_anything | depth_anything |
| Depth median (frame 0) | 9,987.6 m | 20.5 m | 20.5 m* |
| Depth range (frame 0) | 7.0–9,987.8 | 3.2–31.2 | 5.6–57.5 |
| Two-view median NN | 2.19 m | 0.05 m | 0.055 m |
| Two-view p95 NN | 6.65 m | 0.145 m | 0.324 m |
| Cross-view rel. depth err (median) | 99.0 % | 1.8 % | 2.1 % |
| Cross-view rel. depth err (p95) | 99.9 % | 12.8 % | **6.0 %** |
| Cross-view rel. depth err (p99) | 404.8 % | 33.5 % | **9.6 %** |
| Dense score / grade | 45.96 / Poor | 52.4 / Fair | 52.2 / Fair |

*Fix-1 numbers from `data_run_20260913_155203_21e65d`; fix-2 from `data_run_20260913_160634_6d0033`. Global fit quality: `1/Z = a·D_raw + b`, 13.3k correspondences, r² ≈ 0.85 (Z-space evaluation), median |err| ≈ 1.0 m on the fit set.

The p95/p99 improvements (2.1× / 3.5×) are the decisive evidence: the inverse-depth model removes the far-field compression that the Z-affine map imposed, exactly where cross-view fragmentation lived.

## 9. Honest status (unchanged by this fix)

- **Metric scale: ESTIMATED.** SfM scale is arbitrary (no GPS/RTK/GCP on this clip); DA-V2 alignment inherits it. No claim of metric accuracy is made anywhere.
- Depth artifacts remain tagged `metric: false` with the alignment formula recorded.
- Georeferencing: VIDEO_ONLY, honestly skipped, no fake GPS.
- Dense score 52/100 "Fair" reflects real remaining error (median cross-view rel. err ~2 %, tail p99 ~10 %).

## 10. Known remaining limitations

1. **Degenerate K from COLMAP on this clip** (fx/fy ratio 3.2): SfM estimated a physically implausible camera. Pose geometry is still self-consistent (97.9 % in-bounds; 0.05 m two-view agreement), but per-axis distortion of scale is possible. Fix belongs in SfM quality gating (shared-camera model constraints / more frames), explicitly out of scope here per the freeze.
2. `_monocular_fallback_depth` (perspective-gradient synthetic depth) still exists as dead code in `depth_generator.py` — never selected by any backend path; recommend deletion in a cleanup pass.
3. `depth_diagnostics.py` mislabels every run as "Depth-Anything-V2-Small" and its acceptance criteria are partly tautological; it did not cause the fragmentation but it *failed to catch it*. Its criteria need real thresholds (follow-up).
4. Only 10 frames / 9 depth maps from base.mp4 (extraction + registration limits); sparse correspondences (1.4k points) are thin for the affine fit — more frames would sharpen it.

## 11. Diagnostic artifacts

- `backend/data/storage/<run_id>/diagnostics/geometry/` — `single_view.ply`, `two_view.ply`, `geometry_report.json` (all three runs)
- `backend/data/storage/<run_id>/diagnostics/depth/` — raw DA-V2 npy + visuals + valid masks
- `backend/scripts/run_fragmentation_diagnostics.py` — reusable test suite (pose direction, single-view, two-view, multi-view, fusion audit)
- Backend log `depth_global_affine_fitted` lines record the exact fitted parameters per run

## 12. Verification

- Full backend suite: **266 passed, 1 skipped** (includes updated `test_auto_backend_prefers_monocular_depth_anything` regression test)
- Three fresh live runs through the real API on `data/base.mp4`
- All geometric tests re-run on every run; artifacts persisted per run
