# STRATA — PHASE 1B REPORT
**Dense Reconstruction + Multi-Layer Surface Integrity**
Date: 2026-09-20 · All numbers MEASURED from run artifacts unless labelled otherwise.

---

## 1. Files changed
- `app/services/dense_reconstruction.py` — dense stage: `dense_model_raw.ply` export (Part 26), cross-view pair stats + view-separation buckets (Part 4), mesh-input contract (Part 13), mesh-side layer detection, dense→mesh support, component classification, occlusion consistency (Parts 5/6/17/18/8), `mesh_full.ply` authoritative artifact (Part 20), `dense_quality_report.json` (Part 26), stage-dict overwrite fix (measured-offset/fusion-voxel keys were being clobbered by the 0.35 `stage()` update — reported as null in earlier runs despite the measurement having run).
- `app/services/dense_diagnostics.py` — NEW module: `detect_layers` (multi-sheet detector), `classify_layer_source` (CASE A–D), `mesh_support`, `classify_mesh_components` (Part 17), `extended_mesh_quality` (non-manifold/degenerate/normal-conflict/extents), `cross_view_consistency` (occlusion measurement, Part 8), `dense_quality_report` assembly with honest labels.

## 2. Files added
- `app/services/dense_diagnostics.py`
- `tests/test_dense_diagnostics.py` (10 regression tests, all passing)
- `scripts/phase1b_extra_diagnostics.py` (standalone batch-2 diagnostics for runs predating the wiring; read-only sidecars)
- Artifacts per run: `dense/dense_model_raw.ply`, `mesh/mesh_full.ply`, `dense_quality_report.json`, `layer_source_classification.json`, `phase1b_extra_diagnostics.json` (a8/a3 only)

## 3. Dense fusion architecture (unchanged math, now instrumented)
Depth maps (sparse-aligned metric Z) → per-view unprojection with canonical `X_world = R·X_cam + t` → voxel grid at the run's measured cross-view scale → confidence-weighted merge; **sibling-split**: measurements of incompatible surfaces sharing a voxel never average — they become separate output points (this is the occlusion/cross-surface fusion prevention). Provenance sidecar (`dense_fusion_provenance.json`): dominant source frame + pixel + fusion weight per point. New measurements layer: occlusion consistency (per-point cross-view corroboration vs occlusion-isolation at the fusion-voxel threshold).

## 4. Depth alignment results (per-view affine 1/Z = aᵢ·D + bᵢ, persisted per frame)
| scene | views | refined | median err before→after | flags |
|---|---|---|---|---|
| airport8 | 144 | 144 | 23.89 → 3.22 m | none |
| airport3 | 150 | 149 | 4.31 → 1.47 m | 1 unstable_affine_fit |
| berliner_dom1 | 173 | 173 | 20.83 → 4.63 m | none |
| berliner_dom3 | 143 | 143 | 10.85 → 7.11 m | none |
| bundestag3 | 132 | 129 | 12.21 → 6.19 m | none |
Label: ESTIMATED alignments to sparse geometry (Depth Anything V2 is non-metric).

## 5. Dense-layer statistics (detector: cells of 8×voxel footprint, gap > 1.5×voxel on Z)
| scene | dense layered % | dense regions | median sep | mesh layered % | mesh regions |
|---|---|---|---|---|---|
| airport8 | 15.26 | 602 | 3.05 m | 31.87¹ | 713 |
| airport3 | 2.54 | 6 | 4.41 m | 9.64¹ | 9 |
| berliner_dom1 | 1.15 | — | — | 4.16 | 9 |
| berliner_dom3 | 6.45 | 292 | — | **52.46** | 1,187 |
¹ mesh-side values for a8/a3 measured post-hoc by `scripts/phase1b_extra_diagnostics.py`.

## 6. Layer source classification (evidence-based, not visual)
- airport8: **CASE_C_both_contribute** (dense 15.26 %, mesh ≥ dense)
- airport3: **CASE_C_both_contribute** (dense 2.54 %, mesh 9.64 %)
- berliner_dom1: **CASE_B_mesh_origin** (dense 1.15 %, mesh 4.16 %)
- berliner_dom3: **CASE_B_mesh_origin** (dense 6.45 %, mesh **52.46 %** — dominant mesh-created layering; the BPA surface-merge mechanism, known from the 1ae5b0 curtain era, still produces double sheets on this scene)
- baseline (pre-Phase-1A airport8, measured post-hoc): dense 15.23 % / 530 regions; the baseline "mesh" is a 1:1 vertex copy of the dense cloud, so its mesh stats are the dense stats restated.
Conclusion: Phase 1B did **not materially change** dense layering (15.23→15.26 %) — it did not change fusion math; it now measures and localises it, and the classification localises the mesh-created share (dominant in both berliner scenes).

## 7. Mesh statistics
| scene | vertices | faces | components | edge med/p95 | giant faces | up-facing |
|---|---|---|---|---|---|---|
| airport8 | 37,645 | 72,803 | 107 | 5.91/8.53 m | 74 | 94.5 % |
| airport3 | 20,639 | 35,709 | 56 | 2.75/3.88 m | 0 | 93.6 % |
| berliner_dom1 | 27,185 | 42,946 | 178 | — | — | — |
| berliner_dom3 | 35,308 | 65,683 | 95 | — | — | — |
Extended (a8): 0 degenerate, 2,397 non-manifold edges, 6,675 boundary edges, 5.59 % normal-conflict pairs, extents 1020×808×69 m. Baseline mesh (a8) was 1,095,657 vertices ≈ the dense cloud itself — the mesh stage now outputs an actual BPA surface.

## 8. Dense→mesh support (threshold = fusion voxel) + occlusion consistency
| scene | supported % | unsupported regions | corroborated % | occlusion-isolated % |
|---|---|---|---|---|
| airport8 | 95.95 | 1,400 (small) | 40.54 | 22.54 |
| airport3 | 100.0 | 0 | 93.40 | 2.55 |
| berliner_dom1 | 100.0 | 0 | 65.45 | 33.10 |
| berliner_dom3 | 98.35 | — | — | 11.55 |
Component classification never auto-deletes: a8 107 comps = 1 supported_large + 101 supported_small + 5 noise_candidates; dom1 178 = 1 + 177; dom3 95 = 2 + 93.

## 9. airport8 (data_run_20260919_194154_5578f4)
144/144 cams · 100,580 pts · BA 0.2211→0.2178 px (pycolmap_joint, 598,596 obs) · GPS priors 144/144 (rms 0.046 m) · depth align 23.89→3.22 m · fused 1,638,353 → 1,081,258 (34 % filtered) · s2d NN 0.843/2.676/3.333 m · mesh 37,645v/72,803f · support 95.95 %. No regression vs baseline sparse (same frame selection; baseline had no real BA to compare).

## 10. airport3 (data_run_20260920_024829_9b876f)
150/150 cams · 54,076 pts · BA 0.2332→0.2297 px · GPS priors 150/150 (rms 0.173 m) · depth align 4.31→1.47 m · fused 97,277 → 25,180 · s2d NN 1.19/8.47/3.92 m (screened 53,989, med 1.18) · mesh 20,639v/35,709f, 0 giant faces · support 100 % · occlusion-isolated 2.55 %. First airport3 baseline: none existed pre-Phase-1 (report as UNAVAILABLE).

## 11. Berliner/Bundestag results
- berliner_dom1 (ba9bde): 173 cams, 57,849 pts, BA 0.2439→0.2402, GPS rms 0.239 m, fused 227,670, s2d med 2.287 m, CV 3.808 m (5–20 m: 3.79; >20 m: 4.886 — grows with separation), CASE_B, support 100 %, occl-iso 33.1 %.
- berliner_dom3 (abf638): 143 cams, 89,579 pts, BA 0.2357→0.2323, GPS rms 0.029 m, fused 2,043,554, voxel 1.52, s2d 0.914 m, CV 1.52 m (lt5 m: 1.315; 5–20 m: 2.062), CASE_B with 52.46 % mesh layering (worst scene), support 98.35 %.
- bundestag3 (76e40f): 132 cams, 103,229 pts, BA 0.2548→0.2499, GPS rms 0.047 m, fused 133,767 (voxel 4.5 m — scene-adapted), s2d 2.859 m, CV 4.505 m (5–20 m: 5.111; >20 m: 4.19), CASE_C (dense 5.85 % / mesh 7.01 %), support 100 %, corroboration 87.65 %, occl-iso 5.73 %, depth align 12.21→6.19 m, mesh 18,592v/27,442f, 47 comps (1 large + 46 small).

## 12. Performance (CPU-only; no GPU used — honest label)
airport8: frames 36 s · sparse 952 s · depth 405 s · dense 2,937 s (incl. mesh+LOD) · total 4,341 s. Mesh stage dominates; the PLY writers use a per-vertex Python loop (known bottleneck, unchanged this phase by design).

## 13. Known failure cases
1. **berliner_dom3 mesh layering (52.46 %, CASE_B)** — BPA+patch-merge creates double sheets on this scene; mechanism unchanged (deliberately — Phase 2 scope).
2. **airport8 cross-view corroboration only 40.5 %** at the 1.5 m threshold — consistent with the scene's ≈3 m median layer separation; much of the cloud sits in one of two sheets.
3. Depth-alignment residual after per-view fit remains metres-scale (3.2–7.1 m median) — mono-depth ceiling, not a fusion defect.

## 14. Exact remaining limitations
- Dense geometry is limited by Depth Anything V2 + per-view affine alignment: cross-view disagreement 1.5–3.8 m depending on scene/view separation. NO sub-metre dense accuracy is claimed (ESTIMATED scale, UNVALIDATED accuracy).
- Mesh layering on berliner-class scenes is mesh-origin (CASE_B) and unfixed (Phase 2).
- No independent metric validation vs LiDAR was run in this phase (reference validation deferred to Phase 1C gate).
- BA per-camera translation/rotation deltas are not persisted (UNAVAILABLE) — pre-BA poses are not stored.
- airport8's dense stage predates the mesh-input contract wiring, so `mesh_input` for that scene is only in berliner runs (e.g. dom1: 74,295 pts in, spacing median 2.97 m / p95 3.80 m, normal quality 0.993).

## 15. Recommended Phase 1C work
1. Regression gate exactly as specified: before/after vs run 134236 (airport8 baseline, fake BA, no GPS priors, no track gates).
2. Independent reference validation vs `lidar.npz` (airport3/8): point-to-reference distance, scale discrepancy — GT read-only, after production.
3. BPA mesh-layer mechanism fix for CASE_B scenes (berliner_dom3): per-surface merge in the mesh stage (Phase 2 carry-over, measurable via `layer_source_classification.json`).
4. Persist per-camera BA deltas (median/p95/max translation + rotation change) in the BA report.
5. Cross-view disagreement as a function of separation is now measured — use it to bound fusion voxel per scene instead of the single median.
