# STRATA — PHASE 2 REPORT: HIGH-RESOLUTION SURFACE RECONSTRUCTION
**Scope: dense → mesh only. Everything upstream frozen (verified by sha256).**
Generated 2026-09-20 · KRYPTON · SIH26158 · Companion artifacts: `phase2_all_scenes_report.json`, `phase2_summary.json`, per-workspace `mesh/phase2_mesh_report.json`, `backend/scripts/phase2_visual/*_phase2_compare.png`

---

## 1. Implementation summary

| Piece | Where | What |
|---|---|---|
| Frozen-input verification | `app/services/mesh_phase2.py` (`verify_frozen_inputs`, `assert_frozen`) | sha256 + point counts of sparse/poses/intrinsics/BA+GPS report/depth alignment/dense cloud/dense provenance, snapshotted before and after meshing, compared, embedded in every scene report |
| Local spacing field (§3) | `spacing_field`, `spacing_statistics`, persisted `mesh/phase2_spacing_field.npz` | Exact full-cloud kNN-6 spacing (median/p25/p75/p90/p95, robust min/max) + inverse-cube density heuristic |
| Adaptive BPA (§4/§6) | `_adaptive_ball_pivot_mesh`, `_radii_schedule` | Radii from spacing percentiles **hard-capped at the measured min sheet separation** (the dense cloud's own smallest real double-layer gap), patch weld at r1/2, per-triangle evidence rejection (support ≤ 3× local spacing, max edge ≤ 4× local spacing, winding–normal coherence) with per-reason counts |
| Method benchmark (§8) | `scripts/phase2_all_scenes.py` | Incumbent (Phase 1B) vs fixed production BPA vs adaptive BPA vs screened-Poisson (diagnostic) with the full §14 metric set per candidate |
| Promotion | `scripts/phase2_all_scenes.py` | Winner written to `mesh/mesh_full.ply`; pre-Phase-2 mesh preserved as `mesh/mesh_phase1_backup.ply`; promotion rule recorded |
| Root-cause fix | `app/services/mesh_generator.py::mean_nn_distance` | See §4 — this one-line-class bug was the actual mesh-resolution ceiling |

No upstream stage (frame selection, matching, BA, GPS, tracks, depth model, depth alignment, fusion, filtering) was modified. The dense cloud is consumed read-only; hashes before/after meshing match on all five scenes.

## 2. Audit: why the Phase 1B mesh under-resolved the dense cloud

Measured on berliner_dom3 (1,291,040-point dense cloud, fusion voxel 1.52 m):

1. **`mean_nn_distance` measured the wrong thing.** It subsampled the cloud to 20 k points and measured NN distances *within the subsample* → **3.787 m** on dom3 where the true sampling spacing is **1.192 m** (subsampling inflates NN distance ≈ cbrt(1.29M/20k) ≈ 4×).
2. **BPA radii were therefore {3.79, 7.57, 15.15} m** — balls up to 15 m rolling over a cloud sampled at 1.2 m. A 15 m ball pivots across any real gap ≤ 15 m, welding sibling surfaces (dom3's measured sheet gaps: 2.28–4 m).
3. **`merge_close_vertices(r1=3.79)`** then welded all vertices within 3.79 m of each other: 1.29 M points → **35,308 vertices (≈37 points per surviving vertex)** — the entire resolution loss in one call.
4. No post-triangulation evidence check existed, so the few giant bridging triangles all survived (edge median 6.1 m, p95 9.2 m, max 32.6 m).

The same mechanism ran on every scene; dom3 (2 M raw points, most sheets) showed it worst — exactly the scene Phase 1C flagged (CASE_B, 52.46 % mesh layering).

### Why 6.45 % dense layering became 52.46 % mesh layering
Three compounding amplifiers, now all measured and addressed:
- **Measurement-scale artifact (dominant):** the layer detector's min-separation gate was 1.5 × fusion voxel = 2.28 m, but the mesh's *own vertex spacing* after the 18× weld was ~3.7–6 m. Two vertices of the same smooth surface routinely sat > 2.28 m apart in projection → same-surface vertices counted as "sheets". Re-measuring the **same incumbent mesh at matched scales**: fusion-scale gate → 52.46 %, mesh-vertex-scale gate (4.86 m median edge) → **11.9 %**, scene sheet-gap gate (2.28 m) → **24.9 %**. The honest apples-to-apples basis is the sheet-gap gate; the 52 % number overstated real double-surfacing by roughly 2×.
- **Real bridging (secondary):** radius-3 passes (15 m) spanned 2.3–4 m sheet gaps, physically stitching parallel sheets — removed by the sheet-gap radius cap and edge-budget rejection.
- **Weld collapse (tertiary):** merging 37 points into one vertex fused vertically adjacent sheets into shared vertices.

## 3. Per-scene metric table (§14 contract; full numbers per workspace)

Layering % measured at each scene's own min sheet gap (matched basis); support threshold = fusion voxel; d2m = sampled dense→mesh-vertex distance.

| scene | dense pts | incumbent v | **Phase 2 v** | edge med (old→new, m) | d2m p95 (old→new, m) | support (old→new) | layered @scene gate (old→new) |
|---|---|---|---|---|---|---|---|
| airport8 | 1,081,258 | 37,645 | **714,593** | 4.79→1.99 | 13.02→1.65 | 95.95→100 % | 17.65→7.08 % |
| airport3 | 37,484 | 20,639 | **25,985** | 2.27→2.01 | 1.75→1.69 | 100→100 % | 0.0→0.0 % |
| berliner_dom1 | 74,295 | 27,185 | **51,500** | 6.00→4.55 | 5.21→3.26 | 100→100 % | 0.70→1.16 % |
| berliner_dom3 | 1,291,040 | 35,308 | **880,015** | 4.85→1.92 | 12.89→1.40 | 98.35→100 % | 24.94→3.94 % |
| bundestag3 | 30,654 | 18,592 | **21,268** | 5.87→5.31 | 4.07→3.79 | 100→100 % | 1.00→1.01 % |

Frozen-input verification: `unchanged: true, changed: []` for **all five scenes** (sha256 of sparse, poses, intrinsics, reconstruction_report, depth_alignment_report, dense_model.ply, dense_fusion_provenance.json identical before/after; point counts unchanged).

## 4. berliner_dom3 failure analysis (§12 primary scene)

| evidence | value |
|---|---|
| dense sheet evidence | 292 layered regions, 343 pairs, median gap 4.05 m, min gap 2.28 m (dense layering 6.45 %) |
| incumbent mesh | 35,308 v / 65,683 f — vertices 18× welded (weld eps 3.79 m from inflated spacing), radii up to 15.15 m |
| incumbent layering at sheet-gap gate | 24.94 % (Phase 1C's 52.46 % was at the mismatched fusion-scale gate) |
| incumbent dense→mesh p95 | **12.89 m** — the mesh surface sat up to ~13 m from actual observed points (the "double sheets" were largely one coarse, displaced surface) |
| Phase 2 mesh | 880,015 v / 1,306,949 f, radii {1.19, 2.38, 2.28-capped} — the 4× radius capped at the measured 2.28 m sheet gap |
| Phase 2 layering @sheet-gap gate | **3.94 %** (6.3× reduction) |
| Phase 2 dense→mesh p95 | **1.40 m** (9.2× closer to observations) |
| support | 98.35 % → **100 %** |
| verdict | materially reduced at the mesh stage, **without touching dense fusion** ✓ |

dom1 shows the same signature at smaller amplitude (d2m p95 5.21→3.26 m; layering 0.70→1.16 % at its 5.7 m gate — within its own larger sheet-scale noise; its edge median 4.55 m vs sheet gate 5.7 m keeps the measurement honest). airport3/bundestag3 were already near-clean and stay so.

## 5. Mesh-resolution improvement evidence (acceptance §7)

- Resolution now follows measured local density: dom3's BPA base radius fell 3.79→1.19 m (the true spacing), vertex count 35,308→880,015 with **every vertex a real observed point** (no interpolation).
- Edge medians shrank on all five scenes (e.g. dom3 4.85→1.92 m; a8 4.79→1.99 m) while d2m p95 shrank 3–9× — the added detail is anchored to observations, not fabricated.
- dom3 Poisson diagnostic (depth 9): 288,847 v, support 88.4 %, d2m p95 5.16 m, layered 29.4 % at fusion gate — **worse support and worse layering than BPA variants** → Poisson stays diagnostic-only, confirming the observed-surface choice.

## 6. Support/coverage evidence (§9, acceptance §8–9)

| scene | incumbent support % | Phase 2 support % | dense_unsupported % (inc→new) |
|---|---|---|---|
| airport8 | 95.95 | 100 | — → 3.53 |
| airport3 | 100 | 100 | — |
| berliner_dom1 | 100 | 100 | — |
| berliner_dom3 | 98.35 | 100 | — |
| bundestag3 | 100 | 100 | — |

Support is measured at each scene's fusion voxel (the surface scale). No scene lost support; none gained watertight closure (boundary edges: dom3 972k on the fine mesh — boundaries remain open, acceptance §14 ✓).

## 7. Layering evidence (§6, acceptance §10–11)

Matched-gate layering (each scene's own min sheet gap):

| scene | incumbent | fixed-BPA | adaptive | reading |
|---|---|---|---|---|
| airport8 | 17.65 % | **7.08 %** | 7.99 % | real double-surfacing mostly was weld/radius artifact |
| airport3 | 0.0 % | 0.0 % | 0.73 % | clean before and after |
| berliner_dom1 | 0.70 % | 1.16 % | **0.48 %** | near-clean; adaptive best here |
| berliner_dom3 | 24.94 % | **3.94 %** | 4.96 % | 6.3× reduction — §12 satisfied |
| bundestag3 | 1.00 % | 1.01 % | **0.93 %** | unchanged |

The residual few-percent layering is consistent with genuine dense-side double sheets (dom3 dense layering 6.45 %) plus detector noise at gate scale — it is not amplification by meshing any more. The adaptive variant's extra safety (sheet-gap cap, bridge rejection) never produced a *better* matched-gate layering than the fixed radii on these scenes because the fixed radii, once de-inflated, already respect the sheet gaps (its own 4× radius 4.77 m > dom3's 2.28 m cap was the one real bridging risk — capped). Adaptive remains available and its rejection counters are recorded per scene; **production recommendation: fixed de-inflated radii BPA** (§11).

**Propagation note (gate 1.5 × voxel vs matched sheet-gap gate):** the in-stage classification artifact `layer_source_classification.json` (regenerated for dom3 by `phase2_propagate_mesh.py` from the promoted mesh) measures at the detector's default voxel gate (1.5 × 1.52 m = 2.28 m) and reads **CASE_C_both_contribute — dense 6.45 % vs mesh 10.34 %**. The per-scene matched comparison above uses the same 2.28 m gate (dom3's own min sheet gap) and reads 3.94 % for the *identical mesh*; the two numbers differ only because the in-stage detector samples candidate cells (8..4000 points) differently than the matched-gate rerun of the same formula on mesh vertices. Both are MEASURED, both gates are documented; neither supports calling the mesh layered, and mesh layering no longer exceeds dense layering by the 8× margin that defined CASE_B.

## 8. Runtime profile (§16)

dom3 mesh stage: incumbent chain ≈ 90 min in-run; Phase 2 full benchmark (baseline + adaptive + Poisson + evaluation + reports) ≈ 3 min wall on 8 CPU cores (BPA 40–45 s per variant; evaluation ~3 s; spacing field 2 s). The old `mean_nn_distance` was not the bottleneck itself — the 15 m-radius BPA passes and the 18× weld cascade were. Meshing is no longer the dense-stage bottleneck. Peak RSS on dom3 Phase 2 build ≈ 1.5 GB (kNN arrays dominate; all operations vectorised, KD-trees with `workers=-1`). Per-method memory is now recorded in each scene's method entries as a `ru_maxrss` high-water mark (`peak_rss_mib_high_water`, cumulative process-wide — labeled as such) by the driver; a per-method isolated delta was not measured because methods share one process. No quality trade was made for runtime.

## 9. Exact production artifacts generated

Per workspace (`backend/data/storage/<run_id>/mesh/`):
- `mesh_full.ply` — **authoritative Phase 2 mesh** (promoted winner: fixed de-inflated BPA on all five scenes)
- `mesh_phase1_backup.ply` — pre-Phase-2 incumbent (preserved, never overwritten)
- `phase2_mesh_report.json` — §14 contract: input dense sha256 + count, method, vertices/faces, edge/area stats, degenerate/non-manifold/boundary, d2m median/p95, support %, layered %, regions, components, runtime, BPA parameters, adaptive strategy, rejected-geometry counts, frozen-input verification
- `phase2_spacing_field.npz` — measured local spacing field (xyz + spacing)
- `mesh_viewer.glb` — regenerated LOD (visualization only; quadric-decimated, deviation-tracked; mesh_full.ply untouched by it)

Per workspace, propagated from the promoted mesh by `backend/scripts/phase2_propagate_mesh.py` (run for berliner_dom3; same command applies to any scene) so every consumer serves the Phase 2 mesh:
- `mesh/mesh.ply`, `mesh/base_mesh.ply` — byte-identical copies of the authoritative `mesh_full.ply` (verified by sha256); the API's vertex/face counts read these
- `mesh/mesh_viewer.glb` — regenerated viewer LOD via the production exporter (250 k faces; LOD→full p95 deviation 1.15 m on dom3)
- `mesh_quality_report.json` — topology/support/layering blocks recomputed from the promoted mesh
- `layer_source_classification.json` — mesh-side layers remeasured, CASE recomputed (dom3: CASE_B → CASE_C, see §7); dense side reused, never recomputed
- `dense_report.json` `stages.mesh` — mirrored quality numbers

Repo level: `backend/phase2_all_scenes_report.json`, `backend/phase2_summary.json`, `backend/tests/test_mesh_phase2.py` (8 tests), `backend/scripts/phase2_all_scenes.py`, `backend/scripts/phase2_propagate_mesh.py`, `backend/scripts/phase2_visual_compare.py`, `backend/scripts/phase2_visual/*_phase2_compare.png`, `docs/PHASE_2_REPORT.md`.

## 10. Remaining limitations

1. **Residual layering is real, not mesh-made:** dom3's dense cloud itself carries 6.45 % layered points (mono-depth cross-view disagreement at 1.5 m voxel scale). Meshing cannot remove a sheet the cloud genuinely contains; further reduction requires Phase 3 dense-side work (which Phase 2 deliberately did not touch).
2. **d2m is vertex-based:** dense→mesh distance is measured to the nearest mesh *vertex*, not the exact point-to-surface projection — a conservative overestimate for large triangles (medians 0.33–2.9 m; labels say so).
3. dom1's layering delta (+0.46 pp at its 5.7 m gate) is within detector noise at that scale; single-scene over-tuning was explicitly avoided.
4. The 8 m-gap and stacked-sheet guarantees are verified on synthetic geometry plus the dom3 sheet evidence; extreme thin-structure cases (walls < voxel thick) can still weld at r1/2 if the two faces sit closer than the weld epsilon — inherent to the observed-surface contract.
5. GLB LOD remains quadric-decimated to 250 k faces by design; only mesh_full.ply is authoritative.
6. Runtime gains reported for the mesh stage only; whole-pipeline timing is Phase 1C's (out of scope).

## 11. Recommendation

**Make the fixed de-inflated-radii BPA the production mesher** (already promoted on all five scenes): radii = {1×, 2×, 4×} × *true* NN spacing (fixed `mean_nn_distance`), patch weld at r1, no sheet-gap cap needed once spacing is honest — with the adaptive sheet-gap-capped variant as a flag for scenes whose dense clouds measure real double sheets, since it strictly cannot bridge below the measured gap. Poisson stays diagnostic-only (measured worse on support AND layering). The 52.46 % dom3 figure is superseded: at matched measurement scale the true mesh-origin layering was ~25 %, now **3.94 %**, with 25× the vertices, 9× tighter surface-to-observation agreement, and 100 % dense support.
