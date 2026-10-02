# Depth Model Validation (Phase 9.5.1)

## Learned depth: VALIDATED

| Item | Value |
|---|---|
| Model | Depth Anything V2 (ViT-Small, DPT architecture) |
| Checkpoint | `backend/models/weights/depth_anything_v2_vits.pth` — official, 98 MB |
| Architecture | vendored faithful port in `app/services/depth_anything_v2/` (`dpt.py`, `dinov2.py`, `dinov2_layers.py`, `blocks.py`, `transform.py`) |
| Load | `load_state_dict` exact match against the official checkpoint (no weight gaps) |
| Inference | 10/10 views on real Shitan ms1 images (4000×3000), ~2 s/view on CPU |
| Output | per-view `.npy` depth map + `.png` visualization + `.json` metadata |
| Device | cpu |

## Metric depth: NOT VALIDATED

Depth Anything V2 produces **relative** (affine-invariant) depth. There is
no scale reference or ground-truth control in the repository, so:

- outputs are tagged `"metric": false, "metric_scale": "NOT_VALIDATED"`
  in `learned_depth_info.json` and the run manifest;
- learned depth is stored in the separate `depth_learned/` run directory
  and is **never fused into the metric dense cloud**;
- `depth_generator` backend `auto` resolves to **stereo only** so the
  dense pipeline can never silently consume relative depth as meters.

## What was executed

- `python -c` load test: checkpoint loads, one real image inferred → depth map 4000×3000
- Functional run stage `depth_learned`: 10/10 views generated, `metric=false`

## Honest caveats

- No photometric/geometric scale validation was performed (no ground truth).
- CPU-only inference (~2 s per 4000×3000 view); GPU timing is not claimed.