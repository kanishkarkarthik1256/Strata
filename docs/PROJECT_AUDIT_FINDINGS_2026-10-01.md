# Project Audit — Findings Register

**Date:** 2026-10-01
**Scope:** `drone-recon/` only (sibling copies and `export/` deliberately excluded).
**Method:** automated scans (ruff, grep, import-sweep, orphan/dependency analysis)
then targeted deep reads of the truth-carrying path.
**Author:** audit pass; findings are directions, not patches.

## Baseline measured

| metric | value |
|---|---|
| `backend/app` | 157 modules, 44,712 LOC |
| `backend/tests` | 58 files, 16,481 LOC |
| ruff findings | 928 in `app`, 263 in `tests` (both dev-deps configured, neither gated) |
| `except Exception` | 130 in `app` |
| `print(` | 30 in `app` |
| import-sweep on the real venv (Python 3.9.6) | 0 modules fail to import |
| TODO / FIXME / bare `except:` / hardcoded `/Users/` | 0 |

## Severity legend

- **critical** — can corrupt or silently falsify a reported result.
- **high** — a real bug or a guarded invariant that is not actually guarded.
- **medium** — degrades diagnosability, correctness in a rare branch, or waste.
- **low** — cleanup; no behavioural risk today.

---

## A. Correctness / ownership

### AUD-001 — **high** — `data_video_service`: a latent `UnboundLocalError` aborts mission start
`app/services/data_video_service.py:329` (ruff F823).

The function `start_data_video_mission` uses `get_logger(__name__)` in the
`cameras.txt` failure branch, but later (in the SRT branch) does
`from app.logging_config import get_logger` **inside the function body**. A local
`import` anywhere in a function makes that name local for the *whole* function,
so the earlier reference is an unbound local.

- **Why it matters:** if `cameras.txt` conversion raises, instead of logging a
  warning the code raises `UnboundLocalError` from inside the `except` block.
  That new exception replaces the handled one and escapes
  `start_data_video_mission`, so a calibration failure becomes a mission-start
  failure.
- **Direction:** drop the inner imports and use the module-level `log` already
  bound at the top of the file.
- **Accuracy risk:** no — control-flow only.

### AUD-002 — **medium** — duplicated and shadowed definitions
- `app/services/resources.py:287` and `:465` define `tool_available` **twice**
  with identical bodies. The second shadows the first; one is dead.
- `app/services/depth_fusion.py:37` and `:179` both import `ThreadPoolExecutor`.
- `app/services/depth_anything_v2/__init__.py:116` and `:130` both bind `torch`.

- **Why it matters:** shadowed copies drift; a future edit to one leaves the
  other live and wrong. The `resources.py` pair is the exact "two owners of one
  concept" pattern the repo's own STRATA rule forbids.
- **Direction:** delete the shadowed copies; make the import unconditional at
  module top.
- **Accuracy risk:** no.

### AUD-003 — **medium** — an accuracy validator with no production caller
`app/services/ground_truth_validator.py` is not imported by any route, service,
or pipeline stage — only by two test files (and `test_accuracy_single_owner.py`
reads its *source text* to assert an invariant about it).

- **Why it matters:** a validator that never runs in a real mission cannot catch
  a real regression; it only satisfies a test that inspects its source. The
  accuracy single-owner invariant is asserted structurally, not behaviourally.
- **Direction:** either wire it into the validation path (or delete it and move
  the invariant into `metric_validation`), so the single-owner rule is enforced
  by behaviour rather than by reading a file.
- **Accuracy risk:** yes if it is wired naively; it must not become a second
  accuracy computation.

### AUD-004 — **low** — cause chain lost on re-raise
`app/services/pipeline_orchestrator.py:475` raises without `from` (ruff B904).

- **Direction:** `raise ... from err` so the operator sees the true cause.

---

## B. Silent failure / honesty

The repo's STRATA rule is "record the reason rather than let the key vanish."
Several `except Exception: pass` sites violate it. Not all are wrong — but the
ones on the truth path are:

### AUD-005 — **high** — silent fallback to identity intrinsics in the pose path
`app/services/camera_pose_estimator.py:333-334`.

If reading pycolmap's camera calibration matrix fails, `intrinsics` stays
`np.eye(3)` and the pose is emitted with **no note**. A wrong calibration matrix
is exactly the kind of error that moves metric numbers while everything still
"looks" successful.

- **Direction:** record a note/counter when the identity fallback is taken
  (the module already has a note mechanism elsewhere).
- **Accuracy risk:** yes — this is a silent accuracy-affecting fallback.

### AUD-006 — **medium** — silent skips in the accuracy report
`app/services/metric_validation.py:214` and `:1329`.

- `:214` silently omits the `dense_to_mesh` block if the mesh report can't be
  read.
- `:1329` silently skips loading `poses.json`, which silently removes the
  absolute-registration input. The report then carries less than the operator
  expects with no reason attached, contrary to the module's own convention of
  naming why a measurement is absent.
- **Direction:** set a reason field instead of `pass`.
- **Accuracy risk:** no direct falsification (nothing is fabricated), but it
  hides when a validation was not performed.

### AUD-007 — **medium** — stage-cache hash swallows read failure
`app/services/stage_caching.py:37`.

If `stat()`/`read()` raises, the function returns a hash over whatever was
written so far (possibly empty). Two different inputs can then hash equal, and a
cache hit can reuse a stale stage result.

- **Direction:** on failure return a sentinel that forces a miss, and log it.
- **Accuracy risk:** yes — a false cache hit can serve stale geometry.

### AUD-008 — **low** — remaining `except Exception: pass`
`routes/health.py:79` (legit: DB probe), `run_service.py:459`,
`depth_generator.py:242,495`, `pipeline_orchestrator.py:479,1182`,
`performance_profiler.py` (5 sites).

- **Direction:** audit each; the profiler/cache ones are likely fine, the
  pipeline ones should at least log.

---

## C. Performance / scaling

### AUD-009 — **medium** — residual per-component mask gather in the mesh audit
`app/services/mesh_quality.py:156-160` still does
`vertices[labels == cid]` for the top-5 components.

Today this is `O(5 x nv)` — bounded, not super-linear — so it is not the bug the
new regression test guards. But it is the *same pattern* that caused the
`O(n_comp x n_vertices)` blow-up in `classify_mesh_components`, and it is one
`min(5, …)` edit away from returning at full scale.

- **Direction:** reuse the group-once `bbox_min`/`bbox_max` already computed by
  `classify_mesh_components` (it returns `components_top20` with bboxes) instead
  of re-masking. This also removes a duplicate connected-component-derived scan.
- **Accuracy risk:** no — values are order-independent.

### AUD-010 — **medium** — repeated KD-tree construction
39 `cKDTree(...)` sites across `app/services`. `audit_mesh_quality` was just
fixed to reuse one query, but the broader pattern remains: several stages build
a tree over the same cloud more than once.

- **Direction:** where a stage builds a tree and then rebuilds it for a second
  diagnostic, thread the first tree (or the distances) through, as the mesh
  audit now does.
- **Accuracy risk:** no.

### AUD-011 — **low** — Python loop over components for per-component medians
`app/services/dense_diagnostics.py:611` computes `np.median` per component in a
Python loop. Total work is linear, but it is `n_comp` interpreter iterations
where the surrounding code is fully vectorized.

- **Direction:** if component counts grow, use a sorted-group + `reduceat`
  approach (or accept the loop; it is not a measured bottleneck today).

### AUD-012 — **info** — untimed post-fusion chain
`app/services/dense_reconstruction.py` times every fusion substage but the
post-fusion chain (mesh, layer classification, viewer LOD, texture, mesh-input
KD-tree) is one ~692 s untimed block on the reference run.

- **Direction:** give it `stage()` timers so a future regression is visible
  instead of hiding in `total - sum(substages)`.
- **Accuracy risk:** no.

---

## D. Simplification / dead code

### AUD-013 — **medium** — unused declared dependencies
Zero imports anywhere under `backend/` for: **`imagehash`**, **`piexif`**,
**`trimesh`**, **`tqdm`**, **`aiofiles`**.

- **Why it matters:** every one is install weight, attack surface, and renewal
  burden; `trimesh` in particular is a large dependency pulled for nothing.
- **Direction:** drop from `pyproject.toml` after confirming no runtime string
  import (none found).
- **Accuracy risk:** no.

### AUD-014 — **low** — oversized modules
Eight modules exceed 1000 LOC and mix several responsibilities:
`trajectory_sync` (1710), `depth_generator` (1640), `metric_validation` (1430),
`pipeline_orchestrator` (1326), `camera_pose_estimator` (1286),
`sparse_reconstruction` (1223), `mesh_generator` (1218),
`dense_reconstruction` (1145).

- **Direction:** split by the concept each already documents in its docstring
  (e.g., `metric_validation` separates report-building from co-registration).
  Do not split for its own sake; split where a boundary is already visible.
- **Accuracy risk:** no.

### AUD-015 — **low** — untracked build/runtime residue on disk
`frontend/dist` (2.8 MB), `frontend/.vite` (7.1 MB), `__pycache__`, `.DS_Store`.
All correctly ignored and untracked — recorded for completeness, no action
needed.

---

## E. Tests

### AUD-016 — **high** — a disabled assertion that always passes
`tests/test_sparse_conditioning.py:401`:

```python
assert meta["poses_sha256"] == _poses_fingerprint(poses) if False else True
```

Due to conditional-expression precedence this is
`assert ((...) if False else True)` — it is **always true** and `_poses_fingerprint`
is never called (the name is undefined; ruff F821). The intended check that the
stored fingerprint matches the poses is not being made.

- **Direction:** restore the real assertion (define/import the helper) or delete
  the line and keep the following genuine `assert meta["poses_sha256"]`.
- **Accuracy risk:** yes — provenance of the pose input is currently unverified.

### AUD-017 — **medium** — other `if False else` test surgery
`tests/test_trajectory_sync.py:41` and `:384` use `if False else` to swap a
value/behaviour. This is dead-code in tests that hides what is actually being
exercised.

- **Direction:** delete the dead branch and keep the live one.

### AUD-018 — **medium** — undefined annotation in a test
`tests/test_depth_refusal_robustness.py:186` annotates a parameter as `Path`
without importing it (ruff F821). Harmless only because the file has
`from __future__ import annotations`; any tool that resolves hints (or a future
runtime that evaluates them) breaks.

- **Direction:** add the `pathlib.Path` import.

### AUD-019 — **medium** — services with no test coverage
10 of 104 service modules are not mentioned in any test, including two on the
metric path: **`metric_scale_validator`** and **`phase95_validator`**
(`metric_scale_validator` is reachable from `phase95_validator`, which is
reachable from `pipeline_orchestrator`). Also: `area_analysis`,
`canonical_demo_service`, `cpu_budget`, `depth_prefetch`,
`infrastructure_analyzer`, `mission_learning`, `object_detector`, `resources`.

- **Direction:** prioritise `cpu_budget` (it sets the process-wide thread
  budget) and the two metric validators.
- **Accuracy risk:** yes — untested metric validation is exactly where a silent
  regression can hide.

### AUD-020 — **low** — lint debt with no gate
928 app + 263 test ruff findings; `mypy` configured `strict = true` but not run.
38 unused imports and 18 unused variables in `app` point at churn without a
cleanup pass.

- **Direction:** add ruff (and a scoped mypy) to CI on changed files at minimum;
  fix the autofixable subset in one commit.
- **Accuracy risk:** no.

---

## F. Config / dependencies

### AUD-021 — **high** — the project runs on a Python version it declares unsupported
`backend/pyproject.toml` declares `requires-python = ">=3.11"`, `ruff`
`target-version = "py311"`, and `mypy python_version = "3.11"` — but the actual
venv is **Python 3.9.6**. 145 of 157 `app` modules carry
`from __future__ import annotations`, and the code uses PEP-604 `X | None`
unions that only work on 3.9 because of those future imports.

- **Why it matters:** the declared and actual runtimes disagree. On 3.11 a
  module that *lacks* the future import still works, so the drift stays hidden;
  on a fresh 3.9 install the opposite is true. Either the declaration or the venv
  is wrong, and nobody is pinned to the truth.
- **Direction:** pick one runtime, pin it (`requires-python`, CI matrix, Docker
  base, venv), and align `mypy`/`ruff` target versions to it.
- **Accuracy risk:** no direct; environment risk.

### AUD-022 — **medium** — no CI gate
Neither ruff nor mypy runs automatically (both are dev-dependencies). Nothing
prevents AUD-001/AUD-016-class defects from merging.

- **Direction:** add a CI job (or a pre-commit hook) running ruff + mypy +
  `pytest` on the backend.

---

## Overhaul review of the current diff (`dense_diagnostics` / `mesh_quality`)

The vectorization itself is correct and output-identical (verified: old mask
loop vs new function on the real 2.36 M-vertex mesh returned `IDENTICAL: True`,
41.6 s → 0.4 s). The interface has two clean-up opportunities:

1. **`labels` + `n_comp` should travel as one thing.** `classify_mesh_components`
   now takes `labels: np.ndarray | None` *and* `n_comp: int | None`, which must
   be supplied together and can be derived from each other
   (`n_comp = int(labels.max()) + 1`). A single small structure — e.g. a
   `component_labels(vertices, faces) -> (n_comp, labels)` helper, or a
   `MeshComponents` value passed as `components=` — removes the "two params that
   must agree" hazard and gives the label computation one owner (it currently
   exists in three places: `classify_mesh_components`, `audit_mesh_quality`, and
   `mesh_support`'s unsupported-region block).

2. **The `4_000_000` guard is duplicated.** `mesh_quality.py` computes
   `sup_dist if len(cloud_xyz) <= 4_000_000 else None` to mirror
   `mesh_support(..., max_points: int = 4_000_000)`. That magic number now lives
   in two modules and must be kept in sync. Let `mesh_support` decide (it already
   has `max_points`), or expose the constant.

3. **The residual top-5 mask loop** (AUD-009) undercuts the "group once"
   principle the diff establishes; folding it in makes the whole audit
   consistent and removes a second connected-component-derived scan.

None of these change a number; all three are interface hygiene.

---

## Big bets (what has not been considered)

- **Make accuracy a first-class, gated artifact.** AUD-003 + AUD-019 + AUD-006
  together say the accuracy path is partially unwired, partially untested, and
  sometimes silent. A single `accuracy_report` produced by one owner, with a
  required reason for every absent measurement, would convert several "medium"
  findings into a structural guarantee.
- **Turn the STRATA failure-honesty rule into a lint.** A custom ruff rule (or a
  small AST test) forbidding `except Exception: pass` and bare `return None` in
  `app/services` without an adjacent reason/note would prevent AUD-005/006/007
  from recurring — the same way the new regression test prevents the O(n²) mask
  loop.
- **One performance budget file.** The `cKDTree` census (AUD-010) and the untimed
  chain (AUD-012) both stem from performance work being ad hoc. A single manifest
  of "this stage costs X on the reference run" would make the next regression
  visible immediately and replace the manual benchmark scripts.

---

## Verification performed

- New regression test passes; its detector was proven against a faithful replica
  of the old mask loop (OLD: 32·nv → 128·nv, ratio 4.00, **fails**; NEW: 1.00·nv
  at both counts, ratio 1.00, **passes**).
- 135 passed / 7 skipped across the dense, mesh, texture, viewer and STRATA mesh
  suites; 16/16 `test_metric_validation.py` passed (truth path unaffected).
- Import-sweep of all 157 `app` modules on the real venv: 0 failures.

## Not covered in this pass

- `frontend/` was inventoried and scanned for hardcoded URLs but not read
  line-by-line; `docker/` and `scripts/` were not audited.
- Bundled `depth_anything_v2` vendor code was not reviewed (its `E712` findings
  are upstream style, not project defects).
