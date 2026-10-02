# Project Audit + Mesh-Audit Scaling Regression Test — Design

Date: 2026-10-01
Scope: `drone-recon/` only. No sibling copies, no `export/`, no frontend redesign.
Status: design approved in chat; this spec records it.

## Goal

Two deliverables:

1. A **prioritized findings report** over the whole `drone-recon/` project
   ("finishing steps" review), covering every file at least at scan level and
   the load-bearing reconstruction path at read level.
2. **One code change**: a regression test that fails if the mesh-audit
   diagnostic scales super-linearly with component count, so the
   `O(n_comp x n_vertices)` pattern cannot return.

Explicitly out of scope: any fix to application code. Findings are directions,
not patches. The 112 in-flight uncommitted working-tree changes are not ours and
are never staged, reverted, or committed.

## Non-negotiable constraints (STRATA)

- No reported metric may change. The regression test is the only code change.
- No magic scaling constants; thresholds must be derived, not tuned to pass.
- Measurement over assumption: every finding carries `file:line` evidence.

## Deliverable 1 — Findings report

### Method (scan, then read)

Automated scans over `backend/app`, `backend/tests`, `frontend`, `scripts`,
`docker`, and config establish counts; then targeted deep reads of the modules
on the truth-carrying path.

Measured baseline (2026-10-01):

- `backend/app`: 157 Python modules, 44,712 LOC.
- `backend/tests`: 16,481 LOC across 58 test files.
- 130 `except Exception`, 30 `print(`, 17 `type: ignore`/`noqa`.
- 0 TODO/FIXME/HACK, 0 bare `except:`, 0 hardcoded `/Users/` paths in app code.
- Modules >1000 LOC: `trajectory_sync`, `depth_generator`, `metric_validation`,
  `pipeline_orchestrator`, `camera_pose_estimator`, `sparse_reconstruction`,
  `mesh_generator`, `dense_reconstruction`.

### Dimensions

Each finding is filed under exactly one dimension:

- **A. Correctness / ownership** — two implementations of one concept, stale
  duplicate owners, report fields that can disagree.
- **B. Silent failure / honesty** — broad catches that swallow, fallbacks that
  fabricate a value rather than record an absence.
- **C. Performance / scaling** — algorithmic complexity, repeated full passes,
  static resource budgets.
- **D. Simplification / dead code** — duplication, orphaned modules, oversized
  files that hold several responsibilities.
- **E. Tests** — invariants asserted nowhere, coverage gaps on the truth path.
- **F. Config / dependencies** — unused deps, env drift, undocumented required
  tooling.

### Report shape

Findings ranked by severity x impact x effort. Each entry:

| field | meaning |
|---|---|
| id | `AUD-###` |
| dimension | A–F |
| severity | critical / high / medium / low |
| evidence | `path:line` + the observed fact |
| why it matters | the concrete failure it can cause |
| direction | suggested approach, not a patch |
| accuracy risk | whether fixing it could move a reported number |

Delivered as a Markdown file under `drone-recon/docs/`.

## Deliverable 2 — Scaling regression test

### Target

`app/services/dense_diagnostics.py::classify_mesh_components` — the component
step of the mesh audit. Its historical implementation looped
`for cid in range(n_comp): mask = labels == cid` over the full label array, i.e.
`O(n_comp x n_vertices)` (~10^10 comparisons on the 2.36M-vertex reference mesh,
plus ~4,231 tiny sorts). It is now a single grouped pass.

The mesh-audit entry point `app/services/mesh_quality.py::audit_mesh_quality`
also contains one residual `vertices[labels == cid]` gather, bounded to the top
5 components (`O(5 x nv)`, not super-linear). The test guards both so the
unbounded pattern cannot return to either.

### Mechanism: deterministic work-count

Wall-clock timing is rejected: this host runs at 15-minute load averages around
16 on 8 logical cores, so a timing test would be flaky. Instead the test counts
array element touches via an `np.ndarray` subclass that intercepts
`__array_ufunc__` (catches `labels == cid`, which the old loop ran once per
component), `argsort`, and boolean-mask `__getitem__` (counts a full-array
pass). No application code is instrumented.

### Test A — `classify_mesh_components` work does not scale with component count

- Synthetic geometry: two connected meshes with the **same** `nv` and different
  component counts `C` and `4C` (disjoint triangulated patches).
- Pass `labels` and `n_comp` explicitly (the supported interface) as counting
  arrays; `faces` are real and consistent with the labels.
- Assertions:
  1. Element touches on `labels` at `4C` <= 1.5x the touches at `C`
     (old pattern: ~4x).
  2. Absolute budget: touches <= 4 x `nv` at both sizes.
  3. `component_count`, `class_counts`, and the top bbox are correct at both
     sizes, so the test cannot pass by degrading output.

### Test B — considered and rejected

Instrumenting the whole `audit_mesh_quality` path was considered. It is not
feasible as a test-only change: the audit opens with
`vertices = np.asarray(mesh.vertices)`, and `np.asarray` strips an `ndarray`
subclass to the base class, so a counting array cannot survive into the
function. Reaching it would require patching numpy inside application code,
which is out of scope (the only code change is the test).

Instead, the residual `vertices[labels == cid]` gather in `audit_mesh_quality`
(around lines 156-160) is recorded as a **finding** (dimension C). It is
bounded to the top 5 components today, so it is not super-linear; the finding
recommends folding it into the same group-once computation so it, too, becomes
unguardable-pattern-free.

### Verification

- Test A passes on current code.
- Detector proven to catch the pattern: an in-memory replica of the old mask
  loop driven through the same counting array reports ~`C x nv` touches, so the
  assertions would fail. (Proven in a scratch check, not committed.)
- `tests/test_dense_diagnostics.py` and the mesh-quality suites run green.

## Notes on neighbouring skills

- `overhaul`: the current dense-diagnostics diff is reviewed for the cleanest
  interface (whether `labels`/`n_comp`/`support_dist` should become one
  precomputed structure) — reported as a recommendation, not applied.
- `deepen`: nothing in scope; the request is an investigation, not a build.
- `experience`: out of scope; no user-facing interface was requested.
- `brainstorm`: its improvement ideas are section "big bets" of the report.
