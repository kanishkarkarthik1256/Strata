# STRATA — Run & Workspace Conventions

One canonical convention, two legacy roots kept for compatibility. No run IDs are
ever hardcoded in application code; demo runs are identified by **origin**
(the directory they were discovered in), never by name.

## Canonical runtime layout (what the app writes today)

| Path | Purpose | Written by |
|---|---|---|
| `data/<name>.mp4` | Input videos addressable by name (by-name mission flow) | user / repo assets |
| `data/storage/<job_id>/` | Per-mission **upload workspace**: input video copy, frames, selected/, sparse, depth, dense, georef, manifest.json, pipeline_report.json | upload & data-video flows |
| `output/<run_id>/` | Per-run **output workspace** for canonical/data-video runs; mirrors the same stage layout plus the traceability copy of the input video | canonical_demo_service, data_video_service |

Every run directory must contain `manifest.json` (authoritative) and
`pipeline_report.json` (stage timings/notes). Artifacts are always served
through `/api/runs/{run_id}/artifacts/{path}` with traversal protection —
never by raw filesystem path.

## Legacy roots (read for compatibility, never written)

| Path | Contents | Handling |
|---|---|---|
| `backend/outputs/` | Precomputed demo runs shipped with early phases (e.g. the `shitan_ms1_*` fast-demo). **This is the resolved `runs_dir` under the legacy cwd-relative layout**, so `run_service` still discovers them. | Discovered and tagged `is_demo: true` in `/api/runs` responses. `POST /api/demo/fast-demo` selects **only** tagged demo runs (newest first) and returns **404** when none exist — it never substitutes a normal mission. The UI shows them with a `PRECOMPUTED DEMO` badge. |
| `outputs/` (repo root) | Empty stub from an earlier planning document (`app/utils/run_organization.py` describes it). Unused at runtime. | Ignored. `run_organization.py` is retained only because `phase95_validator` imports it. |

`output/` vs `outputs/`: runtime uses **`output/`** (singular). Anything
documenting `outputs/<RUN_ID>/` refers to the unused planning convention.

## Rules the convention enforces

- Run IDs are generated (`<dataset>_<mission>_<YYYYMMDD_HHMMSS>` or UUID job ids);
  concurrent runs get distinct directories and cannot overwrite each other.
- A run's manifest records the exact input video (name + SHA-256 in canonical
  flows), so artifacts trace back to their input.
- Failed runs keep `status: failed` in their report; discovery never hides them.
- Demo runs cannot leak into normal mission results: tagging is by discovery
  root, and the demo endpoints only ever return tagged runs.
- Restarting the app preserves all runs (pure filesystem discovery, 2s cache).
