# Phase 10 — Operations Guide

## Probes

| Probe | Path | Semantics |
|---|---|---|
| Liveness | `GET /api/health` | Process up, DB reachable, GPU/disk/colmap reported |
| Readiness | `GET /api/ready` | DB reachable **and** storage writable **and** queue worker alive |
| Metrics | `GET /api/metrics` | Prometheus text — point scrape jobs here |

Bare aliases (`/health`, `/ready`, `/metrics`) exist for load balancers that
refuse `/api` prefixes.

## Startup sequence

1. `init_db()` creates any missing tables (additive only).
2. `run_migrations()` applies pending ledgered migrations.
3. Bootstrap admin is created if `PLATFORM_BOOTSTRAP_ADMIN_*` are set.
4. The queue worker (`PLATFORM_QUEUE_MAX_WORKERS` concurrent sessions) starts
   and re-adopts jobs a crashed process left in `running`.
5. `GET /api/ready` flips to `ready` once DB + storage + worker are healthy.

## The job queue

- Jobs are rows in `queue_jobs` (status `queued|running|paused|completed|failed|cancelled`).
- Scheduling is priority-desc then FIFO within a priority.
- Executors are looked up by `kind`; the production `pipeline` executor runs
  the existing autonomous orchestrator in a worker thread.
- Failure: executor exceptions retry until `max_attempts`, then the job is
  `failed` and the mission `FAILED` with the real error surfaced.
- Cancellation: sets the streaming engine's cancel flag; the orchestrator
  stops at the next stage boundary and the job lands `cancelled`.
- Pause/resume: pause = cancel-at-boundary + `PAUSED`; resume clears the flag
  and re-enqueues — the orchestrator skips completed stages from artifacts,
  so nothing already finished is redone.
- Crash recovery: on worker startup, jobs stuck `running` are requeued and
  their missions moved `PROCESSING → QUEUED`.

Operational commands:

```bash
# watch queue state
curl -s localhost:8000/api/system/queue
# trigger intermediate-artifact cleanup for one mission (operator+)
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  "localhost:8000/api/missions/$MISSION/cleanup?force=false"
```

## Observability workflow

1. Check `/api/ready` after deploy (DB/storage/worker).
2. Grep logs by `request_id` from client responses.
3. Scrape `/api/metrics`:
   - `drone_http_requests_total{method,path,status}`
   - `drone_queue_jobs_total{kind,result}`
   - `drone_mission_events_total{event}`
   - `drone_queue_worker_alive` (0 ⇒ worker died)
4. Per-mission: `GET /api/missions/{id}/events` (audit), `/logs` (SSE
   history), `/artifacts` (categories).

## Storage layout & cleanup

`data/storage/{project_id}/` is categorized as raw (inputs), processed
(frames/selected), intermediate (depth/checkpoints/tmp — re-generable),
final (dense/mesh/georef/intel), logs. `cleanup_intermediates()` removes
only intermediate content older than `PLATFORM_ARTIFACT_RETENTION_DAYS`
(7 by default) — raw inputs and final outputs are never deleted by policy.

## Recovery / rollback

- Deploys are additive: new tables + migrations only; no destructive DDL.
- A crashed run resumes by re-running the mission (`/start` or worker
  re-adoption): artifact-based resume skips finished stages.
- `DELETE /api/missions/{id}` purges the workspace and DB rows; require
  operator/admin and use deliberately.
