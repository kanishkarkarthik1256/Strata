# Phase 10 — Enterprise Platform Architecture

Phase 10 wraps the Phase 1–9 reconstruction/intelligence stack in a
production-oriented platform layer **without redesigning the pipeline
architecture**. The plugin `PipelineStage` system, the autonomous
orchestrator, and the per-phase chains (`MESH_CHAIN`, `INTEL_CHAIN`,
`PLANNING_CHAIN`) remain the foundation; every new capability integrates
with them.

```
                 ┌────────────────────────────────────────────┐
                 │            HTTP / SSE clients               │
                 └───────────────────┬────────────────────────┘
                                     │
                     ┌───────────────▼───────────────┐
                     │  FastAPI + middleware (auth,   │
                     │  request-id, timing, metrics)  │
                     └───────────────┬───────────────┘
                                     │
        ┌────────────────┬───────────┴────────────┬─────────────────┐
        ▼                ▼                        ▼                 ▼
  /api/auth      /api/missions          /api/system          legacy /api/*
  users · RBAC   lifecycle + audit      ready · metrics      phases 1–9
  sessions       queue_jobs             capabilities        (unchanged routes)
        │                │                                     │
        ▼                ▼                                     ▼
  auth_service    mission_service         queue worker      Pipeline orchestrator
  (PBKDF2,        (state machine,   ┌───► (in-process,      (frames→sparse→depth→
   SHA-256 tokens) audit trail)      │     priority,retry,  dense→georef + chains)
                                     │     pause/resume,    │
                                     │     crash recovery)  │
                                     │                      │
  SQLite (SQLAlchemy async) ◄────────┴─────────── streaming engine (SSE) ◄────┘
  users · missions · queue_jobs ·      disk: data/storage/{project_id}
  projects/jobs/frames/models_3d        categories raw/processed/intermediate/final
```

## Layers

| Layer | Owner | Notes |
|---|---|---|
| HTTP / middleware | `app/main.py`, `app/middleware/*` | Auth enforcement (opt-in `required`), X-Request-ID, timing, Prometheus counters |
| Mission surface | `app/routes/missions.py`, `app/services/mission_service.py` | Lifecycle state machine + append-only audit |
| Auth | `app/routes/auth.py`, `app/services/auth_service.py` | Users, roles (viewer<analyst<operator<admin), bearer sessions |
| Queue | `app/services/job_queue.py` | DB-backed jobs, executor registry (`pipeline` → existing orchestrator) |
| Observability | `app/routes/system.py`, `app/services/metrics.py`, `app/services/resources.py` | /api/ready, /api/metrics, capabilities |
| Storage | `app/services/storage_manager.py` | Deterministic categories + cleanup policy |
| Pipeline | `app/services/pipeline_orchestrator.py`, `app/services/pipeline_stage.py` | **Unchanged** foundation |

## State ownership

- **One owner per mission**: `mission_service.transition()` is the only
  function that changes mission status for user-driven actions; every change
  writes a `mission_events` audit row.
- **One owner per job**: `job_queue.process_next()` is the only executor of
  queue jobs; the worker loop is the only writer of `queue_jobs.status` for
  run progress.
- **Artifact resume**: the orchestrator skips stages whose artifacts exist;
  pause/resume therefore never restarts a completed stage.

## Data flow

1. An operator uploads a dataset (`POST /api/upload`) → a project row +
   workspace is created.
2. `POST /api/missions` binds a mission record (CREATED) to that project.
3. `POST /api/missions/{id}/start` transitions to QUEUED and enqueues a
   `pipeline` job.
4. The queue worker runs the existing orchestrator (stage-by-stage,
   publish over SSE, checkpoint artifacts on disk) and transitions the
   mission PROCESSING → COMPLETED/FAILED.
5. `POST /{id}/pause` sets the streaming engine's cancel flag → the pipeline
   stops at the next stage boundary; `POST /{id}/resume` clears it and
   re-enqueues — completed stages are skipped from artifacts.
6. Clients observe via `GET /api/missions/{id}/events`, `/logs`,
   `/artifacts` and the mission-scoped SSE stream.

## Additions vs. redesigns

New DB tables are strictly additive (`users`, `auth_sessions`, `missions`,
`mission_events`, `queue_jobs`, plus a `schema_migrations` ledger); existing
tables are untouched. Migrations run automatically at startup and are
recorded idempotently.
