# Phase 10 — API Reference

All endpoints are real and served by the running FastAPI application; the
interactive OpenAPI document (`/api/docs`) is generated from the same
route table and therefore matches this list exactly.

## Authentication

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/auth/register` | public | Create an account (viewer/analyst by default; elevated roles require an admin caller) |
| POST | `/api/auth/login` | public | Exchange credentials for a bearer token |
| POST | `/api/auth/logout` | bearer | Revoke the current session |
| GET | `/api/auth/me` | bearer | Current user profile + role |
| GET | `/api/auth/users` | admin | List users |
| POST | `/api/auth/users` | admin | Create a user with an explicit role |

Tokens are returned once (`{"token": "...", "token_type": "bearer"}`) and
sent as `Authorization: Bearer <token>`.

## Missions

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/missions` | bearer | List missions (`?status=QUEUED`, `?mine=true`) |
| POST | `/api/missions` | operator+ | Create a mission around an existing project |
| GET | `/api/missions/{id}` | bearer | Mission detail + lifecycle status |
| POST | `/api/missions/{id}/start` | operator+ | Enqueue a pipeline run (`{plugins, force, priority}`) |
| POST | `/api/missions/{id}/pause` | operator+ | Pause at the next stage boundary |
| POST | `/api/missions/{id}/resume` | operator+ | Resume from stored artifacts |
| POST | `/api/missions/{id}/cancel` | operator+ | Cancel (terminal, re-flight allowed) |
| GET | `/api/missions/{id}/events` | bearer | Audit trail |
| GET | `/api/missions/{id}/logs` | bearer | Streaming-engine run history + pipeline report |
| GET | `/api/missions/{id}/stream` | bearer | SSE live pipeline progress |
| GET | `/api/missions/{id}/artifacts` | bearer | Categorized artifact listing |
| POST | `/api/missions/{id}/cleanup` | operator+ | Purge intermediate artifacts (`?force=true`) |
| DELETE | `/api/missions/{id}` | admin/operator | Delete mission + workspace |

Lifecycle: `CREATED → QUEUED → PROCESSING → COMPLETED | FAILED`; pause from
PROCESSING/QUEUED → `PAUSED → RESUMING → QUEUED`; cancel → `CANCELLED`
(re-flight allowed). Illegal transitions return `409` with the allowed set.

## Observability

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/health` | public | Health (db, GPU, disk, colmap) |
| GET | `/api/ready` | public | Readiness (db reachable, storage writable, queue worker alive) |
| GET | `/api/metrics` | public | Prometheus-text counters/gauges from real events |
| GET | `/api/system/capabilities` | public | CPU/RAM/disk/GPU/tool report |
| GET | `/api/system/queue` | public | Queue depth by job state |
| GET | `/health`, `/ready`, `/metrics` | public | Bare-path aliases for load-balancer probes |

Metrics are derived exclusively from observed events (HTTP requests, queue
jobs, auth events) — nothing is fabricated.

## Legacy endpoints (Phases 1–9)

Unchanged and still available under `/api/*`: upload, frame extraction,
reconstruction, dense/stream, pipeline start/status/cancel, digital-twin,
intel, mission planning/simulation. **When `PLATFORM_AUTH_MODE=required`
these too require a bearer token** except the public allowlist.

## Errors

Consistent envelope throughout:

```json
{ "error": "Mission not found: abc", "status_code": 404, "detail": null }
```

- `400` invalid request / no bound project
- `401` missing/invalid/expired token
- `403` authenticated but insufficient role
- `404` unknown mission/project
- `409` illegal lifecycle transition or already-running state
- `422` validation error
