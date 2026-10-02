# Phase 10 — Troubleshooting

## Symptom → cause → fix

| Symptom | Cause | Fix |
|---|---|---|
| `401 Not authenticated` on every endpoint | `PLATFORM_AUTH_MODE=required` and no/missing token | `POST /api/auth/login`, send `Authorization: Bearer` |
| `401` right after login | Session expired (`PLATFORM_SESSION_TTL_HOURS`) or token revoked by logout | Log in again |
| `403 Requires one of roles operator` on mission actions | Account role is viewer/analyst | Admin creates an operator account (`POST /api/auth/users`) |
| `409 Invalid transition X → Y` | Lifecycle state machine rejects the move (e.g. resume a non-paused mission) | Check `GET /api/missions/{id}` first; allowed targets are in the message |
| `409 Mission is already PROCESSING` | Double start | Use pause/cancel or wait |
| Job `failed` with a pipeline error | Real orchestrator failure | Read `GET /api/missions/{id}/logs` → `pipeline_report`; fix inputs and `/start` again (resumes from artifacts) |
| Job stuck `running` after a crash | Process died mid-run | Worker re-adopts on startup (see `recover_interrupted`); restart the API |
| `drone_queue_worker_alive` = 0 | Worker loop exited | Check logs for `queue_process_error`; restart API |
| `/api/ready` = `not_ready` | DB unreachable / storage unwritable | Inspect `checks` in the JSON body; fix volume permissions or DB URL |
| Upload rejected `Unsupported video format` | Extension not in allowlist | Re-encode to mp4/mov/avi/mkv |
| Mission shows `FAILED` with `depth` error | No depth maps could be generated (real cause in report) | Larger images / adjust stereo params; re-run via `/start` |
| Audit events missing | Not using the lifecycle API | Only `transition()` writes `mission_events`; direct status edits are not audited |

## Failure-handling matrix (verified)

| Scenario | Behavior |
|---|---|
| Invalid upload (bad extension / traversal name) | Rejected 400 before disk write |
| Executor exception (job level) | Retries to `max_attempts`, then job `failed` + mission `FAILED` with the real error |
| Pipeline stage failure | Orchestrator retries then reports; worker marks mission `FAILED`; rerun resumes from last artifact |
| Cancel requested | Job/mission `CANCELLED`; pipeline stops at next stage boundary |
| Pause → resume | `PAUSED` → `RESUMING` → `QUEUED`; completed stages skipped via artifacts |
| Crash mid-job | Startup recovery requeues `running` jobs; mission returns to `QUEUED` |
| GPU absent on a CPU-only box | `/api/system/capabilities` reports it; guarded stages degrade to labelled CPU paths or fail clearly — never silent fakes |
| DB unreachable | `/api/ready` → `not_ready`, error surfaced in `checks`; requests 500 with the real exception logged |
| Unknown queue kind | Job marked `failed` with `No executor registered for kind '…'` |

## Logs

Structured JSON (prod). Search vectors:

```bash
# a specific request through the stack
grep '"request_id":"<id>"' backend.log
# one mission's state transitions
grep '"mission_id":"<id>"' backend.log
# worker failures
grep 'queue_process_error\|job_executor_error' backend.log
```
