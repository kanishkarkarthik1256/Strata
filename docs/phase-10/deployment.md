# Phase 10 — Deployment Guide

## Environments

| | development | testing | production |
|---|---|---|---|
| `DEPLOYMENT` | `development` | `testing` | `production` |
| `PLATFORM_AUTH_MODE` | `disabled` (default) | `required` | `required` |
| Admin bootstrap | optional | set | **set** |
| Logs | console | JSON | JSON |
| DB | sqlite file | sqlite file / tmp | sqlite (single node) or migrate to Postgres |

## Backend (local)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:create_app --factory --reload
# open http://localhost:8000/api/docs
```

On first boot the DB initializes, migrations apply, and (if configured) the
bootstrap admin is created.

## Production bootstrap (first run)

1. `cp .env.example .env` and set at minimum:
   ```
   PLATFORM_AUTH_MODE=required
   PLATFORM_BOOTSTRAP_ADMIN_EMAIL=admin@yourorg.example
   PLATFORM_BOOTSTRAP_ADMIN_PASSWORD=<strong, one-time password>
   ```
2. Start the API. The admin account is created automatically.
3. Immediately `POST /api/auth/login`, then rotate the password via an
   admin-created replacement account + `DELETE` of the bootstrap user
   (user deletion is not yet exposed — disable via DB or keep the account
   behind an email alias you control).
4. Create operators/analysts with `POST /api/auth/users`.

## Docker (docker compose)

```bash
docker compose up --build
# backend health: GET :8000/api/ready
```

- Compose sets `PLATFORM_AUTH_MODE` / bootstrap admin from the environment
  (defaults to `disabled` + no admin; override in `.env`).
- The backend healthcheck targets `/api/ready` (DB + storage + queue worker),
  so the frontend container starts only once the platform is truly ready.
- GPU passthrough requires the NVIDIA Container Toolkit; the service reports
  its absence honestly via `/api/system/capabilities` (`gpu.available=false`)
  instead of faking inference.

## Environment reference (Phase 10 keys)

```
PLATFORM_AUTH_MODE=disabled|required
PLATFORM_SESSION_TTL_HOURS=12
PLATFORM_TOKEN_BYTES=32
PLATFORM_BOOTSTRAP_ADMIN_EMAIL=
PLATFORM_BOOTSTRAP_ADMIN_PASSWORD=
PLATFORM_QUEUE_MAX_WORKERS=1
PLATFORM_QUEUE_POLL_SECONDS=1.0
PLATFORM_QUEUE_DEFAULT_PRIORITY=5
PLATFORM_QUEUE_MAX_ATTEMPTS=2
PLATFORM_ARTIFACT_RETENTION_DAYS=7
DEPLOYMENT=development|testing|production
```

## Scale notes

The platform is single-node: one API process, one in-process queue worker,
SQLite. Moving to multi-node means (in order): Postgres for `queue_jobs` /
`missions` / `auth_sessions`, a shared Redis queue in front of the worker,
and object storage for `data/storage` — each is a drop-in behind the service
boundaries already drawn here. Do not scale the worker beyond one writer per
database until then.
