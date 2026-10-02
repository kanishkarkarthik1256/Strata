# Phase 10 — Security Review

## What is enforced

**Authentication.** Passwords are hashed with PBKDF2-HMAC-SHA256 (per-user
16-byte salt, 200 000 iterations) from the standard library — no plaintext
is stored. Session tokens are 256-bit random values; only their SHA-256
digest is persisted, so a leaked database cannot be replayed directly.
Sessions expire (`PLATFORM_SESSION_TTL_HOURS`) and are revoked on logout.

**Authorization (server-side).** Roles form a strict hierarchy —
`viewer < analyst < operator < admin` — enforced by FastAPI dependencies on
every Phase-10 route:

- mission mutation (create/start/pause/resume/cancel/cleanup): **operator+**
- mission deletion: **admin/operator**
- user management: **admin**
- reads (status/events/logs/artifacts): any authenticated account

In `PLATFORM_AUTH_MODE=required` an auth middleware additionally blocks
unauthenticated access to *every* `/api/*` route except an explicit public
allowlist (`/api/health`, `/api/ready`, `/api/metrics`, docs, login,
register). The default (`disabled`) preserves pre-Phase-10 endpoints open —
documented backward compatibility for local/CI; production must set
`required`.

**Path traversal (fixed in Phase 10).** The upload service previously
wrote `workspace / file.filename` where the filename was client-controlled —
a crafted name (`../../evil.avi`) could escape the project workspace. It now
sanitizes to a plain basename, rejects unsafe characters, and enforces the
extension allowlist **before any bytes are written**; content is validated
after the write and the file is removed on validation failure.

**Subprocesses.** The only external-process calls (`ffprobe` in metadata
extraction) pass an argument list — never a shell string — and are not
user-injectable paths beyond the validated upload.

**Secrets.** No credentials/API keys are hard-coded. Configuration comes
from environment / `.env` (git-ignored); `.env.example` ships defaults only.
Bootstrap admin credentials are environment-supplied and the account is
created idempotently at startup.

**Logs & request IDs.** Every request is tagged with `X-Request-ID`
(inbound header honoured, otherwise generated) and responses echo it.
Structured logs carry the id, method, path, status — never the password or
raw token. `Authorization` headers are not logged.

## Residual risks (not claimed fixed)

1. **Legacy endpoints open by default.** Pre-Phase-10 routes have no
   per-route auth; with `PLATFORM_AUTH_MODE=disabled` (the dev/CI default)
   they are reachable without a token. Deployments must enable `required`.
2. **No TLS/rate limiting at the app layer.** Terminate TLS at the ingress
   and add a rate limiter / WAF in front of `/api/auth/login` and the
   upload endpoint (multi-GB uploads are CPU/disk heavy).
3. **CORS** is allow-list based and environment-configured; keep the origin
   list tight in production. `allow_credentials=True` must not be combined
   with `*` origins.
4. **SQLite single-writer.** The job queue worker is in-process; horizontal
   scaling would require moving the queue to Postgres/Redis. Concurrency is
   bounded by `PLATFORM_QUEUE_MAX_WORKERS`.
5. **Password policy** is minimal (length ≥ 8) — no breach-dictionary or
   rotation enforcement yet.
6. **Session revocation** is per-token, not per-user; an admin can disable
   an account (`is_active=False`) which invalidates lookups immediately.
7. No audit of *data-plane* reads (artifacts/logs views are not logged to
   the mission audit trail) — only state transitions are.

## Test coverage

`tests/test_platform.py` exercises: register/login/me/logout, expired-token
rejection, elevated-role creation requiring an admin actor, admin-only user
management, RBAC 403s, auth-required mode blocking legacy endpoints while
health stays public, and upload traversal rejection.
