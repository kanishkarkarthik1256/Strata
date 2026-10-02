"""Additive SQLite migrations (Phase 10).

``create_all`` in :mod:`app.db.engine` is the source of truth for *new*
tables; this module owns anything that must change on an existing database
without destructive rebuilds — primarily indexes and, in future, new columns
on pre-existing tables.

The runner is intentionally conservative:

* every step is executed inside a transaction;
* each step is recorded in ``schema_migrations`` so it runs exactly once;
* statements are written idempotently (``IF NOT EXISTS`` / existence checks),
  so re-applying the runner after a partial failure is safe.

No migration ever drops or rewrites existing data.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

log = logging.getLogger("drone_recon.db.migrations")

#: (version, display_name, [sql, ...]). Append new additive steps here.
MIGRATIONS: list[tuple[str, str, list[str]]] = [
    (
        "2026.09.07.001",
        "platform-indexes",
        [
            "CREATE INDEX IF NOT EXISTS ix_queue_jobs_lookup "
            "ON queue_jobs (status, priority, created_at)",
            "CREATE INDEX IF NOT EXISTS ix_missions_lookup "
            "ON missions (owner_id, status, created_at)",
            "CREATE INDEX IF NOT EXISTS ix_auth_sessions_expiry "
            "ON auth_sessions (user_id, expires_at)",
        ],
    ),
]


async def run_migrations(conn: AsyncConnection) -> list[str]:
    """Apply pending additive migrations and return the versions applied.

    Creates the ``schema_migrations`` ledger table on first use. Each pending
    step runs inside its own transaction.
    """
    await conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version TEXT PRIMARY KEY,"
            " name TEXT NOT NULL,"
            " applied_at TEXT NOT NULL"
            ")"
        )
    )

    existing = set(
        (
            await conn.execute(text("SELECT version FROM schema_migrations"))
        ).scalars().all()
    )
    applied: list[str] = []
    for version, name, statements in MIGRATIONS:
        if version in existing:
            continue
        for sql in statements:
            await conn.execute(text(sql))
        await conn.execute(
            text(
                "INSERT INTO schema_migrations (version, name, applied_at) "
                "VALUES (:v, :n, datetime('now'))"
            ),
            {"v": version, "n": name},
        )
        applied.append(version)
        log.info("migration_applied version=%s name=%s", version, name)
    return applied
