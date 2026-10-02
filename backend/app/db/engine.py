"""Async SQLAlchemy engine and session factory.

The engine is created lazily on first call to ``init_db`` / ``get_session`` so
that settings can be loaded first.
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.settings import settings
from app.logging_config import get_logger

log = get_logger("drone_recon.db")

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """Create the async engine, session factory, and all tables."""
    global _engine, _session_factory

    log.info("initialising_database", url=settings.database.url)

    _engine = create_async_engine(
        settings.database.url,
        echo=settings.database.echo,
        connect_args={"check_same_thread": False},  # SQLite specific
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Import metadata so create_all knows about all tables.
    from app.db.models import Base  # noqa: F811

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Additive migrations (indexes etc.) — no destructive changes.
        from app.db.migrations import run_migrations

        await run_migrations(conn)

    log.info("database_initialised")


async def close_db() -> None:
    """Dispose the engine (call on shutdown)."""
    global _engine, _session_factory

    if _engine is not None:
        log.info("disposing_database_engine")
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an ``AsyncSession`` — use as a FastAPI dependency."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
