"""Shared test fixtures for DroneRecon backend tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.engine import get_session
from app.db.models import Base
from app.main import create_app


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a small synthetic AVI video for testing.

    Uses MJPG codec in AVI container — works without FFMPEG.
    """
    video_path = tmp_path / "test_drone.avi"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    assert writer.isOpened(), "VideoWriter failed to open"

    for i in range(30):  # 1 second at 30fps
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        x = int(100 + i * 10)
        cv2.rectangle(frame, (x, 200), (x + 80, 280), (0, 255, 0), -1)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture(autouse=True)
def isolate_global_settings():
    """Restore the ``app.config.settings`` singleton around every test.

    Services read settings at call time, so several tests point
    ``settings.storage.base_path`` (and e.g. ``settings.mesh.atlas_width``)
    at their ``tmp_path`` by direct assignment. Nothing restored those
    writes, so each one leaked into every later test in the same process —
    which resolved workspaces inside a previous test's already-deleted
    tmp_path. That is why the suite was order-dependent (an end-to-end
    artifact test passed alone but failed in a full run). One owner for the
    restore, instead of every test remembering to undo its own writes.
    """
    from app.config.settings import settings

    before = settings.model_copy(deep=True)
    yield
    for name, previous in before.__dict__.items():
        current = getattr(settings, name, None)
        if hasattr(previous, "__dict__") and hasattr(current, "__dict__"):
            for field, value in previous.__dict__.items():
                if field in current.__dict__ or hasattr(current, field):
                    setattr(current, field, value)
        else:
            setattr(settings, name, previous)


@pytest_asyncio.fixture
async def db_session():
    """Create an in-memory SQLite session for each test."""
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    """Async test client with overridden DB session dependency."""
    app = create_app()

    async def _override_session():
        yield db_session

    app.dependency_overrides[get_session] = _override_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
