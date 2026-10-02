"""Tests for the Phase 10 enterprise platform layer.

Covers: additive migrations, auth + RBAC (register/login/logout/me/users,
auth-required mode), mission lifecycle state machine + audit trail, the
DB-backed job queue (priority, retry, pause/resume/cancel, crash recovery),
storage catalog/cleanup, observability endpoints (/ready /metrics
capabilities), and the upload security fix (filename traversal).

Queue mechanics are tested with lightweight executors registered only in this
file; the production ``pipeline`` executor is asserted to be registered and
wired to the orchestrator, whose behaviour is covered by the Phase 6–9 suites.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db import migrations as migration_module
from app.db.models import (
    MISSION_CANCELLED,
    MISSION_COMPLETED,
    MISSION_CREATED,
    MISSION_PAUSED,
    MISSION_PROCESSING,
    MISSION_QUEUED,
    MISSION_RESUMING,
    Base,
    Project,
    User,
)
from app.services import job_queue, mission_service
from app.services.auth_service import (
    ensure_bootstrap_admin,
    hash_password,
    verify_password,
)
from app.services.storage_manager import (
    catalog,
    cleanup_intermediates,
    summary,
)
from app.services.streaming_engine import engine

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _seed_project(db: AsyncSession, project_id: str = "p1") -> Project:
    project = Project(id=project_id, name="demo.avi", video_filename="demo.avi",
                      video_path=f"/tmp/{project_id}/demo.avi", status="uploaded")
    db.add(project)
    return project


async def _make_admin(db: AsyncSession, email: str = "admin@example.com") -> User:

    old = (settings.platform.bootstrap_admin_email, settings.platform.bootstrap_admin_password)
    settings.platform.bootstrap_admin_email = email
    settings.platform.bootstrap_admin_password = "AdminPass123!"
    try:
        await ensure_bootstrap_admin(db)
        await db.flush()
        user = (await db.execute(select(User).where(User.email == email))).scalar_one()
    finally:
        settings.platform.bootstrap_admin_email, settings.platform.bootstrap_admin_password = old
    return user


def _monkeypatch_storage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path))


# ---------------------------------------------------------------------------
# Additive migrations
# ---------------------------------------------------------------------------


class TestMigrations:
    async def test_migrations_apply_once_and_idempotent(self, db_session: AsyncSession):
        # Second run on the same connection must be a no-op (recorded versions).
        applied = await migration_module.run_migrations(await db_session.connection())
        applied_again = await migration_module.run_migrations(await db_session.connection())
        assert applied == applied_again or applied_again == []
        versions = (
            await db_session.execute(text("SELECT version FROM schema_migrations"))
        ).scalars().all()
        assert len(versions) == len(set(versions))

    async def test_platform_tables_exist(self, db_session: AsyncSession):
        for table in ("users", "auth_sessions", "missions", "mission_events", "queue_jobs"):
            assert table in Base.metadata.tables


# ---------------------------------------------------------------------------
# Auth + RBAC
# ---------------------------------------------------------------------------


class TestAuth:
    async def test_hash_verify_roundtrip(self):
        stored = hash_password("hunter2secure")
        assert stored.startswith("pbkdf2_sha256$")
        assert verify_password("hunter2secure", stored)
        assert not verify_password("wrong", stored)
        assert not verify_password("x", "not-a-hash")

    async def test_register_login_me_logout(self, client: AsyncClient):
        r = await client.post("/api/auth/register", json={
            "email": "viewer@example.com", "password": "Password123!"})
        assert r.status_code == 201, r.text

        r = await client.post("/api/auth/login", json={
            "email": "viewer@example.com", "password": "Password123!"})
        assert r.status_code == 200
        token = r.json()["token"]
        assert r.json()["user"]["role"] == "analyst"  # default self-register role

        headers = {"Authorization": f"Bearer {token}"}
        me = await client.get("/api/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["email"] == "viewer@example.com"

        # expired/unknown token rejected
        bad = await client.get("/api/auth/me", headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 401

        out = await client.post("/api/auth/logout", headers=headers)
        assert out.status_code == 200 and out.json()["revoked"] is True
        again = await client.get("/api/auth/me", headers=headers)
        assert again.status_code == 401

    async def test_elevated_role_requires_admin(self, client: AsyncClient):
        r = await client.post("/api/auth/register", json={
            "email": "sneaky@example.com", "password": "Password123!",
            "role": "operator"})
        assert r.status_code == 403

    async def test_admin_can_manage_users_and_rbac(self, client: AsyncClient, db_session: AsyncSession):
        await _make_admin(db_session, "boss@example.com")
        await db_session.flush()
        # login
        r = await client.post("/api/auth/login", json={
            "email": "boss@example.com", "password": "AdminPass123!"})
        assert r.status_code == 200
        admin_headers = {"Authorization": f"Bearer {r.json()['token']}"}

        # create operator as admin
        r = await client.post("/api/auth/users", headers=admin_headers, json={
            "email": "op@example.com", "password": "OperatorPass123!", "role": "operator"})
        assert r.status_code == 201
        # list users shows both
        users = await client.get("/api/auth/users", headers=admin_headers)
        assert users.status_code == 200
        assert users.json()["count"] >= 2

        # non-admin cannot list users
        await client.post("/api/auth/register", json={
            "email": "analyst@example.com", "password": "Password123!"})
        login = await client.post("/api/auth/login", json={
            "email": "analyst@example.com", "password": "Password123!"})
        anon_headers = {"Authorization": f"Bearer {login.json()['token']}"}
        denied = await client.get("/api/auth/users", headers=anon_headers)
        assert denied.status_code == 403

    async def test_auth_required_mode_blocks_legacy(self, client: AsyncClient,
                                                    db_session: AsyncSession, monkeypatch):
        monkeypatch.setattr(settings.platform, "auth_mode", "required")
        try:
            # no token → 401 before reaching handlers
            r = await client.get("/api/frame-extraction/nope")
            assert r.status_code == 401
            # health stays public
            h = await client.get("/api/health")
            assert h.status_code == 200
            # valid token passes
            await _make_admin(db_session, "req@example.com")
            await db_session.flush()
            login = await client.post("/api/auth/login", json={
                "email": "req@example.com", "password": "AdminPass123!"})
            assert login.status_code == 200
            token = login.json()["token"]
            r = await client.get("/api/frame-extraction/nope",
                                 headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 404  # reached the handler
        finally:
            monkeypatch.undo()


# ---------------------------------------------------------------------------
# Mission lifecycle + audit
# ---------------------------------------------------------------------------


class TestMissionLifecycle:
    async def test_lifecycle_transitions_and_audit(self, db_session: AsyncSession):
        owner = await _make_admin(db_session)
        _seed_project(db_session, "proj-lifecycle")
        mission = await mission_service.create_mission(
            db_session, owner, "coverage flight", project_id="proj-lifecycle", priority=3)
        assert mission.status == MISSION_CREATED

        # CREATED → QUEUED
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        assert mission.status == MISSION_QUEUED
        # QUEUED → PROCESSING
        await mission_service.transition(db_session, mission, MISSION_PROCESSING, owner)
        # PROCESSING → PAUSED → RESUMING → QUEUED → COMPLETED
        await mission_service.transition(db_session, mission, MISSION_PAUSED, owner)
        await mission_service.transition(db_session, mission, MISSION_RESUMING, owner)
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        await mission_service.transition(db_session, mission, MISSION_PROCESSING, owner)
        await mission_service.transition(db_session, mission, MISSION_COMPLETED, owner)
        assert mission.status == MISSION_COMPLETED

        events = await mission_service.mission_events(db_session, mission.id)
        kinds = [e["event"] for e in events]
        assert kinds.count("to_queued") >= 2 and "to_completed" in kinds
        # every event has an actor
        assert all(e["actor"] == "admin@example.com" for e in events)

    async def test_invalid_transition_rejected(self, db_session: AsyncSession):
        owner = await _make_admin(db_session)
        mission = await mission_service.create_mission(db_session, owner, "m", project_id=None)
        with pytest.raises(Exception) as exc:
            await mission_service.transition(db_session, mission, MISSION_COMPLETED, owner)
        assert exc.value.status_code == 409

    async def test_mission_api_flow(self, client: AsyncClient, db_session: AsyncSession):
        await _make_admin(db_session, "opmgr@example.com")
        await db_session.flush()
        login = await client.post("/api/auth/login", json={
            "email": "opmgr@example.com", "password": "AdminPass123!"})
        headers = {"Authorization": f"Bearer {login.json()['token']}"}

        # create requires an existing project
        bad = await client.post("/api/missions", headers=headers, json={
            "name": "x", "project_id": "missing"})
        assert bad.status_code == 404

        _seed_project(db_session, "proj-api")
        await db_session.flush()
        r = await client.post("/api/missions", headers=headers, json={
            "name": "recon-1", "project_id": "proj-api", "priority": 7})
        assert r.status_code == 201
        mid = r.json()["id"]

        # viewer/analyst cannot create or start a mission
        r = await client.post("/api/missions", headers=headers, json={
            "name": "blocked", "project_id": "proj-api"})
        assert r.status_code == 201  # admin can

        # start → queued job appears (job NOT executed here)
        r = await client.post(f"/api/missions/{mid}/start", headers=headers, json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == MISSION_QUEUED
        assert body["job_id"]

        # duplicate start → 409
        r2 = await client.post(f"/api/missions/{mid}/start", headers=headers, json={})
        assert r2.status_code == 409

        # cancel from queued
        r = await client.post(f"/api/missions/{mid}/cancel", headers=headers)
        assert r.status_code == 200
        detail = await client.get(f"/api/missions/{mid}", headers=headers)
        assert detail.json()["status"] == MISSION_CANCELLED

        # audit trail exposed
        events = await client.get(f"/api/missions/{mid}/events", headers=headers)
        assert events.status_code == 200
        assert len(events.json()["events"]) >= 3

        # unauth users cannot read missions
        noauth = await client.get("/api/missions")
        assert noauth.status_code == 401


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------


class _FakeExecutor:
    """Test-only executor that records calls and returns a scripted outcome."""

    def __init__(self, outcome: dict | None = None, delay: float = 0.0,
                 fail_once: bool = False):
        self.outcome = outcome or {"status": "completed"}
        self.delay = delay
        self.fail_once = fail_once
        self.calls: list[str] = []

    async def __call__(self, job, db):
        self.calls.append(job.id)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_once and len(self.calls) == 1:
            raise RuntimeError("boom (scripted)")
        # honour a real cancellation request like the orchestrator does
        if engine.is_cancelled(job.project_id or job.id):
            return {"status": "cancelled"}
        return self.outcome


@pytest.fixture
def fake_executor_factory(monkeypatch):
    store: dict[str, _FakeExecutor] = {}

    def _make(name: str, outcome: dict | None = None, **kw) -> _FakeExecutor:
        ex = _FakeExecutor(outcome, **kw)
        store[name] = ex
        job_queue.EXECUTORS[name] = ex
        return ex

    yield _make
    for name in store:
        job_queue.EXECUTORS.pop(name, None)


class TestQueue:
    async def test_priority_order_and_fifo(self, db_session: AsyncSession,
                                           fake_executor_factory):
        fake_executor_factory("echo", {"status": "completed"})
        # enqueue in reverse priority order; worker must pick highest first
        ids = []
        for priority in (1, 9, 5, 9):
            job = await job_queue.enqueue(
                db_session, project_id="proj-q", kind="echo", priority=priority)
            ids.append(job.id)
        await db_session.flush()

        processed_ids = []
        for _ in range(4):
            job = await job_queue.process_next(db_session)
            processed_ids.append(job.id)
        # highest priorities first, FIFO within equal priority
        assert processed_ids[0] == ids[1]  # priority 9 (first)
        assert processed_ids[1] == ids[3]  # priority 9 (second)
        assert processed_ids[2] == ids[2]  # priority 5
        assert processed_ids[3] == ids[0]  # priority 1

    async def test_retry_then_fail_and_mission_state(self, db_session: AsyncSession,
                                                     fake_executor_factory):
        owner = await _make_admin(db_session)
        _seed_project(db_session, "proj-retry")
        mission = await mission_service.create_mission(db_session, owner, "r",
                                                       project_id="proj-retry")
        # Route start flow: CREATED → QUEUED, then enqueue.
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        await db_session.flush()

        fake_executor_factory("echo", None, fail_once=True)
        await job_queue.enqueue(
            db_session, project_id="proj-retry", mission_id=mission.id,
            kind="echo", max_attempts=2)
        await db_session.flush()

        first = await job_queue.process_next(db_session)
        assert first.status == "queued"  # retried (attempt 1/2 failed)
        assert mission.status == MISSION_QUEUED  # returned to queue by the retry
        second = await job_queue.process_next(db_session)
        assert second.status == "completed"
        assert mission.status == MISSION_COMPLETED
        assert len(job_queue.EXECUTORS["echo"].calls) == 2

    async def test_final_failure_marks_mission_failed(self, db_session: AsyncSession,
                                                      fake_executor_factory):
        owner = await _make_admin(db_session)
        _seed_project(db_session, "proj-fail")
        mission = await mission_service.create_mission(db_session, owner, "f",
                                                       project_id="proj-fail")
        # Route start flow: CREATED → QUEUED, then enqueue.
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        await db_session.flush()
        fake_executor_factory("boom", None, fail_once=True)
        await job_queue.enqueue(db_session, project_id="proj-fail",
                                mission_id=mission.id, kind="boom", max_attempts=1)
        await db_session.flush()
        done = await job_queue.process_next(db_session)
        assert done.status == "failed"
        assert mission.status == "FAILED"
        assert "boom" in (done.error or "")

    async def test_cancel_queued_mission(self, db_session: AsyncSession,
                                         fake_executor_factory):
        owner = await _make_admin(db_session)
        _seed_project(db_session, "proj-cancel")
        mission = await mission_service.create_mission(db_session, owner, "c",
                                                       project_id="proj-cancel")
        fake_executor_factory("echo")
        # Route start flow: CREATED → QUEUED, then enqueue.
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        await job_queue.enqueue(db_session, project_id="proj-cancel",
                                mission_id=mission.id, kind="echo")
        await db_session.flush()
        # Route cancel flow: QUEUED → CANCELLED + cancel the job rows.
        await mission_service.transition(db_session, mission, MISSION_CANCELLED, owner)
        await job_queue.cancel_job(db_session, mission)
        await db_session.flush()
        # cancelled job must not be picked up
        picked = await job_queue.process_next(db_session)
        assert picked is None
        assert mission.status == MISSION_CANCELLED

    async def test_pause_and_resume(self, db_session: AsyncSession,
                                    fake_executor_factory):
        owner = await _make_admin(db_session)
        _seed_project(db_session, "proj-pr")
        mission = await mission_service.create_mission(db_session, owner, "pr",
                                                       project_id="proj-pr")
        await db_session.flush()
        ex = fake_executor_factory("echo")
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        job = await job_queue.enqueue(db_session, project_id="proj-pr",
                                      mission_id=mission.id, kind="echo")
        await db_session.flush()

        # Route pause flow: QUEUED → PROCESSING → PAUSED + pause the job rows.
        await mission_service.transition(db_session, mission, MISSION_PROCESSING, owner)
        await mission_service.transition(db_session, mission, MISSION_PAUSED, owner)
        await job_queue.pause_job(db_session, mission)
        await db_session.flush()
        assert job.status == "paused"

        # Route resume flow: PAUSED → RESUMING → QUEUED, then re-enqueue.
        await mission_service.transition(db_session, mission, MISSION_RESUMING, owner)
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        resumed = await job_queue.resume_job(db_session, mission)
        await db_session.flush()
        assert resumed.status == "queued"
        done = await job_queue.process_next(db_session)
        assert done.id == resumed.id
        assert done.status == "completed"
        assert mission.status == MISSION_COMPLETED
        # old paused job is completed (not executed twice)
        assert ex.calls == [resumed.id]

    async def test_crash_recovery_requeues_running_jobs(self, db_session: AsyncSession,
                                                        fake_executor_factory):
        owner = await _make_admin(db_session)
        _seed_project(db_session, "proj-crash")
        mission = await mission_service.create_mission(db_session, owner, "cr",
                                                       project_id="proj-crash")
        await db_session.flush()
        fake_executor_factory("echo")
        job = await job_queue.enqueue(db_session, project_id="proj-crash",
                                      mission_id=mission.id, kind="echo")
        await db_session.flush()
        job.status = "running"  # simulate a crash mid-run
        mission.status = "PROCESSING"
        await db_session.flush()

        recovered = await job_queue.recover_interrupted(db_session)
        assert recovered == 1
        assert job.status == "queued"
        assert mission.status == MISSION_QUEUED

    async def test_pipeline_executor_registered(self):
        job_queue.register_default_executors()
        assert "pipeline" in job_queue.EXECUTORS

    async def test_stale_cancelled_return_does_not_cancel_resumed_mission(
            self, db_session: AsyncSession, fake_executor_factory):
        """Pause → resume enqueues a newer job; the old executor's late
        ``cancelled`` return must not cancel the mission (regression)."""
        owner = await _make_admin(db_session)
        _seed_project(db_session, "proj-race")
        mission = await mission_service.create_mission(db_session, owner, "race",
                                                       project_id="proj-race")
        await db_session.flush()
        ex = fake_executor_factory("echo")
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        old = await job_queue.enqueue(db_session, project_id="proj-race",
                                      mission_id=mission.id, kind="echo")
        await db_session.flush()

        # Worker claims the old job (mission PROCESSING, row running), then the
        # operator pauses and immediately resumes before the executor returns.
        await mission_service.transition(db_session, mission, MISSION_PROCESSING, owner)
        old.status = "running"
        await db_session.flush()
        await mission_service.transition(db_session, mission, MISSION_PAUSED, owner)
        await job_queue.pause_job(db_session, mission)
        await mission_service.transition(db_session, mission, MISSION_RESUMING, owner)
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        fresh = await job_queue.resume_job(db_session, mission)  # closes `old`, queues fresh
        await db_session.flush()
        assert fresh.status == "queued"
        assert old.status == "completed"

        # The old executor finally returns "cancelled". process_next must treat
        # it as superseded — the mission must not move to CANCELLED.
        from app.services.job_queue import _settle_executor_result

        await _settle_executor_result(db_session, old, {"status": "cancelled"})
        await db_session.flush()
        assert mission.status == MISSION_QUEUED  # not CANCELLED
        assert old.status == "completed"

        # The resumed run then completes normally.
        done = await job_queue.process_next(db_session)
        assert done.id == fresh.id
        assert done.status == "completed"
        assert mission.status == MISSION_COMPLETED
        assert ex.calls == [fresh.id]

    async def test_cancel_running_job_then_late_completed_return(self,
                                                                 db_session: AsyncSession,
                                                                 fake_executor_factory):
        """A completed executor result arriving after the operator cancelled the
        mission must not resurrect a CANCELLED mission (regression)."""
        owner = await _make_admin(db_session)
        _seed_project(db_session, "proj-cnc")
        mission = await mission_service.create_mission(db_session, owner, "cnc",
                                                       project_id="proj-cnc")
        await db_session.flush()
        fake_executor_factory("echo")
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        job = await job_queue.enqueue(db_session, project_id="proj-cnc",
                                      mission_id=mission.id, kind="echo")
        await db_session.flush()

        # Worker claims, then the operator cancels while the executor is in flight.
        await mission_service.transition(db_session, mission, MISSION_PROCESSING, owner)
        job.status = "running"
        await db_session.flush()
        await mission_service.transition(db_session, mission, MISSION_CANCELLED, owner)
        await job_queue.cancel_job(db_session, mission)
        await db_session.flush()
        assert job.status == "cancelled"

        # Late executor result arrives — must not flip the mission back.
        from app.services.job_queue import _settle_executor_result

        await _settle_executor_result(db_session, job, {"status": "completed"})
        await db_session.flush()
        assert mission.status == MISSION_CANCELLED

    async def test_audit_trail_single_writer(self, db_session: AsyncSession,
                                             fake_executor_factory):
        """Every mission state change is recorded exactly once with a valid
        from/to pair — the queue must not append its own mission events."""
        owner = await _make_admin(db_session)
        _seed_project(db_session, "proj-audit")
        mission = await mission_service.create_mission(db_session, owner, "aud",
                                                       project_id="proj-audit")
        fake_executor_factory("echo")
        await db_session.flush()

        # Full route flow with queue transitions in between.
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        await job_queue.enqueue(db_session, project_id="proj-audit",
                                mission_id=mission.id, kind="echo")
        await mission_service.transition(db_session, mission, MISSION_PROCESSING, owner)
        await mission_service.transition(db_session, mission, MISSION_PAUSED, owner)
        await job_queue.pause_job(db_session, mission)
        await mission_service.transition(db_session, mission, MISSION_RESUMING, owner)
        await mission_service.transition(db_session, mission, MISSION_QUEUED, owner)
        await job_queue.resume_job(db_session, mission)
        await db_session.flush()
        await job_queue.process_next(db_session)  # claims + completes the resumed job
        await db_session.flush()

        events = await mission_service.mission_events(db_session, mission.id)
        events = list(reversed(events))  # chronological
        # Every non-creation event records a real from → to pair — no
        # queue-side duplicates with a null from_status.
        assert events[0]["event"] == "created"
        for e in events[1:]:
            assert e["from_status"] is not None, f"event lacks from_status: {e}"
            assert e["to_status"] is not None
        # chain is contiguous: each to_status equals the next from_status
        for prev, nxt in zip(events[:-1], events[1:]):
            if prev["to_status"] is not None and nxt["from_status"] is not None:
                assert prev["to_status"] == nxt["from_status"], \
                    f"broken chain: {prev} -> {nxt}"


# ---------------------------------------------------------------------------
# Storage manager
# ---------------------------------------------------------------------------


class TestStorage:
    def test_catalog_categories(self, tmp_path: Path, monkeypatch):
        _monkeypatch_storage(tmp_path, monkeypatch)
        ws = settings.storage.project_dir("catproj")
        (ws / "raw").mkdir(parents=True)
        (ws / "raw" / "lidar.las").write_bytes(b"\x00" * 10)
        (ws / "selected").mkdir()
        (ws / "selected" / "f0001.jpg").write_bytes(b"j")
        (ws / "depth").mkdir()
        (ws / "depth" / "f0001.npy").write_bytes(b"d" * 5)
        (ws / "dense").mkdir()
        (ws / "dense" / "dense_model.ply").write_bytes(b"p" * 7)

        entries = catalog("catproj")
        flat = {e.rel_path: e for e in _flatten(entries)}
        assert flat["raw/lidar.las"].category == "raw"
        assert flat["selected/f0001.jpg"].category == "processed"
        assert flat["depth/f0001.npy"].category == "intermediate"
        assert flat["dense/dense_model.ply"].category == "final"

        s = summary("catproj")
        assert s["categories"]["final"] >= 7

    def test_cleanup_only_removes_intermediates(self, tmp_path: Path, monkeypatch):
        _monkeypatch_storage(tmp_path, monkeypatch)
        ws = settings.storage.project_dir("cleanproj")
        (ws / "depth").mkdir()
        old = ws / "depth" / "old.npy"
        old.write_bytes(b"x" * 4)
        import os
        os.utime(old, (time.time() - 99 * 86400, time.time() - 99 * 86400))
        keep = ws / "dense"
        keep.mkdir()
        (keep / "model.ply").write_bytes(b"y" * 8)
        raw = ws / "raw.mp4"
        raw.write_bytes(b"z")

        report = cleanup_intermediates("cleanproj", retention_days=30, force=True)
        assert any("depth" in p for p in report["removed"])
        assert keep.exists() and raw.exists()  # finals + raw untouched

    def test_purge_workspace(self, tmp_path: Path, monkeypatch):
        _monkeypatch_storage(tmp_path, monkeypatch)
        ws = settings.storage.project_dir("purgeproj")  # creates the dir
        (ws / "f.txt").write_text("hi")
        from app.services.storage_manager import purge_workspace

        purge_workspace("purgeproj")
        assert not ws.exists()


def _flatten(entries):
    for e in entries:
        yield e
        yield from _flatten(e.children)


# ---------------------------------------------------------------------------
# Upload security
# ---------------------------------------------------------------------------


class TestUploadSecurity:
    async def test_traversal_filename_rejected(self, client: AsyncClient, tmp_path: Path,
                                               monkeypatch, synthetic_video: Path):
        _monkeypatch_storage(tmp_path, monkeypatch)
        with open(synthetic_video, "rb") as f:
            r = await client.post(
                "/api/upload",
                files={"file": ("../../evil.avi", f, "video/avi")},
            )
        # sanitized basename or 400 — never written outside the workspace.
        # (The workspace may hold more than the video itself — e.g. the
        # provenance sidecar — so assert containment, not an exact count.)
        assert r.status_code in (200, 201)
        ws = settings.storage.project_dir(r.json()["job_id"])
        stored = {p.name for p in ws.iterdir()}
        assert "evil.avi" in stored
        assert all(p.is_relative_to(ws) for p in ws.iterdir())
        assert not (ws / ".." / "evil.avi").exists()

    async def test_non_video_extension_rejected_before_write(self, client: AsyncClient,
                                                             tmp_path: Path, monkeypatch):
        _monkeypatch_storage(tmp_path, monkeypatch)
        r = await client.post("/api/upload", files={"file": ("pwn.sh", b"#!/bin/sh", "text/x-sh")})
        assert r.status_code == 400
        leftover = list((settings.storage.base_dir.parent).glob("pwn.sh"))
        assert not leftover


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class TestObservability:
    async def test_ready_and_metrics_and_capabilities(self, client: AsyncClient):
        ready = await client.get("/api/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] in ("ready", "not_ready")
        assert ready.json()["checks"]["database"] is True

        m = await client.get("/api/metrics")
        assert m.status_code == 200
        assert "drone_http_requests_total" in m.text  # middleware recorded this call

        caps = await client.get("/api/system/capabilities")
        assert caps.status_code == 200
        body = caps.json()
        assert body["compute"]["device"] in ("cuda", "cpu")
        assert isinstance(body["disk"]["free_gb"], (int, float))
        assert "tools" in body

    async def test_root_aliases(self, client: AsyncClient):
        assert (await client.get("/ready")).status_code == 200
        assert (await client.get("/metrics")).status_code == 200

    async def test_request_id_header(self, client: AsyncClient):
        r = await client.get("/api/health")
        assert r.headers.get("x-request-id")


# ---------------------------------------------------------------------------
# Mission deletion + report/artifacts surface
# ---------------------------------------------------------------------------


class TestMissionArtifactsSurface:
    async def test_artifacts_route_lists_workspace(self, client: AsyncClient,
                                                   db_session: AsyncSession,
                                                   tmp_path: Path, monkeypatch):
        _monkeypatch_storage(tmp_path, monkeypatch)
        await _make_admin(db_session, "art@example.com")
        await db_session.flush()
        _seed_project(db_session, "proj-art")
        await db_session.flush()
        ws = settings.storage.project_dir("proj-art")
        (ws / "dense").mkdir(parents=True)
        (ws / "dense" / "model.ply").write_bytes(b"p" * 3)

        login = await client.post("/api/auth/login", json={
            "email": "art@example.com", "password": "AdminPass123!"})
        headers = {"Authorization": f"Bearer {login.json()['token']}"}
        r = await client.post("/api/missions", headers=headers, json={
            "name": "art-mission", "project_id": "proj-art"})
        mid = r.json()["id"]

        arts = await client.get(f"/api/missions/{mid}/artifacts", headers=headers)
        assert arts.status_code == 200
        flat = [a["rel_path"] for a in arts.json()["artifacts"]]
        nested = [c["rel_path"] for a in arts.json()["artifacts"] for c in a.get("children", [])]
        assert any("model.ply" in p for p in flat + nested)

        # delete mission + workspace
        d = await client.delete(f"/api/missions/{mid}", headers=headers)
        assert d.status_code == 200
        assert not ws.exists()
        gone = await client.get(f"/api/missions/{mid}", headers=headers)
        assert gone.status_code == 404
