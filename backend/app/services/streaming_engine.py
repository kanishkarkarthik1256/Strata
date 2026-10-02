"""Streaming engine — in-process pub/sub for live reconstruction progress.

Pipeline stages running in worker threads publish snapshots through
:meth:`StreamingEngine.publish`; the REST stream endpoint subscribes with
:meth:`subscribe` and receives them in real time. Every job also keeps a
bounded event history so a client connecting after the run finished can
replay what happened.

Events are plain dicts with ``type``, ``job_id``, ``timestamp``, and
``payload`` keys. Terminal events use ``type="complete"`` (or
``type="error"``) so subscribers know the stream is done.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

from app.logging_config import get_logger

log = get_logger("drone_recon.services.streaming_engine")

_HISTORY_LIMIT = 2000


class StreamingEngine:
    """Thread-safe job-scoped event broker."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._history: dict[str, deque] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cancelled: dict[str, bool] = {}
        self._lock = threading.Lock()

    # -- cancellation (job-scoped flag, polled by the orchestrator) -----------

    def request_cancel(self, job_id: str) -> None:
        """Request cancellation of a running pipeline for *job_id*."""
        with self._lock:
            self._cancelled[job_id] = True
        self.publish(job_id, "cancel_requested", {})
        log.info("cancel_requested", job_id=job_id)

    def clear_cancel(self, job_id: str) -> None:
        with self._lock:
            self._cancelled.pop(job_id, None)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return bool(self._cancelled.get(job_id))

    # -- publishing (safe to call from worker threads) ----------------------

    def publish(self, job_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Record an event and deliver it to live subscribers.

        Called from inside ``asyncio.to_thread`` workers, so delivery is
        scheduled onto the event loop the engine captured on subscribe.
        """
        event = {
            "type": event_type,
            "job_id": job_id,
            "timestamp": time.time(),
            "payload": payload or {},
        }
        with self._lock:
            hist = self._history.setdefault(job_id, deque(maxlen=_HISTORY_LIMIT))
            hist.append(event)
            targets = list(self._subscribers.get(job_id, ()))

        loop = self._loop
        if loop is None or loop.is_closed() or not targets:
            return
        for queue in targets:
            try:
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            except RuntimeError:  # pragma: no cover - loop died mid-publish
                log.debug("stream_loop_closed", job_id=job_id)

    def replay(self, job_id: str) -> list[dict]:
        """Return the recorded history for a job (most useful post-run)."""
        with self._lock:
            return list(self._history.get(job_id, ()))

    # -- subscription (async context) ----------------------------------------

    def subscribe(self, job_id: str) -> asyncio.Queue:
        """Register a live subscriber queue for *job_id*."""
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            if self._loop is None:
                try:
                    self._loop = asyncio.get_running_loop()
                except RuntimeError:
                    self._loop = None
            self._subscribers.setdefault(job_id, set()).add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs:
                subs.discard(queue)
                if not subs:
                    self._subscribers.pop(job_id, None)

    async def iter_events(self, job_id: str, keep_alive: float = 15.0):
        """Yield historical then live events for *job_id*.

        Stops after a terminal (complete/error) event, or when the caller
        closes the connection.
        """
        for event in self.replay(job_id):
            yield event
            if event["type"] in ("complete", "error"):
                return

        queue = self.subscribe(job_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=keep_alive)
                except TimeoutError:
                    yield {
                        "type": "keepalive",
                        "job_id": job_id,
                        "timestamp": time.time(),
                        "payload": {},
                    }
                    continue
                yield event
                if event["type"] in ("complete", "error"):
                    return
        finally:
            self.unsubscribe(job_id, queue)


# Module-level singleton used by routes and the orchestrator.
engine = StreamingEngine()
