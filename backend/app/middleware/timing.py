"""Request timing middleware.

Adds an ``X-Process-Time`` header to every response so that clients can
observe server-side latency without external monitoring tools.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import get_logger

log = get_logger("drone_recon.middleware.timing")


class TimingMiddleware(BaseHTTPMiddleware):
    """Measures wall-clock time for each request and attaches it as a header."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}"

        # Skip timing logs for health checks (they are noisy)
        if request.url.path != "/api/health":
            log.debug(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                elapsed_ms=round(elapsed_ms, 2),
            )

        return response
