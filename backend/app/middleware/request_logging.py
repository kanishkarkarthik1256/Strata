"""Request logging middleware.

Logs every incoming request / response pair with structured logging, tags
each request with a request ID (honouring an inbound ``X-Request-ID`` or
generating one), and records HTTP counters into the metrics registry.
Health-check requests are logged at DEBUG level to reduce noise.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import get_logger
from app.services.metrics import metrics

log = get_logger("drone_recon.middleware.request")

# Paths that are high-frequency and low-value for logging
_NOISY_PATHS = {"/api/health", "/api/ready", "/api/metrics"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs structured information about each HTTP request / response pair."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        metrics.inc(
            "drone_http_requests_total",
            {"method": request.method, "path": request.url.path,
             "status": str(response.status_code)},
        )

        log_method = log.debug if request.url.path in _NOISY_PATHS else log.info

        log_method(
            "http_request",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else None,
            status_code=response.status_code,
            request_id=request_id,
            client=request.client.host if request.client else None,
        )

        return response
