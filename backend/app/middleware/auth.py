"""Global authentication enforcement (Phase 10).

When ``platform.auth_mode == \"required\"`` every /api route except the public
allowlist requires a valid bearer session; otherwise the request is rejected
with 401 *before* reaching any handler. The legacy phase endpoints therefore
remain reachable without a token only while auth is disabled (the documented
backward-compatible default for local/CI); production deployments enable
``required``.

Session lookup uses the app's dependency override when one is installed
(standard in tests) and the real DB session otherwise, so enforcement cannot
be bypassed by calling a route directly.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config.settings import settings
from app.logging_config import get_logger

log = get_logger("drone_recon.middleware.auth")

_PUBLIC_PREFIXES = ("/api/docs", "/api/redoc", "/api/openapi.json",
                    "/api/auth/login", "/api/auth/register")


class AuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid bearer token when auth is required."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if settings.platform.auth_mode != "required":
            return await call_next(request)

        path = request.url.path
        if path in settings.platform.public_paths or path.startswith(_PUBLIC_PREFIXES):
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Not authenticated", "status_code": 401, "detail": None},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Resolve the session through the app's dependency override when one
        # is installed (tests) — never trust a route-level check to exist.
        from app.db.engine import get_session as _engine_session
        from app.services.auth_service import resolve_token

        db_dep = request.app.dependency_overrides.get(_engine_session, _engine_session)
        token = auth[7:].strip()
        async for db in db_dep():
            user = await resolve_token(db, token)
            if user is None:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or expired token", "status_code": 401,
                             "detail": None},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            request.state.user = user
        return await call_next(request)
