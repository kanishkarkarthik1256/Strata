"""FastAPI exception handlers.

Maps the custom exception hierarchy defined in ``exceptions.py`` into
JSON HTTP responses with consistent structure:

```json
{
  "error": "Project not found: abc123",
  "status_code": 404,
  "detail": null
}
```
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions import DroneReconError
from app.logging_config import get_logger

log = get_logger("drone_recon.errors")


def register_error_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the *app*."""

    @app.exception_handler(DroneReconError)
    async def _handle_drone_recon_error(
        request: Request, exc: DroneReconError
    ) -> JSONResponse:
        log.warning(
            "drone_recon_error",
            error=exc.message,
            status_code=exc.status_code,
            path=str(request.url.path),
            method=request.method,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "status_code": exc.status_code,
                "detail": getattr(exc, "detail", None),
            },
        )

    @app.exception_handler(404)
    async def _handle_not_found(request: Request, exc: Exception) -> JSONResponse:
        log.warning("not_found", path=str(request.url.path), method=request.method)
        # A deliberate HTTPException carries the useful message (e.g. which
        # video/run was not found); a no-route 404 arrives as a plain
        # StarletteHTTPException with a generic default detail.
        detail = getattr(exc, "detail", None)
        detail = detail if isinstance(detail, str) and detail != "Not Found" else None
        return JSONResponse(
            status_code=404,
            content={
                "error": detail or "The requested resource was not found",
                "status_code": 404,
                "detail": detail,
            },
        )

    @app.exception_handler(422)
    async def _handle_validation_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log.warning(
            "validation_error",
            path=str(request.url.path),
            method=request.method,
            detail=str(exc),
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "status_code": 422,
                "detail": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unhandled(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log.exception(
            "unhandled_exception",
            path=str(request.url.path),
            method=request.method,
            exc_type=type(exc).__name__,
            exc_msg=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "detail": None,
            },
        )
