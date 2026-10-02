"""Custom exception hierarchy for DroneRecon.

All application-level errors inherit from ``DroneReconError`` so that
the error handlers in ``error_handlers.py`` can translate them into
consistent HTTP responses.
"""

from __future__ import annotations


class DroneReconError(Exception):
    """Base exception for all DroneRecon errors."""

    def __init__(self, message: str = "An unexpected error occurred", status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# 4xx errors
# ---------------------------------------------------------------------------


class BadRequestError(DroneReconError):
    """Generic client-side error (400)."""

    def __init__(self, message: str = "Bad request") -> None:
        super().__init__(message=message, status_code=400)


class ProjectNotFoundError(DroneReconError):
    """Requested project does not exist (404)."""

    def __init__(self, project_id: str) -> None:
        super().__init__(message=f"Project not found: {project_id}", status_code=404)


class NotFoundError(DroneReconError):
    """Requested resource does not exist (404)."""

    def __init__(self, message: str = "Not found") -> None:
        super().__init__(message=message, status_code=404)


class InvalidVideoError(BadRequestError):
    """Uploaded video is invalid or unsupported (400)."""

    def __init__(self, detail: str = "The uploaded file is not a supported video format") -> None:
        super().__init__(message=detail)


class FileTooLargeError(BadRequestError):
    """Uploaded file exceeds size limit (413)."""

    def __init__(self, max_mb: int) -> None:
        super().__init__(message=f"File exceeds maximum upload size of {max_mb} MB")


# ---------------------------------------------------------------------------
# 5xx errors (processing / infrastructure)
# ---------------------------------------------------------------------------


class ProcessingError(DroneReconError):
    """A pipeline stage failed during processing (500)."""

    def __init__(self, stage: str, detail: str = "") -> None:
        msg = f"Processing failed at stage '{stage}'"
        if detail:
            msg += f": {detail}"
        super().__init__(message=msg, status_code=500)
        self.stage = stage


class ModelNotAvailableError(DroneReconError):
    """An AI model could not be loaded (503)."""

    def __init__(self, model_name: str, detail: str = "") -> None:
        msg = f"AI model '{model_name}' is not available"
        if detail:
            msg += f": {detail}"
        super().__init__(message=msg, status_code=503)


class COLMAPError(ProcessingError):
    """COLMAP binary failed or is missing (500)."""

    def __init__(self, detail: str = "COLMAP operation failed") -> None:
        super().__init__(stage="colmap", detail=detail)


class DiskSpaceError(DroneReconError):
    """Insufficient disk space for processing (507)."""

    def __init__(self) -> None:
        super().__init__(message="Insufficient disk space", status_code=507)
