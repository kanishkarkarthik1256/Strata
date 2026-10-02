"""Structured logging configuration for DroneRecon.

Uses ``structlog`` to produce machine-readable JSON logs in production
and human-friendly coloured output during development.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "info", json_output: bool = True) -> None:
    """Configure structlog and the stdlib logging system.

    Parameters
    ----------
    log_level:
        Minimum log level as a string (``debug``, ``info``, ``warning``, ``error``, ``critical``).
    json_output:
        When *True* each log line is a single JSON object.  When *False* a
        human-friendly coloured renderer is used (good for local development).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: structlog.types.Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "uvicorn.error", "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog-bound logger.

    Usage::

        log = get_logger("drone_recon.pipeline")
        log.info("stage_started", stage="frame_extraction", project_id="abc")
    """
    return structlog.get_logger(name)
