"""Structured logging configuration."""

from __future__ import annotations

from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the research system."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            int(structlog.get_level_from_name(log_level))
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Get a named logger instance."""
    return structlog.get_logger(name)
