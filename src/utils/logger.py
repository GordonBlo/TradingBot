"""Central logging configuration with credential redaction."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable


class _RedactingFormatter(logging.Formatter):
    def __init__(self, *args: object, sensitive_values: Iterable[str], **kwargs: object):
        super().__init__(*args, **kwargs)
        self._sensitive_values = tuple(
            value for value in sensitive_values if value and len(value) >= 4
        )

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        for sensitive_value in self._sensitive_values:
            rendered = rendered.replace(sensitive_value, "***REDACTED***")
        return rendered


def configure_logging(
    level: int = logging.INFO, sensitive_values: Iterable[str] = ()
) -> None:
    """Configure one readable console logger for the entire application."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _RedactingFormatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            sensitive_values=sensitive_values,
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    logging.captureWarnings(True)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that uses the centralized configuration."""

    return logging.getLogger(name)

