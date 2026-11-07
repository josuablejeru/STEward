"""Structured logging helpers.

Log calls use an event name plus key/value fields::

    log_event(logger, logging.INFO, "example_accepted", example_id="stew-...")

Logs never contain secrets. Generated text is not logged; records are
written to the dataset files instead.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Literal, TextIO

LogFormat = Literal["text", "json"]


class JsonFormatter(logging.Formatter):
    """Format each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the JSON representation of ``record``."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class KeyValueFormatter(logging.Formatter):
    """Format records as ``time LEVEL event key=value ...`` lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the text representation of ``record``."""
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        parts = [f"{ts} {record.levelname:<7} {record.getMessage()}"]
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            parts.extend(f"{key}={_short(value)}" for key, value in fields.items())
        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _short(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return json.dumps(text) if " " in text else text


class _StderrHandler(logging.Handler):
    """Write to the *current* ``sys.stderr`` (it can be replaced, e.g. by tests)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def configure_logging(
    level: str = "INFO", fmt: LogFormat = "text", stream: TextIO | None = None
) -> None:
    """Configure the ``steward`` logger hierarchy. Safe to call more than once."""
    handler: logging.Handler = (
        logging.StreamHandler(stream) if stream is not None else _StderrHandler()
    )
    handler.setFormatter(JsonFormatter() if fmt == "json" else KeyValueFormatter())
    logger = logging.getLogger("steward")
    logger.handlers[:] = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Log ``event`` with structured ``fields``."""
    logger.log(level, event, extra={"fields": fields})
