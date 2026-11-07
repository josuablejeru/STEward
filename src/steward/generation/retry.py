"""Retry with exponential backoff for provider calls."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from steward.generation.providers.base import ProviderError
from steward.logs import log_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times to try, and how long to wait between tries."""

    max_attempts: int = 3
    initial_delay_s: float = 1.0
    max_delay_s: float = 30.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        """Check the values."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay_s < 0 or self.max_delay_s < 0:
            raise ValueError("delays must not be negative")

    def delay(self, attempt: int) -> float:
        """Return the wait after failed ``attempt`` (1-based). No jitter, for reproducibility."""
        return float(min(self.max_delay_s, self.initial_delay_s * self.multiplier ** (attempt - 1)))


class RetriesExhaustedError(ProviderError):
    """All attempts failed. ``last_error`` is the final provider error."""

    def __init__(self, last_error: ProviderError, attempts: int) -> None:
        """Wrap the last error."""
        super().__init__(f"Failed after {attempts} attempt(s): {last_error}")
        self.last_error = last_error
        self.attempts = attempts


def call_with_retry[T](
    operation: Callable[[int], T],
    policy: RetryPolicy,
    *,
    label: str,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    """Call ``operation(attempt)`` until it succeeds.

    Retries only :class:`ProviderError` instances with ``retryable=True``.
    Other errors are raised at once.

    Returns:
        The result and the number of attempts used.

    Raises:
        RetriesExhaustedError: If every attempt failed with a retryable error.
    """
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation(attempt), attempt
        except ProviderError as exc:
            if not exc.retryable:
                raise
            if attempt == policy.max_attempts:
                raise RetriesExhaustedError(exc, attempt) from exc
            wait = policy.delay(attempt)
            log_event(
                logger,
                logging.WARNING,
                "retrying",
                label=label,
                attempt=attempt,
                error_type=type(exc).__name__,
                error=str(exc),
                wait_s=wait,
            )
            sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover
