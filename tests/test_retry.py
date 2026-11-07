import pytest

from steward.generation.providers.base import (
    MalformedOutputError,
    ModelNotFoundError,
    ProviderHTTPError,
)
from steward.generation.retry import RetriesExhaustedError, RetryPolicy, call_with_retry


def test_succeeds_after_retryable_errors() -> None:
    sleeps: list[float] = []
    calls: list[int] = []

    def op(attempt: int) -> str:
        calls.append(attempt)
        if attempt < 3:
            raise MalformedOutputError("bad json")
        return "ok"

    result, attempts = call_with_retry(
        op,
        RetryPolicy(max_attempts=3, initial_delay_s=1, multiplier=2),
        label="t",
        sleep=sleeps.append,
    )
    assert (result, attempts) == ("ok", 3)
    assert calls == [1, 2, 3]
    assert sleeps == [1.0, 2.0]


def test_non_retryable_error_is_raised_at_once() -> None:
    calls: list[int] = []

    def op(attempt: int) -> str:
        calls.append(attempt)
        raise ModelNotFoundError("missing")

    with pytest.raises(ModelNotFoundError):
        call_with_retry(op, RetryPolicy(max_attempts=5), label="t", sleep=lambda _: None)
    assert calls == [1]


def test_exhausted_retries_keep_last_error() -> None:
    def op(attempt: int) -> str:
        raise ProviderHTTPError(f"boom {attempt}", status_code=503)

    with pytest.raises(RetriesExhaustedError) as exc:
        call_with_retry(op, RetryPolicy(max_attempts=2), label="t", sleep=lambda _: None)
    assert exc.value.attempts == 2
    assert "boom 2" in str(exc.value.last_error)


def test_delay_is_capped() -> None:
    policy = RetryPolicy(initial_delay_s=10, max_delay_s=15, multiplier=3)
    assert [policy.delay(n) for n in (1, 2, 3)] == [10.0, 15.0, 15.0]


@pytest.mark.parametrize(
    ("status", "retryable"), [(400, False), (408, True), (429, True), (500, True)]
)
def test_http_status_retryability(status: int, retryable: bool) -> None:
    assert ProviderHTTPError("x", status_code=status).retryable is retryable


def test_invalid_policy() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
