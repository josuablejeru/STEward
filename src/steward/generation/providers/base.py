"""Provider-neutral interface for teacher models.

A provider turns a :class:`GenerationRequest` into a
:class:`GenerationResult`. Add a new backend by implementing
:class:`TextGenerationProvider` and registering it in
:mod:`steward.generation.providers`.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from steward.datasets.schema import SamplingParameters


class ChatMessage(BaseModel):
    """One chat message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class GenerationRequest(BaseModel):
    """A single provider call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    messages: list[ChatMessage]
    sampling: SamplingParameters
    response_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema the output must follow, if supported."
    )


class GenerationResult(BaseModel):
    """Text and statistics returned by a provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Identity of a model as reported by its provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    digest: str | None = None
    provider_version: str | None = None
    size_bytes: int | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ProviderError(Exception):
    """Base class for provider failures.

    ``retryable`` tells the caller whether trying again can help.
    """

    retryable: bool = False

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        """Create an error with an optional hint that tells the user what to do."""
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        """Return the message, plus the hint when present."""
        base = super().__str__()
        return f"{base} Hint: {self.hint}" if self.hint else base


class ProviderConnectionError(ProviderError):
    """The provider could not be reached."""

    retryable = True


class ProviderTimeoutError(ProviderError):
    """The provider did not answer in time."""

    retryable = True


class ProviderHTTPError(ProviderError):
    """The provider answered with an HTTP error status."""

    def __init__(self, message: str, *, status_code: int, hint: str | None = None) -> None:
        """Create an HTTP error. 408, 429, and 5xx statuses are retryable."""
        super().__init__(message, hint=hint)
        self.status_code = status_code
        self.retryable = status_code in {408, 429} or status_code >= 500


class ModelNotFoundError(ProviderError):
    """The requested model is not available on the provider."""


class MalformedOutputError(ProviderError):
    """The provider answered, but the output is unusable (bad JSON, truncated, empty)."""

    retryable = True


@runtime_checkable
class TextGenerationProvider(Protocol):
    """The interface every teacher-model backend implements."""

    name: str

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one generation request. Raise :class:`ProviderError` on failure."""
        ...

    def describe_model(self, model: str) -> ModelInfo:
        """Return model identity. Raise :class:`ModelNotFoundError` if absent."""
        ...

    def close(self) -> None:
        """Release network resources."""
        ...
