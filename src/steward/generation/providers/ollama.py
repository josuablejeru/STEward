"""Ollama provider (local inference server, https://ollama.com).

Uses the ``/api/chat`` endpoint without streaming. When a response schema
is given, it is passed as Ollama's ``format`` option for structured output.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from steward.generation.providers.base import (
    GenerationRequest,
    GenerationResult,
    MalformedOutputError,
    ModelInfo,
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    ProviderTimeoutError,
)
from steward.logs import log_event

DEFAULT_BASE_URL = "http://localhost:11434"

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Teacher-model provider backed by a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout_s: float = 120.0,
        connect_timeout_s: float = 5.0,
        keep_alive: str | None = "10m",
        think: bool | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create a provider.

        Args:
            base_url: Ollama server URL.
            timeout_s: Read timeout for one generation call.
            connect_timeout_s: Timeout to open a connection.
            keep_alive: How long Ollama keeps the model loaded after a call.
            think: For reasoning models: ``False`` turns thinking off. ``None``
                sends nothing, which works with every model.
            transport: Optional httpx transport (tests use a mock transport).
        """
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.think = think
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            transport=transport,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> OllamaProvider:
        """Return self."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the HTTP client."""
        self.close()

    # -- public API -------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one chat completion."""
        options: dict[str, Any] = {
            "temperature": request.sampling.temperature,
            "top_p": request.sampling.top_p,
            "num_predict": request.sampling.max_tokens,
        }
        if request.sampling.top_k is not None:
            options["top_k"] = request.sampling.top_k
        if request.sampling.seed is not None:
            options["seed"] = request.sampling.seed
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "options": options,
        }
        if request.response_schema is not None:
            payload["format"] = request.response_schema
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if self.think is not None:
            payload["think"] = self.think

        started = time.perf_counter()
        data = self._post("/api/chat", payload, model=request.model)
        latency = time.perf_counter() - started

        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise MalformedOutputError("Ollama response has no 'message.content' string.")
        finish_reason = data.get("done_reason")
        if finish_reason == "length":
            raise MalformedOutputError(
                f"Output was cut off at max_tokens={request.sampling.max_tokens}.",
                hint="Increase max_tokens in the generation config.",
            )
        if not content.strip():
            raise MalformedOutputError(
                "Ollama returned empty content.",
                hint="Reasoning models may spend all tokens on thinking; try think = false.",
            )
        metadata = {
            key: data[key]
            for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
            if isinstance(data.get(key), int)
        }
        return GenerationResult(
            text=content,
            model=str(data.get("model", request.model)),
            latency_s=latency,
            prompt_tokens=_int_or_none(data.get("prompt_eval_count")),
            completion_tokens=_int_or_none(data.get("eval_count")),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            metadata=metadata,
        )

    def describe_model(self, model: str) -> ModelInfo:
        """Look up the model digest and details from ``/api/tags``."""
        tags = self._get("/api/tags")
        models = tags.get("models")
        if not isinstance(models, list):
            raise ProviderError("Ollama /api/tags response has no 'models' list.")
        wanted = {model, f"{model}:latest"}
        for entry in models:
            if not isinstance(entry, dict):
                continue
            if entry.get("name") in wanted or entry.get("model") in wanted:
                details = entry.get("details")
                return ModelInfo(
                    provider=self.name,
                    model=str(entry.get("name", model)),
                    digest=entry.get("digest") if isinstance(entry.get("digest"), str) else None,
                    provider_version=self._server_version(),
                    size_bytes=entry.get("size") if isinstance(entry.get("size"), int) else None,
                    details=details if isinstance(details, dict) else {},
                )
        available = sorted(str(e.get("name")) for e in models if isinstance(e, dict))
        raise ModelNotFoundError(
            f"Model {model!r} is not available on the Ollama server at {self.base_url}. "
            f"Available: {', '.join(available) or 'none'}.",
            hint=f"Run `ollama pull {model}`.",
        )

    def loaded_vram_mib(self, model: str) -> int | None:
        """Return the VRAM that Ollama reports for a loaded model (``/api/ps``), or None."""
        try:
            data = self._get("/api/ps")
        except ProviderError:
            return None
        for entry in data.get("models") or []:
            if isinstance(entry, dict) and model in {entry.get("name"), entry.get("model")}:
                vram = entry.get("size_vram")
                return vram // (1024 * 1024) if isinstance(vram, int) else None
        return None

    # -- HTTP helpers -----------------------------------------------------

    def _server_version(self) -> str | None:
        try:
            version = self._get("/api/version").get("version")
        except ProviderError:
            return None
        return version if isinstance(version, str) else None

    def _get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def _post(self, path: str, payload: dict[str, Any], *, model: str) -> dict[str, Any]:
        return self._request("POST", path, json=payload, model=model)

    def _request(
        self, method: str, path: str, *, json: Any = None, model: str | None = None
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama did not answer {method} {path} in time ({type(exc).__name__}).",
                hint="Increase --timeout, or use a smaller model.",
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to Ollama at {self.base_url}: {exc}.",
                hint="Start the server with `pixi run -e ollama ollama-serve` or "
                "`ollama serve`, or set --base-url.",
            ) from exc

        if response.status_code >= 400:
            detail = _error_detail(response)
            if response.status_code == 404 and model is not None and "not found" in detail:
                raise ModelNotFoundError(
                    f"Ollama does not have model {model!r}: {detail}",
                    hint=f"Run `ollama pull {model}`.",
                )
            log_event(
                logger, logging.WARNING, "ollama_http_error", status=response.status_code, path=path
            )
            raise ProviderHTTPError(
                f"Ollama {method} {path} failed with HTTP {response.status_code}: {detail}",
                status_code=response.status_code,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise MalformedOutputError(f"Ollama {path} returned a body that is not JSON.") from exc
        if not isinstance(data, dict):
            raise MalformedOutputError(f"Ollama {path} returned JSON that is not an object.")
        return data


def _error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return str(data["error"])
    return response.text[:200]


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None
