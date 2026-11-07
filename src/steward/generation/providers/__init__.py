"""Teacher-model providers and a small registry.

To add a provider, implement
:class:`~steward.generation.providers.base.TextGenerationProvider` and add
a factory to :data:`PROVIDERS`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from steward.generation.providers.base import (
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    MalformedOutputError,
    ModelInfo,
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    ProviderTimeoutError,
    TextGenerationProvider,
)
from steward.generation.providers.ollama import OllamaProvider

ProviderFactory = Callable[..., TextGenerationProvider]


def _transformers(**kwargs: Any) -> TextGenerationProvider:
    # Imported lazily: the module itself is light, but keep the registry import-cheap.
    from steward.inference.transformers_provider import TransformersProvider

    return TransformersProvider(**kwargs)


PROVIDERS: dict[str, ProviderFactory] = {
    "ollama": OllamaProvider,
    "transformers": _transformers,
}


def create_provider(name: str, **kwargs: Any) -> TextGenerationProvider:
    """Create a provider by registry name."""
    try:
        factory = PROVIDERS[name]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown provider {name!r}. Known providers: {known}.") from None
    return factory(**kwargs)


__all__ = [
    "PROVIDERS",
    "ChatMessage",
    "GenerationRequest",
    "GenerationResult",
    "MalformedOutputError",
    "ModelInfo",
    "ModelNotFoundError",
    "OllamaProvider",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderHTTPError",
    "ProviderTimeoutError",
    "TextGenerationProvider",
    "create_provider",
]
