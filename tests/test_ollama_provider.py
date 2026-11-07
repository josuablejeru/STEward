import json

import httpx
import pytest

from fakes import FakeOllama, chat_response
from steward.datasets.schema import SamplingParameters
from steward.generation.providers import create_provider
from steward.generation.providers.base import (
    ChatMessage,
    GenerationRequest,
    MalformedOutputError,
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderTimeoutError,
    TextGenerationProvider,
)
from steward.generation.providers.ollama import OllamaProvider

REQUEST = GenerationRequest(
    model="fake:1b",
    messages=[ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u")],
    sampling=SamplingParameters(temperature=0.3, top_p=0.9, top_k=40, seed=7, max_tokens=99),
    response_schema={"type": "object"},
)


def _provider(handler: object, **kwargs: object) -> OllamaProvider:
    return OllamaProvider(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def test_satisfies_protocol(fake_ollama: FakeOllama) -> None:
    assert isinstance(fake_ollama.provider(), TextGenerationProvider)


def test_request_payload_and_result(fake_ollama: FakeOllama) -> None:
    fake_ollama.chat_queue.append(lambda _: chat_response('{"a": 1}'))
    with fake_ollama.provider() as provider:
        result = provider.generate(REQUEST)
    body = fake_ollama.chat_requests[0]
    assert body["model"] == "fake:1b"
    assert body["stream"] is False
    assert body["format"] == {"type": "object"}
    assert body["options"] == {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "seed": 7,
        "num_predict": 99,
    }
    assert "think" not in body
    assert result.text == '{"a": 1}'
    assert (result.prompt_tokens, result.completion_tokens) == (120, 60)
    assert result.finish_reason == "stop"
    assert result.metadata["total_duration"] == 1_500_000_000


def test_think_flag_is_sent_when_set() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_response("{}")

    _provider(handler, think=False).generate(REQUEST)
    assert seen[0]["think"] is False


def test_connection_error_has_hint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    with pytest.raises(ProviderConnectionError) as exc:
        _provider(handler).generate(REQUEST)
    assert exc.value.retryable
    assert "ollama serve" in str(exc.value)


def test_timeout_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderTimeoutError) as exc:
        _provider(handler).generate(REQUEST)
    assert exc.value.retryable


def test_missing_model_on_chat() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'fake:1b' not found"})

    with pytest.raises(ModelNotFoundError, match="ollama pull fake:1b"):
        _provider(handler).generate(REQUEST)


def test_server_error_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "out of memory"})

    with pytest.raises(ProviderHTTPError, match="out of memory") as exc:
        _provider(handler).generate(REQUEST)
    assert exc.value.retryable


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (chat_response("{}", done_reason="length"), "cut off"),
        (chat_response("   "), "empty"),
        (httpx.Response(200, text="not json"), "not JSON"),
        (httpx.Response(200, json={"done": True}), "message.content"),
    ],
)
def test_malformed_responses(response: httpx.Response, match: str) -> None:
    with pytest.raises(MalformedOutputError, match=match):
        _provider(lambda _: response).generate(REQUEST)


def test_describe_model(fake_ollama: FakeOllama) -> None:
    info = fake_ollama.provider().describe_model("fake:1b")
    assert info.digest == "d" * 64
    assert info.provider_version == "0.0.0-test"
    assert info.details["family"] == "fake"


def test_describe_model_accepts_implicit_latest_tag() -> None:
    fake = FakeOllama(model="tiny:latest")
    assert fake.provider().describe_model("tiny").model == "tiny:latest"


def test_describe_missing_model_lists_available(fake_ollama: FakeOllama) -> None:
    with pytest.raises(ModelNotFoundError, match="Available: fake:1b"):
        fake_ollama.provider().describe_model("other:7b")


def test_registry() -> None:
    provider = create_provider("ollama", base_url="http://x")
    assert provider.name == "ollama"
    provider.close()
    with pytest.raises(ValueError, match="Known providers: ollama"):
        create_provider("nope")
