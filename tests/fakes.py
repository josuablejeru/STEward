"""A fake Ollama server for tests (httpx mock transport, no network)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httpx

from steward.generation.providers.ollama import OllamaProvider

ChatHandler = Callable[[httpx.Request], httpx.Response]


def chat_response(
    content: str, *, done_reason: str = "stop", model: str = "fake:1b"
) -> httpx.Response:
    """Return an Ollama /api/chat response body."""
    return httpx.Response(
        200,
        json={
            "model": model,
            "created_at": "2026-11-06T12:00:00Z",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": done_reason,
            "total_duration": 1_500_000_000,
            "prompt_eval_count": 120,
            "eval_count": 60,
            "eval_duration": 1_000_000_000,
        },
    )


class FakeOllama:
    """A scripted stand-in for an Ollama server.

    By default every source call returns a unique passage and every rewrite
    call returns a valid candidate that keeps the numbers. Push handlers onto
    ``chat_queue`` to inject failures for the next chat calls.
    """

    def __init__(self, model: str = "fake:1b") -> None:
        self.model = model
        self.chat_requests: list[dict[str, Any]] = []
        self.chat_queue: list[ChatHandler] = []
        self.source_calls = 0
        self.source_text: Callable[[int], str] = lambda n: (
            f"The {n + 2} bolts on access panel {n} should be removed while the panel is "
            f"being supported, and the washers must not be dropped. Torque is {10 + n} Nm."
        )
        self.models_available = True
        self.refuse_chat_after: int | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0-test"})
        if path == "/api/tags":
            models = (
                [
                    {
                        "name": self.model,
                        "model": self.model,
                        "digest": "d" * 64,
                        "details": {"family": "fake", "parameter_size": "1B"},
                    }
                ]
                if self.models_available
                else []
            )
            return httpx.Response(200, json={"models": models})
        if path == "/api/chat":
            body = json.loads(request.content)
            self.chat_requests.append(body)
            if (
                self.refuse_chat_after is not None
                and len(self.chat_requests) > self.refuse_chat_after
            ):
                raise httpx.ConnectError("Connection refused", request=request)
            if self.chat_queue:
                return self.chat_queue.pop(0)(request)
            system = body["messages"][0]["content"]
            user = body["messages"][1]["content"]
            if "technical editor" in system:
                return chat_response(json.dumps({"ste_text": self._rewrite(user)}))
            n = self.source_calls
            self.source_calls += 1
            return chat_response(json.dumps({"title": f"Panel {n}", "text": self.source_text(n)}))
        return httpx.Response(404, json={"error": "unknown path"})

    @staticmethod
    def _rewrite(user: str) -> str:
        match = re.search(r"<<<\n(.*)\n>>>", user, re.DOTALL)
        assert match, "rewrite prompt must contain the source between <<< and >>>"
        numbers = re.findall(r"\d+(?:[.,]\d+)*", match.group(1))
        return f"Hold the panel. Do not drop the washers. Use these values: {' '.join(numbers)}."

    def provider(self) -> OllamaProvider:
        return OllamaProvider(
            base_url="http://ollama.test", transport=httpx.MockTransport(self.handler)
        )
