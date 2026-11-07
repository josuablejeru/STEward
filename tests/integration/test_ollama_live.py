"""Opt-in tests against a real local Ollama server.

Not part of ``pixi run test``. Run with::

    STEWARD_OLLAMA_MODEL=qwen3:8b pixi run test-ollama

``OLLAMA_HOST`` selects the server (default http://localhost:11434). Tests
skip when the server or the model is not available.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from steward.cli import default_ollama_url
from steward.datasets.jsonl import read_jsonl
from steward.datasets.schema import DatasetExample
from steward.generation.config import load_config
from steward.generation.pipeline import DatasetGenerator, RunStatus
from steward.generation.providers.ollama import OllamaProvider

pytestmark = pytest.mark.ollama

REPO = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("STEWARD_OLLAMA_MODEL", "")


@pytest.fixture(scope="module")
def provider() -> OllamaProvider:
    if not MODEL:
        pytest.skip("Set STEWARD_OLLAMA_MODEL to run live Ollama tests.")
    url = default_ollama_url()
    try:
        httpx.get(f"{url}/api/version", timeout=3).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"No Ollama server at {url}: {exc}")
    return OllamaProvider(url, timeout_s=300)


def test_model_is_available(provider: OllamaProvider) -> None:
    info = provider.describe_model(MODEL)
    assert info.digest


def test_generates_two_valid_examples(provider: OllamaProvider, tmp_path: Path) -> None:
    output = tmp_path / "live.jsonl"
    generator = DatasetGenerator(
        provider=provider,
        model=MODEL,
        config=load_config(REPO / "configs/generation/default.toml"),
        prompt_dir=REPO / "prompts",
        output=output,
    )
    record = generator.run(2, max_scenarios=6)
    assert record.status is RunStatus.COMPLETED, record.error
    examples = read_jsonl(output, DatasetExample)
    assert len(examples) == 2
    assert all(e.review_state.value == "candidate" for e in examples)
