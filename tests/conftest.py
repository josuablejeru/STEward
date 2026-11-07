"""Shared fixtures. Nothing here needs a network connection or a real model."""

from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FakeOllama
from steward.generation.config import GenerationConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = REPO_ROOT / "prompts"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "generation" / "default.toml"


@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def config() -> GenerationConfig:
    return load_config(DEFAULT_CONFIG)


@pytest.fixture
def prompt_dir() -> Path:
    return PROMPT_DIR
