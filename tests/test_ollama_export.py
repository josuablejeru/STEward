"""Tests for the Ollama Modelfile export (no GPU, no Ollama server)."""

from __future__ import annotations

from pathlib import Path

import pytest

from steward.generation.prompts import load_template
from steward.inference.api import DEFAULT_PROMPT
from steward.inference.ollama_export import (
    GGUF_FILE,
    OllamaExportError,
    convert_to_gguf,
    render_modelfile,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modelfile() -> str:
    return render_modelfile(load_template(ROOT / "prompts", DEFAULT_PROMPT))


def test_modelfile_points_at_gguf_weights(modelfile: str) -> None:
    assert f"FROM ./{GGUF_FILE}\n" in modelfile


def test_modelfile_embeds_exact_training_system_prompt(modelfile: str) -> None:
    system = load_template(ROOT / "prompts", DEFAULT_PROMPT).render(source_text="").system
    assert f"<|im_start|>system\n{system}<|im_end|>\n" in modelfile


def test_modelfile_disables_thinking_like_training(modelfile: str) -> None:
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in modelfile


def test_modelfile_uses_greedy_default_sampling(modelfile: str) -> None:
    assert "PARAMETER temperature 0.0" in modelfile
    assert "PARAMETER seed 1234" in modelfile
    assert 'PARAMETER stop "<|im_end|>"' in modelfile


def test_modelfile_states_non_certification(modelfile: str) -> None:
    assert "not certified or approved ASD-STE100 compliance" in modelfile


def test_convert_without_converter_explains_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("steward.inference.ollama_export.shutil.which", lambda _: None)
    with pytest.raises(OllamaExportError, match="pixi run -e cuda export-ollama"):
        convert_to_gguf(tmp_path, tmp_path / GGUF_FILE)


def test_convert_rejects_unknown_type(tmp_path: Path) -> None:
    with pytest.raises(OllamaExportError, match="Unsupported GGUF type"):
        convert_to_gguf(tmp_path, tmp_path / GGUF_FILE, "q2_k")
