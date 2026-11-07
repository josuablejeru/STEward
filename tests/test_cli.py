import json
import subprocess
import sys

import pytest

import steward
from steward.cli import DISCLAIMER, main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"steward {steward.__version__}"


def test_no_command_prints_help_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage: steward" in capsys.readouterr().out


def test_unknown_command_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["does-not-exist"])
    assert exc.value.code == 2


def test_info_text_includes_disclaimer(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert f"steward_version: {steward.__version__}" in out
    assert DISCLAIMER in out


def test_info_json_is_valid(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["steward_version"] == steward.__version__
    assert data["ste_reference"] == steward.STE_REFERENCE


def test_python_dash_m_entry_point() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "steward", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == f"steward {steward.__version__}"


# -- generate ---------------------------------------------------------------

from pathlib import Path  # noqa: E402

from fakes import FakeOllama  # noqa: E402
from steward.generation import providers  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _patch_provider(monkeypatch: pytest.MonkeyPatch, fake: FakeOllama) -> list[dict[str, object]]:
    seen: list[dict[str, object]] = []

    def factory(**kwargs: object) -> object:
        seen.append(kwargs)
        return fake.provider()

    monkeypatch.setitem(providers.PROVIDERS, "ollama", factory)
    return seen


def _generate_args(output: Path, *extra: str) -> list[str]:
    return [
        "generate",
        "--model",
        "fake:1b",
        "--count",
        "3",
        "--output",
        str(output),
        "--config",
        str(REPO / "configs/generation/default.toml"),
        "--prompt-dir",
        str(REPO / "prompts"),
        *extra,
    ]


def test_generate_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _patch_provider(monkeypatch, FakeOllama())
    output = tmp_path / "data" / "examples.jsonl"
    code = main(_generate_args(output, "--timeout", "42", "--think", "off", "--log-format", "json"))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Accepted:    3 this run, 3 total" in captured.out
    assert DISCLAIMER in captured.out
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3
    assert (tmp_path / "data" / "examples.manifest.json").exists()
    assert seen[0]["timeout_s"] == 42.0
    assert seen[0]["think"] is False
    events = [
        json.loads(line)["event"] for line in captured.err.splitlines() if line.startswith("{")
    ]
    assert events.count("example_accepted") == 3
    assert "run_finished" in events


def test_generate_refuses_existing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_provider(monkeypatch, FakeOllama())
    output = tmp_path / "examples.jsonl"
    assert main(_generate_args(output)) == 0
    assert main(_generate_args(output)) == 1
    assert "--resume" in capsys.readouterr().err
    assert main(_generate_args(output, "--resume")) == 0


def test_generate_reports_unreachable_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeOllama()
    fake.refuse_chat_after = 0
    _patch_provider(monkeypatch, fake)
    code = main(_generate_args(tmp_path / "examples.jsonl", "--max-attempts", "1"))
    err = capsys.readouterr().err
    assert code == 1
    assert "Generation stopped" in err
    assert "--resume" in err


def test_generate_reports_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _generate_args(tmp_path / "x.jsonl")
    args[args.index("--config") + 1] = str(tmp_path / "missing.toml")
    assert main(args) == 1
    assert "Config file not found" in capsys.readouterr().err


def test_generate_requires_model() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["generate"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (None, "http://localhost:11434"),
        ("0.0.0.0:9999", "http://0.0.0.0:9999"),
        ("https://h", "https://h"),
    ],
)
def test_default_ollama_url(
    monkeypatch: pytest.MonkeyPatch, env: str | None, expected: str
) -> None:
    from steward.cli import default_ollama_url

    if env is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", env)
    assert default_ollama_url() == expected
