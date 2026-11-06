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
