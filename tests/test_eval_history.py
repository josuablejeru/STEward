import json
from pathlib import Path

import pytest

from bench_helpers import later, make_record
from steward.evaluation.history import (
    HistoryError,
    append_record,
    read_history,
    verify_history,
)
from steward.evaluation.records import RecordStatus


def test_append_links_records_and_dedupes(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    a, appended_a = append_record(path, make_record("a"))
    b, appended_b = append_record(path, make_record("b", created=later(5)))
    assert appended_a and appended_b
    assert a.prev_record_sha256 is None and b.prev_record_sha256 == a.record_sha256
    again, appended = append_record(path, make_record("a", created=later(9)))
    assert not appended and again.experiment_id == a.experiment_id
    records = read_history(path)
    assert len(records) == 2 and verify_history(records) == []


def test_failed_runs_are_stored(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    stored, _ = append_record(path, make_record("a", status=RecordStatus.FAILED))
    assert read_history(path)[0].status is RecordStatus.FAILED
    assert stored.error


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_edit_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    append_record(path, make_record("a"))
    data = json.loads(_lines(path)[0])
    data["metrics"]["ste_rule_pass_rate"]["value"] = 0.99
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    problems = verify_history(read_history(path))
    assert any("changed after it was written" in p for p in problems)
    with pytest.raises(HistoryError, match="Refusing to append"):
        append_record(path, make_record("b"))


def test_deletion_and_reorder_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    for i, name in enumerate("abc"):
        append_record(path, make_record(name, created=later(i)))
    lines = _lines(path)
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    assert any("chain broken" in p for p in verify_history(read_history(path)))
    path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n", encoding="utf-8")
    assert verify_history(read_history(path))


def test_invalid_line_reports_location(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text('{"record_type": "benchmark"}\n', encoding="utf-8")
    with pytest.raises(HistoryError, match=r"history\.jsonl:1"):
        read_history(path)


def test_missing_history_is_empty(tmp_path: Path) -> None:
    assert read_history(tmp_path / "none.jsonl") == []
