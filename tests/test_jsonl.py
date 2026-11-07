from pathlib import Path

import pytest
from pydantic import BaseModel

from steward.datasets.jsonl import (
    DatasetFileError,
    DuplicateIndex,
    atomic_write_text,
    fingerprint,
    normalize_for_dedup,
    read_jsonl,
    write_jsonl_atomic,
)


class Item(BaseModel):
    name: str
    value: int


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "items.jsonl"
    items = [Item(name="a", value=1), Item(name="b", value=2)]
    assert write_jsonl_atomic(path, items) == 2
    assert read_jsonl(path, Item) == items
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    atomic_write_text(path, "x\n")
    assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]


def test_failed_write_keeps_original_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "out.jsonl"
    atomic_write_text(path, "original\n")

    def broken_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("steward.datasets.jsonl.os.replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(path, "new\n")
    assert path.read_text(encoding="utf-8") == "original\n"
    assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]


def test_read_reports_line_number_on_bad_record(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"name": "a", "value": 1}\n\n{"name": "b"}\n', encoding="utf-8")
    with pytest.raises(DatasetFileError, match=r"bad\.jsonl:3"):
        read_jsonl(path, Item)


def test_normalization_ignores_case_punctuation_and_spacing() -> None:
    assert normalize_for_dedup("Remove  the COVER!") == normalize_for_dedup("remove the cover")
    assert fingerprint("Torque to 10 Nm.") != fingerprint("Torque to 12 Nm.")


def test_duplicate_index() -> None:
    index = DuplicateIndex()
    assert index.find("Remove the cover.") is None
    index.add("stew-1", "Remove the cover.", "Remove cover.")
    assert index.find("REMOVE the cover") == "stew-1"
    assert index.find("other", "remove cover") == "stew-1"
    assert index.find("Install the cover.") is None
    assert len(index) == 2
