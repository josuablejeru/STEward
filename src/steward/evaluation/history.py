"""Append-only experiment history (``benchmarks/results/history.jsonl``).

Integrity rules:

* Each record stores ``record_sha256``: the hash of its content without the
  ID, hash fields, and timestamp. Identical content is never stored twice.
* Each record stores ``prev_record_sha256``: the hash of the record before
  it. A changed, removed, or reordered line breaks the chain, and
  :func:`verify_history` reports it.
* :func:`append_record` only adds a line. It rewrites the file atomically
  after it checks that the existing file is intact.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from steward.datasets.jsonl import atomic_write_text
from steward.evaluation.fingerprints import canonical_sha256
from steward.evaluation.records import BenchmarkRecord, HistoryRecord, TrainingRecord

_ADAPTER: TypeAdapter[BenchmarkRecord | TrainingRecord] = TypeAdapter(HistoryRecord)
_HASH_EXCLUDE = {"experiment_id", "record_sha256", "prev_record_sha256", "created_at"}


class HistoryError(Exception):
    """The history file is invalid or its integrity check failed."""


def content_hash(data: dict[str, Any]) -> str:
    """Hash of a record's content (excluding ID, hash fields, and timestamp)."""
    return canonical_sha256({k: v for k, v in data.items() if k not in _HASH_EXCLUDE})


def finalize_record(
    record_type: type[BenchmarkRecord] | type[TrainingRecord],
    data: dict[str, Any],
    created_at: datetime,
) -> BenchmarkRecord | TrainingRecord:
    """Build a record with its content hash and ID (the chain link is set on append)."""
    body = {k: v for k, v in data.items() if k not in _HASH_EXCLUDE}
    probe = record_type.model_validate(
        {**body, "experiment_id": "x", "record_sha256": "0" * 64, "created_at": created_at}
    )
    digest = content_hash(probe.model_dump(mode="json"))
    prefix = "bench" if record_type is BenchmarkRecord else "train"
    return probe.model_copy(
        update={"experiment_id": f"{prefix}-{digest[:16]}", "record_sha256": digest}
    )


def read_history(path: Path) -> list[BenchmarkRecord | TrainingRecord]:
    """Read and validate all records. An absent file is an empty history."""
    if not path.exists():
        return []
    records: list[BenchmarkRecord | TrainingRecord] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(_ADAPTER.validate_json(line))
        except ValidationError as exc:
            raise HistoryError(
                f"{path}:{line_no}: invalid history record ({exc.error_count()} error(s)): "
                f"{exc.errors()[0]['msg']}"
            ) from exc
    return records


def verify_history(records: list[BenchmarkRecord | TrainingRecord]) -> list[str]:
    """Return integrity problems (empty when the chain is intact)."""
    problems: list[str] = []
    previous: str | None = None
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        expected = content_hash(record.model_dump(mode="json"))
        if record.record_sha256 != expected:
            problems.append(
                f"record {index} ({record.experiment_id}): content was changed after it was "
                "written (hash mismatch)."
            )
        if not record.experiment_id.endswith(record.record_sha256[:16]):
            problems.append(f"record {index}: experiment_id does not match its hash.")
        if record.prev_record_sha256 != previous:
            problems.append(
                f"record {index} ({record.experiment_id}): chain broken. A record before it "
                "was removed, changed, or reordered."
            )
        if record.record_sha256 in seen:
            problems.append(f"record {index}: duplicate of an earlier record.")
        seen.add(record.record_sha256)
        previous = record.record_sha256
    return problems


@contextmanager
def history_lock(path: Path) -> Iterator[None]:
    """Exclusive lock for writers of ``path`` (blocks until free)."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def append_record(
    path: Path, record: BenchmarkRecord | TrainingRecord
) -> tuple[BenchmarkRecord | TrainingRecord, bool]:
    """Append ``record``. Return ``(stored_record, appended)``.

    ``appended`` is False when a record with the same content already exists.

    Raises:
        HistoryError: The existing history is invalid; nothing is written.
    """
    with history_lock(path):
        records = read_history(path)
        problems = verify_history(records)
        if problems:
            raise HistoryError(
                "Refusing to append: history integrity check failed:\n  " + "\n  ".join(problems)
            )
        for existing in records:
            if existing.record_sha256 == record.record_sha256:
                return existing, False
        linked = record.model_copy(
            update={"prev_record_sha256": records[-1].record_sha256 if records else None}
        )
        existing_text = path.read_text(encoding="utf-8") if path.exists() else ""
        if existing_text and not existing_text.endswith("\n"):
            existing_text += "\n"
        atomic_write_text(path, existing_text + linked.model_dump_json() + "\n")
        return linked, True


def benchmarks(records: list[BenchmarkRecord | TrainingRecord]) -> list[BenchmarkRecord]:
    """Return only benchmark records."""
    return [r for r in records if isinstance(r, BenchmarkRecord)]


def trainings(records: list[BenchmarkRecord | TrainingRecord]) -> list[TrainingRecord]:
    """Return only training records."""
    return [r for r in records if isinstance(r, TrainingRecord)]
