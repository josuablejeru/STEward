"""JSONL serialization with atomic writes and duplicate detection."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ValidationError


class DatasetFileError(Exception):
    """Raised when a dataset file cannot be read or parsed."""


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so readers never see a partial file.

    The data goes to a temporary file in the same directory, is flushed to
    disk, and then replaces ``path`` in one ``os.replace`` call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_jsonl_atomic(path: Path, records: Iterable[BaseModel]) -> int:
    """Atomically write records as JSON Lines. Return the number of records."""
    lines = [record.model_dump_json() for record in records]
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))
    return len(lines)


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    """Read and validate every line of a JSONL file against ``model``."""
    records: list[M] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as exc:
                raise DatasetFileError(
                    f"{path}:{line_no}: record does not match {model.__name__} schema: "
                    f"{exc.error_count()} error(s), first: {exc.errors()[0]['msg']}"
                ) from exc
    return records


_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    """Normalize text so trivial differences do not hide duplicates.

    Applies Unicode NFKC, lower case, punctuation removal, and whitespace
    collapsing. Numbers are kept, so texts that differ only in a value are
    *not* duplicates.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def fingerprint(text: str) -> str:
    """Return the SHA-256 hex digest of the normalized text."""
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()


class DuplicateIndex:
    """Detect exact duplicates after normalization.

    This finds texts that are identical after :func:`normalize_for_dedup`.
    It does not detect paraphrases or other near-duplicates.
    """

    def __init__(self) -> None:
        """Create an empty index."""
        self._seen: dict[str, str] = {}

    def __len__(self) -> int:
        """Return the number of fingerprints in the index."""
        return len(self._seen)

    def find(self, *texts: str) -> str | None:
        """Return the owner ID of the first known text, or ``None``."""
        for text in texts:
            owner = self._seen.get(fingerprint(text))
            if owner is not None:
                return owner
        return None

    def add(self, owner_id: str, *texts: str) -> None:
        """Register ``texts`` as belonging to ``owner_id``."""
        for text in texts:
            self._seen.setdefault(fingerprint(text), owner_id)
