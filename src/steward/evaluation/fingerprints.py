"""Content fingerprints for code, datasets, and evaluator versions.

Fingerprints use file content, so they are the same on every machine for
the same files. A small cache keyed by size and mtime avoids hashing large
dataset files again when they did not change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

# Code that can change scores. Plotting, reports, and the CLI are not included.
EVALUATOR_CODE_PATHS = (
    "src/steward/evaluation/metrics.py",
    "src/steward/evaluation/runner.py",
    "src/steward/evaluation/suite.py",
    "src/steward/ste",
    "src/steward/generation/validation.py",
    "configs/ste",
)
WORKTREE_PATHS = ("src/steward", "configs", "prompts", "benchmarks/golden", "pixi.lock")
_SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of one file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path, relative: Iterable[str], suffixes: tuple[str, ...] = ()) -> list[Path]:
    """Return files under ``relative`` paths (sorted, cache folders skipped)."""
    files: list[Path] = []
    for rel in relative:
        base = root / rel
        candidates = [base] if base.is_file() else sorted(base.rglob("*")) if base.is_dir() else []
        for path in candidates:
            if not path.is_file() or _SKIP_PARTS & set(path.parts) or path.suffix == ".pyc":
                continue
            if suffixes and not path.name.endswith(suffixes):
                continue
            files.append(path)
    return sorted(set(files))


def tree_fingerprint(
    root: Path,
    relative: Iterable[str],
    *,
    suffixes: tuple[str, ...] = (),
    cache: dict[str, list[object]] | None = None,
) -> str:
    """Return one hash over the paths and contents of all matching files.

    Args:
        root: Project root. Paths in the hash are relative to it.
        relative: Files or directories to include.
        suffixes: Only include files ending with one of these.
        cache: Optional ``{path: [size, mtime_ns, sha]}`` map, updated in place.
    """
    digest = hashlib.sha256()
    for path in iter_files(root, relative, suffixes):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        cached = cache.get(rel) if cache is not None else None
        if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
            sha = str(cached[2])
        else:
            sha = file_sha256(path)
            if cache is not None:
                cache[rel] = [stat.st_size, stat.st_mtime_ns, sha]
        digest.update(f"{rel}\0{sha}\n".encode())
    return digest.hexdigest()


def evaluator_fingerprint(root: Path) -> str:
    """Fingerprint of the scoring code and rule configuration."""
    return tree_fingerprint(root, EVALUATOR_CODE_PATHS, suffixes=(".py", ".toml"))


def worktree_fingerprint(root: Path) -> str:
    """Fingerprint of the code and configuration that affect results."""
    return tree_fingerprint(root, WORKTREE_PATHS)


def dataset_fingerprint(root: Path, cache: dict[str, list[object]] | None = None) -> str | None:
    """Fingerprint of accepted dataset files under ``data/``, or None when there are none."""
    files = [
        p for p in iter_files(root, ["data"], (".jsonl",)) if not p.name.endswith(".rejected.jsonl")
    ]
    if not files:
        return None
    rels = [p.relative_to(root).as_posix() for p in files]
    return tree_fingerprint(root, rels, cache=cache)


def canonical_sha256(data: object) -> str:
    """SHA-256 of canonical JSON (sorted keys, no spaces)."""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
