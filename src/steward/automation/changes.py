"""Classify changed files and detect changes made by shell commands."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path, PurePosixPath


class ChangeCategory(StrEnum):
    """What kind of benchmark input a file belongs to."""

    DATA = "data"
    GENERATION = "generation"
    TRAINING = "training"
    MODEL = "model"
    EVALUATION = "evaluation"


# Outputs of the benchmark system itself. Writing them never triggers a hook,
# which prevents recursive activation.
IGNORED_PREFIXES = (
    ".git/",
    ".pixi/",
    ".steward/",
    "benchmarks/results/",
    "benchmarks/figures/",
    "docs/benchmarks.md",
)

# Ordered: the first matching prefix wins.
RULES: tuple[tuple[str, ChangeCategory], ...] = (
    ("data/", ChangeCategory.DATA),
    ("src/steward/datasets/", ChangeCategory.DATA),
    ("src/steward/generation/", ChangeCategory.GENERATION),
    ("prompts/benchmark_", ChangeCategory.EVALUATION),
    ("prompts/", ChangeCategory.GENERATION),
    ("configs/generation/", ChangeCategory.GENERATION),
    ("src/steward/training/", ChangeCategory.TRAINING),
    ("configs/training/", ChangeCategory.TRAINING),
    ("configs/sft/", ChangeCategory.TRAINING),
    ("src/steward/inference/", ChangeCategory.EVALUATION),
    ("src/steward/ste/", ChangeCategory.EVALUATION),
    ("src/steward/evaluation/", ChangeCategory.EVALUATION),
    ("configs/ste/", ChangeCategory.EVALUATION),
    ("configs/evaluation/", ChangeCategory.EVALUATION),
    ("benchmarks/golden/", ChangeCategory.EVALUATION),
    ("benchmarks/suites/", ChangeCategory.EVALUATION),
    ("benchmarks/thresholds.toml", ChangeCategory.EVALUATION),
    ("benchmarks/models.toml", ChangeCategory.MODEL),
    ("configs/models/", ChangeCategory.MODEL),
    ("runs/", ChangeCategory.MODEL),
    ("checkpoints/", ChangeCategory.MODEL),
    ("models/", ChangeCategory.MODEL),
    ("configs/", ChangeCategory.GENERATION),
)

WATCH_ROOTS = tuple(
    sorted(
        {prefix.rstrip("/") for prefix, _ in RULES if prefix.endswith("/")}
        | {p for p, _ in RULES if p.endswith(".toml")}
    )
)
_SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"}


def relativize(path: str, root: Path) -> str | None:
    """Return ``path`` relative to ``root`` (POSIX form), or None when outside it."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        rel = candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        return None
    return PurePosixPath(rel).as_posix()


def classify(rel_path: str) -> ChangeCategory | None:
    """Return the category of a project-relative path, or None when irrelevant."""
    parts = set(PurePosixPath(rel_path).parts)
    if parts & _SKIP_PARTS or rel_path.endswith(".pyc"):
        return None
    if any(rel_path == p.rstrip("/") or rel_path.startswith(p) for p in IGNORED_PREFIXES):
        return None
    for prefix, category in RULES:
        if rel_path == prefix.rstrip("/") or rel_path.startswith(prefix):
            return category
    return None


def category_paths(category: ChangeCategory) -> list[str]:
    """The watched prefixes for one category (used for fingerprints)."""
    return [prefix.rstrip("/") for prefix, cat in RULES if cat is category]


Snapshot = dict[str, list[int]]


def take_snapshot(root: Path) -> Snapshot:
    """Return ``{path: [size, mtime_ns]}`` for every watched file (stat only, no reads)."""
    snapshot: Snapshot = {}
    for watched in WATCH_ROOTS:
        base = root / watched
        files = [base] if base.is_file() else base.rglob("*") if base.is_dir() else []
        for path in files:
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if classify(rel) is None:
                continue
            stat = path.stat()
            snapshot[rel] = [stat.st_size, stat.st_mtime_ns]
    return snapshot


def diff_snapshots(before: Snapshot, after: Snapshot) -> list[str]:
    """Paths added, removed, or changed between two snapshots (sorted)."""
    changed = {p for p in after if before.get(p) != after[p]}
    changed |= {p for p in before if p not in after}
    return sorted(changed)
