"""Resolve a ``--model`` argument to a base model and an optional LoRA adapter.

Accepted forms:

* a training run directory (``runs/<id>``, has ``training_summary.json``),
* a LoRA adapter directory (has ``adapter_config.json``),
* a local full-model directory, or
* a Hugging Face model ID such as ``Qwen/Qwen3-0.6B``, or
* ``latest``: the newest completed training run under ``runs/``, by the UTC
  timestamp at the end of its run ID (``...-20261107T011144Z``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from steward.evaluation.records import RecordStatus, TrainingSummary

TRAINING_SUMMARY = "training_summary.json"
LATEST = "latest"
RUNS_DIR = Path("runs")
_RUN_TIMESTAMP = re.compile(r"(\d{8}T\d{6}Z)$")


class ModelRefError(Exception):
    """The model reference cannot be used."""


@dataclass(frozen=True)
class ModelRef:
    """What to load for inference."""

    model: str
    revision: str | None = None
    adapter_path: Path | None = None
    run_dir: Path | None = None


def _project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pixi.toml").exists():
            return candidate
    return Path.cwd()


def _summary(run_dir: Path) -> TrainingSummary:
    try:
        return TrainingSummary.model_validate_json(
            (run_dir / TRAINING_SUMMARY).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ModelRefError(f"Invalid {run_dir / TRAINING_SUMMARY}: {exc}") from exc


def latest_run(runs_dir: Path = RUNS_DIR) -> Path:
    """The newest completed run in ``runs_dir`` that has an adapter."""
    candidates: list[tuple[str, Path]] = []
    for summary_file in runs_dir.glob(f"*/{TRAINING_SUMMARY}"):
        stamp = _RUN_TIMESTAMP.search(summary_file.parent.name)
        if stamp is None:
            continue
        try:
            summary = _summary(summary_file.parent)
        except ModelRefError:
            continue
        if (
            summary.status is RecordStatus.COMPLETED
            and summary.serving
            and summary.serving.adapter_path
        ):
            candidates.append((stamp.group(1), summary_file.parent))
    if not candidates:
        raise ModelRefError(
            f"No completed training run in {runs_dir}/. Train one with `pixi run -e cuda train`."
        )
    return max(candidates)[1]


def resolve_model(ref: str, revision: str | None = None, *, runs_dir: Path = RUNS_DIR) -> ModelRef:
    """Resolve ``ref`` (see the module docstring)."""
    path = latest_run(runs_dir) if ref == LATEST else Path(ref)
    if not path.is_dir():
        if path.exists() or ref.startswith((".", "/", "runs")):
            raise ModelRefError(f"{ref} is not a directory.")
        return ModelRef(model=ref, revision=revision)
    path = path.resolve()
    if (path / TRAINING_SUMMARY).is_file():
        summary = _summary(path)
        if summary.serving is None or summary.serving.adapter_path is None:
            raise ModelRefError(
                f"{path}: the training run has no adapter (status {summary.status})."
            )
        adapter = Path(summary.serving.adapter_path)
        if not adapter.is_absolute():
            adapter = _project_root(path) / adapter
        return ModelRef(
            model=summary.serving.model,
            revision=revision or summary.serving.revision or summary.base_revision,
            adapter_path=adapter,
            run_dir=path,
        )
    if (path / "adapter_config.json").is_file():
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        base = config.get("base_model_name_or_path")
        run_dir = path.parent if (path.parent / TRAINING_SUMMARY).is_file() else None
        adapter_summary = _summary(run_dir) if run_dir else None
        if not base:
            raise ModelRefError(f"{path}/adapter_config.json has no base_model_name_or_path.")
        return ModelRef(
            model=str(base),
            revision=revision or (adapter_summary.base_revision if adapter_summary else None),
            adapter_path=path,
            run_dir=run_dir,
        )
    return ModelRef(model=str(path), revision=revision)
