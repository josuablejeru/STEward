"""Small, controlled hyperparameter sweeps (``steward train-sweep``, ``steward sweep-select``).

A sweep file (``configs/training/sweeps/*.toml``) names one base training config and a
list of trials. Each trial overrides only the listed values, so every other setting,
the dataset, and the train/validation split stay the same across trials:

.. code-block:: toml

    sweep_id = "m4-v0.2"
    base_config = "configs/training/qwen3-0.6b-lora-v0.2.toml"
    selection_suite = "ste-fidelity-dev-v0"

    [[trial]]
    name = "A"
    train = { learning_rate = 2e-4, epochs = 2 }
    lora = { r = 16 }

Trials run one after another. A failed trial is recorded and the next one starts.
The state is written to ``runs/sweeps/<sweep_id>/sweep.json`` after every trial.

Selection never uses training loss, and never uses the final golden suite. The rank
order is: critical failures on the *development* suite, then technical-error rate,
then meaning preservation, then the best validation loss.
"""

from __future__ import annotations

import gc
import json
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from steward.datasets.jsonl import atomic_write_text
from steward.training.config import TrainingConfig, TrainingConfigError, load_training_config


class TrialSpec(BaseModel):
    """One trial: a name and overrides of the base config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    train: dict[str, Any] = Field(default_factory=dict)
    lora: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class SweepConfig(BaseModel):
    """A sweep file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sweep_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    base_config: Path
    selection_suite: str
    description: str = ""
    trial: list[TrialSpec] = Field(min_length=1)


class TrialResult(BaseModel):
    """What happened to one trial."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal["pending", "completed", "failed"] = "pending"
    run_dir: str | None = None
    config_sha256: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    best_val_loss: float | None = None
    best_step: int | None = None
    global_steps: int | None = None
    duration_s: float | None = None
    error: str | None = None
    dev_experiment_id: str | None = None
    dev_metrics: dict[str, float | None] = Field(default_factory=dict)
    rank: int | None = None


class SweepState(BaseModel):
    """``runs/sweeps/<sweep_id>/sweep.json``."""

    model_config = ConfigDict(extra="forbid")

    sweep_id: str
    sweep_file: str
    base_config: str
    selection_suite: str
    trials: list[TrialResult]
    selected: str | None = None
    selection_rule: str = (
        "lowest dev-suite critical failures, then lowest technical-error rate, then highest "
        "meaning preservation, then lowest best validation loss; never training loss, never "
        "the final golden suite"
    )
    updated_at: str = ""


def load_sweep(path: Path) -> SweepConfig:
    """Load and validate a sweep file."""
    try:
        return SweepConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise TrainingConfigError(f"Sweep file not found: {path}") from None
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise TrainingConfigError(f"Invalid sweep file {path}: {exc}") from exc


def trial_config(base: TrainingConfig, sweep: SweepConfig, trial: TrialSpec) -> TrainingConfig:
    """The base config with the trial's overrides (validated again)."""
    data = base.model_dump(mode="json")
    data["train"] = {**data["train"], **trial.train}
    data["lora"] = {**data["lora"], **trial.lora}
    data["run_name"] = f"{base.run_name}-{sweep.sweep_id}-{trial.name}".lower()
    data["version_label"] = f"{base.version_label or base.run_name}-{trial.name}"
    try:
        return TrainingConfig.model_validate(data)
    except ValidationError as exc:
        raise TrainingConfigError(f"Trial {trial.name}: invalid overrides: {exc}") from exc


def state_path(root: Path, sweep_id: str) -> Path:
    """Where the sweep state is written."""
    return root / "runs" / "sweeps" / sweep_id / "sweep.json"


def load_state(root: Path, sweep: SweepConfig, sweep_file: Path) -> SweepState:
    """The saved state, or a new one."""
    path = state_path(root, sweep.sweep_id)
    if path.is_file():
        return SweepState.model_validate_json(path.read_text(encoding="utf-8"))
    return SweepState(
        sweep_id=sweep.sweep_id,
        sweep_file=str(sweep_file),
        base_config=str(sweep.base_config),
        selection_suite=sweep.selection_suite,
        trials=[
            TrialResult(name=t.name, overrides={"train": t.train, "lora": t.lora})
            for t in sweep.trial
        ],
    )


def save_state(root: Path, state: SweepState) -> Path:
    """Write the state atomically."""
    state.updated_at = datetime.now(UTC).isoformat()
    path = state_path(root, state.sweep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, state.model_dump_json(indent=2) + "\n")
    return path


def run_sweep(
    sweep: SweepConfig,
    sweep_file: Path,
    *,
    root: Path,
    train: Callable[[TrainingConfig], Path],
    only: list[str] | None = None,
) -> SweepState:
    """Run the trials that are not completed yet (``only``: a subset by name)."""
    base = load_training_config(
        sweep.base_config if sweep.base_config.is_absolute() else root / sweep.base_config
    )
    state = load_state(root, sweep, sweep_file)
    by_name = {t.name: t for t in state.trials}
    for spec in sweep.trial:
        if only and spec.name not in only:
            continue
        result = by_name.setdefault(spec.name, TrialResult(name=spec.name))
        if result.status == "completed":
            continue
        result.overrides = {"train": spec.train, "lora": spec.lora}
        try:
            config = trial_config(base, sweep, spec)
            result.config_sha256 = config.sha256()
            run_dir = train(config)
        except Exception as exc:  # a failed trial must not stop the sweep
            result.status, result.error = "failed", f"{type(exc).__name__}: {exc}"
            save_state(root, state)
            gc.collect()
            continue
        summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
        hp = summary.get("hyperparameters", {})
        result.status = "completed" if summary.get("status") == "completed" else "failed"
        result.error = summary.get("error")
        result.run_dir = str(run_dir.relative_to(root) if run_dir.is_relative_to(root) else run_dir)
        result.best_val_loss = hp.get("best_val_loss")
        result.best_step = hp.get("best_step")
        result.global_steps = hp.get("global_steps")
        result.duration_s = hp.get("duration_s")
        save_state(root, state)
        gc.collect()
    state.trials = [by_name[t.name] for t in sweep.trial if t.name in by_name]
    save_state(root, state)
    return state


RANK_METRICS = ("critical_failures", "technical_error_rate", "meaning_preservation")


def rank_key(result: TrialResult) -> tuple[float, float, float, float]:
    """Sort key: safety first, then fidelity, then validation loss (lower is better)."""
    m = result.dev_metrics

    def val(name: str, default: float) -> float:
        v = m.get(name)
        return default if v is None else v

    return (
        val("critical_failures", float("inf")),
        val("technical_error_rate", float("inf")),
        -val("meaning_preservation", float("-inf")),
        result.best_val_loss if result.best_val_loss is not None else float("inf"),
    )


def select(state: SweepState) -> str | None:
    """Rank completed trials with dev metrics; return the selected trial name."""
    ranked = sorted(
        (t for t in state.trials if t.status == "completed" and t.dev_metrics),
        key=rank_key,
    )
    for t in state.trials:
        t.rank = None
    for i, t in enumerate(ranked, 1):
        t.rank = i
    state.selected = ranked[0].name if ranked else None
    return state.selected
