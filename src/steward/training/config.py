"""Training configuration (``configs/training/*.toml``)."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class TrainingConfigError(Exception):
    """The training config is missing or invalid."""


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoraSettings(_Config):
    """PEFT LoRA settings."""

    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(min_length=1)


class TrainSettings(_Config):
    """Optimizer and schedule (TRL ``SFTConfig``)."""

    epochs: float = Field(default=3, gt=0)
    max_steps: int = Field(default=-1, description="-1: use epochs.")
    learning_rate: float = Field(default=2e-4, gt=0)
    lr_scheduler: str = "cosine"
    warmup_ratio: float = Field(default=0.05, ge=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    per_device_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_length: int = Field(default=1024, ge=64)
    logging_steps: int = Field(default=5, ge=1)
    eval_steps: int = Field(default=20, ge=1)
    save_steps: int = Field(default=40, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    gradient_checkpointing: bool = False
    select_best: bool = False
    """Keep the checkpoint with the best validation metric as the adapter (not the last).
    Needs ``save_steps`` to be a multiple of ``eval_steps``."""
    metric_for_best: str = "eval_loss"
    early_stopping_patience: int | None = Field(default=None, ge=1)
    """Stop after this many evaluations without improvement (needs ``select_best``)."""
    early_stopping_threshold: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _selection(self) -> TrainSettings:
        if self.select_best and self.save_steps % self.eval_steps:
            raise ValueError("select_best needs save_steps to be a multiple of eval_steps.")
        if self.early_stopping_patience is not None and not self.select_best:
            raise ValueError("early_stopping_patience needs select_best = true.")
        return self


class TrainingConfig(_Config):
    """One LoRA SFT experiment."""

    run_name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    base_model: str
    base_revision: str | None = None
    dataset_dir: Path
    output_root: Path = Path("runs")
    seed: int = 42
    label: Literal["experimental", "pipeline-validation"] = "experimental"
    version_label: str | None = None
    """A model version name for reports, for example ``steward-0.6b-v0.2-A``."""
    eval_suite: str = "ste-multilingual-v1"
    precision: Literal["bf16"] = "bf16"
    lora: LoraSettings
    train: TrainSettings = TrainSettings()

    def sha256(self) -> str:
        """Stable hash of the effective config."""
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


def load_training_config(path: Path) -> TrainingConfig:
    """Load and validate a training config."""
    try:
        return TrainingConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise TrainingConfigError(f"Training config not found: {path}") from None
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise TrainingConfigError(f"Invalid training config {path}: {exc}") from exc
