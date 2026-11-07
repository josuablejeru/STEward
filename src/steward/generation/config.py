"""Generation run configuration, loaded from TOML (see ``configs/generation/``)."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from steward.datasets.schema import SamplingParameters, SourceLanguage, TaskCategory, TaskType


class ConfigError(Exception):
    """Raised when a configuration file is missing or invalid."""


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StageSampling(_Config):
    """Sampling parameters for both stages. Seeds are derived per example."""

    source: SamplingParameters
    rewrite: SamplingParameters


class RequestSettings(_Config):
    """Timeouts and retry settings for provider calls."""

    timeout_s: float = Field(default=120.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    initial_backoff_s: float = Field(default=1.0, ge=0)
    max_backoff_s: float = Field(default=30.0, ge=0)
    max_consecutive_failures: int = Field(default=5, ge=1)


class ValidationSettings(_Config):
    """Limits for the pre-lint sanity checks."""

    max_words_procedural: int = Field(default=20, ge=1)
    max_words_descriptive: int = Field(default=25, ge=1)
    min_length_ratio: float = Field(default=0.3, gt=0)
    max_length_ratio: float = Field(default=3.0, gt=0)


class CategorySpec(_Config):
    """A text category, its sampling weight, and its guidance for the teacher."""

    name: TaskCategory
    weight: float = Field(gt=0)
    guidance: str


GENERATED_TASKS = (
    TaskType.PLAIN_TO_STE_REWRITE,
    TaskType.MULTILINGUAL_TO_STE,
    TaskType.REQUIREMENTS_TO_STE,
)
"""Task types that the teacher generates. Identity and out-of-domain examples are
derived when the training set is prepared (see :mod:`steward.datasets.sft`)."""


_TRANSLATION_DEFAULT: tuple[SourceLanguage, ...] = ("de", "fr", "ro")
_REQUIREMENT_DEFAULT: tuple[SourceLanguage, ...] = ("en", "de", "fr", "ro")


class TaskMix(_Config):
    """Multitask generation: task weights and source languages.

    Weights are relative and need not sum to 1.
    """

    weights: dict[TaskType, float]
    translation_languages: list[SourceLanguage] = Field(
        default_factory=lambda: list(_TRANSLATION_DEFAULT), min_length=1
    )
    requirement_languages: list[SourceLanguage] = Field(
        default_factory=lambda: list(_REQUIREMENT_DEFAULT), min_length=1
    )
    source_formats: dict[TaskType, str] = Field(default_factory=dict)
    """Source-stage ``format_guidance`` per task (part of the config hash)."""
    instructions: dict[TaskType, str] = Field(default_factory=dict)
    """Rewrite-stage ``task_instruction`` per task. ``{language}`` and ``{category}``
    are replaced with the source language name and the text category."""

    @model_validator(mode="after")
    def _generated_only(self) -> TaskMix:
        unknown = [t for t in self.weights if t not in GENERATED_TASKS]
        if unknown:
            raise ValueError(
                f"tasks.weights may only contain {', '.join(GENERATED_TASKS)}; got {unknown}. "
                "Identity and out-of-domain shares are set in the SFT preparation config."
            )
        if not any(w > 0 for w in self.weights.values()):
            raise ValueError("At least one task weight must be positive.")
        if "en" in self.translation_languages:
            raise ValueError("translation_languages must not include 'en'.")
        for name, table in (
            ("source_formats", self.source_formats),
            ("instructions", self.instructions),
        ):
            missing = [t.value for t in self.weights if t not in table]
            if missing:
                raise ValueError(f"tasks.{name} has no entry for {missing}.")
        return self


class FocusSpec(_Config):
    """One targeted failure mode for generation.

    An option is a ``|``-separated string. Its parts are available as ``{o0}``,
    ``{o1}``, ... in ``source`` and ``rewrite``; ``{language}`` is the language name.
    ``required_part`` names the part that the English target must contain.
    """

    weight: float = Field(ge=0)
    tasks: list[TaskType] | None = None
    """Tasks this focus applies to (None: all generated tasks)."""
    options: dict[str, list[str]] = Field(default_factory=dict)
    """Options per source language code, or under ``any``."""
    source: str
    rewrite: str
    label_guidance: str | None = None
    """Replaces the default safety-label guidance of the source stage."""
    required_part: int | None = None


class FocusMix(_Config):
    """Weights and specs of the targeted failure modes (``none`` = no focus)."""

    none_weight: float = Field(default=0.0, ge=0)
    specs: dict[str, FocusSpec]


class GenerationConfig(_Config):
    """All settings for one generation run."""

    seed: int = 0
    source_template: str
    rewrite_template: str
    sampling: StageSampling
    request: RequestSettings = RequestSettings()
    validation: ValidationSettings = ValidationSettings()
    categories: list[CategorySpec] = Field(min_length=1)
    topics: list[str] = Field(min_length=1)
    audiences: list[str] = Field(min_length=1)
    sentence_counts: list[int] = Field(min_length=1)
    data_license: str = "pending-rights-review"
    tasks: TaskMix | None = None
    """Multitask settings. ``None`` generates English rewrites only (schema 1.0 behavior)."""
    focus: FocusMix | None = Field(default=None, exclude_if=lambda v: v is None)
    """Targeted failure modes (Milestone 4). Requires ``tasks``."""

    def sha256(self) -> str:
        """Return a stable hash of the effective configuration."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: Path) -> GenerationConfig:
    """Load and validate a generation config file."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid TOML: {exc}") from exc
    try:
        return GenerationConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config {path}:\n{exc}") from exc
