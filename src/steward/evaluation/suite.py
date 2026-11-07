"""Load the frozen golden suite, the candidate registry, and thresholds."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from steward.datasets.jsonl import DatasetFileError, read_jsonl
from steward.datasets.schema import SamplingParameters, SourceLanguage, TaskCategory, TaskType
from steward.evaluation.records import CandidateRole, SuiteRef


class BenchmarkConfigError(Exception):
    """A suite, registry, or threshold file is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SuiteItem(_Strict):
    """One golden item: a source text and the facts a rewrite must keep.

    Fields after ``must_include`` are used by ``ste_only_text`` suites.
    """

    item_id: str
    category: TaskCategory | None
    safety_critical: bool
    source_text: str = Field(min_length=1)
    must_include: list[str] = Field(default_factory=list)
    task_type: TaskType = TaskType.PLAIN_TO_STE_REWRITE
    source_language: SourceLanguage = "en"
    reference: str | None = None
    """A reference STE-oriented English text (draft, written by the suite author)."""
    expected_unsupported: bool = False
    """Out-of-domain item: the correct answer is the controlled failure."""

    @property
    def category_label(self) -> str:
        """Category name for reports (``out_of_domain`` when there is none)."""
        return self.category.value if self.category else "out_of_domain"


class RepairSettings(_Strict):
    """How often to re-ask for valid JSON."""

    max_repair_attempts: int = Field(default=2, ge=0)


class SuiteManifest(_Strict):
    """``suite.toml``."""

    id: str
    version: int = Field(ge=0)
    review_state: Literal["draft", "human_reviewed"]
    author: str
    description: str
    items_file: str
    items_sha256: str
    item_count: int
    sampling: SamplingParameters
    repair: RepairSettings = RepairSettings()
    contract: Literal["json_rewrite", "ste_only_text"] = "json_rewrite"
    """``json_rewrite``: the model returns ``{"ste_text": ...}``. ``ste_only_text``: the
    model returns only STE text or the unsupported-input sentinel (STE-only contract)."""


class Suite(_Strict):
    """A loaded and verified suite."""

    manifest: SuiteManifest
    items: list[SuiteItem]
    manifest_sha256: str
    path: Path

    @property
    def sha256(self) -> str:
        """Combined hash of the manifest and items."""
        return hashlib.sha256(
            f"{self.manifest_sha256}:{self.manifest.items_sha256}".encode()
        ).hexdigest()

    @property
    def ref(self) -> SuiteRef:
        """Reference stored on benchmark records."""
        return SuiteRef(
            suite_id=self.manifest.id,
            version=self.manifest.version,
            sha256=self.sha256,
            item_count=len(self.items),
            review_state=self.manifest.review_state,
        )


def load_suite(suite_dir: Path) -> Suite:
    """Load a suite and check that its items match the frozen hash."""
    manifest_path = suite_dir / "suite.toml"
    try:
        raw = manifest_path.read_bytes()
        manifest = SuiteManifest.model_validate(tomllib.loads(raw.decode("utf-8")))
    except FileNotFoundError:
        raise BenchmarkConfigError(f"Suite manifest not found: {manifest_path}") from None
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise BenchmarkConfigError(f"Invalid suite manifest {manifest_path}: {exc}") from exc
    items_path = suite_dir / manifest.items_file
    try:
        items_bytes = items_path.read_bytes()
    except FileNotFoundError:
        raise BenchmarkConfigError(f"Suite items not found: {items_path}") from None
    actual = hashlib.sha256(items_bytes).hexdigest()
    if actual != manifest.items_sha256:
        raise BenchmarkConfigError(
            f"Golden suite {items_path} was modified: sha256 {actual[:12]}... does not match "
            f"the frozen {manifest.items_sha256[:12]}... in suite.toml. Golden items are "
            "immutable: revert the change, or create a new suite version."
        )
    try:
        items = read_jsonl(items_path, SuiteItem)
    except DatasetFileError as exc:
        raise BenchmarkConfigError(str(exc)) from exc
    if len(items) != manifest.item_count:
        raise BenchmarkConfigError(
            f"{items_path}: {len(items)} items, but suite.toml says {manifest.item_count}."
        )
    ids = [i.item_id for i in items]
    if len(set(ids)) != len(ids):
        raise BenchmarkConfigError(f"{items_path}: duplicate item IDs.")
    return Suite(
        manifest=manifest,
        items=items,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        path=suite_dir,
    )


class CandidateConfig(_Strict):
    """One entry in ``benchmarks/models.toml``."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    role: CandidateRole
    provider: str = "ollama"
    model: str
    base_model: str
    base_revision: str | None = None
    prompt: str
    think: bool | None = None
    parameter_count: int | None = None
    quantization: str | None = None
    checkpoint_id: str | None = None
    adapter_id: str | None = None
    training_run_id: str | None = None
    release_version: str | None = None
    notes: str | None = None
    suite: str = "ste-golden-v0"
    """Suite directory under ``benchmarks/golden/`` this candidate is evaluated on."""
    adapter_path: str | None = None
    """LoRA adapter directory (provider ``transformers``)."""


class CandidateRegistry(_Strict):
    """``benchmarks/models.toml``."""

    primary_suite: str = "ste-golden-v0"
    """The suite used for model selection, the main report, and the README."""
    candidates: list[CandidateConfig] = Field(default_factory=list)

    def get(self, candidate_id: str) -> CandidateConfig:
        """Return a candidate by ID."""
        for candidate in self.candidates:
            if candidate.id == candidate_id:
                return candidate
        known = ", ".join(c.id for c in self.candidates) or "none"
        raise BenchmarkConfigError(f"Unknown candidate {candidate_id!r}. Known: {known}.")


def load_registry(path: Path) -> CandidateRegistry:
    """Load the candidate registry."""
    try:
        registry = CandidateRegistry.model_validate(tomllib.loads(path.read_text("utf-8")))
    except FileNotFoundError:
        raise BenchmarkConfigError(f"Candidate registry not found: {path}") from None
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise BenchmarkConfigError(f"Invalid candidate registry {path}: {exc}") from exc
    ids = [c.id for c in registry.candidates]
    if len(set(ids)) != len(ids):
        raise BenchmarkConfigError(f"{path}: duplicate candidate IDs.")
    return registry


class Limit(_Strict):
    """A regression limit for one metric."""

    max_drop: float | None = None
    max_increase: float | None = None
    max_increase_ratio: float | None = None
    min_quality_gain: float | None = None


class SelectionRequirements(_Strict):
    """Documented quality requirements for model selection."""

    min_ste_rule_pass_rate: float
    min_meaning_preservation: float
    max_hallucination_rate: float
    max_missing_information_rate: float
    max_critical_failures: int
    min_task_completion_rate: float


class Bound(_Strict):
    """A minimum and/or maximum for one metric."""

    min: float | None = None
    max: float | None = None


class Thresholds(_Strict):
    """``benchmarks/thresholds.toml``."""

    version: int
    regression: dict[str, Limit]
    selection: SelectionRequirements
    selection_by_suite: dict[str, dict[str, Bound]] = Field(default_factory=dict)
    """Suite ID -> metric bounds. Replaces ``selection`` for that suite."""


def load_thresholds(path: Path) -> Thresholds:
    """Load regression thresholds and selection requirements."""
    try:
        return Thresholds.model_validate(tomllib.loads(path.read_text("utf-8")))
    except FileNotFoundError:
        raise BenchmarkConfigError(f"Thresholds file not found: {path}") from None
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise BenchmarkConfigError(f"Invalid thresholds {path}: {exc}") from exc
