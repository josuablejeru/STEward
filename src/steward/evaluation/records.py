"""Schemas for the append-only experiment history.

Two record types share one history file:

* :class:`BenchmarkRecord`: one evaluation of one candidate on one suite.
* :class:`TrainingRecord`: one training run (losses, hyperparameters).

A checkpoint can have many benchmark records. Records are never changed
after they are written. ``stale`` and ``pending`` are *derived* states
(see :mod:`steward.evaluation.status`); they are not written into
completed records.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue

from steward.datasets.schema import PromptRef, SamplingParameters
from steward.ste.rules import RuleCategory, Violation

HISTORY_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _none(value: object) -> bool:
    """``exclude_if`` predicate: fields added after schema 1.0 are not stored when unset.

    This keeps the canonical JSON (and so the content hash) of older records unchanged.
    """
    return value is None


class RecordStatus(StrEnum):
    """Stored status. Only real outcomes are stored."""

    COMPLETED = "completed"
    FAILED = "failed"


class EffectiveStatus(StrEnum):
    """Status shown in reports, derived from stored records and current inputs."""

    COMPLETED = "completed"
    FAILED = "failed"
    STALE = "stale"
    PENDING = "pending"


class CandidateRole(StrEnum):
    """How a candidate relates to the base model."""

    BASE = "base"
    PROMPTED = "prompted"
    FINETUNED = "finetuned"
    QUANTIZED = "quantized"
    RELEASE = "release"


class MetricMethod(StrEnum):
    """Where a metric value comes from. Never mix these in one number."""

    AUTOMATED = "automated"
    HUMAN_REVIEW = "human_review"
    LLM_JUDGE = "llm_judge"


class Metric(_Strict):
    """One aggregate metric with its denominator and uncertainty.

    ``value`` is ``None`` when the metric is not available; ``note`` says why.
    """

    value: float | None
    unit: str
    n: int = Field(ge=0, description="Denominator: items, words, or calls.")
    ci_low: float | None = None
    ci_high: float | None = None
    method: MetricMethod = MetricMethod.AUTOMATED
    higher_is_better: bool | None = None
    note: str | None = None


class CandidateSpec(_Strict):
    """The model, prompt, and role that were evaluated."""

    candidate_id: str
    role: CandidateRole
    provider: str
    model: str
    model_digest: str | None = None
    base_model: str
    base_revision: str | None = None
    checkpoint_id: str | None = None
    adapter_id: str | None = None
    training_run_id: str | None = None
    release_version: str | None = None
    parameter_count: int | None = None
    quantization: str | None = None
    disk_size_bytes: int | None = None
    prompt: PromptRef
    think: bool | None = None
    adapter_path: str | None = Field(default=None, exclude_if=_none)
    adapter_sha256: str | None = Field(default=None, exclude_if=_none)
    adapter_size_bytes: int | None = Field(default=None, exclude_if=_none)


class SuiteRef(_Strict):
    """The frozen evaluation suite that was used."""

    suite_id: str
    version: int
    sha256: str
    item_count: int
    review_state: Literal["draft", "human_reviewed"]


class EvaluatorRef(_Strict):
    """Versions of the scoring code. Different versions are not comparable."""

    evaluator_version: str
    ruleset_version: str
    rules_config_sha256: str
    code_fingerprint: str


class CodeRef(_Strict):
    """The code state that produced a record."""

    steward_version: str
    git_commit: str | None
    git_dirty: bool | None
    worktree_fingerprint: str


class Hardware(_Strict):
    """The machine that ran the experiment."""

    platform: str
    cpu_count: int | None = None
    gpu_name: str | None = None
    gpu_memory_total_mib: int | None = None
    gpu_driver: str | None = None
    provider_version: str | None = None
    note: str | None = None


class ItemResult(_Strict):
    """The result for one suite item."""

    item_id: str
    category: str
    valid_output: bool
    attempts: int
    repair_attempts: int
    output: str | None
    error: str | None = None
    words: int = 0
    violations: list[Violation] = Field(default_factory=list)
    rule_check_passed: bool = False
    facts_total: int = 0
    facts_preserved: int = 0
    missing_facts: list[str] = Field(default_factory=list)
    invented_numbers: list[str] = Field(default_factory=list)
    negation_lost: bool = False
    safety_labels_missing: list[str] = Field(default_factory=list)
    critical_failure: bool = False
    repetition_rate: float = 0.0
    latency_s: float | None = None
    completion_tokens: int | None = None
    tokens_per_s: float | None = None
    # STE-only contract (suites with contract "ste_only_text"); unset for older suites.
    task_type: str | None = Field(default=None, exclude_if=_none)
    source_language: str | None = Field(default=None, exclude_if=_none)
    expected_unsupported: bool | None = Field(default=None, exclude_if=_none)
    guard_status: str | None = Field(default=None, exclude_if=_none)
    guard_critical: list[str] | None = Field(default=None, exclude_if=_none)
    ste_only: bool | None = Field(default=None, exclude_if=_none)
    """Non-empty English text without commentary, Markdown, or reasoning."""
    non_english: bool | None = Field(default=None, exclude_if=_none)
    extra_commentary: bool | None = Field(default=None, exclude_if=_none)
    controlled_failure: bool | None = Field(default=None, exclude_if=_none)
    """The model returned the unsupported-input sentinel."""
    unchanged: bool | None = Field(default=None, exclude_if=_none)
    """Identity items: the output equals the input after normalization."""
    edit_similarity: float | None = Field(default=None, exclude_if=_none)
    chrf: float | None = Field(default=None, exclude_if=_none)
    negation_expected: bool | None = Field(default=None, exclude_if=_none)
    labels_expected: bool | None = Field(default=None, exclude_if=_none)
    terms_matched: list[str] | None = Field(default=None, exclude_if=_none)
    """chrF2 against the reference STE text (heuristic translation-quality proxy)."""


class _HistoryBase(_Strict):
    schema_version: Literal["1.0"] = HISTORY_SCHEMA_VERSION
    experiment_id: str
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prev_record_sha256: str | None = None
    created_at: AwareDatetime
    status: RecordStatus
    error: str | None = None
    code: CodeRef
    hardware: Hardware


class BenchmarkRecord(_HistoryBase):
    """One evaluation of one candidate on one frozen suite."""

    record_type: Literal["benchmark"] = "benchmark"
    candidate: CandidateSpec
    suite: SuiteRef
    evaluator: EvaluatorRef
    sampling: SamplingParameters
    dataset_version: str | None = Field(
        default=None, description="Training dataset fingerprint (fine-tuned candidates only)."
    )
    metrics: dict[str, Metric] = Field(default_factory=dict)
    by_category: dict[str, dict[str, Metric]] = Field(default_factory=dict)
    violation_counts: dict[RuleCategory, int] = Field(default_factory=dict)
    rule_counts: dict[str, int] = Field(default_factory=dict)
    items: list[ItemResult] = Field(default_factory=list)
    duration_s: float = 0.0
    disclaimer: str = (
        "Automated heuristic checks on a draft or reviewed benchmark suite. "
        "Not an ASD-STE100 certification."
    )

    @property
    def comparability_key(self) -> str:
        """Results are directly comparable only when this key is equal."""
        return comparability_key(
            self.suite,
            self.evaluator.evaluator_version,
            self.evaluator.ruleset_version,
            self.evaluator.rules_config_sha256,
        )


def comparability_key(
    suite: SuiteRef, evaluator_version: str, ruleset_version: str, rules_sha256: str
) -> str:
    """Return the key that must match for two results to be directly comparable."""
    return (
        f"{suite.suite_id}@{suite.version}:{suite.sha256[:8]}"
        f"|{evaluator_version}|{ruleset_version}:{rules_sha256[:8]}"
    )


class TrainingPoint(_Strict):
    """One logged training step."""

    step: int = Field(ge=0)
    epoch: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    gpu_allocated_mib: int | None = Field(default=None, exclude_if=_none)
    gpu_peak_mib: int | None = Field(default=None, exclude_if=_none)


class ServingSpec(_Strict):
    """How to reach a trained model for evaluation (for example an Ollama model).

    Fields added after history schema 1.0 are omitted from the stored JSON when
    unset, so the content hashes of older records do not change.
    """

    provider: str
    model: str
    quantization: str | None = None
    revision: str | None = Field(default=None, exclude_if=_none)
    adapter_path: str | None = Field(default=None, exclude_if=_none)
    """Local LoRA adapter directory (provider ``transformers``)."""
    prompt: str | None = Field(default=None, exclude_if=_none)
    """Evaluation prompt reference; the default is the JSON rewrite prompt."""
    suite: str | None = Field(default=None, exclude_if=_none)
    """Benchmark suite directory name for automatic evaluation."""


class TrainingSummary(_Strict):
    """The file a training run writes when it finishes: ``runs/<run_id>/training_summary.json``.

    This is the input contract between training and benchmarking.
    """

    run_id: str
    status: RecordStatus
    error: str | None = None
    base_model: str
    base_revision: str | None = None
    checkpoint_id: str
    adapter_id: str | None = None
    checkpoint_path: str
    dataset_version: str
    dataset_fingerprint: str
    config_sha256: str
    prompt_versions: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, JsonValue] = Field(default_factory=dict)
    parameter_count: int | None = None
    trainable_parameters: int | None = None
    quantization: str | None = None
    curve: list[TrainingPoint] = Field(default_factory=list)
    serving: ServingSpec | None = None


class TrainingRecord(_HistoryBase):
    """A training run stored in the history."""

    record_type: Literal["training"] = "training"
    summary: TrainingSummary


HistoryRecord = Annotated[BenchmarkRecord | TrainingRecord, Field(discriminator="record_type")]
