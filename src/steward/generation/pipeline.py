"""The synthetic-data generation pipeline.

For each example the pipeline:

1. samples a fictional scenario (deterministic from seed and index),
2. asks the teacher to write a plain-English source passage,
3. asks the teacher to rewrite it into an STE-oriented candidate,
4. runs the pre-lint sanity checks,
5. rejects duplicates and failed examples into a separate file,
6. checkpoints all files with atomic writes.

Output files for ``--output data/raw/examples.jsonl``:

* ``examples.jsonl``: accepted :class:`DatasetExample` records,
* ``examples.rejected.jsonl``: :class:`RejectedExample` records,
* ``examples.manifest.json``: one :class:`RunRecord` per run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from steward import __version__
from steward.datasets.jsonl import (
    DuplicateIndex,
    atomic_write_text,
    fingerprint,
    read_jsonl,
    write_jsonl_atomic,
)
from steward.datasets.schema import (
    DatasetExample,
    GenerationParameters,
    PromptVersions,
    Provenance,
    RejectedExample,
    RejectionReason,
    SamplingParameters,
    Scenario,
    StageStats,
    TeacherModel,
)
from steward.environment import collect_environment_info
from steward.generation.config import GenerationConfig
from steward.generation.prompts import PromptTemplate, load_template
from steward.generation.providers.base import (
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    MalformedOutputError,
    ModelInfo,
    ProviderError,
    TextGenerationProvider,
)
from steward.generation.retry import RetriesExhaustedError, RetryPolicy, call_with_retry
from steward.generation.scenarios import derive_seed, focus_text, sample_scenario
from steward.generation.validation import validate_example, validate_multitask
from steward.logs import log_event
from steward.provenance import git_state
from steward.ste import preservation as pv

logger = logging.getLogger(__name__)


# -- teacher output schemas ----------------------------------------------------


class SourceOutput(BaseModel):
    """Structured output of the source-passage stage."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=20)
    identifiers: list[str] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)


class RewriteOutput(BaseModel):
    """Structured output of the rewrite stage."""

    model_config = ConfigDict(extra="ignore")

    ste_text: str = Field(min_length=1)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_json_output[M: BaseModel](text: str, model: type[M]) -> M:
    """Parse teacher output as JSON for ``model``. Tolerates a Markdown code fence."""
    match = _FENCE_RE.match(text)
    body = match.group(1) if match else text.strip()
    try:
        return model.model_validate_json(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "<root>"
        raise MalformedOutputError(
            f"Teacher output does not match {model.__name__}: {where}: {first['msg']} "
            f"({len(text)} chars received)."
        ) from exc


# -- task-specific prompt values -----------------------------------------------


def _label_guidance(language: str) -> str:
    warning = " or ".join(pv.SAFETY_LABELS["WARNING"][language])
    caution = " or ".join(pv.SAFETY_LABELS["CAUTION"][language])
    return (
        f"If the text has a warning, start it with the upper-case label {warning}. If it has "
        f"a caution, start it with the upper-case label {caution}."
    )


def _focus_source_term(scenario: Scenario) -> str | None:
    """The source-language term of a terminology focus (``o0``), if any."""
    if scenario.focus != "terminology" or not scenario.focus_detail:
        return None
    return scenario.focus_detail.split("|")[0].strip() or None


def verified_entities(source_text: str, reported: list[str]) -> list[str]:
    """Identifier codes that must appear unchanged in the target.

    Codes reported by the teacher are kept only if they occur in the source, so a
    teacher error cannot add a requirement. Codes found by the regex are added.
    Measurements are checked separately (see :mod:`steward.ste.preservation`).
    """
    found = [i.strip() for i in reported if i.strip() and i.strip() in source_text]
    found += pv.identifiers(source_text)
    unique: dict[str, None] = {}
    for item in found:
        unique.setdefault(item, None)
    return list(unique)


# -- run bookkeeping -------------------------------------------------------------


class RunStatus(StrEnum):
    """Final state of a generation run."""

    RUNNING = "running"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    ABORTED = "aborted"
    INTERRUPTED = "interrupted"


class RunRecord(BaseModel):
    """Statistics and settings for one generation run (stored in the manifest)."""

    run_id: str
    status: RunStatus
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    requested_total: int
    start_index: int
    scenarios_attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    warning_checks_failed: dict[str, int] = Field(default_factory=dict)
    elapsed_s: float = 0.0
    accepted_per_minute: float | None = None
    mean_example_latency_s: float | None = None
    completion_tokens: int = 0
    completion_tokens_per_s: float | None = None
    error: str | None = None
    teacher: ModelInfo
    prompts: PromptVersions
    config_sha256: str
    config: dict[str, Any]
    steward_version: str
    git_commit: str | None
    git_dirty: bool | None
    environment: dict[str, str]


class Manifest(BaseModel):
    """All runs that wrote to one dataset file."""

    dataset: str
    total_accepted: int = 0
    total_rejected: int = 0
    runs: list[RunRecord] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OutputPaths:
    """Files that belong to one dataset output."""

    examples: Path
    rejected: Path
    manifest: Path

    @classmethod
    def for_output(cls, output: Path) -> OutputPaths:
        """Derive the rejected-file and manifest paths from the examples path."""
        stem = output.name.removesuffix(".jsonl")
        return cls(
            examples=output,
            rejected=output.with_name(f"{stem}.rejected.jsonl"),
            manifest=output.with_name(f"{stem}.manifest.json"),
        )

    def existing(self) -> list[Path]:
        """Return the files that already exist."""
        return [p for p in (self.examples, self.rejected, self.manifest) if p.exists()]


class OutputExistsError(Exception):
    """The output exists and neither resume nor overwrite was requested."""


class GenerationAbortedError(Exception):
    """The run stopped early because of a fatal provider error."""


class _ExampleFailedError(Exception):
    def __init__(
        self,
        reason: RejectionReason,
        message: str,
        *,
        source_text: str | None = None,
        cause: ProviderError | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.source_text = source_text
        self.cause = cause


@dataclass(frozen=True, slots=True)
class _StageOutcome[M: BaseModel]:
    output: M
    stats: StageStats
    params: SamplingParameters


# -- generator ---------------------------------------------------------------------


class DatasetGenerator:
    """Generate STE-oriented rewrite candidates with a teacher model."""

    def __init__(
        self,
        *,
        provider: TextGenerationProvider,
        model: str,
        config: GenerationConfig,
        prompt_dir: Path,
        output: Path,
        resume: bool = False,
        overwrite: bool = False,
        checkpoint_every: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Prepare a run. No provider calls happen until :meth:`run`."""
        if resume and overwrite:
            raise ValueError("Use either resume or overwrite, not both.")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        self.provider = provider
        self.model = model
        self.config = config
        self.paths = OutputPaths.for_output(output)
        self.resume = resume
        self.overwrite = overwrite
        self.checkpoint_every = checkpoint_every
        self._sleep = sleep
        self._clock = clock
        self.source_template: PromptTemplate = load_template(prompt_dir, config.source_template)
        self.rewrite_template: PromptTemplate = load_template(prompt_dir, config.rewrite_template)
        self.retry_policy = RetryPolicy(
            max_attempts=config.request.max_attempts,
            initial_delay_s=config.request.initial_backoff_s,
            max_delay_s=config.request.max_backoff_s,
        )
        self.examples: list[DatasetExample] = []
        self.rejected: list[RejectedExample] = []
        self.manifest = Manifest(dataset=self.paths.examples.name)
        self._dedup = DuplicateIndex()
        self._guidance = {c.name: c.guidance for c in config.categories}

    # -- public -----------------------------------------------------------

    def run(self, total: int, *, max_scenarios: int | None = None) -> RunRecord:
        """Generate examples until the dataset has ``total`` accepted records.

        Args:
            total: Target number of accepted examples in the output file.
            max_scenarios: Upper limit on scenarios tried in this run. The
                default is ``3 * remaining + 5``.
        """
        if total < 1:
            raise ValueError("total must be at least 1")
        start_index = self._prepare_output()
        teacher = self.provider.describe_model(self.model)
        commit, dirty = git_state()
        record = RunRecord(
            run_id=f"run-{self._clock():%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
            status=RunStatus.RUNNING,
            started_at=self._clock(),
            requested_total=total,
            start_index=start_index,
            teacher=teacher,
            prompts=self._prompt_versions(),
            config_sha256=self.config.sha256(),
            config=self.config.model_dump(mode="json"),
            steward_version=__version__,
            git_commit=commit,
            git_dirty=dirty,
            environment=collect_environment_info().to_dict(),
        )
        self._warn_on_mixed_runs(record)
        self.manifest.runs.append(record)
        remaining = total - len(self.examples)
        limit = max_scenarios if max_scenarios is not None else 3 * max(remaining, 0) + 5
        log_event(
            logger,
            logging.INFO,
            "run_started",
            run_id=record.run_id,
            model=teacher.model,
            digest=teacher.digest,
            target=total,
            existing=len(self.examples),
            start_index=start_index,
            output=str(self.paths.examples),
        )

        started = time.perf_counter()
        reasons: Counter[str] = Counter()
        warnings: Counter[str] = Counter()
        latencies: list[float] = []
        tokens = 0
        consecutive_failures = 0
        index = start_index
        status = RunStatus.COMPLETED
        error: str | None = None
        try:
            while len(self.examples) < total:
                if record.scenarios_attempted >= limit:
                    status = RunStatus.INCOMPLETE
                    error = f"Stopped after {limit} scenarios; acceptance rate is too low."
                    break
                scenario = sample_scenario(self.config, index)
                index += 1
                record.scenarios_attempted += 1
                example_started = time.perf_counter()
                try:
                    example = self._generate_one(scenario, teacher, record)
                except _ExampleFailedError as failure:
                    reasons[failure.reason.value] += 1
                    self._reject(record, scenario, failure)
                    if failure.reason is RejectionReason.PROVIDER_ERROR:
                        consecutive_failures += 1
                        if consecutive_failures >= self.config.request.max_consecutive_failures:
                            raise GenerationAbortedError(
                                f"{consecutive_failures} provider failures in a row. "
                                f"Last error: {failure}"
                            ) from failure.cause
                else:
                    consecutive_failures = 0
                    duplicate_of = self._dedup.find(example.source_text, example.ste_candidate)
                    if duplicate_of is not None:
                        reasons[RejectionReason.DUPLICATE.value] += 1
                        self._reject(
                            record,
                            scenario,
                            _ExampleFailedError(
                                RejectionReason.DUPLICATE,
                                f"Duplicate of {duplicate_of}.",
                                source_text=example.source_text,
                            ),
                            candidate=example.ste_candidate,
                            duplicate_of=duplicate_of,
                        )
                    elif not example.validation.passed:
                        reasons[RejectionReason.FAILED_VALIDATION.value] += 1
                        self._reject(
                            record,
                            scenario,
                            _ExampleFailedError(
                                RejectionReason.FAILED_VALIDATION,
                                "Failed checks: " + ", ".join(example.validation.failed_check_ids),
                                source_text=example.source_text,
                            ),
                            candidate=example.ste_candidate,
                            report=example,
                        )
                    else:
                        self._accept(example)
                        record.accepted += 1
                        warnings.update(example.validation.failed_check_ids)
                        latencies.append(time.perf_counter() - example_started)
                        tokens += sum(s.completion_tokens or 0 for s in example.stages)
                        log_event(
                            logger,
                            logging.INFO,
                            "example_accepted",
                            example_id=example.example_id,
                            index=scenario.index,
                            category=scenario.category.value,
                            latency_s=round(latencies[-1], 2),
                            warnings=example.validation.failed_check_ids,
                            progress=f"{len(self.examples)}/{total}",
                        )
                if record.scenarios_attempted % self.checkpoint_every == 0:
                    self._update_stats(record, started, reasons, warnings, latencies, tokens)
                    self._flush()
        except GenerationAbortedError as exc:
            status, error = RunStatus.ABORTED, str(exc)
            raise
        except KeyboardInterrupt:
            status, error = RunStatus.INTERRUPTED, "Interrupted by user."
            raise
        except ProviderError as exc:
            status, error = RunStatus.ABORTED, str(exc)
            raise GenerationAbortedError(str(exc)) from exc
        finally:
            record.status = status
            record.error = error
            record.finished_at = self._clock()
            self._update_stats(record, started, reasons, warnings, latencies, tokens)
            self._flush()
            log_event(
                logger,
                logging.INFO if status is RunStatus.COMPLETED else logging.WARNING,
                "run_finished",
                run_id=record.run_id,
                status=status.value,
                accepted=record.accepted,
                rejected=record.rejected,
                elapsed_s=round(record.elapsed_s, 1),
                accepted_per_minute=record.accepted_per_minute,
                error=error,
            )
        return record

    # -- stages -------------------------------------------------------------

    def _generate_one(
        self, scenario: Scenario, teacher: ModelInfo, record: RunRecord
    ) -> DatasetExample:
        language = scenario.source_language
        language_name = pv.LANGUAGE_NAMES[language]
        category = scenario.category.value.replace("_", " ")
        source_values = {
            "category": category,
            "category_guidance": self._guidance[scenario.category],
            "topic": scenario.topic,
            "audience": scenario.audience,
            "sentence_count": str(scenario.sentence_count),
        }
        tasks = self.config.tasks
        spec = (
            self.config.focus.specs.get(scenario.focus)
            if self.config.focus is not None and scenario.focus
            else None
        )
        if tasks is not None:
            source_values |= {
                "language_name": language_name,
                "format_guidance": tasks.source_formats[scenario.task_type].strip(),
                "label_guidance": _label_guidance(language),
            }
            if spec is not None:
                source_values["format_guidance"] += "\n" + focus_text(
                    spec.source, scenario.focus_detail, language_name
                )
                if spec.label_guidance:
                    source_values["label_guidance"] = focus_text(
                        spec.label_guidance, scenario.focus_detail, language_name
                    )
        source = self._run_stage(
            "source",
            self.source_template,
            source_values,
            SourceOutput,
            self.config.sampling.source,
            scenario.index,
        )
        source_text = source.output.text.strip()
        rewrite_values = {"category": category, "source_text": source_text}
        if tasks is not None:
            rewrite_values["task_instruction"] = (
                tasks.instructions[scenario.task_type]
                .strip()
                .format(language=language_name, category=category)
            )
            if spec is not None:
                rewrite_values["task_instruction"] += "\n" + focus_text(
                    spec.rewrite, scenario.focus_detail, language_name
                )
        rewrite = self._run_stage(
            "rewrite",
            self.rewrite_template,
            rewrite_values,
            RewriteOutput,
            self.config.sampling.rewrite,
            scenario.index,
            source_text=source_text,
        )
        candidate = rewrite.output.ste_text.strip()
        entities: list[str] = []
        constraints: list[str] = []
        if tasks is not None:
            entities = verified_entities(source_text, source.output.identifiers)
            constraints = [c.strip() for c in source.output.safety_constraints if c.strip()]
            report = validate_multitask(
                scenario.task_type,
                language,
                scenario.category,
                source_text,
                candidate,
                entities,
                self.config.validation,
                required_terms=scenario.required_terms,
                focus_source_term=_focus_source_term(scenario),
            )
        else:
            report = validate_example(
                scenario.category, source_text, candidate, self.config.validation
            )
        return DatasetExample(
            example_id="stew-" + fingerprint(source_text)[:16],
            task_type=scenario.task_type,
            task_category=scenario.category,
            source_language=language,
            source_title=source.output.title.strip(),
            source_text=source_text,
            ste_candidate=candidate,
            technical_entities=entities,
            safety_constraints=constraints,
            teacher=TeacherModel(
                provider=teacher.provider,
                model=teacher.model,
                digest=teacher.digest,
                provider_version=teacher.provider_version,
                details=teacher.details,
            ),
            prompt_version=record.prompts,
            generation_parameters=GenerationParameters(
                source=source.params, rewrite=rewrite.params
            ),
            stages=[source.stats, rewrite.stats],
            validation=report,
            provenance=Provenance(
                run_id=record.run_id,
                created_at=self._clock(),
                steward_version=__version__,
                git_commit=record.git_commit,
                git_dirty=record.git_dirty,
                config_sha256=record.config_sha256,
                environment=record.environment,
                scenario=scenario,
                data_license=self.config.data_license,
            ),
        )

    def _run_stage[M: BaseModel](
        self,
        stage: str,
        template: PromptTemplate,
        values: dict[str, str],
        output_model: type[M],
        base_params: SamplingParameters,
        index: int,
        *,
        source_text: str | None = None,
    ) -> _StageOutcome[M]:
        rendered = template.render(**values)
        messages = [
            ChatMessage(role="system", content=rendered.system),
            ChatMessage(role="user", content=rendered.user),
        ]
        stage_seed = derive_seed(self.config.seed, stage, index)
        schema = output_model.model_json_schema()

        def attempt(n: int) -> tuple[M, GenerationResult, SamplingParameters]:
            # A new seed per attempt: with a fixed seed, a retry would repeat a bad output.
            params = base_params.model_copy(update={"seed": stage_seed + n - 1})
            result = self.provider.generate(
                GenerationRequest(
                    model=self.model, messages=messages, sampling=params, response_schema=schema
                )
            )
            return parse_json_output(result.text, output_model), result, params

        started = time.perf_counter()
        try:
            (output, result, params), attempts = call_with_retry(
                attempt, self.retry_policy, label=f"{stage}#{index}", sleep=self._sleep
            )
        except RetriesExhaustedError as exc:
            reason = (
                RejectionReason.MALFORMED_OUTPUT
                if isinstance(exc.last_error, MalformedOutputError)
                else RejectionReason.PROVIDER_ERROR
            )
            raise _ExampleFailedError(
                reason, f"{stage} stage: {exc}", source_text=source_text, cause=exc.last_error
            ) from exc
        return _StageOutcome(
            output=output,
            stats=StageStats(
                stage="source" if stage == "source" else "rewrite",
                attempts=attempts,
                latency_s=round(time.perf_counter() - started, 4),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            ),
            params=params,
        )

    # -- bookkeeping --------------------------------------------------------

    def _prompt_versions(self) -> PromptVersions:
        return PromptVersions(source=self.source_template.ref, rewrite=self.rewrite_template.ref)

    def _prepare_output(self) -> int:
        existing = self.paths.existing()
        if existing and not (self.resume or self.overwrite):
            names = ", ".join(str(p) for p in existing)
            raise OutputExistsError(
                f"Output already exists: {names}. Use --resume to continue or --overwrite "
                "to replace it."
            )
        if self.overwrite or not existing:
            return 0
        if self.paths.examples.exists():
            self.examples = read_jsonl(self.paths.examples, DatasetExample)
        if self.paths.rejected.exists():
            self.rejected = read_jsonl(self.paths.rejected, RejectedExample)
        if self.paths.manifest.exists():
            self.manifest = Manifest.model_validate_json(
                self.paths.manifest.read_text(encoding="utf-8")
            )
        for example in self.examples:
            self._dedup.add(example.example_id, example.source_text, example.ste_candidate)
        indices = [e.provenance.scenario.index for e in self.examples]
        indices += [r.scenario.index for r in self.rejected]
        return max(indices, default=-1) + 1

    def _warn_on_mixed_runs(self, record: RunRecord) -> None:
        previous = [r for r in self.manifest.runs if r.run_id != record.run_id]
        if not previous:
            return
        last = previous[-1]
        if last.config_sha256 != record.config_sha256:
            log_event(logger, logging.WARNING, "resume_config_changed", previous=last.run_id)
        if (last.teacher.model, last.teacher.digest) != (
            record.teacher.model,
            record.teacher.digest,
        ):
            log_event(
                logger,
                logging.WARNING,
                "resume_teacher_changed",
                previous=f"{last.teacher.model}@{last.teacher.digest}",
                current=f"{record.teacher.model}@{record.teacher.digest}",
            )

    def _accept(self, example: DatasetExample) -> None:
        self.examples.append(example)
        self._dedup.add(example.example_id, example.source_text, example.ste_candidate)

    def _reject(
        self,
        record: RunRecord,
        scenario: Scenario,
        failure: _ExampleFailedError,
        *,
        candidate: str | None = None,
        duplicate_of: str | None = None,
        report: DatasetExample | None = None,
    ) -> None:
        key = f"{record.run_id}:{scenario.index}"
        rejected = RejectedExample(
            rejection_id="rej-" + hashlib.sha256(key.encode()).hexdigest()[:16],
            reason=failure.reason,
            message=str(failure),
            created_at=self._clock(),
            run_id=record.run_id,
            scenario=scenario,
            teacher_model=self.model,
            source_text=failure.source_text,
            ste_candidate=candidate,
            duplicate_of=duplicate_of,
            validation=report.validation if report else None,
        )
        self.rejected.append(rejected)
        record.rejected += 1
        log_event(
            logger,
            logging.WARNING,
            "example_rejected",
            index=scenario.index,
            reason=failure.reason.value,
            message=str(failure),
        )

    @staticmethod
    def _update_stats(
        record: RunRecord,
        started: float,
        reasons: Counter[str],
        warnings: Counter[str],
        latencies: list[float],
        tokens: int,
    ) -> None:
        elapsed = time.perf_counter() - started
        record.elapsed_s = round(elapsed, 3)
        record.rejection_reasons = dict(reasons)
        record.warning_checks_failed = dict(warnings)
        record.completion_tokens = tokens
        if elapsed > 0:
            record.accepted_per_minute = round(record.accepted * 60 / elapsed, 3)
            record.completion_tokens_per_s = round(tokens / elapsed, 2)
        if latencies:
            record.mean_example_latency_s = round(sum(latencies) / len(latencies), 3)

    def _flush(self) -> None:
        write_jsonl_atomic(self.paths.examples, self.examples)
        write_jsonl_atomic(self.paths.rejected, self.rejected)
        self.manifest.total_accepted = len(self.examples)
        self.manifest.total_rejected = len(self.rejected)
        atomic_write_text(
            self.paths.manifest,
            json.dumps(self.manifest.model_dump(mode="json"), indent=2) + "\n",
        )
