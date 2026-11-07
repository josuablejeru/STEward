"""High-level benchmark operations shared by the CLI and the hooks."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from steward import __version__
from steward.datasets.jsonl import atomic_write_text
from steward.evaluation.fingerprints import worktree_fingerprint
from steward.evaluation.hardware import describe_hardware
from steward.evaluation.history import append_record, finalize_record
from steward.evaluation.project import BenchmarkProject
from steward.evaluation.records import (
    BenchmarkRecord,
    CandidateRole,
    CodeRef,
    RecordStatus,
    TrainingRecord,
    TrainingSummary,
)
from steward.evaluation.runner import BenchmarkContext, GpuUnavailableError, run_candidate
from steward.evaluation.suite import CandidateConfig
from steward.generation.providers import create_provider
from steward.generation.providers.base import ProviderError, TextGenerationProvider
from steward.generation.retry import RetryPolicy
from steward.logs import log_event
from steward.provenance import git_state

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[CandidateConfig], TextGenerationProvider]
TRAINING_SUMMARY = "training_summary.json"
DEFAULT_EVAL_PROMPT = "ste_rewrite@1"


class Outcome(StrEnum):
    """Result of one benchmark request."""

    APPENDED = "appended"
    SKIPPED = "skipped"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RunOutcome:
    """What happened to one candidate."""

    candidate_id: str
    outcome: Outcome
    message: str
    record: BenchmarkRecord | None = None


def default_provider_factory(base_url: str | None) -> ProviderFactory:
    """Create providers for candidates (Ollama by default)."""

    def factory(candidate: CandidateConfig) -> TextGenerationProvider:
        kwargs: dict[str, Any] = {"timeout_s": 300.0}
        if base_url:
            kwargs["base_url"] = base_url
        if candidate.provider == "ollama" and candidate.think is not None:
            kwargs["think"] = candidate.think
        if candidate.provider == "transformers":
            kwargs = {
                "revision": candidate.base_revision,
                "adapter_path": candidate.adapter_path,
                "think": bool(candidate.think),
            }
        return create_provider(candidate.provider, **kwargs)

    return factory


def has_current_result(project: BenchmarkProject, candidate_id: str) -> bool:
    """True when a completed result exists for the current comparability key."""
    return any(
        r.candidate.candidate_id == candidate_id
        and r.status is RecordStatus.COMPLETED
        and r.comparability_key == project.key_for(_suite_of(project, candidate_id))
        for r in project.benchmarks()
    )


def _suite_of(project: BenchmarkProject, candidate_id: str) -> str:
    try:
        return project.registry.get(candidate_id).suite
    except Exception:
        return project.primary_suite_name


def run_benchmarks(
    project: BenchmarkProject,
    candidates: list[CandidateConfig],
    provider_factory: ProviderFactory,
    *,
    force: bool = False,
    require_gpu: bool = False,
    dataset_version: str | None = None,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] | None = None,
) -> list[RunOutcome]:
    """Benchmark each candidate and append real results to the history.

    Candidates with a current result are skipped unless ``force`` is set.
    A missing model or GPU is reported as ``unavailable``; nothing is
    appended. A run that breaks in the middle is appended as ``failed``.
    """
    outcomes = []
    for candidate in candidates:
        ctx = BenchmarkContext(
            root=project.paths.root,
            suite=project.suite_named(candidate.suite),
            rules=project.rules,
            prompt_dir=project.paths.prompts,
        )
        # Same candidate, same model artifact (adapter), same suite and evaluator.
        current = any(
            r.candidate.candidate_id == candidate.id
            and r.status is RecordStatus.COMPLETED
            and r.comparability_key == project.key_for(candidate.suite)
            and r.candidate.adapter_id == candidate.adapter_id
            for r in project.benchmarks()
        )
        if not force and current:
            outcomes.append(
                RunOutcome(
                    candidate.id,
                    Outcome.SKIPPED,
                    "A current comparable result exists (use --force to re-run).",
                )
            )
            continue
        provider = provider_factory(candidate)
        try:
            kwargs: dict[str, Any] = {
                "policy": policy,
                "require_gpu": require_gpu,
                "dataset_version": dataset_version,
            }
            if sleep is not None:
                kwargs["sleep"] = sleep
            record = run_candidate(provider, candidate, ctx, **kwargs)
        except (ProviderError, GpuUnavailableError) as exc:
            outcomes.append(RunOutcome(candidate.id, Outcome.UNAVAILABLE, str(exc)))
            log_event(
                logger,
                logging.WARNING,
                "benchmark_unavailable",
                candidate=candidate.id,
                error=str(exc),
            )
            continue
        finally:
            provider.close()
        stored, appended = append_record(project.paths.history, record)
        assert isinstance(stored, BenchmarkRecord)
        if candidate.training_run_id:
            write_benchmark_refs(project, candidate.training_run_id)
        if stored.status is RecordStatus.FAILED:
            outcomes.append(
                RunOutcome(candidate.id, Outcome.FAILED, stored.error or "failed", stored)
            )
        else:
            outcomes.append(
                RunOutcome(
                    candidate.id,
                    Outcome.APPENDED if appended else Outcome.SKIPPED,
                    f"{stored.experiment_id}" + ("" if appended else " (identical record exists)"),
                    stored,
                )
            )
    return outcomes


@dataclass(frozen=True)
class TrainingIngest:
    """A training summary found under ``runs/``."""

    path: Path
    record: TrainingRecord | None
    appended: bool
    candidate: CandidateConfig | None
    message: str


def find_training_summaries(project: BenchmarkProject) -> list[Path]:
    """All ``runs/*/training_summary.json`` files."""
    runs = project.paths.runs
    return sorted(runs.glob(f"*/{TRAINING_SUMMARY}")) if runs.is_dir() else []


BENCHMARK_REFS = "benchmark_refs.json"


def write_benchmark_refs(project: BenchmarkProject, run_id: str) -> Path | None:
    """Write ``runs/<run_id>/benchmark_refs.json``: the history records of this run.

    Rebuilt from the history each time, so it is always consistent with it.
    """
    run_dir = project.paths.runs / run_id
    if not run_dir.is_dir():
        return None
    refs = [
        {
            "experiment_id": r.experiment_id,
            "candidate_id": r.candidate.candidate_id,
            "status": r.status.value,
            "suite": f"{r.suite.suite_id}-v{r.suite.version}",
            "evaluator_version": r.evaluator.evaluator_version,
            "comparability_key": r.comparability_key,
            "created_at": r.created_at.isoformat(),
            "history": "benchmarks/results/history.jsonl",
        }
        for r in project.benchmarks()
        if r.candidate.training_run_id == run_id
    ]
    path = run_dir / BENCHMARK_REFS
    atomic_write_text(path, json.dumps({"run_id": run_id, "benchmarks": refs}, indent=2) + "\n")
    return path


def check_training_artifacts(summary: TrainingSummary, root: Path) -> list[str]:
    """Problems that stop a training run from being benchmarked (empty: OK).

    Checks the completion status, the serving spec, the adapter files, and
    that the adapter weights still match the hash in ``adapter_id``.
    """
    if summary.status is not RecordStatus.COMPLETED:
        return [f"training status is {summary.status.value}: {summary.error or 'no error text'}"]
    if summary.serving is None:
        return ["no serving spec (the run cannot be loaded for evaluation)"]
    problems = []
    if not summary.dataset_fingerprint:
        problems.append("no dataset fingerprint")
    adapter = summary.serving.adapter_path
    if adapter:
        path = Path(adapter) if Path(adapter).is_absolute() else root / adapter
        weights = path / "adapter_model.safetensors"
        for f in (path / "adapter_config.json", weights):
            if not f.is_file():
                problems.append(f"missing {f}")
        if weights.is_file() and summary.adapter_id:
            from steward.evaluation.fingerprints import file_sha256

            expected = summary.adapter_id.removeprefix("lora-")
            if not file_sha256(weights).startswith(expected):
                problems.append(f"{weights} does not match adapter_id {summary.adapter_id}")
    return problems


def candidate_from_training(summary: TrainingSummary) -> CandidateConfig | None:
    """A benchmark candidate for a served, completed training run."""
    if summary.status is not RecordStatus.COMPLETED or summary.serving is None:
        return None
    serving = summary.serving
    extra: dict[str, Any] = {}
    if serving.suite:
        extra["suite"] = serving.suite
    return CandidateConfig(
        id=f"ft-{summary.run_id}".lower().replace("_", "-"),
        role=CandidateRole.QUANTIZED if serving.quantization else CandidateRole.FINETUNED,
        provider=serving.provider,
        model=serving.model,
        base_model=summary.base_model,
        base_revision=serving.revision or summary.base_revision,
        adapter_path=serving.adapter_path,
        prompt=serving.prompt or DEFAULT_EVAL_PROMPT,
        **extra,
        think=False,
        parameter_count=summary.parameter_count,
        quantization=summary.serving.quantization or summary.quantization,
        checkpoint_id=summary.checkpoint_id,
        adapter_id=summary.adapter_id,
        training_run_id=summary.run_id,
        release_version=label
        if isinstance(label := summary.hyperparameters.get("version_label"), str)
        else None,
    )


def ingest_training_runs(
    project: BenchmarkProject, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
) -> list[TrainingIngest]:
    """Add finished training runs to the history (idempotent)."""
    results = []
    for path in find_training_summaries(project):
        try:
            summary = TrainingSummary.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            results.append(
                TrainingIngest(path, None, False, None, f"Invalid training summary: {exc}")
            )
            continue
        commit, dirty = git_state(project.paths.root)
        data = {
            "status": summary.status,
            "error": summary.error,
            "code": CodeRef(
                steward_version=__version__,
                git_commit=commit,
                git_dirty=dirty,
                worktree_fingerprint=worktree_fingerprint(project.paths.root),
            ),
            "hardware": describe_hardware(),
            "summary": summary,
        }
        # Hash only the summary so a re-scan never duplicates a run.
        problems = check_training_artifacts(summary, project.paths.root)
        candidate = None if problems else candidate_from_training(summary)
        note = f"; not benchmarkable: {'; '.join(problems)}" if problems else ""
        existing = [t for t in project.trainings() if t.summary == summary]
        if existing:
            results.append(
                TrainingIngest(path, existing[0], False, candidate, "already recorded" + note)
            )
            continue
        record = finalize_record(TrainingRecord, data, clock())
        assert isinstance(record, TrainingRecord)
        stored, appended = append_record(project.paths.history, record)
        assert isinstance(stored, TrainingRecord)
        results.append(
            TrainingIngest(path, stored, appended, candidate, stored.experiment_id + note)
        )
    return results


def regenerate(project: BenchmarkProject, *, figures: bool = True) -> list[Path]:
    """Write figures, the report, the summary, and the README section."""
    from steward.evaluation.plots import render_all
    from steward.evaluation.report import render_text_outputs

    written: list[Path] = []
    if figures:
        written += render_all(
            project.benchmarks(),
            project.trainings(),
            project.paths.figures,
            primary_key=project.key,
        )
    for path, content in render_text_outputs(project).items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            atomic_write_text(path, content)
            written.append(path)
    return written
