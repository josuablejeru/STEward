"""The validation and benchmark workflow that the Stop hook runs.

Steps depend on which categories changed. Expensive training never runs
here. Model benchmarks run only for *new* trained artifacts (and, in
``full`` mode, for registered candidates without current results).
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from steward.automation.changes import ChangeCategory, category_paths
from steward.automation.state import GUARD_ENV
from steward.evaluation.compare import compare, latest_by_candidate, references_for
from steward.evaluation.fingerprints import dataset_fingerprint, tree_fingerprint
from steward.evaluation.history import HistoryError
from steward.evaluation.project import BenchmarkProject
from steward.evaluation.records import CandidateRole, RecordStatus
from steward.evaluation.service import (
    Outcome,
    ProviderFactory,
    ingest_training_runs,
    regenerate,
    run_benchmarks,
)
from steward.evaluation.status import evaluator_version_warnings, pending_candidates
from steward.evaluation.suite import BenchmarkConfigError
from steward.evaluation.verify import (
    IssueLevel,
    check_configuration,
    check_datasets,
    check_generated,
    check_history,
)

CommandRunner = Callable[[Sequence[str], Path], tuple[int, str]]

CODE_SUFFIXES = (".py", ".toml", ".json", ".yaml", ".yml")


def subprocess_runner(timeout_s: float = 1800) -> CommandRunner:
    """Run commands with the reentrancy guard set, returning ``(code, output tail)``."""

    def run(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
        env = {**os.environ, GUARD_ENV: "1"}
        try:
            proc = subprocess.run(
                list(cmd), cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            return 124, f"Timed out after {timeout_s:.0f} s: {' '.join(cmd)}"
        except OSError as exc:
            return 127, f"Cannot run {' '.join(cmd)}: {exc}"
        return proc.returncode, _diagnostics(proc.stdout + "\n" + proc.stderr)

    return run


_FAILURE_LINE = re.compile(r"FAILED|ERROR|Error|error:|failed|^E\s|-->|Found \d+ error")


def _diagnostics(output: str, limit: int = 25) -> str:
    """Failure lines first (test names, error codes, locations), then the last lines."""
    lines = [line for line in output.strip().splitlines() if line.strip()]
    key = [line for line in lines if _FAILURE_LINE.search(line)]
    tail = [line for line in lines[-5:] if line not in key]
    return "\n".join((key[:limit] + tail)[: limit + 5])


@dataclass
class Step:
    """One workflow step."""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class WorkflowReport:
    """Everything the Stop hook reports."""

    steps: list[Step] = field(default_factory=list)
    outstanding: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    fingerprints: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when every step passed."""
        return all(s.ok for s in self.steps)

    def failures(self) -> list[Step]:
        """Failed steps."""
        return [s for s in self.steps if not s.ok]


def category_fingerprints(
    root: Path, categories: set[ChangeCategory], cache: dict[str, list[object]]
) -> dict[str, str]:
    """Content fingerprint for each category (used to skip unchanged inputs)."""
    return {
        cat.value: tree_fingerprint(root, category_paths(cat), cache=cache)
        for cat in sorted(categories)
    }


def _code_changed(paths: list[str]) -> bool:
    return any(
        p.startswith(("src/", "configs/", "prompts/", "benchmarks/")) and p.endswith(CODE_SUFFIXES)
        for p in paths
    )


def run_workflow(
    root: Path,
    pending: dict[str, ChangeCategory],
    *,
    mode: str,
    runner: CommandRunner,
    provider_factory: ProviderFactory,
    cache: dict[str, list[object]] | None = None,
) -> WorkflowReport:
    """Run the steps required by ``pending`` changes."""
    report = WorkflowReport()
    categories = set(pending.values())
    paths = sorted(pending)
    project = BenchmarkProject.at(root)
    history_before = _history_len(project)

    # 1. Pixi validation tasks for code, config, and prompt changes.
    if _code_changed(paths):
        for task in ("lint", "typecheck", "test"):
            code, out = runner(["pixi", "run", "--frozen", task], root)
            report.steps.append(Step(f"pixi run {task}", code == 0, "" if code == 0 else out))

    # 2. Dataset integrity and schema compatibility.
    if ChangeCategory.DATA in categories:
        issues = [i for i in check_datasets(project) if i.level is IssueLevel.ERROR]
        report.steps.append(
            Step("dataset integrity", not issues, "\n".join(i.message for i in issues[:20]))
        )
        current = dataset_fingerprint(root, cache)
        if current:
            trained = [
                r
                for r in project.benchmarks()
                if r.candidate.role in {CandidateRole.FINETUNED, CandidateRole.QUANTIZED}
                and r.status is RecordStatus.COMPLETED
            ]
            stale = [r for r in trained if r.dataset_version != current]
            if stale:
                report.outstanding.append(
                    f"Dataset changed (fingerprint {current[:12]}). "
                    f"{len(stale)} fine-tuned result(s) "
                    "were trained on an older revision and are now stale. Retrain and re-evaluate. "
                    "No new scores were assigned."
                )
            else:
                report.outstanding.append(
                    f"Dataset revision {current[:12]} has no trained model yet. Training and "
                    "evaluation are required before any performance claim."
                )

    # 3. Evaluation definitions: suite, registry, history, version bumps.
    if categories & {ChangeCategory.EVALUATION, ChangeCategory.MODEL}:
        issues = [
            i
            for i in check_configuration(project) + check_history(project)
            if i.level is IssueLevel.ERROR
        ]
        report.steps.append(
            Step(
                "benchmark definitions and history",
                not issues,
                "\n".join(i.message for i in issues[:20]),
            )
        )
        try:
            report.outstanding += evaluator_version_warnings(project, project.benchmarks())
            old = [
                r
                for r in project.benchmarks()
                if r.status is RecordStatus.COMPLETED
                and r.comparability_key not in project.keys.values()
            ]
            if old:
                report.outstanding.append(
                    f"{len(old)} historical result(s) use another suite or evaluator version. They "
                    "are preserved but are not directly comparable with new results."
                )
        except (BenchmarkConfigError, HistoryError) as exc:
            report.steps.append(Step("evaluator comparability", False, str(exc)))

    # 4. Training changes without a new checkpoint.
    ingests = []
    if categories & {ChangeCategory.TRAINING, ChangeCategory.MODEL}:
        try:
            ingests = ingest_training_runs(project)
        except HistoryError as exc:
            report.steps.append(Step("ingest training runs", False, str(exc)))
        bad = [i for i in ingests if i.record is None]
        if bad:
            report.steps.append(
                Step("training summaries", False, "\n".join(f"{i.path}: {i.message}" for i in bad))
            )
        new_runs = [i for i in ingests if i.appended]
        if ChangeCategory.TRAINING in categories and not new_runs:
            report.outstanding.append(
                "Training code or configuration changed, but no new checkpoint exists. The model "
                "benchmark is pending until a training run is completed."
            )

    # 5. Model benchmarks for new trained artifacts (never starts training).
    to_run = [i.candidate for i in ingests if i.appended and i.candidate is not None]
    for ingest in ingests:
        if ingest.appended and ingest.candidate is None and ingest.record is not None:
            report.outstanding.append(
                f"Training run {ingest.record.summary.run_id} was recorded but cannot be "
                f"benchmarked ({ingest.message}). Its evaluation is pending; fix the run or "
                "serve it, then run `pixi run -e cuda benchmark-gpu`."
            )
    if mode == "full" and categories & {ChangeCategory.MODEL, ChangeCategory.EVALUATION}:
        try:
            missing = pending_candidates(project.registry, project.benchmarks(), project.keys)
            to_run += [project.registry.get(cid) for cid in missing]
        except (BenchmarkConfigError, HistoryError):
            pass
    if to_run:
        if mode == "light":
            report.outstanding += [
                f"Benchmark pending for {c.id} (light mode skips model runs)." for c in to_run
            ]
        else:
            outcomes = run_benchmarks(project, to_run, provider_factory, require_gpu=False)
            for o in outcomes:
                if o.outcome is Outcome.UNAVAILABLE:
                    report.outstanding.append(
                        f"Benchmark pending for {o.candidate_id}: {o.message}"
                    )
                elif o.outcome is Outcome.FAILED:
                    report.steps.append(Step(f"benchmark {o.candidate_id}", False, o.message))
                else:
                    report.steps.append(Step(f"benchmark {o.candidate_id}", True, o.message))
    elif categories & {ChangeCategory.MODEL, ChangeCategory.EVALUATION}:
        try:
            for cid in pending_candidates(project.registry, project.benchmarks(), project.keys):
                report.outstanding.append(
                    f"No current result for registered candidate {cid}; run `pixi run benchmark`."
                )
        except (BenchmarkConfigError, HistoryError):
            pass

    # 6. Summaries, figures, report, README.
    history_changed = _history_len(project) != history_before
    if history_changed or categories & {ChangeCategory.EVALUATION, ChangeCategory.MODEL}:
        try:
            regenerate(project)
            issues = check_generated(project)
            report.steps.append(
                Step(
                    "regenerate figures and report",
                    not issues,
                    "\n".join(i.message for i in issues),
                )
            )
        except (BenchmarkConfigError, HistoryError, OSError, ValueError) as exc:
            report.steps.append(Step("regenerate figures and report", False, str(exc)))

    # 7. Improvements and regressions for records added in this run.
    if history_changed:
        try:
            _summarize_changes(project, history_before, report)
        except (BenchmarkConfigError, HistoryError) as exc:
            report.steps.append(Step("regression check", False, str(exc)))
    return report


def _history_len(project: BenchmarkProject) -> int:
    try:
        return len(project.history())
    except HistoryError:
        return -1


def _summarize_changes(project: BenchmarkProject, before: int, report: WorkflowReport) -> None:
    new = [r for r in project.history()[max(before, 0) :]]
    latest = latest_by_candidate(project.benchmarks(), project.key)
    for rec in new:
        if rec.record_type != "benchmark" or rec.status is not RecordStatus.COMPLETED:
            continue
        assert rec.record_type == "benchmark"
        for kind, ref in references_for(rec, latest):
            comp = compare(rec, ref, project.thresholds, kind)
            name = f"{comp.candidate.candidate.candidate_id} vs {kind} {ref.candidate.candidate_id}"
            report.regressions += [f"{name}: {f.metric}: {f.message}" for f in comp.regressions]
            report.improvements += [f"{name}: {f.metric}: {f.message}" for f in comp.improvements]
