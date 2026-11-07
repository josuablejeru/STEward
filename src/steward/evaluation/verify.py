"""Checks behind ``steward benchmark verify`` (also used by CI and the Stop hook)."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from steward.datasets.jsonl import DatasetFileError, fingerprint, read_jsonl
from steward.datasets.schema import DatasetExample, RejectedExample
from steward.evaluation.history import HistoryError, read_history, verify_history
from steward.evaluation.project import BenchmarkProject
from steward.evaluation.records import BenchmarkRecord, RecordStatus, TrainingRecord
from steward.evaluation.suite import BenchmarkConfigError
from steward.generation.prompts import PromptTemplateError, load_template


class IssueLevel(StrEnum):
    """``error`` fails verification; ``warning`` is reported only."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """One verification finding."""

    level: IssueLevel
    check: str
    message: str


def _err(check: str, message: str) -> Issue:
    return Issue(IssueLevel.ERROR, check, message)


def _warn(check: str, message: str) -> Issue:
    return Issue(IssueLevel.WARNING, check, message)


def check_history(project: BenchmarkProject) -> list[Issue]:
    """Schema, hash chain, and reproducibility metadata of every record."""
    try:
        records = read_history(project.paths.history)
    except HistoryError as exc:
        return [_err("history-schema", str(exc))]
    issues = [_err("history-integrity", p) for p in verify_history(records)]
    for rec in records:
        where = rec.experiment_id
        if not rec.code.git_commit and not rec.code.worktree_fingerprint:
            issues.append(
                _err("reproducibility", f"{where}: no git commit or worktree fingerprint.")
            )
        if not rec.hardware.platform:
            issues.append(_err("reproducibility", f"{where}: no hardware platform."))
        if isinstance(rec, BenchmarkRecord) and rec.status is RecordStatus.COMPLETED:
            if not rec.candidate.model_digest and not rec.candidate.checkpoint_id:
                issues.append(
                    _err("reproducibility", f"{where}: no model digest or checkpoint ID.")
                )
            if rec.sampling.seed is None:
                issues.append(_err("reproducibility", f"{where}: no sampling seed."))
            if len(rec.items) != rec.suite.item_count:
                issues.append(
                    _err(
                        "history-schema",
                        f"{where}: {len(rec.items)} item results, suite has "
                        f"{rec.suite.item_count}.",
                    )
                )
            if not rec.metrics:
                issues.append(_err("history-schema", f"{where}: completed record has no metrics."))
            if rec.suite.review_state == "draft":
                issues.append(
                    _warn("suite-review", f"{where}: measured on a draft (unreviewed) suite.")
                )
        if isinstance(rec, TrainingRecord):
            s = rec.summary
            if not (s.dataset_fingerprint and s.config_sha256 and s.base_model):
                issues.append(
                    _err(
                        "reproducibility",
                        f"{where}: training record lacks dataset, config, or base model.",
                    )
                )
    return issues


def check_configuration(project: BenchmarkProject) -> list[Issue]:
    """The suite hash, registry, thresholds, rule config, and prompts."""
    issues: list[Issue] = []
    for name, loader in (
        ("golden-suite", lambda: project.suite),
        ("registry", lambda: project.registry),
        ("thresholds", lambda: project.thresholds),
    ):
        try:
            loader()
        except BenchmarkConfigError as exc:
            issues.append(_err(name, str(exc)))
    try:
        for suite_name in project.suite_names():
            try:
                project.suite_named(suite_name)
            except BenchmarkConfigError as exc:
                issues.append(_err("golden-suite", str(exc)))
    except BenchmarkConfigError:
        pass
    try:
        _ = project.rules
    except (OSError, ValueError) as exc:
        issues.append(_err("rules-config", str(exc)))
    try:
        for candidate in project.registry.candidates:
            try:
                load_template(project.paths.prompts, candidate.prompt)
            except PromptTemplateError as exc:
                issues.append(_err("registry", f"{candidate.id}: {exc}"))
    except BenchmarkConfigError:
        pass
    return issues


def _dataset_files(root: Path) -> list[Path]:
    data = root / "data"
    return sorted(data.rglob("*.jsonl")) if data.is_dir() else []


def _golden_index(project: BenchmarkProject) -> dict[str, str]:
    """Fingerprints of golden sources and references in every suite in use."""
    golden: dict[str, str] = {}
    try:
        names = project.suite_names()
    except BenchmarkConfigError:
        return golden
    for name in names:
        try:
            suite = project.suite_named(name)
        except BenchmarkConfigError:
            continue
        for item in suite.items:
            golden[fingerprint(item.source_text)] = f"{name}/{item.item_id}"
            if item.reference and item.task_type.value != "ste_identity":
                golden[fingerprint(item.reference)] = f"{name}/{item.item_id} (reference)"
    return golden


def _check_sft_file(path: Path, rel: Path, golden: dict[str, str]) -> list[Issue]:
    from steward.datasets.sft import SftRecord

    try:
        records = read_jsonl(path, SftRecord)
    except DatasetFileError as exc:
        return [_err("dataset-schema", str(exc))]
    issues = []
    for record in records:
        for text in (record.input, record.target):
            item = golden.get(fingerprint(text))
            if item:
                issues.append(
                    _err("golden-leakage", f"{rel}: {record.sft_id} matches golden item {item}.")
                )
    return issues


def check_datasets(project: BenchmarkProject) -> list[Issue]:
    """Schema of local dataset files and leakage of golden items into them."""
    issues: list[Issue] = []
    golden = _golden_index(project)
    for path in _dataset_files(project.paths.root):
        rel = path.relative_to(project.paths.root)
        rejected = path.name.endswith(".rejected.jsonl")
        if rel.parts[:2] == ("data", "sft"):
            issues += _check_sft_file(path, rel, golden)
            continue
        try:
            if rejected:
                read_jsonl(path, RejectedExample)
                continue
            examples = read_jsonl(path, DatasetExample)
        except DatasetFileError as exc:
            issues.append(_err("dataset-schema", str(exc)))
            continue
        ids = [e.example_id for e in examples]
        if len(ids) != len(set(ids)):
            issues.append(_err("dataset-duplicates", f"{rel}: duplicate example IDs."))
        for example in examples:
            for text in (example.source_text, example.ste_candidate):
                item = golden.get(fingerprint(text))
                if item:
                    issues.append(
                        _err(
                            "golden-leakage",
                            f"{rel}: {example.example_id} matches "
                            f"golden item {item}. Remove it from the dataset.",
                        )
                    )
    return issues


def check_prompt_leakage(project: BenchmarkProject) -> list[Issue]:
    """Golden source texts must not appear in prompt templates or generation configs."""
    items = []
    try:
        for name in project.suite_names():
            items += project.suite_named(name).items
    except BenchmarkConfigError:
        return []
    issues = []
    for folder in (project.paths.prompts, project.paths.root / "configs"):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.toml")):
            text = path.read_text(encoding="utf-8")
            for item in items:
                if item.source_text[:80] in text:
                    issues.append(
                        _err("golden-leakage", f"{path}: contains golden item {item.item_id}.")
                    )
    return issues


_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def check_links(project: BenchmarkProject) -> list[Issue]:
    """Relative links in README.md and docs/*.md must point to existing files."""
    root = project.paths.root
    docs = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    issues = []
    for doc in docs:
        if not doc.exists():
            continue
        for target in _LINK_RE.findall(doc.read_text(encoding="utf-8")):
            if re.match(r"^[a-z]+://|^#|^mailto:", target):
                continue
            path = (doc.parent / target.split("#")[0]).resolve()
            if not path.exists():
                issues.append(_err("links", f"{doc.relative_to(root)}: broken link {target}"))
    return issues


def check_generated(project: BenchmarkProject) -> list[Issue]:
    """Committed report, summary, README section, and figures match a fresh render."""
    from steward.evaluation.plots import render_all
    from steward.evaluation.report import render_text_outputs

    issues = []
    try:
        texts = render_text_outputs(project)
    except (BenchmarkConfigError, HistoryError) as exc:
        return [_err("generated", f"Cannot render outputs: {exc}")]
    for path, content in texts.items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != content:
            issues.append(
                _err(
                    "generated",
                    f"{path.relative_to(project.paths.root)} is out of date. "
                    "Run `pixi run benchmark-report`.",
                )
            )
    with tempfile.TemporaryDirectory() as tmp:
        fresh = render_all(
            project.benchmarks(),
            project.trainings(),
            Path(tmp),
            formats=("svg",),
            primary_key=project.key,
        )
        for new in fresh:
            committed = project.paths.figures / new.name
            if not committed.exists() or committed.read_bytes() != new.read_bytes():
                issues.append(
                    _err(
                        "generated",
                        f"benchmarks/figures/{new.name} differs from a fresh "
                        "render. Run `pixi run benchmark-plot`.",
                    )
                )
    return issues


def verify(
    project: BenchmarkProject, *, generated: bool = False, datasets: bool = True
) -> list[Issue]:
    """Run all checks. ``generated`` also re-renders figures (slower)."""
    issues = check_configuration(project) + check_history(project) + check_prompt_leakage(project)
    issues += check_links(project)
    if datasets:
        issues += check_datasets(project)
    if generated:
        issues += check_generated(project)
    return issues
