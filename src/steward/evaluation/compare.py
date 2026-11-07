"""Regression detection and model selection.

Only results with the same comparability key (suite version and hash,
evaluator version, rule-set version) are compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from steward.evaluation.records import BenchmarkRecord, CandidateRole, Metric, RecordStatus
from steward.evaluation.suite import Bound, Limit, SelectionRequirements, Thresholds


class FindingKind(StrEnum):
    """Classification of one metric change."""

    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    UNCHANGED = "unchanged"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Finding:
    """One metric compared with a reference."""

    metric: str
    kind: FindingKind
    candidate: float | None
    reference: float | None
    message: str
    significant: bool | None = None


@dataclass
class Comparison:
    """A candidate result compared with one reference result."""

    candidate: BenchmarkRecord
    reference: BenchmarkRecord
    reference_kind: str
    comparable: bool
    findings: list[Finding] = field(default_factory=list)

    @property
    def regressions(self) -> list[Finding]:
        """Regression and critical findings."""
        return [
            f for f in self.findings if f.kind in {FindingKind.REGRESSION, FindingKind.CRITICAL}
        ]

    @property
    def improvements(self) -> list[Finding]:
        """Improvement findings."""
        return [f for f in self.findings if f.kind is FindingKind.IMPROVEMENT]


def _value(record: BenchmarkRecord, name: str) -> float | None:
    metric = record.metrics.get(name)
    return metric.value if metric else None


def _ci_separated(a: Metric | None, b: Metric | None) -> bool | None:
    if a is None or b is None or None in (a.ci_low, a.ci_high, b.ci_low, b.ci_high):
        return None
    assert a.ci_low is not None and a.ci_high is not None
    assert b.ci_low is not None and b.ci_high is not None
    return a.ci_high < b.ci_low or b.ci_high < a.ci_low


def _check(name: str, limit: Limit, cand: BenchmarkRecord, ref: BenchmarkRecord) -> Finding | None:
    c, r = _value(cand, name), _value(ref, name)
    if c is None or r is None:
        return None
    delta = c - r
    significant = _ci_separated(cand.metrics.get(name), ref.metrics.get(name))
    if name == "disk_size_bytes":
        ratio = c / r - 1 if r > 0 else 0.0
        gain = (_value(cand, "ste_rule_pass_rate") or 0.0) - (
            _value(ref, "ste_rule_pass_rate") or 0.0
        )
        if (
            limit.max_increase_ratio is not None
            and ratio > limit.max_increase_ratio
            and (limit.min_quality_gain is None or gain < limit.min_quality_gain)
        ):
            return Finding(
                name,
                FindingKind.REGRESSION,
                c,
                r,
                f"{ratio:+.0%} larger on disk, but STE pass rate changed only {gain:+.3f} "
                f"(needs at least {limit.min_quality_gain:+.3f}).",
            )
        return Finding(name, FindingKind.UNCHANGED, c, r, f"Size change {ratio:+.0%}.")
    if limit.max_drop is not None:
        if delta < -limit.max_drop:
            return Finding(
                name,
                FindingKind.REGRESSION,
                c,
                r,
                f"Dropped by {-delta:.3f} (limit {limit.max_drop}).",
                significant,
            )
        if delta > limit.max_drop:
            return Finding(
                name, FindingKind.IMPROVEMENT, c, r, f"Improved by {delta:.3f}.", significant
            )
    elif limit.max_increase is not None:
        if delta > limit.max_increase:
            return Finding(
                name,
                FindingKind.REGRESSION,
                c,
                r,
                f"Increased by {delta:.3f} (limit {limit.max_increase}).",
                significant,
            )
        if delta < -max(limit.max_increase, 1e-9):
            return Finding(
                name, FindingKind.IMPROVEMENT, c, r, f"Reduced by {-delta:.3f}.", significant
            )
    elif limit.max_increase_ratio is not None and r > 0:
        ratio = c / r - 1
        if ratio > limit.max_increase_ratio:
            return Finding(
                name,
                FindingKind.REGRESSION,
                c,
                r,
                f"{ratio:+.0%} slower (limit {limit.max_increase_ratio:+.0%}).",
            )
        if ratio < -limit.max_increase_ratio:
            return Finding(name, FindingKind.IMPROVEMENT, c, r, f"{ratio:+.0%} faster.")
    return Finding(name, FindingKind.UNCHANGED, c, r, f"Change {delta:+.3f} is within the limit.")


def compare(
    candidate: BenchmarkRecord, reference: BenchmarkRecord, thresholds: Thresholds, kind: str
) -> Comparison:
    """Compare two records. Incomparable records produce no metric findings."""
    comparable = candidate.comparability_key == reference.comparability_key
    comparison = Comparison(candidate, reference, kind, comparable)
    critical = _value(candidate, "critical_failures") or 0.0
    if critical > 0:
        comparison.findings.append(
            Finding(
                "critical_failures",
                FindingKind.CRITICAL,
                critical,
                _value(reference, "critical_failures"),
                f"{critical:.0f} critical safety failure(s): lost WARNING/CAUTION labels or "
                "missing/invented facts in safety-critical items. Human review is required.",
            )
        )
    if not comparable:
        return comparison
    for name, limit in thresholds.regression.items():
        finding = _check(name, limit, candidate, reference)
        if finding is not None:
            comparison.findings.append(finding)
    return comparison


def latest_by_candidate(
    records: list[BenchmarkRecord], key: str | None = None
) -> dict[str, BenchmarkRecord]:
    """Latest completed record per candidate (optionally only with comparability ``key``)."""
    latest: dict[str, BenchmarkRecord] = {}
    for record in sorted(records, key=lambda r: r.created_at):
        if record.status is not RecordStatus.COMPLETED:
            continue
        if key is not None and record.comparability_key != key:
            continue
        latest[record.candidate.candidate_id] = record
    return latest


def _quality(record: BenchmarkRecord) -> tuple[float, float]:
    return (
        _value(record, "ste_rule_pass_rate") or 0.0,
        _value(record, "meaning_preservation") or 0.0,
    )


def references_for(
    candidate: BenchmarkRecord, latest: dict[str, BenchmarkRecord]
) -> list[tuple[str, BenchmarkRecord]]:
    """Return references: untuned base, previous release, best of equal or smaller size."""
    others = [
        r for r in latest.values() if r.candidate.candidate_id != candidate.candidate.candidate_id
    ]
    refs: list[tuple[str, BenchmarkRecord]] = []
    if candidate.candidate.role is not CandidateRole.BASE:
        bases = [
            r
            for r in others
            if r.candidate.role is CandidateRole.BASE
            and r.candidate.base_model == candidate.candidate.base_model
        ]
        if bases:
            refs.append(("untuned base", max(bases, key=lambda r: r.created_at)))
    releases = [
        r
        for r in others
        if r.candidate.role is CandidateRole.RELEASE and r.created_at <= candidate.created_at
    ]
    if releases:
        refs.append(("previous release", max(releases, key=lambda r: r.created_at)))
    size = candidate.candidate.parameter_count
    if size is not None:
        smaller = [
            r
            for r in others
            if r.candidate.parameter_count is not None and r.candidate.parameter_count <= size
        ]
        if smaller:
            best = max(smaller, key=_quality)
            if all(best is not r for _, r in refs):
                refs.append(("best of equal or smaller size", best))
    return refs


def requirement_failures(record: BenchmarkRecord, req: SelectionRequirements) -> list[str]:
    """Return the documented requirements that ``record`` does not meet."""
    checks = [
        ("ste_rule_pass_rate", req.min_ste_rule_pass_rate, True),
        ("meaning_preservation", req.min_meaning_preservation, True),
        ("hallucination_rate", req.max_hallucination_rate, False),
        ("missing_information_rate", req.max_missing_information_rate, False),
        ("critical_failures", float(req.max_critical_failures), False),
        ("task_completion_rate", req.min_task_completion_rate, True),
    ]
    failures = []
    for name, bound, is_min in checks:
        value = _value(record, name)
        if value is None:
            failures.append(f"{name} is not available")
        elif (is_min and value < bound) or (not is_min and value > bound):
            failures.append(f"{name} = {value:.3f} ({'min' if is_min else 'max'} {bound})")
    return failures


def bound_failures(record: BenchmarkRecord, bounds: dict[str, Bound]) -> list[str]:
    """Return the metric bounds that ``record`` does not meet."""
    failures = []
    for name, bound in bounds.items():
        value = _value(record, name)
        if value is None:
            failures.append(f"{name} is not available")
            continue
        if bound.min is not None and value < bound.min:
            failures.append(f"{name} = {value:.3f} (min {bound.min})")
        if bound.max is not None and value > bound.max:
            failures.append(f"{name} = {value:.3f} (max {bound.max})")
    return failures


def select_model(
    latest: dict[str, BenchmarkRecord], req: SelectionRequirements | dict[str, Bound]
) -> tuple[BenchmarkRecord | None, dict[str, list[str]]]:
    """Return the smallest candidate that meets every requirement, and why others fail.

    Every requirement must hold first (safety, STE-only behavior, rules, meaning);
    only then does size decide. A smaller model never wins by size alone.
    """
    failures = {
        cid: bound_failures(r, req) if isinstance(req, dict) else requirement_failures(r, req)
        for cid, r in latest.items()
    }
    passing = [r for cid, r in latest.items() if not failures[cid]]
    if not passing:
        return None, failures
    selected = min(
        passing,
        key=lambda r: (
            r.candidate.parameter_count or float("inf"),
            r.candidate.disk_size_bytes or float("inf"),
            tuple(-x for x in _quality(r)),
        ),
    )
    return selected, failures
