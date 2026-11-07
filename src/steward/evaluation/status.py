"""Derived result states: stale results and pending work.

Stored records are never changed. Instead, this module compares them with
the current suite, evaluator, dataset, and candidate registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from steward.evaluation.metrics import evaluator_version
from steward.evaluation.records import (
    BenchmarkRecord,
    CandidateRole,
    EffectiveStatus,
    RecordStatus,
    comparability_key,
)
from steward.evaluation.suite import CandidateRegistry, Suite
from steward.ste.rules import RULESET_VERSION, RuleConfig

if TYPE_CHECKING:
    from steward.evaluation.project import BenchmarkProject


def current_key(suite: Suite, rules: RuleConfig) -> str:
    """The comparability key that new results would get."""
    return comparability_key(
        suite.ref, evaluator_version(suite.manifest.contract), RULESET_VERSION, rules.source_sha256
    )


@dataclass
class ResultState:
    """A stored record and its derived state."""

    record: BenchmarkRecord
    status: EffectiveStatus
    reasons: list[str] = field(default_factory=list)


def result_states(
    records: list[BenchmarkRecord], key: str | Mapping[str, str], dataset_version: str | None
) -> list[ResultState]:
    """Return the effective status of every benchmark record.

    ``key`` is the current key, or a mapping of suite name to current key.
    """
    current = {key} if isinstance(key, str) else set(key.values())
    states = []
    for record in records:
        if record.status is RecordStatus.FAILED:
            states.append(ResultState(record, EffectiveStatus.FAILED, [record.error or "failed"]))
            continue
        reasons = []
        if record.comparability_key not in current:
            reasons.append(
                "Evaluated with a different suite or evaluator version; "
                "not directly comparable with new results."
            )
        trained = record.candidate.role in {CandidateRole.FINETUNED, CandidateRole.QUANTIZED}
        if trained and dataset_version and record.dataset_version not in {None, dataset_version}:
            reasons.append(
                f"Trained on dataset {str(record.dataset_version)[:12]}; the current dataset is "
                f"{dataset_version[:12]}. Retrain and re-evaluate."
            )
        status = EffectiveStatus.STALE if reasons else EffectiveStatus.COMPLETED
        states.append(ResultState(record, status, reasons))
    return states


def pending_candidates(
    registry: CandidateRegistry, records: list[BenchmarkRecord], key: str | Mapping[str, str]
) -> list[str]:
    """Registered candidates without a completed result for their suite's current key.

    ``key`` is one key for every candidate, or a mapping of suite name to key.
    """
    done = {
        (r.candidate.candidate_id, r.comparability_key)
        for r in records
        if r.status is RecordStatus.COMPLETED
    }
    pending = []
    for c in registry.candidates:
        wanted = key if isinstance(key, str) else key.get(c.suite)
        if wanted is None or (c.id, wanted) not in done:
            pending.append(c.id)
    return pending


_COMPARED_FIELDS = (
    "rule_check_passed",
    "facts_total",
    "facts_preserved",
    "missing_facts",
    "invented_numbers",
    "negation_lost",
    "safety_labels_missing",
    "critical_failure",
    "ste_only",
    "guard_status",
    "controlled_failure",
    "unchanged",
    "chrf",
)


def rescore_item(item: object, output: str, rules: RuleConfig, contract: str) -> dict[str, object]:
    """Score a stored output again with the current code."""
    from steward.evaluation.metrics import score_output
    from steward.evaluation.ste_only import score_ste_only
    from steward.evaluation.suite import SuiteItem

    assert isinstance(item, SuiteItem)
    if contract == "ste_only_text":
        return score_ste_only(item, output, rules)
    return score_output(item, output, rules)


def evaluator_version_warnings(
    project: BenchmarkProject, records: list[BenchmarkRecord]
) -> list[str]:
    """Warn when current scoring code gives different scores without a version bump.

    For the latest completed record of each comparability key with the current
    evaluator and ruleset versions, every stored output is scored again. Any
    difference means the scoring changed and a version must be bumped.
    """
    from steward.evaluation.suite import BenchmarkConfigError

    warnings: list[str] = []
    latest: dict[str, BenchmarkRecord] = {}
    for r in sorted(records, key=lambda r: r.created_at):
        if r.status is RecordStatus.COMPLETED:
            latest[r.comparability_key] = r
    suites_by_id = {}
    for name in project.suite_names():
        try:
            suite = project.suite_named(name)
        except BenchmarkConfigError:
            continue
        suites_by_id[(suite.manifest.id, suite.manifest.version, suite.sha256)] = suite
    for record in latest.values():
        found = suites_by_id.get((record.suite.suite_id, record.suite.version, record.suite.sha256))
        if found is None:
            continue
        suite = found
        contract = suite.manifest.contract
        if (
            record.evaluator.evaluator_version != evaluator_version(contract)
            or record.evaluator.ruleset_version != RULESET_VERSION
            or record.evaluator.rules_config_sha256 != project.rules.source_sha256
        ):
            continue
        items = {i.item_id: i for i in suite.items}
        changed = []
        for stored in record.items:
            if not stored.valid_output or stored.output is None or stored.item_id not in items:
                continue
            fresh = rescore_item(items[stored.item_id], stored.output, project.rules, contract)
            old = stored.model_dump()
            diffs = [
                f for f in _COMPARED_FIELDS if f in fresh and _norm(fresh[f]) != _norm(old.get(f))
            ]
            if len(fresh.get("violations", [])) != len(stored.violations):  # type: ignore[arg-type]
                diffs.append("violations")
            if diffs:
                changed.append(f"{stored.item_id} ({', '.join(diffs)})")
        if changed:
            warnings.append(
                f"Scoring code changed: re-scoring {record.experiment_id} "
                f"({record.candidate.candidate_id}) gives different results for "
                f"{len(changed)} item(s), e.g. {changed[0]}. Bump "
                + (
                    "STE_ONLY_EVALUATOR_VERSION"
                    if contract == "ste_only_text"
                    else "EVALUATOR_VERSION"
                )
                + " or RULESET_VERSION so old and new results are not mixed."
            )
    return warnings


def _norm(value: object) -> object:
    if hasattr(value, "value"):
        return value.value  # enums
    if isinstance(value, float):
        return round(value, 3)
    return value
