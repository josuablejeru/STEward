from bench_helpers import REPO, later, make_record
from steward.evaluation.compare import (
    FindingKind,
    compare,
    latest_by_candidate,
    references_for,
    select_model,
)
from steward.evaluation.records import CandidateRole, RecordStatus
from steward.evaluation.suite import load_thresholds

TH = load_thresholds(REPO / "benchmarks/thresholds.toml")


def kinds(comp) -> dict[str, FindingKind]:  # type: ignore[no-untyped-def]
    return {f.metric: f.kind for f in comp.findings}


def test_detects_each_regression_type() -> None:
    ref = make_record("ref")
    cand = make_record(
        "cand",
        metrics={
            "ste_rule_pass_rate": 0.40,
            "grammar_violations_per_100_words": 3.0,
            "meaning_preservation": 0.90,
            "hallucination_rate": 0.0625,
            "latency_mean_s": 1.5,
        },
    )
    k = kinds(compare(cand, ref, TH, "explicit"))
    for metric in (
        "ste_rule_pass_rate",
        "grammar_violations_per_100_words",
        "meaning_preservation",
        "hallucination_rate",
        "latency_mean_s",
    ):
        assert k[metric] is FindingKind.REGRESSION, metric


def test_improvement_and_significance() -> None:
    ref = make_record(
        "ref", ci={"ste_rule_pass_rate": (0.3, 0.5)}, metrics={"ste_rule_pass_rate": 0.4}
    )
    cand = make_record(
        "cand", ci={"ste_rule_pass_rate": (0.7, 0.9)}, metrics={"ste_rule_pass_rate": 0.8}
    )
    finding = next(
        f for f in compare(cand, ref, TH, "x").findings if f.metric == "ste_rule_pass_rate"
    )
    assert finding.kind is FindingKind.IMPROVEMENT and finding.significant


def test_larger_model_without_quality_gain() -> None:
    ref = make_record("small", disk=1_000_000_000)
    cand = make_record("big", disk=2_000_000_000, metrics={"ste_rule_pass_rate": 0.51})
    assert kinds(compare(cand, ref, TH, "x"))["disk_size_bytes"] is FindingKind.REGRESSION
    better = make_record("big2", disk=2_000_000_000, metrics={"ste_rule_pass_rate": 0.60})
    assert kinds(compare(better, ref, TH, "x"))["disk_size_bytes"] is FindingKind.UNCHANGED


def test_incompatible_versions_are_not_compared() -> None:
    comp = compare(make_record("a", suite_version=1), make_record("b"), TH, "x")
    assert not comp.comparable and comp.findings == []


def test_critical_failures_always_flagged() -> None:
    comp = compare(
        make_record("a", metrics={"critical_failures": 2.0}),
        make_record("b", suite_version=1),
        TH,
        "x",
    )
    assert [f.kind for f in comp.findings] == [FindingKind.CRITICAL]


def test_references_cover_base_release_and_smaller() -> None:
    base = make_record("base", role=CandidateRole.BASE, params=4_000_000_000)
    release = make_record("rel", role=CandidateRole.RELEASE, params=4_000_000_000)
    small = make_record("small", params=1_000_000_000, metrics={"ste_rule_pass_rate": 0.9})
    cand = make_record("ft", role=CandidateRole.FINETUNED, params=4_000_000_000, created=later(10))
    latest = latest_by_candidate([base, release, small, cand])
    refs = {kind: r.candidate.candidate_id for kind, r in references_for(cand, latest)}
    assert refs == {
        "untuned base": "base",
        "previous release": "rel",
        "best of equal or smaller size": "small",
    }


def test_selection_picks_smallest_passing() -> None:
    good = {
        "ste_rule_pass_rate": 0.9,
        "meaning_preservation": 0.99,
        "missing_information_rate": 0.0,
    }
    big = make_record("big", params=8_000_000_000, metrics=good)
    small = make_record("small", params=2_000_000_000, metrics=good)
    tiny_bad = make_record("tiny", params=500_000_000)
    selected, failures = select_model(latest_by_candidate([big, small, tiny_bad]), TH.selection)
    assert selected is not None and selected.candidate.candidate_id == "small"
    assert failures["tiny"] and not failures["big"]


def test_failed_records_are_ignored_for_latest() -> None:
    ok = make_record("a")
    failed = make_record("a", status=RecordStatus.FAILED, created=later(5))
    assert latest_by_candidate([ok, failed])["a"] is ok
