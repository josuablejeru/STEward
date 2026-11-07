import json
import re
from pathlib import Path

import numpy as np
import pytest

from bench_helpers import BenchFake, later, make_project, make_record
from steward.evaluation import hardware
from steward.evaluation.history import append_record
from steward.evaluation.plots import FIGURES, render_all
from steward.evaluation.project import BenchmarkProject
from steward.evaluation.records import CandidateRole, EffectiveStatus, RecordStatus
from steward.evaluation.report import README_END, README_START, update_readme
from steward.evaluation.service import regenerate, run_benchmarks
from steward.evaluation.status import pending_candidates, result_states
from steward.evaluation.verify import IssueLevel, verify
from steward.generation.retry import RetryPolicy


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BenchmarkProject:
    monkeypatch.setattr(hardware, "gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "gpu_memory_used_mib", lambda: None)
    return BenchmarkProject.at(make_project(tmp_path))


def test_states_mark_stale_and_failed() -> None:
    current = make_record("a")
    old_suite = make_record("b", suite_version=1)
    ft = make_record("ft", role=CandidateRole.FINETUNED, dataset_version="old-dataset")
    failed = make_record("c", status=RecordStatus.FAILED)
    states = {
        s.record.candidate.candidate_id: s
        for s in result_states([current, old_suite, ft, failed], current.comparability_key, "new")
    }
    assert states["a"].status is EffectiveStatus.COMPLETED
    assert states["b"].status is EffectiveStatus.STALE
    assert states["ft"].status is EffectiveStatus.STALE
    assert "Retrain" in states["ft"].reasons[0]
    assert states["c"].status is EffectiveStatus.FAILED


def test_pending_candidates(project: BenchmarkProject) -> None:
    assert pending_candidates(project.registry, [], project.key) == ["tiny-base", "tiny-prompted"]


def test_empty_history_figures_show_empty_state(project: BenchmarkProject, tmp_path: Path) -> None:
    out = tmp_path / "fig"
    paths = render_all([], [], out, formats=("svg",))
    assert len(paths) == len(FIGURES)
    text = (out / "ste_compliance_by_model.svg").read_text(encoding="utf-8")
    assert "<svg" in text
    assert not re.search(r"<path[^>]*clip-path=\"url\(#\w+\)\"", text)  # no data marks


def test_figures_are_deterministic(tmp_path: Path) -> None:
    records = [
        make_record("a"),
        make_record(
            "b", params=4_000_000_000, created=later(3), metrics={"ste_rule_pass_rate": 0.8}
        ),
    ]
    names = ("ste_compliance_by_model", "violations_over_time", "pareto_frontier")
    a = render_all(records, [], tmp_path / "a", formats=("svg", "png"), names=names)
    b = render_all(records, [], tmp_path / "b", formats=("svg", "png"), names=names)
    for x, y in zip(a, b, strict=True):
        assert x.read_bytes() == y.read_bytes(), x.name


def test_incompatible_versions_get_separate_series(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    from steward.evaluation.plots import fig_violations_over_time

    records = [
        make_record("a"),
        make_record("a", created=later(1), metrics={"violations_per_100_words": 4}),
        make_record("a", suite_version=1, created=later(2)),
    ]
    fig = fig_violations_over_time(records, [])
    lines = fig.axes[0].get_lines()
    assert sorted(np.atleast_1d(line.get_xdata()).size for line in lines) == [
        1,
        2,
    ]  # never joined across versions
    plt.close(fig)


def test_regenerate_and_verify_end_to_end(project: BenchmarkProject) -> None:
    fake = BenchFake()
    run_benchmarks(
        project,
        project.registry.candidates,
        lambda _: fake.provider(),
        policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
    )
    regenerate(project)
    report = project.paths.report.read_text(encoding="utf-8")
    assert "Draft suite" in report and "tiny-prompted" in report
    assert "Not a certification" in report
    summary = json.loads(project.paths.summary.read_text(encoding="utf-8"))
    assert set(summary["candidates"]) == {"tiny-base", "tiny-prompted"}
    readme = project.paths.readme.read_text(encoding="utf-8")
    assert README_START in readme and "docs/benchmarks.md" in readme
    issues = verify(project, generated=True)
    assert [i for i in issues if i.level is IssueLevel.ERROR] == []

    # A stale committed figure is detected.
    (project.paths.figures / "pareto_frontier.svg").write_text("<svg/>", encoding="utf-8")
    assert any(i.check == "generated" for i in verify(project, generated=True))


def test_verify_detects_broken_links_and_golden_leakage(project: BenchmarkProject) -> None:
    (project.paths.root / "docs" / "x.md").write_text("[gone](missing.md)\n", encoding="utf-8")
    golden = project.suite.items[0].source_text
    data = project.paths.root / "data" / "raw"
    data.mkdir(parents=True)
    from fakes import FakeOllama
    from steward.generation.config import load_config
    from steward.generation.pipeline import DatasetGenerator

    fake = FakeOllama()
    fake.source_text = lambda n: golden
    DatasetGenerator(
        provider=fake.provider(),
        model=fake.model,
        config=load_config(project.paths.root / "configs/generation/default.toml"),
        prompt_dir=project.paths.prompts,
        output=data / "ex.jsonl",
        sleep=lambda _: None,
    ).run(1, max_scenarios=1)
    checks = {i.check for i in verify(project) if i.level is IssueLevel.ERROR}
    assert "links" in checks
    assert "golden-leakage" in checks


def test_verify_reports_tampered_history(project: BenchmarkProject) -> None:
    append_record(project.paths.history, make_record("a"))
    text = project.paths.history.read_text(encoding="utf-8").replace('"cand', '"cand')
    text = text.replace('"value":0.5', '"value":0.9', 1)
    project.paths.history.write_text(text, encoding="utf-8")
    assert any(i.check == "history-integrity" for i in verify(project))


def test_update_readme_replaces_only_marked_block() -> None:
    text = f"intro\n\n{README_START}\nold\n{README_END}\n\noutro\n"
    new = update_readme(text, f"{README_START}\nnew\n{README_END}")
    assert "old" not in new and "new" in new and new.startswith("intro") and "outro" in new
    assert update_readme("x\n", f"{README_START}\ny\n{README_END}").endswith(f"{README_END}\n")


def test_evaluator_change_without_version_bump_warns(
    project: BenchmarkProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    from steward.evaluation.fingerprints import evaluator_fingerprint
    from steward.evaluation.status import evaluator_version_warnings

    fake = BenchFake()
    run_benchmarks(
        project,
        project.registry.candidates[:1],
        lambda _: fake.provider(),
        policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
    )
    records = project.benchmarks()
    assert records[0].evaluator.code_fingerprint == evaluator_fingerprint(project.paths.root)
    # Re-scoring the stored outputs with unchanged code gives the same scores: no warning.
    assert evaluator_version_warnings(project, records) == []
    # A scoring change without a version bump is detected by re-scoring.
    import steward.evaluation.metrics as metrics

    original = metrics.score_output

    def changed(*args: object, **kwargs: object) -> dict[str, object]:
        return original(*args, **kwargs) | {"facts_total": 999}  # type: ignore[arg-type]

    monkeypatch.setattr(metrics, "score_output", changed)
    warnings = evaluator_version_warnings(project, records)
    assert warnings and "Bump EVALUATOR_VERSION" in warnings[0]
