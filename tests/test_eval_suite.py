from pathlib import Path

import pytest

from bench_helpers import REPO, make_project
from steward.evaluation.project import BenchmarkProject
from steward.evaluation.suite import BenchmarkConfigError, load_registry, load_suite


def test_committed_suite_is_intact() -> None:
    suite = load_suite(REPO / "benchmarks/golden/ste-golden-v0")
    assert len(suite.items) == 16
    assert suite.manifest.review_state == "draft"
    assert {i.category_label for i in suite.items} == {
        "procedure",
        "description",
        "safety_notice",
        "troubleshooting",
    }
    assert suite.manifest.sampling.temperature == 0.0


def test_modified_golden_items_are_rejected(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    items = root / "benchmarks/golden/ste-golden-v0/items.jsonl"
    items.write_text(
        items.read_text(encoding="utf-8").replace("8 screws", "9 screws"), encoding="utf-8"
    )
    with pytest.raises(BenchmarkConfigError, match="immutable"):
        load_suite(items.parent)


def test_committed_registry_and_thresholds_load() -> None:
    project = BenchmarkProject.at(REPO)
    assert project.registry.candidates
    assert project.thresholds.selection.max_critical_failures == 0


def test_registry_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    entry = '[[candidates]]\nid="a"\nrole="base"\nmodel="m"\nbase_model="b"\nprompt="p@1"\n'
    path.write_text(entry * 2, encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="duplicate"):
        load_registry(path)
    project = BenchmarkProject.at(REPO)
    with pytest.raises(BenchmarkConfigError, match="Unknown candidate"):
        project.registry.get("nope")
