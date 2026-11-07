from __future__ import annotations

from pathlib import Path

import pytest

from ste_helpers import ScriptedProvider
from steward.datasets.schema import TaskType
from steward.evaluation.project import BenchmarkProject
from steward.evaluation.records import CandidateRole
from steward.evaluation.runner import BenchmarkContext, run_candidate
from steward.evaluation.ste_only import (
    STE_ONLY_EVALUATOR_VERSION,
    chrf,
    edit_similarity,
    score_ste_only,
)
from steward.evaluation.suite import CandidateConfig, load_suite
from steward.ste.rules import load_rule_config

REPO = Path(__file__).resolve().parents[1]
SUITE = load_suite(REPO / "benchmarks/golden/ste-multilingual-v0")
RULES = load_rule_config(REPO / "configs/ste/terms.toml")


def test_committed_multilingual_suite() -> None:
    assert SUITE.manifest.contract == "ste_only_text"
    assert SUITE.manifest.review_state == "draft"
    langs = {i.source_language for i in SUITE.items}
    assert langs == {"en", "de", "fr", "ro"}
    tasks = {i.task_type for i in SUITE.items}
    assert tasks == set(TaskType)
    assert sum(i.expected_unsupported for i in SUITE.items) == 6


def test_reference_texts_pass_the_scoring() -> None:
    for item in SUITE.items:
        if item.reference is None:
            continue
        scored = score_ste_only(item, item.reference, RULES)
        assert scored["ste_only"], item.item_id
        assert not scored["critical_failure"], item.item_id
        assert scored["facts_preserved"] == scored["facts_total"], item.item_id
        assert scored["chrf"] == pytest.approx(100.0)


def test_chrf_and_similarity() -> None:
    assert chrf("Remove the cover.", "Remove the cover.") == pytest.approx(100.0)
    assert 0 < chrf("Remove the lid.", "Remove the cover.") < 100
    assert chrf("", "x") == 0.0
    assert edit_similarity("A b c.", "a B c.") == 1.0


def _item(item_id: str):  # type: ignore[no-untyped-def]
    return next(i for i in SUITE.items if i.item_id == item_id)


def test_copied_input_is_not_an_ste_pass() -> None:
    item = _item("req-en-02")
    scored = score_ste_only(item, item.source_text, RULES)
    assert scored["unchanged"] is True
    assert scored["rule_check_passed"] is False


def test_out_of_domain_scoring() -> None:
    item = _item("ood-en-01")
    assert score_ste_only(item, "STEWARD_UNSUPPORTED_INPUT", RULES)["controlled_failure"]
    assert not score_ste_only(item, "Mountains are nice.", RULES)["controlled_failure"]


def test_false_refusal_on_safety_item_is_critical() -> None:
    scored = score_ste_only(_item("de-safe-01"), "STEWARD_UNSUPPORTED_INPUT", RULES)
    assert scored["critical_failure"]
    assert scored["facts_preserved"] == 0


def test_run_candidate_with_reference_answers(tmp_path: Path) -> None:
    """A provider that returns the reference text gets perfect contract metrics."""
    refs = {i.source_text: i.reference for i in SUITE.items}

    def reply(source: str) -> str:
        ref = refs.get(source)
        return ref if ref else "STEWARD_UNSUPPORTED_INPUT"

    candidate = CandidateConfig(
        id="oracle",
        role=CandidateRole.FINETUNED,
        provider="scripted",
        model="m",
        base_model="b",
        prompt="ste_only@1",
        suite="ste-multilingual-v0",
    )
    ctx = BenchmarkContext(root=REPO, suite=SUITE, rules=RULES, prompt_dir=REPO / "prompts")
    record = run_candidate(ScriptedProvider(reply), candidate, ctx, sleep=lambda _: None)
    m = record.metrics
    assert record.evaluator.evaluator_version == STE_ONLY_EVALUATOR_VERSION
    assert m["ste_only_output_rate"].value == 1.0
    assert m["controlled_failure_reliability"].value == 1.0
    assert m["false_refusal_rate"].value == 0.0
    assert m["unnecessary_rewrite_rate"].value == 0.0
    assert m["critical_failures"].value == 0.0
    assert m["meaning_preservation"].value == 1.0
    assert "lang:de" in record.by_category and "task:out_of_domain" in record.by_category


def test_project_knows_both_suites() -> None:
    project = BenchmarkProject.at(REPO)
    assert project.primary_suite_name == "ste-multilingual-v1"
    keys = project.keys
    assert set(keys) >= {"ste-multilingual-v1", "ste-golden-v0"}
    assert keys["ste-multilingual-v1"] != keys["ste-golden-v0"]
    assert project.key_for("ste-multilingual-v0") != keys["ste-multilingual-v1"]


def test_suite_v1_keeps_v0_items_and_adds_instruction_requests() -> None:
    v1 = load_suite(REPO / "benchmarks/golden/ste-multilingual-v1")
    v0_items = {i.item_id: i for i in SUITE.items}
    new = [i for i in v1.items if i.item_id not in v0_items]
    assert all(v0_items[i.item_id] == i for i in v1.items if i.item_id in v0_items)
    assert len(new) == 4 and all(i.expected_unsupported for i in new)


def test_ste_figures_render_from_a_scored_record(tmp_path: Path) -> None:
    from steward.evaluation.plots import FIGURES, render_all

    refs = {i.source_text: i.reference for i in SUITE.items}
    candidate = CandidateConfig(
        id="oracle",
        role=CandidateRole.BASE,
        provider="scripted",
        model="m",
        base_model="b",
        prompt="ste_only@1",
        suite="ste-multilingual-v0",
    )
    ctx = BenchmarkContext(root=REPO, suite=SUITE, rules=RULES, prompt_dir=REPO / "prompts")
    record = run_candidate(
        ScriptedProvider(lambda s: refs.get(s) or "STEWARD_UNSUPPORTED_INPUT"),
        candidate,
        ctx,
        sleep=lambda _: None,
    )
    names = tuple(FIGURES[11:])
    assert len(names) == len(FIGURES) - 11
    written = render_all([record], [], tmp_path, formats=("svg",), names=names)
    assert len(written) == len(names)
    # Empty state when there is no STE-only record.
    empty = render_all([], [], tmp_path / "empty", formats=("svg",), names=names)
    assert all(
        "No STE-only benchmark results" in p.read_text() or p.stat().st_size > 0 for p in empty
    )
