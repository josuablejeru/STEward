from pathlib import Path

import pytest

from steward.datasets.schema import TaskCategory
from steward.evaluation.metrics import (
    behavior_metrics,
    quality_metrics,
    repetition_rate,
    score_output,
    wilson_interval,
)
from steward.evaluation.records import ItemResult
from steward.evaluation.suite import SuiteItem
from steward.ste.rules import load_rule_config

RULES = load_rule_config(Path(__file__).resolve().parents[1] / "configs/ste/terms.toml")

ITEM = SuiteItem(
    item_id="t1",
    category=TaskCategory.SAFETY_NOTICE,
    safety_critical=True,
    source_text="WARNING: Do not touch the 2 hot pipes for 30 minutes.",
    must_include=["hot pipes"],
)


def _item(output: str, item: SuiteItem = ITEM) -> ItemResult:
    return ItemResult.model_validate(
        {
            "item_id": item.item_id,
            "category": item.category_label,
            "valid_output": True,
            "attempts": 1,
            "repair_attempts": 0,
            "output": output,
            **score_output(item, output, RULES),
        }
    )


def test_faithful_output_keeps_all_facts() -> None:
    r = _item("WARNING: Do not touch the 2 hot pipes for 30 minutes.")
    assert r.facts_preserved == r.facts_total == 5  # 2 numbers, 1 term, negation, WARNING
    assert not r.critical_failure and r.missing_facts == [] and r.invented_numbers == []


def test_missing_label_and_number_is_critical() -> None:
    r = _item("Do not touch the 2 hot pipes.")
    assert set(r.missing_facts) == {"30", "WARNING"}
    assert r.safety_labels_missing == ["WARNING"]
    assert r.critical_failure


def test_invented_number_is_hallucination() -> None:
    r = _item("WARNING: Do not touch the 2 hot pipes for 30 minutes at 90 degrees.")
    assert r.invented_numbers == ["90"]
    assert r.critical_failure  # safety-critical item


def test_lost_negation() -> None:
    r = _item("WARNING: Touch the 2 hot pipes for 30 minutes.")
    assert r.negation_lost and r.critical_failure


def test_step_numbers_are_not_invented_values() -> None:
    r = _item("WARNING: Do not touch the 2 hot pipes.\n1. Wait 30 minutes.")
    assert r.invented_numbers == []


def test_wilson_interval() -> None:
    low, high = wilson_interval(8, 16) or (0, 0)
    assert low < 0.5 < high
    assert wilson_interval(0, 0) is None


def test_quality_metrics_denominators() -> None:
    good = _item("WARNING: Do not touch the 2 hot pipes for 30 minutes.")
    bad = ItemResult(
        item_id="t2",
        category="safety_notice",
        valid_output=False,
        attempts=3,
        repair_attempts=2,
        output=None,
        critical_failure=True,
    )
    metrics = quality_metrics([good, bad])
    assert metrics["task_completion_rate"].value == 0.5
    assert metrics["task_completion_rate"].n == 2
    assert metrics["critical_failures"].value == 1.0
    assert metrics["vocabulary_compliance"].value is None
    assert "dictionary" in (metrics["vocabulary_compliance"].note or "")
    behavior = behavior_metrics([good, bad])
    assert behavior["repair_success_rate"].value == 0.0
    assert behavior["invalid_output_rate"].value == 0.5


def test_repetition_rate() -> None:
    assert repetition_rate("a b c a b c a b c") > 0.4
    assert repetition_rate("remove the cover now") == 0.0
    assert repetition_rate("hi") == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [("8.2B", 8_200_000_000), ("596.59M", 596_590_000), ("bad", None), (None, None)],
)
def test_parse_parameter_size(text: object, expected: int | None) -> None:
    from steward.evaluation.runner import parse_parameter_size

    assert parse_parameter_size(text) == expected
