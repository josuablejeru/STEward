from __future__ import annotations

import json
from pathlib import Path

from ste_helpers import ScriptedProvider, make_example
from steward.datasets.schema import SamplingParameters, TaskCategory
from steward.generation.config import ValidationSettings
from steward.generation.prompts import load_template
from steward.generation.repair import repair_all, repair_example
from steward.ste.rules import load_rule_config

REPO = Path(__file__).resolve().parents[1]
KW = {
    "model": "teacher",
    "template": load_template(REPO / "prompts", "ste_repair@1"),
    "rules": load_rule_config(REPO / "configs/ste/terms.toml"),
    "settings": ValidationSettings(),
    "sampling": SamplingParameters(temperature=0.2),
    "sleep": lambda _: None,
}
PASSIVE = make_example(
    "The 4 bolts on panel P1A must be removed by the technician.",
    "The 4 bolts on panel P1A are removed by the technician.",
    category=TaskCategory.PROCEDURE,
    entities=["P1A"],
)


def test_clean_examples_are_not_sent() -> None:
    clean = make_example("Remove the 4 bolts.", "Remove the 4 bolts.")
    provider = ScriptedProvider(lambda _: "unused")
    outcome = repair_example(clean, provider=provider, **KW)  # type: ignore[arg-type]
    assert outcome.status == "clean" and provider.requests == []


def test_repair_replaces_candidate_and_records_provenance() -> None:
    provider = ScriptedProvider(
        lambda _: json.dumps({"ste_text": "Remove the 4 bolts from panel P1A."})
    )
    outcome = repair_example(PASSIVE, provider=provider, **KW)  # type: ignore[arg-type]
    assert outcome.status == "repaired"
    ex = outcome.example
    assert ex.ste_candidate == "Remove the 4 bolts from panel P1A."
    assert ex.pre_repair_candidate == PASSIVE.ste_candidate
    assert ex.prompt_version.repair is not None
    assert ex.prompt_version.repair.ref == "ste_repair@1"
    assert ex.stages[-1].stage == "repair"
    # The problem list names the rule and the passive text.
    assert "STE-GRAM-PASSIVE" in provider.requests[0].messages[-1].content


def test_repair_that_loses_a_fact_is_not_used() -> None:
    provider = ScriptedProvider(lambda _: json.dumps({"ste_text": "Remove the bolts."}))
    outcome = repair_example(PASSIVE, provider=provider, rounds=2, **KW)  # type: ignore[arg-type]
    assert outcome.status == "unchanged_failed"
    assert outcome.example == PASSIVE
    assert len(provider.requests) == 2


def test_repair_all_keeps_order() -> None:
    other = make_example("Remove the 2 nuts.", "Remove the 2 nuts.", index=5)
    provider = ScriptedProvider(
        lambda _: json.dumps({"ste_text": "Remove the 4 bolts from panel P1A."})
    )
    out, counts = repair_all([PASSIVE, other], workers=2, provider=provider, **KW)  # type: ignore[arg-type]
    assert [e.source_text for e in out] == [PASSIVE.source_text, other.source_text]
    assert counts == {"repaired": 1, "clean": 1}
