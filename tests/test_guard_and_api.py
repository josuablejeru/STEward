from __future__ import annotations

from pathlib import Path

from ste_helpers import ScriptedProvider
from steward.datasets.schema import ReviewState, TaskCategory
from steward.inference.api import SteOnlyService
from steward.inference.guard import GuardPolicy, GuardStatus, guard_output
from steward.ste.rules import RuleConfig, load_rule_config

REPO = Path(__file__).resolve().parents[1]
RULES: RuleConfig = load_rule_config(REPO / "configs/ste/terms.toml")
SOURCE_DE = (
    "VORSICHT: Den Sensor TS-5 nicht mit Druckluft reinigen. Die 4 Schrauben mit 3 N·m anziehen."
)
GOOD = "CAUTION: Do not use compressed air to clean sensor TS-5.\nTighten the 4 screws to 3 N·m."


def _status(output: str, source: str = SOURCE_DE, **kw: object) -> GuardStatus:
    return guard_output(source, output, RULES, source_language="de", **kw).status  # type: ignore[arg-type]


def test_accepts_faithful_english_ste() -> None:
    report = guard_output(SOURCE_DE, GOOD, RULES, category=TaskCategory.PROCEDURE)
    assert report.status is GuardStatus.ACCEPTED
    assert report.source_language == "de"
    assert report.review_state is ReviewState.VALIDATED_AGAINST_AVAILABLE_CHECKS


def test_rejects_untranslated_output() -> None:
    report = guard_output(SOURCE_DE, SOURCE_DE, RULES)
    assert report.status is GuardStatus.REJECTED
    assert {"english_output", "safety_labels_preserved"} <= set(report.critical_failures)


def test_rejects_lost_value_identifier_label_and_negation() -> None:
    cases = {
        "numbers_preserved": GOOD.replace("3 N·m", "N·m"),
        "identifiers_preserved": GOOD.replace("TS-5", "the sensor"),
        "safety_labels_preserved": GOOD.replace("CAUTION: ", ""),
        "negation_preserved": "CAUTION: Use compressed air to clean sensor TS-5.\n"
        "Tighten the 4 screws to 3 N·m.",
        "no_unsupported_additions": GOOD + "\nWait 10 minutes.",
        "output_contract": "Here is the text:\n" + GOOD,
    }
    for check, output in cases.items():
        report = guard_output(SOURCE_DE, output, RULES, source_language="de")
        assert report.status is GuardStatus.REJECTED, check
        assert check in report.critical_failures, (check, report.critical_failures)


def test_large_omission_is_rejected() -> None:
    source = (
        "Remove the 4 bolts from the access door.\nRemove the access door.\n"
        "Examine the seal for cuts and replace the seal if you find damage."
    )
    report = guard_output(source, "Remove the 4 bolts from the access door.", RULES)
    assert "length_plausible" in report.critical_failures


def test_measurement_with_wrong_unit_is_rejected() -> None:
    report = guard_output(SOURCE_DE, GOOD.replace("3 N·m", "3 psi"), RULES)
    assert "measurements_preserved" in report.critical_failures


def test_rule_errors_flag_but_strict_policy_rejects() -> None:
    source = "The cover must be removed by the technician after the power is off."
    output = "The cover is removed by the technician after the power is off."
    assert _status(output, source, category=TaskCategory.PROCEDURE) is GuardStatus.FLAGGED
    strict = GuardPolicy(strict_ste=True)
    assert (
        _status(output, source, category=TaskCategory.PROCEDURE, policy=strict)
        is GuardStatus.REJECTED
    )


def test_sentinel_is_a_controlled_failure() -> None:
    report = guard_output("Write a poem.", "STEWARD_UNSUPPORTED_INPUT", RULES)
    assert report.status is GuardStatus.UNSUPPORTED_INPUT


def test_guard_never_edits_output() -> None:
    service = SteOnlyService(
        ScriptedProvider(lambda _: "  " + GOOD + "  "), "m", RULES, prompt_dir=REPO / "prompts"
    )
    response = service.convert(SOURCE_DE)
    assert response.status is GuardStatus.ACCEPTED
    assert response.text == GOOD
    assert response.generation is not None
    assert response.generation.raw_output == "  " + GOOD + "  "


def test_api_withholds_text_on_rejection_and_keeps_raw_output() -> None:
    service = SteOnlyService(
        ScriptedProvider(lambda src: src), "m", RULES, prompt_dir=REPO / "prompts"
    )
    response = service.convert(SOURCE_DE)
    assert response.status is GuardStatus.REJECTED
    assert response.text is None
    assert response.generation is not None and response.generation.raw_output == SOURCE_DE
    assert not response.ok


def test_api_invalid_input_does_not_call_model() -> None:
    provider = ScriptedProvider(lambda _: GOOD)
    service = SteOnlyService(provider, "m", RULES, prompt_dir=REPO / "prompts")
    assert service.convert("   ").status is GuardStatus.INVALID_INPUT
    assert provider.requests == []


def test_api_rejects_truncated_output() -> None:
    provider = ScriptedProvider(lambda _: GOOD, finish_reason="length")
    service = SteOnlyService(provider, "m", RULES, prompt_dir=REPO / "prompts")
    response = service.convert(SOURCE_DE)
    assert response.status is GuardStatus.REJECTED
    assert "output_truncated" in response.validation.critical_failures


def test_api_uses_the_ste_only_prompt() -> None:
    provider = ScriptedProvider(lambda _: GOOD)
    service = SteOnlyService(provider, "m", RULES, prompt_dir=REPO / "prompts")
    service.convert(SOURCE_DE)
    messages = provider.requests[0].messages
    assert messages[0].role == "system" and "STEWARD_UNSUPPORTED_INPUT" in messages[0].content
    assert messages[1].content == SOURCE_DE
