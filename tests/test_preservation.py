from __future__ import annotations

import pytest

from steward.ste import preservation as pv


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Entfernen Sie die Abdeckung nicht, bevor der Druck 0 bar beträgt.", "de"),
        ("Ne retirez pas le panneau avant que la pression soit à 0 psi.", "fr"),
        ("Nu scoateți capacul până când presiunea este de 1,5 bar.", "ro"),
        ("Do not remove the cover until the pressure is 0 psi.", "en"),
    ],
)
def test_detect_language(text: str, language: str) -> None:
    assert pv.detect_language(text).language == language


def test_short_text_is_unknown_and_counts_as_english() -> None:
    assert pv.detect_language("TP-7").language == "unknown"
    assert pv.is_english("TP-7")


def test_mixed_language_with_diacritics_is_not_english() -> None:
    assert not pv.is_english("Pompa de combustibil operates at 45 psi și este răcită.")


def test_numbers_normalize_decimal_separators_across_languages() -> None:
    assert pv.numbers("2,5 mm und 1.000 kg", "de") == ["2.5", "1000"]
    assert pv.numbers("Torque to 1,250 Nm and 2.5 mm", "en") == ["1250", "2.5"]


def test_numbers_ignore_list_markers_and_identifier_digits() -> None:
    text = "1. Remove 4 bolts from panel 412AL.\nStep 2: Check connector FQ-2 at 50 Ohm."
    assert pv.numbers(text) == ["4", "50"]


def test_measurements_map_units_across_languages() -> None:
    assert pv.measurements("Warten Sie 5 Minuten bei 3 N·m.", "de") == [("5", "min"), ("3", "N·m")]
    assert pv.measurements("Wait 5 minutes at 3 N·m.") == [("5", "min"), ("3", "N·m")]


def test_identifiers() -> None:
    assert pv.identifiers("Open panel 312AR and CB-45; see 32-11-05.") == [
        "312AR",
        "CB-45",
        "32-11-05",
    ]


def test_safety_labels_in_all_languages() -> None:
    labels = pv.safety_labels("WARNUNG: x. ATTENTION : y. AVERTISMENT: z.")
    assert labels == {"WARNING": 2, "CAUTION": 1}
    assert pv.safety_labels("WARNUNG: x", english_only=True) == {}


def test_negation_per_language() -> None:
    assert pv.negation_count("Nicht berühren.", "de") == 1
    assert pv.negation_count("Ne pas toucher. Nettoyer.", "fr") == 1
    assert pv.negation_count("Nu atingeți.", "ro") == 1
    assert pv.negation_count("Do not touch. Don't wait.", "en") == 2


def test_contract_violations() -> None:
    assert pv.contract_violations("Remove the cover.") == []
    assert "assistant_preamble" in pv.contract_violations("Here is the STE text: Remove it.")
    assert "markdown_fence" in pv.contract_violations("```\nRemove it.\n```")
    assert "reasoning_trace" in pv.contract_violations("<think>hmm</think>Remove it.")
    assert "meta_commentary" in pv.contract_violations("Remove it.\nNote: I split the sentence.")


def test_sentinel() -> None:
    assert pv.is_unsupported_sentinel(" STEWARD_UNSUPPORTED_INPUT. ")
    assert not pv.is_unsupported_sentinel("STEWARD_UNSUPPORTED_INPUT Remove it.")
