import pytest

from steward.datasets.schema import TaskCategory
from steward.generation.config import ValidationSettings
from steward.generation.validation import extract_numbers, validate_example
from steward.ste.sentences import split_sentences, word_count

SETTINGS = ValidationSettings()


def _failed(
    source: str, candidate: str, category: TaskCategory = TaskCategory.PROCEDURE
) -> set[str]:
    return set(validate_example(category, source, candidate, SETTINGS).failed_check_ids)


def test_good_rewrite_passes() -> None:
    report = validate_example(
        TaskCategory.PROCEDURE,
        "The 4 bolts should be removed and the cover must not be dropped.",
        "Remove the 4 bolts. Do not drop the cover.",
        SETTINGS,
    )
    assert report.passed
    assert report.failed_check_ids == []
    assert "Not an ASD-STE100 compliance assessment" in report.note


def test_missing_number_is_an_error() -> None:
    report = validate_example(
        TaskCategory.PROCEDURE,
        "Torque to 12.5 Nm and wait 1,000 ms.",
        "Torque to 12.5 Nm.",
        SETTINGS,
    )
    assert not report.passed
    check = next(c for c in report.checks if c.check_id == "numbers_preserved")
    assert check.details["missing"] == ["1000"]


def test_repeated_numbers_must_all_be_kept() -> None:
    assert "numbers_preserved" in _failed("Use 2 nuts and 2 washers.", "Use 2 nuts and washers.")


def test_lost_negation_is_a_warning() -> None:
    report = validate_example(
        TaskCategory.PROCEDURE, "Do not open the valve.", "Open the valve.", SETTINGS
    )
    assert report.passed
    assert "negation_preserved" in report.failed_check_ids


def test_lost_safety_label() -> None:
    assert "safety_labels_preserved" in _failed(
        "WARNING: Hot oil can burn you.", "Hot oil can burn you. Let it cool."
    )


def test_unchanged_candidate_warns() -> None:
    assert "candidate_differs_from_source" in _failed("Remove the cover.", "Remove the cover")


def test_sentence_length_limits_depend_on_category() -> None:
    sentence = " ".join(["word"] * 22) + "."
    assert "sentence_length" in _failed(sentence, sentence + " x", TaskCategory.PROCEDURE)
    assert "sentence_length" not in _failed(sentence, sentence + " x", TaskCategory.DESCRIPTION)


def test_empty_candidate_fails() -> None:
    assert "candidate_nonempty" in _failed("Remove the cover.", "  ")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("10 Nm and 2.5 mm", ["10", "2.5"]), ("1,200 psi", ["1200"]), ("no numbers", [])],
)
def test_extract_numbers(text: str, expected: list[str]) -> None:
    assert extract_numbers(text) == expected


def test_split_sentences_and_word_count() -> None:
    text = "1. Remove the cover. Install the new O-ring.\n- Torque to 2.5 Nm"
    assert split_sentences(text) == [
        "Remove the cover.",
        "Install the new O-ring.",
        "Torque to 2.5 Nm",
    ]
    assert word_count("Install the new O-ring at 2.5 mm.") == 7
