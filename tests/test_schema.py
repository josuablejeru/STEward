from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from steward.datasets.schema import (
    COMPLIANCE_NOTE,
    CheckKind,
    CheckResult,
    DatasetExample,
    GenerationParameters,
    PromptRef,
    PromptVersions,
    Provenance,
    ReviewState,
    SamplingParameters,
    Scenario,
    Severity,
    StageStats,
    TaskCategory,
    TeacherModel,
    ValidationReport,
)


def _check(passed: bool, severity: Severity = Severity.ERROR) -> CheckResult:
    return CheckResult(
        check_id="c", passed=passed, severity=severity, kind=CheckKind.DETERMINISTIC, message="m"
    )


def make_example(**overrides: object) -> DatasetExample:
    ref = PromptRef(template_id="t", version=1, sha256="0" * 64)
    params = SamplingParameters(temperature=0.2, seed=1)
    data: dict[str, object] = {
        "example_id": "stew-0123456789abcdef",
        "task_category": TaskCategory.PROCEDURE,
        "source_title": "Title",
        "source_text": "Remove the cover.",
        "ste_candidate": "Remove the cover.",
        "teacher": TeacherModel(provider="ollama", model="m"),
        "prompt_version": PromptVersions(source=ref, rewrite=ref),
        "generation_parameters": GenerationParameters(source=params, rewrite=params),
        "stages": [StageStats(stage="source", attempts=1, latency_s=0.1)],
        "validation": ValidationReport.from_checks("v", [_check(True)]),
        "provenance": Provenance(
            run_id="r",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            steward_version="0.1.0",
            config_sha256="x",
            environment={},
            scenario=Scenario(
                index=0,
                category=TaskCategory.PROCEDURE,
                topic="t",
                audience="a",
                sentence_count=3,
            ),
        ),
    }
    data.update(overrides)
    return DatasetExample.model_validate(data)


def test_example_round_trips_through_json() -> None:
    example = make_example()
    assert DatasetExample.model_validate_json(example.model_dump_json()) == example


def test_defaults_never_claim_compliance() -> None:
    example = make_example()
    assert example.review_state is ReviewState.CANDIDATE
    assert example.compliance_note == COMPLIANCE_NOTE
    assert "Not verified" in example.compliance_note
    assert example.provenance.synthetic and example.provenance.fictional


def test_extra_fields_are_rejected() -> None:
    data = make_example().model_dump(mode="json")
    data["certified"] = True
    with pytest.raises(ValidationError):
        DatasetExample.model_validate(data)


def test_example_id_format_is_enforced() -> None:
    with pytest.raises(ValidationError):
        make_example(example_id="not-an-id")


def test_naive_timestamps_are_rejected() -> None:
    data = make_example().model_dump(mode="json")
    data["provenance"]["created_at"] = "2026-01-01T00:00:00"
    with pytest.raises(ValidationError):
        DatasetExample.model_validate(data)


def test_validation_report_passed_ignores_warnings() -> None:
    report = ValidationReport.from_checks("v", [_check(True), _check(False, Severity.WARNING)])
    assert report.passed
    assert report.failed_check_ids == ["c"]
    assert not ValidationReport.from_checks("v", [_check(False)]).passed


def test_validation_report_rejects_tampered_passed_flag() -> None:
    data = ValidationReport.from_checks("v", [_check(False)]).model_dump()
    data["passed"] = True
    with pytest.raises(ValidationError, match="passed"):
        ValidationReport.model_validate(data)


def test_sampling_parameters_bounds() -> None:
    with pytest.raises(ValidationError):
        SamplingParameters(temperature=3.0)
    with pytest.raises(ValidationError):
        SamplingParameters(temperature=0.5, top_p=0.0)
