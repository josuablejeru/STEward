from pathlib import Path

import pytest

from bench_helpers import BenchFake, make_project
from steward.evaluation import hardware
from steward.evaluation.project import BenchmarkProject
from steward.evaluation.records import RecordStatus
from steward.evaluation.runner import BenchmarkContext, GpuUnavailableError, run_candidate
from steward.evaluation.service import Outcome, run_benchmarks
from steward.generation.providers.base import ModelNotFoundError
from steward.generation.retry import RetryPolicy

POLICY = RetryPolicy(max_attempts=2, initial_delay_s=0)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BenchmarkProject:
    monkeypatch.setattr(hardware, "gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "gpu_memory_used_mib", lambda: None)
    return BenchmarkProject.at(make_project(tmp_path))


def _ctx(project: BenchmarkProject) -> BenchmarkContext:
    return BenchmarkContext(project.paths.root, project.suite, project.rules, project.paths.prompts)


def test_completed_run_scores_every_item(project: BenchmarkProject) -> None:
    fake = BenchFake()
    record = run_candidate(
        fake.provider(),
        project.registry.get("tiny-prompted"),
        _ctx(project),
        policy=POLICY,
        sleep=lambda _: None,
    )
    assert record.status is RecordStatus.COMPLETED
    assert len(record.items) == 16
    assert record.candidate.parameter_count == 1_200_000_000
    assert record.candidate.quantization == "Q4_K_M"
    assert record.candidate.model_digest == "e" * 64
    # Echoing the source keeps every fact.
    assert record.metrics["meaning_preservation"].value == 1.0
    assert record.metrics["hallucination_rate"].value == 0.0
    assert record.metrics["tokens_per_s"].value == 100.0
    assert record.metrics["model_vram_mib"].value == 2048
    assert record.metrics["gpu_peak_memory_mib"].value is None
    assert record.hardware.gpu_name is None and record.hardware.note
    assert record.suite.review_state == "draft"
    assert set(record.by_category) == {
        "procedure",
        "description",
        "safety_notice",
        "troubleshooting",
    }
    sent = fake.calls[0]
    assert sent["options"]["temperature"] == 0.0 and sent["options"]["seed"] == 1234


def test_repair_attempt_recovers(project: BenchmarkProject) -> None:
    fake = BenchFake()
    fake.queue = ["not json"]
    record = run_candidate(
        fake.provider(),
        project.registry.get("tiny-base"),
        _ctx(project),
        policy=POLICY,
        sleep=lambda _: None,
    )
    first = record.items[0]
    assert first.valid_output and first.repair_attempts == 1
    assert fake.calls[1]["messages"][-1]["content"].startswith("Your answer was not valid JSON")
    assert record.metrics["repair_success_rate"].value == 1.0


def test_persistent_invalid_output_is_scored_as_failure(project: BenchmarkProject) -> None:
    fake = BenchFake()
    fake.queue = ["{bad"] * 3
    record = run_candidate(
        fake.provider(),
        project.registry.get("tiny-base"),
        _ctx(project),
        policy=POLICY,
        sleep=lambda _: None,
    )
    assert not record.items[0].valid_output
    assert record.metrics["task_completion_rate"].value == 15 / 16


def test_provider_failure_mid_run_gives_failed_record(project: BenchmarkProject) -> None:
    fake = BenchFake()
    fake.refuse_after = 3
    record = run_candidate(
        fake.provider(),
        project.registry.get("tiny-base"),
        _ctx(project),
        policy=POLICY,
        sleep=lambda _: None,
    )
    assert record.status is RecordStatus.FAILED
    assert record.metrics == {}
    assert "item 4/16" in (record.error or "")


def test_missing_model_creates_no_record(project: BenchmarkProject) -> None:
    fake = BenchFake()
    fake.available = False
    with pytest.raises(ModelNotFoundError):
        run_candidate(
            fake.provider(), project.registry.get("tiny-base"), _ctx(project), policy=POLICY
        )


def test_require_gpu_without_gpu(project: BenchmarkProject) -> None:
    with pytest.raises(GpuUnavailableError, match="no GPU"):
        run_candidate(
            BenchFake().provider(),
            project.registry.get("tiny-base"),
            _ctx(project),
            require_gpu=True,
        )


def test_service_appends_skips_and_reports_unavailable(project: BenchmarkProject) -> None:
    fake = BenchFake()
    candidates = project.registry.candidates
    first = run_benchmarks(
        project, candidates, lambda _: fake.provider(), policy=POLICY, sleep=lambda _: None
    )
    assert [o.outcome for o in first] == [Outcome.APPENDED, Outcome.APPENDED]
    second = run_benchmarks(project, candidates, lambda _: fake.provider(), policy=POLICY)
    assert [o.outcome for o in second] == [Outcome.SKIPPED, Outcome.SKIPPED]
    gone = BenchFake()
    gone.available = False
    third = run_benchmarks(project, candidates[:1], lambda _: gone.provider(), force=True)
    assert third[0].outcome is Outcome.UNAVAILABLE
    assert len(project.history()) == 2
