import json
from pathlib import Path

import pytest

from fakes import FakeOllama, chat_response
from steward.datasets.jsonl import read_jsonl
from steward.datasets.schema import SCHEMA_VERSION, DatasetExample, RejectedExample, ReviewState
from steward.generation.config import GenerationConfig
from steward.generation.pipeline import (
    DatasetGenerator,
    GenerationAbortedError,
    Manifest,
    OutputExistsError,
    OutputPaths,
    RewriteOutput,
    RunStatus,
    parse_json_output,
)
from steward.generation.providers.base import MalformedOutputError, ModelNotFoundError


def make_generator(
    fake: FakeOllama, config: GenerationConfig, prompt_dir: Path, output: Path, **kwargs: object
) -> DatasetGenerator:
    return DatasetGenerator(
        provider=fake.provider(),
        model=fake.model,
        config=config,
        prompt_dir=prompt_dir,
        output=output,
        sleep=lambda _: None,
        **kwargs,  # type: ignore[arg-type]
    )


def read_outputs(output: Path) -> tuple[list[DatasetExample], list[RejectedExample], Manifest]:
    paths = OutputPaths.for_output(output)
    return (
        read_jsonl(paths.examples, DatasetExample),
        read_jsonl(paths.rejected, RejectedExample),
        Manifest.model_validate_json(paths.manifest.read_text(encoding="utf-8")),
    )


def test_generates_valid_records(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "raw" / "examples.jsonl"
    record = make_generator(fake_ollama, config, prompt_dir, output).run(5)

    assert record.status is RunStatus.COMPLETED
    examples, rejected, manifest = read_outputs(output)
    assert len(examples) == 5 and rejected == []
    assert len({e.example_id for e in examples}) == 5
    first = examples[0]
    assert first.review_state is ReviewState.CANDIDATE
    assert first.teacher.digest == "d" * 64
    assert first.prompt_version.source.template_id == "source_passage"
    assert first.prompt_version.rewrite.ref == "ste_rewrite@1"
    assert first.provenance.config_sha256 == config.sha256()
    assert first.validation.passed
    assert [s.stage for s in first.stages] == ["source", "rewrite"]
    assert first.generation_parameters.rewrite.temperature == 0.2
    assert manifest.total_accepted == 5
    assert manifest.runs[0].accepted == 5
    assert manifest.runs[0].teacher.model == "fake:1b"
    assert manifest.runs[0].completion_tokens == 5 * 2 * 60


def test_requests_use_structured_output_and_derived_seeds(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    make_generator(fake_ollama, config, prompt_dir, tmp_path / "a.jsonl").run(2)
    requests = fake_ollama.chat_requests
    assert len(requests) == 4
    assert requests[0]["format"]["properties"].keys() == {
        "title",
        "text",
        "identifiers",
        "safety_constraints",
    }
    assert requests[1]["format"]["properties"].keys() == {"ste_text"}
    seeds = [r["options"]["seed"] for r in requests]
    assert len(set(seeds)) == 4

    again = FakeOllama()
    make_generator(again, config, prompt_dir, tmp_path / "b.jsonl").run(2)
    assert [r["options"]["seed"] for r in again.chat_requests] == seeds
    assert [r["messages"] for r in again.chat_requests] == [r["messages"] for r in requests]


def test_duplicates_are_rejected(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    fake_ollama.source_text = lambda n: (
        "The 4 bolts should be removed. Torque is 10 Nm."
        if n < 3
        else f"The {n} bolts should be removed and kept. Torque is {n + 1} Nm."
    )
    output = tmp_path / "examples.jsonl"
    record = make_generator(fake_ollama, config, prompt_dir, output).run(3)
    examples, rejected, _ = read_outputs(output)
    assert len(examples) == 3
    assert record.rejection_reasons == {"duplicate": 2}
    assert all(r.duplicate_of == examples[0].example_id for r in rejected)


def test_malformed_output_is_retried(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    fake_ollama.chat_queue.append(lambda _: chat_response("this is not json"))
    output = tmp_path / "examples.jsonl"
    make_generator(fake_ollama, config, prompt_dir, output).run(1)
    examples, rejected, _ = read_outputs(output)
    assert rejected == []
    assert examples[0].stages[0].attempts == 2
    first_seed = fake_ollama.chat_requests[0]["options"]["seed"]
    assert examples[0].generation_parameters.source.seed == first_seed + 1


def test_persistent_malformed_output_is_rejected_and_run_continues(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    attempts = config.request.max_attempts
    fake_ollama.chat_queue.extend([lambda _: chat_response("{bad")] * attempts)
    output = tmp_path / "examples.jsonl"
    record = make_generator(fake_ollama, config, prompt_dir, output).run(2)
    examples, rejected, _ = read_outputs(output)
    assert len(examples) == 2
    assert [r.reason.value for r in rejected] == ["malformed_output"]
    assert record.rejection_reasons == {"malformed_output": 1}
    assert rejected[0].scenario.index == 0
    assert examples[0].provenance.scenario.index == 1


def test_failed_validation_goes_to_rejected_file(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    # A rewrite that drops all numbers fails the numbers_preserved check.
    fake_ollama.chat_queue.extend(
        [
            lambda _: chat_response(
                '{"title": "T", "text": "Remove the 6 bolts. Torque is 9 Nm."}'
            ),
            lambda _: chat_response('{"ste_text": "Remove the bolts."}'),
        ]
    )
    output = tmp_path / "examples.jsonl"
    make_generator(fake_ollama, config, prompt_dir, output).run(1)
    _, rejected, _ = read_outputs(output)
    assert rejected[0].reason.value == "failed_validation"
    assert rejected[0].validation is not None
    assert "numbers_preserved" in rejected[0].validation.failed_check_ids
    assert rejected[0].ste_candidate == "Remove the bolts."


def test_existing_output_needs_resume_or_overwrite(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "examples.jsonl"
    make_generator(fake_ollama, config, prompt_dir, output).run(1)
    with pytest.raises(OutputExistsError, match="--resume"):
        make_generator(FakeOllama(), config, prompt_dir, output).run(1)
    make_generator(FakeOllama(), config, prompt_dir, output, overwrite=True).run(2)
    examples, _, manifest = read_outputs(output)
    assert len(examples) == 2
    assert len(manifest.runs) == 1


def test_resume_continues_to_target(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "examples.jsonl"
    make_generator(fake_ollama, config, prompt_dir, output).run(2)

    resumed = FakeOllama()
    resumed.source_text = lambda n: f"The {n + 50} clamps should be opened. Torque is {n + 70} Nm."
    record = make_generator(resumed, config, prompt_dir, output, resume=True).run(4)
    examples, _, manifest = read_outputs(output)
    assert record.start_index == 2
    assert record.accepted == 2
    assert len(examples) == 4
    assert [e.provenance.scenario.index for e in examples] == [0, 1, 2, 3]
    assert len(manifest.runs) == 2
    assert manifest.total_accepted == 4


def test_resume_when_target_already_met_makes_no_calls(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "examples.jsonl"
    make_generator(fake_ollama, config, prompt_dir, output).run(2)
    again = FakeOllama()
    record = make_generator(again, config, prompt_dir, output, resume=True).run(2)
    assert record.accepted == 0 and again.chat_requests == []
    assert record.status is RunStatus.COMPLETED


def test_missing_model_fails_before_generation(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    fake_ollama.models_available = False
    output = tmp_path / "examples.jsonl"
    with pytest.raises(ModelNotFoundError, match="ollama pull"):
        make_generator(fake_ollama, config, prompt_dir, output).run(1)
    assert not output.exists()


def test_repeated_connection_failures_abort_and_save_progress(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    fake_ollama.refuse_chat_after = 2  # the first example succeeds, then the server goes away
    output = tmp_path / "examples.jsonl"
    with pytest.raises(GenerationAbortedError, match="in a row"):
        make_generator(fake_ollama, config, prompt_dir, output).run(3)

    examples, rejected, manifest = read_outputs(output)
    failures = config.request.max_consecutive_failures
    assert len(examples) == 1
    assert len(rejected) == failures
    assert all(r.reason.value == "provider_error" for r in rejected)
    assert manifest.runs[0].status is RunStatus.ABORTED
    assert len(fake_ollama.chat_requests) == 2 + failures * config.request.max_attempts


def test_low_acceptance_stops_as_incomplete(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    fake_ollama.source_text = lambda n: "The 4 bolts should be removed. Torque is 10 Nm."
    output = tmp_path / "examples.jsonl"
    record = make_generator(fake_ollama, config, prompt_dir, output).run(2, max_scenarios=4)
    assert record.status is RunStatus.INCOMPLETE
    assert record.accepted == 1 and record.rejected == 3
    assert "acceptance rate" in (record.error or "")


def test_parse_json_output_accepts_code_fence() -> None:
    assert parse_json_output('```json\n{"ste_text": "Open it."}\n```', RewriteOutput).ste_text == (
        "Open it."
    )
    with pytest.raises(MalformedOutputError, match="ste_text"):
        parse_json_output('{"other": 1}', RewriteOutput)


def test_records_are_plain_json_lines(
    fake_ollama: FakeOllama, config: GenerationConfig, prompt_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "examples.jsonl"
    make_generator(fake_ollama, config, prompt_dir, output).run(2)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["schema_version"] == SCHEMA_VERSION
