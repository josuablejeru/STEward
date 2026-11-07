"""Helpers for STE-only tests: a scripted provider and example records."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from steward.datasets.schema import (
    DatasetExample,
    GenerationParameters,
    PromptRef,
    PromptVersions,
    Provenance,
    SamplingParameters,
    Scenario,
    SourceLanguage,
    StageStats,
    TaskCategory,
    TaskType,
    TeacherModel,
    ValidationReport,
)
from steward.generation.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelInfo,
)


class ScriptedProvider:
    """Answers with ``reply(user_message)``. Records every request."""

    name = "scripted"

    def __init__(self, reply: Callable[[str], str], *, finish_reason: str = "stop") -> None:
        self.reply = reply
        self.finish_reason = finish_reason
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        text = self.reply(request.messages[-1].content)
        return GenerationResult(
            text=text,
            model=request.model,
            latency_s=0.01,
            completion_tokens=len(text.split()),
            finish_reason=self.finish_reason,
            metadata={"eval_duration": 10_000_000},
        )

    def describe_model(self, model: str) -> ModelInfo:
        return ModelInfo(
            provider=self.name, model=model, digest="d" * 12, details={"parameter_count": 1000}
        )

    def close(self) -> None:
        pass


def make_example(
    source: str,
    target: str,
    *,
    task: TaskType = TaskType.PLAIN_TO_STE_REWRITE,
    language: SourceLanguage = "en",
    category: TaskCategory = TaskCategory.PROCEDURE,
    topic: str = "topic-a",
    index: int = 0,
    entities: list[str] | None = None,
) -> DatasetExample:
    """A schema-1.1 example that passed generation."""
    ref = PromptRef(template_id="t", version=1, sha256="0" * 64)
    params = SamplingParameters(temperature=0.2, seed=1)
    import hashlib

    return DatasetExample(
        example_id="stew-" + hashlib.sha256(source.encode()).hexdigest()[:16],
        task_type=task,
        task_category=category,
        source_language=language,
        source_title="Title",
        source_text=source,
        ste_candidate=target,
        technical_entities=entities or [],
        teacher=TeacherModel(provider="ollama", model="teacher:8b"),
        prompt_version=PromptVersions(source=ref, rewrite=ref),
        generation_parameters=GenerationParameters(source=params, rewrite=params),
        stages=[StageStats(stage="source", attempts=1, latency_s=0.1)],
        validation=ValidationReport.from_checks("prelint-2", []),
        provenance=Provenance(
            run_id="run-1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
            steward_version="0.1.0",
            config_sha256="x",
            environment={},
            scenario=Scenario(
                index=index,
                category=category,
                topic=topic,
                audience="a",
                sentence_count=3,
                task_type=task,
                source_language=language,
            ),
        ),
    )
