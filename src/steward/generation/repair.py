"""Linter-guided repair of accepted candidates (``steward dataset repair``).

For each accepted example whose target has heuristic STE rule *errors*, the
teacher gets the source, the candidate, and the exact list of rule
problems, and returns a corrected candidate. The corrected text replaces the
old one only if it

1. passes every multitask sanity check (facts, labels, language), and
2. has fewer rule errors than before (``require_clean`` asks for zero).

Every example is written to the output file, repaired or not, so the output
can replace the input for SFT preparation. The old candidate is kept in
``pre_repair_candidate`` and the repair prompt and sampling are recorded.
"""

from __future__ import annotations

import functools
import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from steward.datasets.schema import (
    DatasetExample,
    SamplingParameters,
    StageStats,
    TaskCategory,
)
from steward.generation.config import ValidationSettings
from steward.generation.pipeline import RewriteOutput, parse_json_output
from steward.generation.prompts import PromptTemplate
from steward.generation.providers.base import (
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    ProviderError,
    TextGenerationProvider,
)
from steward.generation.retry import RetriesExhaustedError, RetryPolicy, call_with_retry
from steward.generation.scenarios import derive_seed
from steward.generation.validation import validate_multitask
from steward.logs import log_event
from steward.ste.rules import RuleConfig, RuleSeverity, Violation, check_text

logger = logging.getLogger(__name__)

PROCEDURAL = {TaskCategory.PROCEDURE, TaskCategory.TROUBLESHOOTING, TaskCategory.SAFETY_NOTICE}


def rule_errors(text: str, category: TaskCategory, rules: RuleConfig) -> list[Violation]:
    """Error-severity violations for ``text``."""
    return [
        v
        for v in check_text(text, procedural=category in PROCEDURAL, config=rules)
        if v.severity is RuleSeverity.ERROR
    ]


def describe_problems(text: str, violations: Sequence[Violation]) -> str:
    """A numbered problem list for the repair prompt."""
    lines = []
    for n, v in enumerate(violations, 1):
        start = text.rfind("\n", 0, v.start) + 1
        end = text.find("\n", v.end)
        context = text[start : end if end != -1 else len(text)].strip()
        lines.append(f'{n}. {v.rule_id}: {v.message} Text: "{v.text}" (in: "{context[:200]}")')
    return "\n".join(lines)


@dataclass(frozen=True)
class RepairOutcome:
    """What happened to one example."""

    example: DatasetExample
    status: str  # clean | repaired | improved | unchanged_failed | error


def repair_example(
    example: DatasetExample,
    *,
    provider: TextGenerationProvider,
    model: str,
    template: PromptTemplate,
    rules: RuleConfig,
    settings: ValidationSettings,
    sampling: SamplingParameters,
    rounds: int = 2,
    seed: int = 0,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> RepairOutcome:
    """Repair one example. Never raises for provider problems; returns ``error``."""
    errors = rule_errors(example.ste_candidate, example.task_category, rules)
    if not errors:
        return RepairOutcome(example, "clean")
    policy = policy or RetryPolicy(max_attempts=3, initial_delay_s=1.0)
    best_text, best_errors = example.ste_candidate, errors
    stats: list[StageStats] = []
    used: SamplingParameters | None = None
    for round_no in range(rounds):
        rendered = template.render(
            source_text=example.source_text,
            ste_text=best_text,
            problems=describe_problems(best_text, best_errors),
        )
        params = sampling.model_copy(
            update={"seed": derive_seed(seed, "repair", example.example_id, round_no)}
        )
        request = GenerationRequest(
            model=model,
            messages=[
                ChatMessage(role="system", content=rendered.system),
                ChatMessage(role="user", content=rendered.user),
            ],
            sampling=params,
            response_schema=RewriteOutput.model_json_schema(),
        )
        started = time.perf_counter()
        try:
            (text, result), attempts = call_with_retry(
                functools.partial(_ask_attempt, provider, request),
                policy,
                label=f"repair:{example.example_id}",
                sleep=sleep,
            )
        except (RetriesExhaustedError, ProviderError) as exc:
            log_event(
                logger, logging.WARNING, "repair_error", example=example.example_id, error=str(exc)
            )
            break
        stats.append(
            StageStats(
                stage="repair",
                attempts=attempts,
                latency_s=round(time.perf_counter() - started, 4),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
        )
        report = validate_multitask(
            example.task_type,
            example.source_language,
            example.task_category,
            example.source_text,
            text,
            example.technical_entities,
            settings,
            required_terms=example.provenance.scenario.required_terms,
        )
        new_errors = rule_errors(text, example.task_category, rules)
        if report.passed and len(new_errors) < len(best_errors):
            best_text, best_errors, used = text, new_errors, params
        if not best_errors:
            break
    if used is None:
        status = "unchanged_failed" if stats else "error"
        return RepairOutcome(example, status)
    final_report = validate_multitask(
        example.task_type,
        example.source_language,
        example.task_category,
        example.source_text,
        best_text,
        example.technical_entities,
        settings,
        required_terms=example.provenance.scenario.required_terms,
    )
    repaired = example.model_copy(
        update={
            "ste_candidate": best_text,
            "pre_repair_candidate": example.ste_candidate,
            "validation": final_report,
            "stages": [*example.stages, *stats],
            "prompt_version": example.prompt_version.model_copy(update={"repair": template.ref}),
            "generation_parameters": example.generation_parameters.model_copy(
                update={"repair": used}
            ),
        }
    )
    return RepairOutcome(repaired, "repaired" if not best_errors else "improved")


def _ask_attempt(
    provider: TextGenerationProvider, request: GenerationRequest, _attempt: int
) -> tuple[str, GenerationResult]:
    result = provider.generate(request)
    return parse_json_output(result.text, RewriteOutput).ste_text.strip(), result


def repair_all(
    examples: Sequence[DatasetExample],
    *,
    workers: int = 4,
    on_done: Callable[[RepairOutcome], None] | None = None,
    **kwargs: object,
) -> tuple[list[DatasetExample], Counter[str]]:
    """Repair examples in parallel; output order equals input order."""
    counts: Counter[str] = Counter()

    def job(example: DatasetExample) -> RepairOutcome:
        outcome = repair_example(example, **kwargs)  # type: ignore[arg-type]
        if on_done is not None:
            on_done(outcome)
        return outcome

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(job, examples))
    for o in outcomes:
        counts[o.status] += 1
    return [o.example for o in outcomes], counts
