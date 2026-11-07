"""Independent fidelity critique and critique-guided revision (``steward dataset critique``).

A *critic* model from another family than the teacher compares each source text
(in its original language) with the English target and lists errors that change
technical meaning. When the critic reports critical problems, the *reviser* (the
teacher) gets the findings once per round and returns a corrected target. A revision
is kept only when

1. every deterministic and heuristic ``prelint-3`` check still passes,
2. it has no more heuristic STE rule errors than before, and
3. a new critique of the revised target has no critical problem.

The result is stored in ``DatasetExample.llm_review`` with ``method = "llm_judged"``.
It is never treated as proof of correctness and never as a human review. The SFT
preparation can require a passed review (``require_llm_review``).
"""

from __future__ import annotations

import functools
import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from steward.datasets.schema import (
    CritiqueCategory,
    CritiqueProblem,
    DatasetExample,
    LlmReview,
    SamplingParameters,
)
from steward.generation.config import ValidationSettings
from steward.generation.pipeline import RewriteOutput, parse_json_output
from steward.generation.prompts import PromptTemplate
from steward.generation.providers.base import (
    ChatMessage,
    GenerationRequest,
    ProviderError,
    TextGenerationProvider,
)
from steward.generation.repair import rule_errors
from steward.generation.retry import RetriesExhaustedError, RetryPolicy, call_with_retry
from steward.generation.validation import validate_multitask
from steward.logs import log_event
from steward.ste import preservation as pv
from steward.ste.rules import RuleConfig

logger = logging.getLogger(__name__)


class _Problem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: CritiqueCategory
    severity: str = "critical"
    source_span: str = ""
    target_span: str = ""
    explanation: str = ""


class CritiqueOutput(BaseModel):
    """The JSON object the critic returns."""

    model_config = ConfigDict(extra="ignore")

    problems: list[_Problem] = Field(default_factory=list)
    verdict: str = "pass"


def _messages(template: PromptTemplate, **values: str) -> list[ChatMessage]:
    rendered = template.render(**values)
    return [
        ChatMessage(role="system", content=rendered.system),
        ChatMessage(role="user", content=rendered.user),
    ]


def _parse_critique(text: str) -> list[CritiqueProblem]:
    out = parse_json_output(text, CritiqueOutput)
    problems = [
        CritiqueProblem(
            category=p.category,
            severity="minor" if p.severity.strip().lower() == "minor" else "critical",
            source_span=p.source_span[:300],
            target_span=p.target_span[:300],
            explanation=p.explanation[:400],
        )
        for p in out.problems
    ]
    # A "fail" verdict without a critical problem is treated as critical: the critic
    # must not be able to pass an example it rejected.
    if out.verdict.strip().lower() == "fail" and not any(
        p.severity == "critical" for p in problems
    ):
        problems = [p.model_copy(update={"severity": "critical"}) for p in problems] or [
            CritiqueProblem(
                category="contradiction",
                severity="critical",
                explanation="The critic returned 'fail' without details.",
            )
        ]
    return problems


def _ask(
    provider: TextGenerationProvider,
    request: GenerationRequest,
    parse: Callable[[str], object],
    _attempt: int,
) -> object:
    return parse(provider.generate(request).text)


def critique(
    provider: TextGenerationProvider,
    model: str,
    template: PromptTemplate,
    source_text: str,
    language: str,
    ste_text: str,
    sampling: SamplingParameters,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> list[CritiqueProblem]:
    """Run one critique and return the problems."""
    request = GenerationRequest(
        model=model,
        messages=_messages(
            template,
            language_name=pv.LANGUAGE_NAMES.get(language, language),
            source_text=source_text,
            ste_text=ste_text,
        ),
        sampling=sampling,
        response_schema=CritiqueOutput.model_json_schema(),
    )
    result, _ = call_with_retry(
        functools.partial(_ask, provider, request, _parse_critique),
        policy,
        label="critique",
        sleep=sleep,
    )
    assert isinstance(result, list)
    return result


def describe(problems: Sequence[CritiqueProblem]) -> str:
    """A numbered list of findings for the revise prompt."""
    return "\n".join(
        f'{n}. {p.category}: {p.explanation} (source: "{p.source_span}"; target: "{p.target_span}")'
        for n, p in enumerate(problems, 1)
    )


def review_example(
    example: DatasetExample,
    *,
    critic: TextGenerationProvider,
    critic_model: str,
    critique_template: PromptTemplate,
    reviser: TextGenerationProvider | None,
    reviser_model: str | None,
    revise_template: PromptTemplate | None,
    rules: RuleConfig,
    settings: ValidationSettings,
    critic_sampling: SamplingParameters,
    revise_sampling: SamplingParameters,
    rounds: int = 1,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> DatasetExample:
    """Critique one example (and revise it up to ``rounds`` times). Never raises."""
    policy = policy or RetryPolicy(max_attempts=3, initial_delay_s=1.0)
    info = critic.describe_model(critic_model)
    base = {
        "critic_provider": info.provider,
        "critic_model": critic_model,
        "critic_digest": info.digest,
        "prompt": critique_template.ref,
        "reviewed_at": clock(),
    }
    lang = example.source_language
    scenario = example.provenance.scenario
    try:
        problems = critique(
            critic,
            critic_model,
            critique_template,
            example.source_text,
            lang,
            example.ste_candidate,
            critic_sampling,
            policy,
            sleep,
        )
    except (RetriesExhaustedError, ProviderError) as exc:
        log_event(
            logger, logging.WARNING, "critique_error", example=example.example_id, error=str(exc)
        )
        review = LlmReview(verdict="fail", status="error", **base)  # type: ignore[arg-type]
        return example.model_copy(update={"llm_review": review})
    initial = problems
    text = example.ste_candidate
    used_rounds = 0
    for _ in range(rounds):
        if not any(p.severity == "critical" for p in problems):
            break
        if reviser is None or reviser_model is None or revise_template is None:
            break
        request = GenerationRequest(
            model=reviser_model,
            messages=_messages(
                revise_template,
                language_name=pv.LANGUAGE_NAMES.get(lang, lang),
                source_text=example.source_text,
                ste_text=text,
                problems=describe([p for p in problems if p.severity == "critical"]),
            ),
            sampling=revise_sampling,
            response_schema=RewriteOutput.model_json_schema(),
        )
        try:
            revised, _ = call_with_retry(
                functools.partial(
                    _ask,
                    reviser,
                    request,
                    lambda t: parse_json_output(t, RewriteOutput).ste_text.strip(),
                ),
                policy,
                label="revise",
                sleep=sleep,
            )
        except (RetriesExhaustedError, ProviderError):
            break
        assert isinstance(revised, str)
        used_rounds += 1
        report = validate_multitask(
            example.task_type,
            lang,
            example.task_category,
            example.source_text,
            revised,
            example.technical_entities,
            settings,
            required_terms=scenario.required_terms,
        )
        if not report.passed or len(rule_errors(revised, example.task_category, rules)) > len(
            rule_errors(text, example.task_category, rules)
        ):
            continue
        try:
            new_problems = critique(
                critic,
                critic_model,
                critique_template,
                example.source_text,
                lang,
                revised,
                critic_sampling,
                policy,
                sleep,
            )
        except (RetriesExhaustedError, ProviderError):
            break
        text, problems = revised, new_problems
    critical = any(p.severity == "critical" for p in problems)
    revised_ok = text != example.ste_candidate
    status = "failed" if critical else ("revised" if revised_ok else "passed")
    review = LlmReview(
        verdict="fail" if critical else "pass",
        problems=problems,
        rounds=used_rounds,
        status=status,  # type: ignore[arg-type]
        reviser_model=reviser_model if used_rounds else None,
        revise_prompt=revise_template.ref if used_rounds and revise_template else None,
        initial_problems=initial,
        **base,  # type: ignore[arg-type]
    )
    update: dict[str, object] = {"llm_review": review}
    if revised_ok:
        update |= {
            "ste_candidate": text,
            "pre_revision_candidate": example.ste_candidate,
            "validation": validate_multitask(
                example.task_type,
                lang,
                example.task_category,
                example.source_text,
                text,
                example.technical_entities,
                settings,
                required_terms=scenario.required_terms,
            ),
        }
    return example.model_copy(update=update)


def review_all(
    examples: Sequence[DatasetExample],
    *,
    workers: int = 4,
    on_done: Callable[[DatasetExample], None] | None = None,
    **kwargs: object,
) -> tuple[list[DatasetExample], Counter[str]]:
    """Review examples in parallel; output order equals input order."""

    def job(example: DatasetExample) -> DatasetExample:
        out = review_example(example, **kwargs)  # type: ignore[arg-type]
        if on_done is not None:
            on_done(out)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        reviewed = list(pool.map(job, examples))
    counts: Counter[str] = Counter(
        e.llm_review.status if e.llm_review else "none" for e in reviewed
    )
    return reviewed, counts
