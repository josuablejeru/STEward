"""The STE-only application API.

:class:`SteOnlyService` turns technical input into STE-oriented English and
returns an :class:`SteResponse`. The response keeps the *generation* (what
the model wrote) separate from the *validation* (what the guard found), so
callers can inspect both. ``text`` is set only when the guard accepts or
flags the output; for rejected output it is ``None`` and the raw model text
is available in ``generation.raw_output`` for analysis only.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field

from steward.datasets.schema import COMPLIANCE_NOTE, SamplingParameters, TaskCategory
from steward.generation.prompts import PromptTemplate, load_template
from steward.generation.providers.base import (
    ChatMessage,
    GenerationRequest,
    TextGenerationProvider,
)
from steward.inference.guard import (
    GuardPolicy,
    GuardReport,
    GuardStatus,
    check_input,
    guard_output,
)
from steward.ste.rules import RuleConfig, Violation

DEFAULT_PROMPT = "ste_only@1"
DEFAULT_SAMPLING = SamplingParameters(temperature=0.0, top_p=1.0, seed=1234, max_tokens=768)


class GenerationInfo(BaseModel):
    """What the model produced, before validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    model_digest: str | None = None
    adapter_path: str | None = None
    prompt: str
    raw_output: str
    latency_s: float
    completion_tokens: int | None = None
    finish_reason: str | None = None


class SteResponse(BaseModel):
    """Result of one STE-only request (the ``GenerationResult`` of the API).

    * ``text``: the generated text, only when accepted or flagged,
    * ``status``: the validation status,
    * ``violations``: heuristic STE rule violations of the output,
    * ``validation_version``: guard and rule-set versions,
    * ``model_id``: base model plus adapter identity,
    * ``failure_reason``: why no text is returned, else None,
    * ``validation``: every guard check; ``generation``: the raw model output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: GuardStatus
    text: str | None = Field(description="STE-oriented text; None unless accepted or flagged.")
    model_id: str | None = None
    failure_reason: str | None = None
    generation: GenerationInfo | None
    validation: GuardReport
    compliance_note: str = COMPLIANCE_NOTE

    @property
    def ok(self) -> bool:
        """True when text is returned (accepted or flagged)."""
        return self.text is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def validation_version(self) -> str:
        """Guard and rule-set versions that produced ``validation``."""
        return f"{self.validation.guard_version}+{self.validation.ruleset_version}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def violations(self) -> list[Violation]:
        """Heuristic STE rule violations found in the output."""
        return list(self.validation.violations)


def failure_reason(report: GuardReport) -> str | None:
    """A short reason when no text is returned, else None."""
    if report.status is GuardStatus.UNSUPPORTED_INPUT:
        return "The input is outside the supported technical domain (controlled failure)."
    if report.status not in {GuardStatus.REJECTED, GuardStatus.INVALID_INPUT}:
        return None
    failed = {c.check_id: c.message for c in report.checks if c.critical and not c.passed}
    reasons = [failed[cid] for cid in report.critical_failures if cid in failed]
    return " ".join(r if r.endswith(".") else f"{r}." for r in reasons) or None


class SteOnlyService:
    """Generate STE-only output with a provider and check it with the guard."""

    def __init__(
        self,
        provider: TextGenerationProvider,
        model: str,
        rules: RuleConfig,
        *,
        prompt_dir: Path,
        prompt: str = DEFAULT_PROMPT,
        sampling: SamplingParameters = DEFAULT_SAMPLING,
        policy: GuardPolicy | None = None,
    ) -> None:
        """Bind a provider and a model to the STE-only prompt and guard."""
        self.provider = provider
        self.model = model
        self.rules = rules
        self.template: PromptTemplate = load_template(prompt_dir, prompt)
        self.sampling = sampling
        self.policy = policy or GuardPolicy()

    def messages(self, source: str) -> list[ChatMessage]:
        """The chat messages for ``source`` (same format as the training data)."""
        rendered = self.template.render(source_text=source)
        return [
            ChatMessage(role="system", content=rendered.system),
            ChatMessage(role="user", content=rendered.user),
        ]

    def model_id(self, digest: str | None = None) -> str:
        """``model@digest`` (the digest includes the adapter hash when one is loaded)."""
        return f"{self.model}@{digest}" if digest else self.model

    def generate(
        self,
        text: str,
        source_language: str | None = None,
        *,
        category: TaskCategory | None = None,
    ) -> SteResponse:
        """``generate(text, source_language=None) -> GenerationResult`` (see :meth:`convert`)."""
        return self.convert(text, source_language=source_language, category=category)

    def convert(
        self,
        source: str,
        *,
        source_language: str | None = None,
        category: TaskCategory | None = None,
    ) -> SteResponse:
        """Convert ``source`` to STE-oriented English, or return a controlled failure."""
        invalid = check_input(source, self.policy)
        if invalid is not None:
            return SteResponse(
                status=invalid.status,
                text=None,
                model_id=self.model_id(),
                failure_reason=failure_reason(invalid),
                generation=None,
                validation=invalid,
            )
        started = time.perf_counter()
        result = self.provider.generate(
            GenerationRequest(
                model=self.model, messages=self.messages(source), sampling=self.sampling
            )
        )
        info = self.provider.describe_model(self.model)
        report = guard_output(
            source,
            result.text,
            self.rules,
            source_language=source_language,
            category=category,
            policy=self.policy,
        )
        if result.finish_reason == "length" and report.status in {
            GuardStatus.ACCEPTED,
            GuardStatus.FLAGGED,
        }:
            report = report.model_copy(
                update={
                    "status": GuardStatus.REJECTED,
                    "critical_failures": [*report.critical_failures, "output_truncated"],
                    "review_state": None,
                }
            )
        reason = failure_reason(report)
        if "output_truncated" in report.critical_failures:
            reason = ((reason or "") + " The output reached the token limit (truncated).").strip()
        ok = report.status in {GuardStatus.ACCEPTED, GuardStatus.FLAGGED}
        return SteResponse(
            status=report.status,
            text=result.text.strip() if ok else None,
            model_id=self.model_id(info.digest),
            failure_reason=reason,
            generation=GenerationInfo(
                model=self.model,
                model_digest=info.digest,
                adapter_path=str(info.details.get("adapter_path") or "") or None,
                prompt=self.template.ref.ref,
                raw_output=result.text,
                latency_s=round(time.perf_counter() - started, 4),
                completion_tokens=result.completion_tokens,
                finish_reason=result.finish_reason,
            ),
            validation=report,
        )
