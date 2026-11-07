"""Run one benchmark candidate on the frozen golden suite."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from steward import __version__
from steward.datasets.schema import SamplingParameters
from steward.evaluation.fingerprints import evaluator_fingerprint, worktree_fingerprint
from steward.evaluation.hardware import PeakMemorySampler, describe_hardware
from steward.evaluation.history import finalize_record
from steward.evaluation.metrics import (
    behavior_metrics,
    category_metrics,
    efficiency_metrics,
    evaluator_version,
    quality_metrics,
    score_output,
    violation_counts,
)
from steward.evaluation.records import (
    BenchmarkRecord,
    CandidateSpec,
    CodeRef,
    EvaluatorRef,
    ItemResult,
    Metric,
    RecordStatus,
)
from steward.evaluation.suite import CandidateConfig, Suite, SuiteItem
from steward.generation.pipeline import RewriteOutput, parse_json_output
from steward.generation.prompts import PromptTemplate, load_template
from steward.generation.providers.base import (
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    MalformedOutputError,
    ModelInfo,
    TextGenerationProvider,
)
from steward.generation.retry import RetriesExhaustedError, RetryPolicy, call_with_retry
from steward.logs import log_event
from steward.provenance import git_state
from steward.ste.rules import RULESET_VERSION, RuleConfig

logger = logging.getLogger(__name__)

REPAIR_MESSAGE = (
    'Your answer was not valid JSON with a non-empty "ste_text" string. '
    "Return only the JSON object."
)


class GpuUnavailableError(Exception):
    """A GPU benchmark was requested, but no GPU is visible."""


@dataclass(frozen=True)
class BenchmarkContext:
    """Everything that defines how results are scored."""

    root: Path
    suite: Suite
    rules: RuleConfig
    prompt_dir: Path


def parse_parameter_size(text: object) -> int | None:
    """Parse Ollama ``parameter_size`` strings such as ``8.2B`` or ``596.59M``."""
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMBT])\s*", text.upper())
    if match is None:
        return None
    scale = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[match.group(2)]
    return round(float(match.group(1)) * scale)


def candidate_spec(
    candidate: CandidateConfig, info: ModelInfo, template: PromptTemplate
) -> CandidateSpec:
    """Combine registry data with what the provider reports."""
    quant = info.details.get("quantization_level")
    return CandidateSpec(
        candidate_id=candidate.id,
        role=candidate.role,
        provider=candidate.provider,
        model=info.model,
        model_digest=info.digest,
        base_model=candidate.base_model,
        base_revision=candidate.base_revision
        or (f"{candidate.provider}:{info.digest[:12]}" if info.digest else None),
        checkpoint_id=candidate.checkpoint_id,
        adapter_id=candidate.adapter_id,
        training_run_id=candidate.training_run_id,
        release_version=candidate.release_version,
        parameter_count=candidate.parameter_count
        or _int_or_none(info.details.get("parameter_count"))
        or parse_parameter_size(info.details.get("parameter_size")),
        quantization=candidate.quantization or (quant if isinstance(quant, str) else None),
        disk_size_bytes=info.size_bytes,
        prompt=template.ref,
        think=candidate.think,
        adapter_path=candidate.adapter_path,
        adapter_sha256=_str_or_none(info.details.get("adapter_sha256")),
        adapter_size_bytes=_int_or_none(info.details.get("adapter_size_bytes")),
    )


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _generate(
    provider: TextGenerationProvider,
    request: GenerationRequest,
    policy: RetryPolicy,
    sleep: Callable[[float], None],
    label: str,
) -> tuple[GenerationResult | MalformedOutputError, int]:
    """Call the provider. Transport errors are retried; malformed output is returned."""

    def attempt(_: int) -> GenerationResult | MalformedOutputError:
        try:
            return provider.generate(request)
        except MalformedOutputError as exc:
            return exc

    return call_with_retry(attempt, policy, label=label, sleep=sleep)


def run_item(
    provider: TextGenerationProvider,
    model: str,
    template: PromptTemplate,
    item: SuiteItem,
    ctx: BenchmarkContext,
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], None],
) -> ItemResult:
    """Run and score one item. Raises RetriesExhaustedError on transport failure."""
    rendered = template.render(
        category=item.category_label.replace("_", " "), source_text=item.source_text
    )
    messages = [
        ChatMessage(role="system", content=rendered.system),
        ChatMessage(role="user", content=rendered.user),
    ]
    sampling = ctx.suite.manifest.sampling
    schema = RewriteOutput.model_json_schema()
    attempts = 0
    repairs = 0
    latency = 0.0
    tokens: int | None = None
    tokens_per_s: float | None = None
    error: str | None = None
    while True:
        request = GenerationRequest(
            model=model, messages=messages, sampling=sampling, response_schema=schema
        )
        result, calls = _generate(provider, request, policy, sleep, label=item.item_id)
        attempts += calls
        raw: str | None = None
        if isinstance(result, GenerationResult):
            latency += result.latency_s
            tokens = (tokens or 0) + (result.completion_tokens or 0)
            eval_ns = result.metadata.get("eval_duration")
            if result.completion_tokens and isinstance(eval_ns, int) and eval_ns > 0:
                tokens_per_s = result.completion_tokens / (eval_ns / 1e9)
            raw = result.text
            try:
                output = parse_json_output(raw, RewriteOutput).ste_text.strip()
                if output:
                    scored = score_output(item, output, ctx.rules)
                    return ItemResult(
                        item_id=item.item_id,
                        category=item.category_label,
                        valid_output=True,
                        attempts=attempts,
                        repair_attempts=repairs,
                        output=output,
                        latency_s=round(latency, 4),
                        completion_tokens=tokens,
                        tokens_per_s=round(tokens_per_s, 2) if tokens_per_s else None,
                        **scored,  # type: ignore[arg-type]
                    )
                error = "Output has an empty ste_text."
            except MalformedOutputError as exc:
                error = str(exc)
        else:
            error = str(result)
        if repairs >= ctx.suite.manifest.repair.max_repair_attempts:
            break
        repairs += 1
        if raw:
            messages = [*messages, ChatMessage(role="assistant", content=raw)]
        messages = [*messages, ChatMessage(role="user", content=REPAIR_MESSAGE)]
    return ItemResult(
        item_id=item.item_id,
        category=item.category_label,
        valid_output=False,
        attempts=attempts,
        repair_attempts=repairs,
        output=None,
        error=error,
        latency_s=round(latency, 4),
        completion_tokens=tokens,
        critical_failure=item.safety_critical,
    )


def run_item_text(
    provider: TextGenerationProvider,
    model: str,
    template: PromptTemplate,
    item: SuiteItem,
    ctx: BenchmarkContext,
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], None],
) -> ItemResult:
    """Run and score one item under the STE-only text contract (no repair turns)."""
    from steward.evaluation.ste_only import score_ste_only

    rendered = template.render(source_text=item.source_text)
    messages = [
        ChatMessage(role="system", content=rendered.system),
        ChatMessage(role="user", content=rendered.user),
    ]
    request = GenerationRequest(
        model=model, messages=messages, sampling=ctx.suite.manifest.sampling
    )
    result, calls = _generate(provider, request, policy, sleep, label=item.item_id)
    if not isinstance(result, GenerationResult):
        return ItemResult(
            item_id=item.item_id,
            category=item.category_label,
            valid_output=False,
            attempts=calls,
            repair_attempts=0,
            output=None,
            error=str(result),
            critical_failure=item.safety_critical,
            task_type=item.task_type.value,
            source_language=item.source_language,
            expected_unsupported=item.expected_unsupported,
        )
    output = result.text.strip()
    eval_ns = result.metadata.get("eval_duration")
    tokens_per_s = (
        result.completion_tokens / (eval_ns / 1e9)
        if result.completion_tokens and isinstance(eval_ns, int) and eval_ns > 0
        else None
    )
    scored = score_ste_only(item, output, ctx.rules)
    truncated = result.finish_reason == "length"
    return ItemResult(
        item_id=item.item_id,
        category=item.category_label,
        valid_output=bool(output) and not truncated,
        attempts=calls,
        repair_attempts=0,
        output=output or None,
        error="Output reached max_tokens (truncated)." if truncated else None,
        latency_s=round(result.latency_s, 4),
        completion_tokens=result.completion_tokens,
        tokens_per_s=round(tokens_per_s, 2) if tokens_per_s else None,
        **scored,  # type: ignore[arg-type]
    )


def run_candidate(
    provider: TextGenerationProvider,
    candidate: CandidateConfig,
    ctx: BenchmarkContext,
    *,
    policy: RetryPolicy | None = None,
    dataset_version: str | None = None,
    require_gpu: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> BenchmarkRecord:
    """Evaluate ``candidate`` on every suite item and return a finalized record.

    The record has status ``failed`` when the provider stops answering in the
    middle of the run. Errors before the first item (missing model, no GPU)
    are raised, and no record is created.
    """
    policy = policy or RetryPolicy(max_attempts=3, initial_delay_s=2.0)
    template = load_template(ctx.prompt_dir, candidate.prompt)
    info = provider.describe_model(candidate.model)
    hardware = describe_hardware(info.provider_version)
    if require_gpu and hardware.gpu_name is None:
        raise GpuUnavailableError(
            "A GPU benchmark was requested, but nvidia-smi shows no GPU. "
            "Run without --require-gpu, or check the NVIDIA driver."
        )
    spec = candidate_spec(candidate, info, template)
    log_event(
        logger,
        logging.INFO,
        "benchmark_started",
        candidate=candidate.id,
        model=info.model,
        suite=f"{ctx.suite.manifest.id}@{ctx.suite.manifest.version}",
        items=len(ctx.suite.items),
    )
    sampler = PeakMemorySampler().start()
    started = time.perf_counter()
    items: list[ItemResult] = []
    status = RecordStatus.COMPLETED
    error: str | None = None
    try:
        text_contract = ctx.suite.manifest.contract == "ste_only_text"
        runner = run_item_text if text_contract else run_item
        for item in ctx.suite.items:
            result = runner(provider, info.model, template, item, ctx, policy=policy, sleep=sleep)
            items.append(result)
            log_event(
                logger,
                logging.INFO,
                "benchmark_item",
                candidate=candidate.id,
                item=item.item_id,
                valid=result.valid_output,
                passed=result.rule_check_passed,
                violations=len(result.violations),
                critical=result.critical_failure,
            )
    except RetriesExhaustedError as exc:
        status = RecordStatus.FAILED
        error = f"Provider failed at item {len(items) + 1}/{len(ctx.suite.items)}: {exc}"
        log_event(logger, logging.ERROR, "benchmark_failed", candidate=candidate.id, error=error)
    finally:
        peak = sampler.stop()
    vram_probe = getattr(provider, "loaded_vram_mib", None)
    vram = vram_probe(info.model) if callable(vram_probe) else None

    if status is RecordStatus.COMPLETED and len(items) != len(ctx.suite.items):
        status, error = RecordStatus.FAILED, "Not every suite item has a result."
    metrics: dict[str, Any] = {}
    by_category: dict[str, Any] = {}
    counts: tuple[dict[Any, int], dict[str, int]] = ({}, {})
    contract = ctx.suite.manifest.contract
    if status is RecordStatus.COMPLETED and contract == "ste_only_text":
        from steward.evaluation.ste_only import ste_only_all_metrics, ste_only_groups

        metrics = {
            **ste_only_all_metrics(items),
            **behavior_metrics([i for i in items if not i.expected_unsupported]),
            **efficiency_metrics(
                items,
                parameter_count=spec.parameter_count,
                disk_size_bytes=spec.disk_size_bytes,
                gpu_peak_memory_mib=peak,
                model_vram_mib=vram,
            ),
        }
        if spec.adapter_size_bytes:
            metrics["adapter_size_bytes"] = Metric(
                value=float(spec.adapter_size_bytes),
                unit="bytes",
                n=1,
                higher_is_better=False,
                note="LoRA adapter weights only; disk_size_bytes includes the base model.",
            )
        by_category = ste_only_groups(items)
        counts = violation_counts(items)
    elif status is RecordStatus.COMPLETED:
        metrics = {
            **quality_metrics(items),
            **behavior_metrics(items),
            **efficiency_metrics(
                items,
                parameter_count=spec.parameter_count,
                disk_size_bytes=spec.disk_size_bytes,
                gpu_peak_memory_mib=peak,
                model_vram_mib=vram,
            ),
        }
        by_category = category_metrics(items)
        counts = violation_counts(items)
    commit, dirty = git_state(ctx.root)
    data: dict[str, Any] = {
        "status": status,
        "error": error,
        "code": CodeRef(
            steward_version=__version__,
            git_commit=commit,
            git_dirty=dirty,
            worktree_fingerprint=worktree_fingerprint(ctx.root),
        ),
        "hardware": hardware,
        "candidate": spec,
        "suite": ctx.suite.ref,
        "evaluator": EvaluatorRef(
            evaluator_version=evaluator_version(contract),
            ruleset_version=RULESET_VERSION,
            rules_config_sha256=ctx.rules.source_sha256,
            code_fingerprint=evaluator_fingerprint(ctx.root),
        ),
        "sampling": SamplingParameters.model_validate(ctx.suite.manifest.sampling.model_dump()),
        "dataset_version": dataset_version,
        "metrics": metrics,
        "by_category": by_category,
        "violation_counts": counts[0],
        "rule_counts": counts[1],
        "items": items,
        "duration_s": round(time.perf_counter() - started, 3),
    }
    record = finalize_record(BenchmarkRecord, data, clock())
    assert isinstance(record, BenchmarkRecord)
    return record
