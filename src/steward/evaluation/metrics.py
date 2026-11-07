"""Score benchmark outputs and aggregate metrics.

All metrics here are **automated** and heuristic. "Meaning preservation"
and "hallucination" are measured with explicit proxies:

* meaning preservation: share of required facts kept (numbers from the
  source, ``must_include`` terms, WARNING/CAUTION labels, negation),
* hallucination: the output has a number that is not in the source,
* missing information: at least one required fact is missing.

These proxies do not detect a changed meaning that keeps the same words.
Human review is still necessary.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from steward.datasets.schema import TaskCategory
from steward.evaluation.records import ItemResult, Metric, MetricMethod
from steward.evaluation.suite import SuiteItem
from steward.generation.validation import extract_numbers
from steward.ste.rules import RuleCategory, RuleConfig, RuleSeverity, check_text
from steward.ste.sentences import words

EVALUATOR_VERSION = "eval-0.1"
"""Bump when scoring changes. Results with different versions are not comparable."""

PROCEDURAL = {TaskCategory.PROCEDURE, TaskCategory.TROUBLESHOOTING, TaskCategory.SAFETY_NOTICE}

_LIST_MARKER_RE = re.compile(r"^\s*(?:\d{1,2}|[A-Za-z])[.)]\s+", re.MULTILINE)
_NEGATION_RE = re.compile(r"\b(?:not|no|never|none|cannot|without)\b|n't\b", re.IGNORECASE)
_SAFETY_RE = re.compile(r"\b(WARNING|CAUTION)\b")


@dataclass(frozen=True, slots=True)
class MetricInfo:
    """Display metadata for one metric."""

    label: str
    unit: str
    higher_is_better: bool | None
    group: str


METRICS: dict[str, MetricInfo] = {
    "ste_rule_pass_rate": MetricInfo("STE rule-check pass rate", "fraction", True, "quality"),
    "violations_per_100_words": MetricInfo(
        "Violations per 100 words", "per 100 words", False, "quality"
    ),
    "sentence_length_violations_per_100_words": MetricInfo(
        "Sentence-length violations per 100 words", "per 100 words", False, "quality"
    ),
    "grammar_violations_per_100_words": MetricInfo(
        "Grammar violations per 100 words", "per 100 words", False, "quality"
    ),
    "vocabulary_compliance": MetricInfo(
        "Approved-vocabulary compliance", "fraction", True, "quality"
    ),
    "meaning_preservation": MetricInfo(
        "Meaning preservation (fact proxy)", "fraction", True, "quality"
    ),
    "hallucination_rate": MetricInfo("Items with invented numbers", "fraction", False, "quality"),
    "missing_information_rate": MetricInfo(
        "Items with missing facts", "fraction", False, "quality"
    ),
    "critical_failures": MetricInfo("Critical safety failures", "items", False, "quality"),
    "task_completion_rate": MetricInfo("Task completion rate", "fraction", True, "quality"),
    "invalid_output_rate": MetricInfo(
        "Invalid or incomplete outputs", "fraction", False, "behavior"
    ),
    "repair_attempts": MetricInfo("Repair attempts", "attempts", False, "behavior"),
    "repair_success_rate": MetricInfo("Repair success rate", "fraction", True, "behavior"),
    "output_words_mean": MetricInfo("Output length (mean)", "words", None, "behavior"),
    "output_words_p50": MetricInfo("Output length (median)", "words", None, "behavior"),
    "output_words_p90": MetricInfo("Output length (p90)", "words", None, "behavior"),
    "repetition_rate": MetricInfo("Repeated trigram rate", "fraction", False, "behavior"),
    "parameter_count": MetricInfo("Parameters", "parameters", False, "efficiency"),
    "disk_size_bytes": MetricInfo("Model size on disk", "bytes", False, "efficiency"),
    "gpu_peak_memory_mib": MetricInfo(
        "Peak GPU memory (all processes)", "MiB", False, "efficiency"
    ),
    "model_vram_mib": MetricInfo("Model VRAM (provider report)", "MiB", False, "efficiency"),
    "latency_mean_s": MetricInfo("Latency per item (mean)", "s", False, "efficiency"),
    "latency_p90_s": MetricInfo("Latency per item (p90)", "s", False, "efficiency"),
    "tokens_per_s": MetricInfo("Generation speed", "tokens/s", True, "efficiency"),
    # STE-only contract (suites with contract "ste_only_text").
    "ste_only_output_rate": MetricInfo(
        "STE-only output rate (English, no commentary)", "fraction", True, "contract"
    ),
    "non_english_output_rate": MetricInfo("Non-English output rate", "fraction", False, "contract"),
    "extra_commentary_rate": MetricInfo(
        "Commentary, Markdown, or reasoning rate", "fraction", False, "contract"
    ),
    "controlled_failure_reliability": MetricInfo(
        "Out-of-domain controlled-failure rate", "fraction", True, "contract"
    ),
    "false_refusal_rate": MetricInfo("In-domain inputs refused", "fraction", False, "contract"),
    "guard_pass_rate": MetricInfo(
        "Output guard pass rate (accepted or flagged)", "fraction", True, "contract"
    ),
    "unnecessary_rewrite_rate": MetricInfo(
        "Valid STE inputs rewritten", "fraction", False, "contract"
    ),
    "identity_edit_similarity": MetricInfo(
        "Identity items: word similarity to input", "fraction", True, "contract"
    ),
    "chrf_mean": MetricInfo("chrF2 vs. draft reference", "0-100", True, "quality"),
    "numerical_accuracy": MetricInfo("Numbers and measurements exact", "fraction", True, "quality"),
    "terminology_preservation": MetricInfo("Identifier codes kept", "fraction", True, "quality"),
    "negation_preservation": MetricInfo("Negations kept", "fraction", True, "quality"),
    "unsupported_addition_rate": MetricInfo(
        "Items with unsupported additions", "fraction", False, "quality"
    ),
    "sequence_preservation": MetricInfo(
        "Procedure order kept (fact order)", "fraction", True, "quality"
    ),
    "instruction_structure_violations_per_100_words": MetricInfo(
        "Instruction-structure violations per 100 words", "per 100 words", False, "quality"
    ),
    "adapter_size_bytes": MetricInfo("Adapter size on disk", "bytes", False, "efficiency"),
    "technical_error_rate": MetricInfo(
        "Items with a technical error", "fraction", False, "quality"
    ),
    "critical_technical_errors": MetricInfo("Technical errors", "items", False, "quality"),
    "safety_label_preservation": MetricInfo(
        "Safety labels kept exactly", "fraction", True, "quality"
    ),
    "terminology_accuracy": MetricInfo(
        "Fixture terms without a known wrong rendering", "fraction", True, "quality"
    ),
    "source_leakage_rate": MetricInfo(
        "Translations with untranslated words", "fraction", False, "quality"
    ),
}


def evaluator_version(contract: str) -> str:
    """The evaluator version for a suite contract."""
    if contract == "ste_only_text":
        from steward.evaluation.ste_only import STE_ONLY_EVALUATOR_VERSION

        return STE_ONLY_EVALUATOR_VERSION
    return EVALUATOR_VERSION


def _metric(name: str, value: float | None, n: int, **kwargs: object) -> Metric:
    info = METRICS[name]
    return Metric(
        value=value,
        unit=info.unit,
        n=n,
        higher_is_better=info.higher_is_better,
        method=MetricMethod.AUTOMATED,
        **kwargs,  # type: ignore[arg-type]
    )


def _numbers(text: str) -> list[str]:
    return extract_numbers(_LIST_MARKER_RE.sub("", text))


def repetition_rate(text: str) -> float:
    """Return ``1 - unique/total`` for word trigrams (0 for very short texts)."""
    tokens = [w.casefold() for w in words(text)]
    trigrams = list(zip(tokens, tokens[1:], tokens[2:], strict=False))
    if not trigrams:
        return 0.0
    return 1.0 - len(set(trigrams)) / len(trigrams)


def score_output(
    item: SuiteItem,
    output: str,
    rules: RuleConfig,
) -> dict[str, object]:
    """Score one valid output. Returns fields for :class:`ItemResult`."""
    violations = check_text(output, procedural=item.category in PROCEDURAL, config=rules)
    lowered = output.casefold()

    missing: list[str] = []
    remaining = _numbers(output)
    source_numbers = _numbers(item.source_text)
    for number in source_numbers:
        if number in remaining:
            remaining.remove(number)
        else:
            missing.append(number)
    invented = remaining
    for term in item.must_include:
        if term.casefold() not in lowered:
            missing.append(term)
    labels = set(_SAFETY_RE.findall(item.source_text))
    labels_missing = sorted(labels - set(_SAFETY_RE.findall(output)))
    missing.extend(label for label in labels_missing if label not in missing)
    negation_needed = bool(_NEGATION_RE.search(item.source_text))
    negation_lost = negation_needed and not _NEGATION_RE.search(output)

    facts_total = len(source_numbers) + len(item.must_include) + int(negation_needed)
    facts_total += len(labels - {t for t in item.must_include if t in labels})
    facts_missing = len(missing) + int(negation_lost)
    critical = bool(labels_missing) or (
        item.safety_critical and bool(missing or invented or negation_lost)
    )
    return {
        "words": len(words(output)),
        "violations": violations,
        "rule_check_passed": not any(v.severity is RuleSeverity.ERROR for v in violations),
        "facts_total": facts_total,
        "facts_preserved": max(facts_total - facts_missing, 0),
        "missing_facts": missing,
        "invented_numbers": invented,
        "negation_lost": negation_lost,
        "safety_labels_missing": labels_missing,
        "critical_failure": critical,
        "repetition_rate": round(repetition_rate(output), 4),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Return the Wilson score 95% interval for a proportion."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_interval(
    items: Sequence[ItemResult],
    statistic: Callable[[Sequence[ItemResult]], float | None],
    *,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap 95% interval (seeded, so it is reproducible)."""
    if len(items) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = [items[rng.randrange(len(items))] for _ in items]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _rate(
    items: Sequence[ItemResult],
    test: Callable[[ItemResult], bool],
    name: str,
    *,
    note: str | None = None,
) -> Metric:
    hits = sum(1 for i in items if test(i))
    ci = wilson_interval(hits, len(items))
    return _metric(
        name,
        hits / len(items) if items else None,
        len(items),
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
        note=note,
    )


def _per_100_words(
    items: Sequence[ItemResult], name: str, category: RuleCategory | None = None
) -> Metric:
    valid = [i for i in items if i.valid_output]

    def stat(sample: Sequence[ItemResult]) -> float | None:
        total_words = sum(i.words for i in sample)
        if total_words == 0:
            return None
        count = sum(
            1 for i in sample for v in i.violations if category is None or v.category is category
        )
        return 100.0 * count / total_words

    value = stat(valid)
    ci = bootstrap_interval(valid, stat)
    return _metric(
        name,
        round(value, 4) if value is not None else None,
        sum(i.words for i in valid),
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
        note="Denominator is output words of valid outputs.",
    )


def _meaning(items: Sequence[ItemResult]) -> Metric:
    def stat(sample: Sequence[ItemResult]) -> float | None:
        total = sum(i.facts_total for i in sample)
        return sum(i.facts_preserved for i in sample) / total if total else None

    value = stat(items)
    ci = bootstrap_interval(items, stat)
    return _metric(
        "meaning_preservation",
        round(value, 4) if value is not None else None,
        sum(i.facts_total for i in items),
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
        note="Fact proxy: numbers, required terms, safety labels, negation. "
        "Invalid outputs keep 0.",
    )


def quality_metrics(items: Sequence[ItemResult]) -> dict[str, Metric]:
    """Language-quality metrics for a list of item results."""
    return {
        "ste_rule_pass_rate": _rate(
            items, lambda i: i.valid_output and i.rule_check_passed, "ste_rule_pass_rate"
        ),
        "violations_per_100_words": _per_100_words(items, "violations_per_100_words"),
        "sentence_length_violations_per_100_words": _per_100_words(
            items, "sentence_length_violations_per_100_words", RuleCategory.SENTENCE_LENGTH
        ),
        "grammar_violations_per_100_words": _per_100_words(
            items, "grammar_violations_per_100_words", RuleCategory.GRAMMAR
        ),
        "vocabulary_compliance": _metric(
            "vocabulary_compliance",
            None,
            0,
            note="Not available: no licensed ASD-STE100 dictionary is configured.",
        ),
        "meaning_preservation": _meaning(items),
        "hallucination_rate": _rate(
            items, lambda i: bool(i.invented_numbers), "hallucination_rate"
        ),
        "missing_information_rate": _rate(
            items,
            lambda i: (not i.valid_output) or bool(i.missing_facts) or i.negation_lost,
            "missing_information_rate",
        ),
        "critical_failures": _metric(
            "critical_failures",
            float(sum(1 for i in items if i.critical_failure)),
            len(items),
            note="Safety labels lost, or facts lost/invented in a safety-critical item.",
        ),
        "task_completion_rate": _rate(items, lambda i: i.valid_output, "task_completion_rate"),
    }


def behavior_metrics(items: Sequence[ItemResult]) -> dict[str, Metric]:
    """Model-behavior metrics."""
    valid = [i for i in items if i.valid_output]
    lengths = [float(i.words) for i in valid]
    repaired = [i for i in items if i.repair_attempts > 0]
    repair_ok = sum(1 for i in repaired if i.valid_output)
    return {
        "invalid_output_rate": _rate(items, lambda i: not i.valid_output, "invalid_output_rate"),
        "repair_attempts": _metric(
            "repair_attempts", float(sum(i.repair_attempts for i in items)), len(items)
        ),
        "repair_success_rate": _metric(
            "repair_success_rate",
            repair_ok / len(repaired) if repaired else None,
            len(repaired),
            note=None if repaired else "No item needed a repair.",
        ),
        "output_words_mean": _metric(
            "output_words_mean", sum(lengths) / len(lengths) if lengths else None, len(lengths)
        ),
        "output_words_p50": _metric("output_words_p50", _percentile(lengths, 0.5), len(lengths)),
        "output_words_p90": _metric("output_words_p90", _percentile(lengths, 0.9), len(lengths)),
        "repetition_rate": _metric(
            "repetition_rate",
            sum(i.repetition_rate for i in valid) / len(valid) if valid else None,
            len(valid),
        ),
    }


def efficiency_metrics(
    items: Sequence[ItemResult],
    *,
    parameter_count: int | None,
    disk_size_bytes: int | None,
    gpu_peak_memory_mib: int | None,
    model_vram_mib: int | None,
) -> dict[str, Metric]:
    """Model-efficiency metrics."""
    latencies = [i.latency_s for i in items if i.latency_s is not None]
    eval_time = sum(
        i.completion_tokens / i.tokens_per_s
        for i in items
        if i.completion_tokens and i.tokens_per_s
    )
    tokens = sum(i.completion_tokens or 0 for i in items if i.tokens_per_s)
    gpu_note = None if gpu_peak_memory_mib is not None else "No GPU visible to nvidia-smi."
    return {
        "parameter_count": _metric(
            "parameter_count", float(parameter_count) if parameter_count else None, 1
        ),
        "disk_size_bytes": _metric(
            "disk_size_bytes", float(disk_size_bytes) if disk_size_bytes else None, 1
        ),
        "gpu_peak_memory_mib": _metric(
            "gpu_peak_memory_mib",
            float(gpu_peak_memory_mib) if gpu_peak_memory_mib is not None else None,
            1,
            note=gpu_note or "Peak used memory on GPU 0, all processes, sampled every 0.5 s.",
        ),
        "model_vram_mib": _metric(
            "model_vram_mib",
            float(model_vram_mib) if model_vram_mib is not None else None,
            1,
            note=None if model_vram_mib is not None else "Provider did not report VRAM use.",
        ),
        "latency_mean_s": _metric(
            "latency_mean_s", sum(latencies) / len(latencies) if latencies else None, len(latencies)
        ),
        "latency_p90_s": _metric("latency_p90_s", _percentile(latencies, 0.9), len(latencies)),
        "tokens_per_s": _metric(
            "tokens_per_s",
            round(tokens / eval_time, 2) if eval_time > 0 else None,
            tokens,
            note="Completion tokens divided by provider-reported generation time.",
        ),
    }


def category_metrics(items: Sequence[ItemResult]) -> dict[str, dict[str, Metric]]:
    """Quality metrics for each documentation category."""
    keys = (
        "ste_rule_pass_rate",
        "violations_per_100_words",
        "meaning_preservation",
        "critical_failures",
        "task_completion_rate",
    )
    result: dict[str, dict[str, Metric]] = {}
    for category in sorted({i.category for i in items}):
        subset = [i for i in items if i.category == category]
        metrics = quality_metrics(subset)
        result[category] = {k: metrics[k] for k in keys}
    return result


def violation_counts(items: Sequence[ItemResult]) -> tuple[dict[RuleCategory, int], dict[str, int]]:
    """Return violation counts per category and per rule ID."""
    by_category: Counter[RuleCategory] = Counter()
    by_rule: Counter[str] = Counter()
    for item in items:
        for violation in item.violations:
            by_category[violation.category] += 1
            by_rule[violation.rule_id] += 1
    return dict(sorted(by_category.items())), dict(sorted(by_rule.items()))
