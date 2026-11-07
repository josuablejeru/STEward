"""Scoring for suites with the ``ste_only_text`` contract.

The model gets the STE-only prompt and must return only STE-oriented English
or the controlled-failure sentinel. Each output is scored with the same
guard that protects the inference API (:mod:`steward.inference.guard`), plus:

* ``chrF2`` against the draft reference text (a character n-gram overlap
  proxy for translation and rewrite quality; not a meaning judgment),
* identity items: whether the output is unchanged, and the word-level edit
  similarity,
* out-of-domain items: whether the controlled failure was returned.

All metrics are automated heuristics. Results on a draft suite are
indicative only.
"""

from __future__ import annotations

import difflib
from collections import Counter
from collections.abc import Callable, Sequence

from steward.datasets.schema import TaskType
from steward.evaluation.metrics import _metric, _rate, bootstrap_interval, quality_metrics
from steward.evaluation.records import ItemResult, Metric
from steward.evaluation.suite import SuiteItem
from steward.inference.guard import GuardStatus, guard_output
from steward.ste import preservation as pv
from steward.ste.rules import RuleConfig, RuleSeverity
from steward.ste.sentences import words

STE_ONLY_EVALUATOR_VERSION = "ste-only-eval-0.3"
"""Bump when STE-only scoring changes."""

FACT_CHECKS = (
    "numbers_preserved",
    "measurements_preserved",
    "identifiers_preserved",
    "safety_labels_preserved",
    "negation_preserved",
    "no_unsupported_additions",
    "sequence_preserved",
    "terminology_forbidden",
    "no_source_language_words",
    "no_contradiction",
)
TECHNICAL_CHECKS = (*FACT_CHECKS, "english_output", "length_plausible", "sentinel_mixed")
"""Guard checks whose failure is a technical (meaning) error, not a style problem."""


def chrf(hypothesis: str, reference: str, *, n: int = 6, beta: float = 2.0) -> float:
    """Sentence-level chrF (0-100), character n-grams 1..n, whitespace removed."""
    hyp = "".join(hypothesis.split())
    ref = "".join(reference.split())
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i : i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i : i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    rc = sum(recalls) / len(recalls)
    if p == 0 and rc == 0:
        return 0.0
    b2 = beta * beta
    return round(100 * (1 + b2) * p * rc / (b2 * p + rc), 3)


def edit_similarity(a: str, b: str) -> float:
    """Word-level similarity ratio (1.0 = identical after normalization)."""
    return round(
        difflib.SequenceMatcher(
            a=pv.normalize_text(a).split(), b=pv.normalize_text(b).split(), autojunk=False
        ).ratio(),
        4,
    )


def score_ste_only(item: SuiteItem, output: str, rules: RuleConfig) -> dict[str, object]:
    """Score one output. Returns fields for :class:`ItemResult`."""
    text = output.strip()
    sentinel = pv.is_unsupported_sentinel(text)
    base: dict[str, object] = {
        "task_type": item.task_type.value,
        "source_language": item.source_language,
        "expected_unsupported": item.expected_unsupported,
        "controlled_failure": sentinel,
        "words": 0 if sentinel else len(words(text)),
        "repetition_rate": 0.0,
    }
    if item.expected_unsupported:
        return base | {
            "guard_status": GuardStatus.UNSUPPORTED_INPUT.value if sentinel else "answered",
            "ste_only": None,
            "rule_check_passed": sentinel,
        }
    report = guard_output(
        item.source_text,
        text,
        rules,
        source_language=item.source_language,
        category=item.category,
    )
    checks = {c.check_id: c for c in report.checks}
    missing: list[str] = []
    for check_id in ("numbers_preserved", "measurements_preserved", "identifiers_preserved"):
        c = checks.get(check_id)
        if c is not None and not c.passed:
            missing += [str(x) for x in c.details.get("missing", [])]  # type: ignore[union-attr]
    labels_c = checks.get("safety_labels_preserved")
    labels_missing = (
        [str(x) for x in labels_c.details.get("missing", [])]  # type: ignore[union-attr]
        if labels_c is not None and not labels_c.passed
        else []
    )
    if sentinel:
        labels_missing = sorted(pv.safety_labels(item.source_text))
    lowered = text.casefold()
    missing_terms = [t for t in item.must_include if t.casefold() not in lowered]
    neg_c = checks.get("negation_preserved")
    negation_needed = pv.negation_count(item.source_text, item.source_language) > 0
    negation_lost = negation_needed and (sentinel or (neg_c is not None and not neg_c.passed))
    add_c = checks.get("no_unsupported_additions")
    invented = (
        [str(x) for x in add_c.details.get("added", [])]  # type: ignore[union-attr]
        if add_c is not None and not add_c.passed
        else []
    )

    n_numbers = len(pv.numbers(item.source_text, item.source_language))
    n_ids = len(pv.identifiers(item.source_text))
    n_labels = len(pv.safety_labels(item.source_text))
    facts_total = n_numbers + n_ids + n_labels + len(item.must_include) + int(negation_needed)
    if sentinel:
        facts_missing = facts_total
        missing = ["<controlled failure on in-domain input>"]
    else:
        facts_missing = len(missing) + len(labels_missing) + len(missing_terms) + int(negation_lost)
    contract = pv.contract_violations(text)
    english_check = checks.get("english_output")
    english = english_check.passed if english_check is not None else True
    ste_only = bool(text) and not sentinel and english and not contract
    errors = [v for v in report.violations if v.severity is RuleSeverity.ERROR]
    copied = pv.normalize_text(text) == pv.normalize_text(item.source_text)
    fact_failed = any(not checks[c].passed for c in FACT_CHECKS if c in checks)
    critical = bool(labels_missing) or (
        item.safety_critical and (sentinel or fact_failed or bool(missing_terms))
    )
    result = base | {
        "violations": report.violations,
        "rule_check_passed": ste_only
        and not errors
        and (item.task_type is TaskType.STE_IDENTITY or not copied),
        "facts_total": facts_total,
        "facts_preserved": max(facts_total - facts_missing, 0),
        "missing_facts": missing + missing_terms,
        "invented_numbers": invented,
        "negation_lost": negation_lost,
        "negation_expected": negation_needed,
        "labels_expected": bool(pv.safety_labels(item.source_text)),
        "terms_matched": [str(t) for t in checks["terminology_forbidden"].details.get("terms", [])]  # type: ignore[union-attr]
        if "terminology_forbidden" in checks
        else [],
        "safety_labels_missing": labels_missing,
        "critical_failure": critical,
        "repetition_rate": 0.0,
        "guard_status": report.status.value,
        "guard_critical": report.critical_failures,
        "ste_only": ste_only,
        "non_english": bool(text) and not sentinel and not english,
        "extra_commentary": bool(contract),
    }
    result["unchanged"] = copied
    if item.task_type is TaskType.STE_IDENTITY:
        result["edit_similarity"] = edit_similarity(item.source_text, text)
    if item.reference:
        result["chrf"] = 0.0 if sentinel else chrf(text, item.reference)
    return result


def _mean(
    items: Sequence[ItemResult],
    get: Callable[[ItemResult], float | None],
    name: str,
    note: str | None = None,
) -> Metric:
    values = [v for i in items if (v := get(i)) is not None]

    def stat(sample: Sequence[ItemResult]) -> float | None:
        vals = [v for i in sample if (v := get(i)) is not None]
        return sum(vals) / len(vals) if vals else None

    ci = bootstrap_interval([i for i in items if get(i) is not None], stat)
    return _metric(
        name,
        round(sum(values) / len(values), 4) if values else None,
        len(values),
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
        note=note,
    )


def _in_domain(items: Sequence[ItemResult]) -> list[ItemResult]:
    return [i for i in items if not i.expected_unsupported]


def _failed(item: ItemResult, *check_ids: str) -> bool:
    return any(c in (item.guard_critical or []) for c in check_ids)


def _technical_error(item: ItemResult) -> bool:
    if item.controlled_failure or not item.output:
        return True
    return _failed(item, *TECHNICAL_CHECKS)


def ste_only_metrics(items: Sequence[ItemResult]) -> dict[str, Metric]:
    """Contract and behavior metrics for STE-only suites."""
    in_dom = _in_domain(items)
    ood = [i for i in items if i.expected_unsupported]
    identity = [i for i in in_dom if i.task_type == TaskType.STE_IDENTITY.value]
    answered = [i for i in in_dom if i.output and not i.controlled_failure]
    with_ref = [i for i in in_dom if i.chrf is not None]
    return {
        "ste_only_output_rate": _rate(in_dom, lambda i: bool(i.ste_only), "ste_only_output_rate"),
        "non_english_output_rate": _rate(
            answered, lambda i: bool(i.non_english), "non_english_output_rate"
        ),
        "extra_commentary_rate": _rate(
            answered, lambda i: bool(i.extra_commentary), "extra_commentary_rate"
        ),
        "controlled_failure_reliability": _rate(
            ood, lambda i: bool(i.controlled_failure), "controlled_failure_reliability"
        ),
        "false_refusal_rate": _rate(
            in_dom, lambda i: bool(i.controlled_failure), "false_refusal_rate"
        ),
        "guard_pass_rate": _rate(
            in_dom,
            lambda i: i.guard_status in {GuardStatus.ACCEPTED.value, GuardStatus.FLAGGED.value},
            "guard_pass_rate",
        ),
        "unnecessary_rewrite_rate": _rate(
            identity, lambda i: not i.unchanged, "unnecessary_rewrite_rate"
        ),
        "identity_edit_similarity": _mean(
            identity, lambda i: i.edit_similarity, "identity_edit_similarity"
        ),
        "technical_error_rate": _rate(
            in_dom,
            lambda i: _technical_error(i),
            "technical_error_rate",
            note="In-domain items with a failed fidelity check (facts, labels, negation, "
            "order, terminology, untranslated words, contradiction, language, omission) or a "
            "refusal. Heuristic and deterministic guard checks; not a human review.",
        ),
        "critical_technical_errors": _metric(
            "critical_technical_errors",
            float(sum(1 for i in in_dom if _technical_error(i))),
            len(in_dom),
            note="Count of in-domain items with a technical error (see technical_error_rate).",
        ),
        "safety_label_preservation": _rate(
            [i for i in answered if i.labels_expected],
            lambda i: (
                not i.safety_labels_missing
                and not any(x.startswith("label ") for x in i.invented_numbers)
            ),
            "safety_label_preservation",
            note="Items whose source has WARNING/CAUTION/NOTE labels: every label kept, "
            "none changed or added.",
        ),
        "terminology_accuracy": _rate(
            [i for i in answered if i.terms_matched],
            lambda i: not _failed(i, "terminology_forbidden"),
            "terminology_accuracy",
            note="Items with a fixture term (configs/ste/terminology.toml): no known wrong "
            "rendering. Deterministic match; correct renderings are not proven.",
        ),
        "source_leakage_rate": _rate(
            [i for i in answered if i.source_language != "en"],
            lambda i: _failed(i, "no_source_language_words"),
            "source_leakage_rate",
            note="Translation items with untranslated source-language words (wordfreq).",
        ),
        "numerical_accuracy": _rate(
            answered,
            lambda i: (
                not _failed(i, "numbers_preserved", "measurements_preserved")
                and not any(x.startswith("number ") for x in i.invented_numbers)
            ),
            "numerical_accuracy",
            note="Items whose numbers and measurements are all kept and none added.",
        ),
        "terminology_preservation": _rate(
            answered,
            lambda i: not _failed(i, "identifiers_preserved"),
            "terminology_preservation",
            note="Identifier codes (part numbers, codes) kept unchanged. Technical nouns "
            "are only checked through must-include terms in meaning preservation.",
        ),
        "negation_preservation": _rate(
            [i for i in answered if i.negation_expected],
            lambda i: not i.negation_lost,
            "negation_preservation",
            note="Items whose source has a negation; the output must have one too.",
        ),
        "unsupported_addition_rate": _rate(
            answered,
            lambda i: _failed(i, "no_unsupported_additions"),
            "unsupported_addition_rate",
            note="Numbers, identifiers, or safety labels not in the source.",
        ),
        "sequence_preservation": _rate(
            answered,
            lambda i: not _failed(i, "sequence_preserved"),
            "sequence_preservation",
            note="Numbers and identifiers from different source sentences keep their order.",
        ),
        "instruction_structure_violations_per_100_words": _metric(
            "instruction_structure_violations_per_100_words",
            None,
            0,
            note="Not available: the rule set has no instruction-structure rule yet.",
        ),
        "chrf_mean": _mean(
            with_ref,
            lambda i: i.chrf,
            "chrf_mean",
            note="chrF2 against draft reference texts written by the suite author.",
        ),
    }


def ste_only_all_metrics(items: Sequence[ItemResult]) -> dict[str, Metric]:
    """Quality metrics on in-domain items plus the STE-only contract metrics."""
    return {**quality_metrics(_in_domain(items)), **ste_only_metrics(items)}


GROUP_KEYS = (
    "technical_error_rate",
    "critical_technical_errors",
    "negation_preservation",
    "safety_label_preservation",
    "terminology_preservation",
    "ste_rule_pass_rate",
    "meaning_preservation",
    "critical_failures",
    "ste_only_output_rate",
    "non_english_output_rate",
    "chrf_mean",
    "violations_per_100_words",
)


def ste_only_groups(items: Sequence[ItemResult]) -> dict[str, dict[str, Metric]]:
    """Metrics per task (``task:<type>``), language (``lang:<code>``), and category."""
    groups: dict[str, list[ItemResult]] = {}
    for item in _in_domain(items):
        groups.setdefault(f"task:{item.task_type}", []).append(item)
        groups.setdefault(f"lang:{item.source_language}", []).append(item)
        groups.setdefault(item.category, []).append(item)
    result = {}
    for key in sorted(groups):
        metrics = ste_only_all_metrics(groups[key])
        result[key] = {k: metrics[k] for k in GROUP_KEYS}
    ood = [i for i in items if i.expected_unsupported]
    if ood:
        m = ste_only_metrics(ood)
        result["task:out_of_domain"] = {
            "controlled_failure_reliability": m["controlled_failure_reliability"]
        }
    return result
