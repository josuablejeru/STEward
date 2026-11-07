"""Pre-lint sanity checks applied to each generated example.

These checks catch obvious generation failures (empty output, lost numbers,
lost negations). They are *not* the STE linter and do not assess
ASD-STE100 compliance. Checks with ``error`` severity cause rejection;
other failures are recorded on the accepted example for later review.
"""

from __future__ import annotations

import re

from pydantic import JsonValue

from steward.datasets.jsonl import normalize_for_dedup
from steward.datasets.schema import (
    CheckKind,
    CheckResult,
    Severity,
    TaskCategory,
    TaskType,
    ValidationReport,
)
from steward.generation.config import ValidationSettings
from steward.ste.sentences import split_sentences, word_count

CHECKER_VERSION = "prelint-1"

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|nothing|without|cannot|can't|don't|do\s+not|must\s+not)\b",
    re.IGNORECASE,
)
_SAFETY_RE = re.compile(r"\b(WARNING|CAUTION)\b")


def extract_numbers(text: str) -> list[str]:
    """Return numeric tokens with thousands separators removed (``1,000`` -> ``1000``)."""
    return [m.replace(",", "") for m in re.findall(r"\d+(?:[.,]\d+)*", text)]


def validate_example(
    category: TaskCategory,
    source_text: str,
    candidate: str,
    settings: ValidationSettings,
) -> ValidationReport:
    """Run all sanity checks and return a report."""
    checks = [
        _check_nonempty(candidate),
        _check_differs(source_text, candidate),
        _check_numbers(source_text, candidate),
        _check_negation(source_text, candidate),
        _check_safety_words(source_text, candidate),
        _check_length_ratio(source_text, candidate, settings),
        _check_sentence_length(category, candidate, settings),
    ]
    return ValidationReport.from_checks(CHECKER_VERSION, checks)


def _check_nonempty(candidate: str) -> CheckResult:
    ok = bool(candidate.strip())
    return CheckResult(
        check_id="candidate_nonempty",
        passed=ok,
        severity=Severity.ERROR,
        kind=CheckKind.DETERMINISTIC,
        message="Candidate has text." if ok else "Candidate is empty.",
    )


def _check_differs(source: str, candidate: str) -> CheckResult:
    ok = normalize_for_dedup(source) != normalize_for_dedup(candidate)
    return CheckResult(
        check_id="candidate_differs_from_source",
        passed=ok,
        severity=Severity.WARNING,
        kind=CheckKind.DETERMINISTIC,
        message="Candidate differs from the source."
        if ok
        else "Candidate is the same as the source (no rewrite happened).",
    )


def _check_numbers(source: str, candidate: str) -> CheckResult:
    remaining = extract_numbers(candidate)
    missing: list[str] = []
    for number in extract_numbers(source):
        if number in remaining:
            remaining.remove(number)
        else:
            missing.append(number)
    ok = not missing
    return CheckResult(
        check_id="numbers_preserved",
        passed=ok,
        severity=Severity.ERROR,
        kind=CheckKind.DETERMINISTIC,
        message="All numbers in the source are in the candidate."
        if ok
        else f"Numbers missing from the candidate: {', '.join(missing)}.",
        details={
            "missing": list(missing),
            "limitation": "Numbers written as words (for example 'two') are not compared.",
        },
    )


def _check_negation(source: str, candidate: str) -> CheckResult:
    in_source = len(_NEGATION_RE.findall(source))
    in_candidate = len(_NEGATION_RE.findall(candidate))
    ok = in_source == 0 or in_candidate > 0
    return CheckResult(
        check_id="negation_preserved",
        passed=ok,
        severity=Severity.WARNING,
        kind=CheckKind.HEURISTIC,
        message="Negation is present where the source has negation."
        if ok
        else "The source has negation but the candidate has none. Check the meaning.",
        details={"source_negations": in_source, "candidate_negations": in_candidate},
    )


def _check_safety_words(source: str, candidate: str) -> CheckResult:
    needed = set(_SAFETY_RE.findall(source.upper()))
    present = set(_SAFETY_RE.findall(candidate.upper()))
    missing = sorted(needed - present)
    ok = not missing
    return CheckResult(
        check_id="safety_labels_preserved",
        passed=ok,
        severity=Severity.WARNING,
        kind=CheckKind.DETERMINISTIC,
        message="WARNING/CAUTION labels are kept."
        if ok
        else f"Safety labels missing from the candidate: {', '.join(missing)}.",
        details={"missing": list(missing)},
    )


def _check_length_ratio(source: str, candidate: str, settings: ValidationSettings) -> CheckResult:
    src, cand = word_count(source), word_count(candidate)
    ratio = cand / src if src else 0.0
    ok = settings.min_length_ratio <= ratio <= settings.max_length_ratio
    return CheckResult(
        check_id="length_ratio",
        passed=ok,
        severity=Severity.WARNING,
        kind=CheckKind.HEURISTIC,
        message=f"Candidate/source word ratio is {ratio:.2f}."
        + ("" if ok else " Content may be lost or added."),
        details={"source_words": src, "candidate_words": cand, "ratio": round(ratio, 3)},
    )


def _check_sentence_length(
    category: TaskCategory, candidate: str, settings: ValidationSettings
) -> CheckResult:
    procedural = category in {TaskCategory.PROCEDURE, TaskCategory.TROUBLESHOOTING}
    limit = settings.max_words_procedural if procedural else settings.max_words_descriptive
    long_sentences: list[JsonValue] = [
        {"sentence": s, "words": n}
        for s in split_sentences(candidate)
        if (n := word_count(s)) > limit
    ]
    ok = not long_sentences
    return CheckResult(
        check_id="sentence_length",
        passed=ok,
        severity=Severity.WARNING,
        kind=CheckKind.HEURISTIC,
        message=f"All sentences have at most {limit} words."
        if ok
        else f"{len(long_sentences)} sentence(s) have more than {limit} words.",
        details={
            "limit": limit,
            "long_sentences": long_sentences,
            "limitation": "Sentence splitting is approximate; abbreviations can cause errors.",
        },
    )


# -- multitask checks (schema 1.1) ---------------------------------------------------

MULTITASK_CHECKER_VERSION = "prelint-3"
"""Checks for multilingual and requirements tasks. Stricter than ``prelint-1``:
lost or invented facts, lost safety labels, and non-English output are errors,
because these examples can become training targets. ``prelint-3`` adds exact
WARNING/CAUTION/NOTE counts, negation counts, terminology fixtures, untranslated
source words, contradictions, and required terms of the scenario focus."""


def validate_multitask(
    task: TaskType,
    language: str,
    category: TaskCategory,
    source_text: str,
    candidate: str,
    entities: list[str],
    settings: ValidationSettings,
    *,
    required_terms: list[str] | None = None,
    focus_source_term: str | None = None,
) -> ValidationReport:
    """Run the multitask sanity checks (see :data:`MULTITASK_CHECKER_VERSION`).

    ``required_terms`` are English words the target must contain, for example the
    intended rendering of a terminology-focus term (``nut`` for *Mutter*).
    """
    from steward.ste import fidelity as fd
    from steward.ste import preservation as pv

    checks = [_check_nonempty(candidate)]

    contract = pv.contract_violations(candidate)
    checks.append(
        CheckResult(
            check_id="no_commentary",
            passed=not contract,
            severity=Severity.ERROR,
            kind=CheckKind.HEURISTIC,
            message="No commentary, Markdown, or reasoning text."
            if not contract
            else f"Output contract problems: {', '.join(contract)}.",
            details={"problems": list(contract)},
        )
    )

    guess = pv.detect_language(candidate)
    english = guess.language in {"en", "unknown"}
    checks.append(
        CheckResult(
            check_id="target_is_english",
            passed=english,
            severity=Severity.ERROR,
            kind=CheckKind.HEURISTIC,
            message="Target is English."
            if english
            else f"Target looks like {guess.language}, not English.",
            details={"scores": dict(guess.scores)},
        )
    )

    src_numbers = pv.numbers(source_text, language)
    out_numbers = pv.numbers(candidate, "en")
    missing, extra = pv.multiset_missing(src_numbers, out_numbers)
    checks.append(
        CheckResult(
            check_id="numbers_preserved",
            passed=not missing,
            severity=Severity.ERROR,
            kind=CheckKind.DETERMINISTIC,
            message="All source numbers are in the target."
            if not missing
            else f"Numbers missing from the target: {', '.join(missing)}.",
            details={"missing": list(missing)},
        )
    )
    src_measures = set(pv.measurements(source_text, language))
    invented = sorted(
        f"{v} {u}" for v, u in set(pv.measurements(candidate, "en")) - src_measures if v in extra
    )
    checks.append(
        CheckResult(
            check_id="no_invented_measurements",
            passed=not invented,
            severity=Severity.ERROR,
            kind=CheckKind.DETERMINISTIC,
            message="No measurements were added."
            if not invented
            else f"Measurements not in the source: {', '.join(invented)}.",
            details={"invented": list(invented), "extra_numbers": list(extra)},
        )
    )

    lost_ids = [e for e in entities if e not in candidate]
    checks.append(
        CheckResult(
            check_id="identifiers_preserved",
            passed=not lost_ids,
            severity=Severity.ERROR,
            kind=CheckKind.DETERMINISTIC,
            message="All identifiers are in the target."
            if not lost_ids
            else f"Identifiers missing from the target: {', '.join(lost_ids)}.",
            details={"missing": list(lost_ids)},
        )
    )

    src_labels = pv.safety_labels(source_text)
    out_labels = pv.safety_labels(candidate, english_only=True)
    lost_labels = sorted(k for k in src_labels if out_labels[k] < src_labels[k])
    added_labels = sorted(k for k in out_labels if out_labels[k] > src_labels[k])
    checks.append(
        CheckResult(
            check_id="safety_labels_preserved",
            passed=not lost_labels and not added_labels,
            severity=Severity.ERROR,
            kind=CheckKind.DETERMINISTIC,
            message="WARNING/CAUTION/NOTE labels match the source exactly."
            if not lost_labels and not added_labels
            else f"Safety labels lost: {lost_labels or 'none'}; added: {added_labels or 'none'}.",
            details={"lost": list(lost_labels), "added": list(added_labels)},
        )
    )

    src_neg = pv.negation_count(source_text, language)
    out_neg = pv.negation_count(candidate, "en")
    checks.append(
        CheckResult(
            check_id="negation_preserved",
            passed=out_neg >= src_neg,
            severity=Severity.ERROR,
            kind=CheckKind.HEURISTIC,
            message="The target has at least as many negations as the source."
            if out_neg >= src_neg
            else f"The source has {src_neg} negation(s), the target {out_neg}.",
            details={"source_negations": src_neg, "target_negations": out_neg},
        )
    )
    terms = fd.terminology_findings(source_text, candidate, language)
    wrong = [f"{t.term_id}: {', '.join(t.forbidden_found)}" for t in terms if t.forbidden_found]
    checks.append(
        CheckResult(
            check_id="terminology_forbidden",
            passed=not wrong,
            severity=Severity.ERROR,
            kind=CheckKind.DETERMINISTIC,
            message="No known wrong term rendering."
            if not wrong
            else f"Wrong technical terms: {'; '.join(wrong)}.",
            details={"terms": [t.term_id for t in terms], "wrong": list(wrong)},
        )
    )
    unseen = [t.term_id for t in terms if not t.expected_found]
    checks.append(
        CheckResult(
            check_id="terminology_expected",
            passed=not unseen,
            severity=Severity.WARNING,
            kind=CheckKind.HEURISTIC,
            message="Fixture terms have an expected rendering."
            if not unseen
            else f"No expected rendering for: {', '.join(unseen)}.",
            details={"missing": list(unseen)},
        )
    )
    absent_terms = [t for t in (required_terms or []) if not fd.contains_phrase(candidate, t)]
    checks.append(
        CheckResult(
            check_id="required_terms_present",
            passed=not absent_terms,
            severity=Severity.ERROR,
            kind=CheckKind.DETERMINISTIC,
            message="Required focus terms are present."
            if not absent_terms
            else f"Required terms missing: {', '.join(absent_terms)}.",
            details={"required": list(required_terms or []), "missing": list(absent_terms)},
        )
    )
    if focus_source_term:
        in_source = focus_source_term.casefold() in source_text.casefold()
        checks.append(
            CheckResult(
                check_id="focus_term_in_source",
                passed=in_source,
                severity=Severity.ERROR,
                kind=CheckKind.DETERMINISTIC,
                message="The focus term occurs in the source."
                if in_source
                else f"The source does not use the focus term '{focus_source_term}'.",
                details={"term": focus_source_term},
            )
        )
    leaked = fd.source_leakage(source_text, candidate, language)
    checks.append(
        CheckResult(
            check_id="no_source_language_words",
            passed=not leaked,
            severity=Severity.ERROR,
            kind=CheckKind.HEURISTIC,
            message="No untranslated source-language words."
            if not leaked
            else f"Untranslated words: {', '.join(leaked)}.",
            details={"words": list(leaked)},
        )
    )
    contradicted = fd.contradictions(candidate)
    checks.append(
        CheckResult(
            check_id="no_contradiction",
            passed=not contradicted,
            severity=Severity.ERROR,
            kind=CheckKind.DETERMINISTIC,
            message="No contradictory instructions."
            if not contradicted
            else f"Contradictory instructions: {'; '.join(contradicted)}.",
            details={"items": list(contradicted)},
        )
    )

    if task is not TaskType.STE_IDENTITY:
        checks.append(_check_differs(source_text, candidate))
    if task is TaskType.PLAIN_TO_STE_REWRITE:
        checks.append(_check_length_ratio(source_text, candidate, settings))
    checks.append(_check_sentence_length(category, candidate, settings))
    return ValidationReport.from_checks(MULTITASK_CHECKER_VERSION, checks)
