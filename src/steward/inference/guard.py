"""The inference-layer output guard for STE-only generation.

The model cannot guarantee correct output, so every generation is checked
before it is returned. The guard **never edits** the output. It reports
each check and decides a status:

* ``accepted``: every check passed. The text is labelled
  ``validated_against_available_checks``. That is *not* ASD-STE100
  compliance: it means only that the automated heuristics found no problem.
* ``flagged``: only non-critical problems (heuristic STE rule errors). The
  text is returned as a ``candidate`` that needs review.
* ``rejected``: a critical check failed (lost or invented facts, lost
  safety labels, facts from different steps in another order, non-English
  output, commentary). No text is returned.
* ``unsupported_input``: the model returned the controlled-failure
  sentinel for out-of-domain input.
* ``invalid_input``: the input is empty or too long; the model is not called.

Limitations: all checks are heuristic or literal-token comparisons (see
:mod:`steward.ste.preservation`). They miss meaning changes that keep the
same values and words, for example a reversed instruction.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from steward.datasets.schema import CheckKind, ReviewState, TaskCategory
from steward.ste import fidelity as fd
from steward.ste import preservation as pv
from steward.ste.rules import RULESET_VERSION, RuleConfig, RuleSeverity, Violation, check_text
from steward.ste.sentences import words

GUARD_VERSION = "guard-0.3"
PROCEDURAL = {TaskCategory.PROCEDURE, TaskCategory.TROUBLESHOOTING, TaskCategory.SAFETY_NOTICE}


class GuardStatus(StrEnum):
    """Final decision of the guard."""

    ACCEPTED = "accepted"
    FLAGGED = "flagged"
    REJECTED = "rejected"
    UNSUPPORTED_INPUT = "unsupported_input"
    INVALID_INPUT = "invalid_input"


class GuardPolicy(BaseModel):
    """Guard settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strict_ste: bool = False
    """Treat heuristic STE rule errors as critical (reject instead of flag)."""
    max_input_chars: int = Field(default=6000, ge=1)


class GuardCheck(BaseModel):
    """The result of one guard check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    passed: bool
    critical: bool
    kind: CheckKind
    message: str
    details: dict[str, JsonValue] = Field(default_factory=dict)


class GuardReport(BaseModel):
    """Validation metadata for one generation. Separate from the generated text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: GuardStatus
    guard_version: str = GUARD_VERSION
    ruleset_version: str = RULESET_VERSION
    source_language: str
    output_language: str | None = None
    checks: list[GuardCheck] = Field(default_factory=list)
    critical_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)
    review_state: ReviewState | None = None
    terminology_sha256: str | None = None
    note: str = (
        "Automated heuristic checks only. Not an ASD-STE100 compliance assessment or "
        "certification. Human review is required before operational use."
    )


def check_input(text: str, policy: GuardPolicy) -> GuardReport | None:
    """Return an ``invalid_input`` report for unusable input, else None."""
    problem = None
    if not text.strip():
        problem = "The input is empty."
    elif len(text) > policy.max_input_chars:
        problem = f"The input has {len(text)} characters; the limit is {policy.max_input_chars}."
    if problem is None:
        return None
    return GuardReport(
        status=GuardStatus.INVALID_INPUT,
        source_language=pv.detect_language(text).language,
        checks=[
            GuardCheck(
                check_id="input_valid",
                passed=False,
                critical=True,
                kind=CheckKind.DETERMINISTIC,
                message=problem,
            )
        ],
        critical_failures=["input_valid"],
    )


def _check(
    check_id: str,
    passed: bool,
    message_ok: str,
    message_bad: str,
    *,
    critical: bool = True,
    kind: CheckKind = CheckKind.DETERMINISTIC,
    **details: JsonValue,
) -> GuardCheck:
    return GuardCheck(
        check_id=check_id,
        passed=passed,
        critical=critical,
        kind=kind,
        message=message_ok if passed else message_bad,
        details=details,
    )


def guard_output(
    source: str,
    output: str,
    rules: RuleConfig,
    *,
    source_language: str | None = None,
    category: TaskCategory | None = None,
    policy: GuardPolicy | None = None,
) -> GuardReport:
    """Check ``output`` against ``source`` and decide a :class:`GuardStatus`."""
    policy = policy or GuardPolicy()
    language = source_language or pv.detect_language(source).language
    if language not in pv.LANGUAGES:
        language = "en"
    text = output.strip()
    if pv.is_unsupported_sentinel(text):
        return GuardReport(
            status=GuardStatus.UNSUPPORTED_INPUT,
            source_language=language,
            checks=[
                _check(
                    "controlled_failure",
                    True,
                    "The model reported the input as unsupported.",
                    "",
                    critical=False,
                )
            ],
        )

    checks: list[GuardCheck] = []
    checks.append(_check("output_nonempty", bool(text), "Output has text.", "Output is empty."))
    if pv.UNSUPPORTED_SENTINEL in text:
        checks.append(
            _check(
                "sentinel_mixed",
                False,
                "",
                "The output mixes the unsupported-input marker with other text.",
            )
        )
    problems = pv.contract_violations(text)
    checks.append(
        _check(
            "output_contract",
            not problems,
            "No commentary, Markdown, or reasoning text.",
            f"Output contract problems: {', '.join(problems)}.",
            kind=CheckKind.HEURISTIC,
            problems=list(problems),
        )
    )
    guess = pv.detect_language(text)
    foreign = pv.foreign_characters(text)
    english = guess.language in {"en", "unknown"} and foreign < 2
    checks.append(
        _check(
            "english_output",
            english,
            "Output is English.",
            f"Output looks like {guess.language}, not English"
            + (f" ({foreign} German/French/Romanian letters)." if foreign >= 2 else "."),
            kind=CheckKind.HEURISTIC,
            scores=dict(guess.scores),
            foreign_characters=foreign,
        )
    )
    src_words, out_words = len(words(source)), len(words(text))
    ratio = out_words / src_words if src_words else 1.0
    short = src_words >= 8 and ratio < 0.5
    checks.append(
        _check(
            "length_plausible",
            not short,
            f"Output/source length ratio {ratio:.2f}.",
            f"Output has {out_words} words for {src_words} source words (ratio {ratio:.2f}). "
            "Content may be missing.",
            kind=CheckKind.HEURISTIC,
            ratio=round(ratio, 3),
        )
    )

    src_numbers = pv.numbers(source, language)
    out_numbers = pv.numbers(text, "en")
    missing_numbers, extra_numbers = pv.multiset_missing(src_numbers, out_numbers)
    checks.append(
        _check(
            "numbers_preserved",
            not missing_numbers,
            "All source numbers are in the output.",
            f"Numbers missing: {', '.join(missing_numbers)}.",
            missing=list(missing_numbers),
        )
    )
    src_measures = Counter(pv.measurements(source, language))
    out_measures = Counter(pv.measurements(text, "en"))
    lost_measures = sorted(f"{v} {u}" for (v, u) in src_measures if out_measures[(v, u)] == 0)
    checks.append(
        _check(
            "measurements_preserved",
            not lost_measures,
            "All measurements keep their value and unit.",
            f"Measurements missing or with another unit: {', '.join(lost_measures)}.",
            missing=list(lost_measures),
        )
    )
    src_ids = pv.identifiers(source)
    lost_ids = [i for i in src_ids if i not in text]
    checks.append(
        _check(
            "identifiers_preserved",
            not lost_ids,
            "All identifiers are kept unchanged.",
            f"Identifiers missing or changed: {', '.join(lost_ids)}.",
            missing=list(lost_ids),
        )
    )
    inversions = pv.order_inversions(source, text, language)
    checks.append(
        _check(
            "sequence_preserved",
            not inversions,
            "Numbers and identifiers from different steps keep their order.",
            "Facts from different steps or sentences are in another order: "
            + ", ".join(f"{a} before {b}" for a, b in inversions[:5])
            + ". The procedure sequence may have changed.",
            kind=CheckKind.HEURISTIC,
            inverted=[f"{a}<{b}" for a, b in inversions],
        )
    )
    src_labels = pv.safety_labels(source)
    out_labels = pv.safety_labels(text, english_only=True)
    lost_labels = sorted(k for k in src_labels if out_labels[k] < src_labels[k])
    checks.append(
        _check(
            "safety_labels_preserved",
            not lost_labels,
            "Every WARNING, CAUTION, and NOTE is kept.",
            f"Safety labels missing (count decreased): {', '.join(lost_labels)}.",
            missing=list(lost_labels),
            source={k: v for k, v in src_labels.items()},
            output={k: v for k, v in out_labels.items()},
        )
    )
    src_neg = pv.negation_count(source, language)
    out_neg = pv.negation_count(text, "en")
    checks.append(
        _check(
            "negation_preserved",
            out_neg >= src_neg,
            "The output has at least as many negations as the source.",
            f"The source has {src_neg} negation(s) but the output has {out_neg}. A prohibition "
            "or condition may be lost or reversed.",
            kind=CheckKind.HEURISTIC,
            source_negations=src_neg,
            output_negations=out_neg,
        )
    )
    terminology = fd.load_terminology()
    terms = fd.terminology_findings(source, text, language, terminology)
    wrong = [
        f"{t.term_id}: '{t.source_match}' as {', '.join(t.forbidden_found)}"
        for t in terms
        if t.forbidden_found
    ]
    checks.append(
        _check(
            "terminology_forbidden",
            not wrong,
            "No known wrong rendering of a fixture term.",
            f"Wrong technical term: {'; '.join(wrong)}.",
            terms=[t.term_id for t in terms],
            wrong=list(wrong),
        )
    )
    unseen = [f"{t.term_id} ('{t.source_match}')" for t in terms if not t.expected_found]
    checks.append(
        _check(
            "terminology_expected",
            not unseen,
            "Fixture terms have an expected English rendering.",
            f"No expected English rendering for: {', '.join(unseen)}. Check the translation.",
            critical=False,
            kind=CheckKind.HEURISTIC,
            missing=list(unseen),
        )
    )
    leaked = fd.source_leakage(source, text, language)
    checks.append(
        _check(
            "no_source_language_words",
            not leaked,
            "No untranslated source-language words.",
            f"Untranslated {language} words: {', '.join(leaked)}.",
            kind=CheckKind.HEURISTIC,
            words=list(leaked),
        )
    )
    contradicted = fd.contradictions(text)
    checks.append(
        _check(
            "no_contradiction",
            not contradicted,
            "No instruction is given both as a command and as a prohibition.",
            f"Contradictory instructions: {'; '.join(contradicted)}.",
            items=list(contradicted),
        )
    )
    absent = fd.missing_actions(source, text, language)
    checks.append(
        _check(
            "actions_present",
            not absent,
            "Every command verb of the source occurs in the output.",
            f"Actions of the source that are not in the output: {', '.join(absent)}.",
            critical=False,
            kind=CheckKind.HEURISTIC,
            missing=list(absent),
        )
    )
    added_ids = [i for i in pv.identifiers(text) if i not in source]
    added_labels = sorted(k for k in out_labels if out_labels[k] > src_labels[k])
    additions = (
        [f"number {n}" for n in extra_numbers]
        + [f"identifier {i}" for i in added_ids]
        + [f"label {k}" for k in added_labels]
    )
    checks.append(
        _check(
            "no_unsupported_additions",
            not additions,
            "No numbers, identifiers, or safety labels were added.",
            f"Not in the source: {', '.join(additions)}.",
            added=list(additions),
        )
    )

    procedural = (
        category in PROCEDURAL
        if category
        else bool(pv.safety_labels(text) or _looks_procedural(text))
    )
    violations = check_text(text, procedural=procedural, config=rules) if text else []
    errors = [v for v in violations if v.severity is RuleSeverity.ERROR]
    checks.append(
        _check(
            "ste_rules",
            not errors,
            "No heuristic STE rule errors.",
            f"{len(errors)} heuristic STE rule error(s): "
            + ", ".join(sorted({v.rule_id for v in errors})),
            critical=policy.strict_ste,
            kind=CheckKind.HEURISTIC,
            procedural=procedural,
            errors=len(errors),
            warnings=len(violations) - len(errors),
        )
    )

    critical = [c.check_id for c in checks if c.critical and not c.passed]
    warnings = [c.check_id for c in checks if not c.critical and not c.passed]
    if critical:
        status, review = GuardStatus.REJECTED, None
    elif warnings:
        status, review = GuardStatus.FLAGGED, ReviewState.CANDIDATE
    else:
        status, review = GuardStatus.ACCEPTED, ReviewState.VALIDATED_AGAINST_AVAILABLE_CHECKS
    return GuardReport(
        status=status,
        source_language=language,
        output_language=guess.language,
        checks=checks,
        critical_failures=critical,
        warnings=warnings,
        violations=violations,
        review_state=review,
        terminology_sha256=terminology.sha256,
    )


def _looks_procedural(text: str) -> bool:
    """A numbered list, or most sentences start with a common command verb."""
    from steward.ste.sentences import split_sentences

    sentences = split_sentences(text)
    if not sentences:
        return False
    starts = sum(1 for s in sentences if s.split()[0].casefold() in COMMAND_VERBS)
    return starts / len(sentences) >= 0.5


COMMAND_VERBS = frozenset(
    [
        "remove",
        "install",
        "check",
        "make",
        "do",
        "apply",
        "open",
        "close",
        "turn",
        "set",
        "connect",
        "disconnect",
        "replace",
        "clean",
        "examine",
        "measure",
        "tighten",
        "loosen",
        "torque",
        "push",
        "pull",
        "put",
        "make",
        "use",
        "record",
        "stop",
        "start",
        "release",
        "lock",
        "unlock",
        "drain",
        "fill",
        "attach",
        "move",
        "hold",
        "keep",
        "wait",
        "test",
        "inspect",
        "lift",
        "lower",
        "disengage",
        "engage",
        "energize",
        "de-energize",
        "discard",
    ]
)
