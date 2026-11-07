"""Heuristic STE-oriented rule checks with machine-readable diagnostics.

This is a small, versioned subset of checks inspired by widely published
STE writing principles. It is **not** an implementation of ASD-STE100
Issue 9 and does not check the approved dictionary.

Known limitations:

* Sentence splitting is approximate (see :mod:`steward.ste.sentences`).
* The -ing check cannot tell verb forms from nouns; project terms are
  allowed through ``configs/ste/terms.toml``.
* The passive check matches "be" + past participle patterns. It misses
  some passives and flags some adjectives (false positives).
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

RULESET_VERSION = "ste-heuristic-0.1"
"""Bump when a rule changes behavior. Results with different versions are not comparable."""


class RuleCategory(StrEnum):
    """Groups of rules used for reporting."""

    SENTENCE_LENGTH = "sentence_length"
    GRAMMAR = "grammar"


class RuleSeverity(StrEnum):
    """``error`` violations make a text fail the rule check."""

    ERROR = "error"
    WARNING = "warning"


class Violation(BaseModel):
    """One rule violation with its text span."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str
    category: RuleCategory
    severity: RuleSeverity
    start: int
    end: int
    text: str
    message: str


@dataclass(frozen=True)
class RuleConfig:
    """Settings for the rule checks."""

    max_words_procedural: int = 20
    max_words_descriptive: int = 25
    ing_words_allowed: frozenset[str] = field(default_factory=frozenset)
    source_sha256: str = "builtin"


def load_rule_config(path: Path) -> RuleConfig:
    """Load the terminology file (``configs/ste/terms.toml``)."""
    raw = path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    words = data.get("ing_words_allowed", [])
    if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
        raise ValueError(f"{path}: 'ing_words_allowed' must be a list of strings")
    return RuleConfig(
        ing_words_allowed=frozenset(w.casefold() for w in words),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'./][A-Za-z0-9]+)*")
_ING_RE = re.compile(r"\b[A-Za-z][A-Za-z-]*ing\b")
_CONTRACTION_RE = re.compile(r"\b[A-Za-z]+(?:n't|'re|'ve|'ll|'d|'m)\b", re.IGNORECASE)
_IRREGULAR = (
    "made|done|set|put|held|kept|cut|shut|left|found|seen|known|built|sent|told|given|"
    "taken|shown|worn|torn|bent|broken|driven|fitted|tightened|loosened"
)
_PASSIVE_RE = re.compile(
    rf"\b(?:is|are|was|were|be|been|being)\s+(?:[a-z]+ly\s+)?(?:[a-z]+ed|{_IRREGULAR})\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[tuple[int, int, str]]:
    spans = []
    for match in _SENTENCE_RE.finditer(text):
        chunk = match.group(0)
        stripped = chunk.strip()
        if _WORD_RE.search(stripped):
            start = match.start() + (len(chunk) - len(chunk.lstrip()))
            spans.append((start, start + len(stripped), stripped))
    return spans


def check_text(text: str, *, procedural: bool, config: RuleConfig) -> list[Violation]:
    """Return all violations in ``text``.

    Args:
        text: The text to check.
        procedural: True for instructions (procedures, troubleshooting,
            safety notices); False for descriptive text.
        config: Limits and allowed terms.
    """
    violations: list[Violation] = []
    limit = config.max_words_procedural if procedural else config.max_words_descriptive
    for start, end, sentence in _sentences(text):
        count = len(_WORD_RE.findall(sentence))
        if count > limit:
            violations.append(
                Violation(
                    rule_id="STE-SENT-LEN-PROC" if procedural else "STE-SENT-LEN-DESC",
                    category=RuleCategory.SENTENCE_LENGTH,
                    severity=RuleSeverity.ERROR,
                    start=start,
                    end=end,
                    text=sentence,
                    message=f"Sentence has {count} words; the limit is {limit}.",
                )
            )
    for match in _ING_RE.finditer(text):
        word = match.group(0)
        if word.casefold() in config.ing_words_allowed or len(word) <= 4:
            continue
        violations.append(
            Violation(
                rule_id="STE-GRAM-ING",
                category=RuleCategory.GRAMMAR,
                severity=RuleSeverity.WARNING,
                start=match.start(),
                end=match.end(),
                text=word,
                message=f"'{word}' may be an -ing verb form. Use another verb form if possible.",
            )
        )
    for match in _CONTRACTION_RE.finditer(text):
        violations.append(
            Violation(
                rule_id="STE-GRAM-CONTRACTION",
                category=RuleCategory.GRAMMAR,
                severity=RuleSeverity.ERROR,
                start=match.start(),
                end=match.end(),
                text=match.group(0),
                message="Do not use contractions.",
            )
        )
    for match in _PASSIVE_RE.finditer(text):
        violations.append(
            Violation(
                rule_id="STE-GRAM-PASSIVE",
                category=RuleCategory.GRAMMAR,
                severity=RuleSeverity.ERROR if procedural else RuleSeverity.WARNING,
                start=match.start(),
                end=match.end(),
                text=match.group(0),
                message="Possible passive voice. Use the active voice"
                + (" in instructions." if procedural else " where possible."),
            )
        )
    return sorted(violations, key=lambda v: (v.start, v.rule_id))
